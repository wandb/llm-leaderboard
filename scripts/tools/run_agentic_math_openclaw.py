#!/usr/bin/env python3
"""
Run Agentic Math tasks through OpenClaw and score final-answer math.
"""

from __future__ import annotations

import argparse
import atexit
import base64
import hashlib
import json
import os
import re
import shlex
import subprocess
import sys
import threading
import time
import traceback
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path, PurePosixPath
from typing import Any

TOOLS_DIR = Path(__file__).resolve().parent
if str(TOOLS_DIR) not in sys.path:
    sys.path.insert(0, str(TOOLS_DIR))

from weave_agents_native_trace import (
    DEFAULT_AGENT_NAME as DEFAULT_WEAVE_AGENTS_AGENT_NAME,
    DEFAULT_ENV_FILE as DEFAULT_WEAVE_AGENTS_ENV_FILE,
    WEAVE_AGENTS_OUTPUT_COLUMNS,
    empty_weave_agents_evidence,
    env_default_entity,
    env_default_project,
    verify_native_weave_agents_trace,
)


REPO_ROOT = Path(__file__).resolve().parents[2]
PROTOCOL_RUNNER = REPO_ROOT / "scripts" / "tools" / "run_openclaw_agent_protocol.py"
RUNNER_VERSION = "agentic-math-openclaw-2026-07-10-timeup-parallel-v2"
NEMOCLAW_OPENCLAW_CONFIG_PATH = "/sandbox/.openclaw/openclaw.json"
DEFAULT_MAX_INPUT_TOKENS = 500_000
DEFAULT_MAX_TOOL_CALLS = 40
DEFAULT_MAX_AGENT_TURNS = 40
DEFAULT_MAX_TOOL_WALL_SECONDS = 120
OPENCLAW_BUDGET_GUARD_PLUGIN_ID = "nejumi-budget-guard"
OPENCLAW_BUDGET_GUARD_PLUGIN_VERSION = "0.1.0"
OPENCLAW_BUDGET_GUARD_BLOCK_PREFIX = "NEJUMI_BUDGET_GUARD_BLOCKED"
DEFAULT_DENIED_TOOLS = [
    "code_execution",
    "web_search",
    "web_fetch",
    "browser",
    "browser_*",
]
DEFAULT_DENIED_ARGUMENT_PATTERNS = [
    r"https?://",
    r"\b(curl|wget)\b",
    r"\b(?:python(?:3)?\s+-m\s+)?pip(?:3)?\s+install\b",
    r"\b(requests|urllib|httpx)\.",
]


class TaskStartLimiter:
    def __init__(self, min_interval_seconds: float = 0.0) -> None:
        self.min_interval_seconds = max(0.0, float(min_interval_seconds or 0.0))
        self._lock = threading.Lock()
        self._next_start_time = 0.0

    def wait(self) -> None:
        if self.min_interval_seconds <= 0.0:
            return
        with self._lock:
            now = time.monotonic()
            sleep_for = max(0.0, self._next_start_time - now)
            self._next_start_time = max(now, self._next_start_time) + self.min_interval_seconds
        if sleep_for > 0.0:
            time.sleep(sleep_for)


_REGISTERED_NEMOCLAW_GATEWAY_AGENTS: list[tuple[argparse.Namespace, str]] = []
_NEMOCLAW_EXEC_LOCK = threading.Lock()
NEMOCLAW_GATEWAY_CLEANUP_TOTAL_TIMEOUT_SEC = 30.0
NEMOCLAW_GATEWAY_CLEANUP_PER_AGENT_TIMEOUT_SEC = 3
NEMOCLAW_PERMISSION_CLEANUP_RE = re.compile(
    r"OpenClaw permission cleanup failed|normalize_mutable_config_perms|expected 660 group-writable",
    re.IGNORECASE,
)
ANSWER_LINE_RE = re.compile(r"(?im)^\s*(?:final\s+)?(?:answer|答案)\s*[:：]\s*(.+?)\s*$")
ANSWER_RE = re.compile(
    r"(?im)^\s*(?:final\s+)?answer\s*[:：]\s*(?:\\boxed\{)?0*([0-9]{1,3})(?:\})?\b"
)
BOXED_RE = re.compile(r"\\boxed\{\s*0*([0-9]{1,3})\s*\}")
INT_RE = re.compile(r"\b0*([0-9]{1,3})\b")


def run_command(
    command: list[str],
    cwd: Path,
    timeout: int | None = None,
    check: bool = True,
) -> subprocess.CompletedProcess[str]:
    try:
        result = subprocess.run(
            command,
            cwd=str(cwd),
            text=True,
            capture_output=True,
            timeout=timeout,
            check=False,
        )
    except subprocess.TimeoutExpired as exc:
        stdout = exc.stdout.decode("utf-8", errors="replace") if isinstance(exc.stdout, bytes) else (exc.stdout or "")
        stderr = exc.stderr.decode("utf-8", errors="replace") if isinstance(exc.stderr, bytes) else (exc.stderr or "")
        timeout_message = f"\nCommand timed out after {timeout} seconds"
        result = subprocess.CompletedProcess(command, 124, stdout=stdout, stderr=stderr + timeout_message)
    if check and result.returncode != 0:
        raise RuntimeError(
            "Command failed\n"
            f"cmd: {' '.join(shlex.quote(part) for part in command)}\n"
            f"cwd: {cwd}\n"
            f"returncode: {result.returncode}\n"
            f"stdout:\n{result.stdout}\n"
            f"stderr:\n{result.stderr}"
        )
    return result


def write_task_error(task_dir: Path, row: dict[str, Any], exc: BaseException) -> Path:
    error_path = task_dir / "error.json"
    error_path.write_text(
        json.dumps(
            {
                "task_id": row.get("task_id"),
                "error_type": type(exc).__name__,
                "error": str(exc),
                "traceback": traceback.format_exc(),
            },
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    return error_path


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def safe_agent_id(task_id: str, prefix: str) -> str:
    digest = hashlib.sha256(task_id.encode("utf-8")).hexdigest()[:12]
    base = re.sub(r"[^a-zA-Z0-9_-]+", "-", prefix).strip("-").lower()
    return f"{base}-{digest}"


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def command_sha256(command: list[str]) -> str:
    payload = json.dumps(command, ensure_ascii=False, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def safe_id(value: str) -> str:
    return "".join(ch if ch.isalnum() or ch in "._-" else "_" for ch in value)


def resolve_session_prefix(args: argparse.Namespace, default_prefix: str = "agentic-math") -> str:
    configured = str(getattr(args, "session_prefix", "") or "").strip()
    prefix = configured or default_prefix
    wandb_run_id = os.environ.get("WANDB_RUN_ID", "").strip()
    if "{wandb_run_id}" in prefix:
        return prefix.replace("{wandb_run_id}", wandb_run_id or "no-wandb-run-id").lower()
    if wandb_run_id and wandb_run_id not in prefix:
        return f"{wandb_run_id}:{prefix}".lower()
    return prefix.lower()


def build_cache_key(row: dict[str, Any], prompt_text: str, args: argparse.Namespace) -> dict[str, Any]:
    return {
        "runner_version": RUNNER_VERSION,
        "task_id": str(row["task_id"]),
        "prompt_hash": sha256_text(prompt_text),
        "model": args.model or "",
        "thinking": args.thinking,
        "answer_format": str(row.get("answer_format", "integer_0_999")),
        "deny_tools": effective_deny_tools(args),
        "deny_argument_patterns": effective_deny_argument_patterns(args),
        "openclaw_config_source": openclaw_config_cache_source(args),
        "max_input_tokens": int(getattr(args, "max_input_tokens", 0) or 0),
        "max_cumulative_input_tokens": resolved_max_cumulative_input_tokens(args),
        "max_cumulative_output_tokens": resolved_max_cumulative_output_tokens(args),
        "require_actual_token_usage": bool(getattr(args, "require_actual_token_usage", False)),
        "max_tool_calls": int(getattr(args, "max_tool_calls", 0) or 0),
        "max_agent_turns": int(getattr(args, "max_agent_turns", 0) or 0),
        "max_tool_wall_seconds": int(getattr(args, "max_tool_wall_seconds", 0) or 0),
        "openclaw_budget_guard_plugin": OPENCLAW_BUDGET_GUARD_PLUGIN_ID,
        "openclaw_budget_guard_plugin_version": OPENCLAW_BUDGET_GUARD_PLUGIN_VERSION,
        "openclaw_budget_guard_enforcement": "before_tool_call_before_agent_run_and_actual_usage",
        "agent_runtime": "nemoclaw" if getattr(args, "nemoclaw_sandbox", None) else "host",
        "nemoclaw_sandbox": str(getattr(args, "nemoclaw_sandbox", "") or ""),
        "use_task_agent": bool(getattr(args, "use_task_agent", True)),
        "session_prefix": resolve_session_prefix(args),
        "verify_weave_agents": bool(getattr(args, "verify_weave_agents", False)),
        "weave_agents_entity": str(getattr(args, "weave_agents_entity", "") or ""),
        "weave_agents_project": str(getattr(args, "weave_agents_project", "") or ""),
        "weave_agents_agent_name": str(getattr(args, "weave_agents_agent_name", "") or ""),
    }


def cache_key_matches(record: dict[str, Any], cache_key: dict[str, Any]) -> bool:
    return record.get("cache_key") == cache_key


def cache_requires_nemoclaw_session_audit(cache_key: dict[str, Any]) -> bool:
    return (
        cache_key.get("agent_runtime") == "nemoclaw"
        or bool(cache_key.get("nemoclaw_sandbox"))
    )


def default_openclaw_config_template() -> Path:
    return Path(os.environ.get("OPENCLAW_CONFIG_PATH", "~/.openclaw/openclaw.json")).expanduser()


def openclaw_config_cache_source(args: argparse.Namespace) -> str:
    template = getattr(args, "openclaw_config_template", None)
    if template:
        return str(Path(template).expanduser())
    if getattr(args, "nemoclaw_sandbox", None):
        return str(getattr(args, "nemoclaw_openclaw_config_path", NEMOCLAW_OPENCLAW_CONFIG_PATH))
    return str(default_openclaw_config_template())


def effective_deny_tools(args: argparse.Namespace) -> list[str]:
    return sorted(set(str(item) for item in (getattr(args, "deny_tool", None) or DEFAULT_DENIED_TOOLS)))


def effective_deny_argument_patterns(args: argparse.Namespace) -> list[str]:
    return sorted(
        set(str(item) for item in (getattr(args, "deny_argument_pattern", None) or DEFAULT_DENIED_ARGUMENT_PATTERNS))
    )


def resolved_max_cumulative_input_tokens(args: argparse.Namespace) -> int:
    value = getattr(args, "max_cumulative_input_tokens", None)
    if value is None:
        value = getattr(args, "max_input_tokens", 0)
    return int(value or 0)


def resolved_max_cumulative_output_tokens(args: argparse.Namespace) -> int:
    value = getattr(args, "max_cumulative_output_tokens", None)
    return int(value or 0)


def run_nemoclaw_text_command(
    args: argparse.Namespace,
    command: list[str],
    *,
    input_text: str | None = None,
    timeout: int = 60,
    check: bool = True,
) -> subprocess.CompletedProcess[str]:
    full_command = [
        args.nemoclaw_bin,
        "sandbox",
        "exec",
        args.nemoclaw_sandbox,
        "--workdir",
        "/sandbox",
        "--no-tty",
        "--timeout",
        str(timeout),
        "--",
        *command,
    ]
    with _NEMOCLAW_EXEC_LOCK:
        result = run_nemoclaw_subprocess(full_command, input_text=input_text, timeout=timeout)
        if result.returncode != 0 and nemoclaw_permission_cleanup_failed(result):
            repair = repair_nemoclaw_openclaw_permissions_locked(args, timeout=timeout)
            if repair.returncode == 0:
                result = run_nemoclaw_subprocess(full_command, input_text=input_text, timeout=timeout)
            else:
                result.stderr += (
                    "\nNeMoClaw OpenClaw config permission repair failed before retry.\n"
                    f"repair stdout:\n{repair.stdout}\n"
                    f"repair stderr:\n{repair.stderr}"
                )
    if check and result.returncode != 0:
        raise RuntimeError(
            "NeMoClaw command failed\n"
            f"cmd: {' '.join(shlex.quote(part) for part in full_command)}\n"
            f"returncode: {result.returncode}\n"
            f"stdout:\n{result.stdout}\n"
            f"stderr:\n{result.stderr}"
        )
    return result


def run_nemoclaw_subprocess(
    full_command: list[str],
    *,
    input_text: str | None = None,
    timeout: int = 60,
) -> subprocess.CompletedProcess[str]:
    try:
        return subprocess.run(
            full_command,
            text=True,
            input=input_text,
            capture_output=True,
            check=False,
            timeout=max(timeout + 10, 10),
        )
    except subprocess.TimeoutExpired as exc:
        stdout = exc.stdout.decode("utf-8", errors="replace") if isinstance(exc.stdout, bytes) else (exc.stdout or "")
        stderr = exc.stderr.decode("utf-8", errors="replace") if isinstance(exc.stderr, bytes) else (exc.stderr or "")
        return subprocess.CompletedProcess(
            full_command,
            124,
            stdout=stdout,
            stderr=stderr + f"\nNeMoClaw host command timed out after {timeout + 10} seconds",
        )


def nemoclaw_permission_cleanup_failed(result: subprocess.CompletedProcess[str]) -> bool:
    return bool(NEMOCLAW_PERMISSION_CLEANUP_RE.search((result.stderr or "") + "\n" + (result.stdout or "")))


def repair_nemoclaw_openclaw_permissions_locked(
    args: argparse.Namespace,
    *,
    timeout: int = 60,
) -> subprocess.CompletedProcess[str]:
    config_path = str(getattr(args, "nemoclaw_openclaw_config_path", NEMOCLAW_OPENCLAW_CONFIG_PATH))
    config_dir = str(PurePosixPath(config_path).parent)
    script = (
        "set -eu; "
        'config_dir="$1"; config_path="$2"; '
        'test -d "$config_dir"; '
        'chmod 2770 "$config_dir"; '
        'if [ -f "$config_path" ]; then chmod 660 "$config_path"; fi; '
        'hash_path="$config_dir/.config-hash"; '
        'if [ -f "$hash_path" ]; then chmod 660 "$hash_path"; fi; '
        "stat -c '%a %u %g %n' \"$config_dir\" \"$config_path\" \"$hash_path\" 2>/dev/null || true"
    )
    full_command = [
        args.nemoclaw_bin,
        "sandbox",
        "exec",
        args.nemoclaw_sandbox,
        "--workdir",
        "/sandbox",
        "--no-tty",
        "--timeout",
        str(timeout),
        "--",
        "sh",
        "-lc",
        script,
        "repair-openclaw-perms",
        config_dir,
        config_path,
    ]
    return run_nemoclaw_subprocess(full_command, timeout=timeout)


def ensure_nemoclaw_openclaw_permissions(args: argparse.Namespace) -> None:
    if not getattr(args, "nemoclaw_sandbox", None):
        return
    with _NEMOCLAW_EXEC_LOCK:
        result = repair_nemoclaw_openclaw_permissions_locked(args, timeout=60)
    if result.returncode != 0:
        raise RuntimeError(
            "NeMoClaw OpenClaw config permission preflight failed\n"
            f"stdout:\n{result.stdout}\n"
            f"stderr:\n{result.stderr}"
        )


def uses_nemoclaw_gateway_task_agent(args: argparse.Namespace) -> bool:
    return (
        bool(getattr(args, "use_task_agent", True))
        and bool(getattr(args, "no_local", False))
        and bool(getattr(args, "nemoclaw_sandbox", None))
    )


def restart_nemoclaw_gateway_after_task_agent_registration(
    args: argparse.Namespace,
    *,
    label: str,
    timeout: int = 120,
) -> None:
    if (
        not uses_nemoclaw_gateway_task_agent(args)
        or bool(getattr(args, "dry_run", False))
        or not bool(getattr(args, "restart_gateway_after_task_agent_registration", True))
    ):
        return
    command = [
        str(getattr(args, "nemoclaw_bin", "nemoclaw")),
        "sandbox",
        "gateway",
        "restart",
        str(getattr(args, "nemoclaw_sandbox")),
        "--quiet",
    ]
    print(
        f"Restarting NeMoClaw Gateway after {label} task-agent registration.",
        flush=True,
    )
    with _NEMOCLAW_EXEC_LOCK:
        result = run_nemoclaw_subprocess(command, timeout=timeout)
    if result.returncode != 0:
        raise RuntimeError(
            "NeMoClaw Gateway restart failed after task-agent registration\n"
            f"cmd: {' '.join(shlex.quote(part) for part in command)}\n"
            f"returncode: {result.returncode}\n"
            f"stdout:\n{result.stdout}\n"
            f"stderr:\n{result.stderr}"
        )
    print(
        f"NeMoClaw Gateway restarted after {label} task-agent registration.",
        flush=True,
    )


def register_nemoclaw_gateway_task_agent(
    args: argparse.Namespace,
    *,
    agent_id: str,
    workspace: str,
    agent_dir: str,
) -> dict[str, Any]:
    config_path = str(getattr(args, "nemoclaw_openclaw_config_path", NEMOCLAW_OPENCLAW_CONFIG_PATH))
    model = str(getattr(args, "model", "") or "")
    tool_profile = str(getattr(args, "openclaw_tool_profile", "") or "")
    deny_json = json.dumps(effective_deny_tools(args), ensure_ascii=False, separators=(",", ":"))
    session_prefix = resolve_session_prefix(args)
    budget_json = json.dumps(
        {
            "max_input_tokens": int(getattr(args, "max_input_tokens", 0) or 0),
            "max_tool_calls": int(getattr(args, "max_tool_calls", 0) or 0),
            "max_agent_turns": int(getattr(args, "max_agent_turns", 0) or 0),
            "max_tool_wall_seconds": int(getattr(args, "max_tool_wall_seconds", 0) or 0),
            "session_prefix": session_prefix,
            "max_cumulative_input_tokens": resolved_max_cumulative_input_tokens(args),
            "max_cumulative_output_tokens": resolved_max_cumulative_output_tokens(args),
            "require_actual_token_usage": bool(getattr(args, "require_actual_token_usage", False)),
        },
        ensure_ascii=False,
        separators=(",", ":"),
    )
    if bool(getattr(args, "dry_run", False)):
        return {
            "ok": None,
            "skipped": True,
            "reason": "dry_run",
            "agent_id": agent_id,
            "workspace": workspace,
            "agent_dir": agent_dir,
            "config_path": config_path,
        }

    patch_script = r"""
import json
import sys
from pathlib import Path

config_path = Path(sys.argv[1])
agent_id = sys.argv[2]
workspace = sys.argv[3]
agent_dir = sys.argv[4]
model = sys.argv[5]
tool_profile = sys.argv[6]
deny_tools = json.loads(sys.argv[7])
budget = json.loads(sys.argv[8])
max_input_tokens = int(budget.get("max_input_tokens") or 0)
max_tool_calls = int(budget.get("max_tool_calls") or 0)
max_agent_turns = int(budget.get("max_agent_turns") or 0)
max_tool_wall_seconds = int(budget.get("max_tool_wall_seconds") or 0)
session_prefix = str(budget.get("session_prefix") or "")
max_cumulative_input_tokens = int(budget.get("max_cumulative_input_tokens") or 0)
max_cumulative_output_tokens = int(budget.get("max_cumulative_output_tokens") or 0)
require_actual_token_usage = bool(budget.get("require_actual_token_usage"))
session_key_prefixes = [
    session_prefix,
    f"agent:{agent_id}:{session_prefix}",
]

config = json.loads(config_path.read_text(encoding="utf-8"))

def positive_int(value):
    return int(value) if isinstance(value, int) and value > 0 else 0

def context_cap_value(existing, context_window, cap):
    candidates = [cap]
    existing_int = positive_int(existing)
    if existing_int:
        candidates.append(existing_int)
    context_window_int = positive_int(context_window)
    if context_window_int:
        candidates.append(context_window_int)
    return min(candidates)

def resolve_context_cap(config, model, max_input_tokens):
    if max_input_tokens <= 0 or not model or "/" not in model:
        return None
    provider_id, model_id = model.split("/", 1)
    if not provider_id or not model_id:
        return None
    providers = config.setdefault("models", {}).setdefault("providers", {})
    provider = providers.setdefault(provider_id, {})
    if not isinstance(provider, dict):
        raise SystemExit(f"OpenClaw config field models.providers.{provider_id} must be an object")
    models = provider.setdefault("models", [])
    if not isinstance(models, list):
        raise SystemExit(f"OpenClaw config field models.providers.{provider_id}.models must be a list")
    target = None
    for item in models:
        if not isinstance(item, dict):
            continue
        if item.get("id") == model_id or item.get("name") in {model_id, model}:
            target = item
            break
    if target is None:
        target = {}
    context_tokens = context_cap_value(
        target.get("contextTokens"),
        target.get("contextWindow"),
        max_input_tokens,
    )
    return {"provider": provider_id, "model": model_id, "contextTokens": context_tokens}

context_cap = resolve_context_cap(config, model, max_input_tokens)
def run_retries(max_agent_turns):
    if max_agent_turns <= 0:
        return None
    return {
        "base": max_agent_turns,
        "perProfile": 0,
        "min": max_agent_turns,
        "max": max_agent_turns,
    }

turn_run_retries = run_retries(max_agent_turns)
tools = config.setdefault("tools", {})
if isinstance(tools, dict):
    tools["toolSearch"] = False
    exec_config = tools.setdefault("exec", {})
    if max_tool_wall_seconds > 0:
        if not isinstance(exec_config, dict):
            raise SystemExit("OpenClaw config field tools.exec must be an object")
        exec_config["timeoutSec"] = max_tool_wall_seconds
    web = tools.setdefault("web", {})
    if isinstance(web, dict):
        fetch = web.setdefault("fetch", {})
        if isinstance(fetch, dict):
            fetch["enabled"] = False

plugins = config.setdefault("plugins", {})
if not isinstance(plugins, dict):
    raise SystemExit("OpenClaw config field plugins must be an object")
allow = plugins.setdefault("allow", [])
if isinstance(allow, list) and "nejumi-budget-guard" not in allow:
    allow.append("nejumi-budget-guard")
plugin_entries = plugins.setdefault("entries", {})
if not isinstance(plugin_entries, dict):
    raise SystemExit("OpenClaw config field plugins.entries must be an object")
budget_guard = plugin_entries.setdefault("nejumi-budget-guard", {})
if not isinstance(budget_guard, dict):
    raise SystemExit("OpenClaw config field plugins.entries.nejumi-budget-guard must be an object")
budget_guard["enabled"] = True
existing_budget_config = budget_guard.get("config") if isinstance(budget_guard.get("config"), dict) else {}
existing_agent_ids = [
    str(item)
    for item in existing_budget_config.get("agentIds", [])
    if isinstance(item, str) and item
]
if agent_id not in existing_agent_ids:
    existing_agent_ids.append(agent_id)
existing_session_key_prefixes = [
    str(item)
    for item in existing_budget_config.get("sessionKeyPrefixes", [])
    if isinstance(item, str) and item
]
for prefix in session_key_prefixes:
    if prefix and prefix not in existing_session_key_prefixes:
        existing_session_key_prefixes.append(prefix)

budget_guard["config"] = {
    "enabled": True,
    "maxToolCalls": max_tool_calls,
    "maxAgentTurns": max_agent_turns,
    "maxCumulativeInputTokens": max_cumulative_input_tokens,
    "maxCumulativeOutputTokens": max_cumulative_output_tokens,
    "requireActualTokenUsage": require_actual_token_usage,
    "agentIds": existing_agent_ids,
    "sessionKeyPrefixes": existing_session_key_prefixes,
    "blockReasonPrefix": "NEJUMI_BUDGET_GUARD_BLOCKED",
}
budget_guard["hooks"] = {
    "allowConversationAccess": True,
    "timeoutMs": 1000,
}

agents = config.setdefault("agents", {})
entries = agents.setdefault("list", [])
if not isinstance(entries, list):
    raise SystemExit("OpenClaw config field agents.list must be a list")

entry = {
    "id": agent_id,
    "workspace": workspace,
    "agentDir": agent_dir,
    "tools": {
        "profile": tool_profile,
        "deny": deny_tools,
    },
}
if model:
    entry["model"] = model
if context_cap:
    entry["contextTokens"] = context_cap["contextTokens"]
if turn_run_retries:
    entry["runRetries"] = turn_run_retries
entries[:] = [item for item in entries if not (isinstance(item, dict) and item.get("id") == agent_id)]
entries.append(entry)

tmp_path = config_path.with_suffix(config_path.suffix + ".tmp")
tmp_path.write_text(json.dumps(config, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
tmp_path.replace(config_path)
print(json.dumps({"ok": True, "agent_id": agent_id, "config_path": str(config_path), "context_cap": context_cap, "run_retries": turn_run_retries}))
"""
    shell_script = r"""
set -euo pipefail
agent_id="$1"
workspace="$2"
agent_dir="$3"
model="$4"
tool_profile="$5"
deny_json="$6"
config_path="$7"
budget_json="$8"
python3 - "$config_path" "$agent_id" "$agent_dir" <<'PY'
import json
import sys
from pathlib import Path

config_path = Path(sys.argv[1])
agent_id = sys.argv[2]
agent_dir = sys.argv[3]
config = json.loads(config_path.read_text(encoding="utf-8"))
agents = config.setdefault("agents", {})
entries = agents.setdefault("list", [])
if isinstance(entries, list):
    entries[:] = [
        item
        for item in entries
        if not (
            isinstance(item, dict)
            and (item.get("id") == agent_id or item.get("agentDir") == agent_dir)
        )
    ]
tmp_path = config_path.with_suffix(config_path.suffix + ".tmp")
tmp_path.write_text(json.dumps(config, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
tmp_path.replace(config_path)
PY
openclaw agents delete "$agent_id" --force --json >/dev/null 2>&1 || true
cmd=(openclaw agents add "$agent_id" --workspace "$workspace" --agent-dir "$agent_dir" --non-interactive --json)
if [ -n "$model" ]; then
  cmd+=(--model "$model")
fi
"${cmd[@]}"
python3 - "$config_path" "$agent_id" "$workspace" "$agent_dir" "$model" "$tool_profile" "$deny_json" "$budget_json" <<'PY'
""" + patch_script + r"""
PY
"""
    shell_script_b64 = base64.b64encode(shell_script.encode("utf-8")).decode("ascii")
    result = run_nemoclaw_text_command(
        args,
        [
            "env",
            f"OPENCLAW_REGISTER_SCRIPT_B64={shell_script_b64}",
            "bash",
            "-lc",
            'printf %s "$OPENCLAW_REGISTER_SCRIPT_B64" | base64 -d | bash -s -- "$@"',
            "register-task-agent",
            agent_id,
            workspace,
            agent_dir,
            model,
            tool_profile,
            deny_json,
            config_path,
            budget_json,
        ],
        timeout=60,
    )
    _REGISTERED_NEMOCLAW_GATEWAY_AGENTS.append((args, agent_id))
    return {
        "ok": True,
        "agent_id": agent_id,
        "workspace": workspace,
        "agent_dir": agent_dir,
        "config_path": config_path,
        "stdout": result.stdout,
        "stderr": result.stderr,
    }


def unregister_nemoclaw_gateway_task_agent(
    args: argparse.Namespace,
    agent_id: str,
    *,
    timeout: int = 60,
) -> dict[str, Any]:
    if not getattr(args, "nemoclaw_sandbox", None):
        return {"ok": None, "skipped": True, "reason": "no_nemoclaw_sandbox"}
    result = run_nemoclaw_text_command(
        args,
        [
            "bash",
            "-lc",
            'openclaw agents delete "$1" --force --json',
            "delete-task-agent",
            agent_id,
        ],
        timeout=timeout,
        check=False,
    )
    return {
        "ok": result.returncode == 0,
        "agent_id": agent_id,
        "returncode": result.returncode,
        "stdout": result.stdout,
        "stderr": result.stderr,
    }


def cleanup_registered_nemoclaw_gateway_agents() -> None:
    deadline = time.monotonic() + NEMOCLAW_GATEWAY_CLEANUP_TOTAL_TIMEOUT_SEC
    while _REGISTERED_NEMOCLAW_GATEWAY_AGENTS:
        if time.monotonic() >= deadline:
            remaining = len(_REGISTERED_NEMOCLAW_GATEWAY_AGENTS)
            _REGISTERED_NEMOCLAW_GATEWAY_AGENTS.clear()
            print(
                f"WARNING: skipped {remaining} NeMoClaw task-agent cleanup calls after "
                f"{NEMOCLAW_GATEWAY_CLEANUP_TOTAL_TIMEOUT_SEC:.0f}s cleanup budget",
                file=sys.stderr,
                flush=True,
            )
            return
        args, agent_id = _REGISTERED_NEMOCLAW_GATEWAY_AGENTS.pop()
        try:
            unregister_nemoclaw_gateway_task_agent(
                args,
                agent_id,
                timeout=NEMOCLAW_GATEWAY_CLEANUP_PER_AGENT_TIMEOUT_SEC,
            )
        except Exception:
            pass


atexit.register(cleanup_registered_nemoclaw_gateway_agents)


def read_openclaw_config_template(args: argparse.Namespace) -> tuple[dict[str, Any], str]:
    if args.openclaw_config_template:
        template_path = args.openclaw_config_template.expanduser()
        return json.loads(template_path.read_text(encoding="utf-8")), str(template_path)
    if getattr(args, "nemoclaw_sandbox", None):
        sandbox_path = getattr(args, "nemoclaw_openclaw_config_path", NEMOCLAW_OPENCLAW_CONFIG_PATH)
        result = run_nemoclaw_text_command(args, ["cat", sandbox_path], timeout=60)
        return json.loads(result.stdout), sandbox_path
    template_path = default_openclaw_config_template()
    return json.loads(template_path.read_text(encoding="utf-8")), str(template_path)


def configure_openclaw_budget_guard(
    config: dict[str, Any],
    args: argparse.Namespace,
    *,
    agent_id: str,
    session_prefix: str,
) -> None:
    max_tool_calls = int(getattr(args, "max_tool_calls", 0) or 0)
    max_agent_turns = int(getattr(args, "max_agent_turns", 0) or 0)
    max_tool_wall_seconds = int(getattr(args, "max_tool_wall_seconds", 0) or 0)
    max_cumulative_input_tokens = resolved_max_cumulative_input_tokens(args)
    max_cumulative_output_tokens = resolved_max_cumulative_output_tokens(args)
    require_actual_token_usage = bool(getattr(args, "require_actual_token_usage", False))
    if (
        max_tool_calls <= 0
        and max_agent_turns <= 0
        and max_tool_wall_seconds <= 0
        and max_cumulative_input_tokens <= 0
        and max_cumulative_output_tokens <= 0
        and not require_actual_token_usage
    ):
        return

    plugins = config.setdefault("plugins", {})
    if not isinstance(plugins, dict):
        raise RuntimeError("OpenClaw config field plugins must be an object")
    allow = plugins.setdefault("allow", [])
    if isinstance(allow, list) and OPENCLAW_BUDGET_GUARD_PLUGIN_ID not in allow:
        allow.append(OPENCLAW_BUDGET_GUARD_PLUGIN_ID)
    entries = plugins.setdefault("entries", {})
    if not isinstance(entries, dict):
        raise RuntimeError("OpenClaw config field plugins.entries must be an object")
    entry = entries.setdefault(OPENCLAW_BUDGET_GUARD_PLUGIN_ID, {})
    if not isinstance(entry, dict):
        raise RuntimeError(
            f"OpenClaw config field plugins.entries.{OPENCLAW_BUDGET_GUARD_PLUGIN_ID} must be an object"
        )
    entry["enabled"] = True
    entry["config"] = {
        "enabled": True,
        "maxToolCalls": max_tool_calls,
        "maxAgentTurns": max_agent_turns,
        "maxCumulativeInputTokens": max_cumulative_input_tokens,
        "maxCumulativeOutputTokens": max_cumulative_output_tokens,
        "requireActualTokenUsage": require_actual_token_usage,
        "agentIds": [agent_id],
        "sessionKeyPrefixes": [
            session_prefix,
            f"agent:{agent_id}:{session_prefix}",
        ],
        "blockReasonPrefix": OPENCLAW_BUDGET_GUARD_BLOCK_PREFIX,
    }
    entry["hooks"] = {
        "allowConversationAccess": True,
        "timeoutMs": 1000,
    }


def configure_openclaw_exec_timeout(config: dict[str, Any], args: argparse.Namespace) -> dict[str, int] | None:
    max_tool_wall_seconds = int(getattr(args, "max_tool_wall_seconds", 0) or 0)
    if max_tool_wall_seconds <= 0:
        return None
    tools = config.setdefault("tools", {})
    if not isinstance(tools, dict):
        raise RuntimeError("OpenClaw config field tools must be an object")
    exec_config = tools.setdefault("exec", {})
    if not isinstance(exec_config, dict):
        raise RuntimeError("OpenClaw config field tools.exec must be an object")
    exec_config["timeoutSec"] = max_tool_wall_seconds
    return {"timeoutSec": int(exec_config["timeoutSec"])}


def split_openclaw_model_ref(model_ref: str) -> tuple[str, str]:
    model_ref = str(model_ref or "").strip()
    if "/" not in model_ref:
        return "", model_ref
    provider_id, model_id = model_ref.split("/", 1)
    return provider_id.strip(), model_id.strip()


def positive_config_int(value: Any) -> int:
    if isinstance(value, bool):
        return 0
    if isinstance(value, (int, float)) and int(value) == value and value > 0:
        return int(value)
    return 0


def bounded_context_tokens(existing: Any, context_window: Any, max_input_tokens: int) -> int:
    candidates = [max_input_tokens]
    existing_int = positive_config_int(existing)
    if existing_int:
        candidates.append(existing_int)
    context_window_int = positive_config_int(context_window)
    if context_window_int:
        candidates.append(context_window_int)
    return min(candidates)


def configure_openclaw_context_tokens(config: dict[str, Any], args: argparse.Namespace) -> dict[str, Any] | None:
    max_input_tokens = int(getattr(args, "max_input_tokens", 0) or 0)
    if max_input_tokens <= 0:
        return None
    provider_id, model_id = split_openclaw_model_ref(str(getattr(args, "model", "") or ""))
    if not provider_id or not model_id:
        return None
    models_config = config.setdefault("models", {})
    if not isinstance(models_config, dict):
        raise RuntimeError("OpenClaw config field models must be an object")
    providers = models_config.setdefault("providers", {})
    if not isinstance(providers, dict):
        raise RuntimeError("OpenClaw config field models.providers must be an object")
    provider = providers.setdefault(provider_id, {})
    if not isinstance(provider, dict):
        raise RuntimeError(f"OpenClaw config field models.providers.{provider_id} must be an object")
    model_entries = provider.setdefault("models", [])
    if not isinstance(model_entries, list):
        raise RuntimeError(f"OpenClaw config field models.providers.{provider_id}.models must be a list")
    target: dict[str, Any] | None = None
    for entry in model_entries:
        if not isinstance(entry, dict):
            continue
        if entry.get("id") == model_id or entry.get("name") in {model_id, f"{provider_id}/{model_id}"}:
            target = entry
            break
    if target is None:
        target = {"id": model_id, "name": model_id}
        model_entries.append(target)
    target["contextTokens"] = bounded_context_tokens(
        target.get("contextTokens"),
        target.get("contextWindow"),
        max_input_tokens,
    )
    return {
        "provider": provider_id,
        "model": model_id,
        "contextTokens": target["contextTokens"],
    }


def openclaw_run_retries_for_turn_cap(args: argparse.Namespace) -> dict[str, int] | None:
    max_agent_turns = int(getattr(args, "max_agent_turns", 0) or 0)
    if max_agent_turns <= 0:
        return None
    return {
        "base": max_agent_turns,
        "perProfile": 0,
        "min": max_agent_turns,
        "max": max_agent_turns,
    }


def configure_agent_turn_run_retries(
    agent_entry: dict[str, Any],
    args: argparse.Namespace,
) -> dict[str, int] | None:
    run_retries = openclaw_run_retries_for_turn_cap(args)
    if run_retries:
        agent_entry["runRetries"] = run_retries
    return run_retries


def write_nemoclaw_text_file(args: argparse.Namespace, path: str, text: str) -> None:
    code = (
        "import sys; from pathlib import Path; "
        "path = Path(sys.argv[1]); path.parent.mkdir(parents=True, exist_ok=True); "
        "path.write_text(sys.stdin.read(), encoding='utf-8')"
    )
    run_nemoclaw_text_command(args, ["python3", "-c", code, path], input_text=text, timeout=60)


def nemoclaw_task_workdir(row: dict[str, Any], args: argparse.Namespace, agent_id: str) -> str:
    root = str(getattr(args, "nemoclaw_workdir", "") or "/sandbox")
    safe_task = safe_agent_id(str(row["task_id"]), "task")
    if "{task_id}" in root or "{agent_id}" in root:
        return root.format(task_id=safe_task, agent_id=agent_id)
    return str(PurePosixPath(root) / "agentic_math" / safe_task)


def write_task_openclaw_config(
    row: dict[str, Any],
    workspace_dir: Path,
    task_dir: Path,
    args: argparse.Namespace,
) -> tuple[str, Path | None]:
    if not args.use_task_agent:
        return args.agent, None
    if args.no_local and not getattr(args, "nemoclaw_sandbox", None):
        raise RuntimeError("--use-task-agent requires local OpenClaw execution; remove --no-local")

    agent_id = safe_agent_id(str(row["task_id"]), args.task_agent_prefix)
    if getattr(args, "nemoclaw_sandbox", None):
        workspace = nemoclaw_task_workdir(row, args, agent_id)
        agent_dir = str(PurePosixPath(workspace) / "openclaw_agent_state")
        config_path = str(PurePosixPath(workspace) / "openclaw_config.json")
    else:
        workspace = str(workspace_dir.resolve())
        agent_dir = str((task_dir / "openclaw_agent_state").resolve())
        config_path = str(task_dir / "openclaw_config.json")

    if uses_nemoclaw_gateway_task_agent(args):
        canonical_config_path = str(
            getattr(args, "nemoclaw_openclaw_config_path", NEMOCLAW_OPENCLAW_CONFIG_PATH)
        )
        task_agent_path = task_dir / "openclaw_task_agent.json"
        if task_agent_path.exists() and not bool(getattr(args, "redo", False)):
            try:
                existing = json.loads(task_agent_path.read_text(encoding="utf-8"))
            except json.JSONDecodeError:
                existing = {}
            if (
                isinstance(existing, dict)
                and existing.get("agent_id") == agent_id
                and existing.get("workspace") == workspace
                and existing.get("agent_dir") == agent_dir
                and existing.get("config_path") == canonical_config_path
                and isinstance(existing.get("gateway_registered"), dict)
                and existing["gateway_registered"].get("ok") is True
            ):
                return agent_id, None

        registration = register_nemoclaw_gateway_task_agent(
            args,
            agent_id=agent_id,
            workspace=workspace,
            agent_dir=agent_dir,
        )
        task_agent_path.write_text(
            json.dumps(
                {
                    "agent_id": agent_id,
                    "workspace": workspace,
                    "agent_dir": agent_dir,
                    "config_template": canonical_config_path,
                    "config_path": canonical_config_path,
                    "nemoclaw_sandbox": getattr(args, "nemoclaw_sandbox", None),
                    "exec_timeout": (
                        {"timeoutSec": int(getattr(args, "max_tool_wall_seconds", 0) or 0)}
                        if int(getattr(args, "max_tool_wall_seconds", 0) or 0) > 0
                        else None
                    ),
                    "gateway_registered": registration,
                },
                ensure_ascii=False,
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )
        return agent_id, None

    config, template_path = read_openclaw_config_template(args)
    disable_remote_lookup_tools(config)
    exec_timeout = configure_openclaw_exec_timeout(config, args)
    context_cap = configure_openclaw_context_tokens(config, args)
    configure_openclaw_budget_guard(
        config,
        args,
        agent_id=agent_id,
        session_prefix=resolve_session_prefix(args),
    )
    agent_entry = {
        "id": agent_id,
        "workspace": workspace,
        "agentDir": agent_dir,
        "tools": {
            "profile": args.openclaw_tool_profile,
            "deny": effective_deny_tools(args),
        },
    }
    if context_cap:
        agent_entry["contextTokens"] = context_cap["contextTokens"]
    run_retries = configure_agent_turn_run_retries(agent_entry, args)

    agents = config.setdefault("agents", {})
    entries = agents.setdefault("list", [])
    if not isinstance(entries, list):
        raise RuntimeError("OpenClaw config field agents.list must be a list")
    entries[:] = [entry for entry in entries if not (isinstance(entry, dict) and entry.get("id") == agent_id)]
    entries.append(agent_entry)

    config_text = json.dumps(config, ensure_ascii=False, indent=2) + "\n"
    if getattr(args, "nemoclaw_sandbox", None):
        write_nemoclaw_text_file(args, config_path, config_text)
    else:
        Path(config_path).write_text(config_text, encoding="utf-8")
    (task_dir / "openclaw_task_agent.json").write_text(
        json.dumps(
            {
                "agent_id": agent_id,
                "workspace": agent_entry["workspace"],
                "agent_dir": agent_entry["agentDir"],
                    "config_template": template_path,
                    "config_path": config_path,
                    "nemoclaw_sandbox": getattr(args, "nemoclaw_sandbox", None),
                    "context_cap": context_cap,
                    "exec_timeout": exec_timeout,
                    "run_retries": run_retries,
                },
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    return agent_id, Path(config_path)


def task_live_session_dir(task_dir: Path, args: argparse.Namespace) -> Path | None:
    if not getattr(args, "use_task_agent", True):
        return None
    if getattr(args, "nemoclaw_sandbox", None):
        return None
    return task_dir / "openclaw_agent_state" / "sessions"


def task_live_sandbox_session_dir(
    row: dict[str, Any],
    args: argparse.Namespace,
    agent_id: str,
) -> str | None:
    if not getattr(args, "use_task_agent", True):
        return None
    if not getattr(args, "nemoclaw_sandbox", None):
        return None
    if uses_nemoclaw_gateway_task_agent(args):
        return f"/sandbox/.openclaw/agents/{agent_id}/sessions"
    return str(PurePosixPath(nemoclaw_task_workdir(row, args, agent_id)) / "openclaw_agent_state" / "sessions")


PYTHON_SITECUSTOMIZE = """\
from __future__ import annotations

import sys
from pathlib import Path

version = f"{sys.version_info.major}.{sys.version_info.minor}"
candidate_paths = [
    Path("/tmp/.local/lib") / f"python{version}" / "site-packages",
]
for path in candidate_paths:
    text = str(path)
    if path.exists() and text not in sys.path:
        sys.path.insert(0, text)
"""


def write_task_python_sitecustomize(
    row: dict[str, Any],
    workspace_dir: Path,
    args: argparse.Namespace,
    agent_id: str,
) -> None:
    if getattr(args, "nemoclaw_sandbox", None):
        workspace = nemoclaw_task_workdir(row, args, agent_id)
        write_nemoclaw_text_file(args, "/sandbox/sitecustomize.py", PYTHON_SITECUSTOMIZE)
        write_nemoclaw_text_file(
            args,
            str(PurePosixPath(workspace) / "sitecustomize.py"),
            PYTHON_SITECUSTOMIZE,
        )
        return
    workspace_dir.mkdir(parents=True, exist_ok=True)
    (workspace_dir / "sitecustomize.py").write_text(PYTHON_SITECUSTOMIZE, encoding="utf-8")


def disable_remote_lookup_tools(config: dict[str, Any]) -> None:
    tools = config.setdefault("tools", {})
    if not isinstance(tools, dict):
        return
    tools["toolSearch"] = False
    web = tools.setdefault("web", {})
    if isinstance(web, dict):
        fetch = web.setdefault("fetch", {})
        if isinstance(fetch, dict):
            fetch["enabled"] = False


def build_prompt(row: dict[str, Any], max_tool_wall_seconds: int | None = DEFAULT_MAX_TOOL_WALL_SECONDS) -> str:
    benchmark = str(row.get("benchmark", "Agentic Math"))
    answer_format = str(row.get("answer_format", "integer_0_999"))
    wall_limit = int(max_tool_wall_seconds or 0)
    wall_limit_text = f"{wall_limit} seconds" if wall_limit > 0 else "the configured runtime limit"
    if answer_format == "integer_0_999":
        final_answer_instruction = "ANSWER: <integer from 0 to 999>"
        final_answer_note = "Do not put any other number on the ANSWER line."
    else:
        final_answer_instruction = "ANSWER: \\boxed{<exact final answer>}"
        final_answer_note = (
            "Use a concise exact value when possible, such as an integer, fraction, radical, "
            "or interval. Put only the boxed final answer on the ANSWER line."
        )
    return "\n".join(
        [
            "# Agentic Math Task",
            "",
            f"Solve the following {benchmark} problem. You may use the available shell or Python tools",
            "inside the workspace for scratch work, verification, or small computations.",
            "",
            f"請解出下列 {benchmark} 題目。可以使用工作區中的 shell 或 Python 工具作為草稿、驗算或小型計算。",
            "",
            "## Python Tool Guidance",
            "",
            "- Use Python for exact or error-prone calculations, symbolic manipulation, numerical checks,",
            "  small-case brute force, and conjecture testing.",
            "- Use local shell/Python execution in the task workspace; do not use remote code interpreter tools.",
            "- Do not start an interactive shell, Python REPL, notebook, or long-running background process.",
            "- Never call `exec` with `pty=true`; run Python non-interactively with `python3 -c`,",
            "  a heredoc such as `python3 - <<'PY' ... PY`, or a short script file.",
            "- Prefer `sympy` for exact algebra, number theory, equations, and simplification when available.",
            "- Use `math`, `itertools`, and standard-library code for lightweight verification.",
            "- Always print the values you need to inspect, and briefly explain how the computation supports",
            "  the mathematical reasoning.",
            "- Do not rely on unbounded brute force as the solution. Use computation to support a proof,",
            "  verify a closed form, or check edge cases.",
            f"- Each shell/Python execution has a wall-clock limit of {wall_limit_text}.",
            "- Keep computations short and targeted. Do not run broad random searches, long brute-force loops,",
            "  package installs, servers, notebooks, or background jobs.",
            "- If a computation times out, simplify the computation or switch to mathematical reasoning.",
            "- Do not use web search, browser, HTTP, or any external internet lookup.",
            "- If you intend to use a tool, call the tool before writing any `ANSWER:` line.",
            "- Do not write provisional, symbolic, or placeholder answers such as `ANSWER: \\boxed{F_n}`.",
            "- Do not include `ANSWER:` or `\\boxed{...}` in the same assistant turn as a tool call.",
            "",
            "Return your final response with one line exactly in this format:",
            "",
            final_answer_instruction,
            "",
            final_answer_note,
            "",
            "## Metadata",
            "",
            f"- task_id: {row['task_id']}",
            f"- benchmark: {benchmark}",
            f"- source_config: {row.get('source_config', '')}",
            f"- problem_index: {row.get('problem_index', '')}",
            f"- subject: {row.get('subject', '')}",
            "",
            "## Problem",
            "",
            str(row["question"]).strip(),
        ]
    ) + "\n"


def extract_text_from_openclaw(sidecar: dict[str, Any]) -> str:
    stdout_json = sidecar.get("stdout_json")
    if isinstance(stdout_json, dict):
        candidates = [stdout_json]
        result = stdout_json.get("result")
        if isinstance(result, dict):
            candidates.append(result)
            meta = result.get("meta")
            if isinstance(meta, dict):
                candidates.append(meta)
        meta = stdout_json.get("meta")
        if isinstance(meta, dict):
            candidates.append(meta)

        for candidate in candidates:
            payloads = candidate.get("payloads")
            if isinstance(payloads, list):
                texts = [
                    str(payload.get("text", ""))
                    for payload in payloads
                    if isinstance(payload, dict) and payload.get("text") is not None
                ]
                if texts:
                    return "\n\n".join(texts).strip()
        for candidate in candidates:
            for key in ("finalAssistantVisibleText", "finalAssistantRawText"):
                value = candidate.get(key)
                if isinstance(value, str) and value.strip():
                    return value.strip()
    return str(sidecar.get("stdout") or "").strip()


def task_sidecar_path(protocol_output_dir: Path, task_id: str) -> Path:
    return protocol_output_dir / "agentic_math" / task_id / "openclaw_result.json"


def archive_stale_result(task_dir: Path, result_path: Path, cache_key: dict[str, Any]) -> Path | None:
    """Move a stale result out of the active path before a rerun.

    Leaving a mismatched result.json in place is dangerous: if the rerun is
    interrupted, downstream collection can silently pick up the old answer.
    """
    if not result_path.exists():
        return None

    try:
        existing: dict[str, Any] | str = json.loads(result_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        existing = result_path.read_text(encoding="utf-8")

    archive_dir = task_dir / "stale_results"
    archive_dir.mkdir(parents=True, exist_ok=True)
    if isinstance(existing, dict):
        old_hash = str(existing.get("prompt_hash") or existing.get("cache_key", {}).get("prompt_hash") or "unknown")
    else:
        old_hash = "unparseable"
    timestamp = time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())
    archive_path = archive_dir / f"{timestamp}_{old_hash[:12]}.json"
    archive_path.write_text(
        json.dumps(
            {
                "archive_reason": "cache_key_mismatch",
                "runner_version": RUNNER_VERSION,
                "expected_cache_key": cache_key,
                "archived_result": existing,
            },
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    result_path.unlink()
    stale_invocation = task_dir / "openclaw_invocation.json"
    if stale_invocation.exists():
        stale_invocation.unlink()
    return archive_path


def sidecar_error_text(sidecar: dict[str, Any] | None, fallback: str) -> str:
    if sidecar:
        if sidecar_provider_timeout(sidecar):
            stdout_json = sidecar_stdout_json_text(sidecar)
            if stdout_json:
                return stdout_json[-4000:]
        stderr = str(sidecar.get("stderr") or "").strip()
        if stderr:
            return stderr[-4000:]
        stdout_json = sidecar_stdout_json_text(sidecar)
        if stdout_json:
            return stdout_json[-4000:]
    return fallback[-4000:]


OUTER_OPENCLAW_TIMEOUT_RE = re.compile(
    r"\bCommand timed out after \d+(?:\.\d+)? seconds\b",
    flags=re.IGNORECASE,
)

TRANSIENT_OPENCLAW_FAILURE_PATTERNS = [
    r"\bLLM request timed out\b",
    r"\bFailoverError:\s*LLM request timed out\b",
    r"\breason=timeout\b",
    r"\brawError=terminated\b",
    r"\bETIMEDOUT\b",
    r"\bECONNRESET\b",
    r"\bsocket hang up\b",
    r"\b429\b",
    r"\brate limit(?:ed)?\b",
    r"\brate[_ -]?limit(?:ed)?\b",
    r"\bJSON error injected into SSE stream\b",
    r"\blive_provider_timeout\b",
    r"\bprovider_timeout\b",
    r"\bRequest timed out before a response was generated\b",
    r'"status"\s*:\s*"timeout"',
    r'"timeoutPhase"\s*:\s*"provider"',
    r"\btemporarily unavailable\b",
    r"\b503\b",
    r"\b504\b",
    r"\bgateway timeout\b",
]

NON_SCOREABLE_OPENCLAW_FAILURE_PATTERNS = [
    ("unsupported_thinking", r'Thinking level "[^"]+" is not supported'),
    ("insufficient_quota", r"\binsufficient_quota\b|exceeded your current quota"),
    ("authentication", r"\b401\b|unauthorized|invalid[_ -]?api[_ -]?key|incorrect api key|User not found"),
    ("workspace_vanished", r"\bWorkspaceVanishedError\b|workspace appears to have disappeared"),
    ("unknown_model", r"\bmodel .*not found\b|\bunknown model\b|\bNo provider\b"),
    ("nemoclaw_session_audit_failed", r"\bNeMoClaw session audit failed\b"),
]
WORKSPACE_VANISHED_RE = re.compile(
    r"WorkspaceVanishedError:.*?: (?P<workspace>/[^\n]+?)\. Refusing.*?remove (?P<attestation>/[^\s]+\.attested)",
    flags=re.IGNORECASE | re.DOTALL,
)


def openclaw_failure_text(result: subprocess.CompletedProcess[str], sidecar: dict[str, Any] | None) -> str:
    parts = [result.stdout or "", result.stderr or ""]
    if isinstance(sidecar, dict):
        parts.extend(
            [
                str(sidecar.get("stderr") or ""),
                str(sidecar.get("error") or ""),
                sidecar_stdout_json_text(sidecar),
            ]
        )
    return "\n".join(part for part in parts if part)


def is_outer_openclaw_timeout(
    result: subprocess.CompletedProcess[str],
    sidecar: dict[str, Any] | None,
) -> bool:
    if result.returncode != 124:
        return False
    return bool(OUTER_OPENCLAW_TIMEOUT_RE.search(openclaw_failure_text(result, sidecar)))


def sidecar_stdout_json_text(sidecar: dict[str, Any] | None) -> str:
    if not isinstance(sidecar, dict):
        return ""
    stdout_json = sidecar.get("stdout_json")
    if stdout_json is None:
        return ""
    try:
        return json.dumps(stdout_json, ensure_ascii=False, sort_keys=True)
    except TypeError:
        return str(stdout_json)


def sidecar_provider_timeout(sidecar: dict[str, Any] | None) -> bool:
    if not isinstance(sidecar, dict):
        return False
    stdout_json = sidecar.get("stdout_json")
    if not isinstance(stdout_json, dict):
        return False
    status = str(stdout_json.get("status") or "").lower()
    timeout_phase = str(stdout_json.get("timeoutPhase") or "").lower()
    if status == "timeout" or timeout_phase == "provider":
        return True
    text = sidecar_stdout_json_text(sidecar)
    return any(
        re.search(pattern, text, flags=re.IGNORECASE)
        for pattern in TRANSIENT_OPENCLAW_FAILURE_PATTERNS
    )


def is_transient_openclaw_failure(
    result: subprocess.CompletedProcess[str],
    sidecar: dict[str, Any] | None,
) -> bool:
    if isinstance(sidecar, dict):
        if sidecar.get("tool_policy_ok") is False or sidecar.get("tool_policy_violations"):
            return False
        runtime_budget = sidecar.get("runtime_budget")
        if (
            isinstance(runtime_budget, dict)
            and runtime_budget.get("ok") is False
            and runtime_budget.get("violations")
        ):
            return False
    if result.returncode == 0 and not sidecar_provider_timeout(sidecar):
        return False
    text = openclaw_failure_text(result, sidecar)
    return any(re.search(pattern, text, flags=re.IGNORECASE) for pattern in TRANSIENT_OPENCLAW_FAILURE_PATTERNS)


def non_scoreable_openclaw_failure_reason(
    result: subprocess.CompletedProcess[str],
    sidecar: dict[str, Any] | None,
) -> str | None:
    """Return a setup/provider failure reason that must not become an incorrect answer."""
    if result.returncode == 0:
        return None
    text = openclaw_failure_text(result, sidecar)
    for reason, pattern in NON_SCOREABLE_OPENCLAW_FAILURE_PATTERNS:
        if re.search(pattern, text, flags=re.IGNORECASE):
            return reason
    return None


def cached_result_is_reusable(record: dict[str, Any]) -> bool:
    if record.get("scoring_method") != "openclaw_error":
        return True
    text = str(record.get("scoring_error") or "")
    return not any(re.search(pattern, text, flags=re.IGNORECASE) for pattern in TRANSIENT_OPENCLAW_FAILURE_PATTERNS)


def record_nemoclaw_session_audit_matches_cache(record: dict[str, Any], cache_key: dict[str, Any]) -> bool:
    audit = record.get("nemoclaw_session_audit")
    if cache_requires_nemoclaw_session_audit(cache_key):
        return (
            isinstance(audit, dict)
            and audit.get("required") is True
            and audit.get("ok") is True
            and record.get("nemoclaw_session_audit_ok") is True
        )
    if isinstance(audit, dict) and audit.get("required") is True:
        return audit.get("ok") is True and record.get("nemoclaw_session_audit_ok") is True
    return True


def nemoclaw_session_copy_evidence(sidecar: dict[str, Any] | None) -> dict[str, Any]:
    audit = sidecar.get("nemoclaw_session_audit") if isinstance(sidecar, dict) else None
    copy_status = audit.get("copy") if isinstance(audit, dict) else None
    return {
        "nemoclaw_session_copy_source": (
            copy_status.get("source") if isinstance(copy_status, dict) else None
        ),
        "nemoclaw_session_copied_bytes": (
            audit.get("copied_session_bytes") if isinstance(audit, dict) else None
        ),
    }


def record_conversation_order_allows_reuse(record: dict[str, Any]) -> bool:
    order = record.get("conversation_order")
    if record.get("conversation_order_ok") is False:
        return False
    if isinstance(order, dict) and order.get("ok") is False:
        return False
    return True


def record_tool_policy_allows_reuse(record: dict[str, Any]) -> bool:
    if record.get("tool_policy_ok") is False:
        return False
    if record.get("tool_policy_violations"):
        return False
    return True


def record_weave_sidecar_allows_reuse(record: dict[str, Any]) -> bool:
    sidecar = record.get("weave_sidecar")
    if record.get("weave_sidecar_ok") is False:
        return False
    if isinstance(sidecar, dict) and sidecar.get("ok") is False:
        return False
    return True


def record_weave_agents_allows_reuse(record: dict[str, Any], cache_key: dict[str, Any]) -> bool:
    if not cache_key.get("verify_weave_agents"):
        return True
    return (
        record.get("weave_agents_required") is True
        and record.get("weave_agents_ok") is True
        and bool(str(record.get("weave_agents_conversation_id") or "").strip())
        and bool(str(record.get("weave_agents_conversation_url") or "").strip())
        and bool(str(record.get("weave_agents_trace_id") or "").strip())
        and bool(str(record.get("weave_agents_url") or "").strip())
    )


def record_invocation_matches_cache(record: dict[str, Any], cache_key: dict[str, Any]) -> bool:
    if not cache_requires_nemoclaw_session_audit(cache_key):
        return True
    invocation_path_value = record.get("openclaw_invocation_path")
    invocation_sha = record.get("openclaw_invocation_sha256")
    command_sha = record.get("openclaw_command_sha256")
    if not isinstance(invocation_path_value, str) or not invocation_path_value:
        return False
    if not isinstance(invocation_sha, str) or len(invocation_sha) != 64:
        return False
    path = Path(invocation_path_value)
    if not path.exists():
        return False
    try:
        if sha256_file(path) != invocation_sha:
            return False
        invocation = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return False
    if not isinstance(invocation, dict):
        return False
    if invocation.get("cache_key") != cache_key:
        return False
    if isinstance(command_sha, str) and command_sha:
        if invocation.get("command_sha256") != command_sha:
            return False
    elif invocation.get("command"):
        return False
    expected_path = record.get("openclaw_result_path")
    if isinstance(expected_path, str) and expected_path:
        return invocation.get("expected_openclaw_result_path") == expected_path
    return True


def cached_result_matches_cache(record: dict[str, Any], cache_key: dict[str, Any]) -> bool:
    return (
        cache_key_matches(record, cache_key)
        and cached_result_is_reusable(record)
        and record_conversation_order_allows_reuse(record)
        and record_tool_policy_allows_reuse(record)
        and record_weave_sidecar_allows_reuse(record)
        and record_weave_agents_allows_reuse(record, cache_key)
        and record_nemoclaw_session_audit_matches_cache(record, cache_key)
        and record_invocation_matches_cache(record, cache_key)
    )


def append_transient_failure(task_dir: Path, payload: dict[str, Any]) -> None:
    log_path = task_dir / "openclaw_transient_failures.jsonl"
    with log_path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(payload, ensure_ascii=False) + "\n")


def sidecar_conversation_order_failed(sidecar: dict[str, Any] | None) -> bool:
    if not isinstance(sidecar, dict):
        return False
    order = sidecar.get("conversation_order")
    return isinstance(order, dict) and order.get("ok") is False


def recover_workspace_vanished_failure(
    task_dir: Path,
    result: subprocess.CompletedProcess[str],
    sidecar: dict[str, Any] | None,
) -> dict[str, Any] | None:
    text = openclaw_failure_text(result, sidecar)
    match = WORKSPACE_VANISHED_RE.search(text)
    if not match:
        return None
    workspace = Path(match.group("workspace")).resolve()
    attestation = Path(match.group("attestation")).resolve()
    attestation_root = (Path.home() / ".openclaw" / "workspace-attestations").resolve()
    try:
        attestation.relative_to(attestation_root)
    except ValueError:
        return None
    if not workspace.exists() or not attestation.exists():
        return None
    recovery_dir = task_dir / "recovered_workspace_attestations"
    recovery_dir.mkdir(parents=True, exist_ok=True)
    destination = recovery_dir / f"{int(time.time())}_{attestation.name}"
    attestation.replace(destination)
    return {
        "workspace": str(workspace),
        "attestation": str(attestation),
        "moved_to": str(destination),
    }


def build_openclaw_error_record(
    row: dict[str, Any],
    result: subprocess.CompletedProcess[str],
    sidecar_path: Path,
    cache_key: dict[str, Any],
    attempt_metadata: dict[str, Any] | None = None,
) -> dict[str, Any]:
    sidecar = None
    if sidecar_path.exists():
        try:
            sidecar = json.loads(sidecar_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            sidecar = None

    usage = {}
    openclaw_returncode = result.returncode
    if isinstance(sidecar, dict):
        openclaw_returncode = sidecar.get("returncode", result.returncode)
        stdout_json = sidecar.get("stdout_json")
        if isinstance(stdout_json, dict):
            usage = (
                stdout_json.get("meta", {})
                .get("agentMeta", {})
                .get("usage", {})
            )
    runtime_budget = sidecar.get("runtime_budget", {}) if isinstance(sidecar, dict) else {}
    runtime_budget_exceeded = (
        isinstance(runtime_budget, dict) and bool(runtime_budget.get("violations"))
    )
    conversation_order_failed = sidecar_conversation_order_failed(sidecar)
    disqualified_reason = (
        "runtime_budget_exceeded"
        if runtime_budget_exceeded
        else "time_up"
        if is_outer_openclaw_timeout(result, sidecar)
        else "conversation_order_violation"
        if conversation_order_failed
        else ""
    )

    return {
        **row,
        "response": "",
        "predicted_answer": None,
        "gold_answer": normalized_answer(row["answer"]),
        "correct": False,
        "scoring_method": "openclaw_error",
        "scoring_error": sidecar_error_text(sidecar, result.stderr or result.stdout),
        "openclaw_result_path": str(sidecar_path) if sidecar_path.exists() else "",
        "openclaw_returncode": openclaw_returncode,
        "openclaw_usage": usage,
        "openclaw_tool_call_count": sidecar.get("tool_call_count", 0) if isinstance(sidecar, dict) else 0,
        "openclaw_tool_error_count": sidecar.get("tool_error_count", 0) if isinstance(sidecar, dict) else 0,
        "runtime_budget": runtime_budget,
        "openclaw_disqualified_reason": disqualified_reason,
        "tool_policy_ok": sidecar.get("tool_policy_ok") if isinstance(sidecar, dict) else None,
        "tool_policy_violations": sidecar.get("tool_policy_violations", []) if isinstance(sidecar, dict) else [],
        "weave_sidecar": sidecar.get("weave_sidecar", {}) if isinstance(sidecar, dict) else {},
        "weave_sidecar_ok": (
            (sidecar.get("weave_sidecar") or {}).get("ok") if isinstance(sidecar, dict) else None
        ),
        "conversation_order_ok": (
            (sidecar.get("conversation_order") or {}).get("ok") if isinstance(sidecar, dict) else None
        ),
        "conversation_order": sidecar.get("conversation_order", {}) if isinstance(sidecar, dict) else {},
        "nemoclaw_session_audit_ok": (
            (sidecar.get("nemoclaw_session_audit") or {}).get("ok") if isinstance(sidecar, dict) else None
        ),
        "nemoclaw_session_audit": (
            sidecar.get("nemoclaw_session_audit", {}) if isinstance(sidecar, dict) else {}
        ),
        **nemoclaw_session_copy_evidence(sidecar),
        **default_weave_agents_evidence(args=argparse.Namespace(), required=False),
        **(attempt_metadata or {}),
        "openclaw_config_source": cache_key.get("openclaw_config_source", ""),
        "prompt_hash": cache_key["prompt_hash"],
        "runner_version": RUNNER_VERSION,
        "cache_key": cache_key,
    }


def build_scored_record_from_sidecar(
    row: dict[str, Any],
    sidecar_path: Path,
    sidecar: dict[str, Any],
    cache_key: dict[str, Any],
    attempt_metadata: dict[str, Any],
) -> dict[str, Any]:
    response_text = extract_text_from_openclaw(sidecar)
    prediction = extract_answer(response_text)
    gold = normalized_answer(row["answer"])
    score = answers_equivalent(prediction, row["answer"])
    conversation_order_failed = sidecar_conversation_order_failed(sidecar)
    disqualified_reason = (
        "runtime_budget_exceeded"
        if isinstance(sidecar.get("runtime_budget"), dict)
        and sidecar["runtime_budget"].get("violations")
        else "conversation_order_violation"
        if conversation_order_failed
        else ""
    )
    correct = bool(score["equivalent"]) and not disqualified_reason
    return {
        **row,
        "response": response_text,
        "predicted_answer": prediction,
        "gold_answer": gold,
        "correct": correct,
        "scoring_method": score.get("method"),
        "scoring_error": score.get("error", ""),
        "openclaw_result_path": str(sidecar_path),
        "openclaw_returncode": sidecar.get("returncode"),
        "openclaw_usage": (
            sidecar.get("stdout_json", {})
            .get("meta", {})
            .get("agentMeta", {})
            .get("usage", {})
            if isinstance(sidecar.get("stdout_json"), dict)
            else {}
        ),
        "openclaw_tool_call_count": sidecar.get("tool_call_count", 0),
        "openclaw_tool_error_count": sidecar.get("tool_error_count", 0),
        "runtime_budget": sidecar.get("runtime_budget", {}),
        "openclaw_disqualified_reason": disqualified_reason,
        "tool_policy_ok": sidecar.get("tool_policy_ok"),
        "tool_policy_violations": sidecar.get("tool_policy_violations", []),
        "weave_sidecar": sidecar.get("weave_sidecar", {}),
        "weave_sidecar_ok": (sidecar.get("weave_sidecar") or {}).get("ok"),
        "conversation_order_ok": (sidecar.get("conversation_order") or {}).get("ok"),
        "conversation_order": sidecar.get("conversation_order", {}),
        "nemoclaw_session_audit_ok": (sidecar.get("nemoclaw_session_audit") or {}).get("ok"),
        "nemoclaw_session_audit": sidecar.get("nemoclaw_session_audit", {}),
        **nemoclaw_session_copy_evidence(sidecar),
        **default_weave_agents_evidence(args=argparse.Namespace(), required=False),
        **attempt_metadata,
        "openclaw_config_source": cache_key.get("openclaw_config_source", ""),
        "prompt_hash": cache_key["prompt_hash"],
        "runner_version": RUNNER_VERSION,
        "cache_key": cache_key,
    }


def is_weave_sidecar_failure(sidecar: dict[str, Any] | None) -> bool:
    if not isinstance(sidecar, dict):
        return False
    weave_sidecar = sidecar.get("weave_sidecar")
    return (
        isinstance(weave_sidecar, dict)
        and weave_sidecar.get("ok") is False
        and sidecar.get("returncode") == 0
    )


def should_verify_weave_agents(args: argparse.Namespace) -> bool:
    return bool(getattr(args, "verify_weave_agents", False)) and not bool(
        getattr(args, "dry_run", False)
    )


def default_weave_agents_evidence(
    args: argparse.Namespace,
    *,
    required: bool,
    session_key: str = "",
    agent_id: str = "",
) -> dict[str, Any]:
    conversation_key = weave_agents_conversation_key(agent_id=agent_id, session_key=session_key)
    return empty_weave_agents_evidence(
        required=required,
        entity=str(getattr(args, "weave_agents_entity", "") or env_default_entity()),
        project=str(getattr(args, "weave_agents_project", "") or env_default_project()),
        agent_name=str(
            getattr(args, "weave_agents_agent_name", "") or DEFAULT_WEAVE_AGENTS_AGENT_NAME
        ),
        conversation_id=conversation_key,
        conversation_id_contains=conversation_key,
    )


def weave_agents_conversation_key(*, agent_id: str, session_key: str) -> str:
    session_key = str(session_key or "")
    agent_id = str(agent_id or "")
    if not session_key:
        return ""
    if session_key.startswith("agent:") or not agent_id:
        return session_key.lower()
    return f"agent:{agent_id}:{session_key}".lower()


def verify_weave_agents_for_attempt(
    row: dict[str, Any],
    task_dir: Path,
    args: argparse.Namespace,
    *,
    session_key: str,
    agent_id: str,
    sidecar: dict[str, Any],
) -> dict[str, Any]:
    conversation_key = weave_agents_conversation_key(agent_id=agent_id, session_key=session_key)
    if not should_verify_weave_agents(args):
        return default_weave_agents_evidence(
            args,
            required=False,
            session_key=session_key,
            agent_id=agent_id,
        )
    tool_count = int(sidecar.get("tool_call_count") or 0)
    verifier_json = task_dir / "weave_agents_verifications" / f"{safe_id(conversation_key)}.json"
    entity = str(getattr(args, "weave_agents_entity", "") or env_default_entity())
    project = str(getattr(args, "weave_agents_project", "") or env_default_project())
    agent_name = str(
        getattr(args, "weave_agents_agent_name", "") or DEFAULT_WEAVE_AGENTS_AGENT_NAME
    )
    try:
        return verify_native_weave_agents_trace(
            entity=entity,
            project=project,
            agent_name=agent_name,
            conversation_id_contains=conversation_key,
            verifier_json=verifier_json,
            env_file=Path(getattr(args, "weave_agents_env_file", DEFAULT_WEAVE_AGENTS_ENV_FILE)),
            expected_model=str(getattr(args, "model", "") or ""),
            required_texts=[f"task_id: {row['task_id']}"],
            require_tool_trace=tool_count > 0,
            require_usage=True,
            limit=int(getattr(args, "weave_agents_limit", 50) or 50),
            timeout_seconds=float(getattr(args, "weave_agents_verification_timeout", 120.0) or 0),
            poll_seconds=float(getattr(args, "weave_agents_poll_seconds", 5.0) or 5.0),
        )
    except Exception as exc:
        error = str(exc)
        print(
            "Native weave-openclaw Agents trace verification failed for "
            f"{row['task_id']}; recording per-instance evidence failure and continuing: "
            f"{error}",
            file=sys.stderr,
            flush=True,
        )
        evidence = empty_weave_agents_evidence(
            required=True,
            entity=entity,
            project=project,
            agent_name=agent_name,
            conversation_id=conversation_key,
            conversation_id_contains=conversation_key,
            verifier_json=str(verifier_json),
            error=error,
        )
        evidence["weave_agents_ok"] = False
        return evidence


def sidecar_nemoclaw_session_audit_matches_cache(sidecar: dict[str, Any], cache_key: dict[str, Any]) -> bool:
    audit = sidecar.get("nemoclaw_session_audit")
    if cache_requires_nemoclaw_session_audit(cache_key):
        return (
            isinstance(audit, dict)
            and audit.get("required") is True
            and audit.get("ok") is True
        )
    if isinstance(audit, dict) and audit.get("required") is True:
        return audit.get("ok") is True
    return True


def sidecar_matches_cache(sidecar: dict[str, Any], cache_key: dict[str, Any]) -> bool:
    metadata = sidecar.get("metadata") if isinstance(sidecar.get("metadata"), dict) else {}
    policy = sidecar.get("tool_policy") if isinstance(sidecar.get("tool_policy"), dict) else {}
    expected_config = cache_key.get("openclaw_config_source")
    observed_config = metadata.get("openclaw_config_source")
    return (
        sidecar.get("returncode") == 0
        and metadata.get("task_id") == cache_key["task_id"]
        and metadata.get("prompt_hash") == cache_key["prompt_hash"]
        and (not cache_key.get("model") or metadata.get("model_id") == cache_key.get("model"))
        and (not expected_config or observed_config == expected_config)
        and sorted(policy.get("deny_tools") or []) == sorted(cache_key.get("deny_tools") or [])
        and sorted(policy.get("deny_argument_patterns") or [])
        == sorted(cache_key.get("deny_argument_patterns") or [])
        and sidecar.get("tool_policy_ok") is not False
        and not sidecar.get("tool_policy_violations")
        and (not isinstance(sidecar.get("conversation_order"), dict) or sidecar["conversation_order"].get("ok") is not False)
        and sidecar_nemoclaw_session_audit_matches_cache(sidecar, cache_key)
    )


def attempt_metadata_from_sidecar_path(task_dir: Path, sidecar_path: Path) -> dict[str, Any]:
    attempt_id = ""
    try:
        parts = sidecar_path.relative_to(task_dir).parts
    except ValueError:
        parts = ()
    if len(parts) >= 2 and parts[0] == "openclaw_attempts":
        attempt_id = parts[1]
    invocation_path = (
        task_dir / "openclaw_invocations" / f"{attempt_id}.json"
        if attempt_id
        else task_dir / "openclaw_invocation.json"
    )
    metadata = {
        "openclaw_attempt_id": attempt_id,
        "openclaw_attempt_output_dir": str(task_dir / "openclaw_attempts" / attempt_id) if attempt_id else "",
        "openclaw_invocation_path": str(invocation_path),
    }
    if invocation_path.exists():
        try:
            invocation = json.loads(invocation_path.read_text(encoding="utf-8"))
            metadata["openclaw_invocation_sha256"] = sha256_file(invocation_path)
            command = invocation.get("command") if isinstance(invocation, dict) else None
            if isinstance(command, list):
                metadata["openclaw_command_sha256"] = invocation.get("command_sha256") or command_sha256(command)
        except (OSError, json.JSONDecodeError):
            pass
    return metadata


def sidecar_invocation_matches_cache(task_dir: Path, sidecar_path: Path, cache_key: dict[str, Any]) -> bool:
    metadata = attempt_metadata_from_sidecar_path(task_dir, sidecar_path)
    invocation_path_value = metadata.get("openclaw_invocation_path")
    if not isinstance(invocation_path_value, str) or not invocation_path_value:
        return False
    invocation_path = Path(invocation_path_value)
    if not invocation_path.exists():
        return False
    try:
        invocation = json.loads(invocation_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return False
    return invocation.get("cache_key") == cache_key


def relog_existing_sidecar(sidecar_path: Path, args: argparse.Namespace) -> dict[str, Any]:
    _ = (sidecar_path, args)
    raise RuntimeError(
        "Manual Weave sidecar relogging is disabled. Use native weave-openclaw "
        "Agents traces only; JSON relog traces are not valid evidence."
    )


def recover_existing_success_sidecar(
    row: dict[str, Any],
    task_dir: Path,
    cache_key: dict[str, Any],
    args: argparse.Namespace,
) -> dict[str, Any] | None:
    candidates = sorted(
        task_dir.glob(f"openclaw_attempts/*/agentic_math/{row['task_id']}/openclaw_result.json"),
        key=lambda path: path.stat().st_mtime,
        reverse=True,
    )
    for sidecar_path in candidates:
        try:
            sidecar = json.loads(sidecar_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            continue
        if not sidecar_matches_cache(sidecar, cache_key):
            continue
        if not sidecar_invocation_matches_cache(task_dir, sidecar_path, cache_key):
            continue
        if args.weave_sidecar and (is_weave_sidecar_failure(sidecar) or not sidecar.get("weave_sidecar")):
            if not args.weave_sidecar_strict:
                continue
            print(f"Relogging successful OpenClaw sidecar to Weave: {sidecar_path}", flush=True)
            sidecar = relog_existing_sidecar(sidecar_path, args)
            if not sidecar_matches_cache(sidecar, cache_key):
                continue
        if is_weave_sidecar_failure(sidecar):
            continue
        return build_scored_record_from_sidecar(
            row,
            sidecar_path,
            sidecar,
            cache_key,
            attempt_metadata_from_sidecar_path(task_dir, sidecar_path),
        )
    return None


def _strip_wrapping_math(text: str) -> str:
    text = text.strip()
    wrappers = [
        ("\\(", "\\)"),
        ("\\[", "\\]"),
        ("$", "$"),
    ]
    changed = True
    while changed:
        changed = False
        for left, right in wrappers:
            if text.startswith(left) and text.endswith(right):
                text = text[len(left) : len(text) - len(right)].strip()
                changed = True
    return text


def _balanced_argument(text: str, open_index: int, open_char: str = "{", close_char: str = "}") -> tuple[str, int] | None:
    if open_index >= len(text) or text[open_index] != open_char:
        return None
    depth = 0
    start = open_index + 1
    index = open_index
    while index < len(text):
        char = text[index]
        if char == open_char:
            depth += 1
        elif char == close_char:
            depth -= 1
            if depth == 0:
                return text[start:index], index + 1
        index += 1
    return None


def _extract_command_arguments(text: str, command_start: int, command: str, count: int) -> tuple[list[str], int] | None:
    index = command_start + len(command)
    args: list[str] = []
    for _ in range(count):
        while index < len(text) and text[index].isspace():
            index += 1
        parsed = _balanced_argument(text, index)
        if parsed is None:
            return None
        arg, index = parsed
        args.append(arg)
    return args, index


def _extract_optional_bracket(text: str, index: int) -> tuple[str | None, int]:
    while index < len(text) and text[index].isspace():
        index += 1
    if index >= len(text) or text[index] != "[":
        return None, index
    parsed = _balanced_argument(text, index, "[", "]")
    if parsed is None:
        return None, index
    return parsed


def _replace_latex_commands(text: str) -> str:
    commands = ("\\dfrac", "\\tfrac", "\\frac")
    changed = True
    while changed:
        changed = False
        for command in commands:
            start = text.find(command)
            if start < 0:
                continue
            parsed = _extract_command_arguments(text, start, command, 2)
            if parsed is None:
                continue
            (num, den), end = parsed
            replacement = f"(({_replace_latex_commands(num)})/({_replace_latex_commands(den)}))"
            text = text[:start] + replacement + text[end:]
            changed = True
            break

    while True:
        start = text.find("\\sqrt")
        if start < 0:
            break
        index = start + len("\\sqrt")
        degree, index = _extract_optional_bracket(text, index)
        while index < len(text) and text[index].isspace():
            index += 1
        parsed = _balanced_argument(text, index)
        if parsed is None:
            break
        radicand, end = parsed
        radicand = _replace_latex_commands(radicand)
        if degree:
            replacement = f"(({radicand})**(1/({_replace_latex_commands(degree)})))"
        else:
            replacement = f"sqrt({radicand})"
        text = text[:start] + replacement + text[end:]
    return text


def clean_answer_text(value: Any) -> str:
    text = "" if value is None else str(value)
    text = text.strip()
    text = re.sub(r"(?is)^.*?(?:answer|答案)\s*[:：]\s*", "", text).strip()
    text = re.sub(r"(?i)^(?:the\s+)?(?:final\s+)?answer\s+is\s+", "", text).strip()
    text = text.strip(" \t\r\n。；;")
    text = _strip_wrapping_math(text)
    boxed = extract_boxed_argument(text)
    if boxed is not None and boxed.strip():
        text = boxed.strip()
    return _strip_wrapping_math(text.strip())


def canonical_answer_string(value: Any) -> str:
    text = clean_answer_text(value)
    if re.fullmatch(r"[+-]?0*[0-9]+", text):
        return str(int(text))
    return text


def format_for_math_verify(value: Any) -> str:
    answer = clean_answer_text(value)
    if not answer:
        return "$.$"
    if answer.startswith("$"):
        answer = answer[1:]
    if answer.endswith("$"):
        answer = answer[:-1]
    answer = answer.strip()
    return f"${answer}$" if answer else "$.$"


def string_compare_answers(extracted: Any, gold: Any) -> bool:
    def normalize(text: Any) -> str:
        normalized = "" if text is None else str(text)
        normalized = re.sub(r"\s+", "", normalized)
        normalized = normalized.replace("\\frac", "")
        normalized = normalized.replace("\\cdot", "*").replace("\\times", "*")
        normalized = re.sub(r"\\[a-zA-Z]+", "", normalized)
        return normalized

    extracted_norm = normalize(extracted)
    gold_norm = normalize(gold)
    return (
        extracted_norm == gold_norm
        or (gold_norm and gold_norm in extracted_norm)
        or (extracted_norm and extracted_norm in gold_norm)
    )


def math_verify_compare(prediction: Any, gold: Any) -> dict[str, Any]:
    try:
        from math_verify import parse, verify
    except Exception as exc:
        return {"available": False, "equivalent": None, "error": str(exc)}

    try:
        gold_parsed = parse(format_for_math_verify(gold))
        pred_parsed = parse(format_for_math_verify(prediction))
        return {
            "available": True,
            "equivalent": bool(verify(gold_parsed, pred_parsed)),
            "error": "",
        }
    except Exception as exc:
        return {"available": True, "equivalent": None, "error": str(exc)}


def extract_boxed_argument(text: str) -> str | None:
    matches: list[str] = []
    search_from = 0
    while True:
        start = text.find("\\boxed", search_from)
        if start < 0:
            break
        index = start + len("\\boxed")
        while index < len(text) and text[index].isspace():
            index += 1
        parsed = _balanced_argument(text, index)
        if parsed is None:
            search_from = index + 1
            continue
        arg, end = parsed
        matches.append(arg)
        search_from = end
    return matches[-1] if matches else None


def extract_answer(text: str) -> str | None:
    answer_line_matches = ANSWER_LINE_RE.findall(text or "")
    for candidate in reversed(answer_line_matches):
        candidate = canonical_answer_string(candidate)
        if candidate:
            return candidate

    boxed = extract_boxed_argument(text or "")
    if boxed is not None:
        candidate = canonical_answer_string(boxed)
        if candidate:
            return candidate

    for pattern in (ANSWER_RE, BOXED_RE):
        matches = pattern.findall(text or "")
        if matches:
            return str(int(matches[-1]))
    lines = [line.strip() for line in (text or "").splitlines() if line.strip()]
    if lines:
        matches = INT_RE.findall(lines[-1])
        if matches:
            return str(int(matches[-1]))
    return None


def normalized_answer(value: Any) -> str:
    return canonical_answer_string(value)


def _parse_expr_text(text: str):
    from sympy import E, I, pi, sqrt
    from sympy.parsing.sympy_parser import (
        convert_xor,
        implicit_multiplication_application,
        parse_expr,
        standard_transformations,
    )

    cleaned = clean_answer_text(text)
    cleaned = cleaned.replace("\\left", "").replace("\\right", "")
    cleaned = cleaned.replace("\\,", "").replace("\\;", "")
    cleaned = cleaned.replace("\\cdot", "*").replace("\\times", "*")
    cleaned = cleaned.replace("×", "*").replace("·", "*")
    cleaned = cleaned.replace("\\pi", "pi").replace("π", "pi")
    cleaned = cleaned.replace("\\infty", "oo").replace("∞", "oo")
    cleaned = _replace_latex_commands(cleaned)
    cleaned = cleaned.replace("^", "**")
    cleaned = cleaned.replace("{", "(").replace("}", ")")
    cleaned = cleaned.replace("[", "(").replace("]", ")")
    cleaned = cleaned.strip()
    transformations = standard_transformations + (implicit_multiplication_application, convert_xor)
    return parse_expr(
        cleaned,
        local_dict={"sqrt": sqrt, "pi": pi, "E": E, "I": I},
        transformations=transformations,
        evaluate=True,
    )


def _split_top_level_commas(text: str) -> list[str]:
    parts: list[str] = []
    start = 0
    stack: list[str] = []
    pairs = {"(": ")", "[": "]", "{": "}"}
    closing = set(pairs.values())
    for index, char in enumerate(text):
        if char in pairs:
            stack.append(pairs[char])
        elif char in closing and stack and stack[-1] == char:
            stack.pop()
        elif char == "," and not stack:
            parts.append(text[start:index].strip())
            start = index + 1
    parts.append(text[start:].strip())
    return parts


def _parse_interval(text: str) -> dict[str, Any] | None:
    cleaned = clean_answer_text(text).replace("\\left", "").replace("\\right", "").strip()
    inequality = _parse_chained_interval_inequality(cleaned)
    if inequality is not None:
        return inequality
    if len(cleaned) >= 5 and cleaned[0] in "[(" and cleaned[-1] in "])":
        inside = cleaned[1:-1].strip()
        parts = _split_top_level_commas(inside)
        if len(parts) != 2:
            return None
        return {
            "left_closed": cleaned[0] == "[",
            "right_closed": cleaned[-1] == "]",
            "left": _parse_expr_text(parts[0]),
            "right": _parse_expr_text(parts[1]),
        }
    return None


def _parse_chained_interval_inequality(text: str) -> dict[str, Any] | None:
    normalized = text
    replacements = {
        "\\leq": "<=",
        "\\le": "<=",
        "≤": "<=",
        "\\geq": ">=",
        "\\ge": ">=",
        "≥": ">=",
    }
    for old, new in replacements.items():
        normalized = normalized.replace(old, new)
    parts = re.split(r"\s*(<=|>=|<|>)\s*", normalized)
    if len(parts) != 5:
        return None
    left_text, left_op, middle, right_op, right_text = [part.strip() for part in parts]
    if not re.fullmatch(r"(?:[A-Za-z]|\\[A-Za-z]+)(?:[A-Za-z0-9_']*)?", middle):
        return None
    if left_op in {"<=", "<"} and right_op in {"<=", "<"}:
        return {
            "left_closed": left_op == "<=",
            "right_closed": right_op == "<=",
            "left": _parse_expr_text(left_text),
            "right": _parse_expr_text(right_text),
        }
    if left_op in {">=", ">"} and right_op in {">=", ">"}:
        return {
            "left_closed": right_op == ">=",
            "right_closed": left_op == ">=",
            "left": _parse_expr_text(right_text),
            "right": _parse_expr_text(left_text),
        }
    return None


def _sympy_equal(left: Any, right: Any) -> bool:
    from sympy import N, simplify

    diff = simplify(left - right)
    if diff == 0:
        return True
    try:
        return abs(float(N(diff, 30))) <= 1e-9
    except Exception:
        return False


def answers_equivalent(prediction: Any, gold: Any) -> dict[str, Any]:
    if prediction is None:
        return {"equivalent": False, "method": "missing_prediction"}

    pred_text = canonical_answer_string(prediction)
    gold_text = canonical_answer_string(gold)
    if pred_text == gold_text:
        return {"equivalent": True, "method": "exact"}

    pred_norm = re.sub(r"\s+", "", pred_text).lower()
    gold_norm = re.sub(r"\s+", "", gold_text).lower()
    if pred_norm == gold_norm:
        return {"equivalent": True, "method": "normalized_text"}

    math_verify_result = math_verify_compare(pred_text, gold_text)
    if math_verify_result["equivalent"] is True:
        return {"equivalent": True, "method": "math_verify"}

    try:
        pred_interval = _parse_interval(pred_text)
        gold_interval = _parse_interval(gold_text)
        if pred_interval is not None or gold_interval is not None:
            equivalent = (
                pred_interval is not None
                and gold_interval is not None
                and pred_interval["left_closed"] == gold_interval["left_closed"]
                and pred_interval["right_closed"] == gold_interval["right_closed"]
                and _sympy_equal(pred_interval["left"], gold_interval["left"])
                and _sympy_equal(pred_interval["right"], gold_interval["right"])
            )
            return {"equivalent": equivalent, "method": "interval_sympy"}

        pred_parts = _split_top_level_commas(clean_answer_text(pred_text))
        gold_parts = _split_top_level_commas(clean_answer_text(gold_text))
        if len(pred_parts) > 1 or len(gold_parts) > 1:
            equivalent = len(pred_parts) == len(gold_parts) and all(
                _sympy_equal(_parse_expr_text(pred_part), _parse_expr_text(gold_part))
                for pred_part, gold_part in zip(pred_parts, gold_parts)
            )
            return {"equivalent": equivalent, "method": "tuple_sympy"}

        equivalent = _sympy_equal(_parse_expr_text(pred_text), _parse_expr_text(gold_text))
        return {"equivalent": equivalent, "method": "expression_sympy"}
    except Exception as exc:
        if string_compare_answers(pred_text, gold_text):
            return {"equivalent": True, "method": "string_compare"}
        if math_verify_result["equivalent"] is False:
            return {"equivalent": False, "method": "math_verify"}
        math_verify_error = math_verify_result.get("error", "")
        error = f"math_verify={math_verify_error}; fallback={exc}" if math_verify_error else str(exc)
        return {"equivalent": False, "method": "parse_error", "error": error}


def run_openclaw_for_task(
    row: dict[str, Any],
    task_dir: Path,
    args: argparse.Namespace,
) -> dict[str, Any]:
    result_path = task_dir / "result.json"
    prompt_text = build_prompt(row, max_tool_wall_seconds=int(getattr(args, "max_tool_wall_seconds", 0) or 0))
    cache_key = build_cache_key(row, prompt_text, args)
    if result_path.exists() and not args.redo:
        existing = json.loads(result_path.read_text(encoding="utf-8"))
        if cached_result_matches_cache(existing, cache_key):
            print(f"Reusing existing Agentic Math result: {row['task_id']}", flush=True)
            return existing
        print(
            f"Existing Agentic Math result is stale for {row['task_id']}; "
            "cache key changed or cached OpenClaw error is transient, rerunning.",
            flush=True,
        )
        archive_path = archive_stale_result(task_dir, result_path, cache_key)
        if archive_path:
            print(f"Archived stale result: {archive_path}", flush=True)

    if not args.redo:
        recovered = recover_existing_success_sidecar(row, task_dir, cache_key, args)
        if recovered is not None:
            result_path.write_text(
                json.dumps(recovered, ensure_ascii=False, indent=2) + "\n",
                encoding="utf-8",
            )
            print(f"Recovered existing Agentic Math sidecar: {row['task_id']}", flush=True)
            return recovered

    workspace_dir = task_dir / "workspace"
    workspace_dir.mkdir(parents=True, exist_ok=True)
    prompt_file = task_dir / "prompt.md"
    prompt_file.write_text(prompt_text, encoding="utf-8")
    agent_id, openclaw_config_path = write_task_openclaw_config(row, workspace_dir, task_dir, args)
    write_task_python_sitecustomize(row, workspace_dir, args, agent_id)

    invocation_dir = task_dir / "openclaw_invocations"
    invocation_dir.mkdir(parents=True, exist_ok=True)
    max_attempts = max(1, int(args.openclaw_max_attempts))
    last_result: subprocess.CompletedProcess[str] | None = None
    last_sidecar_path: Path | None = None
    last_attempt_metadata: dict[str, Any] = {}
    for attempt_number in range(1, max_attempts + 1):
        attempt_id = f"{int(time.time())}-{os.getpid()}-{attempt_number}"
        session_key = f"{resolve_session_prefix(args)}:{row['task_id']}:{attempt_id}"
        attempt_output_dir = task_dir / "openclaw_attempts" / attempt_id
        sidecar_path = task_sidecar_path(attempt_output_dir, str(row["task_id"]))
        invocation_path = invocation_dir / f"{attempt_id}.json"
        attempt_metadata = {
            "openclaw_attempt_id": attempt_id,
            "openclaw_attempt_number": attempt_number,
            "openclaw_max_attempts": max_attempts,
            "openclaw_attempt_output_dir": str(attempt_output_dir),
            "openclaw_invocation_path": str(invocation_path),
            "session_key": session_key,
        }
        command = [
            sys.executable,
            str(PROTOCOL_RUNNER),
            "run",
            "--benchmark-id",
            "agentic_math",
            "--task-id",
            str(row["task_id"]),
            "--prompt-file",
            str(prompt_file),
            "--agent",
            agent_id,
            "--session-key",
            session_key,
            "--output-dir",
            str(attempt_output_dir),
            "--cwd",
            str(workspace_dir),
            "--timeout",
            str(args.openclaw_timeout),
            "--thinking",
            args.thinking,
            "--max-input-tokens",
            str(int(getattr(args, "max_input_tokens", 0) or 0)),
            "--max-cumulative-input-tokens",
            str(resolved_max_cumulative_input_tokens(args)),
            "--max-cumulative-output-tokens",
            str(resolved_max_cumulative_output_tokens(args)),
            "--max-tool-calls",
            str(int(getattr(args, "max_tool_calls", 0) or 0)),
            "--max-agent-turns",
            str(int(getattr(args, "max_agent_turns", 0) or 0)),
            "--max-tool-wall-seconds",
            str(int(getattr(args, "max_tool_wall_seconds", 0) or 0)),
        ]
        if bool(getattr(args, "require_actual_token_usage", False)):
            command.append("--require-actual-token-usage")
        if args.profile:
            command.extend(["--profile", args.profile])
        if args.model:
            command.extend(["--model", args.model])
        if args.nemoclaw_sandbox:
            command.extend(["--nemoclaw-sandbox", args.nemoclaw_sandbox])
            command.extend(["--nemoclaw-bin", args.nemoclaw_bin])
            nemoclaw_workdir = (
                nemoclaw_task_workdir(row, args, agent_id)
                if args.use_task_agent
                else args.nemoclaw_workdir
            )
            command.extend(["--nemoclaw-workdir", nemoclaw_workdir])
        if args.no_local:
            command.append("--no-local")
        if args.allow_failed_preflight:
            command.append("--allow-failed-preflight")
        command.append("--no-weave-sidecar")
        for denied_tool in effective_deny_tools(args):
            command.extend(["--deny-tool", denied_tool])
        for pattern in effective_deny_argument_patterns(args):
            command.extend(["--deny-argument-pattern", pattern])
        if openclaw_config_path:
            command.extend(["--openclaw-config-path", str(openclaw_config_path)])
        command.extend(["--openclaw-config-source", str(cache_key["openclaw_config_source"])])
        live_session_dir = task_live_session_dir(task_dir, args)
        if live_session_dir is not None:
            command.extend(["--live-session-dir", str(live_session_dir)])
        live_sandbox_session_dir = task_live_sandbox_session_dir(row, args, agent_id)
        if live_sandbox_session_dir is not None:
            command.extend(["--live-sandbox-session-dir", live_sandbox_session_dir])
        if args.dry_run:
            command.append("--dry-run")

        result = run_command(command, cwd=REPO_ROOT, timeout=args.openclaw_timeout + 60, check=False)
        last_result = result
        last_sidecar_path = sidecar_path
        last_attempt_metadata = attempt_metadata
        invocation = {
            "attempt_id": attempt_id,
            "attempt_number": attempt_number,
            "max_attempts": max_attempts,
            "command": command,
            "command_sha256": command_sha256(command),
            "session_key": session_key,
            "runner_version": RUNNER_VERSION,
            "cache_key": cache_key,
            "expected_openclaw_result_path": str(sidecar_path),
            "returncode": result.returncode,
            "stdout": result.stdout,
            "stderr": result.stderr,
        }
        invocation_path.write_text(
            json.dumps(invocation, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        attempt_metadata["openclaw_invocation_sha256"] = sha256_file(invocation_path)
        attempt_metadata["openclaw_command_sha256"] = invocation["command_sha256"]
        (task_dir / "openclaw_invocation.json").write_text(
            json.dumps(
                {
                    "latest_invocation_path": str(invocation_path),
                    "latest_attempt_id": attempt_id,
                    "latest_openclaw_result_path": str(sidecar_path),
                },
                ensure_ascii=False,
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )

        sidecar = None
        if sidecar_path.exists():
            try:
                sidecar = json.loads(sidecar_path.read_text(encoding="utf-8"))
            except json.JSONDecodeError:
                sidecar = None
        if result.returncode == 0 and not is_transient_openclaw_failure(result, sidecar):
            break

        if result_path.exists() and not args.redo:
            existing = json.loads(result_path.read_text(encoding="utf-8"))
            if cached_result_matches_cache(existing, cache_key):
                print(
                    f"OpenClaw returned {result.returncode} for {row['task_id']}, "
                    "but a matching result.json is available; reusing it.",
                    flush=True,
                )
                return existing

        if is_weave_sidecar_failure(sidecar):
            raise RuntimeError(
                f"Diagnostic Weave sidecar logging failed for {row['task_id']}.\n"
                f"stdout:\n{result.stdout}\n"
                f"stderr:\n{result.stderr}"
            )
        non_scoreable_reason = non_scoreable_openclaw_failure_reason(result, sidecar)
        if non_scoreable_reason:
            recovery = None
            if non_scoreable_reason == "workspace_vanished" and attempt_number < max_attempts:
                recovery = recover_workspace_vanished_failure(task_dir, result, sidecar)
            if recovery:
                append_transient_failure(
                    task_dir,
                    {
                        "task_id": row["task_id"],
                        "attempt_id": attempt_id,
                        "attempt_number": attempt_number,
                        "max_attempts": max_attempts,
                        "returncode": result.returncode,
                        "openclaw_result_path": str(sidecar_path),
                        "failure_text": sidecar_error_text(sidecar, openclaw_failure_text(result, sidecar)),
                        "recovered_non_scoreable_reason": non_scoreable_reason,
                        "recovery": recovery,
                    },
                )
                delay = max(0.0, float(args.openclaw_retry_base_seconds)) * attempt_number
                print(
                    f"OpenClaw recoverable workspace attestation failure for {row['task_id']} on attempt "
                    f"{attempt_number}/{max_attempts}; retrying after {delay:.1f}s.",
                    flush=True,
                )
                if delay:
                    time.sleep(delay)
                continue
            failure_text = sidecar_error_text(sidecar, openclaw_failure_text(result, sidecar))
            raise RuntimeError(
                f"OpenClaw non-scoreable failure for {row['task_id']} "
                f"({non_scoreable_reason}). This is an execution/configuration failure, "
                f"not a benchmark incorrect answer.\n{failure_text}"
            )
        if is_transient_openclaw_failure(result, sidecar):
            append_transient_failure(
                task_dir,
                {
                    "task_id": row["task_id"],
                    "attempt_id": attempt_id,
                    "attempt_number": attempt_number,
                    "max_attempts": max_attempts,
                    "returncode": result.returncode,
                    "openclaw_result_path": str(sidecar_path),
                    "failure_text": sidecar_error_text(sidecar, openclaw_failure_text(result, sidecar)),
                    "exhausted": attempt_number >= max_attempts,
                },
            )
            if attempt_number < max_attempts:
                delay = max(0.0, float(args.openclaw_retry_base_seconds)) * attempt_number
                print(
                    f"OpenClaw transient failure for {row['task_id']} on attempt "
                    f"{attempt_number}/{max_attempts}; retrying after {delay:.1f}s.",
                    flush=True,
                )
                if delay:
                    time.sleep(delay)
                continue
            attempt_metadata["openclaw_disqualified_reason"] = "provider_transient_exhausted"
            print(
                f"OpenClaw transient failure exhausted for {row['task_id']} after "
                f"{attempt_number}/{max_attempts} attempts; recording as provider_transient_exhausted.",
                flush=True,
            )

        if isinstance(sidecar, dict):
            attempt_metadata.update(
                verify_weave_agents_for_attempt(
                    row,
                    task_dir,
                    args,
                    session_key=session_key,
                    agent_id=agent_id,
                    sidecar=sidecar,
                )
            )
        record = build_openclaw_error_record(row, result, sidecar_path, cache_key, attempt_metadata)
        result_path.write_text(
            json.dumps(record, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        print(
            f"OpenClaw failed for {row['task_id']} with return code {result.returncode}; "
            "recording as incorrect and continuing.",
            flush=True,
        )
        if args.fail_fast:
            raise RuntimeError(
                f"OpenClaw failed for {row['task_id']} with return code {result.returncode}\n"
                f"stdout:\n{result.stdout}\n"
                f"stderr:\n{result.stderr}"
            )
        return record

    if last_result is None or last_sidecar_path is None:
        raise RuntimeError(f"OpenClaw did not run for {row['task_id']}")
    result = last_result
    sidecar_path = last_sidecar_path
    attempt_metadata = last_attempt_metadata

    sidecar = json.loads(sidecar_path.read_text(encoding="utf-8"))
    if not sidecar_nemoclaw_session_audit_matches_cache(sidecar, cache_key):
        raise RuntimeError(
            f"OpenClaw NeMoClaw session audit mismatch for {row['task_id']}: "
            f"{sidecar_path}"
        )
    if not sidecar_matches_cache(sidecar, cache_key):
        raise RuntimeError(
            f"OpenClaw sidecar metadata mismatch for {row['task_id']}: "
            f"{sidecar_path}"
        )
    if is_weave_sidecar_failure(sidecar):
        raise RuntimeError(
            f"Diagnostic Weave sidecar logging failed for {row['task_id']}.\n"
            f"stdout:\n{result.stdout}\n"
            f"stderr:\n{result.stderr}"
        )
    attempt_metadata.update(
        verify_weave_agents_for_attempt(
            row,
            task_dir,
            args,
            session_key=str(attempt_metadata.get("session_key") or ""),
            agent_id=agent_id,
            sidecar=sidecar,
        )
    )
    record = build_scored_record_from_sidecar(row, sidecar_path, sidecar, cache_key, attempt_metadata)
    result_path.write_text(
        json.dumps(record, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return record


def write_summary(output_dir: Path, results: list[dict[str, Any]], args: argparse.Namespace) -> dict[str, Any]:
    total = len(results)
    scored = [row for row in results if row.get("predicted_answer") is not None]
    correct = [row for row in results if row.get("correct") is True]
    tool_called = [row for row in results if int(row.get("openclaw_tool_call_count") or 0) > 0]
    tool_errors = [row for row in results if int(row.get("openclaw_tool_error_count") or 0) > 0]
    tool_policy_violations = [row for row in results if row.get("tool_policy_violations")]
    conversation_order_violations = [row for row in results if row.get("conversation_order_ok") is False]
    runtime_budget_exceeded = [
        row
        for row in results
        if row.get("openclaw_disqualified_reason") == "runtime_budget_exceeded"
    ]
    session_audit_required = [
        row
        for row in results
        if isinstance(row.get("nemoclaw_session_audit"), dict)
        and row["nemoclaw_session_audit"].get("required") is True
    ]
    session_audit_passed = [row for row in session_audit_required if row.get("nemoclaw_session_audit_ok") is True]
    session_audit_failed = [row for row in session_audit_required if row.get("nemoclaw_session_audit_ok") is False]
    weave_agents_required = [row for row in results if row.get("weave_agents_required") is True]
    weave_agents_passed = [row for row in weave_agents_required if row.get("weave_agents_ok") is True]
    weave_agents_failed = [row for row in weave_agents_required if row.get("weave_agents_ok") is not True]
    by_subject: dict[str, dict[str, Any]] = {}
    for row in results:
        subject = str(row.get("subject") or "unknown")
        bucket = by_subject.setdefault(subject, {"total": 0, "correct": 0})
        bucket["total"] += 1
        if row.get("correct") is True:
            bucket["correct"] += 1
    for bucket in by_subject.values():
        bucket["accuracy"] = bucket["correct"] / bucket["total"] if bucket["total"] else 0.0
    summary = {
        "total_instances": total,
        "answered_instances": len(scored),
        "correct_instances": len(correct),
        "incorrect_instances": total - len(correct),
        "accuracy": len(correct) / total if total else 0.0,
        "correctness": len(correct) / total if total else 0.0,
        "correct_ids": [row["task_id"] for row in correct],
        "incorrect_ids": [row["task_id"] for row in results if row.get("correct") is not True],
        "model": args.model,
        "thinking": args.thinking,
        "runner_version": RUNNER_VERSION,
        "num_workers": int(getattr(args, "num_workers", 1) or 1),
        "task_start_min_interval_seconds": float(
            getattr(args, "task_start_min_interval_seconds", 0.0) or 0.0
        ),
        "weave_sidecar": args.weave_sidecar,
        "weave_sidecar_strict": args.weave_sidecar_strict,
        "tool_called_instances": len(tool_called),
        "tool_error_instances": len(tool_errors),
        "tool_policy_violation_instances": len(tool_policy_violations),
        "conversation_order_violation_instances": len(conversation_order_violations),
        "runtime_budget": {
            "max_input_tokens": int(getattr(args, "max_input_tokens", 0) or 0) or None,
            "max_tool_calls": int(getattr(args, "max_tool_calls", 0) or 0) or None,
            "max_agent_turns": int(getattr(args, "max_agent_turns", 0) or 0) or None,
            "max_tool_wall_seconds": int(getattr(args, "max_tool_wall_seconds", 0) or 0) or None,
        },
        "runtime_budget_exceeded_instances": len(runtime_budget_exceeded),
        "disqualified_instances": sum(1 for row in results if row.get("openclaw_disqualified_reason")),
        "nemoclaw_session_audit_required_instances": len(session_audit_required),
        "nemoclaw_session_audit_passed_instances": len(session_audit_passed),
        "nemoclaw_session_audit_failed_instances": len(session_audit_failed),
        "weave_agents_required_instances": len(weave_agents_required),
        "weave_agents_passed_instances": len(weave_agents_passed),
        "weave_agents_failed_instances": len(weave_agents_failed),
        "dry_run": args.dry_run,
        "by_subject": by_subject,
    }
    (output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-jsonl", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/agentic_math_openclaw"))
    parser.add_argument("--prefix", default="openclaw")
    parser.add_argument("--agent", default="main")
    parser.add_argument("--profile")
    parser.add_argument("--model")
    parser.add_argument("--thinking", default="high")
    parser.add_argument("--nemoclaw-bin", default="nemoclaw")
    parser.add_argument(
        "--nemoclaw-sandbox",
        help="Run OpenClaw inside this NeMoClaw sandbox via `nemoclaw sandbox exec`.",
    )
    parser.add_argument("--nemoclaw-workdir", default="/sandbox")
    parser.add_argument("--nemoclaw-openclaw-config-path", default=NEMOCLAW_OPENCLAW_CONFIG_PATH)
    parser.add_argument("--openclaw-config-template", type=Path)
    parser.add_argument("--openclaw-tool-profile", default="coding")
    parser.add_argument("--deny-tool", action="append")
    parser.add_argument("--deny-argument-pattern", action="append")
    parser.add_argument("--use-task-agent", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--task-agent-prefix", default="nejumi-math")
    parser.add_argument(
        "--restart-gateway-after-task-agent-registration",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "Restart the NeMoClaw Gateway after bulk task-agent registration so "
            "parallel dynamic agents are visible to the running Gateway."
        ),
    )
    parser.add_argument(
        "--session-prefix",
        help=(
            "OpenClaw session-key prefix. When WANDB_RUN_ID is set, the run id is "
            "prepended unless this value already contains it or uses {wandb_run_id}."
        ),
    )
    parser.add_argument("--openclaw-timeout", type=int, default=1200)
    parser.add_argument(
        "--num-workers",
        type=int,
        default=1,
        help="Number of Agentic Math tasks to run concurrently.",
    )
    parser.add_argument(
        "--task-start-min-interval-seconds",
        type=float,
        default=0.0,
        help="Minimum interval between starting Agentic Math OpenClaw tasks.",
    )
    parser.add_argument(
        "--max-input-tokens",
        type=int,
        default=DEFAULT_MAX_INPUT_TOKENS,
        help=(
            "Agentic Math per-call/context input-token cap. If "
            "--max-cumulative-input-tokens is omitted, this value is also used "
            "as the cumulative provider input-token cap."
        ),
    )
    parser.add_argument(
        "--max-cumulative-input-tokens",
        type=int,
        default=None,
        help="Agentic Math per-task cumulative provider input-token cap. 0 disables it.",
    )
    parser.add_argument(
        "--max-cumulative-output-tokens",
        type=int,
        default=None,
        help="Agentic Math per-task cumulative provider output-token cap. 0 disables it.",
    )
    parser.add_argument(
        "--require-actual-token-usage",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Fail tasks when provider token usage is not available from OpenClaw.",
    )
    parser.add_argument(
        "--max-tool-calls",
        type=int,
        default=DEFAULT_MAX_TOOL_CALLS,
        help="Agentic Math per-task tool-call budget. 0 disables the budget.",
    )
    parser.add_argument(
        "--max-agent-turns",
        type=int,
        default=DEFAULT_MAX_AGENT_TURNS,
        help="Agentic Math per-task assistant-turn budget. 0 disables the budget.",
    )
    parser.add_argument(
        "--max-tool-wall-seconds",
        type=int,
        default=DEFAULT_MAX_TOOL_WALL_SECONDS,
        help=(
            "Hard wall-clock cap for each OpenClaw exec tool call. The generated "
            "OpenClaw config sets tools.exec.timeoutSec to this value, and the "
            "Nejumi OpenClaw runtime patch clamps model-supplied exec timeouts to it. "
            "0 disables this harness-level setting."
        ),
    )
    parser.add_argument(
        "--openclaw-max-attempts",
        type=int,
        default=3,
        help="Maximum attempts per task for transient OpenClaw/provider failures.",
    )
    parser.add_argument(
        "--openclaw-retry-base-seconds",
        type=float,
        default=15.0,
        help="Linear backoff base seconds between transient OpenClaw retries.",
    )
    parser.add_argument("--allow-failed-preflight", action="store_true")
    parser.add_argument("--no-local", action="store_true")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--redo", action="store_true", help="Re-run tasks even if result.json exists.")
    parser.add_argument("--fail-fast", action="store_true", help="Abort on the first OpenClaw task error.")
    parser.add_argument(
        "--weave-sidecar",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Diagnostic fallback only; native weave-openclaw plugin is the production trace path.",
    )
    parser.add_argument(
        "--weave-sidecar-strict",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Fail the task if --weave-sidecar is enabled and the diagnostic sidecar trace cannot be logged.",
    )
    parser.add_argument(
        "--verify-weave-agents",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Require native weave-openclaw Agents traces to be visible after each task.",
    )
    parser.add_argument("--weave-agents-entity", default=env_default_entity())
    parser.add_argument("--weave-agents-project", default=env_default_project())
    parser.add_argument("--weave-agents-agent-name", default=DEFAULT_WEAVE_AGENTS_AGENT_NAME)
    parser.add_argument("--weave-agents-env-file", type=Path, default=DEFAULT_WEAVE_AGENTS_ENV_FILE)
    parser.add_argument("--weave-agents-limit", type=int, default=50)
    parser.add_argument("--weave-agents-verification-timeout", type=float, default=120.0)
    parser.add_argument("--weave-agents-poll-seconds", type=float, default=5.0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if getattr(args, "weave_sidecar", False) or getattr(args, "weave_sidecar_strict", False):
        raise SystemExit(
            "Weave sidecar logging is disabled for Agentic Math. Use native "
            "weave-openclaw Agents traces only; manual sidecar traces are not valid evidence."
        )
    rows = read_jsonl(args.dataset_jsonl)
    if args.limit is not None:
        rows = rows[: args.limit]
    args.output_dir.mkdir(parents=True, exist_ok=True)
    ensure_nemoclaw_openclaw_permissions(args)

    results_by_index: dict[int, dict[str, Any]] = {}
    partial_results_path = args.output_dir / "results.partial.jsonl"

    def ordered_results() -> list[dict[str, Any]]:
        return [results_by_index[index] for index in sorted(results_by_index)]

    def run_one(index: int, row: dict[str, Any]) -> tuple[int, dict[str, Any]]:
        task_dir = args.output_dir / str(row["task_id"])
        task_start_limiter.wait()
        print(f"[{index}/{len(rows)}] Agentic Math task: {row['task_id']}", flush=True)
        return index, run_openclaw_for_task(row, task_dir, args)

    def pre_register_task_agents() -> None:
        if not uses_nemoclaw_gateway_task_agent(args) or bool(getattr(args, "dry_run", False)):
            return
        print(
            "Pre-registering NeMoClaw Gateway task agents before parallel Agentic Math execution.",
            flush=True,
        )
        for index, row in enumerate(rows, start=1):
            task_dir = args.output_dir / str(row["task_id"])
            workspace_dir = task_dir / "workspace"
            workspace_dir.mkdir(parents=True, exist_ok=True)
            agent_id, _ = write_task_openclaw_config(row, workspace_dir, task_dir, args)
            write_task_python_sitecustomize(row, workspace_dir, args, agent_id)
            print(
                f"[{index}/{len(rows)}] Pre-registered Agentic Math task agent: {row['task_id']}",
                flush=True,
            )
        restart_nemoclaw_gateway_after_task_agent_registration(args, label="Agentic Math")

    num_workers = max(1, int(getattr(args, "num_workers", 1) or 1))
    task_start_limiter = TaskStartLimiter(
        float(getattr(args, "task_start_min_interval_seconds", 0.0) or 0.0)
    )
    print(f"Agentic Math task workers: {num_workers}", flush=True)
    if task_start_limiter.min_interval_seconds:
        print(
            "Agentic Math task start min interval: "
            f"{task_start_limiter.min_interval_seconds:.1f}s",
            flush=True,
        )
    if num_workers == 1:
        for index, row in enumerate(rows, start=1):
            task_dir = args.output_dir / str(row["task_id"])
            try:
                completed_index, record = run_one(index, row)
                results_by_index[completed_index] = record
            except Exception as exc:
                error_path = write_task_error(task_dir, row, exc)
                write_jsonl(partial_results_path, ordered_results())
                print(
                    f"Agentic Math task failed: {row['task_id']} (details: {error_path})",
                    file=sys.stderr,
                    flush=True,
                )
                raise
            write_jsonl(partial_results_path, ordered_results())
    else:
        pre_register_task_agents()
        executor = ThreadPoolExecutor(max_workers=num_workers)
        try:
            future_to_task = {
                executor.submit(run_one, index, row): (index, row)
                for index, row in enumerate(rows, start=1)
            }
            for future in as_completed(future_to_task):
                index, row = future_to_task[future]
                task_dir = args.output_dir / str(row["task_id"])
                try:
                    completed_index, record = future.result()
                    results_by_index[completed_index] = record
                except Exception as exc:
                    error_path = write_task_error(task_dir, row, exc)
                    write_jsonl(partial_results_path, ordered_results())
                    print(
                        f"Agentic Math task failed: {row['task_id']} (details: {error_path})",
                        file=sys.stderr,
                        flush=True,
                    )
                    executor.shutdown(wait=False, cancel_futures=True)
                    raise
                write_jsonl(partial_results_path, ordered_results())
        finally:
            executor.shutdown(wait=True, cancel_futures=False)

    results = ordered_results()
    write_jsonl(args.output_dir / "results.jsonl", results)
    summary = write_summary(args.output_dir, results, args)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
