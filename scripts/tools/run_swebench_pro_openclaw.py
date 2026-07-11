#!/usr/bin/env python3
"""
Generate SWE-bench Pro patches with OpenClaw on real repository checkouts.

The runner creates or resets a checkout for each SWE-bench Pro instance at
`base_commit`, runs the Nejumi OpenClaw protocol in that checkout, then captures
`git diff --binary` plus untracked intent-to-add files as the model patch.
The output JSON is compatible with Scale's `swe_bench_pro_eval.py`.
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
import tempfile
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

TOOLS_DIR = Path(__file__).resolve().parent
if str(TOOLS_DIR) not in sys.path:
    sys.path.insert(0, str(TOOLS_DIR))

from weave_agents_native_trace import (
    DEFAULT_AGENT_NAME as DEFAULT_WEAVE_AGENTS_AGENT_NAME,
    DEFAULT_ENV_FILE as DEFAULT_WEAVE_AGENTS_ENV_FILE,
    empty_weave_agents_evidence,
    env_default_entity,
    env_default_project,
    verify_native_weave_agents_trace,
)


REPO_ROOT = Path(__file__).resolve().parents[2]
PROTOCOL_RUNNER = REPO_ROOT / "scripts" / "tools" / "run_openclaw_agent_protocol.py"
RUNNER_VERSION = "swebench-pro-openclaw-2026-07-10-timeup-parallel-v2"
PATCH_CAPTURE_VERSION = "git-diff-with-untracked-excluding-selected-tests-v2"
DEFAULT_MAX_INPUT_TOKENS = 1_000_000
DEFAULT_MAX_TOOL_CALLS = 40
DEFAULT_MAX_AGENT_TURNS = 40
DEFAULT_MAX_TOOL_WALL_SECONDS = 300
OPENCLAW_BUDGET_GUARD_PLUGIN_ID = "nejumi-budget-guard"
OPENCLAW_BUDGET_GUARD_PLUGIN_VERSION = "0.1.0"
OPENCLAW_BUDGET_GUARD_BLOCK_PREFIX = "NEJUMI_BUDGET_GUARD_BLOCKED"
SCOREABLE_OPENCLAW_DISQUALIFIED_REASONS = {
    "runtime_budget_exceeded",
    "tool_policy_violation",
    "conversation_order_violation",
    "time_up",
}
OPENCLAW_RUNTIME_DIR = ".nejumi_openclaw"
RUNTIME_EXCLUDED_PATHS = [OPENCLAW_RUNTIME_DIR]
NEMOCLAW_OPENCLAW_CONFIG_PATH = "/sandbox/.openclaw/openclaw.json"
DEFAULT_DENIED_TOOLS = [
    "code_execution",
    "web_search",
    "web_fetch",
    "browser",
    "browser_*",
]
DEFAULT_DENIED_ARGUMENT_PATTERNS = [
    r"https?://",
    r"\b(?:ftp|sftp|ssh)://",
    r"\bgit\+",
    r"\bgit\s+(?:clone|fetch|pull|ls-remote)\b",
    r"\b(curl|wget)\b",
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


def run_command(
    command: list[str],
    cwd: Path | None = None,
    timeout: int | None = None,
    check: bool = True,
) -> subprocess.CompletedProcess[str]:
    try:
        result = subprocess.run(
            command,
            cwd=str(cwd) if cwd else None,
            text=True,
            capture_output=True,
            timeout=timeout,
            check=False,
        )
    except subprocess.TimeoutExpired as exc:
        stdout = exc.stdout.decode("utf-8", errors="replace") if isinstance(exc.stdout, bytes) else (exc.stdout or "")
        stderr = exc.stderr.decode("utf-8", errors="replace") if isinstance(exc.stderr, bytes) else (exc.stderr or "")
        result = subprocess.CompletedProcess(
            command,
            124,
            stdout=stdout,
            stderr=stderr + f"\nCommand timed out after {timeout} seconds",
        )
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


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def safe_id(instance_id: str) -> str:
    return "".join(ch if ch.isalnum() or ch in "._-" else "_" for ch in instance_id)


def safe_agent_id(instance_id: str, prefix: str) -> str:
    digest = hashlib.sha1(instance_id.encode("utf-8")).hexdigest()[:12]
    normalized_prefix = "".join(
        ch.lower() if ch.isalnum() else "-" for ch in prefix.strip()
    ).strip("-")
    return f"{normalized_prefix or 'nejumi-swe'}-{digest}"


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def command_sha256(command: list[str]) -> str:
    payload = json.dumps(command, ensure_ascii=False, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def resolve_session_prefix(args: argparse.Namespace, default_prefix: str = "swebench-pro") -> str:
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
        "instance_id": str(row["instance_id"]),
        "prompt_hash": sha256_text(prompt_text),
        "model": args.model or "",
        "thinking": args.thinking,
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
        "nemoclaw_sandbox": str(getattr(args, "nemoclaw_sandbox", "") or ""),
        "nemoclaw_checkout_sandbox_root": str(
            getattr(args, "nemoclaw_checkout_sandbox_root", "") or ""
        ),
        "nemoclaw_checkout_transfer_mode": str(
            getattr(args, "nemoclaw_checkout_transfer_mode", "visible") or "visible"
        ),
        "session_prefix": resolve_session_prefix(args),
        "verify_weave_agents": bool(getattr(args, "verify_weave_agents", False)),
        "weave_agents_entity": str(getattr(args, "weave_agents_entity", "") or ""),
        "weave_agents_project": str(getattr(args, "weave_agents_project", "") or ""),
        "weave_agents_agent_name": str(getattr(args, "weave_agents_agent_name", "") or ""),
    }


def cache_key_matches(record: dict[str, Any], cache_key: dict[str, Any]) -> bool:
    return record.get("cache_key") == cache_key


def cache_requires_nemoclaw_session_audit(cache_key: dict[str, Any]) -> bool:
    return bool(cache_key.get("nemoclaw_sandbox"))


def sidecar_identity_matches_cache(sidecar: dict[str, Any], cache_key: dict[str, Any]) -> bool:
    metadata = sidecar.get("metadata") if isinstance(sidecar.get("metadata"), dict) else {}
    policy = sidecar.get("tool_policy") if isinstance(sidecar.get("tool_policy"), dict) else {}
    expected_config = cache_key.get("openclaw_config_source")
    observed_config = metadata.get("openclaw_config_source")
    return (
        metadata.get("task_id") == cache_key["instance_id"]
        and metadata.get("prompt_hash") == cache_key["prompt_hash"]
        and (not cache_key.get("model") or metadata.get("model_id") == cache_key.get("model"))
        and (not expected_config or observed_config == expected_config)
        and sorted(policy.get("deny_tools") or []) == sorted(cache_key.get("deny_tools") or [])
        and sorted(policy.get("deny_argument_patterns") or [])
        == sorted(cache_key.get("deny_argument_patterns") or [])
    )


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


def patch_record_scoreable_disqualification(record: dict[str, Any]) -> bool:
    return str(record.get("openclaw_disqualified_reason") or "") in SCOREABLE_OPENCLAW_DISQUALIFIED_REASONS


def patch_record_nemoclaw_session_audit_matches_cache(record: dict[str, Any], cache_key: dict[str, Any]) -> bool:
    if patch_record_scoreable_disqualification(record):
        return True
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


def patch_record_conversation_order_allows_reuse(record: dict[str, Any]) -> bool:
    if patch_record_scoreable_disqualification(record):
        return True
    order = record.get("conversation_order")
    if record.get("conversation_order_ok") is False:
        return False
    if isinstance(order, dict) and order.get("ok") is False:
        return False
    return True


def patch_record_tool_policy_allows_reuse(record: dict[str, Any]) -> bool:
    if patch_record_scoreable_disqualification(record):
        return True
    if record.get("tool_policy_ok") is False:
        return False
    if record.get("tool_policy_violations"):
        return False
    return True


def patch_record_weave_sidecar_allows_reuse(record: dict[str, Any]) -> bool:
    sidecar = record.get("weave_sidecar")
    if record.get("weave_sidecar_ok") is False:
        return False
    if isinstance(sidecar, dict) and sidecar.get("ok") is False:
        return False
    return True


def patch_record_weave_agents_allows_reuse(record: dict[str, Any], cache_key: dict[str, Any]) -> bool:
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


def patch_record_invocation_matches_cache(record: dict[str, Any], cache_key: dict[str, Any]) -> bool:
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


def patch_mentions_paths(patch: str, paths: list[str]) -> bool:
    for path in paths:
        if f" a/{path} " in patch or f" b/{path}" in patch:
            return True
        if f"a/{path}\n" in patch or f"b/{path}\n" in patch:
            return True
    return False


def load_cached_patch_record(
    task_dir: Path,
    cache_key: dict[str, Any],
    prefix: str,
    excluded_paths: list[str] | None = None,
) -> dict[str, Any] | None:
    record_path = task_dir / "patch_record.json"
    if not record_path.exists():
        return None
    try:
        record = json.loads(record_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return None
    if not isinstance(record, dict) or not cache_key_matches(record, cache_key):
        return None
    if record.get("instance_id") != cache_key["instance_id"] or "patch" not in record:
        return None
    if not patch_record_nemoclaw_session_audit_matches_cache(record, cache_key):
        return None
    if not patch_record_conversation_order_allows_reuse(record):
        return None
    if not patch_record_tool_policy_allows_reuse(record):
        return None
    if not patch_record_weave_sidecar_allows_reuse(record):
        return None
    if not patch_record_weave_agents_allows_reuse(record, cache_key):
        return None
    if not patch_record_invocation_matches_cache(record, cache_key):
        return None
    if not record.get("patch") and record.get("patch_capture_version") != PATCH_CAPTURE_VERSION:
        return None
    if (
        record.get("patch_capture_version") != PATCH_CAPTURE_VERSION
        and patch_mentions_paths(str(record.get("patch") or ""), excluded_paths or [])
    ):
        return None
    cached = dict(record)
    if not cached.get("openclaw_usage"):
        result_path = cached.get("openclaw_result_path")
        if isinstance(result_path, str) and result_path:
            try:
                sidecar = json.loads(Path(result_path).read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                sidecar = None
            if isinstance(sidecar, dict):
                usage = sidecar_usage(sidecar)
                if usage:
                    cached["openclaw_usage"] = usage
                    record["openclaw_usage"] = usage
                    try:
                        record_path.write_text(
                            json.dumps(record, ensure_ascii=False, indent=2) + "\n",
                            encoding="utf-8",
                        )
                    except OSError:
                        pass
    cached["prefix"] = prefix
    return cached


def task_sidecar_path(protocol_output_dir: Path, instance_id: str, benchmark_id: str = "agentic_swe") -> Path:
    return protocol_output_dir / benchmark_id / instance_id / "openclaw_result.json"


def protocol_benchmark_id(row: dict[str, Any]) -> str:
    """Map source benchmark names onto the protocol runner's supported IDs."""
    benchmark_id = str(row.get("benchmark_id") or "agentic_swe")
    if benchmark_id in {"swebench_lite", "swebench_pro", "swebench"}:
        return "agentic_swe"
    return benchmark_id


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
    workdir: str = "/sandbox",
) -> subprocess.CompletedProcess[str]:
    full_command = [
        getattr(args, "nemoclaw_bin", "nemoclaw"),
        "sandbox",
        "exec",
        args.nemoclaw_sandbox,
        "--workdir",
        workdir,
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
    config_dir = config_path.rsplit("/", 1)[0] or "."
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
        getattr(args, "nemoclaw_bin", "nemoclaw"),
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


def run_nemoclaw_binary_command(
    args: argparse.Namespace,
    command: list[str],
    *,
    input_bytes: bytes | None = None,
    timeout: int = 60,
    check: bool = True,
    workdir: str = "/sandbox",
) -> subprocess.CompletedProcess[bytes]:
    full_command = [
        args.nemoclaw_bin,
        "sandbox",
        "exec",
        args.nemoclaw_sandbox,
        "--workdir",
        workdir,
        "--no-tty",
        "--timeout",
        str(timeout),
        "--",
        *command,
    ]
    result = subprocess.run(
        full_command,
        input=input_bytes,
        capture_output=True,
        check=False,
        timeout=max(timeout + 10, 10),
    )
    if check and result.returncode != 0:
        raise RuntimeError(
            "NeMoClaw command failed\n"
            f"cmd: {' '.join(shlex.quote(part) for part in full_command)}\n"
            f"returncode: {result.returncode}\n"
            f"stdout:\n{result.stdout.decode('utf-8', errors='replace')}\n"
            f"stderr:\n{result.stderr.decode('utf-8', errors='replace')}"
        )
    return result


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


def openclaw_config_cache_source(args: argparse.Namespace) -> str:
    template = getattr(args, "openclaw_config_template", None)
    if template:
        return str(Path(template).expanduser())
    if getattr(args, "nemoclaw_sandbox", None):
        return str(getattr(args, "nemoclaw_openclaw_config_path", NEMOCLAW_OPENCLAW_CONFIG_PATH))
    return str(default_openclaw_config_template())


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
            required_texts=[f"instance_id: {row['instance_id']}"],
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
            f"{row['instance_id']}; recording per-instance evidence failure and continuing: "
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


def is_runtime_budget_exceeded(sidecar: dict[str, Any] | None) -> bool:
    if not isinstance(sidecar, dict):
        return False
    runtime_budget = sidecar.get("runtime_budget")
    return (
        isinstance(runtime_budget, dict)
        and runtime_budget.get("ok") is False
        and bool(runtime_budget.get("violations"))
    )


def sidecar_conversation_order_failed(sidecar: dict[str, Any] | None) -> bool:
    if not isinstance(sidecar, dict):
        return False
    conversation_order = sidecar.get("conversation_order")
    return isinstance(conversation_order, dict) and conversation_order.get("ok") is False


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


def is_outer_openclaw_timeout(
    result: subprocess.CompletedProcess[str],
    sidecar: dict[str, Any] | None,
) -> bool:
    if result.returncode != 124:
        return False
    return bool(OUTER_OPENCLAW_TIMEOUT_RE.search(openclaw_failure_text(result, sidecar)))


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
    """Return a setup/provider failure reason that must not become a benchmark patch."""
    if result.returncode == 0:
        return None
    text = openclaw_failure_text(result, sidecar)
    for reason, pattern in NON_SCOREABLE_OPENCLAW_FAILURE_PATTERNS:
        if re.search(pattern, text, flags=re.IGNORECASE):
            return reason
    return None


def append_transient_failure(task_dir: Path, payload: dict[str, Any]) -> None:
    with (task_dir / "openclaw_transient_failures.jsonl").open("a", encoding="utf-8") as f:
        f.write(json.dumps(payload, ensure_ascii=False) + "\n")


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


def sidecar_usage(sidecar: dict[str, Any]) -> dict[str, Any]:
    stdout_json = sidecar.get("stdout_json")
    if isinstance(stdout_json, dict):
        for meta_parent in (stdout_json.get("result"), stdout_json):
            if not isinstance(meta_parent, dict):
                continue
            usage = meta_parent.get("meta", {}).get("agentMeta", {}).get("usage", {})
            if isinstance(usage, dict) and usage:
                return usage
    runtime_budget = sidecar.get("runtime_budget")
    observed = runtime_budget.get("observed") if isinstance(runtime_budget, dict) else None
    actual_usage = observed.get("actual_usage") if isinstance(observed, dict) else None
    if isinstance(actual_usage, dict) and actual_usage:
        return actual_usage
    return {}


def repo_url(repo: str) -> str:
    if repo.startswith("http://") or repo.startswith("https://") or repo.endswith(".git"):
        return repo
    return f"https://github.com/{repo}.git"


def prepare_checkout(row: dict[str, Any], checkout_root: Path, reset: bool) -> Path:
    instance_id = str(row["instance_id"])
    checkout_dir = checkout_root / safe_id(instance_id)
    checkout_root.mkdir(parents=True, exist_ok=True)

    new_checkout = False
    if not checkout_dir.exists():
        run_command(["git", "clone", repo_url(str(row["repo"])), str(checkout_dir)])
        new_checkout = True

    if reset or new_checkout:
        run_command(["git", "fetch", "--all", "--tags", "--prune"], cwd=checkout_dir)
        run_command(["git", "reset", "--hard"], cwd=checkout_dir)
        run_command(["git", "clean", "-fdx"], cwd=checkout_dir)
    else:
        return checkout_dir

    base_commit = str(row["base_commit"])
    checkout = run_command(["git", "checkout", "--force", base_commit], cwd=checkout_dir, check=False)
    if checkout.returncode != 0:
        run_command(["git", "fetch", "origin", base_commit], cwd=checkout_dir)
        run_command(["git", "checkout", "--force", base_commit], cwd=checkout_dir)
    run_command(["git", "reset", "--hard", base_commit], cwd=checkout_dir)
    run_command(["git", "clean", "-fdx"], cwd=checkout_dir)
    return checkout_dir


def list_text(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, list):
        return "\n".join(f"- {item}" for item in value)
    if isinstance(value, tuple):
        return "\n".join(f"- {item}" for item in value)
    return str(value)


def normalize_prompt_newlines(text: str) -> str:
    return text.replace("\r\n", "\n").replace("\r", "\n")


def build_prompt(row: dict[str, Any], max_tool_wall_seconds: int | None = DEFAULT_MAX_TOOL_WALL_SECONDS) -> str:
    benchmark_name = str(row.get("benchmark_name") or "SWE-bench Pro")
    issue_categories = list_text(row.get("issue_categories"))
    selected_tests = list_text(row.get("selected_test_files_to_run"))
    requirements = str(row.get("requirements") or "").strip()
    interface = str(row.get("interface") or "").strip()
    wall_limit = int(max_tool_wall_seconds or 0)
    wall_limit_text = f"{wall_limit} seconds" if wall_limit > 0 else "the configured runtime limit"

    parts = [
        f"# {benchmark_name} Task",
        "",
        "You are running inside a checkout of the target repository at the base commit.",
        "Inspect the repository, edit files as needed, and leave the working tree with the minimal fix.",
        "Do not create commits. The harness will collect `git diff --binary` after you finish.",
        "",
        "## Metadata",
        "",
        f"- instance_id: {row.get('instance_id')}",
        f"- repo: {row.get('repo')}",
        f"- base_commit: {row.get('base_commit')}",
        f"- language: {row.get('repo_language')}",
        f"- issue_specificity: {row.get('issue_specificity')}",
        f"- docker image tag: {row.get('dockerhub_tag')}",
        "",
        "## Issue",
        "",
        str(row.get("problem_statement") or "").strip(),
    ]
    if requirements:
        parts.extend(["", "## Requirements", "", requirements])
    if interface:
        parts.extend(["", "## Interface", "", interface])
    if issue_categories:
        parts.extend(["", "## Issue Categories", "", issue_categories])
    if selected_tests:
        parts.extend(["", "## Selected Test Files", "", selected_tests])
    before_cmd = str(row.get("before_repo_set_cmd") or "").strip()
    if before_cmd:
        parts.extend(["", "## Evaluation Setup Hint", "", before_cmd])
    parts.extend(
        [
            "",
            "## Agent Workflow",
            "",
            "First orient yourself with local repository tools such as `pwd`, `ls`, `find`, or `rg`.",
            "Use repository search before reading specific files unless the exact path is already proven.",
            "If a file read returns missing-path errors, stop guessing paths and search the checkout.",
            "Run the relevant local tests when practical after making changes.",
            f"Each shell execution has a wall-clock limit of {wall_limit_text}.",
            "Keep commands targeted: prefer `rg`, focused file reads, selected tests, and small verification commands.",
            "Do not run broad repository-wide builds, servers, notebooks, or background jobs unless required by the issue.",
            "Package installs with `pip install ...` are allowed when needed to run local tests.",
            "Do not install from explicit git URLs, HTTP(S) URLs, or other remote source URLs.",
            "Do not use package managers as web search or data-fetch tools.",
            "If a command times out, narrow the command or continue with static reasoning from the repository.",
        ]
    )
    parts.extend(
        [
            "",
            "## Output Contract",
            "",
            "Finish with the repository files edited. The benchmark runner, not you, will collect the patch.",
            "Keep the change minimal and avoid unrelated formatting or dependency churn.",
            "Use local shell execution in the target checkout; do not use remote code interpreter tools.",
            "Do not use web search, browser, HTTP, or any external internet lookup.",
            "If an install command is needed, keep it targeted to the repository or its normal test dependencies.",
            "Do not write `FINAL ANSWER`, `ANSWER:`, or any equivalent final-answer marker before all tool use is complete.",
            "Do not include a final-answer marker in the same assistant turn as a tool call.",
        ]
    )
    return normalize_prompt_newlines("\n".join(parts) + "\n")


def selected_test_paths(row: dict[str, Any]) -> list[str]:
    raw = row.get("selected_test_files_to_run") or []
    if isinstance(raw, str):
        raw = [raw]
    paths: list[str] = []
    for item in raw:
        path = str(item).split("::", 1)[0].strip().lstrip("./")
        if path:
            paths.append(path)
    return sorted(set(paths))


def _is_excluded_path(path: str, excluded_paths: list[str]) -> bool:
    normalized = path.strip().lstrip("./")
    for excluded in excluded_paths:
        excluded = excluded.rstrip("/")
        if normalized == excluded or normalized.startswith(f"{excluded}/"):
            return True
    return False


def capture_patch(checkout_dir: Path, excluded_paths: list[str] | None = None) -> str:
    excluded_paths = sorted(set((excluded_paths or []) + RUNTIME_EXCLUDED_PATHS))
    if excluded_paths:
        for start in range(0, len(excluded_paths), 200):
            run_command(
                ["git", "reset", "-q", "--", *excluded_paths[start : start + 200]],
                cwd=checkout_dir,
                check=False,
            )
    untracked = run_command(
        ["git", "ls-files", "--others", "--exclude-standard", "-z"],
        cwd=checkout_dir,
    )
    paths = [
        path
        for path in untracked.stdout.split("\0")
        if path and not _is_excluded_path(path, excluded_paths)
    ]
    for start in range(0, len(paths), 200):
        run_command(["git", "add", "-N", "--", *paths[start : start + 200]], cwd=checkout_dir)
    diff_command = ["git", "diff", "--binary"]
    if excluded_paths:
        diff_command.extend(["--", ".", *[f":(exclude){path}" for path in excluded_paths]])
    result = run_command(diff_command, cwd=checkout_dir)
    return result.stdout


def _filter_untracked_paths(raw: str, excluded_paths: list[str]) -> list[str]:
    return [
        path
        for path in raw.split("\0")
        if path and not _is_excluded_path(path, excluded_paths)
    ]


def capture_patch_nemoclaw(
    checkout_dir: Path,
    args: argparse.Namespace,
    excluded_paths: list[str] | None = None,
) -> str:
    excluded_paths = sorted(set((excluded_paths or []) + RUNTIME_EXCLUDED_PATHS))
    sandbox_dir = str(sandbox_checkout_dir(checkout_dir, args))
    if excluded_paths:
        for start in range(0, len(excluded_paths), 200):
            run_nemoclaw_text_command(
                args,
                [
                    "bash",
                    "-lc",
                    'git reset -q -- "$@"',
                    "git-reset",
                    *excluded_paths[start : start + 200],
                ],
                workdir=sandbox_dir,
                check=False,
            )
    untracked = run_nemoclaw_text_command(
        args,
        ["git", "ls-files", "--others", "--exclude-standard", "-z"],
        workdir=sandbox_dir,
    )
    paths = _filter_untracked_paths(untracked.stdout, excluded_paths)
    for start in range(0, len(paths), 200):
        run_nemoclaw_text_command(
            args,
            [
                "bash",
                "-lc",
                'git add -N -- "$@"',
                "git-add",
                *paths[start : start + 200],
            ],
            workdir=sandbox_dir,
        )
    diff_paths = ["."]
    if excluded_paths:
        diff_paths.extend([f":(exclude){path}" for path in excluded_paths])
    result = run_nemoclaw_text_command(
        args,
        ["bash", "-lc", 'git diff --binary -- "$@"', "git-diff", *diff_paths],
        workdir=sandbox_dir,
    )
    return result.stdout


def should_force_empty_patch(openclaw_metadata: dict[str, Any]) -> bool:
    return bool(openclaw_metadata.get("openclaw_disqualified_reason"))


def default_openclaw_config_template() -> Path:
    return Path(os.environ.get("OPENCLAW_CONFIG_PATH", "~/.openclaw/openclaw.json")).expanduser()


def sandbox_checkout_dir(checkout_dir: Path, args: argparse.Namespace) -> Path:
    sandbox_root = getattr(args, "nemoclaw_checkout_sandbox_root", None)
    if sandbox_root:
        return Path(str(sandbox_root)) / checkout_dir.name
    transfer_mode = str(getattr(args, "nemoclaw_checkout_transfer_mode", "visible") or "visible")
    if getattr(args, "nemoclaw_sandbox", None) and transfer_mode == "copy":
        return Path("/sandbox/checkouts") / checkout_dir.name
    return checkout_dir.resolve()


def openclaw_nemoclaw_workdir(checkout_dir: Path, args: argparse.Namespace) -> str | None:
    if not getattr(args, "nemoclaw_sandbox", None):
        return None
    return str(sandbox_checkout_dir(checkout_dir, args))


def nemoclaw_checkout_visible(checkout_dir: Path, args: argparse.Namespace) -> bool:
    if not getattr(args, "nemoclaw_sandbox", None):
        return True
    sandbox_dir = str(sandbox_checkout_dir(checkout_dir, args))
    if "\n" in sandbox_dir or "\r" in sandbox_dir:
        raise RuntimeError(f"Invalid sandbox checkout path: {sandbox_dir!r}")
    transfer_mode = str(getattr(args, "nemoclaw_checkout_transfer_mode", "visible") or "visible")
    check_script = 'test -d "$1"'
    if transfer_mode == "copy":
        check_script = 'test -d "$1" && git -C "$1" rev-parse --is-inside-work-tree >/dev/null 2>&1'
    result = run_nemoclaw_text_command(
        args,
        ["bash", "-lc", check_script, "check-checkout", sandbox_dir],
        timeout=30,
        check=False,
    )
    return result.returncode == 0


def assert_nemoclaw_checkout_visible(checkout_dir: Path, args: argparse.Namespace) -> None:
    if not getattr(args, "nemoclaw_sandbox", None):
        return
    if not nemoclaw_checkout_visible(checkout_dir, args):
        raise RuntimeError(
            "SWE-Bench Pro checkout is not visible inside the NeMoClaw sandbox: "
            f"{sandbox_checkout_dir(checkout_dir, args)}. Mount or prepare the checkout under the sandbox-visible root "
            "before running the agent."
        )


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def create_checkout_archive(checkout_dir: Path, archive_path: Path) -> None:
    run_command(
        [
            "tar",
            "-C",
            str(checkout_dir),
            "--exclude",
            "./.git",
            "--exclude",
            f"./{OPENCLAW_RUNTIME_DIR}",
            "-czf",
            str(archive_path),
            ".",
        ],
        cwd=REPO_ROOT,
    )


def upload_file_to_nemoclaw(
    local_path: Path,
    sandbox_path: str,
    args: argparse.Namespace,
) -> dict[str, Any]:
    host_sha = sha256_file(local_path)
    run_nemoclaw_text_command(
        args,
        ["bash", "-lc", 'mkdir -p "$(dirname "$1")"', "init-upload", sandbox_path],
        timeout=30,
    )
    upload_timeout = max(60, int(getattr(args, "nemoclaw_checkout_transfer_timeout", 300) or 300))
    upload_command = [
        getattr(args, "nemoclaw_bin", "nemoclaw"),
        "sandbox",
        "upload",
        args.nemoclaw_sandbox,
        str(local_path),
        sandbox_path,
    ]
    with _NEMOCLAW_EXEC_LOCK:
        run_command(upload_command, cwd=REPO_ROOT, timeout=upload_timeout)
    result = run_nemoclaw_text_command(args, ["sha256sum", sandbox_path], timeout=60)
    sandbox_sha = result.stdout.strip().split()[0] if result.stdout.strip() else ""
    if sandbox_sha != host_sha:
        raise RuntimeError(
            "Transferred NeMoClaw checkout archive sha256 mismatch: "
            f"host={host_sha} sandbox={sandbox_sha}"
        )
    return {
        "archive_bytes": local_path.stat().st_size,
        "archive_sha256": host_sha,
        "transport": "nemoclaw_sandbox_upload",
        "chunk_bytes": None,
        "chunk_count": None,
        "sandbox_archive": sandbox_path,
    }


def sync_checkout_to_nemoclaw_copy(
    checkout_dir: Path,
    task_dir: Path,
    args: argparse.Namespace,
) -> dict[str, Any]:
    sandbox_dir = str(sandbox_checkout_dir(checkout_dir, args))
    transfer_dir = task_dir / "nemoclaw_checkout_transfer"
    transfer_dir.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=transfer_dir) as tmp_name:
        archive_path = Path(tmp_name) / "checkout.tgz"
        create_checkout_archive(checkout_dir, archive_path)
        sandbox_archive = f"/sandbox/tmp/nejumi_swe_checkout/{safe_id(checkout_dir.name)}.tgz"
        upload = upload_file_to_nemoclaw(archive_path, sandbox_archive, args)
    run_nemoclaw_text_command(
        args,
        [
            "bash",
            "-lc",
            (
                'rm -rf "$1" && mkdir -p "$1" && tar -xzf "$2" -C "$1" '
                '&& cd "$1" && git init -q '
                '&& git config user.email "nejumi-swe@example.local" '
                '&& git config user.name "Nejumi SWE Harness" '
                '&& git add -f -A '
                '&& git commit -q -m baseline'
            ),
            "extract-checkout",
            sandbox_dir,
            sandbox_archive,
        ],
        timeout=max(60, int(getattr(args, "nemoclaw_checkout_transfer_timeout", 300))),
    )
    return {
        "mode": "copy",
        "sandbox_checkout_dir": sandbox_dir,
        **upload,
    }


def ensure_nemoclaw_checkout_ready(
    checkout_dir: Path,
    task_dir: Path,
    args: argparse.Namespace,
) -> dict[str, Any] | None:
    if not getattr(args, "nemoclaw_sandbox", None):
        return None
    if nemoclaw_checkout_visible(checkout_dir, args):
        return {"mode": "visible", "sandbox_checkout_dir": str(sandbox_checkout_dir(checkout_dir, args))}
    mode = str(getattr(args, "nemoclaw_checkout_transfer_mode", "visible") or "visible")
    if mode != "copy":
        assert_nemoclaw_checkout_visible(checkout_dir, args)
    return sync_checkout_to_nemoclaw_copy(checkout_dir, task_dir, args)


def task_openclaw_config_paths(
    checkout_dir: Path,
    args: argparse.Namespace,
) -> tuple[Path, Path, Path, Path]:
    host_runtime_dir = checkout_dir / OPENCLAW_RUNTIME_DIR
    host_config_path = host_runtime_dir / "openclaw_config.json"
    host_agent_dir = host_runtime_dir / "agent_state"
    if getattr(args, "nemoclaw_sandbox", None):
        sandbox_runtime_dir = sandbox_checkout_dir(checkout_dir, args) / OPENCLAW_RUNTIME_DIR
        sandbox_config_path = sandbox_runtime_dir / "openclaw_config.json"
        sandbox_agent_dir = sandbox_runtime_dir / "agent_state"
    else:
        sandbox_config_path = host_config_path
        sandbox_agent_dir = host_agent_dir
    return host_config_path, host_agent_dir, sandbox_config_path, sandbox_agent_dir


def task_live_session_dir(
    checkout_dir: Path,
    task_dir: Path,
    args: argparse.Namespace,
) -> Path | None:
    if not getattr(args, "use_task_agent", True):
        return None
    if getattr(args, "nemoclaw_sandbox", None):
        if uses_nemoclaw_gateway_task_agent(args):
            return None
        transfer_mode = str(
            getattr(args, "nemoclaw_checkout_transfer_mode", "visible") or "visible"
        )
        if transfer_mode != "visible":
            return None
        _, host_agent_dir, _, _ = task_openclaw_config_paths(checkout_dir, args)
        return host_agent_dir / "sessions"
    return task_dir / "openclaw_agent_state" / "sessions"


def task_live_sandbox_session_dir(
    checkout_dir: Path,
    args: argparse.Namespace,
    agent_id: str | None = None,
) -> str | None:
    if not getattr(args, "use_task_agent", True):
        return None
    if not getattr(args, "nemoclaw_sandbox", None):
        return None
    if uses_nemoclaw_gateway_task_agent(args):
        if not agent_id:
            agent_id = safe_agent_id(checkout_dir.name, args.task_agent_prefix)
        return f"/sandbox/.openclaw/agents/{agent_id}/sessions"
    transfer_mode = str(getattr(args, "nemoclaw_checkout_transfer_mode", "visible") or "visible")
    if transfer_mode == "visible":
        return None
    _, _, _, sandbox_agent_dir = task_openclaw_config_paths(checkout_dir, args)
    return str(sandbox_agent_dir / "sessions")


def write_task_openclaw_config(
    row: dict[str, Any],
    checkout_dir: Path,
    task_dir: Path,
    args: argparse.Namespace,
) -> tuple[str, Path | None]:
    if not args.use_task_agent:
        return args.agent, None
    if args.no_local and not getattr(args, "nemoclaw_sandbox", None):
        raise RuntimeError("--use-task-agent requires local OpenClaw execution; remove --no-local")

    agent_id = safe_agent_id(str(row["instance_id"]), args.task_agent_prefix)
    if getattr(args, "nemoclaw_sandbox", None):
        host_config_path, host_agent_dir, sandbox_config_path, sandbox_agent_dir = task_openclaw_config_paths(
            checkout_dir,
            args,
        )
        workspace = str(sandbox_checkout_dir(checkout_dir, args))
    else:
        host_config_path = task_dir / "openclaw_config.json"
        host_agent_dir = task_dir / "openclaw_agent_state"
        sandbox_config_path = host_config_path
        sandbox_agent_dir = host_agent_dir
        workspace = str(checkout_dir.resolve())
    host_config_path.parent.mkdir(parents=True, exist_ok=True)

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
                and existing.get("agent_dir") == str(sandbox_agent_dir)
                and existing.get("config_path") == canonical_config_path
                and isinstance(existing.get("gateway_registered"), dict)
                and existing["gateway_registered"].get("ok") is True
            ):
                return agent_id, None

        registration = register_nemoclaw_gateway_task_agent(
            args,
            agent_id=agent_id,
            workspace=workspace,
            agent_dir=str(sandbox_agent_dir),
        )
        task_agent_path.write_text(
            json.dumps(
                {
                    "agent_id": agent_id,
                    "workspace": workspace,
                    "agent_dir": str(sandbox_agent_dir),
                    "host_config_path": str(host_config_path),
                    "host_agent_dir": str(host_agent_dir),
                    "sandbox_config_path": canonical_config_path,
                    "sandbox_agent_dir": str(sandbox_agent_dir),
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
        "agentDir": str(sandbox_agent_dir),
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
    host_config_path.write_text(config_text, encoding="utf-8")
    if getattr(args, "nemoclaw_sandbox", None) and str(
        getattr(args, "nemoclaw_checkout_transfer_mode", "visible") or "visible"
    ) == "copy":
        run_nemoclaw_text_command(
            args,
            [
                "python3",
                "-c",
                (
                    "import sys; from pathlib import Path; "
                    "path = Path(sys.argv[1]); path.parent.mkdir(parents=True, exist_ok=True); "
                    "path.write_text(sys.stdin.read(), encoding='utf-8')"
                ),
                str(sandbox_config_path),
            ],
            input_text=config_text,
            timeout=60,
        )
    (task_dir / "openclaw_task_agent.json").write_text(
        json.dumps(
            {
                "agent_id": agent_id,
                "workspace": agent_entry["workspace"],
                "agent_dir": agent_entry["agentDir"],
                "host_config_path": str(host_config_path),
                "host_agent_dir": str(host_agent_dir),
                "sandbox_config_path": str(sandbox_config_path),
                "sandbox_agent_dir": str(sandbox_agent_dir),
                "config_template": template_path,
                "config_path": str(sandbox_config_path),
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
    return agent_id, sandbox_config_path


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


def run_openclaw_for_task(
    row: dict[str, Any],
    checkout_dir: Path,
    task_dir: Path,
    args: argparse.Namespace,
) -> dict[str, Any]:
    task_dir.mkdir(parents=True, exist_ok=True)
    prompt_text = build_prompt(row, max_tool_wall_seconds=int(getattr(args, "max_tool_wall_seconds", 0) or 0))
    prompt_hash = sha256_text(prompt_text)
    cache_key = build_cache_key(row, prompt_text, args)
    prompt_file = task_dir / "prompt.md"
    prompt_file.write_text(prompt_text, encoding="utf-8")
    checkout_transfer = ensure_nemoclaw_checkout_ready(checkout_dir, task_dir, args)
    agent_id, openclaw_config_path = write_task_openclaw_config(row, checkout_dir, task_dir, args)

    invocation_dir = task_dir / "openclaw_invocations"
    invocation_dir.mkdir(parents=True, exist_ok=True)
    max_attempts = max(1, int(args.openclaw_max_attempts))
    metadata: dict[str, Any] | None = None
    sidecar_path: Path | None = None
    for attempt_number in range(1, max_attempts + 1):
        attempt_id = f"{int(time.time())}-{os.getpid()}-{attempt_number}"
        benchmark_id = protocol_benchmark_id(row)
        session_key = f"{resolve_session_prefix(args)}:{row['instance_id']}:{attempt_id}"
        attempt_output_dir = task_dir / "openclaw_attempts" / attempt_id
        sidecar_path = task_sidecar_path(attempt_output_dir, str(row["instance_id"]), benchmark_id=benchmark_id)
        invocation_path = invocation_dir / f"{attempt_id}.json"
        command = [
            sys.executable,
            str(PROTOCOL_RUNNER),
            "run",
            "--benchmark-id",
            benchmark_id,
            "--task-id",
            str(row["instance_id"]),
            "--prompt-file",
            str(prompt_file),
            "--cwd",
            str(checkout_dir),
            "--output-dir",
            str(attempt_output_dir),
            "--agent",
            agent_id,
            "--session-key",
            session_key,
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
        if openclaw_config_path:
            command.extend(["--openclaw-config-path", str(openclaw_config_path)])
        command.extend(["--openclaw-config-source", str(cache_key["openclaw_config_source"])])
        live_session_dir = task_live_session_dir(checkout_dir, task_dir, args)
        if live_session_dir is not None:
            command.extend(["--live-session-dir", str(live_session_dir)])
        live_sandbox_session_dir = task_live_sandbox_session_dir(checkout_dir, args, agent_id)
        if live_sandbox_session_dir is not None:
            command.extend(["--live-sandbox-session-dir", live_sandbox_session_dir])
        if getattr(args, "nemoclaw_sandbox", None):
            command.extend(["--nemoclaw-bin", str(getattr(args, "nemoclaw_bin", "nemoclaw"))])
            command.extend(["--nemoclaw-sandbox", str(args.nemoclaw_sandbox)])
            nemoclaw_workdir = openclaw_nemoclaw_workdir(checkout_dir, args)
            command.extend(["--nemoclaw-workdir", str(nemoclaw_workdir)])
        if args.profile:
            command.extend(["--profile", args.profile])
        if args.model:
            command.extend(["--model", args.model])
        if args.no_local:
            command.append("--no-local")
        if args.allow_failed_preflight:
            command.append("--allow-failed-preflight")
        command.append("--no-weave-sidecar")
        for denied_tool in effective_deny_tools(args):
            command.extend(["--deny-tool", denied_tool])
        for pattern in effective_deny_argument_patterns(args):
            command.extend(["--deny-argument-pattern", pattern])
        if args.dry_run:
            command.append("--dry-run")

        started_at = time.time()
        result = run_command(command, cwd=REPO_ROOT, timeout=args.openclaw_timeout + 60, check=False)
        ended_at = time.time()
        metadata = {
            "attempt_id": attempt_id,
            "attempt_number": attempt_number,
            "max_attempts": max_attempts,
            "command": command,
            "command_sha256": command_sha256(command),
            "session_key": session_key,
            "runner_version": RUNNER_VERSION,
            "cache_key": cache_key,
            "prompt_hash": prompt_hash,
            "expected_openclaw_result_path": str(sidecar_path),
            "nemoclaw_sandbox": getattr(args, "nemoclaw_sandbox", None),
            "nemoclaw_workdir": openclaw_nemoclaw_workdir(checkout_dir, args),
            "nemoclaw_checkout_sandbox_root": getattr(args, "nemoclaw_checkout_sandbox_root", None),
            "nemoclaw_checkout_transfer": checkout_transfer,
            "returncode": result.returncode,
            "stdout": result.stdout,
            "stderr": result.stderr,
            "started_at": started_at,
            "ended_at": ended_at,
            "wall_clock_time": ended_at - started_at,
        }
        invocation_path.write_text(
            json.dumps(metadata, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        metadata["openclaw_invocation_path"] = str(invocation_path)
        metadata["openclaw_invocation_sha256"] = sha256_file(invocation_path)
        metadata["openclaw_command_sha256"] = metadata["command_sha256"]
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
        if is_weave_sidecar_failure(sidecar):
            raise RuntimeError(
                f"Diagnostic Weave sidecar logging failed for {row['instance_id']}"
            )
        if is_runtime_budget_exceeded(sidecar):
            metadata["openclaw_disqualified_reason"] = "runtime_budget_exceeded"
            metadata["runtime_budget"] = sidecar.get("runtime_budget")
            break
        if isinstance(sidecar, dict) and (
            sidecar.get("tool_policy_ok") is False or sidecar.get("tool_policy_violations")
        ):
            metadata["openclaw_disqualified_reason"] = "tool_policy_violation"
            break
        if sidecar_conversation_order_failed(sidecar):
            metadata["openclaw_disqualified_reason"] = "conversation_order_violation"
            break
        if is_outer_openclaw_timeout(result, sidecar):
            metadata["openclaw_disqualified_reason"] = "time_up"
            metadata["openclaw_error"] = sidecar_error_text(sidecar, openclaw_failure_text(result, sidecar))
            print(
                f"OpenClaw timed out for {row['instance_id']} on attempt "
                f"{attempt_number}/{max_attempts}; recording an empty patch.",
                flush=True,
            )
            break
        non_scoreable_reason = non_scoreable_openclaw_failure_reason(result, sidecar)
        if non_scoreable_reason:
            recovery = None
            if non_scoreable_reason == "workspace_vanished" and attempt_number < max_attempts:
                recovery = recover_workspace_vanished_failure(task_dir, result, sidecar)
            if recovery:
                append_transient_failure(
                    task_dir,
                    {
                        "instance_id": row["instance_id"],
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
                    f"OpenClaw recoverable workspace attestation failure for {row['instance_id']} on attempt "
                    f"{attempt_number}/{max_attempts}; retrying after {delay:.1f}s.",
                    flush=True,
                )
                if delay:
                    time.sleep(delay)
                continue
            raise RuntimeError(
                f"OpenClaw non-scoreable setup/provider failure for {row['instance_id']}: "
                f"{non_scoreable_reason}"
            )
        if is_transient_openclaw_failure(result, sidecar):
            append_transient_failure(
                task_dir,
                {
                    "instance_id": row["instance_id"],
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
                    f"OpenClaw transient failure for {row['instance_id']} on attempt "
                    f"{attempt_number}/{max_attempts}; retrying after {delay:.1f}s.",
                    flush=True,
                )
                if delay:
                    time.sleep(delay)
                continue
            metadata["openclaw_disqualified_reason"] = "provider_transient_exhausted"
            print(
                f"OpenClaw transient failure exhausted for {row['instance_id']} after "
                f"{attempt_number}/{max_attempts} attempts; recording an empty patch.",
                flush=True,
            )
            break
        raise RuntimeError(f"OpenClaw failed for {row['instance_id']} with code {result.returncode}")
    if metadata is None or sidecar_path is None:
        raise RuntimeError(f"OpenClaw did not run for {row['instance_id']}")
    if not sidecar_path.exists():
        if metadata.get("openclaw_disqualified_reason") == "time_up":
            metadata.update(
                {
                    "openclaw_result_path": "",
                    "openclaw_returncode": metadata.get("returncode"),
                    "openclaw_usage": {},
                    "openclaw_tool_call_count": 0,
                    "openclaw_tool_error_count": 0,
                    "tool_policy_ok": None,
                    "tool_policy_violations": [],
                    "conversation_order_ok": None,
                    "conversation_order": {},
                    "nemoclaw_session_audit_ok": None,
                    "nemoclaw_session_audit": {},
                    "openclaw_config_source": cache_key.get("openclaw_config_source", ""),
                    "runtime_budget": {},
                    "weave_sidecar": {},
                    "weave_sidecar_ok": None,
                    **default_weave_agents_evidence(args, required=False),
                }
            )
            return metadata
        raise RuntimeError(f"OpenClaw completed without sidecar result: {sidecar_path}")
    sidecar = json.loads(sidecar_path.read_text(encoding="utf-8"))
    if not sidecar_identity_matches_cache(sidecar, cache_key):
        raise RuntimeError(
            f"OpenClaw sidecar metadata mismatch for {row['instance_id']}: "
            f"{sidecar_path}"
        )
    if is_weave_sidecar_failure(sidecar):
        raise RuntimeError(f"Diagnostic Weave sidecar logging failed for {row['instance_id']}")
    disqualified_reason = metadata.get("openclaw_disqualified_reason", "")
    if not disqualified_reason and sidecar_conversation_order_failed(sidecar):
        disqualified_reason = "conversation_order_violation"
    if not sidecar_nemoclaw_session_audit_matches_cache(sidecar, cache_key):
        if not disqualified_reason:
            raise RuntimeError(
                f"OpenClaw NeMoClaw session audit mismatch for {row['instance_id']}: "
                f"{sidecar_path}"
            )
        print(
            "OpenClaw NeMoClaw session audit failed for "
            f"{row['instance_id']} after {disqualified_reason}; "
            "recording as disqualified instead of aborting the run.",
            flush=True,
        )
    weave_agents_evidence = verify_weave_agents_for_attempt(
        row,
        task_dir,
        args,
        session_key=str(metadata.get("session_key") or ""),
        agent_id=agent_id,
        sidecar=sidecar,
    )
    metadata.update(
        {
            "openclaw_result_path": str(sidecar_path),
            "openclaw_returncode": sidecar.get("returncode"),
            "openclaw_usage": sidecar_usage(sidecar),
            "openclaw_tool_call_count": sidecar.get("tool_call_count", 0),
            "openclaw_tool_error_count": sidecar.get("tool_error_count", 0),
            "tool_policy_ok": sidecar.get("tool_policy_ok"),
            "tool_policy_violations": sidecar.get("tool_policy_violations", []),
            "conversation_order_ok": (sidecar.get("conversation_order") or {}).get("ok"),
            "conversation_order": sidecar.get("conversation_order", {}),
            "nemoclaw_session_audit_ok": (sidecar.get("nemoclaw_session_audit") or {}).get("ok"),
            "nemoclaw_session_audit": sidecar.get("nemoclaw_session_audit", {}),
            **nemoclaw_session_copy_evidence(sidecar),
            "openclaw_config_source": cache_key.get("openclaw_config_source", ""),
            "runtime_budget": sidecar.get("runtime_budget", {}),
            "weave_sidecar": sidecar.get("weave_sidecar", {}),
            "weave_sidecar_ok": (sidecar.get("weave_sidecar") or {}).get("ok"),
            "openclaw_disqualified_reason": disqualified_reason,
            **weave_agents_evidence,
        }
    )
    return metadata


def write_outputs(
    output_dir: Path,
    patches: list[dict[str, Any]],
    rows: list[dict[str, Any]],
    args: argparse.Namespace,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    patch_path = output_dir / "patches.json"
    patch_path.write_text(json.dumps(patches, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    jsonl_path = output_dir / "patches.jsonl"
    with jsonl_path.open("w", encoding="utf-8") as f:
        for patch in patches:
            f.write(json.dumps(patch, ensure_ascii=False) + "\n")
    summary = {
        "total_requested": len(rows),
        "patches_written": len(patches),
        "empty_patches": sum(1 for patch in patches if not patch.get("patch")),
        "runner_version": RUNNER_VERSION,
        "openclaw_num_workers": int(getattr(args, "openclaw_num_workers", 1) or 1),
        "openclaw_task_start_min_interval_seconds": float(
            getattr(args, "openclaw_task_start_min_interval_seconds", 0.0) or 0.0
        ),
        "runtime_budget": runtime_budget_summary(patches, args),
        "traced_patches": sum(1 for patch in patches if patch.get("openclaw_result_path")),
        "tool_called_patches": sum(1 for patch in patches if int(patch.get("openclaw_tool_call_count") or 0) > 0),
        "tool_error_patches": sum(1 for patch in patches if int(patch.get("openclaw_tool_error_count") or 0) > 0),
        "tool_policy_violation_patches": sum(1 for patch in patches if patch.get("tool_policy_violations")),
        "conversation_order_violation_patches": sum(
            1 for patch in patches if patch.get("conversation_order_ok") is False
        ),
        "nemoclaw_session_audit_required_patches": sum(
            1
            for patch in patches
            if isinstance(patch.get("nemoclaw_session_audit"), dict)
            and patch["nemoclaw_session_audit"].get("required") is True
        ),
        "nemoclaw_session_audit_passed_patches": sum(
            1
            for patch in patches
            if isinstance(patch.get("nemoclaw_session_audit"), dict)
            and patch["nemoclaw_session_audit"].get("required") is True
            and patch.get("nemoclaw_session_audit_ok") is True
        ),
        "nemoclaw_session_audit_failed_patches": sum(
            1
            for patch in patches
            if isinstance(patch.get("nemoclaw_session_audit"), dict)
            and patch["nemoclaw_session_audit"].get("required") is True
            and patch.get("nemoclaw_session_audit_ok") is False
        ),
        "runtime_budget_exceeded_patches": sum(
            1 for patch in patches if patch.get("openclaw_disqualified_reason") == "runtime_budget_exceeded"
        ),
        "disqualified_patches": sum(1 for patch in patches if patch.get("openclaw_disqualified_reason")),
        "patch_path": str(patch_path),
        "jsonl_path": str(jsonl_path),
    }
    (output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def runtime_budget_summary(
    patches: list[dict[str, Any]],
    args: argparse.Namespace,
) -> dict[str, int | None]:
    def observed_limit(name: str) -> int:
        return max(
            (
                int(((patch.get("runtime_budget") or {}).get("limits") or {}).get(name) or 0)
                for patch in patches
            ),
            default=0,
        )

    configured_input = int(getattr(args, "max_input_tokens", 0) or 0)
    configured_tools = int(getattr(args, "max_tool_calls", 0) or 0)
    configured_turns = int(getattr(args, "max_agent_turns", 0) or 0)
    configured_tool_wall = int(getattr(args, "max_tool_wall_seconds", 0) or 0)
    max_input_tokens = configured_input or observed_limit("max_input_tokens")
    max_tool_calls = configured_tools or observed_limit("max_tool_calls")
    max_agent_turns = configured_turns or observed_limit("max_agent_turns")
    return {
        "max_input_tokens": max_input_tokens or None,
        "max_tool_calls": max_tool_calls or None,
        "max_agent_turns": max_agent_turns or None,
        "max_tool_wall_seconds": configured_tool_wall or None,
    }


def filter_rows(rows: list[dict[str, Any]], args: argparse.Namespace) -> list[dict[str, Any]]:
    if args.instance_id:
        wanted = set(args.instance_id)
        rows = [row for row in rows if str(row.get("instance_id")) in wanted]
    if args.limit is not None:
        rows = rows[: args.limit]
    return rows


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-jsonl", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/swebench_pro_openclaw"))
    parser.add_argument("--checkout-root", type=Path, default=Path("outputs/swebench_pro_checkouts"))
    parser.add_argument("--prefix", default="openclaw")
    parser.add_argument("--model")
    parser.add_argument("--thinking", default="medium")
    parser.add_argument("--agent", default="main")
    parser.add_argument("--profile")
    parser.add_argument("--openclaw-config-template", type=Path)
    parser.add_argument("--openclaw-tool-profile", default="coding")
    parser.add_argument("--nemoclaw-bin", default="nemoclaw")
    parser.add_argument("--nemoclaw-sandbox")
    parser.add_argument("--nemoclaw-openclaw-config-path", default=NEMOCLAW_OPENCLAW_CONFIG_PATH)
    parser.add_argument(
        "--nemoclaw-workdir",
        help=(
            "Working directory inside the NeMoClaw sandbox. Defaults to the "
            "sandbox-visible checkout path."
        ),
    )
    parser.add_argument(
        "--nemoclaw-checkout-sandbox-root",
        help=(
            "Sandbox-visible root corresponding to --checkout-root. If omitted, "
            "the host checkout absolute path is assumed to be mounted unchanged."
        ),
    )
    parser.add_argument(
        "--nemoclaw-checkout-transfer-mode",
        choices=["visible", "copy"],
        default="visible",
        help=(
            "How to make host checkouts available to the NeMoClaw sandbox. "
            "'visible' requires a pre-mounted/shared checkout path. 'copy' "
            "uploads the checkout archive to the sandbox and captures git diff "
            "inside the sandbox."
        ),
    )
    parser.add_argument(
        "--nemoclaw-checkout-transfer-timeout",
        type=int,
        default=300,
        help="Timeout in seconds for extracting copied checkouts in the sandbox.",
    )
    parser.add_argument("--deny-tool", action="append")
    parser.add_argument("--deny-argument-pattern", action="append")
    parser.add_argument("--use-task-agent", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--task-agent-prefix", default="nejumi-swe")
    parser.add_argument(
        "--restart-gateway-after-task-agent-registration",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "Restart the NeMoClaw Gateway after bulk task-agent registration so "
            "parallel dynamic agents are visible to the running Gateway."
        ),
    )
    parser.add_argument("--session-prefix", default="swebench-pro")
    parser.add_argument("--openclaw-timeout", type=int, default=3600)
    parser.add_argument(
        "--openclaw-num-workers",
        type=int,
        default=1,
        help="Number of SWE-Bench Pro OpenClaw patch-generation tasks to run concurrently.",
    )
    parser.add_argument(
        "--openclaw-task-start-min-interval-seconds",
        type=float,
        default=0.0,
        help="Minimum interval between starting SWE-Bench Pro OpenClaw tasks.",
    )
    parser.add_argument(
        "--max-input-tokens",
        type=int,
        default=DEFAULT_MAX_INPUT_TOKENS,
        help=(
            "SWE-Bench Pro per-call/context input-token cap. If "
            "--max-cumulative-input-tokens is omitted, this value is also used "
            "as the cumulative provider input-token cap."
        ),
    )
    parser.add_argument(
        "--max-cumulative-input-tokens",
        type=int,
        default=None,
        help="SWE-Bench Pro per-task cumulative provider input-token cap. 0 disables it.",
    )
    parser.add_argument(
        "--max-cumulative-output-tokens",
        type=int,
        default=None,
        help="SWE-Bench Pro per-task cumulative provider output-token cap. 0 disables it.",
    )
    parser.add_argument(
        "--require-actual-token-usage",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Fail instances when provider token usage is not available from OpenClaw.",
    )
    parser.add_argument(
        "--max-tool-calls",
        type=int,
        default=DEFAULT_MAX_TOOL_CALLS,
        help="SWE-Bench Pro per-task tool-call budget. 0 disables the budget.",
    )
    parser.add_argument(
        "--max-agent-turns",
        type=int,
        default=DEFAULT_MAX_AGENT_TURNS,
        help="SWE-Bench Pro per-task assistant-turn budget. 0 disables the budget.",
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
        help="Maximum attempts per instance for transient OpenClaw/provider failures.",
    )
    parser.add_argument(
        "--openclaw-retry-base-seconds",
        type=float,
        default=15.0,
        help="Linear backoff base seconds between transient OpenClaw retries.",
    )
    parser.add_argument("--limit", type=int)
    parser.add_argument("--instance-id", action="append")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--no-local", action="store_true")
    parser.add_argument("--allow-failed-preflight", action="store_true")
    parser.add_argument("--no-reset", action="store_true")
    parser.add_argument("--redo", action="store_true", help="Regenerate patches even when cache keys match.")
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
        help="Fail if --weave-sidecar is enabled and the diagnostic sidecar trace cannot be logged.",
    )
    parser.add_argument(
        "--verify-weave-agents",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Require native weave-openclaw Agents traces to be visible after each instance.",
    )
    parser.add_argument("--weave-agents-entity", default=env_default_entity())
    parser.add_argument("--weave-agents-project", default=env_default_project())
    parser.add_argument("--weave-agents-agent-name", default=DEFAULT_WEAVE_AGENTS_AGENT_NAME)
    parser.add_argument("--weave-agents-env-file", type=Path, default=DEFAULT_WEAVE_AGENTS_ENV_FILE)
    parser.add_argument("--weave-agents-limit", type=int, default=50)
    parser.add_argument("--weave-agents-verification-timeout", type=float, default=120.0)
    parser.add_argument("--weave-agents-poll-seconds", type=float, default=5.0)
    parser.add_argument(
        "--skip-agent",
        action="store_true",
        help="Only prepare prompts/checkouts and collect any existing diff.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if getattr(args, "weave_sidecar", False) or getattr(args, "weave_sidecar_strict", False):
        raise SystemExit(
            "Weave sidecar logging is disabled for SWE-Bench Pro. Use native "
            "weave-openclaw Agents traces only; manual sidecar traces are not valid evidence."
        )
    rows = filter_rows(read_jsonl(args.dataset_jsonl), args)
    if not rows:
        raise SystemExit("No rows selected")

    patches_by_index: dict[int, dict[str, Any]] = {}
    args.output_dir.mkdir(parents=True, exist_ok=True)
    ensure_nemoclaw_openclaw_permissions(args)

    def ordered_patches() -> list[dict[str, Any]]:
        return [patches_by_index[index] for index in sorted(patches_by_index)]

    def build_patch_record(index: int, row: dict[str, Any]) -> tuple[int, dict[str, Any]]:
        instance_id = str(row["instance_id"])
        task_start_limiter.wait()
        print(f"[{index}/{len(rows)}] {instance_id}")
        task_dir = args.output_dir / safe_id(instance_id)
        task_dir.mkdir(parents=True, exist_ok=True)
        prompt_text = build_prompt(row, max_tool_wall_seconds=int(getattr(args, "max_tool_wall_seconds", 0) or 0))
        cache_key = build_cache_key(row, prompt_text, args)
        if not args.redo:
            cached_patch = load_cached_patch_record(
                task_dir,
                cache_key,
                args.prefix,
                selected_test_paths(row),
            )
            if cached_patch is not None:
                print(f"Reusing existing SWE-bench Pro patch: {instance_id}", flush=True)
                return index, cached_patch
        openclaw_metadata: dict[str, Any] = {}
        if args.dry_run:
            (task_dir / "prompt.md").write_text(prompt_text, encoding="utf-8")
            patch = ""
        else:
            checkout_dir = prepare_checkout(row, args.checkout_root, reset=not args.no_reset)
            if not args.skip_agent:
                openclaw_metadata = run_openclaw_for_task(row, checkout_dir, task_dir, args)
            elif (
                getattr(args, "nemoclaw_sandbox", None)
                and str(getattr(args, "nemoclaw_checkout_transfer_mode", "visible") or "visible") == "copy"
            ):
                openclaw_metadata["nemoclaw_checkout_transfer"] = ensure_nemoclaw_checkout_ready(
                    checkout_dir,
                    task_dir,
                    args,
                )
            if should_force_empty_patch(openclaw_metadata):
                patch = ""
            elif (
                getattr(args, "nemoclaw_sandbox", None)
                and str(getattr(args, "nemoclaw_checkout_transfer_mode", "visible") or "visible") == "copy"
            ):
                patch = capture_patch_nemoclaw(checkout_dir, args, selected_test_paths(row))
            else:
                patch = capture_patch(checkout_dir, selected_test_paths(row))
            (task_dir / "patch.diff").write_text(patch, encoding="utf-8")
        patch_record = {
            "instance_id": instance_id,
            "patch": patch,
            "prefix": args.prefix,
            "cache_key": cache_key,
            "patch_capture_version": PATCH_CAPTURE_VERSION,
            **default_weave_agents_evidence(args, required=False),
            **{
                key: value
                for key, value in openclaw_metadata.items()
                if key
                in {
                    "attempt_id",
                    "attempt_number",
                    "max_attempts",
                    "runner_version",
                    "prompt_hash",
                    "session_key",
                    "openclaw_invocation_path",
                    "openclaw_invocation_sha256",
                    "openclaw_command_sha256",
                    "openclaw_config_source",
                    "openclaw_result_path",
                    "openclaw_returncode",
                    "openclaw_usage",
                    "openclaw_tool_call_count",
                    "openclaw_tool_error_count",
                    "tool_policy_ok",
                    "tool_policy_violations",
                    "conversation_order_ok",
                    "conversation_order",
                    "nemoclaw_session_audit_ok",
                    "nemoclaw_session_audit",
                    "openclaw_disqualified_reason",
                    "nemoclaw_sandbox",
                    "nemoclaw_workdir",
                    "nemoclaw_checkout_sandbox_root",
                    "nemoclaw_checkout_transfer",
                    "runtime_budget",
                    "weave_sidecar",
                    "weave_sidecar_ok",
                    "weave_agents_ok",
                    "weave_agents_required",
                    "weave_agents_agent_name",
                    "weave_agents_conversation_id",
                    "weave_agents_conversation_id_contains",
                    "weave_agents_conversation_url",
                    "weave_agents_conversation_link_html",
                    "weave_agents_trace_id",
                    "weave_agents_url",
                    "weave_agents_trace_url",
                    "weave_agents_verifier_json",
                    "weave_agents_error",
                    "openclaw_error",
                }
            },
        }
        (task_dir / "patch_record.json").write_text(
            json.dumps(patch_record, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        return index, patch_record

    def merge_disk_patch_records() -> None:
        for index, row in enumerate(rows, start=1):
            if index in patches_by_index:
                continue
            task_dir = args.output_dir / safe_id(str(row["instance_id"]))
            prompt_text = build_prompt(
                row,
                max_tool_wall_seconds=int(getattr(args, "max_tool_wall_seconds", 0) or 0),
            )
            cache_key = build_cache_key(row, prompt_text, args)
            cached_patch = load_cached_patch_record(
                task_dir,
                cache_key,
                args.prefix,
                selected_test_paths(row),
            )
            if cached_patch is not None:
                patches_by_index[index] = cached_patch

    def pre_register_task_agents() -> None:
        if (
            not uses_nemoclaw_gateway_task_agent(args)
            or bool(getattr(args, "dry_run", False))
            or bool(getattr(args, "skip_agent", False))
        ):
            return
        print(
            "Pre-registering NeMoClaw Gateway task agents before parallel SWE-Bench Pro execution.",
            flush=True,
        )
        for index, row in enumerate(rows, start=1):
            instance_id = str(row["instance_id"])
            task_dir = args.output_dir / safe_id(instance_id)
            task_dir.mkdir(parents=True, exist_ok=True)
            checkout_dir = prepare_checkout(row, args.checkout_root, reset=not args.no_reset)
            write_task_openclaw_config(row, checkout_dir, task_dir, args)
            print(
                f"[{index}/{len(rows)}] Pre-registered SWE-Bench Pro task agent: {instance_id}",
                flush=True,
            )
        restart_nemoclaw_gateway_after_task_agent_registration(args, label="SWE-Bench Pro")

    openclaw_num_workers = max(1, int(getattr(args, "openclaw_num_workers", 1) or 1))
    task_start_limiter = TaskStartLimiter(
        float(getattr(args, "openclaw_task_start_min_interval_seconds", 0.0) or 0.0)
    )
    print(f"SWE-Bench Pro OpenClaw task workers: {openclaw_num_workers}", flush=True)
    if task_start_limiter.min_interval_seconds:
        print(
            "SWE-Bench Pro OpenClaw task start min interval: "
            f"{task_start_limiter.min_interval_seconds:.1f}s",
            flush=True,
        )
    if openclaw_num_workers == 1:
        for index, row in enumerate(rows, start=1):
            completed_index, patch_record = build_patch_record(index, row)
            patches_by_index[completed_index] = patch_record
            write_outputs(args.output_dir, ordered_patches(), rows, args)
    else:
        pre_register_task_agents()
        executor = ThreadPoolExecutor(max_workers=openclaw_num_workers)
        try:
            future_to_task = {
                executor.submit(build_patch_record, index, row): (index, row)
                for index, row in enumerate(rows, start=1)
            }
            for future in as_completed(future_to_task):
                completed_index, patch_record = future.result()
                patches_by_index[completed_index] = patch_record
                write_outputs(args.output_dir, ordered_patches(), rows, args)
        except Exception:
            merge_disk_patch_records()
            write_outputs(args.output_dir, ordered_patches(), rows, args)
            executor.shutdown(wait=False, cancel_futures=True)
            raise
        finally:
            executor.shutdown(wait=True, cancel_futures=False)


if __name__ == "__main__":
    main()
