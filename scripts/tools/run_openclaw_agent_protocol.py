#!/usr/bin/env python3
"""
Run Nejumi Taiwan agentic tasks through OpenClaw with W&B Weave tracing enabled.

The script intentionally does not edit ~/.openclaw/openclaw.json by default. Use
`write-weave-config` to generate the config snippet, inspect it, then merge it
into the OpenClaw gateway config used by the evaluation host.
"""

from __future__ import annotations

import argparse
import base64
import fnmatch
import tempfile
import hashlib
import json
import os
import re
import shlex
import shutil
import signal
import subprocess
import sys
import time
import urllib.error
import urllib.request
from datetime import datetime, timezone
from math import ceil
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[2]
PROTOCOL_VERSION = "nejumi-agent-protocol-2026.04"
MIN_NODE_VERSION = (22, 19, 0)
MIN_OPENCLAW_VERSION = (2026, 4, 25)
DEFAULT_ENV_FILE = REPO_ROOT / ".env"
WEAVE_SIDECAR_SCRIPT = REPO_ROOT / "scripts" / "tools" / "log_openclaw_result_to_weave.mjs"
DEFAULT_NATIVE_WEAVE_AGENT_NAME = "nejumi-taiwan-openclaw"
DEFAULT_DIAGNOSTIC_WEAVE_AGENT_NAME = "nejumi-taiwan-sidecar-diagnostic"
DEFAULT_NEMOCLAW_OPENCLAW_CONFIG_PATH = Path("/sandbox/.openclaw/openclaw.json")
DEFAULT_NEMOCLAW_PATCHED_OPENCLAW_BIN_DIR = "/sandbox/.npm-global/bin"
DEFAULT_NEMOCLAW_DEEPSWE_TOOL_BIN_DIR = "/sandbox/.deepswe-tools/go/bin"
SANDBOX_OPENCLAW_ENV_PASSTHROUGH = ("OPENCLAW_GATEWAY_URL",)
AGENTS_API_BASE_URL = "https://trace.wandb.ai"
AGENTS_QUERY_ENDPOINT = "/agents/query"
AGENTS_SPANS_QUERY_ENDPOINT = "/agents/spans/query"
AGENTS_TRACES_CHAT_ENDPOINT = "/agents/traces/chat"
AGENTS_SPANS_QUERY_MAX_LIMIT = 10_000
AGENTS_DIAGNOSTIC_SCHEMA_VERSION = 1
SANDBOX_LIVE_SESSION_SCAN_TIMEOUT = 10
SANDBOX_LIVE_SESSION_POLL_SECONDS = 1.0
BUDGET_GUARD_BLOCK_MARKER = "NEJUMI_BUDGET_GUARD_BLOCKED"
OPENCLAW_NO_RESPONSE_MARKERS = (
    "agent couldn't generate a response",
    "agent could not generate a response",
)
SANDBOX_LIVE_SESSION_SCAN_SCRIPT = r"""
import json
import sys
import time
from pathlib import Path

threshold = float(sys.argv[1])
rows = []
BUDGET_GUARD_BLOCK_MARKER = "NEJUMI_BUDGET_GUARD_BLOCKED"


def budget_guard_block_kind(text):
    if "tool_policy_violation" in text:
        return "tool_policy_violation"
    if "agent_turn_limit_exceeded" in text:
        return "agent_turn"
    if "tool_call_limit_exceeded" in text:
        return "tool_call"
    if "cumulative_input_tokens_limit_exceeded" in text:
        return "cumulative_input_tokens"
    if "cumulative_output_tokens_limit_exceeded" in text:
        return "cumulative_output_tokens"
    if "missing_actual_input_tokens" in text:
        return "missing_actual_input_tokens"
    if "missing_actual_output_tokens" in text:
        return "missing_actual_output_tokens"
    if "missing_actual_token_usage" in text:
        return "missing_actual_token_usage"
    return "unknown"


def tool_call_count(message):
    seen = set()
    count = 0

    def add_call(value, fallback):
        nonlocal count
        if not isinstance(value, dict):
            return
        call_id = value.get("id") or value.get("toolCallId") or value.get("tool_use_id")
        key = str(call_id) if isinstance(call_id, str) and call_id else fallback
        if key in seen:
            return
        seen.add(key)
        count += 1

    content = message.get("content")
    if isinstance(content, list):
        for index, part in enumerate(content):
            if isinstance(part, dict) and part.get("type") in {"toolCall", "tool_use"}:
                add_call(part, f"content:{index}")

    for key in ("tool_calls", "toolCalls"):
        calls = message.get(key)
        if isinstance(calls, list):
            for index, call in enumerate(calls):
                add_call(call, f"{key}:{index}")
    return count


def iter_tool_calls(message):
    content = message.get("content")
    if isinstance(content, list):
        for index, part in enumerate(content):
            if isinstance(part, dict) and part.get("type") in {"toolCall", "tool_use"}:
                yield part, f"content:{index}"

    for key in ("tool_calls", "toolCalls"):
        calls = message.get(key)
        if isinstance(calls, list):
            for index, call in enumerate(calls):
                if isinstance(call, dict):
                    yield call, f"{key}:{index}"


def tool_name(value):
    name = value.get("name") or value.get("toolName")
    function = value.get("function")
    if not isinstance(name, str) and isinstance(function, dict):
        name = function.get("name")
    return name if isinstance(name, str) else ""


def tool_arguments(value):
    arguments = value.get("arguments")
    if arguments is None:
        arguments = value.get("input")
    function = value.get("function")
    if arguments is None and isinstance(function, dict):
        arguments = function.get("arguments")
    if isinstance(arguments, str):
        try:
            return json.loads(arguments)
        except json.JSONDecodeError:
            return {"raw": arguments}
    return arguments if isinstance(arguments, dict) else {}


def forbidden_live_tool_policy_violations(message):
    violations = []
    for value, source in iter_tool_calls(message):
        name = tool_name(value)
        arguments = tool_arguments(value)
        name_norm = name.lower()
        if name_norm == "exec" and arguments.get("pty") is True:
            violations.append({"type": "forbidden_interactive_exec_pty", "toolName": name, "source": source})
    return violations


def live_provider_timeout_error(message):
    if not isinstance(message, dict):
        return None
    text = "\n".join(
        str(message.get(key) or "")
        for key in ("errorMessage", "errorCode", "errorBody")
    )
    if not text.strip():
        return None
    timeout_patterns = [
        "upstream idle timeout",
        "llm request timed out",
        "gateway timeout",
        "etimedout",
        "timeout exceeded",
        "timed out",
    ]
    is_timeout = any(pattern in text.lower() for pattern in timeout_patterns)
    error_code = str(message.get("errorCode") or "")
    if error_code in {"408", "504"}:
        is_timeout = True
    if not is_timeout:
        return None
    return {
        "type": "provider_timeout",
        "errorCode": error_code or None,
        "errorMessage": str(message.get("errorMessage") or "")[:500],
        "errorBody": str(message.get("errorBody") or "")[:1000],
    }


def text_from_value(value):
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    if isinstance(value, list):
        return "\n".join(text_from_value(item) for item in value)
    if isinstance(value, dict):
        texts = []
        for key in ("text", "content", "thinking", "arguments", "input", "partialArgs"):
            if key in value:
                texts.append(text_from_value(value.get(key)))
        if texts:
            return "\n".join(text for text in texts if text)
        try:
            return json.dumps(value, ensure_ascii=False)
        except TypeError:
            return str(value)
    return str(value)


def estimated_tokens(text):
    cjk = sum(1 for char in text if "\u3400" <= char <= "\u9fff" or "\uf900" <= char <= "\ufaff")
    non_cjk = max(0, len(text) - cjk)
    return cjk + int((non_cjk + 3) // 4)


now = time.time()
for raw_dir in sys.argv[2:]:
    sessions_dir = Path(raw_dir)
    if not sessions_dir.exists() or not sessions_dir.is_dir():
        continue
    for path in sessions_dir.glob("*.jsonl"):
        if path.name.endswith(".trajectory.jsonl"):
            continue
        try:
            stat = path.stat()
        except OSError:
            continue
        if stat.st_mtime < threshold:
            continue
        tool_calls = 0
        try:
            lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
        except OSError:
            continue
        estimated_input_tokens = 0
        agent_turn_count = 0
        executed_tool_results = 0
        blocked_tool_results = 0
        policy_violations = []
        budget_guard_blocks = []
        provider_timeouts = []
        last_message_role = None
        last_assistant_tool_call_count = None
        last_assistant_text_length = 0
        for line_index, raw in enumerate(lines):
            if BUDGET_GUARD_BLOCK_MARKER in raw:
                budget_guard_blocks.append(
                    {
                        "line_index": line_index,
                        "marker": BUDGET_GUARD_BLOCK_MARKER,
                        "kind": budget_guard_block_kind(raw),
                        "text": raw[:1000],
                    }
                )
            try:
                event = json.loads(raw)
            except json.JSONDecodeError:
                continue
            message = event.get("message") if isinstance(event, dict) else None
            if not isinstance(message, dict):
                continue
            estimated_input_tokens += estimated_tokens(text_from_value(message.get("content")))
            role = message.get("role")
            if isinstance(role, str):
                last_message_role = role
            if role == "assistant":
                assistant_tool_calls = tool_call_count(message)
                agent_turn_count += 1
                tool_calls += assistant_tool_calls
                last_assistant_tool_call_count = assistant_tool_calls
                last_assistant_text_length = len(text_from_value(message.get("content")).strip())
                policy_violations.extend(forbidden_live_tool_policy_violations(message))
                provider_timeout = live_provider_timeout_error(message)
                if provider_timeout is not None:
                    provider_timeout["line_index"] = line_index
                    provider_timeouts.append(provider_timeout)
            elif role in {"toolResult", "tool"}:
                tool_result_text = "\n".join(
                    [
                        text_from_value(message.get("content")),
                        text_from_value(message.get("details")),
                    ]
                )
                if BUDGET_GUARD_BLOCK_MARKER in tool_result_text:
                    blocked_tool_results += 1
                else:
                    executed_tool_results += 1
        lock_exists = Path(str(path) + ".lock").exists()
        idle_seconds = max(0.0, now - stat.st_mtime)
        final_assistant_idle_done = (
            not lock_exists
            and last_message_role == "assistant"
            and last_assistant_tool_call_count == 0
            and last_assistant_text_length > 0
        )
        rows.append(
            {
                "path": str(path),
                "mtime": stat.st_mtime,
                "idle_seconds": idle_seconds,
                "lock_exists": lock_exists,
                "last_message_role": last_message_role,
                "last_assistant_tool_call_count": last_assistant_tool_call_count,
                "last_assistant_text_length": last_assistant_text_length,
                "final_assistant_idle_done": final_assistant_idle_done,
                "tool_call_count": tool_calls,
                "blocked_tool_call_count": blocked_tool_results,
                "executed_tool_call_count": executed_tool_results,
                "estimated_input_tokens": estimated_input_tokens,
                "agent_turn_count": agent_turn_count,
                "live_tool_policy_violation_count": len(policy_violations),
                "live_tool_policy_violations": policy_violations[:10],
                "budget_guard_block_count": len(budget_guard_blocks),
                "budget_guard_blocks": budget_guard_blocks[:10],
                "live_provider_timeout_count": len(provider_timeouts),
                "live_provider_timeouts": provider_timeouts[:10],
            }
        )
rows.sort(
    key=lambda row: (
        row["tool_call_count"],
        row["estimated_input_tokens"],
        row["agent_turn_count"],
        row["mtime"],
    ),
    reverse=True,
)
print(json.dumps({"ok": True, "sessions": rows}, ensure_ascii=False))
""".strip()
FINAL_ANSWER_MARKERS = (
    "ANSWER:",
    "FINAL ANSWER",
    "Final answer",
    "CANARY_RESULT",
)
TOOL_CALL_TYPE_MARKERS = {
    "toolCall",
    "tool_call",
    "toolUse",
    "tool_use",
    "function_call",
}
TOOL_CALL_KEYS = (
    "tool_call",
    "tool_calls",
    "toolCalls",
    "toolCall",
    "toolUse",
    "tool_use",
    "function_call",
)


def parse_version(text: str) -> tuple[int, int, int] | None:
    match = re.search(r"(\d+)\.(\d+)\.(\d+)", text)
    if not match:
        return None
    return tuple(int(part) for part in match.groups())


def version_gte(current: tuple[int, int, int] | None, minimum: tuple[int, int, int]) -> bool:
    return current is not None and current >= minimum


def run_quiet(command: list[str], env: dict[str, str] | None = None) -> subprocess.CompletedProcess[str] | None:
    try:
        return subprocess.run(command, text=True, capture_output=True, check=False, env=env)
    except FileNotFoundError:
        return None


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def read_text(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def load_env_file(env: dict[str, str], env_file: Path | None) -> tuple[dict[str, str], bool]:
    if env_file is None:
        return env, False
    path = env_file.expanduser()
    if not path.exists():
        return env, False
    loaded = False
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        if not key or not re.match(r"^[A-Za-z_][A-Za-z0-9_]*$", key):
            continue
        value = value.strip().strip('"').strip("'")
        if key not in env:
            env[key] = value
            loaded = True
    return env, loaded


def nvm_bin_dirs() -> list[Path]:
    root = Path.home() / ".nvm" / "versions" / "node"
    if not root.exists():
        return []
    dirs: list[Path] = []
    for candidate in root.glob("v*/bin"):
        node = candidate / "node"
        openclaw = candidate / "openclaw"
        if node.exists() or openclaw.exists():
            dirs.append(candidate)

    def key(path: Path) -> tuple[int, int, int]:
        return parse_version(path.parent.name) or (0, 0, 0)

    return sorted(dirs, key=key, reverse=True)


def resolve_openclaw_bin(openclaw_bin: str, env: dict[str, str]) -> str | None:
    path = Path(openclaw_bin).expanduser()
    if path.is_absolute() or "/" in openclaw_bin:
        return str(path) if path.exists() else None
    found = shutil.which(openclaw_bin, path=env.get("PATH"))
    if found:
        return found
    for bin_dir in nvm_bin_dirs():
        candidate = bin_dir / openclaw_bin
        if candidate.exists():
            return str(candidate)
    return None


def resolve_executable_bin(binary: str, env: dict[str, str]) -> str | None:
    path = Path(binary).expanduser()
    if path.is_absolute() or "/" in binary:
        return str(path) if path.exists() else None
    found = shutil.which(binary, path=env.get("PATH"))
    if found:
        return found
    for bin_dir in nvm_bin_dirs():
        candidate = bin_dir / binary
        if candidate.exists():
            return str(candidate)
    return None


def prepare_env(env_file: Path | None, openclaw_bin: str | None = None) -> tuple[dict[str, str], bool, str | None]:
    env, loaded = load_env_file(os.environ.copy(), env_file)
    resolved = resolve_openclaw_bin(openclaw_bin or "openclaw", env) if openclaw_bin else None
    if resolved:
        bin_dir = str(Path(resolved).parent)
        env["PATH"] = bin_dir + os.pathsep + env.get("PATH", "")
    else:
        for bin_dir in nvm_bin_dirs():
            env["PATH"] = str(bin_dir) + os.pathsep + env.get("PATH", "")
            break
    return env, loaded, resolved


def preflight(
    openclaw_bin: str,
    env_file: Path | None = DEFAULT_ENV_FILE,
    *,
    nemoclaw_bin: str = "nemoclaw",
    nemoclaw_sandbox: str | None = None,
) -> dict[str, Any]:
    env, env_loaded, resolved_openclaw = prepare_env(env_file, openclaw_bin)
    node_path = shutil.which("node", path=env.get("PATH"))
    node_result = run_quiet([node_path, "--version"], env=env) if node_path else None
    node_version = parse_version(node_result.stdout if node_result else "")

    openclaw_path = resolved_openclaw
    openclaw_result = run_quiet([openclaw_path, "--version"], env=env) if openclaw_path else None
    openclaw_version = parse_version(
        (openclaw_result.stdout + openclaw_result.stderr) if openclaw_result else ""
    )

    resolved_nemoclaw = resolve_executable_bin(nemoclaw_bin, env)
    nemoclaw_result = (
        run_quiet([resolved_nemoclaw, "--version"], env=env)
        if resolved_nemoclaw
        else None
    )
    nemoclaw_status_result = (
        run_quiet([resolved_nemoclaw, "sandbox", "status", nemoclaw_sandbox], env=env)
        if resolved_nemoclaw and nemoclaw_sandbox
        else None
    )
    nemoclaw_openclaw_result = (
        run_quiet(
            [
                resolved_nemoclaw,
                "sandbox",
                "exec",
                nemoclaw_sandbox,
                "--no-tty",
                "--timeout",
                "30",
                "--",
                "openclaw",
                "--version",
            ],
            env=env,
        )
        if resolved_nemoclaw and nemoclaw_sandbox
        else None
    )

    result = {
        "node_min_version": ".".join(map(str, MIN_NODE_VERSION)),
        "node_path": node_path,
        "node_version": ".".join(map(str, node_version)) if node_version else None,
        "node_ok": version_gte(node_version, MIN_NODE_VERSION),
        "openclaw_min_version": ".".join(map(str, MIN_OPENCLAW_VERSION)),
        "openclaw_path": openclaw_path,
        "openclaw_version": ".".join(map(str, openclaw_version)) if openclaw_version else None,
        "openclaw_ok": version_gte(openclaw_version, MIN_OPENCLAW_VERSION),
        "env_file": str(env_file.expanduser()) if env_file else None,
        "env_file_loaded": env_loaded,
        "env_probe": {
            "WANDB_API_KEY": bool(env.get("WANDB_API_KEY")),
            "OPENAI_API_KEY": bool(env.get("OPENAI_API_KEY")),
            "OPENCLAW_CONFIG_PATH": bool(env.get("OPENCLAW_CONFIG_PATH")),
        },
        "weave_plugin_install_command": "openclaw plugins install weave-openclaw",
        "nemoclaw": {
            "sandbox": nemoclaw_sandbox,
            "bin": resolved_nemoclaw,
            "version_output": (
                (nemoclaw_result.stdout + nemoclaw_result.stderr).strip()
                if nemoclaw_result
                else None
            ),
            "installed": bool(resolved_nemoclaw and nemoclaw_result and nemoclaw_result.returncode == 0),
            "sandbox_status_ok": (
                bool(nemoclaw_status_result and nemoclaw_status_result.returncode == 0)
                if nemoclaw_sandbox
                else None
            ),
            "sandbox_openclaw_ok": (
                bool(nemoclaw_openclaw_result and nemoclaw_openclaw_result.returncode == 0)
                if nemoclaw_sandbox
                else None
            ),
        },
    }
    result["ok"] = bool(result["node_ok"] and result["openclaw_ok"])
    if nemoclaw_sandbox:
        result["ok"] = bool(
            result["node_ok"]
            and result["nemoclaw"]["installed"]
            and result["nemoclaw"]["sandbox_status_ok"]
            and result["nemoclaw"]["sandbox_openclaw_ok"]
        )
    return result


def write_weave_config(args: argparse.Namespace) -> None:
    config = {
        "diagnostics": {"enabled": True},
        "plugins": {
            "allow": ["weave"],
            "entries": {
                "weave": {
                    "enabled": True,
                    "config": {
                        "entity": args.entity,
                        "project": args.project,
                        "apiKey": {
                            "source": "env",
                            "provider": "default",
                            "id": "WANDB_API_KEY",
                        },
                        "serviceName": args.service_name,
                        "agentName": args.agent_name,
                        "agentVersion": args.agent_version,
                        "agentDescription": args.agent_description,
                        "captureContent": args.capture_content,
                        "flushIntervalMs": args.flush_interval_ms,
                    },
                    "hooks": {"allowConversationAccess": args.allow_conversation_access},
                }
            },
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(config, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(args.output)


def metadata_header(args: argparse.Namespace, prompt_text: str) -> tuple[str, dict[str, Any]]:
    tool_policy = tool_policy_text(args)
    verifier = read_text(args.verifier) if args.verifier else ""
    metadata = {
        "protocol_version": PROTOCOL_VERSION,
        "benchmark_id": args.benchmark_id,
        "task_id": args.task_id,
        "model_id": args.model or "",
        "agent_runtime": "openclaw",
        "prompt_hash": sha256_text(prompt_text),
        "tool_policy_hash": sha256_text(tool_policy),
        "verifier_hash": sha256_text(verifier),
        "openclaw_config_source": getattr(args, "openclaw_config_source", None) or "",
    }
    header = "\n".join(
        [
            "<nejumi_agent_protocol>",
            *[f"{key}: {value}" for key, value in metadata.items()],
            "</nejumi_agent_protocol>",
            "",
        ]
    )
    return header, metadata


def load_tool_policy(args: argparse.Namespace) -> dict[str, Any]:
    policy: dict[str, Any] = {
        "deny_tools": list(args.deny_tool or []),
        "deny_argument_patterns": list(args.deny_argument_pattern or []),
    }
    if args.tool_policy:
        raw = read_text(args.tool_policy)
        try:
            parsed = json.loads(raw)
        except json.JSONDecodeError:
            parsed = None
        if isinstance(parsed, dict):
            deny_tools = parsed.get("deny_tools") or parsed.get("deny")
            if isinstance(deny_tools, list):
                policy["deny_tools"].extend(str(item) for item in deny_tools)
            tools = parsed.get("tools")
            if isinstance(tools, dict) and isinstance(tools.get("deny"), list):
                policy["deny_tools"].extend(str(item) for item in tools["deny"])
            deny_args = parsed.get("deny_argument_patterns") or parsed.get("deny_args")
            if isinstance(deny_args, list):
                policy["deny_argument_patterns"].extend(str(item) for item in deny_args)
    policy["deny_tools"] = sorted(set(str(item) for item in policy["deny_tools"] if str(item).strip()))
    policy["deny_argument_patterns"] = sorted(
        set(str(item) for item in policy["deny_argument_patterns"] if str(item).strip())
    )
    return policy


def tool_policy_text(args: argparse.Namespace) -> str:
    raw = read_text(args.tool_policy) if args.tool_policy else ""
    explicit = {
        "deny_tools": list(args.deny_tool or []),
        "deny_argument_patterns": list(args.deny_argument_pattern or []),
    }
    return raw + "\n" + json.dumps(explicit, ensure_ascii=False, sort_keys=True)


def _tool_name_matches(tool_name: str, pattern: str) -> bool:
    tool_name_norm = tool_name.lower()
    pattern_norm = pattern.lower()
    if pattern_norm.startswith("re:"):
        return re.search(pattern_norm[3:], tool_name_norm) is not None
    if "*" in pattern_norm or "?" in pattern_norm:
        return fnmatch.fnmatch(tool_name_norm, pattern_norm)
    return tool_name_norm == pattern_norm


def _tool_arguments_text(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    if isinstance(value, dict):
        return "\n".join(_tool_arguments_text(item) for item in value.values())
    if isinstance(value, (list, tuple)):
        return "\n".join(_tool_arguments_text(item) for item in value)
    return str(value)


def _tool_arguments_may_execute(tool_name: str) -> bool:
    tool_name_norm = tool_name.lower()
    executable_names = {
        "bash",
        "code_execution",
        "exec",
        "python",
        "python_exec",
        "shell",
        "terminal",
    }
    if tool_name_norm in executable_names:
        return True
    return any(token in tool_name_norm for token in ("exec", "shell", "terminal", "bash"))


def _is_bare_url_pattern(pattern: str) -> bool:
    return pattern.strip() in {r"https?://", "https?://"}


def tool_policy_violations(tool_events: list[dict[str, Any]], policy: dict[str, Any]) -> list[dict[str, Any]]:
    violations: list[dict[str, Any]] = []
    deny_tools = [str(item) for item in policy.get("deny_tools", [])]
    deny_argument_patterns = [str(item) for item in policy.get("deny_argument_patterns", [])]
    for event in tool_events:
        if event.get("type") != "tool_call":
            continue
        tool_name = str(event.get("toolName") or "")
        for pattern in deny_tools:
            if _tool_name_matches(tool_name, pattern):
                violations.append(
                    {
                        "type": "denied_tool",
                        "toolName": tool_name,
                        "pattern": pattern,
                        "toolCallId": event.get("toolCallId"),
                        "index": event.get("index"),
                    }
                )
        if not _tool_arguments_may_execute(tool_name):
            continue
        args_text = _tool_arguments_text(event.get("arguments"))
        for pattern in deny_argument_patterns:
            if _is_bare_url_pattern(pattern):
                continue
            if re.search(pattern, args_text, flags=re.IGNORECASE):
                violations.append(
                    {
                        "type": "denied_argument_pattern",
                        "toolName": tool_name,
                        "pattern": pattern,
                        "toolCallId": event.get("toolCallId"),
                        "index": event.get("index"),
                    }
                )
    return violations


def build_openclaw_agent_args(
    args: argparse.Namespace,
    message_text: str,
    openclaw_bin: str | None = None,
) -> list[str]:
    command = [openclaw_bin or args.openclaw_bin]
    if args.profile:
        command.extend(["--profile", args.profile])
    command.extend(
        [
            "agent",
            "--agent",
            args.agent,
            "--session-key",
            args.session_key or f"{args.benchmark_id}:{args.task_id}",
            "--message",
            message_text,
            "--timeout",
            str(args.timeout),
            "--json",
        ]
    )
    if args.local:
        command.append("--local")
    if args.model:
        command.extend(["--model", args.model])
    if args.thinking:
        command.extend(["--thinking", args.thinking])
    return command


def build_openclaw_command(args: argparse.Namespace, message_text: str, openclaw_bin: str | None = None) -> list[str]:
    return build_openclaw_command_with_message_source(args, message_text, openclaw_bin)


def build_openclaw_command_with_message_source(
    args: argparse.Namespace,
    message_text: str,
    openclaw_bin: str | None = None,
    sandbox_message_path: str | None = None,
) -> list[str]:
    openclaw_command = build_openclaw_agent_args(args, message_text, openclaw_bin)
    if not getattr(args, "nemoclaw_sandbox", None):
        return openclaw_command
    command = [
        args.nemoclaw_bin,
        "sandbox",
        "exec",
        args.nemoclaw_sandbox,
    ]
    nemoclaw_workdir = getattr(args, "nemoclaw_workdir", None)
    if nemoclaw_workdir:
        command.extend(["--workdir", nemoclaw_workdir])
    command.extend(["--no-tty", "--timeout", str(args.timeout + 60), "--"])
    message_b64 = base64.b64encode(message_text.encode("utf-8")).decode("ascii")
    shell_parts: list[str] = []
    replace_next_message = False
    for part in openclaw_command:
        if replace_next_message:
            shell_parts.append('"$OPENCLAW_MESSAGE"')
            replace_next_message = False
            continue
        shell_parts.append(shlex.quote(str(part)))
        if part == "--message":
            replace_next_message = True
    patched_openclaw_bin_dir = str(
        getattr(args, "nemoclaw_patched_openclaw_bin_dir", None)
        or DEFAULT_NEMOCLAW_PATCHED_OPENCLAW_BIN_DIR
    )
    if sandbox_message_path:
        message_loader = 'OPENCLAW_MESSAGE="$(cat "$OPENCLAW_MESSAGE_FILE")"; '
    else:
        message_loader = 'OPENCLAW_MESSAGE="$(printf %s "$OPENCLAW_MESSAGE_B64" | base64 -d)"; '
    deepswe_tool_bin_dir = DEFAULT_NEMOCLAW_DEEPSWE_TOOL_BIN_DIR
    extra_path_entries = [
        str(item)
        for item in (getattr(args, "nemoclaw_extra_path", None) or [])
        if str(item).strip()
    ]
    extra_pythonpath_entries = [
        str(item)
        for item in (getattr(args, "nemoclaw_extra_pythonpath", None) or [])
        if str(item).strip()
    ]
    path_entries = [
        deepswe_tool_bin_dir,
        *extra_path_entries,
        patched_openclaw_bin_dir,
    ]
    export_parts = [
        "export PATH="
        + ":".join(shlex.quote(item) for item in path_entries)
        + ":$PATH; "
    ]
    if extra_pythonpath_entries:
        export_parts.append(
            "export PYTHONPATH="
            + ":".join(shlex.quote(item) for item in extra_pythonpath_entries)
            + ":${PYTHONPATH:-}; "
        )
    shell_command = (
        "".join(export_parts)
        + message_loader
        + "exec "
        + " ".join(shell_parts)
    )
    config_path = effective_openclaw_config_path(args)
    command.append("env")
    if config_path:
        command.append(f"OPENCLAW_CONFIG_PATH={config_path}")
    for key in SANDBOX_OPENCLAW_ENV_PASSTHROUGH:
        value = os.environ.get(key)
        if value and "\n" not in value and "\r" not in value:
            command.append(f"{key}={value}")
    if sandbox_message_path:
        command.append(f"OPENCLAW_MESSAGE_FILE={sandbox_message_path}")
    else:
        command.append(f"OPENCLAW_MESSAGE_B64={message_b64}")
    command.extend(["bash", "-c", shell_command])
    return command


def write_nemoclaw_message_file(
    args: argparse.Namespace,
    message_text: str,
    env: dict[str, str] | None,
) -> str:
    message_hash = sha256_text(message_text)[:24]
    sandbox_path = f"/tmp/nejumi-openclaw-messages/{args.benchmark_id}-{args.task_id}-{message_hash}.md"
    script = (
        "import sys; "
        "from pathlib import Path; "
        "path = Path(sys.argv[1]); "
        "path.parent.mkdir(parents=True, exist_ok=True); "
        "path.write_text(sys.stdin.read(), encoding='utf-8')"
    )
    command = [
        args.nemoclaw_bin,
        "sandbox",
        "exec",
        args.nemoclaw_sandbox,
        "--workdir",
        "/sandbox",
        "--no-tty",
        "--timeout",
        "60",
        "--",
        "python3",
        "-c",
        script,
        sandbox_path,
    ]
    result = subprocess.run(
        command,
        input=message_text,
        text=True,
        capture_output=True,
        check=False,
        env=env,
    )
    if result.returncode != 0:
        raise RuntimeError(
            "Failed to write OpenClaw message into NeMoClaw sandbox\n"
            f"cmd: {' '.join(shlex.quote(str(part)) for part in command)}\n"
            f"returncode: {result.returncode}\n"
            f"stdout:\n{result.stdout}\n"
            f"stderr:\n{result.stderr}"
        )
    return sandbox_path


def effective_openclaw_config_path(args: argparse.Namespace) -> Path | None:
    configured = getattr(args, "openclaw_config_path", None)
    if configured:
        return Path(configured)
    if getattr(args, "nemoclaw_sandbox", None):
        return DEFAULT_NEMOCLAW_OPENCLAW_CONFIG_PATH
    return None


def openclaw_state_dir(args: argparse.Namespace) -> Path:
    explicit = os.environ.get("OPENCLAW_STATE_DIR")
    if explicit:
        return Path(explicit).expanduser()
    profile = getattr(args, "profile", None)
    if profile:
        return Path.home() / f".openclaw-{profile}"
    return Path.home() / ".openclaw"


def agent_sessions_dir(args: argparse.Namespace) -> Path:
    return openclaw_state_dir(args) / "agents" / str(args.agent) / "sessions"


def configured_live_session_dirs(args: argparse.Namespace) -> list[Path]:
    dirs: list[Path] = []
    seen: set[str] = set()
    for value in getattr(args, "live_session_dir", None) or []:
        path = Path(value).expanduser()
        key = str(path)
        if key not in seen:
            dirs.append(path)
            seen.add(key)
    default_dir = agent_sessions_dir(args)
    key = str(default_dir)
    if key not in seen:
        dirs.append(default_dir)
    return dirs


def configured_live_sandbox_session_dirs(args: argparse.Namespace) -> list[str]:
    dirs: list[str] = []
    seen: set[str] = set()

    def add_dir(value: str | None) -> None:
        if value is None:
            return
        path = str(value).strip()
        if not path or "\n" in path or "\r" in path or "\x00" in path:
            return
        if path not in seen:
            dirs.append(path)
            seen.add(path)

    for value in getattr(args, "live_sandbox_session_dir", None) or []:
        add_dir(value)
    agent = str(getattr(args, "agent", "") or "").strip()
    if getattr(args, "nemoclaw_sandbox", None) and agent and not any(
        char in agent for char in ("/", "\n", "\r", "\x00")
    ):
        add_dir(f"/sandbox/.openclaw/agents/{agent}/sessions")
    return dirs


def live_session_candidates(args: argparse.Namespace, started_at: float) -> list[Path]:
    candidates: list[Path] = []
    threshold = started_at - 5.0
    for sessions_dir in configured_live_session_dirs(args):
        if not sessions_dir.exists():
            continue
        for path in sessions_dir.glob("*.jsonl"):
            if path.name.endswith(".trajectory.jsonl"):
                continue
            try:
                if path.stat().st_mtime >= threshold:
                    candidates.append(path)
            except OSError:
                continue
    return sorted(
        candidates,
        key=lambda path: path.stat().st_mtime if path.exists() else 0.0,
        reverse=True,
    )


def estimate_text_tokens(text: str) -> int:
    cjk_chars = sum(
        1
        for char in text
        if ("\u3400" <= char <= "\u9fff") or ("\uf900" <= char <= "\ufaff")
    )
    non_cjk_chars = max(0, len(text) - cjk_chars)
    return cjk_chars + ceil(non_cjk_chars / 4)


def estimate_session_input_tokens_from_timeline(timeline_events: list[dict[str, Any]]) -> int:
    total = 0
    for event in timeline_events:
        event_type = event.get("type")
        if event_type in {"user_message", "assistant_message", "assistant_reasoning", "tool_result"}:
            total += estimate_text_tokens(_jsonish_text(event.get("content")))
        elif event_type == "tool_call":
            total += estimate_text_tokens(_jsonish_text(event.get("arguments")))
    return total


def session_message_role_counts(path: Path) -> dict[str, int]:
    counts = {"assistant": 0, "user": 0, "tool": 0}
    if not path.exists():
        return counts
    for raw in path.read_text(encoding="utf-8", errors="replace").splitlines():
        try:
            event = json.loads(raw)
        except json.JSONDecodeError:
            continue
        message = event.get("message") if isinstance(event, dict) else None
        if not isinstance(message, dict):
            continue
        role = message.get("role")
        if role == "assistant":
            counts["assistant"] += 1
        elif role == "user":
            counts["user"] += 1
        elif role in {"tool", "toolResult"}:
            counts["tool"] += 1
    return counts


def session_budget_guard_blocks(path: Path) -> list[dict[str, Any]]:
    marker = "NEJUMI_BUDGET_GUARD_BLOCKED"
    blocks: list[dict[str, Any]] = []
    if not path.exists():
        return blocks
    for line_index, raw in enumerate(path.read_text(encoding="utf-8", errors="replace").splitlines()):
        if marker in raw:
            blocks.append(
                {
                    "source": "session",
                    "line_index": line_index,
                    "marker": marker,
                    "kind": budget_guard_block_kind(raw),
                    "text": raw[:1000],
                }
            )
    return blocks


def budget_guard_block_kind(text: str) -> str:
    if "tool_policy_violation" in text:
        return "tool_policy_violation"
    if "agent_turn_limit_exceeded" in text:
        return "agent_turn"
    if "tool_call_limit_exceeded" in text:
        return "tool_call"
    if "cumulative_input_tokens_limit_exceeded" in text:
        return "cumulative_input_tokens"
    if "cumulative_output_tokens_limit_exceeded" in text:
        return "cumulative_output_tokens"
    if "missing_actual_input_tokens" in text:
        return "missing_actual_input_tokens"
    if "missing_actual_output_tokens" in text:
        return "missing_actual_output_tokens"
    if "missing_actual_token_usage" in text:
        return "missing_actual_token_usage"
    return "unknown"


def budget_guard_block_fields(text: str) -> dict[str, Any]:
    fields: dict[str, Any] = {}
    for match in re.finditer(r"\b([A-Za-z][A-Za-z0-9_]*)=([^\s]+)", text or ""):
        key = match.group(1)
        value = match.group(2)
        if key in {"observed", "limit", "observedCalls"} and re.fullmatch(r"-?\d+", value):
            fields[key] = int(value)
        else:
            fields[key] = value
    return fields


def stdio_budget_guard_blocks(sidecar: dict[str, Any]) -> list[dict[str, Any]]:
    marker = BUDGET_GUARD_BLOCK_MARKER
    blocks: list[dict[str, Any]] = []
    for stream_name in ("stdout", "stderr"):
        stream_text = str(sidecar.get(stream_name) or "")
        for line_index, line in enumerate(stream_text.splitlines()):
            if marker not in line:
                continue
            blocks.append(
                {
                    "source": stream_name,
                    "line_index": line_index,
                    "marker": marker,
                    "kind": budget_guard_block_kind(line),
                    "text": line[:1000],
                }
            )
    return blocks


def is_budget_guard_blocked_tool_result(event: dict[str, Any]) -> bool:
    if event.get("type") != "tool_result":
        return False
    text = "\n".join(
        [
            _jsonish_text(event.get("content")),
            _jsonish_text(event.get("details")),
        ]
    )
    return "NEJUMI_BUDGET_GUARD_BLOCKED" in text


def tool_result_execution_counts(tool_events: list[dict[str, Any]]) -> tuple[int, int]:
    executed = 0
    blocked = 0
    for event in tool_events:
        if event.get("type") != "tool_result":
            continue
        if is_budget_guard_blocked_tool_result(event):
            blocked += 1
        else:
            executed += 1
    return executed, blocked


def session_budget_observation(path: Path) -> dict[str, Any]:
    sidecar = {"live_session_file": str(path)}
    tool_events = extract_tool_events(sidecar)
    timeline_events = extract_timeline_events(sidecar)
    role_counts = session_message_role_counts(path)
    budget_guard_blocks = session_budget_guard_blocks(path)
    tool_call_count = sum(1 for event in tool_events if event.get("type") == "tool_call")
    executed_tool_call_count, blocked_tool_call_count = tool_result_execution_counts(tool_events)
    return {
        "source": "host",
        "tool_call_count": tool_call_count,
        "blocked_tool_call_count": blocked_tool_call_count,
        "executed_tool_call_count": executed_tool_call_count,
        "estimated_input_tokens": estimate_session_input_tokens_from_timeline(timeline_events),
        "agent_turn_count": role_counts["assistant"],
        "session_file": str(path),
        "budget_guard_block_count": len(budget_guard_blocks),
        "budget_guard_blocks": budget_guard_blocks[:10],
    }


def scan_nemoclaw_live_sessions(
    args: argparse.Namespace,
    started_at: float,
    env: dict[str, str] | None = None,
) -> dict[str, Any]:
    sandbox_dirs = configured_live_sandbox_session_dirs(args)
    if not getattr(args, "nemoclaw_sandbox", None) or not sandbox_dirs:
        return {
            "enabled": False,
            "ok": None,
            "reason": "not_configured",
            "sandbox": getattr(args, "nemoclaw_sandbox", None),
            "session_dirs": sandbox_dirs,
            "sessions": [],
        }
    scan_script_b64 = base64.b64encode(SANDBOX_LIVE_SESSION_SCAN_SCRIPT.encode("utf-8")).decode(
        "ascii"
    )
    command = [
        args.nemoclaw_bin,
        "sandbox",
        "exec",
        args.nemoclaw_sandbox,
        "--workdir",
        "/sandbox",
        "--no-tty",
        "--timeout",
        str(SANDBOX_LIVE_SESSION_SCAN_TIMEOUT),
        "--",
        "env",
        f"OPENCLAW_SCAN_SCRIPT_B64={scan_script_b64}",
        "bash",
        "-lc",
        'OPENCLAW_SCAN_SCRIPT="$(printf %s "$OPENCLAW_SCAN_SCRIPT_B64" | base64 -d)"; exec python3 -c "$OPENCLAW_SCAN_SCRIPT" "$@"',
        "openclaw-session-scan",
        str(started_at - 5.0),
        *sandbox_dirs,
    ]
    result = subprocess.run(
        command,
        text=True,
        capture_output=True,
        check=False,
        env=env,
    )
    scan = {
        "enabled": True,
        "ok": result.returncode == 0,
        "sandbox": getattr(args, "nemoclaw_sandbox", None),
        "session_dirs": sandbox_dirs,
        "command_returncode": result.returncode,
        "stderr_tail": result.stderr[-1000:] if result.stderr else "",
        "sessions": [],
    }
    if result.returncode != 0:
        scan["reason"] = "sandbox_scan_failed"
        return scan
    parsed = parse_last_json_line(result.stdout)
    if not isinstance(parsed, dict) or parsed.get("ok") is not True:
        scan["ok"] = False
        scan["reason"] = "sandbox_scan_invalid_json"
        scan["stdout_tail"] = result.stdout[-1000:] if result.stdout else ""
        return scan
    sessions = parsed.get("sessions")
    if isinstance(sessions, list):
        scan["sessions"] = [session for session in sessions if isinstance(session, dict)]
    return scan


def live_tool_budget_status(
    args: argparse.Namespace,
    started_at: float,
    env: dict[str, str] | None = None,
) -> dict[str, Any]:
    now = time.time()
    max_input_tokens = int(getattr(args, "max_input_tokens", 0) or 0)
    max_tool_calls = int(getattr(args, "max_tool_calls", 0) or 0)
    max_agent_turns = int(getattr(args, "max_agent_turns", 0) or 0)
    max_tool_wall_seconds = int(getattr(args, "max_tool_wall_seconds", 0) or 0)
    llm_response_idle_timeout_seconds = max(
        0.0,
        float(getattr(args, "llm_response_idle_timeout_seconds", 0.0) or 0.0),
    )
    session_dirs = [str(path) for path in configured_live_session_dirs(args)]
    sandbox_session_dirs = configured_live_sandbox_session_dirs(args)
    budget_enabled = (
        max_tool_calls > 0
        or max_input_tokens > 0
        or max_agent_turns > 0
        or llm_response_idle_timeout_seconds > 0
    )
    observations: list[dict[str, Any]] = []
    for path in live_session_candidates(args, started_at):
        observations.append(session_budget_observation(path))
    sandbox_scan = scan_nemoclaw_live_sessions(args, started_at, env)
    if sandbox_scan.get("ok") is True:
        for session in sandbox_scan.get("sessions", []):
            count = session.get("tool_call_count")
            executed_count = session.get("executed_tool_call_count")
            blocked_count = session.get("blocked_tool_call_count")
            estimated_input_tokens = session.get("estimated_input_tokens")
            agent_turn_count = session.get("agent_turn_count")
            path = session.get("path")
            live_policy_count = session.get("live_tool_policy_violation_count")
            live_policy_violations = session.get("live_tool_policy_violations")
            budget_guard_block_count = session.get("budget_guard_block_count")
            budget_guard_blocks = session.get("budget_guard_blocks")
            live_provider_timeout_count = session.get("live_provider_timeout_count")
            live_provider_timeouts = session.get("live_provider_timeouts")
            final_assistant_idle_done = session.get("final_assistant_idle_done")
            final_assistant_idle_seconds = session.get("idle_seconds")
            session_lock_exists = session.get("lock_exists")
            session_idle_seconds = session.get("idle_seconds")
            last_message_role = session.get("last_message_role")
            if isinstance(count, (int, float)) and isinstance(path, str) and path:
                blocked_count_int = (
                    int(blocked_count)
                    if isinstance(blocked_count, (int, float))
                    else (
                        int(budget_guard_block_count)
                        if isinstance(budget_guard_block_count, (int, float))
                        else 0
                    )
                )
                executed_count_int = (
                    int(executed_count)
                    if isinstance(executed_count, (int, float))
                    else max(0, int(count) - blocked_count_int)
                )
                observations.append(
                    {
                        "source": "nemoclaw_sandbox",
                        "tool_call_count": int(count),
                        "blocked_tool_call_count": blocked_count_int,
                        "executed_tool_call_count": executed_count_int,
                        "estimated_input_tokens": (
                            int(estimated_input_tokens)
                            if isinstance(estimated_input_tokens, (int, float))
                            else None
                        ),
                        "agent_turn_count": (
                            int(agent_turn_count)
                            if isinstance(agent_turn_count, (int, float))
                            else None
                        ),
                        "session_file": path,
                        "live_tool_policy_violation_count": (
                            int(live_policy_count)
                            if isinstance(live_policy_count, (int, float))
                            else 0
                        ),
                        "live_tool_policy_violations": (
                            live_policy_violations
                            if isinstance(live_policy_violations, list)
                            else []
                        ),
                        "budget_guard_block_count": (
                            int(budget_guard_block_count)
                            if isinstance(budget_guard_block_count, (int, float))
                            else 0
                        ),
                        "budget_guard_blocks": (
                            budget_guard_blocks
                            if isinstance(budget_guard_blocks, list)
                            else []
                        ),
                        "live_provider_timeout_count": (
                            int(live_provider_timeout_count)
                            if isinstance(live_provider_timeout_count, (int, float))
                            else 0
                        ),
                        "live_provider_timeouts": (
                            live_provider_timeouts
                            if isinstance(live_provider_timeouts, list)
                            else []
                        ),
                        "final_assistant_idle_done": bool(final_assistant_idle_done),
                        "final_assistant_idle_seconds": (
                            float(final_assistant_idle_seconds)
                            if isinstance(final_assistant_idle_seconds, (int, float))
                            else None
                        ),
                        "session_lock_exists": bool(session_lock_exists),
                        "session_idle_seconds": (
                            float(session_idle_seconds)
                            if isinstance(session_idle_seconds, (int, float))
                            else None
                        ),
                        "last_message_role": (
                            str(last_message_role) if isinstance(last_message_role, str) else None
                        ),
                    }
                )
    tool_observations = sorted(
        observations,
        key=lambda item: int(item.get("executed_tool_call_count") or item.get("tool_call_count") or 0),
        reverse=True,
    )
    input_observations = sorted(
        observations,
        key=lambda item: int(item.get("estimated_input_tokens") or 0),
        reverse=True,
    )
    turn_observations = sorted(
        observations,
        key=lambda item: int(item.get("agent_turn_count") or 0),
        reverse=True,
    )
    best_tool = tool_observations[0] if tool_observations else {}
    best_input = input_observations[0] if input_observations else {}
    best_turn = turn_observations[0] if turn_observations else {}
    policy_observations = sorted(
        observations,
        key=lambda item: int(item.get("live_tool_policy_violation_count") or 0),
        reverse=True,
    )
    best_policy = policy_observations[0] if policy_observations else {}
    budget_guard_observations = sorted(
        observations,
        key=lambda item: int(item.get("budget_guard_block_count") or 0),
        reverse=True,
    )
    best_budget_guard = budget_guard_observations[0] if budget_guard_observations else {}
    provider_timeout_observations = sorted(
        observations,
        key=lambda item: int(item.get("live_provider_timeout_count") or 0),
        reverse=True,
    )
    best_provider_timeout = provider_timeout_observations[0] if provider_timeout_observations else {}
    final_assistant_observations = [
        item for item in observations if item.get("final_assistant_idle_done") is True
    ]
    final_assistant_observations = sorted(
        final_assistant_observations,
        key=lambda item: float(item.get("final_assistant_idle_seconds") or 0.0),
        reverse=True,
    )
    best_final_assistant = (
        final_assistant_observations[0] if final_assistant_observations else {}
    )
    response_idle_observations = [
        item
        for item in observations
        if item.get("last_message_role") in {"user", "tool", "toolResult"}
        and isinstance(item.get("session_idle_seconds"), (int, float))
    ]
    response_idle_observations = sorted(
        response_idle_observations,
        key=lambda item: float(item.get("session_idle_seconds") or 0.0),
        reverse=True,
    )
    best_response_idle = response_idle_observations[0] if response_idle_observations else {}
    no_session_response_idle_exceeded = bool(
        llm_response_idle_timeout_seconds > 0
        and not observations
        and now - started_at >= llm_response_idle_timeout_seconds
    )
    best_budget_guard_blocks = (
        best_budget_guard.get("budget_guard_blocks", [])
        if best_budget_guard
        else []
    )
    if not isinstance(best_budget_guard_blocks, list):
        best_budget_guard_blocks = []
    budget_guard_policy_blocks = [
        block
        for block in best_budget_guard_blocks
        if isinstance(block, dict)
        and (
            block.get("kind") == "tool_policy_violation"
            or "tool_policy_violation" in str(block.get("text") or "")
        )
    ]
    non_policy_budget_guard_blocks = [
        block
        for block in best_budget_guard_blocks
        if isinstance(block, dict) and block not in budget_guard_policy_blocks
    ]
    best_count = int(best_tool.get("tool_call_count") or 0) if best_tool else None
    best_executed_count = (
        int(best_tool.get("executed_tool_call_count") or 0) if best_tool else None
    )
    best_blocked_count = (
        int(best_tool.get("blocked_tool_call_count") or 0) if best_tool else None
    )
    best_input_tokens = (
        int(best_input.get("estimated_input_tokens") or 0) if best_input else None
    )
    best_turn_count = (
        int(best_turn.get("agent_turn_count") or 0) if best_turn else None
    )
    tool_exceeded = bool(
        max_tool_calls > 0
        and best_tool
        and best_executed_count is not None
        and best_executed_count > max_tool_calls
    )
    input_exceeded = bool(
        max_input_tokens > 0
        and best_input
        and best_input_tokens is not None
        and best_input_tokens > max_input_tokens
    )
    turn_exceeded = bool(
        max_agent_turns > 0
        and best_turn
        and best_turn_count is not None
        and best_turn_count > max_agent_turns
    )
    turn_limit_reached = bool(
        max_agent_turns > 0
        and best_turn
        and best_turn_count is not None
        and best_turn_count >= max_agent_turns
    )
    policy_violation_count = (
        int(best_policy.get("live_tool_policy_violation_count") or 0) if best_policy else 0
    )
    policy_observed = policy_violation_count > 0 or bool(budget_guard_policy_blocks)
    budget_guard_block_count = (
        int(best_budget_guard.get("budget_guard_block_count") or 0)
        if best_budget_guard
        else 0
    )
    provider_timeout_count = (
        int(best_provider_timeout.get("live_provider_timeout_count") or 0)
        if best_provider_timeout
        else 0
    )
    provider_timeouts = (
        best_provider_timeout.get("live_provider_timeouts", [])
        if best_provider_timeout
        else []
    )
    exceeded_limits = []
    if input_exceeded:
        exceeded_limits.append("max_input_tokens_exceeded")
    if tool_exceeded:
        exceeded_limits.append("max_tool_calls_exceeded")
    if turn_exceeded:
        exceeded_limits.append("max_agent_turns_exceeded")
    budget_guard_exceeded = bool(non_policy_budget_guard_blocks)
    if budget_guard_exceeded:
        exceeded_limits.append("budget_guard_blocked")
    provider_timeout_exceeded = provider_timeout_count > 0
    if provider_timeout_exceeded:
        exceeded_limits.append("live_provider_timeout")
    response_idle_exceeded = bool(
        no_session_response_idle_exceeded
        or (
            llm_response_idle_timeout_seconds > 0
            and best_response_idle
            and float(best_response_idle.get("session_idle_seconds") or 0.0)
            >= llm_response_idle_timeout_seconds
        )
    )
    if response_idle_exceeded:
        exceeded_limits.append("llm_response_idle_timeout")
    interrupt_limits = list(exceeded_limits)
    if turn_limit_reached and "max_agent_turns_exceeded" not in interrupt_limits:
        interrupt_limits.append("max_agent_turns_reached")
    reason = None
    if len(exceeded_limits) == 1:
        reason = exceeded_limits[0]
    elif len(exceeded_limits) > 1:
        reason = "runtime_budget_exceeded"
    interrupt_reason = None
    if len(interrupt_limits) == 1:
        interrupt_reason = interrupt_limits[0]
    elif len(interrupt_limits) > 1:
        interrupt_reason = "runtime_budget_interrupted"
    return {
        "enabled": budget_enabled,
        "max_input_tokens": max_input_tokens or None,
        "max_tool_calls": max_tool_calls or None,
        "max_agent_turns": max_agent_turns or None,
        "llm_response_idle_timeout_seconds": llm_response_idle_timeout_seconds or None,
        "estimated_input_tokens": best_input_tokens,
        "tool_call_count": best_count,
        "executed_tool_call_count": best_executed_count,
        "blocked_tool_call_count": best_blocked_count,
        "agent_turn_count": best_turn_count,
        "session_file": best_tool.get("session_file") if best_tool else None,
        "session_source": best_tool.get("source") if best_tool else None,
        "input_session_file": best_input.get("session_file") if best_input else None,
        "input_session_source": best_input.get("source") if best_input else None,
        "turn_session_file": best_turn.get("session_file") if best_turn else None,
        "turn_session_source": best_turn.get("source") if best_turn else None,
        "policy_session_file": best_policy.get("session_file") if best_policy else None,
        "policy_session_source": best_policy.get("source") if best_policy else None,
        "budget_guard_session_file": best_budget_guard.get("session_file") if best_budget_guard else None,
        "budget_guard_session_source": best_budget_guard.get("source") if best_budget_guard else None,
        "live_tool_policy_violation_count": policy_violation_count,
        "live_tool_policy_violations": best_policy.get("live_tool_policy_violations", [])
        if best_policy
        else [],
        "live_tool_policy_observed": policy_observed,
        "budget_guard_policy_block_count": len(budget_guard_policy_blocks),
        "budget_guard_policy_blocks": budget_guard_policy_blocks[:10],
        "budget_guard_block_count": budget_guard_block_count,
        "budget_guard_blocks": best_budget_guard_blocks,
        "non_policy_budget_guard_block_count": len(non_policy_budget_guard_blocks),
        "non_policy_budget_guard_blocks": non_policy_budget_guard_blocks[:10],
        "live_provider_timeout_count": provider_timeout_count,
        "live_provider_timeouts": provider_timeouts if isinstance(provider_timeouts, list) else [],
        "session_dirs": session_dirs,
        "sandbox_session_dirs": sandbox_session_dirs,
        "sandbox_scan": sandbox_scan,
        "final_assistant_idle_done": bool(best_final_assistant),
        "final_assistant_idle_seconds": (
            float(best_final_assistant.get("final_assistant_idle_seconds") or 0.0)
            if best_final_assistant
            else None
        ),
        "final_assistant_session_file": (
            best_final_assistant.get("session_file") if best_final_assistant else None
        ),
        "final_assistant_session_source": (
            best_final_assistant.get("source") if best_final_assistant else None
        ),
        "final_assistant_session_lock_exists": (
            bool(best_final_assistant.get("session_lock_exists")) if best_final_assistant else None
        ),
        "llm_response_idle_exceeded": response_idle_exceeded,
        "llm_response_idle_seconds": (
            float(now - started_at)
            if no_session_response_idle_exceeded
            else
            float(best_response_idle.get("session_idle_seconds") or 0.0)
            if best_response_idle
            else None
        ),
        "llm_response_idle_session_file": (
            best_response_idle.get("session_file") if best_response_idle else None
        ),
        "llm_response_idle_session_source": (
            "no_session_observed"
            if no_session_response_idle_exceeded
            else
            best_response_idle.get("source") if best_response_idle else None
        ),
        "llm_response_idle_last_message_role": (
            best_response_idle.get("last_message_role") if best_response_idle else None
        ),
        "llm_response_idle_session_lock_exists": (
            bool(best_response_idle.get("session_lock_exists")) if best_response_idle else None
        ),
        "exceeded": bool(exceeded_limits),
        "exceeded_limits": exceeded_limits,
        "reason": reason,
        "turn_limit_reached": turn_limit_reached,
        "interrupt": bool(interrupt_limits),
        "interrupt_limits": interrupt_limits,
        "interrupt_reason": interrupt_reason,
    }


def terminate_process(process: subprocess.Popen[str], grace_seconds: float = 10.0) -> tuple[str, str]:
    try:
        os.killpg(process.pid, signal.SIGTERM)
    except ProcessLookupError:
        pass
    except OSError:
        process.terminate()
    try:
        return process.communicate(timeout=grace_seconds)
    except subprocess.TimeoutExpired:
        try:
            os.killpg(process.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        except OSError:
            process.kill()
        return process.communicate()


def gracefully_complete_process(
    process: subprocess.Popen[str],
    interrupt_grace_seconds: float = 30.0,
) -> tuple[str, str, dict[str, Any]]:
    """Let OpenClaw run finalizers and telemetry flush before forced cleanup."""
    signal_method = "process_group_sigint"
    try:
        os.killpg(process.pid, signal.SIGINT)
    except ProcessLookupError:
        signal_method = "process_already_exited"
    except OSError:
        signal_method = "process_sigint"
        process.send_signal(signal.SIGINT)
    try:
        stdout, stderr = process.communicate(timeout=interrupt_grace_seconds)
        return stdout, stderr, {
            "mode": "graceful_sigint",
            "signal_method": signal_method,
            "interrupt_grace_seconds": interrupt_grace_seconds,
            "forced_after_sigint": False,
        }
    except subprocess.TimeoutExpired:
        stdout, stderr = terminate_process(process)
        return stdout, stderr, {
            "mode": "sigint_then_forced_termination",
            "signal_method": signal_method,
            "interrupt_grace_seconds": interrupt_grace_seconds,
            "forced_after_sigint": True,
        }


def run_openclaw_command_with_live_budget(
    command: list[str],
    args: argparse.Namespace,
    env: dict[str, str],
    started_at: float,
) -> tuple[subprocess.CompletedProcess[str], dict[str, Any]]:
    process = subprocess.Popen(
        command,
        cwd=args.cwd,
        env=env,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        start_new_session=True,
    )
    last_status: dict[str, Any] = live_tool_budget_status(args, started_at, env)
    poll_seconds = (
        SANDBOX_LIVE_SESSION_POLL_SECONDS
        if configured_live_sandbox_session_dirs(args)
        else 1.0
    )
    final_idle_salvage_seconds = max(
        0.0,
        float(getattr(args, "final_assistant_idle_salvage_seconds", 0.0) or 0.0),
    )
    final_idle_shutdown_grace_seconds = max(
        0.0,
        float(
            getattr(args, "final_assistant_shutdown_grace_seconds", 30.0)
            or 0.0
        ),
    )
    while True:
        try:
            stdout, stderr = process.communicate(timeout=poll_seconds)
            final_status = live_tool_budget_status(args, started_at, env)
            return (
                subprocess.CompletedProcess(command, process.returncode or 0, stdout=stdout, stderr=stderr),
                {
                    **final_status,
                    "interrupted": False,
                },
            )
        except subprocess.TimeoutExpired:
            last_status = live_tool_budget_status(args, started_at, env)
            final_idle_seconds = last_status.get("final_assistant_idle_seconds")
            if (
                final_idle_salvage_seconds > 0
                and last_status.get("final_assistant_idle_done") is True
                and isinstance(final_idle_seconds, (int, float))
                and float(final_idle_seconds) >= final_idle_salvage_seconds
            ):
                stdout, stderr, shutdown = gracefully_complete_process(
                    process,
                    interrupt_grace_seconds=final_idle_shutdown_grace_seconds,
                )
                reason = (
                    "Live OpenClaw final assistant idle salvage: "
                    f"session_file={last_status.get('final_assistant_session_file')} "
                    f"idle_seconds={float(final_idle_seconds):.1f} "
                    f"threshold_seconds={final_idle_salvage_seconds:.1f}"
                )
                stderr = (stderr or "") + "\n" + reason
                return (
                    subprocess.CompletedProcess(command, 0, stdout=stdout, stderr=stderr),
                    {
                        **last_status,
                        "interrupted": False,
                        "final_assistant_idle_salvaged": True,
                        "final_assistant_idle_salvage_seconds": final_idle_salvage_seconds,
                        "final_assistant_shutdown": shutdown,
                    },
                )
            if not last_status.get("interrupt"):
                continue
            # Budget and idle-timeout exits still need OpenClaw's shutdown hooks.
            # In particular, weave-openclaw closes and flushes the parent
            # agent/message spans from those hooks.  A direct SIGTERM can leave
            # only already-completed tool spans in W&B.
            stdout, stderr, shutdown = gracefully_complete_process(
                process,
                interrupt_grace_seconds=final_idle_shutdown_grace_seconds,
            )
            reason_name = (
                last_status.get("interrupt_reason")
                or last_status.get("reason")
                or "runtime_budget_interrupted"
            )
            reason = (
                "Live OpenClaw interrupt: "
                f"estimated_input_tokens={last_status.get('estimated_input_tokens')} "
                f"max_input_tokens={last_status.get('max_input_tokens')} "
                f"tool_call_count={last_status.get('tool_call_count')} "
                f"executed_tool_call_count={last_status.get('executed_tool_call_count')} "
                f"blocked_tool_call_count={last_status.get('blocked_tool_call_count')} "
                f"max_tool_calls={last_status.get('max_tool_calls')} "
                f"agent_turn_count={last_status.get('agent_turn_count')} "
                f"max_agent_turns={last_status.get('max_agent_turns')} "
                f"provider_timeout_count={last_status.get('live_provider_timeout_count')} "
                f"provider_timeouts={json.dumps(last_status.get('live_provider_timeouts') or [], ensure_ascii=False)[:2000]} "
                f"llm_response_idle_seconds={last_status.get('llm_response_idle_seconds')} "
                f"llm_response_idle_timeout_seconds={last_status.get('llm_response_idle_timeout_seconds')} "
                f"llm_response_idle_session_file={last_status.get('llm_response_idle_session_file')}"
            )
            stderr = (stderr or "") + "\n" + reason
            return (
                subprocess.CompletedProcess(command, process.returncode or 125, stdout=stdout, stderr=stderr),
                {
                    **last_status,
                    "interrupted": True,
                    "reason": reason_name,
                    "interrupt_shutdown": shutdown,
                },
            )


def run_agent(args: argparse.Namespace) -> None:
    if args.weave_sidecar or args.weave_sidecar_strict:
        raise SystemExit(
            "Weave sidecar logging is disabled for Taiwan agentic benchmarks. "
            "Use the native weave-openclaw Agents integration; manual sidecar/relog "
            "traces are not valid evidence."
        )
    args.openclaw_config_path = effective_openclaw_config_path(args)
    status = preflight(
        args.openclaw_bin,
        args.env_file,
        nemoclaw_bin=getattr(args, "nemoclaw_bin", "nemoclaw"),
        nemoclaw_sandbox=getattr(args, "nemoclaw_sandbox", None),
    )
    if not status["ok"] and not args.allow_failed_preflight:
        print(json.dumps(status, ensure_ascii=False, indent=2), file=sys.stderr)
        raise SystemExit("OpenClaw preflight failed")

    prompt_text = read_text(args.prompt_file)
    header, metadata = metadata_header(args, prompt_text)
    policy = load_tool_policy(args)
    message_text = header + prompt_text

    output_dir = args.output_dir / args.benchmark_id / args.task_id
    output_dir.mkdir(parents=True, exist_ok=True)
    message_file = output_dir / "message.md"
    message_file.write_text(message_text, encoding="utf-8")

    env, _, _ = prepare_env(args.env_file, args.openclaw_bin)
    env_for_session_copy: dict[str, str] | None = env
    if args.openclaw_config_path and not getattr(args, "nemoclaw_sandbox", None):
        env["OPENCLAW_CONFIG_PATH"] = str(args.openclaw_config_path.resolve())

    openclaw_bin = None if getattr(args, "nemoclaw_sandbox", None) else status.get("openclaw_path")
    sandbox_message_path = None
    if getattr(args, "nemoclaw_sandbox", None) and not args.dry_run:
        sandbox_message_path = write_nemoclaw_message_file(args, message_text, env)
    command = build_openclaw_command_with_message_source(
        args,
        message_text,
        openclaw_bin,
        sandbox_message_path=sandbox_message_path,
    )
    sidecar = {
        "metadata": metadata,
        "preflight": status,
        "command": command,
        "cwd": str(args.cwd.resolve()),
        "openclaw_config_path": str(args.openclaw_config_path.resolve()) if args.openclaw_config_path else None,
        "sandbox_message_path": sandbox_message_path,
        "tool_policy": policy,
        "started_at": time.time(),
        "dry_run": args.dry_run,
    }

    live_runtime_budget: dict[str, Any] = {}
    if args.dry_run:
        result = subprocess.CompletedProcess(command, 0, stdout="", stderr="")
    else:
        result, live_runtime_budget = run_openclaw_command_with_live_budget(
            command,
            args,
            env,
            sidecar["started_at"],
        )

    sidecar.update(
        {
            "ended_at": time.time(),
            "returncode": result.returncode,
            "stdout": result.stdout,
            "stderr": result.stderr,
        }
    )
    if live_runtime_budget:
        sidecar["live_runtime_budget"] = live_runtime_budget
        if live_runtime_budget.get("session_file"):
            sidecar["live_session_file"] = live_runtime_budget.get("session_file")
    sidecar["stdout_json"] = parse_last_json_line(result.stdout) if result.stdout.strip() else None
    sidecar["gateway_transport"] = gateway_transport_status(sidecar, args)

    if getattr(args, "nemoclaw_sandbox", None) and not args.dry_run and env_for_session_copy is not None:
        sidecar["nemoclaw_session_copy"] = copy_nemoclaw_session_file(
            sidecar,
            args,
            output_dir,
            env_for_session_copy,
        )

    enrich_sidecar_with_tool_events(sidecar)
    sidecar["runtime_budget"] = runtime_budget_status(sidecar, args)
    sidecar["nemoclaw_session_audit"] = nemoclaw_session_audit_status(sidecar, args)
    sidecar["model_completion"] = model_completion_status(sidecar)
    violations = tool_policy_violations(sidecar.get("tool_events", []), policy)
    sidecar["tool_policy_violations"] = violations
    sidecar["tool_policy_ok"] = not violations

    sidecar_path = output_dir / "openclaw_result.json"
    sidecar_path.write_text(json.dumps(sidecar, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    if should_log_weave_sidecar(args, sidecar):
        weave_result = log_weave_sidecar(args, sidecar_path, sidecar, message_text)
        sidecar["weave_sidecar"] = weave_result
        sidecar_path.write_text(json.dumps(sidecar, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        if args.weave_sidecar_strict and not weave_result.get("ok"):
            raise SystemExit("Weave sidecar logging failed")
    else:
        sidecar["weave_sidecar"] = {
            "ok": None,
            "skipped": True,
            "reason": "native weave-openclaw plugin is the authoritative Agents trace path",
        }
        sidecar_path.write_text(json.dumps(sidecar, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    if violations:
        print(json.dumps({"tool_policy_violations": violations}, ensure_ascii=False, indent=2), file=sys.stderr)
    gateway_transport = sidecar.get("gateway_transport")
    if isinstance(gateway_transport, dict) and gateway_transport.get("ok") is False:
        print(json.dumps({"gateway_transport": gateway_transport}, ensure_ascii=False, indent=2), file=sys.stderr)
        raise SystemExit("OpenClaw Gateway transport violation")
    runtime_budget = sidecar.get("runtime_budget")
    if isinstance(runtime_budget, dict) and runtime_budget.get("violations"):
        print(json.dumps({"runtime_budget": runtime_budget}, ensure_ascii=False, indent=2), file=sys.stderr)
        raise SystemExit("Runtime budget exceeded")
    conversation_order = sidecar.get("conversation_order")
    if isinstance(conversation_order, dict) and conversation_order.get("ok") is False:
        print(
            json.dumps({"conversation_order": conversation_order}, ensure_ascii=False, indent=2),
            file=sys.stderr,
        )
        raise SystemExit("Conversation order violation")
    nemoclaw_session_audit = sidecar.get("nemoclaw_session_audit")
    if isinstance(nemoclaw_session_audit, dict) and nemoclaw_session_audit.get("ok") is False:
        print(
            json.dumps({"nemoclaw_session_audit": nemoclaw_session_audit}, ensure_ascii=False, indent=2),
            file=sys.stderr,
        )
        raise SystemExit("NeMoClaw session audit failed")
    model_completion = sidecar.get("model_completion")
    if isinstance(model_completion, dict) and model_completion.get("ok") is False:
        print(
            json.dumps({"model_completion": model_completion}, ensure_ascii=False, indent=2),
            file=sys.stderr,
        )
        raise SystemExit(str(model_completion.get("reason") or "Model response incomplete"))
    print(sidecar_path)
    if result.returncode != 0:
        raise SystemExit(result.returncode)


def should_log_weave_sidecar(args: argparse.Namespace, sidecar: dict[str, Any]) -> bool:
    if args.dry_run or not args.weave_sidecar:
        return False
    if not WEAVE_SIDECAR_SCRIPT.exists():
        return False
    if sidecar.get("stdout_json") is None and not sidecar.get("stdout"):
        return bool(args.weave_sidecar_strict)
    return True


def extract_openclaw_meta(sidecar: dict[str, Any]) -> dict[str, Any]:
    stdout_json = sidecar.get("stdout_json")
    if isinstance(stdout_json, dict):
        meta = stdout_json.get("meta")
        if isinstance(meta, dict):
            return meta
        result = stdout_json.get("result")
        if isinstance(result, dict):
            merged = dict(result)
            result_meta = result.get("meta")
            if isinstance(result_meta, dict):
                merged.update(result_meta)
            return merged
    return {}


def gateway_transport_status(sidecar: dict[str, Any], args: argparse.Namespace) -> dict[str, Any]:
    gateway_required = not bool(getattr(args, "local", True))
    meta = extract_openclaw_meta(sidecar)
    transport = meta.get("transport")
    fallback_from = meta.get("fallbackFrom")
    stderr = str(sidecar.get("stderr") or "")
    fallback_marker = "EMBEDDED FALLBACK" in stderr
    pairing_marker = "pairing required" in stderr.lower()
    violation = gateway_required and (
        transport == "embedded"
        or fallback_from == "gateway"
        or fallback_marker
    )
    reasons: list[str] = []
    if transport == "embedded":
        reasons.append("embedded_transport")
    if fallback_from == "gateway":
        reasons.append("fallback_from_gateway")
    if fallback_marker:
        reasons.append("embedded_fallback_stderr")
    if pairing_marker:
        reasons.append("pairing_required_stderr")
    return {
        "ok": not violation,
        "gateway_required": gateway_required,
        "transport": transport,
        "fallbackFrom": fallback_from,
        "stderr_embedded_fallback": fallback_marker,
        "stderr_pairing_required": pairing_marker,
        "reasons": reasons,
    }


def extract_agent_meta(sidecar: dict[str, Any]) -> dict[str, Any]:
    meta = extract_openclaw_meta(sidecar)
    agent_meta = meta.get("agentMeta")
    copied_session_file = sidecar.get("copied_session_file")
    sandbox_session_file = sidecar.get("sandbox_session_file")
    live_session_file = sidecar.get("live_session_file")
    if isinstance(agent_meta, dict):
        merged = dict(agent_meta)
        if isinstance(copied_session_file, str) and copied_session_file:
            merged["sandboxSessionFile"] = merged.get("sessionFile")
            merged["sessionFile"] = copied_session_file
        return merged
    if isinstance(copied_session_file, str) and copied_session_file:
        merged = {"sessionFile": copied_session_file}
        if isinstance(sandbox_session_file, str) and sandbox_session_file:
            merged["sandboxSessionFile"] = sandbox_session_file
        elif isinstance(live_session_file, str) and live_session_file:
            merged["sandboxSessionFile"] = live_session_file
        return merged
    if isinstance(live_session_file, str) and live_session_file:
        return {"sessionFile": live_session_file}
    return {}


def live_nemoclaw_session_file(sidecar: dict[str, Any]) -> str | None:
    live_budget = sidecar.get("live_runtime_budget")
    if not isinstance(live_budget, dict):
        return None
    if live_budget.get("session_source") != "nemoclaw_sandbox":
        return None
    session_file = live_budget.get("session_file")
    if not isinstance(session_file, str) or not session_file.startswith("/"):
        return None
    return session_file


def copy_nemoclaw_session_file(
    sidecar: dict[str, Any],
    args: argparse.Namespace,
    output_dir: Path,
    env: dict[str, str],
) -> dict[str, Any]:
    """Copy the sandbox-local OpenClaw session JSONL to the host audit dir."""
    if not getattr(args, "nemoclaw_sandbox", None):
        return {"attempted": False, "ok": None, "reason": "not_nemoclaw"}
    raw_meta = extract_openclaw_meta(sidecar)
    agent_meta = raw_meta.get("agentMeta") if isinstance(raw_meta.get("agentMeta"), dict) else {}
    live_budget = sidecar.get("live_runtime_budget")
    final_session = (
        live_budget.get("final_assistant_session_file")
        if isinstance(live_budget, dict)
        and live_budget.get("final_assistant_idle_salvaged") is True
        and live_budget.get("final_assistant_session_source") == "nemoclaw_sandbox"
        else None
    )
    if isinstance(final_session, str) and final_session.startswith("/"):
        # A delegated subagent can remain active after the parent has emitted its
        # final answer. The parent session remains the auditable benchmark record.
        session_file = final_session
        session_source = "final_assistant_idle_salvage"
    else:
        session_file = agent_meta.get("sessionFile") if isinstance(agent_meta, dict) else None
        session_source = "stdout_agent_meta"
    if not isinstance(session_file, str) or not session_file.startswith("/"):
        live_session = live_nemoclaw_session_file(sidecar)
        if isinstance(live_session, str):
            session_file = live_session
            session_source = "live_runtime_budget"
    if not isinstance(session_file, str) or not session_file.startswith("/"):
        return {"attempted": False, "ok": None, "reason": "missing_sandbox_session_file"}
    if "\n" in session_file or "\r" in session_file:
        return {
            "attempted": True,
            "ok": False,
            "reason": "invalid_session_file_path",
            "source": session_source,
        }

    copied_path = output_dir / "nemoclaw_session.jsonl"
    command = [
        args.nemoclaw_bin,
        "sandbox",
        "exec",
        args.nemoclaw_sandbox,
        "--workdir",
        "/sandbox",
        "--no-tty",
        "--timeout",
        "60",
        "--",
        "cat",
        session_file,
    ]
    result = subprocess.run(command, text=True, capture_output=True, check=False, env=env)
    if result.returncode != 0:
        return {
            "attempted": True,
            "ok": False,
            "reason": "copy_failed",
            "sandbox_session_file": session_file,
            "source": session_source,
            "returncode": result.returncode,
            "stderr": result.stderr,
        }
    copied_path.write_text(result.stdout, encoding="utf-8")
    sidecar["copied_session_file"] = str(copied_path)
    sidecar["sandbox_session_file"] = session_file
    return {
        "attempted": True,
        "ok": True,
        "sandbox_session_file": session_file,
        "copied_session_file": str(copied_path),
        "source": session_source,
        "bytes": copied_path.stat().st_size,
    }


def nemoclaw_session_audit_status(sidecar: dict[str, Any], args: argparse.Namespace) -> dict[str, Any]:
    if not getattr(args, "nemoclaw_sandbox", None):
        return {"required": False, "ok": None, "reason": "not_nemoclaw"}
    if getattr(args, "dry_run", False):
        return {"required": False, "ok": None, "reason": "dry_run"}

    errors: list[str] = []
    copy_status = sidecar.get("nemoclaw_session_copy")
    if not isinstance(copy_status, dict):
        errors.append("missing_nemoclaw_session_copy_status")
        copy_status = {}
    elif copy_status.get("ok") is not True:
        errors.append(f"session_copy_{copy_status.get('reason') or 'failed'}")

    copied_session_file = sidecar.get("copied_session_file")
    copied_session_path: Path | None = None
    copied_session_bytes: int | None = None
    if not isinstance(copied_session_file, str) or not copied_session_file:
        errors.append("missing_copied_session_file")
    else:
        copied_session_path = Path(copied_session_file).expanduser()
        if not copied_session_path.exists():
            errors.append("copied_session_file_missing_on_host")
        else:
            copied_session_bytes = copied_session_path.stat().st_size
            if copied_session_bytes <= 0:
                errors.append("copied_session_file_empty")

    conversation_order = sidecar.get("conversation_order")
    if not isinstance(conversation_order, dict):
        errors.append("missing_conversation_order")
        conversation_order = {}
    elif conversation_order.get("checked") is not True:
        errors.append("conversation_order_not_checked")
    elif conversation_order.get("ok") is not True:
        errors.append("conversation_order_not_ok")

    event_count = sidecar.get("timeline_event_count")
    if not isinstance(event_count, int) or event_count <= 0:
        errors.append("missing_timeline_events")

    user_count = conversation_order.get("user_message_count")
    assistant_count = conversation_order.get("assistant_message_count")
    tool_call_count = conversation_order.get("tool_call_count")
    if not isinstance(user_count, int) or user_count <= 0:
        errors.append("missing_user_message")
    if not (
        (isinstance(assistant_count, int) and assistant_count > 0)
        or (isinstance(tool_call_count, int) and tool_call_count > 0)
    ):
        errors.append("missing_assistant_or_tool_activity")

    return {
        "required": True,
        "ok": not errors,
        "sandbox": getattr(args, "nemoclaw_sandbox", None),
        "copy": copy_status,
        "copied_session_file": copied_session_file if isinstance(copied_session_file, str) else None,
        "copied_session_bytes": copied_session_bytes,
        "timeline_event_count": event_count,
        "conversation_order_checked": conversation_order.get("checked"),
        "conversation_order_ok": conversation_order.get("ok"),
        "user_message_count": user_count,
        "assistant_message_count": assistant_count,
        "tool_call_count": tool_call_count,
        "errors": errors,
    }


def model_completion_status(sidecar: dict[str, Any]) -> dict[str, Any]:
    """Classify model-side incomplete output separately from harness failures."""
    output_text = "\n".join(
        str(sidecar.get(key) or "") for key in ("stdout", "stderr")
    ).lower()
    warning_marker = next(
        (marker for marker in OPENCLAW_NO_RESPONSE_MARKERS if marker in output_text),
        None,
    )

    meta = extract_openclaw_meta(sidecar)
    final_stop_reason = meta.get("stopReason") if isinstance(meta, dict) else None
    final_assistant_message_seen = False
    session_file = extract_agent_meta(sidecar).get("sessionFile")
    if isinstance(session_file, str) and session_file:
        session_path = Path(session_file).expanduser()
        if session_path.exists():
            for raw in session_path.read_text(encoding="utf-8", errors="replace").splitlines():
                try:
                    event = json.loads(raw)
                except json.JSONDecodeError:
                    continue
                message = event.get("message") if isinstance(event, dict) else None
                if not isinstance(message, dict) or message.get("role") != "assistant":
                    continue
                final_assistant_message_seen = True
                observed_reason = message.get("stopReason") or message.get("stop_reason")
                if isinstance(observed_reason, str) and observed_reason:
                    final_stop_reason = observed_reason

    normalized_stop_reason = str(final_stop_reason or "").strip().lower()
    truncated = normalized_stop_reason in {
        "length",
        "max_tokens",
        "max_output_tokens",
    }
    if truncated:
        reason = "model_output_truncated"
    elif warning_marker:
        reason = "openclaw_no_response"
    else:
        reason = None
    return {
        "ok": reason is None,
        "failure_category": "model" if reason else None,
        "reason": reason,
        "scoreable": bool(reason),
        "retryable": False,
        "final_stop_reason": final_stop_reason,
        "final_assistant_message_seen": final_assistant_message_seen,
        "openclaw_no_response_warning": bool(warning_marker),
        "warning_marker": warning_marker,
    }


def extract_assistant_text(sidecar: dict[str, Any]) -> str:
    stdout_json = sidecar.get("stdout_json")
    if isinstance(stdout_json, dict):
        candidates = []
        meta = stdout_json.get("meta")
        if isinstance(meta, dict):
            candidates.append(meta)
        result = stdout_json.get("result")
        if isinstance(result, dict):
            candidates.append(result)
        candidates.append(stdout_json)
        for meta in candidates:
            for key in ("finalAssistantVisibleText", "finalAssistantRawText"):
                value = meta.get(key)
                if isinstance(value, str) and value.strip():
                    return value.strip()
            payloads = meta.get("payloads")
            if isinstance(payloads, list):
                texts = [
                    str(payload.get("text", ""))
                    for payload in payloads
                    if isinstance(payload, dict) and payload.get("text") is not None
                ]
                if texts:
                    return "\n\n".join(texts).strip()
    return str(sidecar.get("stdout") or "").strip()


def extract_prompt_text(sidecar: dict[str, Any], fallback_message: str) -> str:
    meta = extract_openclaw_meta(sidecar)
    prompt = meta.get("finalPromptText")
    return prompt if isinstance(prompt, str) and prompt.strip() else fallback_message


def normalize_usage(agent_meta: dict[str, Any]) -> dict[str, int]:
    usage = agent_meta.get("usage")
    if not isinstance(usage, dict):
        return {}
    mapped = {
        "inputTokens": usage.get("input"),
        "outputTokens": usage.get("output"),
        "reasoningTokens": usage.get("reasoningTokens"),
        "cacheReadInputTokens": usage.get("cacheRead"),
        "cacheCreationInputTokens": usage.get("cacheWrite"),
    }
    return {key: int(value) for key, value in mapped.items() if isinstance(value, (int, float))}


def copied_session_usage(sidecar: dict[str, Any]) -> dict[str, int]:
    """Recover cumulative provider usage when an interrupted CLI omits agentMeta."""
    copy_status = sidecar.get("nemoclaw_session_copy")
    if not isinstance(copy_status, dict) or copy_status.get("ok") is not True:
        return {}
    session_file = copy_status.get("copied_session_file")
    if not isinstance(session_file, str) or not session_file:
        return {}
    path = Path(session_file).expanduser()
    totals = {
        "inputTokens": 0,
        "outputTokens": 0,
        "reasoningTokens": 0,
        "cacheReadInputTokens": 0,
        "cacheCreationInputTokens": 0,
    }
    found = False
    try:
        for line in path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            entry = json.loads(line)
            message = entry.get("message") if isinstance(entry, dict) else None
            usage = message.get("usage") if isinstance(message, dict) else None
            if not isinstance(usage, dict):
                continue
            found = True
            mapping = {
                "inputTokens": usage.get("input"),
                "outputTokens": usage.get("output"),
                "reasoningTokens": usage.get("reasoningTokens"),
                "cacheReadInputTokens": usage.get("cacheRead"),
                "cacheCreationInputTokens": usage.get("cacheWrite"),
            }
            for key, value in mapping.items():
                if isinstance(value, (int, float)):
                    totals[key] += int(value)
    except (OSError, json.JSONDecodeError):
        return {}
    return totals if found else {}


def enrich_sidecar_with_tool_events(sidecar: dict[str, Any]) -> dict[str, Any]:
    tool_events = extract_tool_events(sidecar)
    timeline_events = (
        sidecar.get("timeline_events")
        if isinstance(sidecar.get("timeline_events"), list)
        else extract_timeline_events(sidecar)
    )
    agent_meta = extract_agent_meta(sidecar)
    session_file = agent_meta.get("sessionFile")
    role_counts = (
        session_message_role_counts(Path(session_file).expanduser())
        if isinstance(session_file, str) and session_file
        else {"assistant": 0, "user": 0, "tool": 0}
    )
    budget_guard_blocks = (
        session_budget_guard_blocks(Path(session_file).expanduser())
        if isinstance(session_file, str) and session_file
        else []
    )
    budget_guard_blocks = [*budget_guard_blocks, *stdio_budget_guard_blocks(sidecar)]
    tool_call_count = sum(1 for event in tool_events if event.get("type") == "tool_call")
    executed_tool_call_count, blocked_tool_call_count = tool_result_execution_counts(tool_events)
    sidecar["tool_events"] = tool_events
    sidecar["timeline_events"] = timeline_events
    sidecar["tool_call_count"] = tool_call_count
    sidecar["blocked_tool_call_count"] = blocked_tool_call_count
    sidecar["executed_tool_call_count"] = executed_tool_call_count
    sidecar["budget_guard_block_count"] = len(budget_guard_blocks)
    sidecar["budget_guard_blocks"] = budget_guard_blocks[:10]
    sidecar["agent_turn_count"] = role_counts["assistant"]
    sidecar["tool_error_count"] = sum(
        1 for event in tool_events if event.get("type") == "tool_result" and event.get("isError")
    )
    sidecar["timeline_event_count"] = len(timeline_events)
    sidecar["conversation_order"] = conversation_order_status(timeline_events)
    return sidecar


def _timeline_event_index(event: dict[str, Any], fallback: int) -> int:
    value = event.get("timelineIndex")
    if isinstance(value, (int, float)):
        return int(value)
    return fallback


def _event_text_contains_final_answer(event: dict[str, Any]) -> bool:
    if event.get("type") != "assistant_message":
        return False
    text = _jsonish_text(event.get("content"))
    if not text.strip():
        return False
    return any(marker in text for marker in FINAL_ANSWER_MARKERS)


def conversation_order_status(timeline_events: list[dict[str, Any]]) -> dict[str, Any]:
    """Check local OpenClaw session order before accepting a task sidecar.

    The W&B Agents verifier checks remote span timestamps. This local check uses
    the session JSONL order itself, which is the authoritative source for the
    task runner before any diagnostic conversion or later W&B verification.
    """
    if not timeline_events:
        return {
            "ok": True,
            "checked": False,
            "source": "openclaw_session_jsonl",
            "reason": "no_timeline_events",
            "event_count": 0,
            "issues": [],
        }

    first_user_index: int | None = None
    first_tool_call_index: int | None = None
    last_tool_call_index: int | None = None
    first_final_answer_index: int | None = None
    seen_tool_calls: set[str] = set()
    issues: list[dict[str, Any]] = []
    warnings: list[dict[str, Any]] = []

    for fallback_index, event in enumerate(timeline_events):
        if not isinstance(event, dict):
            issues.append(
                {
                    "type": "invalid_timeline_event",
                    "index": fallback_index,
                }
            )
            continue
        index = _timeline_event_index(event, fallback_index)
        event_type = event.get("type")
        if event_type == "user_message" and first_user_index is None:
            first_user_index = index
        if event_type == "tool_call":
            if first_tool_call_index is None:
                first_tool_call_index = index
            last_tool_call_index = index
            tool_call_id = event.get("toolCallId")
            if isinstance(tool_call_id, str) and tool_call_id:
                seen_tool_calls.add(tool_call_id)
        if event_type == "tool_result":
            tool_call_id = event.get("toolCallId")
            if not isinstance(tool_call_id, str) or not tool_call_id:
                issues.append(
                    {
                        "type": "tool_result_missing_call_id",
                        "index": index,
                    }
                )
            elif tool_call_id not in seen_tool_calls:
                issues.append(
                    {
                        "type": "tool_result_before_matching_call",
                        "index": index,
                        "toolCallId": tool_call_id,
                    }
                )
        if _event_text_contains_final_answer(event) and first_final_answer_index is None:
            first_final_answer_index = index

    if first_tool_call_index is not None:
        if first_user_index is None:
            issues.append(
                {
                    "type": "tool_present_without_user_message",
                    "first_tool_call_index": first_tool_call_index,
                }
            )
        elif first_tool_call_index <= first_user_index:
            issues.append(
                {
                    "type": "tool_before_or_at_first_user_message",
                    "first_user_index": first_user_index,
                    "first_tool_call_index": first_tool_call_index,
                }
            )

    if (
        first_final_answer_index is not None
        and last_tool_call_index is not None
        and last_tool_call_index > first_final_answer_index
    ):
        warnings.append(
            {
                "type": "tool_after_final_answer",
                "first_final_answer_index": first_final_answer_index,
                "last_tool_call_index": last_tool_call_index,
            }
        )

    return {
        "ok": not issues,
        "checked": True,
        "source": "openclaw_session_jsonl",
        "event_count": len(timeline_events),
        "user_message_count": sum(1 for event in timeline_events if event.get("type") == "user_message"),
        "tool_call_count": sum(1 for event in timeline_events if event.get("type") == "tool_call"),
        "tool_result_count": sum(1 for event in timeline_events if event.get("type") == "tool_result"),
        "assistant_message_count": sum(
            1 for event in timeline_events if event.get("type") == "assistant_message"
        ),
        "first_user_index": first_user_index,
        "first_tool_call_index": first_tool_call_index,
        "last_tool_call_index": last_tool_call_index,
        "first_final_answer_index": first_final_answer_index,
        "issues": issues,
        "warnings": warnings,
    }


def runtime_budget_status(sidecar: dict[str, Any], args: argparse.Namespace) -> dict[str, Any]:
    """Validate post-run usage against benchmark runtime budgets.

    OpenClaw 2026.6.9 does not expose a CLI flag for exact native live usage
    interruption. This function records exact post-run usage when available and
    merges conservative live estimates from session JSONL monitoring. The
    subprocess wrapper can terminate live when the session estimate or observed
    tool-call count crosses a configured cap.
    """
    max_input_tokens = int(getattr(args, "max_input_tokens", 0) or 0)
    max_cumulative_input_tokens = int(getattr(args, "max_cumulative_input_tokens", 0) or 0)
    max_cumulative_output_tokens = int(getattr(args, "max_cumulative_output_tokens", 0) or 0)
    require_actual_token_usage = bool(getattr(args, "require_actual_token_usage", False))
    max_tool_calls = int(getattr(args, "max_tool_calls", 0) or 0)
    max_agent_turns = int(getattr(args, "max_agent_turns", 0) or 0)
    max_tool_wall_seconds = int(getattr(args, "max_tool_wall_seconds", 0) or 0)
    agent_meta = extract_agent_meta(sidecar)
    usage = agent_meta.get("usage") if isinstance(agent_meta.get("usage"), dict) else {}
    normalized_usage = normalize_usage(agent_meta)
    usage_source = "stdout_agent_meta" if normalized_usage else None
    if not normalized_usage:
        normalized_usage = copied_session_usage(sidecar)
        if normalized_usage:
            usage_source = "copied_nemoclaw_session"
    input_tokens = usage.get("input")
    input_tokens = int(input_tokens) if isinstance(input_tokens, (int, float)) else None
    if input_tokens is None and normalized_usage:
        input_tokens = normalized_usage.get("inputTokens")
    actual_input_tokens = None
    actual_output_tokens = None
    if normalized_usage:
        raw_input = normalized_usage.get("inputTokens")
        raw_output = normalized_usage.get("outputTokens")
        cache_read = normalized_usage.get("cacheReadInputTokens") or 0
        cache_write = normalized_usage.get("cacheCreationInputTokens") or 0
        total = normalized_usage.get("totalTokens")
        if isinstance(raw_input, int):
            actual_input_tokens = raw_input + int(cache_read or 0) + int(cache_write or 0)
        if isinstance(raw_output, int):
            actual_output_tokens = raw_output
        if actual_input_tokens is None and isinstance(total, int) and isinstance(actual_output_tokens, int):
            actual_input_tokens = max(0, total - actual_output_tokens)
        if actual_output_tokens is None and isinstance(total, int) and isinstance(actual_input_tokens, int):
            actual_output_tokens = max(0, total - actual_input_tokens)
    tool_call_count = sidecar.get("tool_call_count")
    tool_call_count = int(tool_call_count) if isinstance(tool_call_count, (int, float)) else None
    blocked_tool_call_count = sidecar.get("blocked_tool_call_count")
    blocked_tool_call_count = (
        int(blocked_tool_call_count)
        if isinstance(blocked_tool_call_count, (int, float))
        else 0
    )
    raw_executed_tool_call_count = sidecar.get("executed_tool_call_count")
    has_sidecar_executed_tool_count = isinstance(raw_executed_tool_call_count, (int, float))
    executed_tool_call_count = raw_executed_tool_call_count
    executed_tool_call_count = (
        int(executed_tool_call_count)
        if isinstance(executed_tool_call_count, (int, float))
        else (
            max(0, tool_call_count - blocked_tool_call_count)
            if tool_call_count is not None
            else None
        )
    )
    agent_turn_count = sidecar.get("agent_turn_count")
    agent_turn_count = int(agent_turn_count) if isinstance(agent_turn_count, (int, float)) else None
    live_budget = sidecar.get("live_runtime_budget")
    live_tool_call_count = None
    live_executed_tool_call_count = None
    live_blocked_tool_call_count = 0
    live_agent_turn_count = None
    live_tool_exceeded = False
    live_estimated_input_tokens = None
    live_input_exceeded = False
    live_turn_exceeded = False
    live_policy_observed = False
    live_policy_violation_count = 0
    live_policy_violations: list[Any] = []
    live_budget_guard_exceeded = False
    live_budget_guard_block_count = 0
    live_budget_guard_blocks: list[Any] = []
    sidecar_budget_guard_blocks = (
        sidecar.get("budget_guard_blocks")
        if isinstance(sidecar.get("budget_guard_blocks"), list)
        else []
    )
    sidecar_agent_turn_guard_blocked = any(
        isinstance(block, dict)
        and (
            block.get("kind") == "agent_turn"
            or "agent_turn_limit_exceeded" in str(block.get("text") or "")
        )
        for block in sidecar_budget_guard_blocks
    )
    sidecar_tool_call_guard_blocked = any(
        isinstance(block, dict)
        and (
            block.get("kind") == "tool_call"
            or "tool_call_limit_exceeded" in str(block.get("text") or "")
        )
        for block in sidecar_budget_guard_blocks
    )
    if isinstance(live_budget, dict):
        live_count = live_budget.get("tool_call_count")
        if isinstance(live_count, (int, float)):
            live_tool_call_count = int(live_count)
        live_executed_count = live_budget.get("executed_tool_call_count")
        if isinstance(live_executed_count, (int, float)):
            live_executed_tool_call_count = int(live_executed_count)
        live_blocked_count = live_budget.get("blocked_tool_call_count")
        if isinstance(live_blocked_count, (int, float)):
            live_blocked_tool_call_count = int(live_blocked_count)
        live_turns = live_budget.get("agent_turn_count")
        if isinstance(live_turns, (int, float)):
            live_agent_turn_count = int(live_turns)
        live_input = live_budget.get("estimated_input_tokens")
        if isinstance(live_input, (int, float)):
            live_estimated_input_tokens = int(live_input)
        exceeded_limits = live_budget.get("exceeded_limits")
        exceeded_limit_set = set(exceeded_limits) if isinstance(exceeded_limits, list) else set()
        reason = live_budget.get("reason")
        live_tool_exceeded = (
            "max_tool_calls_exceeded" in exceeded_limit_set
            or reason in {"max_tool_calls_exceeded", "max_input_tokens_and_tool_calls_exceeded"}
        )
        live_input_exceeded = (
            "max_input_tokens_exceeded" in exceeded_limit_set
            or reason in {"max_input_tokens_exceeded", "max_input_tokens_and_tool_calls_exceeded"}
        )
        live_turn_exceeded = (
            "max_agent_turns_exceeded" in exceeded_limit_set
            or "max_agent_turns_reached" in exceeded_limit_set
            or reason in {"max_agent_turns_exceeded", "max_agent_turns_reached"}
        )
        live_policy_observed = (
            "live_tool_policy_violation" in exceeded_limit_set
            or reason == "live_tool_policy_violation"
        )
        live_policy_count = live_budget.get("live_tool_policy_violation_count")
        if isinstance(live_policy_count, (int, float)):
            live_policy_violation_count = int(live_policy_count)
        live_policy_rows = live_budget.get("live_tool_policy_violations")
        if isinstance(live_policy_rows, list):
            live_policy_violations = live_policy_rows
        live_budget_guard_exceeded = "budget_guard_blocked" in exceeded_limit_set or reason == "budget_guard_blocked"
        live_budget_guard_count = live_budget.get("budget_guard_block_count")
        if isinstance(live_budget_guard_count, (int, float)):
            live_budget_guard_block_count = int(live_budget_guard_count)
        live_budget_guard_rows = live_budget.get("budget_guard_blocks")
        if isinstance(live_budget_guard_rows, list):
            live_budget_guard_blocks = live_budget_guard_rows
    if sidecar_budget_guard_blocks:
        live_budget_guard_exceeded = True
        live_budget_guard_block_count = max(
            live_budget_guard_block_count,
            len(sidecar_budget_guard_blocks),
        )
        if live_budget_guard_blocks:
            seen_blocks = {_jsonish_text(block) for block in live_budget_guard_blocks}
            for block in sidecar_budget_guard_blocks:
                block_key = _jsonish_text(block)
                if block_key not in seen_blocks:
                    live_budget_guard_blocks.append(block)
                    seen_blocks.add(block_key)
        else:
            live_budget_guard_blocks = list(sidecar_budget_guard_blocks)
    if live_tool_call_count is not None:
        if tool_call_count is None:
            tool_call_count = live_tool_call_count
        else:
            tool_call_count = max(tool_call_count, live_tool_call_count)
    if live_blocked_tool_call_count:
        blocked_tool_call_count = max(blocked_tool_call_count, live_blocked_tool_call_count)
    if live_executed_tool_call_count is not None:
        if executed_tool_call_count is None:
            executed_tool_call_count = live_executed_tool_call_count
        elif not has_sidecar_executed_tool_count:
            executed_tool_call_count = max(executed_tool_call_count, live_executed_tool_call_count)
    elif tool_call_count is not None:
        executed_tool_call_count = max(0, tool_call_count - blocked_tool_call_count)
    if live_agent_turn_count is not None:
        if agent_turn_count is None:
            agent_turn_count = live_agent_turn_count
        else:
            agent_turn_count = max(agent_turn_count, live_agent_turn_count)

    violations: list[dict[str, Any]] = []
    if max_input_tokens > 0 and (
        (input_tokens is not None and input_tokens > max_input_tokens)
        or live_input_exceeded
    ):
        violations.append(
            {
                "type": "max_input_tokens_exceeded",
                "observed": input_tokens if input_tokens is not None else live_estimated_input_tokens,
                "limit": max_input_tokens,
                "source": (
                    "provider_usage"
                    if input_tokens is not None and input_tokens > max_input_tokens
                    else "live_session_estimate"
                ),
            }
        )
    if require_actual_token_usage and not normalized_usage:
        violations.append(
            {
                "type": "missing_actual_token_usage",
                "observed": None,
                "limit": "required",
                "source": "provider_usage",
            }
        )
    if max_cumulative_input_tokens > 0:
        if actual_input_tokens is None:
            violations.append(
                {
                    "type": "missing_actual_input_tokens",
                    "observed": None,
                    "limit": max_cumulative_input_tokens,
                    "source": "provider_usage",
                }
            )
        elif actual_input_tokens > max_cumulative_input_tokens:
            violations.append(
                {
                    "type": "max_cumulative_input_tokens_exceeded",
                    "observed": actual_input_tokens,
                    "limit": max_cumulative_input_tokens,
                    "source": "provider_usage",
                }
            )
    if max_cumulative_output_tokens > 0:
        if actual_output_tokens is None:
            violations.append(
                {
                    "type": "missing_actual_output_tokens",
                    "observed": None,
                    "limit": max_cumulative_output_tokens,
                    "source": "provider_usage",
                }
            )
        elif actual_output_tokens > max_cumulative_output_tokens:
            violations.append(
                {
                    "type": "max_cumulative_output_tokens_exceeded",
                    "observed": actual_output_tokens,
                    "limit": max_cumulative_output_tokens,
                    "source": "provider_usage",
                }
            )
    if max_tool_calls > 0 and (
        (executed_tool_call_count is not None and executed_tool_call_count > max_tool_calls)
        or live_tool_exceeded
        or sidecar_tool_call_guard_blocked
    ):
        violations.append(
            {
                "type": "max_tool_calls_exceeded",
                "observed": (
                    max_tool_calls + 1
                    if sidecar_tool_call_guard_blocked
                    and (
                        executed_tool_call_count is None
                        or executed_tool_call_count <= max_tool_calls
                    )
                    else executed_tool_call_count
                ),
                "limit": max_tool_calls,
                "source": (
                    "openclaw_runtime_patch"
                    if sidecar_tool_call_guard_blocked
                    else "live_runtime_budget"
                    if live_tool_exceeded
                    else "openclaw_session_jsonl"
                ),
            }
        )
    if max_agent_turns > 0 and (
        (agent_turn_count is not None and agent_turn_count > max_agent_turns)
        or live_turn_exceeded
        or sidecar_agent_turn_guard_blocked
    ):
        violations.append(
            {
                "type": "max_agent_turns_exceeded",
                "observed": (
                    max_agent_turns + 1
                    if sidecar_agent_turn_guard_blocked and (
                        agent_turn_count is None or agent_turn_count <= max_agent_turns
                    )
                    else agent_turn_count
                ),
                "limit": max_agent_turns,
                "source": (
                    "openclaw_runtime_patch"
                    if sidecar_agent_turn_guard_blocked
                    else "live_runtime_budget"
                    if live_turn_exceeded
                    else "openclaw_session_jsonl"
                ),
            }
        )
    for block in live_budget_guard_blocks:
        if not isinstance(block, dict):
            continue
        kind = str(block.get("kind") or "")
        if kind not in {
            "cumulative_input_tokens",
            "cumulative_output_tokens",
            "missing_actual_token_usage",
            "missing_actual_input_tokens",
            "missing_actual_output_tokens",
        }:
            continue
        fields = budget_guard_block_fields(str(block.get("text") or ""))
        if kind == "cumulative_input_tokens":
            violation_type = "max_cumulative_input_tokens_exceeded"
        elif kind == "cumulative_output_tokens":
            violation_type = "max_cumulative_output_tokens_exceeded"
        else:
            violation_type = kind
        violation_limit = fields.get("limit")
        if violation_type == "missing_actual_token_usage":
            violation_limit = "required"
        if any(item.get("type") == violation_type and item.get("source") == "openclaw_runtime_patch" for item in violations):
            continue
        violations.append(
            {
                "type": violation_type,
                "observed": fields.get("observed"),
                "limit": violation_limit,
                "source": "openclaw_runtime_patch",
                "block": block,
            }
        )
    return {
        "ok": not violations,
        "enforced": bool(
            max_input_tokens > 0
            or max_cumulative_input_tokens > 0
            or max_cumulative_output_tokens > 0
            or require_actual_token_usage
            or max_tool_calls > 0
            or max_agent_turns > 0
            or max_tool_wall_seconds > 0
        ),
        "limits": {
            "max_input_tokens": max_input_tokens or None,
            "max_cumulative_input_tokens": max_cumulative_input_tokens or None,
            "max_cumulative_output_tokens": max_cumulative_output_tokens or None,
            "require_actual_token_usage": require_actual_token_usage,
            "max_tool_calls": max_tool_calls or None,
            "max_agent_turns": max_agent_turns or None,
            "max_tool_wall_seconds": max_tool_wall_seconds or None,
        },
        "observed": {
            "input_tokens": input_tokens,
            "actual_cumulative_input_tokens": actual_input_tokens,
            "actual_cumulative_output_tokens": actual_output_tokens,
            "actual_usage": normalized_usage,
            "actual_usage_source": usage_source,
            "estimated_input_tokens": live_estimated_input_tokens,
            "tool_call_count": tool_call_count,
            "executed_tool_call_count": executed_tool_call_count,
            "blocked_tool_call_count": blocked_tool_call_count,
            "agent_turn_count": agent_turn_count,
        },
        "budget_guard": {
            "blocked": bool(live_budget_guard_exceeded or blocked_tool_call_count),
            "block_count": max(blocked_tool_call_count, live_budget_guard_block_count),
            "blocks": live_budget_guard_blocks,
            "policy_observed": live_policy_observed,
            "policy_violation_count": live_policy_violation_count,
            "policy_violations": live_policy_violations,
        },
        "live": live_budget if isinstance(live_budget, dict) else {},
        "violations": violations,
    }


def extract_reasoning_text(sidecar: dict[str, Any]) -> str:
    agent_meta = extract_agent_meta(sidecar)
    session_file = agent_meta.get("sessionFile")
    if not isinstance(session_file, str) or not session_file:
        return ""
    session_path = Path(session_file).expanduser()
    if not session_path.exists():
        return ""

    chunks: list[str] = []
    for raw in session_path.read_text(encoding="utf-8", errors="replace").splitlines():
        try:
            event = json.loads(raw)
        except json.JSONDecodeError:
            continue
        message = event.get("message") if isinstance(event, dict) else None
        if not isinstance(message, dict) or message.get("role") != "assistant":
            continue
        content = message.get("content")
        if not isinstance(content, list):
            continue
        for part in content:
            if not isinstance(part, dict):
                continue
            reasoning = None
            if part.get("type") == "thinking":
                reasoning = part.get("thinking")
            if reasoning is None and part.get("thinkingSignature") == "reasoning_content":
                reasoning = part.get("thinking")
            if isinstance(reasoning, str) and reasoning.strip():
                chunks.append(reasoning.strip())
    return "\n\n---\n\n".join(chunks)


def _extract_content_text(content: Any) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        texts: list[str] = []
        for part in content:
            if not isinstance(part, dict):
                continue
            text = part.get("text") or part.get("content")
            if isinstance(text, str):
                texts.append(text)
        return "\n\n".join(texts)
    if content is None:
        return ""
    return json.dumps(content, ensure_ascii=False)


def extract_tool_events(sidecar: dict[str, Any]) -> list[dict[str, Any]]:
    agent_meta = extract_agent_meta(sidecar)
    session_file = agent_meta.get("sessionFile")
    if not isinstance(session_file, str) or not session_file:
        return []
    session_path = Path(session_file).expanduser()
    if not session_path.exists():
        return []

    events: list[dict[str, Any]] = []
    for raw in session_path.read_text(encoding="utf-8", errors="replace").splitlines():
        try:
            event = json.loads(raw)
        except json.JSONDecodeError:
            continue
        message = event.get("message") if isinstance(event, dict) else None
        if not isinstance(message, dict):
            continue
        timestamp = message.get("timestamp") or event.get("timestamp")
        role = message.get("role")
        if role == "assistant":
            seen_tool_calls: set[str] = set()
            content = message.get("content")
            if isinstance(content, list):
                for part in content:
                    if not isinstance(part, dict):
                        continue
                    if part.get("type") not in {"toolCall", "tool_use"}:
                        continue
                    tool_call_id = part.get("id") or part.get("toolCallId") or part.get("tool_use_id")
                    if isinstance(tool_call_id, str) and tool_call_id:
                        seen_tool_calls.add(tool_call_id)
                    tool_name = part.get("name") or part.get("toolName")
                    arguments = part.get("arguments")
                    if arguments is None:
                        arguments = part.get("input")
                    if arguments is None:
                        arguments = part.get("partialArgs")
                    events.append(
                        {
                            "type": "tool_call",
                            "toolCallId": tool_call_id,
                            "toolName": tool_name or "unknown",
                            "arguments": arguments,
                            "timestamp": timestamp,
                        }
                    )
            for key in ("tool_calls", "toolCalls"):
                calls = message.get(key)
                if not isinstance(calls, list):
                    continue
                for call in calls:
                    if not isinstance(call, dict):
                        continue
                    tool_call_id = call.get("id") or call.get("toolCallId") or call.get("tool_use_id")
                    if isinstance(tool_call_id, str) and tool_call_id:
                        if tool_call_id in seen_tool_calls:
                            continue
                        seen_tool_calls.add(tool_call_id)
                    function = call.get("function") if isinstance(call.get("function"), dict) else {}
                    tool_name = call.get("name") or call.get("toolName") or function.get("name")
                    arguments = call.get("arguments")
                    if arguments is None:
                        arguments = function.get("arguments")
                    if arguments is None:
                        arguments = call.get("input")
                    events.append(
                        {
                            "type": "tool_call",
                            "toolCallId": tool_call_id,
                            "toolName": tool_name or "unknown",
                            "arguments": arguments,
                            "timestamp": timestamp,
                        }
                    )
        elif role in {"toolResult", "tool"}:
            details = message.get("details")
            events.append(
                {
                    "type": "tool_result",
                    "toolCallId": message.get("toolCallId") or message.get("tool_call_id"),
                    "toolName": message.get("toolName") or message.get("name") or "unknown",
                    "content": _extract_content_text(message.get("content")),
                    "details": details if isinstance(details, dict) else None,
                    "isError": bool(message.get("isError")),
                    "timestamp": timestamp,
                }
            )
    for index, event in enumerate(events):
        event["index"] = index
    return events


def extract_timeline_events(sidecar: dict[str, Any]) -> list[dict[str, Any]]:
    agent_meta = extract_agent_meta(sidecar)
    session_file = agent_meta.get("sessionFile")
    if not isinstance(session_file, str) or not session_file:
        return []
    session_path = Path(session_file).expanduser()
    if not session_path.exists():
        return []

    events: list[dict[str, Any]] = []
    for raw in session_path.read_text(encoding="utf-8", errors="replace").splitlines():
        try:
            event = json.loads(raw)
        except json.JSONDecodeError:
            continue
        message = event.get("message") if isinstance(event, dict) else None
        if not isinstance(message, dict):
            continue
        timestamp = message.get("timestamp") or event.get("timestamp")
        role = message.get("role")
        content = message.get("content")
        if role == "user":
            events.append(
                {
                    "type": "user_message",
                    "role": "user",
                    "content": _extract_content_text(content),
                    "timestamp": timestamp,
                }
            )
        elif role == "assistant":
            seen_tool_calls: set[str] = set()
            if not isinstance(content, list):
                text = _extract_content_text(content)
                if text.strip():
                    events.append(
                        {
                            "type": "assistant_message",
                            "role": "assistant",
                            "content": text,
                            "timestamp": timestamp,
                        }
                    )
            else:
                for part in content:
                    if not isinstance(part, dict):
                        continue
                    part_type = part.get("type")
                    if part_type in {"toolCall", "tool_use"}:
                        tool_call_id = part.get("id") or part.get("toolCallId") or part.get("tool_use_id")
                        if isinstance(tool_call_id, str) and tool_call_id:
                            seen_tool_calls.add(tool_call_id)
                        arguments = part.get("arguments")
                        if arguments is None:
                            arguments = part.get("input")
                        if arguments is None:
                            arguments = part.get("partialArgs")
                        events.append(
                            {
                                "type": "tool_call",
                                "role": "assistant",
                                "toolCallId": tool_call_id,
                                "toolName": part.get("name") or part.get("toolName") or "unknown",
                                "arguments": arguments,
                                "timestamp": timestamp,
                            }
                        )
                        continue
                    reasoning = None
                    if part_type == "thinking":
                        reasoning = part.get("thinking")
                    if reasoning is None and part.get("thinkingSignature") == "reasoning_content":
                        reasoning = part.get("thinking")
                    if isinstance(reasoning, str) and reasoning.strip():
                        events.append(
                            {
                                "type": "assistant_reasoning",
                                "role": "assistant",
                                "content": reasoning.strip(),
                                "timestamp": timestamp,
                            }
                        )
                        continue
                    text = part.get("text") or part.get("content")
                    if isinstance(text, str) and text.strip():
                        events.append(
                            {
                                "type": "assistant_message",
                                "role": "assistant",
                                "content": text,
                                "timestamp": timestamp,
                            }
                        )
            for key in ("tool_calls", "toolCalls"):
                calls = message.get(key)
                if not isinstance(calls, list):
                    continue
                for call in calls:
                    if not isinstance(call, dict):
                        continue
                    tool_call_id = call.get("id") or call.get("toolCallId") or call.get("tool_use_id")
                    if isinstance(tool_call_id, str) and tool_call_id:
                        if tool_call_id in seen_tool_calls:
                            continue
                        seen_tool_calls.add(tool_call_id)
                    function = call.get("function") if isinstance(call.get("function"), dict) else {}
                    tool_name = call.get("name") or call.get("toolName") or function.get("name")
                    arguments = call.get("arguments")
                    if arguments is None:
                        arguments = function.get("arguments")
                    if arguments is None:
                        arguments = call.get("input")
                    events.append(
                        {
                            "type": "tool_call",
                            "role": "assistant",
                            "toolCallId": tool_call_id,
                            "toolName": tool_name or "unknown",
                            "arguments": arguments,
                            "timestamp": timestamp,
                        }
                    )
        elif role in {"toolResult", "tool"}:
            details = message.get("details")
            events.append(
                {
                    "type": "tool_result",
                    "role": "tool",
                    "toolCallId": message.get("toolCallId") or message.get("tool_call_id"),
                    "toolName": message.get("toolName") or message.get("name") or "unknown",
                    "content": _extract_content_text(content),
                    "details": details if isinstance(details, dict) else None,
                    "isError": bool(message.get("isError")),
                    "timestamp": timestamp,
                }
            )
    for index, event in enumerate(events):
        event["timelineIndex"] = index
    return events


def parse_last_json_line(text: str) -> dict[str, Any] | None:
    decoder = json.JSONDecoder()
    for match in re.finditer(r"\{", text or ""):
        suffix = (text or "")[match.start() :]
        try:
            parsed, end = decoder.raw_decode(suffix)
        except json.JSONDecodeError:
            continue
        if isinstance(parsed, dict) and not suffix[end:].strip():
            return parsed
    for line in reversed((text or "").splitlines()):
        stripped = line.strip()
        if not stripped or not stripped.startswith("{"):
            continue
        try:
            parsed = json.loads(stripped)
        except json.JSONDecodeError:
            continue
        return parsed if isinstance(parsed, dict) else None
    return None


def is_transient_weave_sidecar_error(stdout: str, stderr: str) -> bool:
    text = f"{stdout}\n{stderr}".lower()
    transient_markers = (
        "concurrent export limit reached",
        "rate limit",
        "too many requests",
        "timeout",
        "timed out",
        "econnreset",
        "etimedout",
        "temporarily unavailable",
        "503",
        "429",
    )
    return any(marker in text for marker in transient_markers)


def log_weave_sidecar(
    args: argparse.Namespace,
    sidecar_path: Path,
    sidecar: dict[str, Any],
    message_text: str,
) -> dict[str, Any]:
    env, env_loaded, _ = prepare_env(args.env_file, args.openclaw_bin)
    if not env.get("WANDB_API_KEY"):
        result = {"ok": False, "skipped": True, "reason": "WANDB_API_KEY is not set", "env_file_loaded": env_loaded}
        if args.weave_sidecar_strict:
            return result
        return result

    node_path = shutil.which("node", path=env.get("PATH"))
    if not node_path:
        return {"ok": False, "skipped": True, "reason": "node is not available on PATH"}

    agent_meta = extract_agent_meta(sidecar)
    meta = extract_openclaw_meta(sidecar)
    metadata = sidecar.get("metadata", {}) if isinstance(sidecar.get("metadata"), dict) else {}
    benchmark_id = metadata.get("benchmark_id")
    task_id = metadata.get("task_id")
    reasoning_text = extract_reasoning_text(sidecar)
    tool_events = sidecar.get("tool_events") if isinstance(sidecar.get("tool_events"), list) else extract_tool_events(sidecar)
    timeline_events = extract_timeline_events(sidecar)
    tool_call_count = sidecar.get("tool_call_count")
    if not isinstance(tool_call_count, int):
        tool_call_count = sum(1 for event in tool_events if event.get("type") == "tool_call")
    tool_error_count = sidecar.get("tool_error_count")
    if not isinstance(tool_error_count, int):
        tool_error_count = sum(1 for event in tool_events if event.get("type") == "tool_result" and event.get("isError"))
    payload = {
        "entity": args.weave_entity,
        "project": args.weave_project,
        "agentName": args.weave_agent_name,
        "agentVersion": args.weave_agent_version,
        "agentDescription": args.weave_agent_description,
        "benchmarkId": benchmark_id,
        "taskId": task_id,
        "protocolVersion": metadata.get("protocol_version"),
        "promptHash": metadata.get("prompt_hash"),
        "toolPolicyHash": metadata.get("tool_policy_hash"),
        "verifierHash": metadata.get("verifier_hash"),
        "model": agent_meta.get("model") or getattr(args, "model", None) or metadata.get("model_id") or "unknown",
        "providerName": agent_meta.get("provider") or args.weave_provider_name or "openclaw",
        "conversationId": agent_meta.get("sessionId")
        or getattr(args, "session_key", None)
        or f"{benchmark_id}:{task_id}",
        "sessionFile": agent_meta.get("sessionFile"),
        "cwd": sidecar.get("cwd"),
        "openclawConfigPath": sidecar.get("openclaw_config_path"),
        "openclawResultPath": str(sidecar_path),
        "prompt": extract_prompt_text(sidecar, message_text),
        "assistant": extract_assistant_text(sidecar),
        "reasoning": reasoning_text,
        "toolEvents": tool_events,
        "timelineEvents": timeline_events,
        "toolCallCount": tool_call_count,
        "toolErrorCount": tool_error_count,
        "toolPolicy": sidecar.get("tool_policy", {}),
        "toolPolicyOk": sidecar.get("tool_policy_ok"),
        "toolPolicyViolations": sidecar.get("tool_policy_violations", []),
        "conversationOrder": sidecar.get("conversation_order", {}),
        "usage": normalize_usage(agent_meta),
        "startedAt": sidecar.get("started_at"),
        "endedAt": sidecar.get("ended_at"),
        "durationMs": meta.get("durationMs") if isinstance(meta, dict) else None,
        "returncode": sidecar.get("returncode"),
        "stopReason": meta.get("stopReason") if isinstance(meta, dict) else None,
        "thinking": (meta.get("requestShaping") or {}).get("thinking")
        if isinstance(meta, dict)
        else getattr(args, "thinking", None),
        "nativePluginExpected": True,
    }
    if sidecar.get("stderr"):
        payload["stderrTail"] = str(sidecar.get("stderr"))[-4000:]

    max_attempts = max(1, 1 + int(getattr(args, "weave_sidecar_retries", 6) or 0))
    base_sleep = max(0.0, float(getattr(args, "weave_sidecar_retry_base_seconds", 5.0) or 0.0))
    attempts: list[dict[str, Any]] = []
    last_result: dict[str, Any] | None = None
    with tempfile.TemporaryDirectory(prefix="nejumi-weave-sidecar-") as temp_dir:
        payload_path = Path(temp_dir) / "payload.json"
        payload_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        for attempt_index in range(max_attempts):
            completed = subprocess.run(
                [node_path, str(WEAVE_SIDECAR_SCRIPT), str(payload_path)],
                cwd=str(REPO_ROOT),
                env=env,
                text=True,
                capture_output=True,
                check=False,
            )
            transient = is_transient_weave_sidecar_error(completed.stdout, completed.stderr)
            result: dict[str, Any] = {
                "ok": completed.returncode == 0,
                "returncode": completed.returncode,
                "stdout": completed.stdout,
                "stderr": completed.stderr,
                "agent_name": args.weave_agent_name,
                "project": f"{args.weave_entity}/{args.weave_project}",
                "attempt": attempt_index + 1,
                "max_attempts": max_attempts,
                "transient": transient,
            }
            result["stdout_json"] = parse_last_json_line(completed.stdout)
            attempt_summary = {
                "attempt": attempt_index + 1,
                "returncode": completed.returncode,
                "transient": transient,
                "stdout_tail": completed.stdout[-1000:],
                "stderr_tail": completed.stderr[-1000:],
            }
            attempts.append(attempt_summary)
            result["attempts"] = attempts
            last_result = result
            if result["ok"]:
                return result
            if not transient or attempt_index + 1 >= max_attempts:
                break
            sleep_seconds = min(60.0, base_sleep * (2**attempt_index))
            if sleep_seconds:
                time.sleep(sleep_seconds)
    assert last_result is not None
    return last_result


def relog_sidecar(args: argparse.Namespace) -> None:
    raise SystemExit(
        "relog-sidecar is disabled for Taiwan agentic benchmarks. Use native "
        "weave-openclaw Agents traces only; manual relogging is not valid evidence."
    )
    sidecar_path = args.sidecar_path
    sidecar = json.loads(sidecar_path.read_text(encoding="utf-8"))
    enrich_sidecar_with_tool_events(sidecar)
    message_path = args.message_file or sidecar_path.parent / "message.md"
    message_text = message_path.read_text(encoding="utf-8") if message_path.exists() else ""
    weave_result = log_weave_sidecar(args, sidecar_path, sidecar, message_text)
    sidecar["weave_sidecar"] = weave_result
    sidecar_path.write_text(json.dumps(sidecar, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(weave_result, ensure_ascii=False, indent=2))
    if args.weave_sidecar_strict and not weave_result.get("ok"):
        raise SystemExit("Weave sidecar logging failed")


def print_preflight(args: argparse.Namespace) -> None:
    print(
        json.dumps(
            preflight(
                args.openclaw_bin,
                args.env_file,
                nemoclaw_bin=args.nemoclaw_bin,
                nemoclaw_sandbox=args.nemoclaw_sandbox,
            ),
            ensure_ascii=False,
            indent=2,
        )
    )


def agents_api_post(env: dict[str, str], path: str, payload: dict[str, Any]) -> dict[str, Any]:
    api_key = env.get("WANDB_API_KEY")
    if not api_key:
        raise SystemExit("WANDB_API_KEY is required for W&B Agents API checks")
    token = base64.b64encode(f"api:{api_key}".encode("utf-8")).decode("ascii")
    request = urllib.request.Request(
        f"{AGENTS_API_BASE_URL}{path}",
        data=json.dumps(payload).encode("utf-8"),
        method="POST",
        headers={
            "Authorization": f"Basic {token}",
            "Content-Type": "application/json",
        },
    )
    try:
        with urllib.request.urlopen(request, timeout=30) as response:
            return json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")[:1000]
        raise SystemExit(f"W&B Agents API check failed: HTTP {exc.code}: {detail}") from exc


def _message_count(value: Any) -> int:
    return len(value) if isinstance(value, list) else 0


def _jsonish_text(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    try:
        return json.dumps(value, ensure_ascii=False, sort_keys=True)
    except (TypeError, ValueError):
        return str(value)


def _chat_messages(trace_chat_payload: dict[str, Any] | None) -> list[dict[str, Any]]:
    if not isinstance(trace_chat_payload, dict):
        return []
    messages = trace_chat_payload.get("messages", [])
    if not isinstance(messages, list):
        return []
    return [message for message in messages if isinstance(message, dict)]


def _chat_component_text(message: dict[str, Any], component: str, fields: tuple[str, ...]) -> str:
    payload = message.get(component)
    if not isinstance(payload, dict):
        return ""
    parts = [_jsonish_text(payload.get(field)) for field in fields if payload.get(field)]
    return "\n".join(part for part in parts if part)


def _chat_user_text(message: dict[str, Any]) -> str:
    return _chat_component_text(message, "user_message", ("text", "content", "message", "prompt"))


def _chat_assistant_text(message: dict[str, Any]) -> str:
    return _chat_component_text(
        message,
        "assistant_message",
        ("text", "content", "message", "response"),
    )


def _chat_tool_content(message: dict[str, Any]) -> str:
    return _chat_component_text(
        message,
        "tool_call",
        ("tool_arguments", "tool_result", "arguments", "result", "input", "output"),
    )


def _chat_timestamp(message: dict[str, Any]) -> float | None:
    return _parse_span_timestamp(message.get("started_at"))


def _chat_is_tool_call(message: dict[str, Any]) -> bool:
    return isinstance(message.get("tool_call"), dict)


def _contains_tool_call(value: Any) -> bool:
    if isinstance(value, list):
        return any(_contains_tool_call(item) for item in value)
    if not isinstance(value, dict):
        return False
    type_value = value.get("type")
    if isinstance(type_value, str) and type_value in TOOL_CALL_TYPE_MARKERS:
        return True
    if any(value.get(key) for key in TOOL_CALL_KEYS):
        return True
    return any(_contains_tool_call(item) for item in value.values())


def _chat_is_final_answer(message: dict[str, Any]) -> bool:
    if _chat_is_tool_call(message) or _contains_tool_call(message.get("assistant_message")):
        return False
    assistant_text = _chat_assistant_text(message)
    return any(marker in assistant_text for marker in FINAL_ANSWER_MARKERS)


def summarize_chat_message(message: dict[str, Any]) -> dict[str, Any]:
    user_text = _chat_user_text(message)
    assistant_text = _chat_assistant_text(message)
    tool_content = _chat_tool_content(message)
    return {
        "type": message.get("type"),
        "started_at": message.get("started_at"),
        "has_user_message": bool(user_text),
        "has_assistant_message": bool(assistant_text),
        "has_tool_call": _chat_is_tool_call(message),
        "has_tool_content": bool(tool_content),
        "is_final_answer": _chat_is_final_answer(message),
        "user_text_preview": user_text[:500],
        "assistant_text_preview": assistant_text[:500],
        "tool_content_preview": tool_content[:500],
    }


def _span_output_text(span: dict[str, Any]) -> str:
    return _jsonish_text(span.get("output_messages"))


def _is_final_answer_span(span: dict[str, Any]) -> bool:
    if span.get("operation_name") not in {"chat", "invoke_agent"}:
        return False
    if _contains_tool_call(span.get("output_messages")):
        return False
    output_text = _span_output_text(span)
    return any(marker in output_text for marker in FINAL_ANSWER_MARKERS)


def _chronological_key(span: dict[str, Any]) -> tuple[str, str, str]:
    return (
        str(span.get("started_at") or ""),
        str(span.get("ended_at") or ""),
        str(span.get("span_id") or ""),
    )


def _parse_span_timestamp(value: Any) -> float | None:
    if not isinstance(value, str) or not value.strip():
        return None
    text = value.strip()
    if text.endswith("Z"):
        text = f"{text[:-1]}+00:00"
    try:
        parsed = datetime.fromisoformat(text)
    except ValueError:
        return None
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed.timestamp()


def _span_timestamps(span: dict[str, Any]) -> tuple[float | None, float | None]:
    return (
        _parse_span_timestamp(span.get("started_at")),
        _parse_span_timestamp(span.get("ended_at")),
    )


def _span_start(span: dict[str, Any]) -> float | None:
    return _span_timestamps(span)[0]


def _span_end(span: dict[str, Any]) -> float | None:
    return _span_timestamps(span)[1]


def _timestamp_quality_issues(spans: list[dict[str, Any]]) -> list[dict[str, Any]]:
    issues: list[dict[str, Any]] = []
    for index, span in enumerate(spans, start=1):
        started_at = span.get("started_at")
        ended_at = span.get("ended_at")
        started_ts, ended_ts = _span_timestamps(span)
        if started_ts is None or ended_ts is None:
            issues.append(
                {
                    "index": index,
                    "span_id": span.get("span_id"),
                    "started_at": started_at,
                    "ended_at": ended_at,
                    "error": "missing_or_invalid_timestamp",
                }
            )
            continue
        if ended_ts < started_ts:
            issues.append(
                {
                    "index": index,
                    "span_id": span.get("span_id"),
                    "started_at": started_at,
                    "ended_at": ended_at,
                    "error": "ended_before_started",
                }
            )
    return issues


def summarize_agent_span(span: dict[str, Any]) -> dict[str, Any]:
    input_messages = span.get("input_messages")
    output_messages = span.get("output_messages")
    return {
        "started_at": span.get("started_at"),
        "ended_at": span.get("ended_at"),
        "span_name": span.get("span_name"),
        "operation_name": span.get("operation_name"),
        "agent_name": span.get("agent_name"),
        "provider_name": span.get("provider_name"),
        "request_model": span.get("request_model"),
        "conversation_id": span.get("conversation_id"),
        "trace_id": span.get("trace_id"),
        "span_id": span.get("span_id"),
        "parent_span_id": span.get("parent_span_id"),
        "tool_name": span.get("tool_name"),
        "error_type": span.get("error_type"),
        "has_input_messages": bool(input_messages),
        "has_output_messages": bool(output_messages),
        "input_message_count": _message_count(input_messages),
        "output_message_count": _message_count(output_messages),
        "has_tool_call_arguments": bool(span.get("tool_call_arguments")),
        "has_tool_call_result": bool(span.get("tool_call_result")),
        "has_final_answer_marker": _is_final_answer_span(span),
    }


def _latest_trace_id(raw_spans: list[dict[str, Any]]) -> str | None:
    trace_latest_keys: dict[str, tuple[str, str, str]] = {}
    for span in raw_spans:
        trace_id = span.get("trace_id")
        if not isinstance(trace_id, str) or not trace_id:
            continue
        key = _chronological_key(span)
        if trace_id not in trace_latest_keys or key > trace_latest_keys[trace_id]:
            trace_latest_keys[trace_id] = key
    if not trace_latest_keys:
        return None
    return max(trace_latest_keys.items(), key=lambda item: (item[1], item[0]))[0]


def _span_time_label(spans: list[dict[str, Any]], key: str, *, minimum: bool) -> str | None:
    values = [str(span.get(key) or "") for span in spans if span.get(key)]
    if not values:
        return None
    return min(values) if minimum else max(values)


def build_agents_trace_order_health(
    spans: list[dict[str, Any]],
    timestamp_issues: list[dict[str, Any]],
    *,
    trace_chat_messages: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    chat_messages = trace_chat_messages or []
    message_spans = [span for span in spans if span.get("operation_name") in {"chat", "invoke_agent"}]
    message_spans_with_input = [span for span in message_spans if span.get("has_input_messages")]
    tool_spans = [span for span in spans if span.get("operation_name") == "execute_tool"]
    final_answer_spans = [span for span in message_spans if span.get("has_final_answer_marker")]
    chat_messages_with_input = [message for message in chat_messages if _chat_user_text(message)]
    chat_tool_calls = [message for message in chat_messages if _chat_is_tool_call(message)]
    chat_final_answer_messages = [message for message in chat_messages if _chat_is_final_answer(message)]
    input_start_records = [
        (_span_start(span), str(span.get("started_at") or ""))
        for span in message_spans_with_input
    ] + [
        (_chat_timestamp(message), str(message.get("started_at") or ""))
        for message in chat_messages_with_input
    ]
    input_start_records = [record for record in input_start_records if record[0] is not None]
    tool_start_records = [
        (_span_start(span), str(span.get("started_at") or ""))
        for span in tool_spans
    ] + [
        (_chat_timestamp(message), str(message.get("started_at") or ""))
        for message in chat_tool_calls
    ]
    tool_start_records = [record for record in tool_start_records if record[0] is not None]
    final_answer_boundary_records = [
        (_span_end(span), str(span.get("ended_at") or ""))
        for span in final_answer_spans
    ] + [
        (_chat_timestamp(message), str(message.get("started_at") or ""))
        for message in chat_final_answer_messages
    ]
    final_answer_boundary_records = [
        record for record in final_answer_boundary_records if record[0] is not None
    ]
    message_input_count = len(message_spans_with_input) + len(chat_messages_with_input)
    final_answer_count = len(final_answer_spans) + len(chat_final_answer_messages)
    health: dict[str, Any] = {
        "timestamp_quality_ok": not timestamp_issues,
        "timestamp_issue_count": len(timestamp_issues),
        "timestamp_issues": timestamp_issues[:10],
        "message_span_count": len(message_spans),
        "message_spans_with_input": message_input_count,
        "tool_span_count": len(tool_spans),
        "final_answer_span_count": final_answer_count,
        "trace_order_ok": True,
        "trace_user_message_order_ok": True,
        "trace_final_answer_order_ok": True,
        "order_issues": [],
        "order_warnings": [],
        "first_message_started_at": _span_time_label(message_spans, "started_at", minimum=True),
        "first_input_message_started_at": _span_time_label(
            message_spans_with_input,
            "started_at",
            minimum=True,
        ),
        "first_tool_started_at": _span_time_label(tool_spans, "started_at", minimum=True),
        "last_tool_started_at": _span_time_label(tool_spans, "started_at", minimum=False),
        "first_final_answer_ended_at": _span_time_label(final_answer_spans, "ended_at", minimum=True),
        "chat_messages_with_input": len(chat_messages_with_input),
        "chat_tool_call_count": len(chat_tool_calls),
        "chat_final_answer_message_count": len(chat_final_answer_messages),
    }
    order_issues = health["order_issues"]
    order_warnings = health["order_warnings"]
    if timestamp_issues:
        if message_spans and tool_spans:
            health["trace_order_ok"] = False
            order_issues.append("trace_order_not_comparable_invalid_timestamps")
        if tool_spans:
            health["trace_user_message_order_ok"] = False
            order_issues.append("trace_user_message_order_not_comparable_invalid_timestamps")
        if final_answer_spans and tool_spans:
            health["trace_final_answer_order_ok"] = False
            order_issues.append("trace_final_answer_order_not_comparable_invalid_timestamps")
        return health

    if message_spans and tool_spans:
        first_message = min(_span_start(span) for span in message_spans)
        first_tool = min(_span_start(span) for span in tool_spans)
        if first_message is None or first_tool is None or first_tool <= first_message:
            health["trace_order_ok"] = False
            order_issues.append("tool_started_before_or_at_first_message")

    if tool_spans:
        if not input_start_records:
            health["trace_user_message_order_ok"] = False
            order_issues.append("tool_present_without_visible_user_input")
        else:
            first_input_message, first_input_started_at = min(input_start_records)
            first_tool, first_tool_started_at = min(tool_start_records)
            health["first_input_message_started_at"] = first_input_started_at
            health["first_tool_started_at"] = first_tool_started_at
            if first_input_message is None or first_tool is None or first_tool <= first_input_message:
                health["trace_user_message_order_ok"] = False
                order_issues.append("tool_started_before_or_at_visible_user_input")

    if final_answer_count > 0 and tool_spans:
        if not final_answer_boundary_records or not tool_start_records:
            health["trace_final_answer_order_ok"] = False
            order_issues.append("trace_final_answer_order_not_comparable_invalid_timestamps")
        else:
            first_final_answer_boundary, first_final_answer_at = min(final_answer_boundary_records)
            last_tool_start, last_tool_started_at = max(tool_start_records)
            health["first_final_answer_at"] = first_final_answer_at
            health["last_tool_started_at"] = last_tool_started_at
        if (
            not final_answer_boundary_records
            or not tool_start_records
            or last_tool_start >= first_final_answer_boundary
        ):
            order_warnings.append("tool_started_after_or_at_final_answer_end")

    return health


def agents_conversation_matches(
    span: dict[str, Any],
    *,
    conversation_id: str | None,
    conversation_id_contains: str | None,
) -> bool:
    value = str(span.get("conversation_id") or "")
    if conversation_id and value != conversation_id:
        return False
    if conversation_id_contains and conversation_id_contains not in value:
        return False
    return True


def agents_span_matches(
    span: dict[str, Any],
    *,
    agent_name: str | None,
    conversation_id: str | None,
    conversation_id_contains: str | None,
) -> bool:
    if not agents_conversation_matches(
        span,
        conversation_id=conversation_id,
        conversation_id_contains=conversation_id_contains,
    ):
        return False
    if not agent_name or span.get("agent_name") == agent_name:
        return True
    return span.get("agent_name") in {"", None} and bool(
        conversation_id or conversation_id_contains
    )


def build_agents_check_summary(
    agents: dict[str, Any],
    spans: dict[str, Any],
    *,
    trace_chat_payload: dict[str, Any] | None = None,
    entity: str,
    project: str,
    agent_name: str | None,
    limit: int,
    span_limit: int | None = None,
    conversation_id: str | None = None,
    conversation_id_contains: str | None = None,
) -> dict[str, Any]:
    project_id = f"{entity}/{project}"
    agents_list = agents.get("agents", []) if isinstance(agents.get("agents"), list) else []
    spans_list = spans.get("spans", []) if isinstance(spans.get("spans"), list) else []
    raw_spans = [
        span
        for span in spans_list
        if isinstance(span, dict)
        and agents_span_matches(
            span,
            agent_name=agent_name,
            conversation_id=conversation_id,
            conversation_id_contains=conversation_id_contains,
        )
    ]
    latest_spans_api_order = [summarize_agent_span(span) for span in raw_spans[:limit]]
    latest_trace_id = _latest_trace_id(raw_spans)
    latest_trace_spans = [
        summarize_agent_span(span)
        for span in raw_spans
        if latest_trace_id and span.get("trace_id") == latest_trace_id
    ]
    latest_trace_spans_chronological = sorted(latest_trace_spans, key=_chronological_key)
    timestamp_issues = _timestamp_quality_issues(latest_trace_spans_chronological)
    trace_chat_messages = _chat_messages(trace_chat_payload)
    chat_messages_with_content = [
        message
        for message in trace_chat_messages
        if _chat_user_text(message) or _chat_assistant_text(message)
    ]
    chat_messages_with_input = [
        message for message in trace_chat_messages if _chat_user_text(message)
    ]
    chat_tool_calls = [message for message in trace_chat_messages if _chat_is_tool_call(message)]
    chat_tool_calls_with_content = [
        message for message in chat_tool_calls if _chat_tool_content(message)
    ]
    chat_final_answer_messages = [
        message for message in trace_chat_messages if _chat_is_final_answer(message)
    ]
    trace_order_health = build_agents_trace_order_health(
        latest_trace_spans_chronological,
        timestamp_issues,
        trace_chat_messages=trace_chat_messages,
    )
    message_span_content_count = sum(
        1
        for span in latest_trace_spans_chronological
        if span.get("has_input_messages") or span.get("has_output_messages")
    )
    tool_span_content_count = sum(
        1
        for span in latest_trace_spans_chronological
        if span.get("has_tool_call_arguments") or span.get("has_tool_call_result")
    )
    content_capture_health = {
        "span_count_checked": len(latest_trace_spans_chronological),
        "message_span_count": trace_order_health["message_span_count"],
        "message_spans_with_content": message_span_content_count + len(chat_messages_with_content),
        "message_spans_with_input": trace_order_health["message_spans_with_input"],
        "tool_span_count": trace_order_health["tool_span_count"],
        "tool_spans_with_content": tool_span_content_count + len(chat_tool_calls_with_content),
        "final_answer_span_count": trace_order_health["final_answer_span_count"],
        "spans_with_valid_timestamps": len(latest_trace_spans_chronological) - len(timestamp_issues),
        "spans_with_invalid_timestamps": len(timestamp_issues),
        "trace_timestamp_quality_ok": trace_order_health["timestamp_quality_ok"],
        "trace_order_ok": trace_order_health["trace_order_ok"],
        "trace_user_message_order_ok": trace_order_health["trace_user_message_order_ok"],
        "trace_final_answer_order_ok": trace_order_health["trace_final_answer_order_ok"],
        "trace_order_warnings": trace_order_health.get("order_warnings", []),
        "trace_final_answer_after_tool_warning": (
            "tool_started_after_or_at_final_answer_end"
            in trace_order_health.get("order_warnings", [])
        ),
    }
    if trace_chat_payload is not None:
        content_capture_health.update(
            {
                "chat_message_count": len(trace_chat_messages),
                "chat_messages_with_content": len(chat_messages_with_content),
                "chat_messages_with_input": len(chat_messages_with_input),
                "chat_tool_call_count": len(chat_tool_calls),
                "chat_tool_calls_with_content": len(chat_tool_calls_with_content),
                "chat_final_answer_message_count": len(chat_final_answer_messages),
            }
        )
    query_source = {
        "kind": "wandb_agents_api",
        "api_base_url": AGENTS_API_BASE_URL,
        "agents_endpoint": AGENTS_QUERY_ENDPOINT,
        "spans_endpoint": AGENTS_SPANS_QUERY_ENDPOINT,
        "project_id": project_id,
        "agent_name": agent_name or "",
        "conversation_id": conversation_id or "",
        "conversation_id_contains": conversation_id_contains or "",
        "limit": limit,
        "span_limit": span_limit if isinstance(span_limit, int) else max(limit, limit * 4),
        "agents_count": len(agents_list),
        "spans_count": len(spans_list),
        "matching_span_count": len(raw_spans),
        "latest_trace_span_count": len(latest_trace_spans_chronological),
    }
    if trace_chat_payload is not None:
        query_source.update(
            {
                "trace_chat_endpoint": AGENTS_TRACES_CHAT_ENDPOINT,
                "trace_chat_message_count": len(trace_chat_messages),
            }
        )
    return {
        "diagnostic_schema_version": AGENTS_DIAGNOSTIC_SCHEMA_VERSION,
        "generated_at": time.time(),
        "project_id": project_id,
        "agent_name_filter": agent_name,
        "agents_url": f"https://wandb.ai/{entity}/{project}/weave/agents",
        "query_source": query_source,
        "agents": agents_list,
        "total_count": agents.get("total_count", 0),
        "latest_trace_id": latest_trace_id,
        "latest_trace_spans_chronological": latest_trace_spans_chronological,
        "latest_trace_chat_messages_chronological": [
            summarize_chat_message(message) for message in trace_chat_messages
        ],
        "content_capture_health": content_capture_health,
        "trace_order_health": trace_order_health,
        "latest_spans_api_order": latest_spans_api_order,
    }


def query_trace_chat(
    *,
    env: dict[str, str],
    entity: str,
    project: str,
    trace_id: str,
) -> dict[str, Any]:
    return agents_api_post(
        env,
        AGENTS_TRACES_CHAT_ENDPOINT,
        {
            "project_id": f"{entity}/{project}",
            "trace_id": trace_id,
            "include_feedback": False,
        },
    )


def check_agents(args: argparse.Namespace) -> None:
    env, _, _ = prepare_env(args.env_file)
    project_id = f"{args.entity}/{args.project}"
    filters = {"agent_name": args.agent_name} if args.agent_name else {}
    span_limit = max(args.limit, args.limit * 4)
    agents_payload = {
        "project_id": project_id,
        "filters": filters,
        "limit": args.limit,
        "offset": 0,
    }
    span_conditions: list[dict[str, Any]] = []
    if args.conversation_id:
        span_conditions.append(
            {
                "$eq": [
                    {"$getField": "conversation_id"},
                    {"$literal": args.conversation_id},
                ]
            }
        )
    if args.conversation_id_contains:
        span_conditions.append(
            {
                "$contains": {
                    "input": {"$getField": "conversation_id"},
                    "substr": {"$literal": args.conversation_id_contains},
                    "case_insensitive": False,
                }
            }
        )
    if not span_conditions and args.agent_name:
        span_conditions.append(
            {
                "$eq": [
                    {"$getField": "agent_name"},
                    {"$literal": args.agent_name},
                ]
            }
        )
    span_query = None
    if len(span_conditions) == 1:
        span_query = {"$expr": span_conditions[0]}
    elif span_conditions:
        span_query = {"$expr": {"$and": span_conditions}}
    if args.conversation_id or args.conversation_id_contains:
        span_limit = AGENTS_SPANS_QUERY_MAX_LIMIT
    spans_payload = {
        "project_id": project_id,
        "query": span_query,
        "include_details": True,
        "limit": min(AGENTS_SPANS_QUERY_MAX_LIMIT, span_limit),
        "offset": 0,
    }
    agents = agents_api_post(env, AGENTS_QUERY_ENDPOINT, agents_payload)
    spans = agents_api_post(env, AGENTS_SPANS_QUERY_ENDPOINT, spans_payload)
    raw_spans = [
        span
        for span in (spans.get("spans", []) if isinstance(spans.get("spans"), list) else [])
        if isinstance(span, dict)
        and agents_span_matches(
            span,
            agent_name=args.agent_name,
            conversation_id=args.conversation_id,
            conversation_id_contains=args.conversation_id_contains,
        )
    ]
    latest_trace_id = _latest_trace_id(raw_spans)
    trace_chat_payload = (
        query_trace_chat(
            env=env,
            entity=args.entity,
            project=args.project,
            trace_id=latest_trace_id,
        )
        if latest_trace_id
        else None
    )
    result = build_agents_check_summary(
        agents,
        spans,
        trace_chat_payload=trace_chat_payload,
        entity=args.entity,
        project=args.project,
        agent_name=args.agent_name,
        limit=args.limit,
        span_limit=span_limit,
        conversation_id=args.conversation_id,
        conversation_id_contains=args.conversation_id_contains,
    )
    output = json.dumps(result, ensure_ascii=False, indent=2)
    if args.json:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(output + "\n", encoding="utf-8")
    print(output)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Nejumi Taiwan OpenClaw agent protocol runner")
    subparsers = parser.add_subparsers(required=True)

    preflight_parser = subparsers.add_parser("preflight")
    preflight_parser.add_argument("--openclaw-bin", default="openclaw")
    preflight_parser.add_argument("--nemoclaw-bin", default="nemoclaw")
    preflight_parser.add_argument("--nemoclaw-sandbox")
    preflight_parser.add_argument("--env-file", type=Path, default=DEFAULT_ENV_FILE)
    preflight_parser.set_defaults(func=print_preflight)

    agents_parser = subparsers.add_parser("check-agents")
    agents_parser.add_argument("--entity", default="llm-leaderboard")
    agents_parser.add_argument("--project", default="tc-leaderboard")
    agents_parser.add_argument("--agent-name", default=DEFAULT_NATIVE_WEAVE_AGENT_NAME)
    agents_parser.add_argument("--env-file", type=Path, default=DEFAULT_ENV_FILE)
    agents_parser.add_argument("--limit", type=int, default=10)
    agents_parser.add_argument("--conversation-id")
    agents_parser.add_argument("--conversation-id-contains")
    agents_parser.add_argument("--json", type=Path, help="Write the diagnostic payload to this JSON file.")
    agents_parser.set_defaults(func=check_agents)

    relog_parser = subparsers.add_parser("relog-sidecar")
    relog_parser.add_argument("sidecar_path", type=Path)
    relog_parser.add_argument("--message-file", type=Path)
    relog_parser.add_argument("--openclaw-bin", default="openclaw")
    relog_parser.add_argument("--env-file", type=Path, default=DEFAULT_ENV_FILE)
    relog_parser.add_argument("--model")
    relog_parser.add_argument("--thinking", default="medium")
    relog_parser.add_argument("--session-key")
    relog_parser.add_argument("--weave-sidecar-strict", action="store_true")
    relog_parser.add_argument("--weave-entity", default="llm-leaderboard")
    relog_parser.add_argument("--weave-project", default="tc-leaderboard")
    relog_parser.add_argument("--weave-agent-name", default=DEFAULT_DIAGNOSTIC_WEAVE_AGENT_NAME)
    relog_parser.add_argument("--weave-agent-version", default=PROTOCOL_VERSION)
    relog_parser.add_argument("--weave-agent-description", default="Nejumi 4.5 Taiwan OpenClaw sidecar trace")
    relog_parser.add_argument("--weave-provider-name", default=None)
    relog_parser.add_argument("--weave-sidecar-retries", type=int, default=6)
    relog_parser.add_argument("--weave-sidecar-retry-base-seconds", type=float, default=5.0)
    relog_parser.set_defaults(func=relog_sidecar)

    config_parser = subparsers.add_parser("write-weave-config")
    config_parser.add_argument("--entity", default="llm-leaderboard")
    config_parser.add_argument("--project", default="tc-leaderboard")
    config_parser.add_argument("--agent-name", default=DEFAULT_NATIVE_WEAVE_AGENT_NAME)
    config_parser.add_argument("--agent-version", default=PROTOCOL_VERSION)
    config_parser.add_argument("--agent-description", default="Nejumi 4.5 Taiwan agentic evaluation")
    config_parser.add_argument("--service-name", default="openclaw-agent")
    config_parser.add_argument("--flush-interval-ms", type=int, default=1000)
    config_parser.add_argument("--capture-content", action=argparse.BooleanOptionalAction, default=True)
    config_parser.add_argument(
        "--allow-conversation-access",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    config_parser.add_argument("--output", type=Path, default=Path("configs/openclaw_weave.example.json"))
    config_parser.set_defaults(func=write_weave_config)

    run_parser = subparsers.add_parser("run")
    run_parser.add_argument("--benchmark-id", required=True, choices=["agentic_math", "agentic_swe", "deepswe"])
    run_parser.add_argument("--task-id", required=True)
    run_parser.add_argument("--prompt-file", type=Path, required=True)
    run_parser.add_argument("--tool-policy", type=Path)
    run_parser.add_argument("--deny-tool", action="append", default=[])
    run_parser.add_argument("--deny-argument-pattern", action="append", default=[])
    run_parser.add_argument("--verifier", type=Path)
    run_parser.add_argument("--openclaw-bin", default="openclaw")
    run_parser.add_argument("--nemoclaw-bin", default="nemoclaw")
    run_parser.add_argument(
        "--nemoclaw-sandbox",
        help=(
            "Run OpenClaw inside this NeMoClaw sandbox via "
            "`nemoclaw sandbox exec <sandbox> -- openclaw ...`."
        ),
    )
    run_parser.add_argument(
        "--nemoclaw-workdir",
        default="/sandbox",
        help="Working directory inside the NeMoClaw sandbox for sandbox exec.",
    )
    run_parser.add_argument(
        "--nemoclaw-patched-openclaw-bin-dir",
        default=DEFAULT_NEMOCLAW_PATCHED_OPENCLAW_BIN_DIR,
        help=(
            "Sandbox-local bin directory containing the patched OpenClaw copy. "
            "Prepended to PATH for NeMoClaw runs."
        ),
    )
    run_parser.add_argument(
        "--nemoclaw-extra-path",
        action="append",
        default=[],
        help="Extra sandbox PATH entry to prepend for this OpenClaw invocation.",
    )
    run_parser.add_argument(
        "--nemoclaw-extra-pythonpath",
        action="append",
        default=[],
        help="Extra sandbox PYTHONPATH entry to prepend for this OpenClaw invocation.",
    )
    run_parser.add_argument("--env-file", type=Path, default=DEFAULT_ENV_FILE)
    run_parser.add_argument("--agent", default="main")
    run_parser.add_argument("--profile")
    run_parser.add_argument("--session-key")
    run_parser.add_argument("--model")
    run_parser.add_argument("--thinking", default="medium")
    run_parser.add_argument("--timeout", type=int, default=600)
    run_parser.add_argument(
        "--max-input-tokens",
        type=int,
        default=0,
        help=(
            "Per-call/context input-token cap used by older configs and by "
            "the live estimate fallback. 0 disables this legacy budget."
        ),
    )
    run_parser.add_argument(
        "--max-cumulative-input-tokens",
        type=int,
        default=0,
        help=(
            "Problem-level cumulative provider input-token budget based on "
            "actual model-call usage. 0 disables this exact budget."
        ),
    )
    run_parser.add_argument(
        "--max-cumulative-output-tokens",
        type=int,
        default=0,
        help=(
            "Problem-level cumulative provider output-token budget based on "
            "actual model-call usage. 0 disables this exact budget."
        ),
    )
    run_parser.add_argument(
        "--require-actual-token-usage",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Fail if OpenClaw/model calls do not expose actual token usage.",
    )
    run_parser.add_argument(
        "--max-tool-calls",
        type=int,
        default=0,
        help="Harness-side live session JSONL tool-call budget. 0 disables the budget.",
    )
    run_parser.add_argument(
        "--max-agent-turns",
        type=int,
        default=0,
        help="Harness-side live session JSONL assistant-turn budget. 0 disables the budget.",
    )
    run_parser.add_argument(
        "--max-tool-wall-seconds",
        type=int,
        default=0,
        help=(
            "Configured hard wall-clock cap for each exec tool call. Enforcement "
            "is applied by OpenClaw tools.exec.timeoutSec plus the Nejumi runtime "
            "patch that clamps model-supplied exec timeout values."
        ),
    )
    run_parser.add_argument(
        "--live-session-dir",
        type=Path,
        action="append",
        default=[],
        help=(
            "Additional OpenClaw session JSONL directory to monitor for live "
            "tool-call budget enforcement. Can be supplied multiple times."
        ),
    )
    run_parser.add_argument(
        "--live-sandbox-session-dir",
        action="append",
        default=[],
        help=(
            "OpenClaw session JSONL directory inside the NeMoClaw sandbox to "
            "poll for live tool-call budget enforcement. Can be supplied multiple times."
        ),
    )
    run_parser.add_argument(
        "--final-assistant-idle-salvage-seconds",
        type=float,
        default=60.0,
        help=(
            "If OpenClaw has emitted a final assistant message with no tool call, "
            "the session lock is gone, and the live session has been idle this "
            "many seconds, terminate the stuck OpenClaw process and write a "
            "scoreable sidecar. 0 disables this salvage path."
        ),
    )
    run_parser.add_argument(
        "--final-assistant-shutdown-grace-seconds",
        type=float,
        default=30.0,
        help=(
            "After final-answer idle salvage triggers, send SIGINT and allow this "
            "many seconds for OpenClaw finalizers and telemetry flush before "
            "falling back to forced termination."
        ),
    )
    run_parser.add_argument(
        "--llm-response-idle-timeout-seconds",
        type=float,
        default=0.0,
        help=(
            "If the live session's last message is a tool result and no assistant "
            "response is recorded for this many seconds, terminate OpenClaw as a "
            "provider-side response idle timeout so the caller can retry. "
            "0 disables this watchdog."
        ),
    )
    run_parser.add_argument("--local", action=argparse.BooleanOptionalAction, default=True)
    run_parser.add_argument("--cwd", type=Path, default=Path.cwd())
    run_parser.add_argument("--openclaw-config-path", type=Path)
    run_parser.add_argument(
        "--openclaw-config-source",
        help=(
            "Source config/template identity used by the caller to create "
            "--openclaw-config-path. Recorded in sidecar metadata for cache validation."
        ),
    )
    run_parser.add_argument("--output-dir", type=Path, default=Path("outputs/openclaw_agent"))
    run_parser.add_argument("--dry-run", action="store_true")
    run_parser.add_argument("--allow-failed-preflight", action="store_true")
    run_parser.add_argument(
        "--weave-sidecar",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "Diagnostic fallback only. Production OpenClaw traces should be "
            "captured by the native weave-openclaw plugin."
        ),
    )
    run_parser.add_argument("--weave-sidecar-strict", action="store_true")
    run_parser.add_argument("--weave-entity", default="llm-leaderboard")
    run_parser.add_argument("--weave-project", default="tc-leaderboard")
    run_parser.add_argument("--weave-agent-name", default=DEFAULT_DIAGNOSTIC_WEAVE_AGENT_NAME)
    run_parser.add_argument("--weave-agent-version", default=PROTOCOL_VERSION)
    run_parser.add_argument("--weave-agent-description", default="Nejumi 4.5 Taiwan OpenClaw sidecar trace")
    run_parser.add_argument("--weave-provider-name", default=None)
    run_parser.add_argument("--weave-sidecar-retries", type=int, default=6)
    run_parser.add_argument("--weave-sidecar-retry-base-seconds", type=float, default=5.0)
    run_parser.set_defaults(func=run_agent)

    return parser


def main() -> None:
    args = build_parser().parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
