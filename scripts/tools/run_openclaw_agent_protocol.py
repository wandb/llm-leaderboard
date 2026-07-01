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
import shutil
import subprocess
import sys
import time
import urllib.error
import urllib.request
from datetime import datetime, timezone
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
AGENTS_API_BASE_URL = "https://trace.wandb.ai"
AGENTS_QUERY_ENDPOINT = "/agents/query"
AGENTS_SPANS_QUERY_ENDPOINT = "/agents/spans/query"
AGENTS_DIAGNOSTIC_SCHEMA_VERSION = 1
SANDBOX_LIVE_SESSION_SCAN_TIMEOUT = 10
SANDBOX_LIVE_SESSION_POLL_SECONDS = 5.0
SANDBOX_LIVE_SESSION_SCAN_SCRIPT = r"""
import json
import sys
from pathlib import Path

threshold = float(sys.argv[1])
rows = []


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
        for raw in lines:
            try:
                event = json.loads(raw)
            except json.JSONDecodeError:
                continue
            message = event.get("message") if isinstance(event, dict) else None
            if not isinstance(message, dict) or message.get("role") != "assistant":
                continue
            tool_calls += tool_call_count(message)
        rows.append(
            {
                "path": str(path),
                "mtime": stat.st_mtime,
                "tool_call_count": tool_calls,
            }
        )
rows.sort(key=lambda row: (row["tool_call_count"], row["mtime"]), reverse=True)
print(json.dumps({"ok": True, "sessions": rows}, ensure_ascii=False))
""".strip()
FINAL_ANSWER_MARKERS = (
    "ANSWER:",
    "FINAL ANSWER",
    "Final answer",
    "答案",
    "\\boxed",
    "CANARY_RESULT",
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
    if getattr(args, "openclaw_config_path", None):
        command.extend(["env", f"OPENCLAW_CONFIG_PATH={args.openclaw_config_path}"])
    command.extend(openclaw_command)
    return command


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
    for value in getattr(args, "live_sandbox_session_dir", None) or []:
        path = str(value).strip()
        if not path or "\n" in path or "\r" in path:
            continue
        if path not in seen:
            dirs.append(path)
            seen.add(path)
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
        "python3",
        "-c",
        SANDBOX_LIVE_SESSION_SCAN_SCRIPT,
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
    max_tool_calls = int(getattr(args, "max_tool_calls", 0) or 0)
    session_dirs = [str(path) for path in configured_live_session_dirs(args)]
    sandbox_session_dirs = configured_live_sandbox_session_dirs(args)
    if max_tool_calls <= 0:
        return {
            "enabled": False,
            "max_tool_calls": None,
            "tool_call_count": None,
            "session_file": None,
            "session_source": None,
            "session_dirs": session_dirs,
            "sandbox_session_dirs": sandbox_session_dirs,
            "sandbox_scan": {"enabled": False, "ok": None, "reason": "budget_disabled"},
            "exceeded": False,
        }
    observations: list[dict[str, Any]] = []
    for path in live_session_candidates(args, started_at):
        events = extract_tool_events({"live_session_file": str(path)})
        count = sum(1 for event in events if event.get("type") == "tool_call")
        observations.append(
            {
                "source": "host",
                "tool_call_count": count,
                "session_file": str(path),
            }
        )
    sandbox_scan = scan_nemoclaw_live_sessions(args, started_at, env)
    if sandbox_scan.get("ok") is True:
        for session in sandbox_scan.get("sessions", []):
            count = session.get("tool_call_count")
            path = session.get("path")
            if isinstance(count, (int, float)) and isinstance(path, str) and path:
                observations.append(
                    {
                        "source": "nemoclaw_sandbox",
                        "tool_call_count": int(count),
                        "session_file": path,
                    }
                )
    observations.sort(
        key=lambda item: int(item.get("tool_call_count") or 0),
        reverse=True,
    )
    best = observations[0] if observations else {}
    best_count = int(best.get("tool_call_count") or 0) if best else None
    return {
        "enabled": True,
        "max_tool_calls": max_tool_calls,
        "tool_call_count": best_count,
        "session_file": best.get("session_file") if best else None,
        "session_source": best.get("source") if best else None,
        "session_dirs": session_dirs,
        "sandbox_session_dirs": sandbox_session_dirs,
        "sandbox_scan": sandbox_scan,
        "exceeded": bool(best and best_count is not None and best_count > max_tool_calls),
    }


def terminate_process(process: subprocess.Popen[str], grace_seconds: float = 10.0) -> tuple[str, str]:
    process.terminate()
    try:
        return process.communicate(timeout=grace_seconds)
    except subprocess.TimeoutExpired:
        process.kill()
        return process.communicate()


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
    )
    last_status: dict[str, Any] = live_tool_budget_status(args, started_at, env)
    poll_seconds = (
        SANDBOX_LIVE_SESSION_POLL_SECONDS
        if configured_live_sandbox_session_dirs(args)
        else 1.0
    )
    while True:
        try:
            stdout, stderr = process.communicate(timeout=poll_seconds)
            return (
                subprocess.CompletedProcess(command, process.returncode or 0, stdout=stdout, stderr=stderr),
                {
                    **last_status,
                    "interrupted": False,
                    "reason": None,
                },
            )
        except subprocess.TimeoutExpired:
            last_status = live_tool_budget_status(args, started_at, env)
            if not last_status.get("exceeded"):
                continue
            stdout, stderr = terminate_process(process)
            reason = (
                "Live runtime budget exceeded: "
                f"tool_call_count={last_status.get('tool_call_count')} "
                f"max_tool_calls={last_status.get('max_tool_calls')}"
            )
            stderr = (stderr or "") + "\n" + reason
            return (
                subprocess.CompletedProcess(command, process.returncode or 125, stdout=stdout, stderr=stderr),
                {
                    **last_status,
                    "interrupted": True,
                    "reason": "max_tool_calls_exceeded",
                },
            )


def run_agent(args: argparse.Namespace) -> None:
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

    openclaw_bin = None if getattr(args, "nemoclaw_sandbox", None) else status.get("openclaw_path")
    command = build_openclaw_command(args, message_text, openclaw_bin)
    sidecar = {
        "metadata": metadata,
        "preflight": status,
        "command": command,
        "cwd": str(args.cwd.resolve()),
        "openclaw_config_path": str(args.openclaw_config_path.resolve()) if args.openclaw_config_path else None,
        "tool_policy": policy,
        "started_at": time.time(),
        "dry_run": args.dry_run,
    }

    live_runtime_budget: dict[str, Any] = {}
    env_for_session_copy: dict[str, str] | None = None
    if args.dry_run:
        result = subprocess.CompletedProcess(command, 0, stdout="", stderr="")
    else:
        env, _, _ = prepare_env(args.env_file, args.openclaw_bin)
        env_for_session_copy = env
        if args.openclaw_config_path and not getattr(args, "nemoclaw_sandbox", None):
            env["OPENCLAW_CONFIG_PATH"] = str(args.openclaw_config_path.resolve())
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
        raise SystemExit("Tool policy violation")
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


def extract_agent_meta(sidecar: dict[str, Any]) -> dict[str, Any]:
    meta = extract_openclaw_meta(sidecar)
    agent_meta = meta.get("agentMeta")
    if isinstance(agent_meta, dict):
        merged = dict(agent_meta)
        copied_session_file = sidecar.get("copied_session_file")
        if isinstance(copied_session_file, str) and copied_session_file:
            merged["sandboxSessionFile"] = merged.get("sessionFile")
            merged["sessionFile"] = copied_session_file
        return merged
    live_session_file = sidecar.get("live_session_file")
    if isinstance(live_session_file, str) and live_session_file:
        return {"sessionFile": live_session_file}
    return {}


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
    session_file = agent_meta.get("sessionFile") if isinstance(agent_meta, dict) else None
    if not isinstance(session_file, str) or not session_file.startswith("/"):
        return {"attempted": False, "ok": None, "reason": "missing_sandbox_session_file"}
    if "\n" in session_file or "\r" in session_file:
        return {"attempted": True, "ok": False, "reason": "invalid_session_file_path"}

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
        "bash",
        "-lc",
        'cat "$1"',
        "cat-session",
        session_file,
    ]
    result = subprocess.run(command, text=True, capture_output=True, check=False, env=env)
    if result.returncode != 0:
        return {
            "attempted": True,
            "ok": False,
            "reason": "copy_failed",
            "sandbox_session_file": session_file,
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


def enrich_sidecar_with_tool_events(sidecar: dict[str, Any]) -> dict[str, Any]:
    tool_events = extract_tool_events(sidecar)
    timeline_events = (
        sidecar.get("timeline_events")
        if isinstance(sidecar.get("timeline_events"), list)
        else extract_timeline_events(sidecar)
    )
    sidecar["tool_events"] = tool_events
    sidecar["timeline_events"] = timeline_events
    sidecar["tool_call_count"] = sum(1 for event in tool_events if event.get("type") == "tool_call")
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
        issues.append(
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
    }


def runtime_budget_status(sidecar: dict[str, Any], args: argparse.Namespace) -> dict[str, Any]:
    """Validate post-run usage against benchmark runtime budgets.

    OpenClaw 2026.6.9 does not expose a CLI flag for native live usage
    interruption. This function records the harness contract after sidecar
    enrichment; the subprocess wrapper can additionally terminate live when a
    session JSONL shows too many tool calls.
    """
    max_input_tokens = int(getattr(args, "max_input_tokens", 0) or 0)
    max_tool_calls = int(getattr(args, "max_tool_calls", 0) or 0)
    agent_meta = extract_agent_meta(sidecar)
    usage = agent_meta.get("usage") if isinstance(agent_meta.get("usage"), dict) else {}
    input_tokens = usage.get("input")
    input_tokens = int(input_tokens) if isinstance(input_tokens, (int, float)) else None
    tool_call_count = sidecar.get("tool_call_count")
    tool_call_count = int(tool_call_count) if isinstance(tool_call_count, (int, float)) else None
    live_budget = sidecar.get("live_runtime_budget")
    live_tool_call_count = None
    live_tool_exceeded = False
    if isinstance(live_budget, dict):
        live_count = live_budget.get("tool_call_count")
        if isinstance(live_count, (int, float)):
            live_tool_call_count = int(live_count)
        live_tool_exceeded = (
            live_budget.get("exceeded") is True
            or live_budget.get("reason") == "max_tool_calls_exceeded"
        )
    if live_tool_call_count is not None:
        if tool_call_count is None:
            tool_call_count = live_tool_call_count
        else:
            tool_call_count = max(tool_call_count, live_tool_call_count)

    violations: list[dict[str, Any]] = []
    if max_input_tokens > 0 and input_tokens is not None and input_tokens > max_input_tokens:
        violations.append(
            {
                "type": "max_input_tokens_exceeded",
                "observed": input_tokens,
                "limit": max_input_tokens,
            }
        )
    if max_tool_calls > 0 and (
        (tool_call_count is not None and tool_call_count > max_tool_calls) or live_tool_exceeded
    ):
        violations.append(
            {
                "type": "max_tool_calls_exceeded",
                "observed": tool_call_count,
                "limit": max_tool_calls,
                "source": "live_runtime_budget" if live_tool_exceeded else "openclaw_session_jsonl",
            }
        )
    return {
        "ok": not violations,
        "enforced": bool(max_input_tokens > 0 or max_tool_calls > 0),
        "limits": {
            "max_input_tokens": max_input_tokens or None,
            "max_tool_calls": max_tool_calls or None,
        },
        "observed": {
            "input_tokens": input_tokens,
            "tool_call_count": tool_call_count,
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


def _span_output_text(span: dict[str, Any]) -> str:
    return _jsonish_text(span.get("output_messages"))


def _is_final_answer_span(span: dict[str, Any]) -> bool:
    if span.get("operation_name") not in {"chat", "invoke_agent"}:
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
) -> dict[str, Any]:
    message_spans = [span for span in spans if span.get("operation_name") in {"chat", "invoke_agent"}]
    message_spans_with_input = [span for span in message_spans if span.get("has_input_messages")]
    tool_spans = [span for span in spans if span.get("operation_name") == "execute_tool"]
    final_answer_spans = [span for span in message_spans if span.get("has_final_answer_marker")]
    health: dict[str, Any] = {
        "timestamp_quality_ok": not timestamp_issues,
        "timestamp_issue_count": len(timestamp_issues),
        "timestamp_issues": timestamp_issues[:10],
        "message_span_count": len(message_spans),
        "message_spans_with_input": len(message_spans_with_input),
        "tool_span_count": len(tool_spans),
        "final_answer_span_count": len(final_answer_spans),
        "trace_order_ok": True,
        "trace_user_message_order_ok": True,
        "trace_final_answer_order_ok": True,
        "order_issues": [],
        "first_message_started_at": _span_time_label(message_spans, "started_at", minimum=True),
        "first_input_message_started_at": _span_time_label(
            message_spans_with_input,
            "started_at",
            minimum=True,
        ),
        "first_tool_started_at": _span_time_label(tool_spans, "started_at", minimum=True),
        "last_tool_started_at": _span_time_label(tool_spans, "started_at", minimum=False),
        "first_final_answer_ended_at": _span_time_label(final_answer_spans, "ended_at", minimum=True),
    }
    order_issues = health["order_issues"]
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
        if not message_spans_with_input:
            health["trace_user_message_order_ok"] = False
            order_issues.append("tool_present_without_visible_user_input")
        else:
            first_input_message = min(_span_start(span) for span in message_spans_with_input)
            first_tool = min(_span_start(span) for span in tool_spans)
            if first_input_message is None or first_tool is None or first_tool <= first_input_message:
                health["trace_user_message_order_ok"] = False
                order_issues.append("tool_started_before_or_at_visible_user_input")

    if final_answer_spans and tool_spans:
        first_final_answer_end = min(_span_end(span) for span in final_answer_spans)
        last_tool_start = max(_span_start(span) for span in tool_spans)
        if first_final_answer_end is None or last_tool_start is None or last_tool_start >= first_final_answer_end:
            health["trace_final_answer_order_ok"] = False
            order_issues.append("tool_started_after_or_at_final_answer_end")

    return health


def build_agents_check_summary(
    agents: dict[str, Any],
    spans: dict[str, Any],
    *,
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
        and (not agent_name or span.get("agent_name") == agent_name)
        and (not conversation_id or span.get("conversation_id") == conversation_id)
        and (
            not conversation_id_contains
            or (
                isinstance(span.get("conversation_id"), str)
                and conversation_id_contains in span.get("conversation_id")
            )
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
    trace_order_health = build_agents_trace_order_health(
        latest_trace_spans_chronological,
        timestamp_issues,
    )
    content_capture_health = {
        "span_count_checked": len(latest_trace_spans_chronological),
        "message_span_count": trace_order_health["message_span_count"],
        "message_spans_with_content": sum(
            1
            for span in latest_trace_spans_chronological
            if span.get("has_input_messages") or span.get("has_output_messages")
        ),
        "message_spans_with_input": trace_order_health["message_spans_with_input"],
        "tool_span_count": trace_order_health["tool_span_count"],
        "tool_spans_with_content": sum(
            1
            for span in latest_trace_spans_chronological
            if span.get("has_tool_call_arguments") or span.get("has_tool_call_result")
        ),
        "final_answer_span_count": trace_order_health["final_answer_span_count"],
        "spans_with_valid_timestamps": len(latest_trace_spans_chronological) - len(timestamp_issues),
        "spans_with_invalid_timestamps": len(timestamp_issues),
        "trace_timestamp_quality_ok": trace_order_health["timestamp_quality_ok"],
        "trace_order_ok": trace_order_health["trace_order_ok"],
        "trace_user_message_order_ok": trace_order_health["trace_user_message_order_ok"],
        "trace_final_answer_order_ok": trace_order_health["trace_final_answer_order_ok"],
    }
    return {
        "diagnostic_schema_version": AGENTS_DIAGNOSTIC_SCHEMA_VERSION,
        "generated_at": time.time(),
        "project_id": project_id,
        "agent_name_filter": agent_name,
        "agents_url": f"https://wandb.ai/{entity}/{project}/weave/agents",
        "query_source": {
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
        },
        "agents": agents_list,
        "total_count": agents.get("total_count", 0),
        "latest_trace_id": latest_trace_id,
        "latest_trace_spans_chronological": latest_trace_spans_chronological,
        "content_capture_health": content_capture_health,
        "trace_order_health": trace_order_health,
        "latest_spans_api_order": latest_spans_api_order,
    }


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
    spans_payload = dict(agents_payload)
    spans_payload["limit"] = span_limit
    agents = agents_api_post(env, AGENTS_QUERY_ENDPOINT, agents_payload)
    spans = agents_api_post(env, AGENTS_SPANS_QUERY_ENDPOINT, spans_payload)
    result = build_agents_check_summary(
        agents,
        spans,
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
    run_parser.add_argument("--benchmark-id", required=True, choices=["agentic_math", "agentic_swe"])
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
        help="Harness-side post-run input token budget. 0 disables the budget.",
    )
    run_parser.add_argument(
        "--max-tool-calls",
        type=int,
        default=0,
        help="Harness-side live session JSONL tool-call budget. 0 disables the budget.",
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
