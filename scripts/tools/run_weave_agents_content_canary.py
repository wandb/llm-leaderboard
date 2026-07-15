#!/usr/bin/env python3
"""Run a fresh OpenClaw canary and verify W&B Weave Agents content capture.

Default mode is prepare-only and does not call a model. Use --execute with an
explicit --model when you intentionally want to create a fresh live trace.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from external_action_approval_checks import (
    approval_results as normalized_approval_results,
    extract_paid_api_approval,
    validate_approval_results,
)

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_ENV_FILE = REPO_ROOT / ".env"
DEFAULT_AGENT_NAME = "nejumi-taiwan-openclaw"
DEFAULT_ENTITY = "llm-leaderboard"
DEFAULT_PROJECT = "tc-leaderboard"
DEFAULT_OUTPUT_DIR = Path("outputs/weave_agents_content_canary")
DEFAULT_NEMOCLAW_OPENCLAW_CONFIG_PATH = "/sandbox/.openclaw/openclaw.json"
DEFAULT_NEMOCLAW_OPENCLAW_CONFIG_SOURCE_TRACE_TEXT = (
    f"openclaw_config_source: {DEFAULT_NEMOCLAW_OPENCLAW_CONFIG_PATH}"
)
NETWORK_DENY_PATTERNS = ("curl", "wget", "requests", "urllib", "httpx", r"https?://")
CANARY_GATE_RUNNER = REPO_ROOT / "scripts" / "tools" / "verify_weave_agents_content_canary_result.py"


@dataclass(frozen=True)
class CanaryPaths:
    canary_id: str
    task_id: str
    prompt_file: Path
    expected_sidecar: Path
    plan_file: Path
    command_result_file: Path
    gate_result_file: Path
    verifier_dir: Path
    agents_diagnostic_file: Path


def utc_canary_id() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--execute", action="store_true", help="Actually run OpenClaw and call the selected model.")
    parser.add_argument("--model", help="OpenClaw model id. Required with --execute.")
    parser.add_argument("--thinking", default="low")
    parser.add_argument("--timeout", type=int, default=180)
    parser.add_argument("--canary-id", help="Stable canary id. Defaults to current UTC timestamp.")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--entity", default=os.environ.get("WANDB_ENTITY", DEFAULT_ENTITY))
    parser.add_argument("--project", default=os.environ.get("WANDB_PROJECT", DEFAULT_PROJECT))
    parser.add_argument("--agent-name", default=DEFAULT_AGENT_NAME)
    parser.add_argument("--env-file", type=Path, default=DEFAULT_ENV_FILE)
    parser.add_argument("--openclaw-bin", default="openclaw")
    parser.add_argument("--openclaw-config-path", type=Path)
    parser.add_argument("--cwd", type=Path, default=REPO_ROOT)
    parser.add_argument("--agent", default="main")
    parser.add_argument("--profile")
    parser.add_argument("--local", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--nemoclaw-bin", default="nemoclaw")
    parser.add_argument("--nemoclaw-sandbox")
    parser.add_argument("--nemoclaw-workdir", default="/sandbox")
    parser.add_argument(
        "--nemoclaw-openclaw-config-path",
        default=DEFAULT_NEMOCLAW_OPENCLAW_CONFIG_PATH,
        help=(
            "Sandbox-local OpenClaw config path checked immediately before live "
            "NeMoClaw canary execution."
        ),
    )
    parser.add_argument("--allow-failed-preflight", action="store_true")
    parser.add_argument("--verify-attempts", type=int, default=8)
    parser.add_argument("--verify-sleep-seconds", type=float, default=10.0)
    parser.add_argument("--agents-limit", type=int, default=30)
    parser.add_argument("--no-require-tool", action="store_true")
    parser.add_argument("--no-require-usage", action="store_true")
    parser.add_argument(
        "--external-action-approval-report-json",
        type=Path,
        help=(
            "Source-bound verifier report produced by "
            "verify_external_action_approval_packet.py. Required with --execute."
        ),
    )
    parser.add_argument(
        "--external-action-approval-source-packet-json",
        type=Path,
        help=(
            "Source external_action_approval_packet.json that the verifier report "
            "must be bound to. Required with --execute so a stale approval report "
            "from another release bundle cannot authorize a live canary."
        ),
    )
    return parser.parse_args(argv)


def load_json_object(path: Path) -> tuple[dict[str, Any] | None, str | None]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        return None, str(exc)
    if not isinstance(payload, dict):
        return None, f"{path} must contain a JSON object"
    return payload, None


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def command_sha256(command: list[str]) -> str:
    payload = json.dumps(command, ensure_ascii=False, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def build_external_action_approval_record(
    approval_report_path: Path | None,
    *,
    required_before_external_action: bool,
    expected_source_packet_path: Path | None = None,
) -> dict[str, Any]:
    record: dict[str, Any] = {
        "required_before_external_action": bool(required_before_external_action),
        "path": str(approval_report_path) if approval_report_path else "",
        "present": False,
        "valid": False,
        "sha256": "",
        "schema_version": None,
        "status": "",
        "generated_at": "",
        "approval_packet_json": "",
        "external_action_checklist_sha256": "",
        "required_approval_count": None,
        "granted_approval_count": None,
        "all_required_approvals_granted": False,
        "source_binding": {},
        "expected_source_packet_json": str(expected_source_packet_path)
        if expected_source_packet_path
        else "",
        "expected_source_packet_sha256": "",
        "source_packet_path_matches_expected": False,
        "source_packet_sha256_matches_expected": False,
        "will_execute_external_actions": None,
        "approval_results": [],
        "approval_results_validation": {},
        "paid_api_approval": {},
        "errors": [],
    }
    if required_before_external_action and expected_source_packet_path is None:
        record["errors"].append(
            "external-action approval source packet is required before live canary execution"
        )
    if expected_source_packet_path is not None:
        if not expected_source_packet_path.exists():
            record["errors"].append(
                "external-action approval source packet does not exist: "
                f"{expected_source_packet_path}"
            )
        else:
            try:
                record["expected_source_packet_sha256"] = sha256_file(
                    expected_source_packet_path
                )
            except OSError as exc:
                record["errors"].append(
                    f"external-action approval source packet sha256 failed: {exc}"
                )
    if approval_report_path is None:
        if required_before_external_action:
            record["errors"].append(
                "external-action approval verifier report is required before live canary execution"
            )
        return record

    payload, error = load_json_object(approval_report_path)
    if payload is None:
        record["errors"].append(error or "external-action approval report JSON could not be loaded")
        return record

    record["present"] = True
    try:
        record["sha256"] = sha256_file(approval_report_path)
    except OSError as exc:
        record["errors"].append(f"external-action approval report sha256 failed: {exc}")

    source_binding = (
        payload.get("source_binding")
        if isinstance(payload.get("source_binding"), dict)
        else {}
    )
    source_errors = source_binding.get("errors")
    if not isinstance(source_errors, list):
        source_errors = ["source_binding.errors is missing or not a list"]

    required_count = payload.get("required_approval_count")
    granted_count = payload.get("granted_approval_count")
    approval_results = normalized_approval_results(payload)
    approval_results_validation = validate_approval_results(payload)
    record.update(
        {
            "schema_version": payload.get("schema_version"),
            "status": str(payload.get("status") or ""),
            "generated_at": payload.get("generated_at") or "",
            "approval_packet_json": str(payload.get("approval_packet_json") or ""),
            "external_action_checklist_sha256": str(
                payload.get("external_action_checklist_sha256") or ""
            ),
            "required_approval_count": required_count,
            "granted_approval_count": granted_count,
            "all_required_approvals_granted": bool(
                payload.get("all_required_approvals_granted")
            ),
            "source_binding": source_binding,
            "will_execute_external_actions": payload.get("will_execute_external_actions"),
            "approval_results": approval_results,
            "approval_results_validation": approval_results_validation,
            "paid_api_approval": extract_paid_api_approval(payload),
        }
    )
    record["errors"].extend(
        str(error)
        for error in approval_results_validation.get("errors", [])
        if isinstance(error, str)
    )

    if payload.get("schema_version") != 1:
        record["errors"].append("schema_version must be 1")
    if payload.get("ok") is not True:
        record["errors"].append("ok must be true")
    if payload.get("status") != "approved":
        record["errors"].append("status must be approved")
    if not isinstance(required_count, int) or required_count <= 0:
        record["errors"].append("required_approval_count must be a positive integer")
    if not isinstance(granted_count, int):
        record["errors"].append("granted_approval_count must be an integer")
    elif isinstance(required_count, int) and granted_count != required_count:
        record["errors"].append("granted_approval_count must equal required_approval_count")
    if payload.get("all_required_approvals_granted") is not True:
        record["errors"].append("all_required_approvals_granted must be true")
    if payload.get("will_execute_external_actions") is not False:
        record["errors"].append("will_execute_external_actions must be false")
    if source_binding.get("bound") is not True:
        record["errors"].append("source_binding.bound must be true")
    if source_binding.get("source_packet_readable") is not True:
        record["errors"].append("source_binding.source_packet_readable must be true")
    source_sha = str(source_binding.get("source_approval_packet_sha256") or "")
    if len(source_sha) != 64 or any(char not in "0123456789abcdef" for char in source_sha):
        record["errors"].append(
            "source_binding.source_approval_packet_sha256 must be 64 lowercase hex characters"
        )
    if expected_source_packet_path is not None:
        source_packet_value = str(source_binding.get("source_packet_json") or "")
        if not source_packet_value:
            record["errors"].append("source_binding.source_packet_json is missing")
        else:
            source_packet_path = Path(source_packet_value)
            if not source_packet_path.is_absolute():
                source_packet_path = (approval_report_path.parent / source_packet_path).resolve()
            expected_resolved = expected_source_packet_path.resolve()
            source_resolved = source_packet_path.resolve()
            record["source_packet_path_matches_expected"] = source_resolved == expected_resolved
            if source_resolved != expected_resolved:
                record["errors"].append(
                    "source_binding.source_packet_json does not match "
                    "--external-action-approval-source-packet-json"
                )
        expected_sha = str(record.get("expected_source_packet_sha256") or "")
        if expected_sha and source_sha:
            record["source_packet_sha256_matches_expected"] = source_sha == expected_sha
            if source_sha != expected_sha:
                record["errors"].append(
                    "source_binding.source_approval_packet_sha256 does not match "
                    "--external-action-approval-source-packet-json"
                )
    if source_errors:
        record["errors"].extend(f"source_binding: {item}" for item in source_errors)

    record["valid"] = not record["errors"]
    return record


def canary_prompt(canary_id: str) -> str:
    return f"""# W&B Agents Content Capture Canary

Canary ID: {canary_id}

Use one available code or Python tool exactly once to compute `7 * 13`.
After the tool result is available, your final response must be exactly this single line and nothing else:

CANARY_RESULT {canary_id} 91

Do not use web search, network access, package installation, or external files.
"""


def canary_paths(output_dir: Path, canary_id: str) -> CanaryPaths:
    safe_id = "".join(char if char.isalnum() else "_" for char in canary_id)
    task_id = f"weave_agents_content_canary_{safe_id}"
    prompt_file = output_dir / "prompts" / f"{task_id}.md"
    expected_sidecar = output_dir / "agentic_math" / task_id / "openclaw_result.json"
    plan_file = output_dir / "plans" / f"{task_id}.json"
    command_result_file = output_dir / "plans" / f"{task_id}.command_result.json"
    gate_result_file = output_dir / "plans" / f"{task_id}.gate.json"
    verifier_dir = output_dir / "verifier" / task_id
    agents_diagnostic_file = output_dir / "agents_diagnostics" / f"{task_id}.agents.json"
    return CanaryPaths(
        canary_id=canary_id,
        task_id=task_id,
        prompt_file=prompt_file,
        expected_sidecar=expected_sidecar,
        plan_file=plan_file,
        command_result_file=command_result_file,
        gate_result_file=gate_result_file,
        verifier_dir=verifier_dir,
        agents_diagnostic_file=agents_diagnostic_file,
    )


def openclaw_config_source_trace_text(config_path: str | os.PathLike[str] | None) -> str:
    value = str(config_path or "").strip()
    if not value:
        return ""
    return f"openclaw_config_source: {value}"


def required_trace_texts(args: argparse.Namespace, paths: CanaryPaths) -> list[str]:
    texts = [
        paths.canary_id,
        f"CANARY_RESULT {paths.canary_id} 91",
    ]
    if args.nemoclaw_sandbox:
        config_source = openclaw_config_source_trace_text(args.nemoclaw_openclaw_config_path)
        if config_source:
            texts.append(config_source)
    return texts


def build_run_command(args: argparse.Namespace, paths: CanaryPaths) -> list[str]:
    command = [
        sys.executable,
        str(REPO_ROOT / "scripts" / "tools" / "run_openclaw_agent_protocol.py"),
        "run",
        "--benchmark-id",
        "agentic_math",
        "--task-id",
        paths.task_id,
        "--prompt-file",
        str(paths.prompt_file),
        "--output-dir",
        str(args.output_dir),
        "--session-key",
        paths.task_id,
        "--agent",
        args.agent,
        "--thinking",
        args.thinking,
        "--timeout",
        str(args.timeout),
        "--cwd",
        str(args.cwd),
        "--env-file",
        str(args.env_file),
        "--openclaw-bin",
        args.openclaw_bin,
    ]
    if args.model:
        command.extend(["--model", args.model])
    if args.profile:
        command.extend(["--profile", args.profile])
    if args.openclaw_config_path:
        command.extend(["--openclaw-config-path", str(args.openclaw_config_path)])
    if not getattr(args, "local", True):
        command.append("--no-local")
    if args.nemoclaw_sandbox:
        config_source = str(args.nemoclaw_openclaw_config_path or "").strip()
        command.extend(
            [
                "--nemoclaw-bin",
                args.nemoclaw_bin,
                "--nemoclaw-sandbox",
                args.nemoclaw_sandbox,
                "--nemoclaw-workdir",
                args.nemoclaw_workdir,
            ]
        )
        if config_source:
            command.extend(["--openclaw-config-source", config_source])
        command.extend(["--live-sandbox-session-dir", f"/sandbox/.openclaw/agents/{args.agent}/sessions"])
    if args.allow_failed_preflight:
        command.append("--allow-failed-preflight")
    for pattern in NETWORK_DENY_PATTERNS:
        command.extend(["--deny-argument-pattern", pattern])
    return command


def build_nemoclaw_metadata(args: argparse.Namespace) -> dict[str, Any]:
    enabled = bool(args.nemoclaw_sandbox)
    return {
        "required": enabled,
        "enabled": enabled,
        "bin": args.nemoclaw_bin if enabled else "",
        "sandbox": args.nemoclaw_sandbox or "",
        "workdir": args.nemoclaw_workdir if enabled else "",
    }


def openclaw_provider_and_model(openclaw_model: str | None) -> tuple[str, str]:
    provider, sep, model_id = str(openclaw_model or "").strip().partition("/")
    return provider, model_id if sep else ""


def _preflight_check(name: str, ok: bool, detail: str) -> dict[str, Any]:
    return {"name": name, "ok": bool(ok), "detail": detail}


def build_nemoclaw_openclaw_config_preflight(
    args: argparse.Namespace,
    *,
    run: bool,
) -> dict[str, Any]:
    required = bool(args.execute and args.nemoclaw_sandbox)
    config_path = str(args.nemoclaw_openclaw_config_path or "").strip()
    record: dict[str, Any] = {
        "required_before_openclaw": required,
        "ran": False,
        "ok": not required,
        "model": args.model or "",
        "provider": "",
        "model_id": "",
        "config_path": config_path,
        "command": [],
        "returncode": None,
        "checks": [],
        "errors": [],
    }
    if not required:
        return record

    provider_id, model_id = openclaw_provider_and_model(args.model)
    record["provider"] = provider_id
    record["model_id"] = model_id
    if not provider_id or not model_id:
        record["errors"].append("model must be provider/model for NeMoClaw OpenClaw config preflight")
    if not config_path:
        record["errors"].append("nemoclaw OpenClaw config path is required")
    if record["errors"] or not run:
        record["ok"] = False
        return record

    command = [
        args.nemoclaw_bin,
        "sandbox",
        "exec",
        args.nemoclaw_sandbox,
        "--no-tty",
        "--timeout",
        "30",
        "--",
        "cat",
        config_path,
    ]
    record["command"] = command
    completed = run_subprocess(command, cwd=REPO_ROOT)
    record["ran"] = True
    record["returncode"] = completed.returncode
    readable = completed.returncode == 0
    record["checks"].append(
        _preflight_check(
            f"NeMoClaw sandbox OpenClaw config is readable: {config_path}",
            readable,
            f"bytes={len(completed.stdout.encode('utf-8'))}"
            if readable
            else (completed.stderr or completed.stdout)[-1000:],
        )
    )
    if not readable:
        record["errors"].append("NeMoClaw sandbox OpenClaw config is not readable")
        record["ok"] = False
        return record

    try:
        data = json.loads(completed.stdout)
    except json.JSONDecodeError as exc:
        record["checks"].append(
            _preflight_check(
                f"NeMoClaw sandbox OpenClaw config parses: {config_path}",
                False,
                str(exc),
            )
        )
        record["errors"].append("NeMoClaw sandbox OpenClaw config is not valid JSON")
        record["ok"] = False
        return record
    if not isinstance(data, dict):
        record["checks"].append(
            _preflight_check(
                f"NeMoClaw sandbox OpenClaw config parses: {config_path}",
                False,
                "JSON output is not an object",
            )
        )
        record["errors"].append("NeMoClaw sandbox OpenClaw config JSON is not an object")
        record["ok"] = False
        return record

    provider = data.get("models", {}).get("providers", {}).get(provider_id, {})
    models = provider.get("models") if isinstance(provider, dict) else None
    model_ids = [model.get("id") for model in models or [] if isinstance(model, dict)]
    entries = data.get("plugins", {}).get("entries", {})
    weave = entries.get("weave") if isinstance(entries, dict) else {}
    checks = [
        _preflight_check(
            f"NeMoClaw sandbox OpenClaw {provider_id} provider exists",
            bool(provider),
            "present" if provider else "missing",
        ),
        _preflight_check(
            f"NeMoClaw sandbox OpenClaw model is registered: {args.model}",
            model_id in model_ids,
            json.dumps(model_ids, ensure_ascii=False),
        ),
        _preflight_check(
            "NeMoClaw sandbox OpenClaw Weave plugin is enabled",
            bool(weave and weave.get("enabled")),
            str(weave.get("enabled") if isinstance(weave, dict) else None),
        ),
    ]
    record["checks"].extend(checks)
    record["errors"] = [check["name"] for check in record["checks"] if not check["ok"]]
    record["ok"] = not record["errors"]
    return record


def request_model_aliases(model_id: str | None) -> list[str]:
    value = str(model_id or "").strip()
    if not value:
        return []
    aliases = [value]
    if value.startswith("openai-direct/"):
        aliases.append(value.removeprefix("openai-direct/"))
    if "/" in value:
        provider, remainder = value.split("/", 1)
        if (provider.endswith("-direct") or provider == "wandb-inference") and remainder:
            aliases.append(remainder)
        aliases.append(value.rsplit("/", 1)[-1])
    return list(dict.fromkeys(alias for alias in aliases if alias))


def _sidecar_get_agent_meta(sidecar: dict[str, Any]) -> dict[str, Any]:
    stdout_json = sidecar.get("stdout_json")
    candidates: list[dict[str, Any]] = []
    if isinstance(stdout_json, dict):
        candidates.append(stdout_json)
        meta = stdout_json.get("meta")
        if isinstance(meta, dict):
            candidates.append(meta)
        result = stdout_json.get("result")
        if isinstance(result, dict):
            candidates.append(result)
            result_meta = result.get("meta")
            if isinstance(result_meta, dict):
                candidates.append(result_meta)
    for candidate in candidates:
        agent_meta = candidate.get("agentMeta")
        if isinstance(agent_meta, dict):
            return agent_meta
    return {}


def _sidecar_find_session_key(value: Any) -> str | None:
    if isinstance(value, dict):
        session_key = value.get("sessionKey")
        if isinstance(session_key, str) and session_key.strip():
            return session_key.strip()
        for nested in value.values():
            found = _sidecar_find_session_key(nested)
            if found:
                return found
    if isinstance(value, list):
        for nested in value:
            found = _sidecar_find_session_key(nested)
            if found:
                return found
    return None


def extract_conversation_id(sidecar_path: Path) -> str | None:
    if not sidecar_path.exists():
        return None
    try:
        sidecar = json.loads(sidecar_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    session_key = _sidecar_find_session_key(sidecar)
    if session_key:
        return session_key.lower()
    agent_meta = _sidecar_get_agent_meta(sidecar)
    for key in ("sessionKey", "conversationId", "sessionId"):
        value = agent_meta.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    return None


def build_verify_command(
    args: argparse.Namespace,
    paths: CanaryPaths,
    json_path: Path,
    *,
    conversation_id: str | None = None,
) -> list[str]:
    command = [
        sys.executable,
        str(REPO_ROOT / "scripts" / "tools" / "verify_taiwan_weave_agents.py"),
        "--entity",
        args.entity,
        "--project",
        args.project,
        "--agent-name",
        args.agent_name,
        "--limit",
        str(args.agents_limit),
        "--min-trace-spans",
        "1",
        "--min-message-spans",
        "1",
        "--env-file",
        str(args.env_file),
        "--json",
        str(json_path),
    ]
    if conversation_id:
        command.extend(["--conversation-id", conversation_id])
    else:
        command.extend(["--conversation-id-contains", paths.task_id.lower()])
    if not args.no_require_tool:
        command.extend(["--require-tool-span", "--require-tool-content"])
    if not args.no_require_usage:
        command.append("--require-usage")
    for model in request_model_aliases(args.model):
        command.extend(["--expected-request-model", model])
    for required_text in required_trace_texts(args, paths):
        command.extend(["--require-text", required_text])
    return command


def build_agents_diagnostic_command(
    args: argparse.Namespace,
    paths: CanaryPaths,
    *,
    conversation_id: str | None = None,
) -> list[str]:
    command = [
        sys.executable,
        str(REPO_ROOT / "scripts" / "tools" / "run_openclaw_agent_protocol.py"),
        "check-agents",
        "--entity",
        args.entity,
        "--project",
        args.project,
        "--agent-name",
        args.agent_name,
        "--limit",
        str(args.agents_limit),
        "--env-file",
        str(args.env_file),
        "--json",
        str(paths.agents_diagnostic_file),
    ]
    if conversation_id:
        command.extend(["--conversation-id", conversation_id])
    else:
        command.extend(["--conversation-id-contains", paths.task_id.lower()])
    return command


def write_canary_files(
    args: argparse.Namespace,
    paths: CanaryPaths,
    run_command: list[str],
    *,
    external_action_approval: dict[str, Any],
    nemoclaw_openclaw_config_preflight: dict[str, Any],
) -> dict[str, Any]:
    paths.prompt_file.parent.mkdir(parents=True, exist_ok=True)
    paths.plan_file.parent.mkdir(parents=True, exist_ok=True)
    paths.verifier_dir.mkdir(parents=True, exist_ok=True)
    paths.agents_diagnostic_file.parent.mkdir(parents=True, exist_ok=True)
    paths.prompt_file.write_text(canary_prompt(paths.canary_id), encoding="utf-8")
    plan = {
        "canary_id": paths.canary_id,
        "task_id": paths.task_id,
        "will_call_paid_model_api": bool(args.execute),
        "will_execute_external_actions": bool(args.execute),
        "model": args.model,
        "thinking": args.thinking,
        "timeout": args.timeout,
        "prompt_file": str(paths.prompt_file),
        "expected_sidecar": str(paths.expected_sidecar),
        "agents_diagnostic_file": str(paths.agents_diagnostic_file),
        "gate_result_file": str(paths.gate_result_file),
        "run_command": run_command,
        "run_command_sha256": command_sha256(run_command),
        "verify_attempts": args.verify_attempts,
        "verify_sleep_seconds": args.verify_sleep_seconds,
        "verification_requirements": {
            "require_content": True,
            "require_tool_span": not bool(args.no_require_tool),
            "require_tool_content": not bool(args.no_require_tool),
            "require_usage": not bool(args.no_require_usage),
            "expected_request_models": request_model_aliases(args.model),
            "required_texts": required_trace_texts(args, paths),
        },
        "agent_name": args.agent_name,
        "entity": args.entity,
        "project": args.project,
        "nemoclaw": build_nemoclaw_metadata(args),
        "nemoclaw_openclaw_config_preflight": nemoclaw_openclaw_config_preflight,
        "external_action_approval": external_action_approval,
    }
    paths.plan_file.write_text(json.dumps(plan, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return plan


def run_subprocess(command: list[str], *, cwd: Path) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        command,
        cwd=cwd,
        text=True,
        capture_output=True,
        check=False,
    )


def write_command_result(paths: CanaryPaths, payload: dict[str, Any]) -> dict[str, Any]:
    paths.command_result_file.parent.mkdir(parents=True, exist_ok=True)
    paths.command_result_file.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return payload


def command_result_identity(plan: dict[str, Any], paths: CanaryPaths) -> dict[str, Any]:
    return {
        "canary_id": plan.get("canary_id") or paths.canary_id,
        "task_id": plan.get("task_id") or paths.task_id,
        "model": plan.get("model"),
        "thinking": plan.get("thinking"),
        "plan_file": str(paths.plan_file),
        "prompt_file": plan.get("prompt_file") or str(paths.prompt_file),
        "expected_sidecar": plan.get("expected_sidecar") or str(paths.expected_sidecar),
        "run_command": plan.get("run_command") if isinstance(plan.get("run_command"), list) else [],
        "run_command_sha256": plan.get("run_command_sha256") or "",
    }


def write_blocked_command_result(
    paths: CanaryPaths,
    *,
    plan: dict[str, Any],
    failure_kind: str,
    failure_detail: str,
    payload: dict[str, Any],
) -> dict[str, Any]:
    now = time.time()
    command_result = {
        "ok": False,
        "returncode": 2,
        "started_at": None,
        "ended_at": now,
        "blocked_before_openclaw": True,
        "paid_api_attempted": False,
        "will_execute_external_actions": True,
        "stdout_tail": "",
        "stderr_tail": "",
        "failure": {
            "kind": failure_kind,
            "detail": failure_detail,
        },
        **command_result_identity(plan, paths),
        **payload,
    }
    return write_command_result(paths, command_result)


def write_gate_result(
    paths: CanaryPaths,
    *,
    verifier_json: str | None = None,
    agents_diagnostic_json: str | None = None,
) -> dict[str, Any]:
    command = [
        sys.executable,
        str(CANARY_GATE_RUNNER),
        "--plan-file",
        str(paths.plan_file),
        "--command-result-file",
        str(paths.command_result_file),
        "--json",
        str(paths.gate_result_file),
    ]
    if verifier_json:
        command.extend(["--verifier-json", verifier_json])
    if agents_diagnostic_json:
        command.extend(["--agents-diagnostic-json", agents_diagnostic_json])
    completed = run_subprocess(command, cwd=REPO_ROOT)
    try:
        payload = json.loads(completed.stdout)
    except json.JSONDecodeError:
        payload = {
            "ok": False,
            "gate": "weave_agents_content_canary",
            "status": "gate_summary_failed",
            "detail": "gate summarizer stdout was not JSON",
            "command": command,
            "returncode": completed.returncode,
            "stdout_tail": completed.stdout[-2000:],
            "stderr_tail": completed.stderr[-2000:],
        }
        paths.gate_result_file.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        return payload
    if completed.returncode != 0:
        payload["status"] = payload.get("status") or "gate_summary_failed"
        payload["gate_summary_returncode"] = completed.returncode
        payload["gate_summary_stderr_tail"] = completed.stderr[-2000:]
        paths.gate_result_file.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
    return payload


def classify_openclaw_failure(sidecar_path: Path, stderr_tail: str) -> dict[str, str] | None:
    text = stderr_tail
    if sidecar_path.exists():
        try:
            sidecar = json.loads(sidecar_path.read_text(encoding="utf-8"))
            text = f"{sidecar.get('stderr') or ''}\n{text}"
        except (OSError, json.JSONDecodeError):
            pass
    lowered = text.lower()
    if "insufficient_quota" in lowered or "exceeded your current quota" in lowered:
        return {
            "kind": "provider_quota",
            "detail": "provider returned insufficient_quota before a scoreable canary trace was produced",
        }
    if "unknown model" in lowered or "model_not_found" in lowered:
        return {
            "kind": "model_not_found",
            "detail": "OpenClaw could not resolve the requested model id",
        }
    if "authentication" in lowered or "unauthorized" in lowered or "401" in lowered:
        return {
            "kind": "provider_auth",
            "detail": "provider authentication failed before a scoreable canary trace was produced",
        }
    if "rate_limit" in lowered or "429" in lowered:
        return {
            "kind": "provider_rate_limit",
            "detail": "provider rate limit failed the canary before trace verification",
        }
    return None


def verify_with_retries(args: argparse.Namespace, paths: CanaryPaths) -> dict[str, Any]:
    conversation_id = extract_conversation_id(paths.expected_sidecar)
    attempts: list[dict[str, Any]] = []
    max_attempts = max(1, args.verify_attempts)
    sleep_seconds = max(0.0, args.verify_sleep_seconds)

    for attempt in range(1, max_attempts + 1):
        json_path = paths.verifier_dir / f"attempt_{attempt:02d}.json"
        command = build_verify_command(args, paths, json_path, conversation_id=conversation_id)
        completed = run_subprocess(command, cwd=REPO_ROOT)
        record = {
            "attempt": attempt,
            "returncode": completed.returncode,
            "json_path": str(json_path),
            "command": command,
            "stdout_tail": completed.stdout[-2000:],
            "stderr_tail": completed.stderr[-2000:],
        }
        attempts.append(record)
        if completed.returncode == 0:
            return {
                "ok": True,
                "conversation_id": conversation_id,
                "attempts": attempts,
                "final_json": str(json_path),
            }
        if attempt < max_attempts and sleep_seconds > 0:
            time.sleep(sleep_seconds)

    return {
        "ok": False,
        "conversation_id": conversation_id,
        "attempts": attempts,
        "final_json": attempts[-1]["json_path"] if attempts else None,
    }


def run_agents_diagnostic(
    args: argparse.Namespace,
    paths: CanaryPaths,
    *,
    conversation_id: str | None = None,
) -> dict[str, Any]:
    paths.agents_diagnostic_file.parent.mkdir(parents=True, exist_ok=True)
    command = build_agents_diagnostic_command(
        args,
        paths,
        conversation_id=conversation_id,
    )
    completed = run_subprocess(command, cwd=REPO_ROOT)
    return {
        "ok": completed.returncode == 0,
        "returncode": completed.returncode,
        "json_path": str(paths.agents_diagnostic_file),
        "json_exists": paths.agents_diagnostic_file.exists(),
        "command": command,
        "stdout_tail": completed.stdout[-2000:],
        "stderr_tail": completed.stderr[-2000:],
    }


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    if args.execute and not args.model:
        raise SystemExit("--model is required with --execute")
    if args.execute and not args.nemoclaw_sandbox:
        raise SystemExit("--nemoclaw-sandbox is required with --execute")
    canary_id = args.canary_id or utc_canary_id()
    paths = canary_paths(args.output_dir, canary_id)
    run_command = build_run_command(args, paths)
    external_action_approval = build_external_action_approval_record(
        args.external_action_approval_report_json,
        required_before_external_action=bool(args.execute),
        expected_source_packet_path=args.external_action_approval_source_packet_json,
    )
    nemoclaw_openclaw_config_preflight = build_nemoclaw_openclaw_config_preflight(
        args,
        run=False,
    )
    plan = write_canary_files(
        args,
        paths,
        run_command,
        external_action_approval=external_action_approval,
        nemoclaw_openclaw_config_preflight=nemoclaw_openclaw_config_preflight,
    )

    if not args.execute:
        gate_result = write_gate_result(paths)
        print(
            json.dumps(
                {"ok": True, "executed": False, "plan": plan, "gate_result": gate_result},
                ensure_ascii=False,
                indent=2,
            )
        )
        return

    if not external_action_approval.get("valid"):
        command_result = write_blocked_command_result(
            paths,
            plan=plan,
            failure_kind="external_action_approval_missing",
            failure_detail=(
                "external-action approval was missing or invalid; OpenClaw and "
                "provider execution were not started"
            ),
            payload={"external_action_approval": external_action_approval},
        )
        gate_result = write_gate_result(paths)
        result = {
            "ok": False,
            "executed": False,
            "blocked_before_openclaw": True,
            "plan_file": str(paths.plan_file),
            "command_result_file": str(paths.command_result_file),
            "gate_result_file": str(paths.gate_result_file),
            "command_result": command_result,
            "gate_result": gate_result,
            "external_action_approval": external_action_approval,
            "missing_fields": [
                flag
                for flag, value in {
                    "--external-action-approval-source-packet-json": (
                        args.external_action_approval_source_packet_json
                        if external_action_approval.get("source_packet_path_matches_expected")
                        and external_action_approval.get("source_packet_sha256_matches_expected")
                        else None
                    ),
                    "--external-action-approval-report-json": (
                        args.external_action_approval_report_json
                        if external_action_approval.get("present")
                        else None
                    ),
                }.items()
                if not value
            ],
        }
        print(json.dumps(result, ensure_ascii=False, indent=2))
        raise SystemExit(2)

    nemoclaw_openclaw_config_preflight = build_nemoclaw_openclaw_config_preflight(
        args,
        run=True,
    )
    plan = write_canary_files(
        args,
        paths,
        run_command,
        external_action_approval=external_action_approval,
        nemoclaw_openclaw_config_preflight=nemoclaw_openclaw_config_preflight,
    )
    if not nemoclaw_openclaw_config_preflight.get("ok"):
        command_result = write_blocked_command_result(
            paths,
            plan=plan,
            failure_kind="nemoclaw_config_preflight_failed",
            failure_detail=(
                "NeMoClaw sandbox OpenClaw config preflight failed before "
                "OpenClaw or provider execution"
            ),
            payload={
                "nemoclaw_openclaw_config_preflight": (
                    nemoclaw_openclaw_config_preflight
                )
            },
        )
        gate_result = write_gate_result(paths)
        result = {
            "ok": False,
            "executed": False,
            "blocked_before_openclaw": True,
            "plan_file": str(paths.plan_file),
            "command_result_file": str(paths.command_result_file),
            "gate_result_file": str(paths.gate_result_file),
            "command_result": command_result,
            "gate_result": gate_result,
            "nemoclaw_openclaw_config_preflight": nemoclaw_openclaw_config_preflight,
        }
        print(json.dumps(result, ensure_ascii=False, indent=2))
        raise SystemExit(2)

    started_at = time.time()
    completed = run_subprocess(run_command, cwd=REPO_ROOT)
    command_result = {
        "ok": completed.returncode == 0,
        "returncode": completed.returncode,
        "started_at": started_at,
        "ended_at": time.time(),
        "paid_api_attempted": True,
        "blocked_before_openclaw": False,
        "stdout_tail": completed.stdout[-4000:],
        "stderr_tail": completed.stderr[-4000:],
        **command_result_identity(plan, paths),
    }
    failure = classify_openclaw_failure(paths.expected_sidecar, completed.stderr)
    if failure:
        command_result["failure"] = failure
    write_command_result(paths, command_result)
    gate_result = write_gate_result(paths)
    if completed.returncode != 0:
        print(
            json.dumps(
                {
                    "ok": False,
                    "executed": True,
                    "command_result": command_result,
                    "gate_result": gate_result,
                },
                ensure_ascii=False,
                indent=2,
            )
        )
        raise SystemExit(completed.returncode)

    verification = verify_with_retries(args, paths)
    agents_diagnostic = (
        run_agents_diagnostic(
            args,
            paths,
            conversation_id=verification.get("conversation_id")
            if isinstance(verification.get("conversation_id"), str)
            else None,
        )
        if verification.get("ok")
        else {
            "ok": False,
            "returncode": None,
            "json_path": str(paths.agents_diagnostic_file),
            "json_exists": False,
            "skipped": True,
        }
    )
    gate_result = write_gate_result(
        paths,
        verifier_json=verification.get("final_json"),
        agents_diagnostic_json=str(paths.agents_diagnostic_file),
    )
    result = {
        "ok": bool(verification.get("ok"))
        and bool(agents_diagnostic.get("ok"))
        and bool(gate_result.get("ok")),
        "executed": True,
        "plan_file": str(paths.plan_file),
        "command_result_file": str(paths.command_result_file),
        "gate_result_file": str(paths.gate_result_file),
        "sidecar": str(paths.expected_sidecar),
        "agents_diagnostic": agents_diagnostic,
        "verification": verification,
        "gate_result": gate_result,
    }
    print(json.dumps(result, ensure_ascii=False, indent=2))
    if not result["ok"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
