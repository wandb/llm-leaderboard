#!/usr/bin/env python3
"""Run post-install NeMoClaw verification for the Taiwan leaderboard.

This verifier is intentionally offline with respect to model providers and W&B.
It runs local setup checks, sandbox preflight, canary readiness, and the ADR
adoption doctor, then writes a single JSON/Markdown result.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shlex
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUTPUT_DIR = Path("temp")
DEFAULT_INSTALL_CHECK_SCRIPT = REPO_ROOT / "scripts" / "setup" / "install_nemoclaw.sh"
DEFAULT_PROTOCOL_SCRIPT = REPO_ROOT / "scripts" / "tools" / "run_openclaw_agent_protocol.py"
DEFAULT_CANARY_READINESS_SCRIPT = REPO_ROOT / "scripts" / "tools" / "check_taiwan_canary_readiness.py"
DEFAULT_ADOPTION_SCRIPT = REPO_ROOT / "scripts" / "tools" / "check_taiwan_nemoclaw_adoption.py"
DEFAULT_CANARY_MANIFEST = REPO_ROOT / "configs" / "taiwan_openai_canary_models.yaml"
DEFAULT_CANARY_SLUG = "gpt-4_1-mini-openai-direct-canary"
DEFAULT_CANARY_OPENCLAW_MODEL = "openai-direct/gpt-4.1-mini-2025-04-14"
DEFAULT_GENERATED_FULL_DIR = REPO_ROOT / "configs" / "taiwan_full" / "generated_openai_canary"
DEFAULT_GENERATED_NONAGENTIC_DIR = (
    REPO_ROOT / "configs" / "taiwan_full" / "generated_openai_canary_nonagentic"
)
DEFAULT_GENERATED_AGENTIC_NEMOCLAW_DIR = (
    REPO_ROOT / "configs" / "taiwan_full" / "generated_openai_canary_agentic_nemoclaw"
)
DEFAULT_GENERATED_AGENTIC_AGGREGATE_DIR = (
    REPO_ROOT / "configs" / "taiwan_full" / "generated_openai_canary_agentic_aggregate"
)
DEFAULT_STEP_TIMEOUT_SECONDS = 300.0
FORBIDDEN_POST_INSTALL_COMMAND_EXACT_TOKENS = {
    "--install",
    "--onboard",
    "--upload",
    "--wandb",
    "--yes-i-accept-third-party-software",
}
FORBIDDEN_POST_INSTALL_COMMAND_PREFIXES = (
    "--upload",
    "--wandb",
    "ANTHROPIC_API_KEY=",
    "GEMINI_API_KEY=",
    "GOOGLE_API_KEY=",
    "OPENAI_API_KEY=",
    "OPENROUTER_",
    "OPENROUTER_API_KEY=",
    "WANDB_",
    "WEAVE_",
    "XAI_API_KEY=",
)
FORBIDDEN_POST_INSTALL_COMMAND_MARKERS = (
    "openrouter",
    "wandb",
    "weave",
)
REQUIRED_POST_INSTALL_COMMAND_TOKENS = {
    "setup_check": ("--check-only", "--json"),
    "protocol_preflight": ("preflight",),
    "canary_readiness": (
        "--require-nemoclaw",
        "--json",
    ),
    "adoption_check": (
        "--setup-json",
        "--readiness-json",
        "--json",
        "--markdown",
    ),
}


def utc_timestamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def repo_path(path: Path | str) -> Path:
    value = Path(path).expanduser()
    if value.is_absolute():
        return value
    return REPO_ROOT / value


def path_display(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(REPO_ROOT))
    except ValueError:
        return str(path)


def python_command(value: str) -> list[str]:
    parts = shlex.split(value)
    return parts or [sys.executable]


def extend_optional_path(command: list[str], flag: str, value: Path | None) -> None:
    if value is not None:
        command.extend([flag, str(repo_path(value))])


def extend_optional_value(command: list[str], flag: str, value: str | None) -> None:
    if value:
        command.extend([flag, value])


def extend_repeated_paths(command: list[str], flag: str, values: list[Path] | None) -> None:
    for value in values or []:
        command.extend([flag, str(repo_path(value))])


def extend_repeated_values(command: list[str], flag: str, values: list[str] | None) -> None:
    for value in values or []:
        if value:
            command.extend([flag, value])


def timeout_text(value: str | bytes | None) -> str:
    if value is None:
        return ""
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    return value


def run_command(command: list[str], *, timeout_seconds: float) -> subprocess.CompletedProcess[str]:
    try:
        return subprocess.run(
            command,
            cwd=REPO_ROOT,
            text=True,
            capture_output=True,
            check=False,
            timeout=timeout_seconds,
        )
    except subprocess.TimeoutExpired as exc:
        stdout = timeout_text(exc.stdout)
        stderr = timeout_text(exc.stderr)
        timeout_message = f"\nCommand timed out after {timeout_seconds:g} seconds"
        return subprocess.CompletedProcess(
            command,
            124,
            stdout=stdout,
            stderr=stderr + timeout_message,
        )


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def read_json(path: Path) -> dict[str, Any] | None:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    return payload if isinstance(payload, dict) else None


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_file_or_empty(path: Path | None) -> str:
    if path is None or not path.exists() or not path.is_file():
        return ""
    try:
        return sha256_file(path)
    except OSError:
        return ""


def nested_get(payload: dict[str, Any], dotted: str) -> Any:
    current: Any = payload
    for part in dotted.split("."):
        if not isinstance(current, dict) or part not in current:
            return None
        current = current[part]
    return current


def require_bool(
    errors: list[str],
    payload: dict[str, Any],
    dotted: str,
    expected: bool,
) -> None:
    if nested_get(payload, dotted) is not expected:
        errors.append(f"{dotted} must be {str(expected).lower()}")


def require_empty_list(errors: list[str], payload: dict[str, Any], dotted: str) -> None:
    value = nested_get(payload, dotted)
    if value != []:
        errors.append(f"{dotted} must be an empty list")


def require_check_ok(
    errors: list[str],
    payload: dict[str, Any],
    name: str,
) -> None:
    checks = payload.get("checks")
    if not isinstance(checks, list):
        errors.append("checks must be a list")
        return
    for row in checks:
        if isinstance(row, dict) and row.get("name") == name:
            if row.get("ok") is not True:
                errors.append(f"check {name!r} must have ok=true")
            return
    errors.append(f"missing required check {name!r}")


def require_criterion_ok(
    errors: list[str],
    payload: dict[str, Any],
    name: str,
) -> None:
    criteria = payload.get("criteria")
    if not isinstance(criteria, list):
        errors.append("criteria must be a list")
        return
    for row in criteria:
        if isinstance(row, dict) and row.get("name") == name:
            if row.get("ok") is not True:
                errors.append(f"criterion {name!r} must have ok=true")
            return
    errors.append(f"missing required criterion {name!r}")


def validate_step_payload_contract(
    name: str,
    payload: dict[str, Any] | None,
    *,
    sandbox: str,
) -> list[str]:
    if not isinstance(payload, dict):
        return ["output JSON must be a readable object"]

    errors: list[str] = []
    require_bool(errors, payload, "ok", True)

    if name == "setup_check":
        require_bool(errors, payload, "check_only", True)
        require_bool(errors, payload, "install_requested", False)
        require_bool(errors, payload, "onboard_requested", False)
        require_bool(errors, payload, "host_prerequisites_ok", True)
        require_bool(errors, payload, "runtime_installed", True)
        require_bool(errors, payload, "sandbox_configured", True)
        require_empty_list(errors, payload, "missing_required_commands")
        require_bool(errors, payload, "commands.nemoclaw.available", True)
        require_bool(errors, payload, "commands.openshell.available", True)
        require_bool(errors, payload, "commands.docker.available", True)
        require_bool(errors, payload, "commands.docker.info_ok", True)
        require_bool(errors, payload, "setup_plan.will_launch_model_inference", False)
        require_bool(errors, payload, "setup_plan.sandbox_configured", True)
        require_bool(errors, payload, "setup_plan.sandbox_readiness_required", True)
        require_bool(
            errors,
            payload,
            "setup_plan.install_or_onboard_requires_explicit_acceptance",
            True,
        )
    elif name == "protocol_preflight":
        require_bool(errors, payload, "node_ok", True)
        require_bool(errors, payload, "nemoclaw.installed", True)
        require_bool(errors, payload, "nemoclaw.sandbox_status_ok", True)
        require_bool(errors, payload, "nemoclaw.sandbox_openclaw_ok", True)
        if nested_get(payload, "nemoclaw.sandbox") != sandbox:
            errors.append(f"nemoclaw.sandbox must be {sandbox!r}")
    elif name == "canary_readiness":
        for check_name in (
            "NeMoClaw command is available",
            "OpenShell command is available",
            "NeMoClaw version command succeeds",
            f"NeMoClaw sandbox status succeeds: {sandbox}",
            f"OpenClaw runs inside NeMoClaw sandbox: {sandbox}",
            f"NeMoClaw sandbox runtime policy is introspectable: {sandbox}",
            f"NeMoClaw W&B/Weave runtime policy is present: {sandbox}",
        ):
            require_check_ok(errors, payload, check_name)
    elif name == "adoption_check":
        status = payload.get("status")
        allowed_statuses = {
            "adoptable_for_agentic_math",
            "adoptable_for_agentic_benchmarks",
        }
        if status not in allowed_statuses:
            errors.append(
                "status must be adoptable_for_agentic_math or "
                "adoptable_for_agentic_benchmarks"
            )
        require_bool(errors, payload, "summary.ready_for_use", True)
        require_bool(errors, payload, "adoption_decision.ready_for_use", True)
        require_bool(errors, payload, "adoption_decision.design_ready", True)
        scope = nested_get(payload, "adoption_decision.scope")
        allowed_scopes = {
            "agentic_math_only",
            "agentic_math_and_swebench_pro",
        }
        if scope not in allowed_scopes:
            errors.append(
                "adoption_decision.scope must be agentic_math_only or "
                "agentic_math_and_swebench_pro"
            )
        if status == "adoptable_for_agentic_math" and scope != "agentic_math_only":
            errors.append("adoptable_for_agentic_math requires scope agentic_math_only")
        if (
            status == "adoptable_for_agentic_benchmarks"
            and scope != "agentic_math_and_swebench_pro"
        ):
            errors.append(
                "adoptable_for_agentic_benchmarks requires scope "
                "agentic_math_and_swebench_pro"
            )
        require_empty_list(errors, payload, "adoption_decision.blockers")
        require_empty_list(errors, payload, "summary.blockers")
        require_criterion_ok(errors, payload, "runtime_wandb_weave_policy")
        require_criterion_ok(errors, payload, "runtime_network_policy_allowlist")
    else:
        errors.append(f"unknown post-install step {name!r}")

    return errors


def last_json_object(text: str) -> dict[str, Any] | None:
    for line in reversed(text.splitlines()):
        stripped = line.strip()
        if not stripped.startswith("{"):
            continue
        try:
            payload = json.loads(stripped)
        except json.JSONDecodeError:
            continue
        if isinstance(payload, dict):
            return payload
    try:
        payload = json.loads(text)
    except json.JSONDecodeError:
        return None
    return payload if isinstance(payload, dict) else None


def command_record(
    *,
    name: str,
    command: list[str],
    completed: subprocess.CompletedProcess[str],
    output_json: Path | None = None,
    ok_from_json_key: str = "ok",
    sandbox: str,
) -> dict[str, Any]:
    payload = read_json(output_json) if output_json else None
    payload_ok = bool(payload.get(ok_from_json_key)) if isinstance(payload, dict) else None
    payload_contract_errors = validate_step_payload_contract(name, payload, sandbox=sandbox)
    payload_contract_ok = not payload_contract_errors
    returncode_ok = completed.returncode == 0
    ok = returncode_ok and (payload_ok if payload_ok is not None else True) and payload_contract_ok
    return {
        "name": name,
        "ok": ok,
        "returncode": completed.returncode,
        "returncode_ok": returncode_ok,
        "payload_ok": payload_ok,
        "payload_contract_ok": payload_contract_ok,
        "payload_contract_errors": payload_contract_errors,
        "timed_out": completed.returncode == 124 and "Command timed out after" in completed.stderr,
        "command": command,
        "output_json": path_display(output_json) if output_json else "",
        "output_json_sha256": sha256_file_or_empty(output_json),
        "stdout_tail": completed.stdout[-4000:],
        "stderr_tail": completed.stderr[-4000:],
        "payload_status": payload.get("status") if isinstance(payload, dict) else None,
    }


def command_safety_report(steps: list[dict[str, Any]]) -> dict[str, Any]:
    records: list[dict[str, Any]] = []
    for step in steps:
        name = str(step.get("name") or "")
        command = step.get("command")
        command_tokens = [str(token) for token in command] if isinstance(command, list) else []
        command_text = " ".join(command_tokens)
        forbidden_tokens = forbidden_post_install_command_tokens(command_tokens)
        missing_required_tokens = [
            token
            for token in REQUIRED_POST_INSTALL_COMMAND_TOKENS.get(name, ())
            if token not in command_text
        ]
        records.append(
            {
                "name": name,
                "ok": bool(command_tokens) and not forbidden_tokens and not missing_required_tokens,
                "forbidden_tokens": forbidden_tokens,
                "missing_required_tokens": missing_required_tokens,
            }
        )
    forbidden_token_count = sum(len(record["forbidden_tokens"]) for record in records)
    missing_required_token_count = sum(
        len(record["missing_required_tokens"]) for record in records
    )
    missing_command_count = sum(1 for record in records if not record["name"])
    return {
        "ok": (
            forbidden_token_count == 0
            and missing_required_token_count == 0
            and missing_command_count == 0
            and len(records) == len(REQUIRED_POST_INSTALL_COMMAND_TOKENS)
        ),
        "forbidden_tokens": sorted(FORBIDDEN_POST_INSTALL_COMMAND_EXACT_TOKENS),
        "forbidden_prefixes": list(FORBIDDEN_POST_INSTALL_COMMAND_PREFIXES),
        "forbidden_markers": list(FORBIDDEN_POST_INSTALL_COMMAND_MARKERS),
        "required_step_tokens": {
            name: list(tokens)
            for name, tokens in sorted(REQUIRED_POST_INSTALL_COMMAND_TOKENS.items())
        },
        "forbidden_token_count": forbidden_token_count,
        "missing_required_token_count": missing_required_token_count,
        "missing_command_count": missing_command_count,
        "records": records,
    }


def forbidden_post_install_command_tokens(command_tokens: list[str]) -> list[str]:
    forbidden: list[str] = []
    for token in command_tokens:
        token_text = str(token)
        token_casefold = token_text.casefold()
        if token_text in FORBIDDEN_POST_INSTALL_COMMAND_EXACT_TOKENS:
            forbidden.append(token_text)
            continue
        if any(token_text.startswith(prefix) for prefix in FORBIDDEN_POST_INSTALL_COMMAND_PREFIXES):
            forbidden.append(token_text)
            continue
        if any(marker in token_casefold for marker in FORBIDDEN_POST_INSTALL_COMMAND_MARKERS):
            forbidden.append(token_text)
    return forbidden


def write_preflight_json(path: Path, completed: subprocess.CompletedProcess[str]) -> dict[str, Any]:
    payload = last_json_object(completed.stdout)
    if payload is None:
        payload = {
            "ok": False,
            "status": "invalid_preflight_output",
            "stdout_tail": completed.stdout[-4000:],
            "stderr_tail": completed.stderr[-4000:],
        }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return payload


def build_markdown(report: dict[str, Any]) -> str:
    lines = [
        "# Taiwan NeMoClaw Post-Install Verification",
        "",
        f"Status: `{report['status']}`",
        f"OK: `{str(report['ok']).lower()}`",
        f"Sandbox: `{report['sandbox']}`",
        "",
        "| Step | OK | Return Code | Output JSON |",
        "|---|---:|---:|---|",
    ]
    for step in report.get("steps", []):
        if not isinstance(step, dict):
            continue
        lines.append(
            "| "
            + " | ".join(
                [
                    str(step.get("name", "")),
                    str(bool(step.get("ok"))).lower(),
                    str(step.get("returncode", "")),
                    f"`{step.get('output_json', '')}`",
                ]
            )
            + " |"
        )
    lines.extend(["", "## Next Actions", ""])
    for step in report.get("steps", []):
        if isinstance(step, dict) and not step.get("ok"):
            lines.append(f"- `{step.get('name')}`: inspect `{step.get('output_json')}`")
    if all(isinstance(step, dict) and step.get("ok") for step in report.get("steps", [])):
        lines.append("- none")
    lines.append("")
    return "\n".join(lines)


def build_report(args: argparse.Namespace) -> dict[str, Any]:
    output_dir = repo_path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    timestamp = args.timestamp
    setup_json = repo_path(args.setup_json or output_dir / f"nemoclaw_setup_check_{timestamp}.json")
    preflight_json = repo_path(args.preflight_json or output_dir / f"nemoclaw_protocol_preflight_{timestamp}.json")
    readiness_json = repo_path(args.readiness_json or output_dir / f"nemoclaw_canary_readiness_{timestamp}.json")
    adoption_json = repo_path(args.adoption_json or output_dir / f"taiwan_nemoclaw_adoption_check_{timestamp}.json")
    adoption_md = repo_path(args.adoption_markdown or output_dir / f"taiwan_nemoclaw_adoption_check_{timestamp}.md")

    py = python_command(args.python_command)
    install_command = [
        str(repo_path(args.install_check_script)),
        "--check-only",
        "--sandbox",
        args.sandbox,
        "--json",
        str(setup_json),
    ]
    setup_completed = run_command(
        install_command,
        timeout_seconds=args.step_timeout_seconds,
    )

    preflight_command = [
        *py,
        str(repo_path(args.protocol_script)),
        "preflight",
        "--nemoclaw-bin",
        args.nemoclaw_bin,
        "--nemoclaw-sandbox",
        args.sandbox,
    ]
    preflight_completed = run_command(
        preflight_command,
        timeout_seconds=args.step_timeout_seconds,
    )
    write_preflight_json(preflight_json, preflight_completed)

    readiness_command = [
        *py,
        str(repo_path(args.canary_readiness_script)),
        "--require-nemoclaw",
        "--nemoclaw-bin",
        args.nemoclaw_bin,
        "--nemoclaw-sandbox",
        args.sandbox,
        "--json",
        str(readiness_json),
    ]
    extend_optional_path(readiness_command, "--manifest", args.canary_manifest)
    extend_optional_value(readiness_command, "--canary-slug", args.canary_slug)
    extend_optional_value(readiness_command, "--openclaw-model", args.canary_openclaw_model)
    extend_optional_value(
        readiness_command,
        "--expected-pretrained-model",
        args.canary_expected_pretrained_model,
    )
    extend_optional_path(readiness_command, "--generated-full-dir", args.generated_full_dir)
    extend_optional_path(
        readiness_command,
        "--generated-nonagentic-dir",
        args.generated_nonagentic_dir,
    )
    extend_optional_path(readiness_command, "--generated-agentic-dir", args.generated_agentic_dir)
    extend_optional_path(
        readiness_command,
        "--generated-agentic-aggregate-dir",
        args.generated_agentic_aggregate_dir,
    )
    readiness_completed = run_command(
        readiness_command,
        timeout_seconds=args.step_timeout_seconds,
    )

    adoption_command = [
        *py,
        str(repo_path(args.adoption_script)),
        "--setup-json",
        str(setup_json),
        "--readiness-json",
        str(readiness_json),
        "--sandbox",
        args.sandbox,
        "--json",
        str(adoption_json),
        "--markdown",
        str(adoption_md),
    ]
    extend_repeated_paths(adoption_command, "--agentic-config", args.adoption_agentic_config)
    extend_repeated_values(
        adoption_command,
        "--agentic-config-glob",
        args.adoption_agentic_config_glob,
    )
    if args.fail_adoption_step:
        adoption_command.append("--fail-on-not-adoptable")
    adoption_completed = run_command(
        adoption_command,
        timeout_seconds=args.step_timeout_seconds,
    )

    steps = [
        command_record(
            name="setup_check",
            command=install_command,
            completed=setup_completed,
            output_json=setup_json,
            sandbox=args.sandbox,
        ),
        command_record(
            name="protocol_preflight",
            command=preflight_command,
            completed=preflight_completed,
            output_json=preflight_json,
            sandbox=args.sandbox,
        ),
        command_record(
            name="canary_readiness",
            command=readiness_command,
            completed=readiness_completed,
            output_json=readiness_json,
            sandbox=args.sandbox,
        ),
        command_record(
            name="adoption_check",
            command=adoption_command,
            completed=adoption_completed,
            output_json=adoption_json,
            sandbox=args.sandbox,
        ),
    ]
    command_safety = command_safety_report(steps)
    ok = all(step["ok"] for step in steps) and bool(command_safety["ok"])
    outputs = {
        "setup_json": path_display(setup_json),
        "preflight_json": path_display(preflight_json),
        "readiness_json": path_display(readiness_json),
        "adoption_json": path_display(adoption_json),
        "adoption_markdown": path_display(adoption_md),
    }
    outputs_sha256 = {
        "setup_json": sha256_file_or_empty(setup_json),
        "preflight_json": sha256_file_or_empty(preflight_json),
        "readiness_json": sha256_file_or_empty(readiness_json),
        "adoption_json": sha256_file_or_empty(adoption_json),
        "adoption_markdown": sha256_file_or_empty(adoption_md),
    }
    return {
        "schema_version": 1,
        "ok": ok,
        "status": "passed" if ok else "failed",
        "generated_at": time.time(),
        "sandbox": args.sandbox,
        "will_launch_model_inference": False,
        "will_query_wandb": False,
        "will_install_or_onboard": False,
        "step_timeout_seconds": args.step_timeout_seconds,
        "command_safety": command_safety,
        "steps": steps,
        "outputs": outputs,
        "outputs_sha256": outputs_sha256,
    }


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--timestamp", default=utc_timestamp())
    parser.add_argument("--sandbox", default="nejumi-taiwan")
    parser.add_argument("--nemoclaw-bin", default="nemoclaw")
    parser.add_argument("--python-command", default="uv run python")
    parser.add_argument("--install-check-script", type=Path, default=DEFAULT_INSTALL_CHECK_SCRIPT)
    parser.add_argument("--protocol-script", type=Path, default=DEFAULT_PROTOCOL_SCRIPT)
    parser.add_argument("--canary-readiness-script", type=Path, default=DEFAULT_CANARY_READINESS_SCRIPT)
    parser.add_argument("--adoption-script", type=Path, default=DEFAULT_ADOPTION_SCRIPT)
    parser.add_argument("--canary-manifest", type=Path, default=DEFAULT_CANARY_MANIFEST)
    parser.add_argument("--canary-slug", default=DEFAULT_CANARY_SLUG)
    parser.add_argument("--canary-openclaw-model", default=DEFAULT_CANARY_OPENCLAW_MODEL)
    parser.add_argument("--canary-expected-pretrained-model")
    parser.add_argument("--generated-full-dir", type=Path, default=DEFAULT_GENERATED_FULL_DIR)
    parser.add_argument(
        "--generated-nonagentic-dir",
        type=Path,
        default=DEFAULT_GENERATED_NONAGENTIC_DIR,
    )
    parser.add_argument(
        "--generated-agentic-dir",
        type=Path,
        default=DEFAULT_GENERATED_AGENTIC_NEMOCLAW_DIR,
    )
    parser.add_argument(
        "--generated-agentic-aggregate-dir",
        type=Path,
        default=DEFAULT_GENERATED_AGENTIC_AGGREGATE_DIR,
    )
    parser.add_argument("--adoption-agentic-config", type=Path, action="append")
    parser.add_argument("--adoption-agentic-config-glob", action="append")
    parser.add_argument("--setup-json", type=Path)
    parser.add_argument("--preflight-json", type=Path)
    parser.add_argument("--readiness-json", type=Path)
    parser.add_argument("--adoption-json", type=Path)
    parser.add_argument("--adoption-markdown", type=Path)
    parser.add_argument("--step-timeout-seconds", type=float, default=DEFAULT_STEP_TIMEOUT_SECONDS)
    parser.add_argument("--json", type=Path)
    parser.add_argument("--markdown", type=Path)
    parser.add_argument(
        "--fail-adoption-step",
        action="store_true",
        help="Pass --fail-on-not-adoptable to the adoption doctor.",
    )
    parser.add_argument("--fail-on-failed", action="store_true")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    report = build_report(args)
    if args.json:
        report["path"] = str(repo_path(args.json))
    if args.markdown:
        report["markdown_path"] = str(repo_path(args.markdown))
    if args.json:
        write_json(repo_path(args.json), report)
    if args.markdown:
        path = repo_path(args.markdown)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(build_markdown(report), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    if args.fail_on_failed and not report["ok"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main(sys.argv[1:])
