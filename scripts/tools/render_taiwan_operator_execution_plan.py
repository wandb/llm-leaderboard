#!/usr/bin/env python3
"""Render Taiwan release operator-plan templates with reviewed values.

This tool does not execute external actions. It turns the machine-generated
operator plan into a concrete pre-execution review artifact, so paid API,
W&B, and NeMoClaw commands are not copied from placeholder templates by hand.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shlex
import stat
import sys
import time
from pathlib import Path
from typing import Any

from weave_content_canary_gate_contract import (
    weave_content_canary_gate_contract_issues,
)


PLACEHOLDER_TOKENS = (
    "CONTENT_CANARY_YYYYMMDDTHHMM",
    "WEAVE_CONTENT_CANARY_GATE",
    "YYYYMMDDTHHMM",
    "YYYYMMDD",
    "MODEL_SLUG",
    "RUN_ID",
    "PHASE",
)
DEFAULT_WEAVE_CONTENT_CANARY_MAX_AGE_SECONDS = 24 * 60 * 60
OPENAI_DIRECT_CANARY_MANIFEST = "configs/taiwan_openai_canary_models.yaml"
OPENAI_DIRECT_CANARY_GENERATED_DIRS = {
    "configs/taiwan_full/generated_openai_canary",
    "configs/taiwan_full/generated_openai_canary_nonagentic",
    "configs/taiwan_full/generated_openai_canary_agentic_nemoclaw",
    "configs/taiwan_full/generated_openai_canary_agentic_aggregate",
}
CANARY_FORBIDDEN_PROVIDER_MARKERS = (
    "openrouter",
    "anthropic",
    "claude",
    "gemini",
    "opus",
    "sonnet",
)
OPENAI_DIRECT_CANARY_APPROVAL_SCOPE_MARKER = "openai-direct/gpt-4.1-mini"
NEMOCLAW_OPENCLAW_CONFIG_PATH = "/sandbox/.openclaw/openclaw.json"
WEAVE_CONTENT_CANARY_BLOCKED_NEXT_ACTIONS = {
    "external_action_approval_missing": (
        "approve the source-bound external-action packet before rerunning; "
        "OpenClaw/provider execution was blocked before any paid API attempt"
    ),
    "nemoclaw_config_preflight_failed": (
        "fix the NeMoClaw sandbox OpenClaw config path/provider/model mapping "
        "before rerunning; OpenClaw/provider execution was blocked before any "
        "paid API attempt"
    ),
}


def read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise ValueError(f"{path} must contain a JSON object")
    return payload


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def write_text(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def unresolved_tokens(values: list[str]) -> list[str]:
    found: list[str] = []
    for token in PLACEHOLDER_TOKENS:
        if any(token in value for value in values):
            found.append(token)
    return found


def replacement_map(args: argparse.Namespace) -> dict[str, str]:
    mapping: dict[str, str] = {}
    if args.timestamp:
        timestamp = args.timestamp
        mapping["YYYYMMDDTHHMM"] = timestamp
        mapping["YYYYMMDD"] = timestamp[:8]
        mapping["CONTENT_CANARY_YYYYMMDDTHHMM"] = (
            args.content_canary_id or f"CONTENT_CANARY_{timestamp}"
        )
    if args.content_canary_id:
        mapping["CONTENT_CANARY_YYYYMMDDTHHMM"] = args.content_canary_id
    if args.weave_content_canary_gate:
        mapping["WEAVE_CONTENT_CANARY_GATE"] = args.weave_content_canary_gate
    if args.run_id:
        mapping["RUN_ID"] = args.run_id
    if args.model_slug:
        mapping["MODEL_SLUG"] = args.model_slug
    if args.phase:
        mapping["PHASE"] = args.phase
    return mapping


def apply_replacements(value: str, replacements: dict[str, str]) -> str:
    rendered = value
    for token in PLACEHOLDER_TOKENS:
        if token in replacements:
            rendered = rendered.replace(token, replacements[token])
    return rendered


def command_option_value(raw_command: str, option: str) -> str:
    parts = split_command(raw_command)
    return option_value(parts, option) or ""


def replace_command_option_value(command: str, option: str, value: Path | None) -> str:
    if value is None:
        return command
    parts = split_command(command)
    if option not in parts:
        return command
    output: list[str] = []
    index = 0
    replaced = False
    while index < len(parts):
        output.append(parts[index])
        if parts[index] == option and index + 1 < len(parts):
            output.append(str(value))
            index += 2
            replaced = True
            continue
        index += 1
    if not replaced:
        return command
    return shlex.join(output)


def bind_template_command_approval_paths(
    raw_command: str,
    rendered_command: str,
    *,
    external_action_approval_source_packet_json: Path | None,
    external_action_approval_report_json: Path | None,
) -> str:
    """Bind placeholder approval paths to the reviewed source/report paths."""

    source_value = command_option_value(
        raw_command,
        "--external-action-approval-source-packet-json",
    )
    if "YYYYMMDDTHHMM" in source_value:
        rendered_command = replace_command_option_value(
            rendered_command,
            "--external-action-approval-source-packet-json",
            external_action_approval_source_packet_json,
        )
    report_value = command_option_value(
        raw_command,
        "--external-action-approval-report-json",
    )
    if "YYYYMMDDTHHMM" in report_value:
        rendered_command = replace_command_option_value(
            rendered_command,
            "--external-action-approval-report-json",
            external_action_approval_report_json,
        )
    return rendered_command


def count_required(steps: list[dict[str, Any]], key: str) -> int:
    return sum(1 for step in steps if bool(step.get(key)))


def requires_external_action(steps: list[dict[str, Any]]) -> bool:
    return any(
        bool(step.get(key))
        for step in steps
        for key in (
            "requires_paid_api",
            "requires_wandb_access",
            "requires_wandb_write",
            "requires_third_party_acceptance",
            "requires_nemoclaw_install",
            "requires_scope_confirmation",
        )
    )


def is_executable_command(command: str) -> bool:
    stripped = command.strip()
    executable_prefixes = (
        "uv ",
        "python ",
        "python3 ",
        "bash ",
        "sh ",
        "./",
        "scripts/",
    )
    return stripped.startswith(executable_prefixes)


def split_command(command: str) -> list[str]:
    try:
        return shlex.split(command)
    except ValueError:
        return command.split()


def command_invokes(parts: list[str], script_name: str) -> bool:
    return any(part.endswith(script_name) for part in parts)


def option_present(parts: list[str], option: str) -> bool:
    return option in parts


def option_value(parts: list[str], option: str) -> str:
    try:
        index = parts.index(option)
    except ValueError:
        return ""
    if index + 1 >= len(parts):
        return ""
    return parts[index + 1]


def has_unresolved_placeholder(value: str) -> bool:
    return any(token in value for token in PLACEHOLDER_TOKENS)


def numeric_timestamp(value: Any) -> float | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str):
        try:
            return float(value)
        except ValueError:
            return None
    return None


def validate_weave_content_canary_gate_option(
    parts: list[str],
    *,
    base_dir: Path,
) -> list[str]:
    """Validate a concrete Weave content-canary gate path used for execution."""

    errors: list[str] = []
    gate_value = option_value(parts, "--weave-content-canary-gate")
    if not gate_value:
        errors.append("--weave-content-canary-gate value is missing")
        return errors
    if has_unresolved_placeholder(gate_value):
        return errors

    gate_path = Path(gate_value)
    if not gate_path.is_absolute():
        gate_path = (base_dir / gate_path).resolve()
    if not gate_path.exists():
        errors.append(f"Weave content canary gate JSON does not exist: {gate_value}")
        return errors
    try:
        payload = read_json(gate_path)
    except (OSError, json.JSONDecodeError, ValueError) as exc:
        errors.append(f"Weave content canary gate JSON is not readable: {gate_value}: {exc}")
        return errors

    if payload.get("ok") is not True or payload.get("status") != "passed":
        errors.append(failed_weave_content_canary_gate_error(payload, gate_value))
        return errors

    contract_issues = weave_content_canary_gate_contract_issues(payload)
    if contract_issues:
        errors.append(
            "Weave content canary gate is missing native Weave verifier "
            f"contract evidence: {'; '.join(contract_issues)}"
        )
        return errors

    max_age_seconds = DEFAULT_WEAVE_CONTENT_CANARY_MAX_AGE_SECONDS
    max_age_value = option_value(parts, "--weave-content-canary-max-age-seconds")
    if max_age_value:
        try:
            max_age_seconds = int(max_age_value)
        except ValueError:
            errors.append(
                "--weave-content-canary-max-age-seconds must be an integer "
                f"when supplied: {max_age_value}"
            )
            return errors
    if max_age_seconds < 0:
        return errors

    generated_at = numeric_timestamp(payload.get("generated_at"))
    freshness_timestamp = generated_at if generated_at is not None else gate_path.stat().st_mtime
    age_seconds = time.time() - freshness_timestamp
    if age_seconds > max_age_seconds:
        errors.append(
            "Weave content canary gate is stale: "
            f"{gate_value} age_seconds={age_seconds:.0f} "
            f"max_age_seconds={max_age_seconds}"
        )
    return errors


def failed_weave_content_canary_gate_error(payload: dict[str, Any], gate_value: str) -> str:
    status = payload.get("status")
    failure_kind = payload.get("failure_kind")
    detail_parts = [
        f"{gate_value} has ok={payload.get('ok')!r}",
        f"status={status!r}",
        f"failure_kind={failure_kind!r}",
    ]
    paid_api_attempted = payload.get("paid_api_attempted")
    if isinstance(paid_api_attempted, bool):
        detail_parts.append(f"paid_api_attempted={paid_api_attempted!r}")
    recommended_next_action = payload.get("recommended_next_action")
    if isinstance(recommended_next_action, str) and recommended_next_action.strip():
        detail_parts.append(
            f"recommended_next_action={recommended_next_action.strip()!r}"
        )
    status_action = (
        WEAVE_CONTENT_CANARY_BLOCKED_NEXT_ACTIONS.get(status)
        if isinstance(status, str)
        else None
    )
    failure_action = (
        WEAVE_CONTENT_CANARY_BLOCKED_NEXT_ACTIONS.get(failure_kind)
        if isinstance(failure_kind, str)
        else None
    )
    next_action = status_action or failure_action
    if next_action:
        detail_parts.append(f"next_action={next_action!r}")
    return (
        "Weave content canary gate must have ok=true and status=passed: "
        + ", ".join(detail_parts)
    )


def validate_external_action_source_packet_option(
    parts: list[str],
    *,
    base_dir: Path,
    expected_source_packet_path: Path | None,
) -> list[str]:
    """Validate command-level approval source packet against the reviewed packet."""

    if expected_source_packet_path is None:
        return []
    value = option_value(parts, "--external-action-approval-source-packet-json")
    if not value or has_unresolved_placeholder(value):
        return []
    observed = Path(value)
    if not observed.is_absolute():
        observed = (base_dir / observed).resolve()
    expected = expected_source_packet_path.resolve()
    if observed != expected:
        return [
            "--external-action-approval-source-packet-json command value "
            f"does not match reviewed source packet: {value} != {expected}"
        ]
    return []


def validate_external_action_approval_report_option(
    parts: list[str],
    *,
    base_dir: Path,
    expected_report_path: Path | None,
) -> list[str]:
    """Validate command-level approval verifier report against reviewed report."""

    if expected_report_path is None:
        return []
    value = option_value(parts, "--external-action-approval-report-json")
    if not value or has_unresolved_placeholder(value):
        return []
    observed = Path(value)
    if not observed.is_absolute():
        observed = (base_dir / observed).resolve()
    expected = expected_report_path.resolve()
    if observed != expected:
        return [
            "--external-action-approval-report-json command value does not "
            f"match reviewed verifier report: {value} != {expected}"
        ]
    return []


def validate_openai_direct_canary_batch_command(
    parts: list[str],
    *,
    command: str,
) -> list[str]:
    """Keep the current one-model canary on the approved OpenAI-direct path."""

    if not option_present(parts, "--canary"):
        return []
    errors: list[str] = []
    lower_command = command.lower()
    markers = [
        marker
        for marker in CANARY_FORBIDDEN_PROVIDER_MARKERS
        if marker in lower_command
    ]
    if markers:
        errors.append(
            "OpenAI-direct canary command uses forbidden provider marker(s): "
            + ", ".join(sorted(set(markers)))
        )
    manifest = option_value(parts, "--manifest")
    if manifest != OPENAI_DIRECT_CANARY_MANIFEST:
        errors.append(
            "OpenAI-direct canary command must use "
            f"--manifest {OPENAI_DIRECT_CANARY_MANIFEST}"
        )
    generated_dir = option_value(parts, "--generated-config-dir")
    if generated_dir not in OPENAI_DIRECT_CANARY_GENERATED_DIRS:
        errors.append(
            "OpenAI-direct canary command must use an approved generated config dir: "
            f"{sorted(OPENAI_DIRECT_CANARY_GENERATED_DIRS)}"
        )
    return errors


def command_is_paid_canary_batch(command: str) -> bool:
    parts = split_command(command)
    return (
        command_invokes(parts, "run_taiwan_full_eval_batch.py")
        and option_present(parts, "--canary")
        and not option_present(parts, "--prepare-only")
    )


def extract_paid_api_approval(payload: dict[str, Any]) -> dict[str, Any]:
    result = {
        "present": False,
        "approved": False,
        "approved_budget_usd": None,
        "approved_model_scope": "",
    }
    approval_results = payload.get("approval_results")
    if not isinstance(approval_results, list):
        return result
    for item in approval_results:
        if not isinstance(item, dict) or item.get("requirement") != "paid_api":
            continue
        result["present"] = True
        result["approved"] = bool(item.get("approved"))
        result["approved_budget_usd"] = item.get("approved_budget_usd")
        result["approved_model_scope"] = (
            item.get("approved_model_scope")
            if isinstance(item.get("approved_model_scope"), str)
            else ""
        )
        break
    return result


def validate_canary_approval_scope(
    steps: list[dict[str, Any]],
    approval: dict[str, Any],
) -> dict[str, Any]:
    requires_scope = any(
        command_is_paid_canary_batch(str(command))
        for step in steps
        if isinstance(step, dict)
        for command in step.get("executable_commands") or []
    )
    errors: list[str] = []
    paid_api = (
        approval.get("paid_api_approval")
        if isinstance(approval.get("paid_api_approval"), dict)
        else {}
    )
    if requires_scope:
        if paid_api.get("approved") is not True:
            errors.append(
                "paid canary execution requires an approved paid_api approval result"
            )
        scope = (
            paid_api.get("approved_model_scope")
            if isinstance(paid_api.get("approved_model_scope"), str)
            else ""
        )
        lower_scope = scope.lower()
        if OPENAI_DIRECT_CANARY_APPROVAL_SCOPE_MARKER not in lower_scope:
            errors.append(
                "paid canary approval scope must include "
                f"{OPENAI_DIRECT_CANARY_APPROVAL_SCOPE_MARKER}"
            )
        markers = [
            marker
            for marker in CANARY_FORBIDDEN_PROVIDER_MARKERS
            if marker in lower_scope
        ]
        if markers:
            errors.append(
                "paid canary approval scope contains forbidden provider marker(s): "
                + ", ".join(sorted(set(markers)))
            )
    return {
        "valid": not errors,
        "required": requires_scope,
        "expected_model_scope_marker": OPENAI_DIRECT_CANARY_APPROVAL_SCOPE_MARKER
        if requires_scope
        else "",
        "observed_model_scope": paid_api.get("approved_model_scope") or "",
        "errors": errors,
    }


def validate_command_policy(
    steps: list[dict[str, Any]],
    *,
    base_dir: Path | None = None,
    expected_external_action_source_packet_path: Path | None = None,
    expected_external_action_approval_report_path: Path | None = None,
) -> dict[str, Any]:
    """Validate rendered executable commands before producing runnable shell."""

    errors: list[str] = []
    warnings: list[str] = []
    records: list[dict[str, Any]] = []
    checked_command_count = 0
    base_dir = (base_dir or Path.cwd()).resolve()
    for step in steps:
        if not isinstance(step, dict):
            continue
        gate = str(step.get("gate") or "")
        for command in step.get("executable_commands") or []:
            checked_command_count += 1
            command = str(command)
            parts = split_command(command)
            command_errors: list[str] = []
            lower_command = command.lower()
            if "openrouter" in lower_command:
                command_errors.append("command must not reference OpenRouter")
            if option_present(parts, "--external-action-approval-source-packet-json"):
                command_errors.extend(
                    validate_external_action_source_packet_option(
                        parts,
                        base_dir=base_dir,
                        expected_source_packet_path=(
                            expected_external_action_source_packet_path
                        ),
                    )
                )
            if option_present(parts, "--external-action-approval-report-json"):
                command_errors.extend(
                    validate_external_action_approval_report_option(
                        parts,
                        base_dir=base_dir,
                        expected_report_path=(
                            expected_external_action_approval_report_path
                        ),
                    )
                )

            if command_invokes(parts, "run_weave_agents_content_canary.py") and option_present(
                parts, "--execute"
            ):
                for option in (
                    "--external-action-approval-source-packet-json",
                    "--external-action-approval-report-json",
                    "--nemoclaw-sandbox",
                    "--nemoclaw-openclaw-config-path",
                ):
                    if not option_present(parts, option):
                        command_errors.append(
                            f"run_weave_agents_content_canary.py --execute is missing {option}"
                        )
                config_path = option_value(parts, "--nemoclaw-openclaw-config-path")
                if config_path and config_path != NEMOCLAW_OPENCLAW_CONFIG_PATH:
                    command_errors.append(
                        "run_weave_agents_content_canary.py --execute "
                        "--nemoclaw-openclaw-config-path must be "
                        f"{NEMOCLAW_OPENCLAW_CONFIG_PATH}"
                    )

            if command_invokes(parts, "run_taiwan_full_eval_batch.py"):
                command_errors.extend(
                    validate_openai_direct_canary_batch_command(
                        parts,
                        command=command,
                    )
                )
                phase = option_value(parts, "--phase") or "full"
                prepare_only = option_present(parts, "--prepare-only")
                external_phase = phase in {"full", "nonagentic", "agentic", "agentic_aggregate"}
                if external_phase and not prepare_only:
                    for option in (
                        "--external-action-approval-source-packet-json",
                        "--external-action-approval-report-json",
                    ):
                        if not option_present(parts, option):
                            command_errors.append(
                                f"run_taiwan_full_eval_batch.py {phase} execution is missing {option}"
                            )
                if phase in {"full", "agentic"} and not prepare_only:
                    for option in (
                        "--require-nemoclaw-agentic-config",
                        "--agentic-math-nemoclaw-sandbox",
                        "--swebench-pro-nemoclaw-sandbox",
                        "--agentic-math-nemoclaw-openclaw-config-path",
                        "--swebench-pro-nemoclaw-openclaw-config-path",
                    ):
                        if not option_present(parts, option):
                            command_errors.append(
                                f"run_taiwan_full_eval_batch.py {phase} command is missing {option}"
                            )
                    for option in (
                        "--agentic-math-nemoclaw-openclaw-config-path",
                        "--swebench-pro-nemoclaw-openclaw-config-path",
                    ):
                        value = option_value(parts, option)
                        if value and value != NEMOCLAW_OPENCLAW_CONFIG_PATH:
                            command_errors.append(
                                f"run_taiwan_full_eval_batch.py {phase} command "
                                f"{option} must be {NEMOCLAW_OPENCLAW_CONFIG_PATH}"
                            )
                    transfer_mode = option_value(
                        parts,
                        "--swebench-pro-nemoclaw-checkout-transfer-mode",
                    )
                    checkout_root = option_value(
                        parts,
                        "--swebench-pro-nemoclaw-checkout-sandbox-root",
                    )
                    if transfer_mode not in {"copy", "visible"} and not checkout_root:
                        command_errors.append(
                            "run_taiwan_full_eval_batch.py "
                            f"{phase} command must set SWE checkout transfer mode "
                            "to copy/visible or provide --swebench-pro-nemoclaw-checkout-sandbox-root"
                        )
                    for option in (
                        "--weave-content-canary-gate",
                        "--require-weave-content-canary",
                        "--verify-wandb-completion",
                        "--verify-weave-agents",
                        "--wandb-run-id-prefix",
                        "--weave-agents-require-tool-span",
                        "--weave-agents-require-tool-content",
                        "--weave-agents-require-usage",
                    ):
                        if not option_present(parts, option):
                            command_errors.append(
                                f"run_taiwan_full_eval_batch.py {phase} execution is missing {option}"
                            )
                    if option_present(parts, "--weave-content-canary-gate"):
                        command_errors.extend(
                            validate_weave_content_canary_gate_option(
                                parts,
                                base_dir=base_dir,
                            )
                        )
                if option_present(parts, "--verify-weave-agents"):
                    for option in (
                        "--wandb-run-id-prefix",
                        "--weave-agents-require-tool-span",
                        "--weave-agents-require-tool-content",
                        "--weave-agents-require-usage",
                    ):
                        if not option_present(parts, option):
                            command_errors.append(
                                f"run_taiwan_full_eval_batch.py Weave verification is missing {option}"
                            )
                if option_present(parts, "--verify-wandb-completion") and not option_present(
                    parts,
                    "--wandb-run-id-prefix",
                ):
                    command_errors.append(
                        "run_taiwan_full_eval_batch.py W&B completion verification is missing --wandb-run-id-prefix"
                    )

            records.append(
                {
                    "gate": gate,
                    "command": command,
                    "ok": not command_errors,
                    "errors": command_errors,
                }
            )
            errors.extend(command_errors)

    return {
        "valid": not errors,
        "checked_command_count": checked_command_count,
        "error_count": len(errors),
        "warning_count": len(warnings),
        "errors": errors,
        "warnings": warnings,
        "records": records,
    }


def build_external_action_approval_record(
    report_path: Path | None,
    *,
    required_before_external_action: bool,
    expected_source_packet_path: Path | None = None,
) -> dict[str, Any]:
    record: dict[str, Any] = {
        "required_before_external_action": bool(required_before_external_action),
        "path": str(report_path) if report_path else "",
        "present": False,
        "valid": False,
        "sha256": "",
        "schema_version": None,
        "status": "",
        "approval_packet_json": "",
        "required_approval_count": None,
        "granted_approval_count": None,
        "all_required_approvals_granted": False,
        "source_binding": {},
        "paid_api_approval": {},
        "expected_source_packet_json": str(expected_source_packet_path)
        if expected_source_packet_path
        else "",
        "expected_source_packet_sha256": "",
        "source_packet_path_matches_expected": False,
        "source_packet_sha256_matches_expected": False,
        "will_execute_external_actions": None,
        "errors": [],
    }
    if required_before_external_action and expected_source_packet_path is None:
        record["errors"].append(
            "external-action approval source packet is required before shell execution"
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
    if report_path is None:
        if required_before_external_action:
            record["errors"].append(
                "external-action approval verifier report is required before shell execution"
            )
        return record
    if not report_path.exists():
        record["errors"].append(f"external-action approval verifier report does not exist: {report_path}")
        return record
    try:
        payload = read_json(report_path)
    except (OSError, json.JSONDecodeError, ValueError) as exc:
        record["errors"].append(
            f"external-action approval verifier report is not readable JSON: {exc}"
        )
        return record

    record["present"] = True
    try:
        record["sha256"] = sha256_file(report_path)
    except OSError as exc:
        record["errors"].append(f"external-action approval verifier report sha256 failed: {exc}")
    source_binding = (
        payload.get("source_binding")
        if isinstance(payload.get("source_binding"), dict)
        else {}
    )
    source_errors = source_binding.get("errors")
    if not isinstance(source_errors, list):
        source_errors = ["source_binding.errors is missing or not a list"]
    record.update(
        {
            "schema_version": payload.get("schema_version"),
            "status": str(payload.get("status") or ""),
            "approval_packet_json": str(payload.get("approval_packet_json") or ""),
            "required_approval_count": payload.get("required_approval_count"),
            "granted_approval_count": payload.get("granted_approval_count"),
            "all_required_approvals_granted": bool(
                payload.get("all_required_approvals_granted")
            ),
            "source_binding": source_binding,
            "paid_api_approval": extract_paid_api_approval(payload),
            "will_execute_external_actions": payload.get("will_execute_external_actions"),
        }
    )
    if payload.get("schema_version") != 1:
        record["errors"].append("schema_version must be 1")
    if payload.get("ok") is not True:
        record["errors"].append("ok must be true")
    if payload.get("status") != "approved":
        record["errors"].append("status must be approved")
    required_count = payload.get("required_approval_count")
    granted_count = payload.get("granted_approval_count")
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
                source_packet_path = (report_path.parent / source_packet_path).resolve()
            expected_resolved = expected_source_packet_path.resolve()
            source_resolved = source_packet_path.resolve()
            record["source_packet_path_matches_expected"] = (
                source_resolved == expected_resolved
            )
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


def resolve_relative_path(path_value: str, *, base_dir: Path) -> Path:
    path = Path(path_value)
    if path.is_absolute():
        return path.resolve()
    return (base_dir / path).resolve()


def build_release_gate_binding_record(
    operator_plan_path: Path,
    operator_plan: dict[str, Any],
    expected_release_gate_json: Path | None,
) -> dict[str, Any]:
    outputs = operator_plan.get("outputs") if isinstance(operator_plan.get("outputs"), dict) else {}
    source_value = (
        operator_plan.get("source_release_gate_json")
        or operator_plan.get("release_gate_json")
        or outputs.get("release_gate_json")
        or ""
    )
    record: dict[str, Any] = {
        "required": expected_release_gate_json is not None,
        "source_release_gate_json": str(source_value),
        "expected_release_gate_json": str(expected_release_gate_json)
        if expected_release_gate_json
        else "",
        "source_path_matches_expected": False,
        "source_exists": False,
        "expected_exists": False,
        "valid": expected_release_gate_json is None,
        "errors": [],
    }
    if isinstance(source_value, str) and source_value.strip():
        source_path = resolve_relative_path(
            source_value,
            base_dir=operator_plan_path.parent,
        )
        record["source_release_gate_json_resolved"] = str(source_path)
        record["source_exists"] = source_path.exists()
    else:
        if expected_release_gate_json is not None:
            record["errors"].append("operator plan source release gate JSON is missing")

    if expected_release_gate_json is not None:
        expected_path = expected_release_gate_json.resolve()
        record["expected_release_gate_json_resolved"] = str(expected_path)
        record["expected_exists"] = expected_path.exists()
        if not expected_path.exists():
            record["errors"].append(
                f"expected release gate JSON does not exist: {expected_release_gate_json}"
            )
        source_resolved = record.get("source_release_gate_json_resolved")
        if isinstance(source_resolved, str) and source_resolved:
            record["source_path_matches_expected"] = source_resolved == str(expected_path)
            if source_resolved != str(expected_path):
                record["errors"].append(
                    "operator plan source release gate JSON does not match "
                    "--release-gate-json"
                )
        record["valid"] = not record["errors"]
    return record


def select_steps(
    operator_plan: dict[str, Any],
    selected_gates: set[str],
    replacements: dict[str, str],
    *,
    external_action_approval_source_packet_json: Path | None = None,
    external_action_approval_report_json: Path | None = None,
) -> list[dict[str, Any]]:
    operator_next_steps = operator_plan.get("operator_next_steps")
    if not isinstance(operator_next_steps, dict):
        raise ValueError("operator_plan.operator_next_steps must be an object")
    raw_steps = operator_next_steps.get("steps")
    if not isinstance(raw_steps, list):
        raise ValueError("operator_plan.operator_next_steps.steps must be a list")

    rendered_steps: list[dict[str, Any]] = []
    for raw_step in raw_steps:
        if not isinstance(raw_step, dict):
            continue
        gate = raw_step.get("gate")
        if selected_gates and gate not in selected_gates:
            continue
        commands = []
        for command in raw_step.get("commands") or []:
            raw_command = str(command)
            rendered_command = apply_replacements(raw_command, replacements)
            rendered_command = bind_template_command_approval_paths(
                raw_command,
                rendered_command,
                external_action_approval_source_packet_json=(
                    external_action_approval_source_packet_json
                ),
                external_action_approval_report_json=(
                    external_action_approval_report_json
                ),
            )
            commands.append(rendered_command)
        evidence = [
            apply_replacements(str(path), replacements)
            for path in raw_step.get("evidence_to_produce") or []
        ]
        unresolved = unresolved_tokens(commands + evidence)
        executable_commands = [command for command in commands if is_executable_command(command)]
        non_executable_notes = [
            command for command in commands if not is_executable_command(command)
        ]
        rendered_steps.append(
            {
                "order": raw_step.get("order"),
                "gate": gate,
                "status": raw_step.get("status"),
                "next_action": raw_step.get("next_action"),
                "requires_paid_api": bool(raw_step.get("requires_paid_api")),
                "requires_wandb_access": bool(raw_step.get("requires_wandb_access")),
                "requires_wandb_write": bool(raw_step.get("requires_wandb_write")),
                "requires_third_party_acceptance": bool(
                    raw_step.get("requires_third_party_acceptance")
                ),
                "requires_nemoclaw_install": bool(raw_step.get("requires_nemoclaw_install")),
                "requires_scope_confirmation": bool(
                    raw_step.get("requires_scope_confirmation")
                ),
                "commands": commands,
                "command_count": len(commands),
                "executable_commands": executable_commands,
                "executable_command_count": len(executable_commands),
                "non_executable_notes": non_executable_notes,
                "non_executable_note_count": len(non_executable_notes),
                "evidence_to_produce": evidence,
                "evidence_path_count": len(evidence),
                "unresolved_placeholder_tokens": unresolved,
                "ready_to_execute_without_placeholder": not unresolved,
                "warnings": raw_step.get("warnings") or [],
            }
        )
    return rendered_steps


def build_execution_plan(
    *,
    operator_plan_path: Path,
    operator_plan: dict[str, Any],
    selected_gates: set[str],
    replacements: dict[str, str],
    external_action_approval_report_json: Path | None = None,
    external_action_approval_source_packet_json: Path | None = None,
    expected_release_gate_json: Path | None = None,
) -> dict[str, Any]:
    steps = select_steps(
        operator_plan,
        selected_gates,
        replacements,
        external_action_approval_source_packet_json=(
            external_action_approval_source_packet_json
        ),
        external_action_approval_report_json=external_action_approval_report_json,
    )
    all_values = [
        value
        for step in steps
        for value in [*step["commands"], *step["evidence_to_produce"]]
    ]
    unresolved = unresolved_tokens(all_values)
    all_ready = bool(steps) and not unresolved
    requirement_counts = {
        "paid_api_steps": count_required(steps, "requires_paid_api"),
        "wandb_access_steps": count_required(steps, "requires_wandb_access"),
        "wandb_write_steps": count_required(steps, "requires_wandb_write"),
        "third_party_acceptance_steps": count_required(
            steps,
            "requires_third_party_acceptance",
        ),
        "nemoclaw_install_steps": count_required(steps, "requires_nemoclaw_install"),
        "scope_confirmation_steps": count_required(
            steps,
            "requires_scope_confirmation",
        ),
    }
    external_action_required = requires_external_action(steps)
    external_action_approval = build_external_action_approval_record(
        external_action_approval_report_json,
        required_before_external_action=external_action_required,
        expected_source_packet_path=external_action_approval_source_packet_json,
    )
    release_gate_binding = build_release_gate_binding_record(
        operator_plan_path,
        operator_plan,
        expected_release_gate_json,
    )
    command_policy = validate_command_policy(
        steps,
        base_dir=Path.cwd(),
        expected_external_action_source_packet_path=(
            external_action_approval_source_packet_json
        ),
        expected_external_action_approval_report_path=(
            external_action_approval_report_json
        ),
    )
    approval_scope_policy = validate_canary_approval_scope(
        steps,
        external_action_approval,
    )
    return {
        "schema_version": 1,
        "generated_at": time.time(),
        "status": "placeholder_ready" if all_ready else "templates_pending_values",
        "source_operator_plan": str(operator_plan_path),
        "source_operator_plan_sha256": sha256_file(operator_plan_path),
        "selected_gates": sorted(selected_gates),
        "replacement_values_supplied": sorted(replacements),
        "unresolved_placeholder_tokens": unresolved,
        "all_ready_to_execute_without_placeholder": all_ready,
        "external_action_approval": external_action_approval,
        "approval_scope_policy": approval_scope_policy,
        "release_gate_binding": release_gate_binding,
        "command_policy": command_policy,
        "all_ready_for_external_execution": all_ready
        and (
            not external_action_required
            or bool(external_action_approval.get("valid"))
        )
        and bool(release_gate_binding.get("valid"))
        and bool(command_policy.get("valid"))
        and bool(approval_scope_policy.get("valid")),
        "no_external_action_performed": True,
        "total_command_count": sum(
            step.get("command_count", 0) for step in steps
        ),
        "total_executable_command_count": sum(
            step.get("executable_command_count", 0) for step in steps
        ),
        "total_evidence_path_count": sum(
            step.get("evidence_path_count", 0) for step in steps
        ),
        "requirement_counts": requirement_counts,
        "steps": steps,
    }


def markdown(plan: dict[str, Any]) -> str:
    lines = [
        "# Taiwan Operator Execution Plan",
        "",
        f"Status: `{plan.get('status')}`",
        f"Source operator plan: `{plan.get('source_operator_plan')}`",
        f"Source operator plan SHA-256: `{plan.get('source_operator_plan_sha256')}`",
        f"No external action performed: `{str(plan.get('no_external_action_performed')).lower()}`",
        "",
        "## Readiness",
        "",
        f"- All placeholders resolved: `{str(plan.get('all_ready_to_execute_without_placeholder')).lower()}`",
        f"- All external approvals verified: `{str(plan.get('all_ready_for_external_execution')).lower()}`",
        f"- Total commands: `{plan.get('total_command_count')}`",
        f"- Total executable commands: `{plan.get('total_executable_command_count')}`",
        f"- Total evidence paths: `{plan.get('total_evidence_path_count')}`",
    ]
    unresolved = plan.get("unresolved_placeholder_tokens")
    if isinstance(unresolved, list) and unresolved:
        lines.append("- Unresolved placeholders: " + ", ".join(f"`{token}`" for token in unresolved))
    else:
        lines.append("- Unresolved placeholders: none")

    counts = plan.get("requirement_counts")
    if isinstance(counts, dict):
        lines.extend(["", "## External Requirements", "", "| Requirement | Steps |", "|---|---:|"])
        for key, label in [
            ("paid_api_steps", "Paid API"),
            ("wandb_access_steps", "W&B access"),
            ("wandb_write_steps", "W&B write"),
            ("third_party_acceptance_steps", "Third-party acceptance"),
            ("nemoclaw_install_steps", "NeMoClaw install"),
            ("scope_confirmation_steps", "Scope confirmation"),
        ]:
            lines.append(f"| {label} | {counts.get(key, 0)} |")

    approval = plan.get("external_action_approval")
    if isinstance(approval, dict):
        lines.extend(
            [
                "",
                "## External Action Approval",
                "",
                f"- Required before external action: `{str(approval.get('required_before_external_action')).lower()}`",
                f"- Present: `{str(approval.get('present')).lower()}`",
                f"- Valid: `{str(approval.get('valid')).lower()}`",
                f"- Report JSON: `{approval.get('path') or ''}`",
                f"- Report SHA-256: `{approval.get('sha256') or ''}`",
                f"- Expected source packet JSON: `{approval.get('expected_source_packet_json') or ''}`",
                f"- Expected source packet SHA-256: `{approval.get('expected_source_packet_sha256') or ''}`",
                f"- Status: `{approval.get('status') or ''}`",
            ]
        )
        source_binding = approval.get("source_binding")
        if isinstance(source_binding, dict):
            lines.append(
                f"- Source bound: `{str(source_binding.get('bound')).lower()}`"
            )
        errors = approval.get("errors")
        if isinstance(errors, list) and errors:
            lines.extend(["", "Approval report errors:", ""])
            lines.extend(f"- {error}" for error in errors)

    release_binding = plan.get("release_gate_binding")
    if isinstance(release_binding, dict):
        lines.extend(
            [
                "",
                "## Release Gate Binding",
                "",
                f"- Required: `{str(release_binding.get('required')).lower()}`",
                f"- Valid: `{str(release_binding.get('valid')).lower()}`",
                f"- Source release gate JSON: `{release_binding.get('source_release_gate_json') or ''}`",
                f"- Expected release gate JSON: `{release_binding.get('expected_release_gate_json') or ''}`",
                f"- Source path matches expected: `{str(release_binding.get('source_path_matches_expected')).lower()}`",
            ]
        )
        binding_errors = release_binding.get("errors")
        if isinstance(binding_errors, list) and binding_errors:
            lines.extend(["", "Release gate binding errors:", ""])
            lines.extend(f"- {error}" for error in binding_errors)

    approval_scope = plan.get("approval_scope_policy")
    if isinstance(approval_scope, dict):
        lines.extend(
            [
                "",
                "## Approval Scope Policy",
                "",
                f"- Required: `{str(approval_scope.get('required')).lower()}`",
                f"- Valid: `{str(approval_scope.get('valid')).lower()}`",
                f"- Expected model scope marker: `{approval_scope.get('expected_model_scope_marker') or ''}`",
                f"- Observed model scope: `{approval_scope.get('observed_model_scope') or ''}`",
            ]
        )
        scope_errors = approval_scope.get("errors")
        if isinstance(scope_errors, list) and scope_errors:
            lines.extend(["", "Approval scope errors:", ""])
            lines.extend(f"- {error}" for error in scope_errors)

    command_policy = plan.get("command_policy")
    if isinstance(command_policy, dict):
        lines.extend(
            [
                "",
                "## Command Policy",
                "",
                f"- Valid: `{str(command_policy.get('valid')).lower()}`",
                f"- Checked commands: `{command_policy.get('checked_command_count')}`",
                f"- Error count: `{command_policy.get('error_count')}`",
            ]
        )
        policy_errors = command_policy.get("errors")
        if isinstance(policy_errors, list) and policy_errors:
            lines.extend(["", "Command policy errors:", ""])
            lines.extend(f"- {error}" for error in policy_errors)

    lines.extend(["", "## Steps", ""])
    steps = plan.get("steps")
    if not isinstance(steps, list) or not steps:
        lines.append("- none")
    else:
        for step in steps:
            if not isinstance(step, dict):
                continue
            required_labels = step_requirement_labels(step)
            lines.extend(
                [
                    f"### {step.get('order')}. {step.get('gate')}",
                    "",
                    f"- Status: `{step.get('status')}`",
                    f"- Ready without placeholder edit: `{str(step.get('ready_to_execute_without_placeholder')).lower()}`",
                    f"- Commands: `{step.get('command_count')}` total, `{step.get('executable_command_count')}` executable, `{step.get('non_executable_note_count')}` notes",
                    f"- Evidence paths: `{step.get('evidence_path_count')}`",
                    "- Requires: " + (", ".join(required_labels) or "none"),
                    f"- Next action: {step.get('next_action') or ''}",
                    "",
                    "Executable commands:",
                    "",
                ]
            )
            executable = step.get("executable_commands")
            if isinstance(executable, list) and executable:
                lines.extend(f"- `{command}`" for command in executable)
            else:
                lines.append("- none")
            notes = step.get("non_executable_notes")
            if isinstance(notes, list) and notes:
                lines.extend(["", "Notes:", ""])
                lines.extend(f"- {note}" for note in notes)
            evidence = step.get("evidence_to_produce")
            if isinstance(evidence, list) and evidence:
                lines.extend(["", "Evidence to produce:", ""])
                lines.extend(f"- `{path}`" for path in evidence)
            lines.append("")
    return "\n".join(lines)


def step_requirement_labels(step: dict[str, Any]) -> list[str]:
    return [
        label
        for label, key in [
            ("paid API", "requires_paid_api"),
            ("W&B access", "requires_wandb_access"),
            ("W&B write", "requires_wandb_write"),
            ("third-party acceptance", "requires_third_party_acceptance"),
            ("NeMoClaw install", "requires_nemoclaw_install"),
            ("scope confirmation", "requires_scope_confirmation"),
        ]
        if step.get(key)
    ]


def shell_script(plan: dict[str, Any]) -> str:
    lines = [
        "#!/usr/bin/env bash",
        "set -euo pipefail",
        "",
        "# Generated by render_taiwan_operator_execution_plan.py.",
        f"# Source operator plan: {plan.get('source_operator_plan') or ''}",
        f"# Source operator plan SHA-256: {plan.get('source_operator_plan_sha256') or ''}",
        "# Review paid API, W&B write, and third-party acceptance requirements before running.",
        "",
    ]
    approval = plan.get("external_action_approval")
    if isinstance(approval, dict) and approval.get("required_before_external_action"):
        lines.extend(
            [
                f"# External action approval report: {approval.get('path') or ''}",
                f"# External action approval SHA-256: {approval.get('sha256') or ''}",
                f"# External action approval source packet: {approval.get('expected_source_packet_json') or ''}",
                f"# External action approval source packet SHA-256: {approval.get('expected_source_packet_sha256') or ''}",
                "",
            ]
        )
    release_binding = plan.get("release_gate_binding")
    if isinstance(release_binding, dict) and release_binding.get("required"):
        lines.extend(
            [
                f"# Release gate JSON: {release_binding.get('expected_release_gate_json') or ''}",
                f"# Release gate binding valid: {str(release_binding.get('valid')).lower()}",
                "",
            ]
        )
    command_policy = plan.get("command_policy")
    if isinstance(command_policy, dict):
        lines.extend(
            [
                f"# Command policy valid: {str(command_policy.get('valid')).lower()}",
                f"# Command policy checked commands: {command_policy.get('checked_command_count')}",
                "",
            ]
        )
    for step in plan.get("steps") or []:
        if not isinstance(step, dict):
            continue
        lines.append(f"# Gate: {step.get('gate')} ({step.get('status')})")
        required_labels = step_requirement_labels(step)
        lines.append("# Requires: " + (", ".join(required_labels) or "none"))
        lines.append(
            f"# Commands: {step.get('command_count')} total, "
            f"{step.get('executable_command_count')} executable, "
            f"{step.get('non_executable_note_count')} notes"
        )
        lines.append(f"# Evidence paths: {step.get('evidence_path_count')}")
        for note in step.get("non_executable_notes") or []:
            lines.append(f"# NOTE: {note}")
        for command in step.get("executable_commands") or []:
            lines.append(command)
        lines.append("")
    return "\n".join(lines)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--operator-plan-json", type=Path, required=True)
    parser.add_argument("--gate", action="append", default=[])
    parser.add_argument("--timestamp")
    parser.add_argument("--content-canary-id")
    parser.add_argument("--weave-content-canary-gate")
    parser.add_argument("--run-id")
    parser.add_argument("--model-slug")
    parser.add_argument("--phase")
    parser.add_argument(
        "--release-gate-json",
        type=Path,
        help=(
            "Expected source release gate JSON for this operator plan. When set, "
            "the operator plan must point back to the same release gate before "
            "--require-ready or shell generation can pass."
        ),
    )
    parser.add_argument(
        "--external-action-approval-report-json",
        type=Path,
        help=(
            "Source-bound verifier report produced by "
            "verify_external_action_approval_packet.py. Required before writing "
            "or requiring a shell plan that executes external actions."
        ),
    )
    parser.add_argument(
        "--external-action-approval-source-packet-json",
        type=Path,
        help=(
            "Source external_action_approval_packet.json that the verifier report "
            "must be bound to. Required for shell plans that execute external "
            "actions, so stale approval reports from another release bundle cannot "
            "authorize this plan."
        ),
    )
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--markdown", type=Path)
    parser.add_argument("--shell-script", type=Path)
    parser.add_argument("--require-ready", action="store_true")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    operator_plan = read_json(args.operator_plan_json)
    replacements = replacement_map(args)
    plan = build_execution_plan(
        operator_plan_path=args.operator_plan_json,
        operator_plan=operator_plan,
        selected_gates=set(args.gate or []),
        replacements=replacements,
        external_action_approval_report_json=args.external_action_approval_report_json,
        external_action_approval_source_packet_json=(
            args.external_action_approval_source_packet_json
        ),
        expected_release_gate_json=args.release_gate_json,
    )
    write_json(args.output_json, plan)
    if args.markdown:
        write_text(args.markdown, markdown(plan))
    if args.shell_script:
        if not plan["all_ready_to_execute_without_placeholder"]:
            print(
                "refusing to write shell script while placeholders remain: "
                + ", ".join(plan["unresolved_placeholder_tokens"]),
                file=sys.stderr,
            )
            return 2
        if not plan["all_ready_for_external_execution"]:
            errors = plan.get("external_action_approval", {}).get("errors", [])
            binding_errors = plan.get("release_gate_binding", {}).get("errors", [])
            command_policy_errors = plan.get("command_policy", {}).get("errors", [])
            approval_scope_errors = plan.get("approval_scope_policy", {}).get(
                "errors",
                [],
            )
            print(
                "refusing to write shell script without valid external-action approval: "
                + ", ".join(
                    str(error)
                    for error in [
                        *errors,
                        *binding_errors,
                        *command_policy_errors,
                        *approval_scope_errors,
                    ]
                ),
                file=sys.stderr,
            )
            return 2
        write_text(args.shell_script, shell_script(plan))
        args.shell_script.chmod(args.shell_script.stat().st_mode | stat.S_IXUSR)
    if args.require_ready and not plan["all_ready_to_execute_without_placeholder"]:
        print(
            "operator execution plan still has unresolved placeholders: "
            + ", ".join(plan["unresolved_placeholder_tokens"]),
            file=sys.stderr,
        )
        return 2
    if args.require_ready and not plan["all_ready_for_external_execution"]:
        errors = plan.get("external_action_approval", {}).get("errors", [])
        binding_errors = plan.get("release_gate_binding", {}).get("errors", [])
        command_policy_errors = plan.get("command_policy", {}).get("errors", [])
        approval_scope_errors = plan.get("approval_scope_policy", {}).get(
            "errors",
            [],
        )
        print(
            "operator execution plan external-action approval is not valid: "
            + ", ".join(
                str(error)
                for error in [
                    *errors,
                    *binding_errors,
                    *command_policy_errors,
                    *approval_scope_errors,
                ]
            ),
            file=sys.stderr,
        )
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
