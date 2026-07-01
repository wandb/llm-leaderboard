#!/usr/bin/env python3
"""Check NeMoClaw adoption readiness for the Taiwan leaderboard.

This command is offline. It reads local setup/readiness/config files and
reports whether NeMoClaw can be used for Taiwan Agentic Math under the ADR.
It does not install NeMoClaw, query W&B, or run model inference.
"""

from __future__ import annotations

import argparse
import json
import re
import shlex
import time
from pathlib import Path
from typing import Any

from omegaconf import OmegaConf

import build_taiwan_production_readiness_report as readiness


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_AGENTIC_CONFIG_GLOBS = (
    "configs/taiwan_full/generated_openai_canary_agentic_nemoclaw/*.yaml",
    "configs/taiwan_full/generated_openai_canary_agentic/*.yaml",
    "configs/taiwan_full/generated_openai_canary_agentic_aggregate/*.yaml",
    "configs/taiwan_full/generated_openai_canary_nonagentic/*.yaml",
    "configs/taiwan_full/generated_openai_canary/*.yaml",
    "configs/taiwan_full/generated_canary_agentic_nemoclaw/*.yaml",
    "configs/taiwan_full/generated_canary_agentic/*.yaml",
    "configs/taiwan_full/generated_canary_agentic_aggregate/*.yaml",
    "configs/taiwan_full/generated_canary_nonagentic/*.yaml",
    "configs/taiwan_full/generated_canary/*.yaml",
    "configs/taiwan_full/generated_agentic_nemoclaw/*.yaml",
    "configs/taiwan_full/generated_agentic/*.yaml",
    "configs/taiwan_full/generated_agentic_aggregate/*.yaml",
    "configs/taiwan_full/generated_nonagentic/*.yaml",
    "configs/taiwan_full/generated/*.yaml",
    "configs/config-taiwan-*.yaml",
    "configs/base_config_taiwan.yaml",
)
RUNTIME_BLOCKER_CRITERIA = {
    "setup_installed",
    "sandbox_readiness",
    "runtime_wandb_weave_policy",
    "runtime_network_policy_allowlist",
}
DESIGN_ADOPTION_CRITERIA = {
    "setup_plan_safety",
    "agentic_math_config",
    "swebench_pro_non_adoption_guard",
}
WANDB_WEAVE_POLICY_CHECK_PREFIX = "NeMoClaw W&B/Weave runtime policy is present:"
NETWORK_POLICY_ALLOWLIST_CHECK_PREFIX = "NeMoClaw runtime network policies are allowlisted:"
REQUIRED_POLICY_TIER = "restricted"
ALLOWED_POLICY_TIERS = ["restricted", "balanced", "open"]
REQUIRED_INSTALLER_LOCK_JSON = "scripts/setup/nemoclaw_installer_lock.json"
INSTALLER_PROVENANCE_NOTE = (
    "Installer integrity is not verified by this script; operator review is required before install/onboard."
)
REQUIRED_PRODUCTION_INSTALL_AND_ONBOARD_MARKERS = (
    "scripts/setup/install_nemoclaw.sh",
    "--install",
    "--onboard",
    "--installer-lock-json",
    "--installer-sha256",
    "--installer-review-json",
    "--yes-i-accept-third-party-software",
    f"--policy-tier {REQUIRED_POLICY_TIER}",
    "--json",
)
REQUIRED_INSTALLER_REVIEW_COMMAND_MARKERS = (
    "scripts/setup/review_nemoclaw_installer.py",
    "--url",
    "--install-ref",
    "--expected-sha256",
    "--lock-json",
    "--json",
    "--markdown",
)
REQUIRED_POST_INSTALL_VERIFICATION_COMMAND_MARKERS = (
    "scripts/setup/verify_nemoclaw_post_install.py",
    "--nemoclaw-openclaw-config-path",
    "--json",
    "--markdown",
    "--fail-on-failed",
)
REQUIRED_ADOPTION_CHECK_COMMAND_MARKERS = (
    "scripts/tools/check_taiwan_nemoclaw_adoption.py",
    "--setup-json",
    "--readiness-json",
    "--sandbox",
    "--agentic-config-glob",
    "--json",
    "--markdown",
)
REQUIRED_OPERATOR_SEQUENCE = (
    (
        "setup_check",
        "post_install_check_command",
        "temp/nemoclaw_setup_check_YYYYMMDDTHHMM.json",
        False,
    ),
    (
        "installer_review",
        "installer_review_command",
        "temp/nemoclaw_installer_review_YYYYMMDDTHHMM.json",
        False,
    ),
    (
        "install_and_onboard",
        "production_install_and_onboard_command",
        "temp/nemoclaw_install_onboard_YYYYMMDDTHHMM.json",
        True,
    ),
    (
        "post_install_verification",
        "post_install_verification_command",
        "temp/nemoclaw_post_install_verification_YYYYMMDDTHHMM.json",
        False,
    ),
    (
        "canary_readiness",
        "canary_readiness_command",
        "outputs/taiwan_full_eval/openai_canary_readiness_nemoclaw.json",
        False,
    ),
    (
        "adoption_check",
        "adoption_check_command",
        "temp/taiwan_nemoclaw_adoption_check_YYYYMMDDTHHMM.json",
        False,
    ),
    (
        "production_readiness",
        "production_readiness_command",
        "outputs/taiwan_full_eval/taiwan_production_readiness_report.json",
        False,
    ),
)
REQUIRED_OPERATOR_EVIDENCE_PATHS = (
    "temp/nemoclaw_setup_check_YYYYMMDDTHHMM.json",
    "temp/nemoclaw_installer_review_YYYYMMDDTHHMM.json",
    "temp/nemoclaw_installer_review_YYYYMMDDTHHMM.md",
    "temp/nemoclaw_install_onboard_YYYYMMDDTHHMM.json",
    "temp/nemoclaw_install_onboard_YYYYMMDDTHHMM.install.log",
    "temp/nemoclaw_install_onboard_YYYYMMDDTHHMM.onboard.log",
    "temp/nemoclaw_post_install_verification_YYYYMMDDTHHMM.json",
    "temp/nemoclaw_post_install_verification_YYYYMMDDTHHMM.md",
    "outputs/taiwan_full_eval/openai_canary_readiness_nemoclaw.json",
    "temp/taiwan_nemoclaw_adoption_check_YYYYMMDDTHHMM.json",
    "temp/taiwan_nemoclaw_adoption_check_YYYYMMDDTHHMM.md",
    "outputs/taiwan_full_eval/taiwan_production_readiness_report.json",
)


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


def read_json(path: Path) -> dict[str, Any] | None:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    return payload if isinstance(payload, dict) else None


def command_parts(command: str) -> list[str]:
    try:
        return shlex.split(command)
    except ValueError:
        return command.split()


def command_flag_value(parts: list[str], flag: str) -> tuple[str | None, bool]:
    for index, part in enumerate(parts):
        if part == flag:
            if index + 1 < len(parts):
                return parts[index + 1], True
            return None, True
        if part.startswith(flag + "="):
            return part.split("=", 1)[1], True
    return None, False


def installer_lock_sha256() -> str:
    lock = read_json(repo_path(REQUIRED_INSTALLER_LOCK_JSON)) or {}
    value = str(lock.get("sha256") or "").strip().lower()
    return value if re.fullmatch(r"[0-9a-f]{64}", value) else ""


def latest_by_mtime(paths: list[Path]) -> Path | None:
    return max(
        paths,
        key=lambda path: path.stat().st_mtime if path.exists() else 0,
        default=None,
    )


def operator_handoff_from_setup_payload(
    payload: dict[str, Any] | None,
    *,
    source_setup_json: str | None,
) -> dict[str, Any]:
    plan = payload.get("setup_plan") if isinstance(payload, dict) else {}
    if not isinstance(plan, dict):
        plan = {}
    raw_steps = plan.get("operator_sequence")
    steps: list[dict[str, Any]] = []
    if isinstance(raw_steps, list):
        for row in raw_steps:
            if not isinstance(row, dict):
                continue
            steps.append(
                {
                    "step": row.get("step") if isinstance(row.get("step"), str) else None,
                    "command": (
                        row.get("command") if isinstance(row.get("command"), str) else None
                    ),
                    "expected_evidence_path": (
                        row.get("expected_evidence_path")
                        if isinstance(row.get("expected_evidence_path"), str)
                        else None
                    ),
                    "requires_external_action": row.get("requires_external_action") is True,
                    "required": row.get("required") is True,
                }
            )
    raw_evidence_paths = plan.get("expected_evidence_paths")
    expected_evidence_paths = (
        [path for path in raw_evidence_paths if isinstance(path, str) and path.strip()]
        if isinstance(raw_evidence_paths, list)
        else []
    )
    command_fields = (
        "post_install_check_command",
        "installer_review_command",
        "install_and_onboard_command",
        "production_install_and_onboard_command",
        "post_install_verification_command",
        "canary_readiness_command",
        "adoption_check_command",
        "production_readiness_command",
    )
    handoff: dict[str, Any] = {
        "available": bool(steps),
        "source_setup_json": source_setup_json,
        "step_count": len(steps),
        "required_step_count": sum(1 for row in steps if row.get("required") is True),
        "external_action_step_count": sum(
            1 for row in steps if row.get("requires_external_action") is True
        ),
        "evidence_path_count": len(expected_evidence_paths),
        "steps": steps,
        "expected_evidence_paths": expected_evidence_paths,
    }
    for field in command_fields:
        value = plan.get(field)
        handoff[field] = value if isinstance(value, str) and value.strip() else None
    return handoff


def operator_handoff_summary(setup_paths: list[Path]) -> dict[str, Any]:
    latest = latest_by_mtime(setup_paths)
    if latest is None or not latest.exists() or not latest.is_file():
        return operator_handoff_from_setup_payload(None, source_setup_json=None)
    return operator_handoff_from_setup_payload(
        read_json(latest),
        source_setup_json=path_display(latest),
    )


def criterion(
    *,
    name: str,
    ok: bool,
    status: str,
    requirement: str,
    evidence_paths: list[Path],
    detail: str,
    next_action: str,
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    row = {
        "name": name,
        "ok": ok,
        "status": status,
        "requirement": requirement,
        "evidence_paths": [path_display(path) for path in evidence_paths],
        "detail": detail,
        "next_action": next_action,
    }
    if extra:
        row.update(extra)
    return row


def bool_field(payload: dict[str, Any], field: str) -> bool | None:
    value = payload.get(field)
    return value if isinstance(value, bool) else None


def int_or_none(value: Any) -> int | None:
    return value if isinstance(value, int) and not isinstance(value, bool) else None


def operation_failure(operation_results: dict[str, Any], operation: str) -> dict[str, Any]:
    record = operation_results.get(operation) if isinstance(operation_results, dict) else {}
    if not isinstance(record, dict):
        return {}
    failure = record.get("failure")
    if not isinstance(failure, dict):
        return {}
    return failure if str(failure.get("failure_kind") or "").strip() else {}


def setup_plan_safety(setup_paths: list[Path]) -> dict[str, Any]:
    latest = latest_by_mtime(setup_paths)
    requirement = (
        "The latest NeMoClaw setup JSON must include a reviewable setup_plan "
        "that does not launch model inference, requires explicit acceptance, "
        "and includes install, onboard, post-install verification, canary "
        "readiness, production readiness commands, and install/onboard "
        "operation result fields. It must also identify provider preflight "
        "state and sandbox configuration separately from runtime readiness. "
        "It must also identify the third-party "
        "software, installer URL/ref, pinned installer lock JSON, acceptance "
        "ledger fields, and the installer integrity/provenance review fields, "
        "and the "
        f"{REQUIRED_POLICY_TIER} policy tier required for production "
        "benchmark sandboxing. All operator commands that produce evidence "
        "must include JSON/report output paths. If install or onboard was "
        "requested or attempted, the setup JSON must also prove explicit "
        "third-party acceptance and include internally consistent operation "
        "results, non-empty operation logs, and integer return codes for "
        "attempted operations. Install/onboard evidence must not be accepted "
            "unless installer review is verified, installer integrity is verified, "
            "and provenance is locked."
    )
    if latest is None:
        return criterion(
            name="setup_plan_safety",
            ok=False,
            status="missing",
            requirement=requirement,
            evidence_paths=[],
            detail="No NeMoClaw setup JSON was found.",
            next_action="Run scripts/setup/install_nemoclaw.sh --check-only --json temp/nemoclaw_setup_check_TIMESTAMP.json.",
        )
    payload = read_json(latest) or {}
    plan = payload.get("setup_plan") if isinstance(payload.get("setup_plan"), dict) else {}
    required_payload_checks = {
        "provider": lambda value: isinstance(value, str) and bool(value),
        "provider_preflight": lambda value: isinstance(value, dict),
        "sandbox_configured": lambda value: isinstance(value, bool),
    }
    missing_payload_fields = [
        field
        for field, predicate in required_payload_checks.items()
        if not predicate(payload.get(field))
    ]
    required_setup_plan_checks = {
        "provider": lambda value: isinstance(value, str) and bool(value),
        "provider_preflight": lambda value: isinstance(value, dict),
        "sandbox_configured": lambda value: isinstance(value, bool),
        "sandbox_readiness_required": lambda value: value is True,
        "third_party_software_name": lambda value: isinstance(value, str) and bool(value),
        "installer_url": lambda value: isinstance(value, str) and bool(value),
        "install_ref": lambda value: isinstance(value, str) and bool(value),
            "installer_sha256": lambda value: isinstance(value, str),
            "installer_signature": lambda value: isinstance(value, str),
            "installer_lock_json": lambda value: isinstance(value, str) and bool(value),
            "installer_review_json": lambda value: isinstance(value, str),
            "installer_review_verified": lambda value: isinstance(value, bool),
            "installer_integrity_verified": lambda value: isinstance(value, bool),
        "installer_provenance_locked": lambda value: isinstance(value, bool),
        "installer_provenance_note": lambda value: isinstance(value, str) and bool(value),
        "policy_tier": lambda value: isinstance(value, str) and bool(value),
        "policy_tier_allowed_values": lambda value: isinstance(value, list),
        "policy_tier_valid": lambda value: isinstance(value, bool),
            "acceptance_ledger_fields": lambda value: isinstance(value, list),
            "installer_review_command": lambda value: isinstance(value, str) and bool(value),
            "install_command": lambda value: isinstance(value, str) and bool(value),
        "onboard_command": lambda value: isinstance(value, str) and bool(value),
        "install_and_onboard_command": lambda value: isinstance(value, str) and bool(value),
        "production_install_and_onboard_command": lambda value: isinstance(value, str)
        and bool(value),
        "post_install_check_command": lambda value: isinstance(value, str) and bool(value),
        "post_install_verification_command": lambda value: isinstance(value, str) and bool(value),
        "canary_readiness_command": lambda value: isinstance(value, str) and bool(value),
        "adoption_check_command": lambda value: isinstance(value, str) and bool(value),
        "production_readiness_command": lambda value: isinstance(value, str) and bool(value),
        "operator_sequence": lambda value: isinstance(value, list) and bool(value),
        "expected_evidence_paths": lambda value: isinstance(value, list) and bool(value),
    }
    missing_fields = [
        field
        for field, predicate in required_setup_plan_checks.items()
        if not predicate(plan.get(field))
    ]
    evidence_output_requirements = {
        "install_command": "--json",
        "onboard_command": "--json",
        "install_and_onboard_command": "--json",
        "production_install_and_onboard_command": "--json",
        "post_install_check_command": "--json",
        "post_install_verification_command": "--json",
        "canary_readiness_command": "--json",
        "adoption_check_command": "--json",
        "production_readiness_command": "--report-json",
    }
    missing_evidence_output_fields = [
        field
        for field, marker in evidence_output_requirements.items()
        if isinstance(plan.get(field), str) and marker not in str(plan.get(field))
    ]
    missing_restricted_policy_command_fields = [
        field
        for field in (
            "onboard_command",
            "install_and_onboard_command",
            "production_install_and_onboard_command",
        )
        if isinstance(plan.get(field), str)
        and f"--policy-tier {REQUIRED_POLICY_TIER}" not in str(plan.get(field))
    ]
    production_command = plan.get("production_install_and_onboard_command")
    missing_production_install_and_onboard_markers: list[str] = []
    production_install_command_consistency_errors: list[str] = []
    if isinstance(production_command, str):
        missing_production_install_and_onboard_markers = [
            marker
            for marker in REQUIRED_PRODUCTION_INSTALL_AND_ONBOARD_MARKERS
            if marker not in production_command
        ]
        if (
            isinstance(plan.get("install_and_onboard_command"), str)
            and production_command != plan.get("install_and_onboard_command")
        ):
            production_install_command_consistency_errors.append(
                "production_install_and_onboard_command must match install_and_onboard_command"
            )
    installer_review_command = plan.get("installer_review_command")
    missing_installer_review_command_markers: list[str] = []
    installer_review_command_errors: list[str] = []
    expected_installer_sha256 = installer_lock_sha256()
    if isinstance(installer_review_command, str):
        missing_installer_review_command_markers = [
            marker
            for marker in REQUIRED_INSTALLER_REVIEW_COMMAND_MARKERS
            if marker not in installer_review_command
        ]
        review_parts = command_parts(installer_review_command)
        expected_sha_arg, expected_sha_present = command_flag_value(
            review_parts,
            "--expected-sha256",
        )
        if not expected_sha_present:
            installer_review_command_errors.append(
                "installer_review_command must include --expected-sha256"
            )
        elif not expected_sha_arg:
            installer_review_command_errors.append(
                "installer_review_command has empty --expected-sha256"
            )
        elif not re.fullmatch(r"[0-9a-f]{64}", str(expected_sha_arg).lower()):
            installer_review_command_errors.append(
                "installer_review_command --expected-sha256 must be 64 lowercase hex"
            )
        elif expected_installer_sha256 and str(expected_sha_arg).lower() != expected_installer_sha256:
            installer_review_command_errors.append(
                f"installer_review_command --expected-sha256 must match {REQUIRED_INSTALLER_LOCK_JSON}"
            )
        lock_json_arg, lock_json_present = command_flag_value(review_parts, "--lock-json")
        if not lock_json_present:
            installer_review_command_errors.append(
                "installer_review_command must include --lock-json"
            )
        elif lock_json_arg != REQUIRED_INSTALLER_LOCK_JSON:
            installer_review_command_errors.append(
                f"installer_review_command --lock-json must be {REQUIRED_INSTALLER_LOCK_JSON}"
            )
    post_install_verification_command = plan.get("post_install_verification_command")
    missing_post_install_verification_command_markers: list[str] = []
    if isinstance(post_install_verification_command, str):
        missing_post_install_verification_command_markers = [
            marker
            for marker in REQUIRED_POST_INSTALL_VERIFICATION_COMMAND_MARKERS
            if marker not in post_install_verification_command
        ]
    adoption_check_command = plan.get("adoption_check_command")
    missing_adoption_check_command_markers: list[str] = []
    if isinstance(adoption_check_command, str):
        missing_adoption_check_command_markers = [
            marker
            for marker in REQUIRED_ADOPTION_CHECK_COMMAND_MARKERS
            if marker not in adoption_check_command
        ]
    operator_sequence = (
        plan.get("operator_sequence")
        if isinstance(plan.get("operator_sequence"), list)
        else []
    )
    operator_sequence_by_step = {
        str(row.get("step")): row
        for row in operator_sequence
        if isinstance(row, dict) and isinstance(row.get("step"), str)
    }
    missing_operator_sequence_steps: list[str] = []
    operator_sequence_errors: list[str] = []
    for step, command_field, expected_path, requires_external_action in REQUIRED_OPERATOR_SEQUENCE:
        row = operator_sequence_by_step.get(step)
        if not isinstance(row, dict):
            missing_operator_sequence_steps.append(step)
            continue
        if row.get("command") != plan.get(command_field):
            operator_sequence_errors.append(
                f"operator_sequence.{step}.command must match setup_plan.{command_field}"
            )
        if row.get("expected_evidence_path") != expected_path:
            operator_sequence_errors.append(
                f"operator_sequence.{step}.expected_evidence_path must be {expected_path}"
            )
        if row.get("requires_external_action") is not requires_external_action:
            operator_sequence_errors.append(
                "operator_sequence."
                f"{step}.requires_external_action must be {str(requires_external_action).lower()}"
            )
        if row.get("required") is not True:
            operator_sequence_errors.append(f"operator_sequence.{step}.required must be true")
    expected_evidence_paths = (
        plan.get("expected_evidence_paths")
        if isinstance(plan.get("expected_evidence_paths"), list)
        else []
    )
    missing_operator_evidence_paths = [
        path
        for path in REQUIRED_OPERATOR_EVIDENCE_PATHS
        if path not in expected_evidence_paths
    ]
    invalid_policy_fields = []
    if payload.get("policy_tier") != REQUIRED_POLICY_TIER:
        invalid_policy_fields.append("policy_tier")
    if payload.get("policy_tier_allowed_values") != ALLOWED_POLICY_TIERS:
        invalid_policy_fields.append("policy_tier_allowed_values")
    if payload.get("policy_tier_valid") is not True:
        invalid_policy_fields.append("policy_tier_valid")
    if plan.get("policy_tier") != REQUIRED_POLICY_TIER:
        invalid_policy_fields.append("setup_plan.policy_tier")
    if plan.get("policy_tier_allowed_values") != ALLOWED_POLICY_TIERS:
        invalid_policy_fields.append("setup_plan.policy_tier_allowed_values")
    if plan.get("policy_tier_valid") is not True:
        invalid_policy_fields.append("setup_plan.policy_tier_valid")
    operation_results = (
        payload.get("operation_results")
        if isinstance(payload.get("operation_results"), dict)
        else {}
    )
    missing_operation_result_fields: list[str] = []
    if not operation_results:
        missing_operation_result_fields.append("operation_results")
    else:
        install_result = (
            operation_results.get("install")
            if isinstance(operation_results.get("install"), dict)
            else {}
        )
        onboard_result = (
            operation_results.get("onboard")
            if isinstance(operation_results.get("onboard"), dict)
            else {}
        )
        if not install_result:
            missing_operation_result_fields.append("operation_results.install")
        else:
            for field in ("requested", "attempted", "returncode", "log_path"):
                if field not in install_result:
                    missing_operation_result_fields.append(f"operation_results.install.{field}")
            for field in ("requested", "attempted"):
                if field in install_result and not isinstance(install_result.get(field), bool):
                    missing_operation_result_fields.append(f"operation_results.install.{field}:bool")
            if "log_path" in install_result and not isinstance(install_result.get("log_path"), str):
                missing_operation_result_fields.append("operation_results.install.log_path:str")
        if not onboard_result:
            missing_operation_result_fields.append("operation_results.onboard")
        else:
            for field in ("requested", "attempted", "skipped", "returncode", "log_path"):
                if field not in onboard_result:
                    missing_operation_result_fields.append(f"operation_results.onboard.{field}")
            for field in ("requested", "attempted", "skipped"):
                if field in onboard_result and not isinstance(onboard_result.get(field), bool):
                    missing_operation_result_fields.append(f"operation_results.onboard.{field}:bool")
            if "log_path" in onboard_result and not isinstance(onboard_result.get("log_path"), str):
                missing_operation_result_fields.append("operation_results.onboard.log_path:str")
    install_result = (
        operation_results.get("install")
        if isinstance(operation_results.get("install"), dict)
        else {}
    )
    onboard_result = (
        operation_results.get("onboard")
        if isinstance(operation_results.get("onboard"), dict)
        else {}
    )
    latest_onboard_failure = operation_failure(operation_results, "onboard")
    provider_preflight = (
        payload.get("provider_preflight")
        if isinstance(payload.get("provider_preflight"), dict)
        else {}
    )
    plan_provider_preflight = (
        plan.get("provider_preflight")
        if isinstance(plan.get("provider_preflight"), dict)
        else {}
    )
    provider_preflight_consistency_errors: list[str] = []
    if provider_preflight and provider_preflight.get("will_launch_probe") is not False:
        provider_preflight_consistency_errors.append(
            "provider_preflight.will_launch_probe must be false"
        )
    if provider_preflight and provider_preflight.get("will_launch_benchmark_inference") is not False:
        provider_preflight_consistency_errors.append(
            "provider_preflight.will_launch_benchmark_inference must be false"
        )
    if provider_preflight and not isinstance(
        provider_preflight.get("credential_available"),
        bool,
    ):
        provider_preflight_consistency_errors.append(
            "provider_preflight.credential_available must be a boolean"
        )
    if provider_preflight and not isinstance(
        provider_preflight.get("ready_for_noninteractive_onboard_preflight"),
        bool,
    ):
        provider_preflight_consistency_errors.append(
            "provider_preflight.ready_for_noninteractive_onboard_preflight must be a boolean"
        )
    if provider_preflight and plan_provider_preflight != provider_preflight:
        provider_preflight_consistency_errors.append(
            "setup_plan.provider_preflight must match top-level provider_preflight"
        )
    if (
        isinstance(payload.get("sandbox_configured"), bool)
        and isinstance(plan.get("sandbox_configured"), bool)
        and payload.get("sandbox_configured") != plan.get("sandbox_configured")
    ):
        provider_preflight_consistency_errors.append(
            "setup_plan.sandbox_configured must match top-level sandbox_configured"
        )
    if (
        isinstance(payload.get("provider"), str)
        and isinstance(plan.get("provider"), str)
        and payload.get("provider") != plan.get("provider")
    ):
        provider_preflight_consistency_errors.append(
            "setup_plan.provider must match top-level provider"
        )
    top_install_requested = bool_field(payload, "install_requested")
    top_onboard_requested = bool_field(payload, "onboard_requested")
    accepted_third_party = bool_field(payload, "accepted_third_party_software")
    install_requested = bool_field(install_result, "requested")
    install_attempted = bool_field(install_result, "attempted")
    onboard_requested = bool_field(onboard_result, "requested")
    onboard_attempted = bool_field(onboard_result, "attempted")
    onboard_skipped = bool_field(onboard_result, "skipped")
    install_returncode = int_or_none(install_result.get("returncode"))
    onboard_returncode = int_or_none(onboard_result.get("returncode"))
    operation_result_consistency_errors: list[str] = []
    if top_install_requested is not None and install_requested is not None:
        if top_install_requested != install_requested:
            operation_result_consistency_errors.append(
                "install_requested does not match operation_results.install.requested"
            )
    if top_onboard_requested is not None and onboard_requested is not None:
        if top_onboard_requested != onboard_requested:
            operation_result_consistency_errors.append(
                "onboard_requested does not match operation_results.onboard.requested"
            )
    if install_requested is True and install_attempted is not True:
        operation_result_consistency_errors.append(
            "operation_results.install.requested requires attempted=true"
        )
    if install_attempted is True:
        if not install_result.get("log_path"):
            operation_result_consistency_errors.append(
                "operation_results.install.attempted requires non-empty log_path"
            )
        if install_returncode is None:
            operation_result_consistency_errors.append(
                "operation_results.install.attempted requires integer returncode"
            )
    if onboard_requested is True:
        if onboard_attempted is not True and onboard_skipped is not True:
            operation_result_consistency_errors.append(
                "operation_results.onboard.requested requires attempted=true or skipped=true"
            )
    if onboard_attempted is True and onboard_skipped is True:
        operation_result_consistency_errors.append(
            "operation_results.onboard cannot be both attempted and skipped"
        )
    if onboard_attempted is True:
        if not onboard_result.get("log_path"):
            operation_result_consistency_errors.append(
                "operation_results.onboard.attempted requires non-empty log_path"
            )
        if onboard_returncode is None:
            operation_result_consistency_errors.append(
                "operation_results.onboard.attempted requires integer returncode"
            )
    if onboard_skipped is True:
        if install_requested is not True:
            operation_result_consistency_errors.append(
                "operation_results.onboard.skipped requires install requested"
            )
        if install_returncode is None or install_returncode == 0:
            operation_result_consistency_errors.append(
                "operation_results.onboard.skipped requires failed install returncode"
            )
    third_party = (
        payload.get("third_party_software")
        if isinstance(payload.get("third_party_software"), dict)
        else {}
    )
    required_third_party_checks = {
        "name": isinstance(third_party.get("name"), str) and bool(third_party.get("name")),
        "vendor": third_party.get("vendor") == "NVIDIA",
        "repository_url": (
            isinstance(third_party.get("repository_url"), str)
            and bool(third_party.get("repository_url"))
        ),
        "documentation_url": (
            isinstance(third_party.get("documentation_url"), str)
            and bool(third_party.get("documentation_url"))
        ),
        "installer_url": (
            isinstance(third_party.get("installer_url"), str)
            and bool(third_party.get("installer_url"))
        ),
        "install_ref": (
            isinstance(third_party.get("install_ref"), str)
            and bool(third_party.get("install_ref"))
        ),
            "installer_sha256": isinstance(third_party.get("installer_sha256"), str),
            "installer_signature": isinstance(third_party.get("installer_signature"), str),
            "installer_lock_json": (
                isinstance(third_party.get("installer_lock_json"), str)
                and bool(third_party.get("installer_lock_json"))
            ),
            "installer_review_json": isinstance(third_party.get("installer_review_json"), str),
            "installer_review_verified": isinstance(
                third_party.get("installer_review_verified"),
                bool,
            ),
            "installer_integrity_verified": isinstance(
                third_party.get("installer_integrity_verified"),
                bool,
        ),
        "installer_provenance_locked": isinstance(
            third_party.get("installer_provenance_locked"),
            bool,
        ),
        "installer_provenance_note": (
            isinstance(third_party.get("installer_provenance_note"), str)
            and bool(third_party.get("installer_provenance_note"))
        ),
        "acceptance_required": third_party.get("acceptance_required") is True,
        "acceptance_flag": third_party.get("acceptance_flag") == "--yes-i-accept-third-party-software",
        "accepted": isinstance(third_party.get("accepted"), bool),
        "install_or_onboard_requested": isinstance(
            third_party.get("install_or_onboard_requested"),
            bool,
        ),
        "operator_review_required_before_install": (
            third_party.get("operator_review_required_before_install") is True
        ),
    }
    missing_third_party_software_fields = [
        field for field, present in required_third_party_checks.items() if not present
    ]
    ledger_fields = (
        plan.get("acceptance_ledger_fields")
        if isinstance(plan.get("acceptance_ledger_fields"), list)
        else []
    )
    required_ledger_fields = [
        "accepted_third_party_software",
        "third_party_software.name",
        "third_party_software.vendor",
        "third_party_software.installer_url",
        "third_party_software.install_ref",
            "third_party_software.installer_sha256",
            "third_party_software.installer_signature",
            "third_party_software.installer_lock_json",
            "third_party_software.installer_review_json",
            "third_party_software.installer_review_verified",
            "third_party_software.installer_integrity_verified",
        "third_party_software.installer_provenance_locked",
        "third_party_software.installer_provenance_note",
        "third_party_software.acceptance_required",
        "third_party_software.acceptance_flag",
        "third_party_software.accepted",
        "policy_tier",
        "policy_tier_allowed_values",
        "policy_tier_valid",
        "setup_plan.policy_tier",
        "setup_plan.policy_tier_allowed_values",
        "setup_plan.policy_tier_valid",
            "setup_plan.installer_sha256",
            "setup_plan.installer_signature",
            "setup_plan.installer_lock_json",
            "setup_plan.installer_review_json",
            "setup_plan.installer_review_verified",
            "setup_plan.installer_integrity_verified",
        "setup_plan.installer_provenance_locked",
        "setup_plan.installer_provenance_note",
        "operation_results.install.log_path",
        "operation_results.onboard.log_path",
    ]
    missing_acceptance_ledger_fields = [
        field for field in required_ledger_fields if field not in ledger_fields
    ]
    any_requested_or_attempted = any(
        value is True
        for value in (
            top_install_requested,
            top_onboard_requested,
            install_requested,
            onboard_requested,
            install_attempted,
            onboard_attempted,
        )
    )
    third_party_accepted = bool_field(third_party, "accepted")
    third_party_requested = bool_field(third_party, "install_or_onboard_requested")
    expected_requested = bool(top_install_requested or top_onboard_requested)
    acceptance_consistency_errors: list[str] = []
    if accepted_third_party is None:
        acceptance_consistency_errors.append(
            "accepted_third_party_software must be a boolean"
        )
    if third_party_accepted is None:
        acceptance_consistency_errors.append("third_party_software.accepted must be a boolean")
    elif accepted_third_party is not None and third_party_accepted != accepted_third_party:
        acceptance_consistency_errors.append(
            "third_party_software.accepted does not match accepted_third_party_software"
        )
    if third_party_requested is None:
        acceptance_consistency_errors.append(
            "third_party_software.install_or_onboard_requested must be a boolean"
        )
    elif third_party_requested != expected_requested:
        acceptance_consistency_errors.append(
            "third_party_software.install_or_onboard_requested does not match requested operations"
        )
    if any_requested_or_attempted and accepted_third_party is not True:
        acceptance_consistency_errors.append(
            "install/onboard requested or attempted without accepted_third_party_software=true"
        )
    if any_requested_or_attempted and third_party_accepted is not True:
        acceptance_consistency_errors.append(
            "install/onboard requested or attempted without third_party_software.accepted=true"
        )
    installer_provenance_consistency_errors: list[str] = []
    for field in (
        "installer_url",
        "install_ref",
        "installer_sha256",
        "installer_signature",
        "installer_lock_json",
        "installer_review_json",
        "installer_review_verified",
        "installer_integrity_verified",
        "installer_provenance_locked",
        "installer_provenance_note",
    ):
        if field in third_party or field in plan:
            if third_party.get(field) != plan.get(field):
                installer_provenance_consistency_errors.append(
                    f"third_party_software.{field} does not match setup_plan.{field}"
                )
    if third_party.get("installer_provenance_note") != INSTALLER_PROVENANCE_NOTE:
        installer_provenance_consistency_errors.append(
            "third_party_software.installer_provenance_note has unexpected value"
        )
    if plan.get("installer_provenance_note") != INSTALLER_PROVENANCE_NOTE:
        installer_provenance_consistency_errors.append(
            "setup_plan.installer_provenance_note has unexpected value"
        )
    for source_name, source in (
        ("third_party_software", third_party),
        ("setup_plan", plan),
    ):
        if (
            isinstance(source.get("installer_lock_json"), str)
            and source.get("installer_lock_json")
            and source.get("installer_lock_json") != REQUIRED_INSTALLER_LOCK_JSON
        ):
            installer_provenance_consistency_errors.append(
                f"{source_name}.installer_lock_json must be {REQUIRED_INSTALLER_LOCK_JSON}"
            )
        if source.get("installer_review_verified") is True and not source.get("installer_lock_json"):
            installer_provenance_consistency_errors.append(
                f"{source_name}.installer_review_verified=true requires non-empty installer_lock_json"
            )
        if source.get("installer_review_verified") is True and not source.get("installer_review_json"):
            installer_provenance_consistency_errors.append(
                f"{source_name}.installer_review_verified=true requires non-empty installer_review_json"
            )
        if source.get("installer_integrity_verified") is True and not source.get("installer_sha256"):
            installer_provenance_consistency_errors.append(
                f"{source_name}.installer_integrity_verified=true requires non-empty installer_sha256"
            )
        if source.get("installer_integrity_verified") is True and source.get("installer_review_verified") is not True:
            installer_provenance_consistency_errors.append(
                f"{source_name}.installer_integrity_verified=true requires installer_review_verified=true"
            )
        if source.get("installer_provenance_locked") is True and source.get("installer_integrity_verified") is not True:
            installer_provenance_consistency_errors.append(
                f"{source_name}.installer_provenance_locked=true requires installer_integrity_verified=true"
            )
        if source.get("installer_provenance_locked") is True and not source.get("installer_sha256"):
            installer_provenance_consistency_errors.append(
                f"{source_name}.installer_provenance_locked=true requires non-empty installer_sha256"
            )
        if source.get("installer_provenance_locked") is True and not source.get("installer_lock_json"):
            installer_provenance_consistency_errors.append(
                f"{source_name}.installer_provenance_locked=true requires non-empty installer_lock_json"
            )
    if any_requested_or_attempted:
        for source_name, source in (
            ("third_party_software", third_party),
            ("setup_plan", plan),
        ):
            if source.get("installer_review_verified") is not True:
                installer_provenance_consistency_errors.append(
                    f"install/onboard requested or attempted without {source_name}.installer_review_verified=true"
                )
            if not source.get("installer_lock_json"):
                installer_provenance_consistency_errors.append(
                    f"install/onboard requested or attempted without {source_name}.installer_lock_json"
                )
            if source.get("installer_integrity_verified") is not True:
                installer_provenance_consistency_errors.append(
                    f"install/onboard requested or attempted without {source_name}.installer_integrity_verified=true"
                )
            if source.get("installer_provenance_locked") is not True:
                installer_provenance_consistency_errors.append(
                    f"install/onboard requested or attempted without {source_name}.installer_provenance_locked=true"
                )
    ok = (
        plan.get("will_launch_model_inference") is False
        and plan.get("install_or_onboard_requires_explicit_acceptance") is True
        and plan.get("acceptance_flag") == "--yes-i-accept-third-party-software"
        and plan.get("third_party_software_name") == third_party.get("name")
        and isinstance(plan.get("installer_url"), str)
        and bool(plan.get("installer_url"))
        and isinstance(plan.get("install_ref"), str)
        and bool(plan.get("install_ref"))
        and not missing_fields
        and not missing_payload_fields
        and not missing_evidence_output_fields
        and not missing_restricted_policy_command_fields
        and not missing_production_install_and_onboard_markers
        and not production_install_command_consistency_errors
        and not missing_installer_review_command_markers
        and not installer_review_command_errors
        and not missing_post_install_verification_command_markers
        and not missing_adoption_check_command_markers
        and not missing_operator_sequence_steps
        and not operator_sequence_errors
        and not missing_operator_evidence_paths
        and not invalid_policy_fields
        and not missing_operation_result_fields
        and not operation_result_consistency_errors
        and not missing_third_party_software_fields
        and not missing_acceptance_ledger_fields
        and not acceptance_consistency_errors
        and not installer_provenance_consistency_errors
        and not provider_preflight_consistency_errors
        and "verify_nemoclaw_post_install.py" in str(plan.get("post_install_verification_command", ""))
        and "--require-nemoclaw" in str(plan.get("canary_readiness_command", ""))
    )
    return criterion(
        name="setup_plan_safety",
        ok=ok,
        status="passed" if ok else "invalid_setup_plan",
        requirement=requirement,
        evidence_paths=[latest],
        detail=(
            "The setup plan is explicit and does not launch inference."
            if ok
            else "The setup plan is missing required safety or verification fields."
        ),
        next_action=(
            "Review setup_plan before install/onboard."
            if ok
            else "Refresh setup JSON with scripts/setup/install_nemoclaw.sh --check-only --json."
        ),
        extra={
            "latest_setup_plan": plan,
            "latest_provider_preflight": provider_preflight,
            "latest_sandbox_configured": payload.get("sandbox_configured"),
            "latest_onboard_failure": latest_onboard_failure,
            "latest_third_party_software": third_party,
            "latest_operation_results": operation_results,
            "missing_payload_fields": missing_payload_fields,
            "missing_setup_plan_fields": missing_fields,
            "missing_evidence_output_fields": missing_evidence_output_fields,
            "missing_restricted_policy_command_fields": missing_restricted_policy_command_fields,
            "missing_production_install_and_onboard_markers": missing_production_install_and_onboard_markers,
            "production_install_command_consistency_errors": production_install_command_consistency_errors,
            "missing_installer_review_command_markers": missing_installer_review_command_markers,
            "installer_review_command_errors": installer_review_command_errors,
            "missing_post_install_verification_command_markers": (
                missing_post_install_verification_command_markers
            ),
            "missing_adoption_check_command_markers": missing_adoption_check_command_markers,
            "expected_installer_sha256": expected_installer_sha256,
            "missing_operator_sequence_steps": missing_operator_sequence_steps,
            "operator_sequence_errors": operator_sequence_errors,
            "missing_operator_evidence_paths": missing_operator_evidence_paths,
            "invalid_policy_fields": invalid_policy_fields,
            "missing_operation_result_fields": missing_operation_result_fields,
            "operation_result_consistency_errors": operation_result_consistency_errors,
            "missing_third_party_software_fields": missing_third_party_software_fields,
            "missing_acceptance_ledger_fields": missing_acceptance_ledger_fields,
            "acceptance_consistency_errors": acceptance_consistency_errors,
            "installer_provenance_consistency_errors": installer_provenance_consistency_errors,
            "provider_preflight_consistency_errors": provider_preflight_consistency_errors,
        },
    )


def setup_installed(setup_paths: list[Path]) -> dict[str, Any]:
    latest = latest_by_mtime(setup_paths)
    requirement = (
        "The latest NeMoClaw setup JSON must show Docker ready and both "
        "nemoclaw and openshell commands available."
    )
    if latest is None:
        return criterion(
            name="setup_installed",
            ok=False,
            status="missing",
            requirement=requirement,
            evidence_paths=[],
            detail="No NeMoClaw setup JSON was found.",
            next_action="Run the setup check, then install/onboard NeMoClaw with explicit acceptance if commands are missing.",
        )
    payload = read_json(latest) or {}
    commands = payload.get("commands") if isinstance(payload.get("commands"), dict) else {}
    docker = commands.get("docker") if isinstance(commands.get("docker"), dict) else {}
    nemoclaw = commands.get("nemoclaw") if isinstance(commands.get("nemoclaw"), dict) else {}
    openshell = commands.get("openshell") if isinstance(commands.get("openshell"), dict) else {}
    host_prerequisites_ok = payload.get("host_prerequisites_ok")
    if not isinstance(host_prerequisites_ok, bool):
        host_prerequisites_ok = docker.get("available") is True and docker.get("info_ok") is True
    runtime_installed = payload.get("runtime_installed")
    if not isinstance(runtime_installed, bool):
        runtime_installed = nemoclaw.get("available") is True and openshell.get("available") is True
    ok = (
        payload.get("ok") is True
        and docker.get("available") is True
        and docker.get("info_ok") is True
        and nemoclaw.get("available") is True
        and openshell.get("available") is True
    )
    missing = [
        name
        for name, command in (
            ("docker", docker),
            ("nemoclaw", nemoclaw),
            ("openshell", openshell),
        )
        if command.get("available") is not True
    ]
    if docker.get("available") is True and docker.get("info_ok") is not True:
        missing.append("docker_info")
    missing_required_commands = payload.get("missing_required_commands")
    if not isinstance(missing_required_commands, list):
        missing_required_commands = missing
    operation_results = (
        payload.get("operation_results")
        if isinstance(payload.get("operation_results"), dict)
        else {}
    )
    return criterion(
        name="setup_installed",
        ok=ok,
        status="passed" if ok else "missing_or_not_ready",
        requirement=requirement,
        evidence_paths=[latest],
        detail=(
            "NeMoClaw/OpenShell setup check passed."
            if ok
            else "Missing or not-ready setup components: " + ", ".join(missing or ["unknown"])
        ),
        next_action=(
            "Run sandbox readiness preflight."
            if ok
            else "Install/onboard NeMoClaw and OpenShell, then rerun --check-only."
        ),
        extra={
            "host_prerequisites_ok": host_prerequisites_ok,
            "runtime_installed": runtime_installed,
            "sandbox_configured": payload.get("sandbox_configured"),
            "provider": payload.get("provider"),
            "provider_preflight": (
                payload.get("provider_preflight")
                if isinstance(payload.get("provider_preflight"), dict)
                else {}
            ),
            "latest_onboard_failure": operation_failure(operation_results, "onboard"),
            "missing_required_commands": missing_required_commands,
            "missing_components": missing,
            "latest_setup_commands": commands,
        },
    )


def sandbox_readiness(readiness_paths: list[Path], setup_paths: list[Path]) -> dict[str, Any]:
    gate = readiness.evaluate_nemoclaw_readiness(
        readiness_paths,
        require=True,
        setup_paths=setup_paths,
    )
    return criterion(
        name="sandbox_readiness",
        ok=bool(gate.get("ok")),
        status=str(gate.get("status") or ""),
        requirement=str(gate.get("requirement") or ""),
        evidence_paths=[repo_path(path) for path in gate.get("evidence_paths", []) if isinstance(path, str)],
        detail=str(gate.get("detail") or ""),
        next_action=str(gate.get("next_action") or ""),
        extra={
            "readiness_gate": gate,
        },
    )


def _readiness_check_by_prefix(payload: dict[str, Any], prefix: str) -> dict[str, Any] | None:
    checks = payload.get("checks")
    if not isinstance(checks, list):
        return None
    for row in checks:
        if isinstance(row, dict) and str(row.get("name") or "").startswith(prefix):
            return row
    return None


def _json_detail(value: Any) -> dict[str, Any]:
    if isinstance(value, dict):
        return value
    if not isinstance(value, str) or not value.strip():
        return {}
    try:
        payload = json.loads(value)
    except json.JSONDecodeError:
        return {}
    return payload if isinstance(payload, dict) else {}


def runtime_wandb_weave_policy(
    readiness_paths: list[Path],
    setup_paths: list[Path],
) -> dict[str, Any]:
    requirement = (
        "The accepted NeMoClaw canary readiness JSON must prove that the sandbox "
        "has the W&B/Weave runtime network policy applied."
    )
    readiness_gate = readiness.evaluate_nemoclaw_readiness(
        readiness_paths,
        require=True,
        setup_paths=setup_paths,
    )
    if readiness_gate.get("ok") is not True:
        return criterion(
            name="runtime_wandb_weave_policy",
            ok=False,
            status="readiness_not_passed",
            requirement=requirement,
            evidence_paths=[
                repo_path(path)
                for path in readiness_gate.get("evidence_paths", [])
                if isinstance(path, str)
            ],
            detail=(
                "No accepted readiness JSON is available for W&B/Weave runtime "
                "policy validation."
            ),
            next_action=(
                "Rerun check_taiwan_canary_readiness.py --require-nemoclaw "
                "after NeMoClaw setup and policy application."
            ),
            extra={"readiness_gate": readiness_gate},
        )

    evidence_paths = [
        repo_path(path)
        for path in readiness_gate.get("evidence_paths", [])
        if isinstance(path, str)
    ]
    latest = latest_by_mtime(evidence_paths)
    payload = read_json(latest) if latest is not None else None
    check = (
        _readiness_check_by_prefix(payload, WANDB_WEAVE_POLICY_CHECK_PREFIX)
        if isinstance(payload, dict)
        else None
    )
    detail_payload = _json_detail(check.get("detail")) if isinstance(check, dict) else {}
    policies = detail_payload.get("policies")
    if not isinstance(policies, list):
        policies = []
    detailed_status_network_policies = detail_payload.get("detailed_status_network_policies")
    if not isinstance(detailed_status_network_policies, list):
        detailed_status_network_policies = []
    policy_count = int_or_none(detail_payload.get("policy_count"))
    detailed_policy_count = int_or_none(
        detail_payload.get("detailed_status_network_policy_count")
    )
    ok = (
        isinstance(check, dict)
        and check.get("ok") is True
        and detail_payload.get("wandb_weave_policy_present") is True
        and "wandb-weave" in {str(item) for item in policies}
        and isinstance(detailed_policy_count, int)
        and detailed_policy_count > 0
    )
    status = "passed" if ok else "missing_or_invalid_policy_evidence"
    return criterion(
        name="runtime_wandb_weave_policy",
        ok=ok,
        status=status,
        requirement=requirement,
        evidence_paths=evidence_paths,
        detail=(
            "NeMoClaw W&B/Weave runtime policy evidence is present."
            if ok
            else "The accepted readiness JSON does not prove W&B/Weave runtime policy presence."
        ),
        next_action=(
            "Keep the same sandbox policy when running Agentic Math/SWE."
            if ok
            else (
                "Apply configs/nemoclaw/policies/wandb_weave.yaml to the sandbox, "
                "rerun canary readiness, then rerun the adoption check."
            )
        ),
        extra={
            "readiness_gate": readiness_gate,
            "source_check_name": check.get("name") if isinstance(check, dict) else "",
            "source_check_ok": check.get("ok") if isinstance(check, dict) else None,
            "wandb_weave_policy_present": detail_payload.get("wandb_weave_policy_present"),
            "policy_count": policy_count,
            "policies": [str(item) for item in policies],
            "summary_policy_count": int_or_none(detail_payload.get("summary_policy_count")),
            "summary_policies": (
                detail_payload.get("summary_policies")
                if isinstance(detail_payload.get("summary_policies"), list)
                else []
            ),
            "detailed_status_network_policy_count": detailed_policy_count,
            "detailed_status_network_policies": [
                str(item) for item in detailed_status_network_policies
            ],
            "non_wandb_network_policies": (
                detail_payload.get("non_wandb_network_policies")
                if isinstance(detail_payload.get("non_wandb_network_policies"), list)
                else []
            ),
        },
    )


def runtime_network_policy_allowlist(
    readiness_paths: list[Path],
    setup_paths: list[Path],
) -> dict[str, Any]:
    requirement = (
        "The accepted NeMoClaw canary readiness JSON must prove that detailed "
        "runtime network policies contain only the approved NeMoClaw/OpenClaw, "
        "managed inference, NVIDIA endpoint, npm bootstrap, and W&B/Weave entries."
    )
    readiness_gate = readiness.evaluate_nemoclaw_readiness(
        readiness_paths,
        require=True,
        setup_paths=setup_paths,
    )
    if readiness_gate.get("ok") is not True:
        return criterion(
            name="runtime_network_policy_allowlist",
            ok=False,
            status="readiness_not_passed",
            requirement=requirement,
            evidence_paths=[
                repo_path(path)
                for path in readiness_gate.get("evidence_paths", [])
                if isinstance(path, str)
            ],
            detail=(
                "No accepted readiness JSON is available for runtime network "
                "policy allowlist validation."
            ),
            next_action=(
                "Rerun check_taiwan_canary_readiness.py --require-nemoclaw "
                "after NeMoClaw setup and policy application."
            ),
            extra={"readiness_gate": readiness_gate},
        )

    evidence_paths = [
        repo_path(path)
        for path in readiness_gate.get("evidence_paths", [])
        if isinstance(path, str)
    ]
    latest = latest_by_mtime(evidence_paths)
    payload = read_json(latest) if latest is not None else None
    check = (
        _readiness_check_by_prefix(payload, NETWORK_POLICY_ALLOWLIST_CHECK_PREFIX)
        if isinstance(payload, dict)
        else None
    )
    detail_payload = _json_detail(check.get("detail")) if isinstance(check, dict) else {}
    unknown_policies = detail_payload.get("unknown_runtime_network_policies")
    if not isinstance(unknown_policies, list):
        unknown_policies = []
    detailed_status_network_policies = detail_payload.get("detailed_status_network_policies")
    if not isinstance(detailed_status_network_policies, list):
        detailed_status_network_policies = []
    ok = (
        isinstance(check, dict)
        and check.get("ok") is True
        and detail_payload.get("runtime_network_policy_allowlist_ok") is True
        and not unknown_policies
        and int_or_none(detail_payload.get("detailed_status_network_policy_count")) is not None
    )
    status = "passed" if ok else "unknown_runtime_network_policy"
    return criterion(
        name="runtime_network_policy_allowlist",
        ok=ok,
        status=status,
        requirement=requirement,
        evidence_paths=evidence_paths,
        detail=(
            "NeMoClaw runtime network policies are within the approved allowlist."
            if ok
            else "The accepted readiness JSON does not prove a clean runtime network policy allowlist."
        ),
        next_action=(
            "Keep the same sandbox policy when running Agentic Math/SWE."
            if ok
            else (
                "Remove unknown runtime network policies from the NeMoClaw sandbox "
                "or explicitly review and add them to the allowlist before production execution."
            )
        ),
        extra={
            "readiness_gate": readiness_gate,
            "source_check_name": check.get("name") if isinstance(check, dict) else "",
            "source_check_ok": check.get("ok") if isinstance(check, dict) else None,
            "runtime_network_policy_allowlist_ok": detail_payload.get(
                "runtime_network_policy_allowlist_ok"
            ),
            "unknown_runtime_network_policies": [str(item) for item in unknown_policies],
            "allowed_runtime_network_policies": (
                detail_payload.get("allowed_runtime_network_policies")
                if isinstance(detail_payload.get("allowed_runtime_network_policies"), list)
                else []
            ),
            "detailed_status_network_policy_count": int_or_none(
                detail_payload.get("detailed_status_network_policy_count")
            ),
            "detailed_status_network_policies": [
                str(item) for item in detailed_status_network_policies
            ],
            "non_wandb_network_policies": (
                detail_payload.get("non_wandb_network_policies")
                if isinstance(detail_payload.get("non_wandb_network_policies"), list)
                else []
            ),
        },
    )


def load_config(path: Path) -> dict[str, Any] | None:
    try:
        payload = OmegaConf.to_container(OmegaConf.load(path), resolve=True)
    except Exception:
        return None
    return payload if isinstance(payload, dict) else None


def config_records(config_paths: list[Path], *, sandbox: str) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for path in config_paths:
        payload = load_config(path)
        if payload is None:
            records.append({"path": path_display(path), "valid": False})
            continue
        agentic_math = payload.get("agentic_math") if isinstance(payload.get("agentic_math"), dict) else {}
        swebench_pro = payload.get("swebench_pro") if isinstance(payload.get("swebench_pro"), dict) else {}
        run = payload.get("run") if isinstance(payload.get("run"), dict) else {}
        swebench_nemoclaw_keys = sorted(
            key
            for key in swebench_pro
            if str(key).startswith("nemoclaw")
        )
        records.append(
            {
                "path": path_display(path),
                "valid": True,
                "agentic_math_enabled": run.get("agentic_math") is True,
                "agentic_math_nemoclaw_sandbox": agentic_math.get("nemoclaw_sandbox"),
                "agentic_math_use_task_agent": agentic_math.get("use_task_agent"),
                "agentic_math_matches_sandbox": agentic_math.get("nemoclaw_sandbox") == sandbox,
                "swebench_pro_enabled": run.get("swebench_pro") is True,
                "swebench_pro_nemoclaw_keys": swebench_nemoclaw_keys,
                "swebench_pro_nemoclaw_sandbox": swebench_pro.get("nemoclaw_sandbox"),
                "swebench_pro_matches_sandbox": swebench_pro.get("nemoclaw_sandbox") == sandbox,
            }
        )
    return records


def agentic_math_config_ready(config_paths: list[Path], *, sandbox: str) -> dict[str, Any]:
    requirement = (
        "At least one Agentic Math config must enable NeMoClaw for the target "
        "sandbox and keep the task-agent path enabled."
    )
    records = config_records(config_paths, sandbox=sandbox)
    passing = [
        record
        for record in records
        if record.get("valid")
        and record.get("agentic_math_enabled")
        and record.get("agentic_math_matches_sandbox")
        and record.get("agentic_math_use_task_agent") is not False
    ]
    ok = bool(passing)
    return criterion(
        name="agentic_math_config",
        ok=ok,
        status="passed" if ok else "missing_nemoclaw_agentic_math_config",
        requirement=requirement,
        evidence_paths=[repo_path(record["path"]) for record in passing if isinstance(record.get("path"), str)],
        detail=(
            "Agentic Math has a NeMoClaw-ready generated config."
            if ok
            else "No generated Agentic Math config is ready for NeMoClaw."
        ),
        next_action=(
            "Use the passing config for the Agentic Math canary."
            if ok
            else (
                "Run prepare_taiwan_full_eval_configs.py --canary --phase agentic "
                f"--agentic-math-nemoclaw-sandbox {sandbox}; do not disable task-agent mode."
            )
        ),
        extra={"records": records},
    )


def swebench_guard(config_paths: list[Path], *, sandbox: str) -> dict[str, Any]:
    requirement = (
        "SWE-Bench Pro configs may route through NeMoClaw only when they set "
        "nemoclaw_sandbox to the target sandbox. Checkout sharing and patch "
        "capture must be handled by the SWE runner."
    )
    records = config_records(config_paths, sandbox=sandbox)
    offending = [
        record
        for record in records
        if record.get("valid")
        and record.get("swebench_pro_nemoclaw_keys")
        and not record.get("swebench_pro_matches_sandbox")
    ]
    passing = [
        record
        for record in records
        if record.get("valid")
        and record.get("swebench_pro_enabled")
        and record.get("swebench_pro_nemoclaw_keys")
        and record.get("swebench_pro_matches_sandbox")
    ]
    ok = bool(records) and not offending
    configured_count = sum(
        1
        for record in records
        if record.get("valid") and record.get("swebench_pro_nemoclaw_keys")
    )
    return criterion(
        name="swebench_pro_non_adoption_guard",
        ok=ok,
        status=(
            "passed"
            if ok
            else "swebench_pro_invalid_nemoclaw_config"
            if offending
            else "missing_configs"
        ),
        requirement=requirement,
        evidence_paths=[repo_path(record["path"]) for record in records if isinstance(record.get("path"), str)],
        detail=(
            "At least one enabled SWE-Bench Pro config is NeMoClaw-ready for the target sandbox."
            if passing
            else "No checked SWE-Bench Pro config contains NeMoClaw fields; SWE remains on host OpenClaw."
            if ok
            else "At least one SWE-Bench Pro config has incomplete or mismatched NeMoClaw fields."
            if offending
            else "No generated configs were available to verify."
        ),
        next_action=(
            "Use the passing SWE-Bench Pro config for the NeMoClaw canary."
            if passing
            else f"Optionally generate SWE configs with --swebench-pro-nemoclaw-sandbox {sandbox}."
            if ok
            else f"Set swebench_pro.nemoclaw_sandbox to {sandbox}, or remove partial NeMoClaw fields."
            if offending
            else "Generate or pass the Agentic Taiwan configs to inspect."
        ),
        extra={
            "records": records,
            "offending_records": offending,
            "passing_records": passing,
            "swebench_pro_nemoclaw_config_count": configured_count,
            "swebench_pro_nemoclaw_ready": bool(passing),
        },
    )


def discover_config_paths(paths: list[Path] | None, globs: list[str] | None = None) -> list[Path]:
    if paths:
        return [repo_path(path) for path in paths]
    return readiness.discover_paths(tuple(globs or DEFAULT_AGENTIC_CONFIG_GLOBS))


def adoption_decision(
    *,
    criteria: list[dict[str, Any]],
    setup_paths: list[Path],
    sandbox: str,
) -> dict[str, Any]:
    by_name = {
        str(row.get("name")): row
        for row in criteria
        if isinstance(row, dict) and row.get("name")
    }
    blockers = [name for name, row in by_name.items() if not row.get("ok")]
    runtime_blockers = [name for name in blockers if name in RUNTIME_BLOCKER_CRITERIA]
    design_blockers = [name for name in blockers if name in DESIGN_ADOPTION_CRITERIA]
    other_blockers = [
        name
        for name in blockers
        if name not in RUNTIME_BLOCKER_CRITERIA and name not in DESIGN_ADOPTION_CRITERIA
    ]
    design_ready = all(
        bool(by_name.get(name, {}).get("ok")) for name in DESIGN_ADOPTION_CRITERIA
    )
    ready_for_use = not blockers
    swebench_pro_ready = bool(
        by_name.get("swebench_pro_non_adoption_guard", {}).get("swebench_pro_nemoclaw_ready")
    )
    scope = "agentic_math_and_swebench_pro" if swebench_pro_ready else "agentic_math_only"
    if ready_for_use:
        recommendation = (
            "adopt_for_agentic_benchmarks"
            if swebench_pro_ready
            else "adopt_for_agentic_math"
        )
        rationale = (
            "All adoption criteria pass; NeMoClaw can be used for Taiwan Agentic Math and SWE-Bench Pro."
            if swebench_pro_ready
            else "All adoption criteria pass; NeMoClaw can be used for Taiwan Agentic Math."
        )
        next_action = (
            "Run the Agentic Math and SWE-Bench Pro canaries through the NeMoClaw configs and keep W&B/Weave proof."
            if swebench_pro_ready
            else "Run the Agentic Math canary through the NeMoClaw config and keep W&B/Weave proof."
        )
    elif setup_paths and design_ready and blockers and not design_blockers and not other_blockers:
        recommendation = (
            "conditional_adopt_for_agentic_benchmarks"
            if swebench_pro_ready
            else "conditional_adopt_for_agentic_math"
        )
        rationale = (
            "Design and policy criteria pass, but local runtime installation or sandbox "
            "readiness is not proven yet."
        )
        next_action = (
            f"Install/onboard NeMoClaw for sandbox {sandbox} with explicit acceptance, "
            "then rerun post-install verification and the release gate."
        )
    elif not setup_paths:
        recommendation = "insufficient_evidence"
        rationale = "No NeMoClaw setup evidence was provided or discovered."
        next_action = "Run install_nemoclaw.sh --check-only and rerun the adoption doctor."
    else:
        recommendation = "do_not_adopt_until_remediated"
        rationale = "One or more design, policy, or config criteria failed."
        next_action = "Fix the listed non-runtime blockers before claiming NeMoClaw adoption."
    return {
        "recommendation": recommendation,
        "ready_for_use": ready_for_use,
        "scope": scope,
        "sandbox": sandbox,
        "design_ready": design_ready,
        "runtime_blockers": runtime_blockers,
        "design_blockers": design_blockers,
        "other_blockers": other_blockers,
        "blockers": blockers,
        "rationale": rationale,
        "next_action": next_action,
    }


def build_report(args: argparse.Namespace) -> dict[str, Any]:
    setup_paths = (
        [repo_path(path) for path in args.setup_json]
        if args.setup_json
        else readiness.discover_paths((readiness.DEFAULT_NEMOCLAW_SETUP_GLOB,))
    )
    readiness_paths = (
        [repo_path(path) for path in args.readiness_json]
        if args.readiness_json
        else readiness.filter_canary_readiness_paths(
            readiness.discover_paths(readiness.DEFAULT_READINESS_GLOBS)
        )
    )
    config_paths = discover_config_paths(args.agentic_config, args.agentic_config_glob)
    criteria = [
        setup_plan_safety(setup_paths),
        setup_installed(setup_paths),
        sandbox_readiness(readiness_paths, setup_paths),
        runtime_wandb_weave_policy(readiness_paths, setup_paths),
        runtime_network_policy_allowlist(readiness_paths, setup_paths),
        agentic_math_config_ready(config_paths, sandbox=args.sandbox),
        swebench_guard(config_paths, sandbox=args.sandbox),
    ]
    operator_handoff = operator_handoff_summary(setup_paths)
    operator_handoff_summary_fields = {
        "available": operator_handoff.get("available"),
        "source_setup_json": operator_handoff.get("source_setup_json"),
        "step_count": operator_handoff.get("step_count"),
        "required_step_count": operator_handoff.get("required_step_count"),
        "external_action_step_count": operator_handoff.get("external_action_step_count"),
        "evidence_path_count": operator_handoff.get("evidence_path_count"),
    }
    blockers = [row for row in criteria if not row["ok"]]
    setup_criterion = next(
        (row for row in criteria if row.get("name") == "setup_installed"),
        {},
    )
    setup_runtime = {
        "host_prerequisites_ok": setup_criterion.get("host_prerequisites_ok"),
        "runtime_installed": setup_criterion.get("runtime_installed"),
        "sandbox_configured": setup_criterion.get("sandbox_configured"),
        "provider": setup_criterion.get("provider"),
        "provider_preflight": setup_criterion.get("provider_preflight"),
        "latest_onboard_failure": setup_criterion.get("latest_onboard_failure"),
        "missing_required_commands": (
            setup_criterion.get("missing_required_commands")
            if isinstance(setup_criterion.get("missing_required_commands"), list)
            else []
        ),
        "missing_components": (
            setup_criterion.get("missing_components")
            if isinstance(setup_criterion.get("missing_components"), list)
            else []
        ),
    }
    decision = adoption_decision(
        criteria=criteria,
        setup_paths=setup_paths,
        sandbox=args.sandbox,
    )
    if not setup_paths:
        status = "missing_setup_evidence"
    elif any(row["name"] == "setup_installed" and not row["ok"] for row in criteria):
        status = "not_installed"
    elif blockers:
        status = "not_adoptable"
    else:
        status = (
            "adoptable_for_agentic_benchmarks"
            if decision.get("scope") == "agentic_math_and_swebench_pro"
            else "adoptable_for_agentic_math"
        )
    missing_required_commands = (
        setup_runtime.get("missing_required_commands")
        if isinstance(setup_runtime.get("missing_required_commands"), list)
        else []
    )
    missing_components = (
        setup_runtime.get("missing_components")
        if isinstance(setup_runtime.get("missing_components"), list)
        else []
    )
    return {
        "schema_version": 1,
        "ok": not blockers,
        "status": status,
        "adoption_decision": decision,
        "ready_for_use": decision["ready_for_use"],
        "adoption_recommendation": decision["recommendation"],
        "adoption_scope": decision["scope"],
        "design_ready": decision["design_ready"],
        "blockers": decision["blockers"],
        "runtime_blockers": decision["runtime_blockers"],
        "design_blockers": decision["design_blockers"],
        "other_blockers": decision["other_blockers"],
        "setup_runtime": setup_runtime,
        "operator_handoff": operator_handoff,
        "missing_required_commands": missing_required_commands,
        "missing_components": missing_components,
        "generated_at": time.time(),
        "sandbox": args.sandbox,
        "setup_paths": [path_display(path) for path in setup_paths],
        "readiness_paths": [path_display(path) for path in readiness_paths],
        "agentic_config_paths": [path_display(path) for path in config_paths],
        "summary": {
            "criterion_count": len(criteria),
            "blocker_count": len(blockers),
            "blockers": [row["name"] for row in blockers],
            "adoption_recommendation": decision["recommendation"],
            "ready_for_use": decision["ready_for_use"],
            "adoption_scope": decision["scope"],
            "setup_runtime": setup_runtime,
            "operator_handoff": operator_handoff_summary_fields,
        },
        "criteria": criteria,
    }


def markdown(report: dict[str, Any]) -> str:
    lines = [
        "# Taiwan NeMoClaw Adoption Check",
        "",
        f"Status: `{report['status']}`",
        f"OK: `{str(report['ok']).lower()}`",
        f"Sandbox: `{report['sandbox']}`",
        "",
        "## Adoption Decision",
        "",
        f"- Recommendation: `{(report.get('adoption_decision') or {}).get('recommendation')}`",
        f"- Ready for use: `{str((report.get('adoption_decision') or {}).get('ready_for_use')).lower()}`",
        f"- Scope: `{(report.get('adoption_decision') or {}).get('scope')}`",
        f"- Runtime blockers: `{report.get('runtime_blockers')}`",
        f"- Design blockers: `{report.get('design_blockers')}`",
        f"- Other blockers: `{report.get('other_blockers')}`",
        f"- Rationale: {(report.get('adoption_decision') or {}).get('rationale')}",
        f"- Next action: {(report.get('adoption_decision') or {}).get('next_action')}",
        "",
        "## Setup Runtime",
        "",
    ]
    setup_runtime = report.get("setup_runtime")
    if not isinstance(setup_runtime, dict):
        setup_runtime = ((report.get("summary") or {}).get("setup_runtime") or {})
    if isinstance(setup_runtime, dict):
        lines.extend(
            [
                f"- Host prerequisites OK: `{setup_runtime.get('host_prerequisites_ok')}`",
                f"- Runtime installed: `{setup_runtime.get('runtime_installed')}`",
                f"- Sandbox configured: `{setup_runtime.get('sandbox_configured')}`",
                f"- Provider: `{setup_runtime.get('provider')}`",
                f"- Provider preflight: `{setup_runtime.get('provider_preflight')}`",
                f"- Latest onboard failure: `{setup_runtime.get('latest_onboard_failure')}`",
                f"- Missing required commands: `{setup_runtime.get('missing_required_commands')}`",
                "",
            ]
        )
    operator_handoff = report.get("operator_handoff")
    if not isinstance(operator_handoff, dict):
        operator_handoff = {}
    lines.extend(
        [
            "## Operator Handoff",
            "",
            f"- Source setup JSON: `{operator_handoff.get('source_setup_json')}`",
            f"- Available: `{str(operator_handoff.get('available')).lower()}`",
            f"- Steps: `{operator_handoff.get('step_count')}`",
            f"- Required steps: `{operator_handoff.get('required_step_count')}`",
            f"- External-action steps: `{operator_handoff.get('external_action_step_count')}`",
            f"- Expected evidence paths: `{operator_handoff.get('evidence_path_count')}`",
            "",
            "| Step | External action | Evidence path |",
            "|---|---:|---|",
        ]
    )
    for row in operator_handoff.get("steps") if isinstance(operator_handoff.get("steps"), list) else []:
        if not isinstance(row, dict):
            continue
        lines.append(
            "| "
            + " | ".join(
                [
                    str(row.get("step") or ""),
                    str(row.get("requires_external_action") is True).lower(),
                    str(row.get("expected_evidence_path") or "").replace("|", "\\|"),
                ]
            )
            + " |"
        )
    lines.append("")
    lines.extend(
        [
        "## Criteria",
        "",
        "| Criterion | OK | Status | Detail |",
        "|---|---:|---|---|",
        ]
    )
    for row in report.get("criteria", []):
        if not isinstance(row, dict):
            continue
        lines.append(
            "| "
            + " | ".join(
                [
                    str(row.get("name", "")),
                    str(bool(row.get("ok"))).lower(),
                    str(row.get("status", "")),
                    str(row.get("detail", "")).replace("|", "\\|"),
                ]
            )
            + " |"
        )
    lines.extend(["", "## Next Actions", ""])
    for row in report.get("criteria", []):
        if isinstance(row, dict) and not row.get("ok"):
            lines.append(f"- `{row.get('name')}`: {row.get('next_action')}")
    if all(isinstance(row, dict) and row.get("ok") for row in report.get("criteria", [])):
        lines.append("- none")
    lines.append("")
    return "\n".join(lines)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--setup-json", type=Path, action="append")
    parser.add_argument("--readiness-json", type=Path, action="append")
    parser.add_argument("--agentic-config", type=Path, action="append")
    parser.add_argument(
        "--agentic-config-glob",
        action="append",
        help=(
            "Glob for generated Taiwan configs to inspect when --agentic-config is "
            "not supplied. Repeatable. The adoption doctor uses these configs for "
            "both Agentic Math NeMoClaw readiness and SWE-Bench Pro NeMoClaw "
            "migration guard checks."
        ),
    )
    parser.add_argument("--sandbox", default="nejumi-taiwan")
    parser.add_argument("--json", type=Path)
    parser.add_argument("--markdown", type=Path)
    parser.add_argument("--fail-on-not-adoptable", action="store_true")
    return parser.parse_args(argv)


def write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    report = build_report(args)
    if args.json:
        report["path"] = str(repo_path(args.json))
    if args.markdown:
        report["markdown_path"] = str(repo_path(args.markdown))
    if args.json:
        write_text(repo_path(args.json), json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    if args.markdown:
        write_text(repo_path(args.markdown), markdown(report))
    print(json.dumps(report, ensure_ascii=False, indent=2))
    if args.fail_on_not_adoptable and not report["ok"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
