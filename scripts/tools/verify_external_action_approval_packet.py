#!/usr/bin/env python3
"""Verify a reviewed external-action approval packet before execution.

This verifier is offline. It does not query W&B, write W&B, install NeMoClaw,
or launch model/provider calls. Operators should run it on a reviewed copy of
external_action_approval_packet.json after filling the required reviewer fields
and before executing any paid API, W&B write, third-party install, or scope
adoption command.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[2]
SCHEMA_VERSION = 1
SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
EXTERNAL_ACTION_REQUIREMENTS = (
    ("paid_api", "Paid API"),
    ("wandb_access", "W&B access"),
    ("wandb_write", "W&B write"),
    ("third_party_acceptance", "Third-party acceptance"),
    ("nemoclaw_install", "NeMoClaw install"),
    ("scope_confirmation", "Scope confirmation"),
)
COMMON_REVIEWER_FIELDS = ("approved_by", "approved_at", "approval_reference")
APPROVAL_REQUIREMENT_BASE_FIELDS = (
    "requirement",
    "label",
    "count",
    "required",
    "required_before_gates",
    "reviewer_fields",
)
SOURCE_BOUND_TOP_LEVEL_FIELDS = (
    "readiness_status",
    "readiness_ok",
    "blocking_gates",
    "source",
    "external_action_checklist",
    "external_action_checklist_sha256",
    "approval_requirement_count",
    "required_approval_count",
    "execution_policy",
    "approval_verifier",
    "approval_template_renderer",
    "outputs",
)
PLACEHOLDER_TOKENS = (
    "TODO",
    "TBD",
    "PLACEHOLDER",
    "PENDING",
    "REVIEW",
    "REVIEWED_",
    "YYYY",
    "RUN_ID",
    "MODEL_SLUG",
    "ACTUAL_OR_BILLING",
    "$ACTUAL",
    "あとで",
    "仮",
    "未定",
    "不明",
)
SCOPE_ATTESTATION_PLACEHOLDER_TOKENS = tuple(
    token for token in PLACEHOLDER_TOKENS if token not in {"REVIEW", "REVIEWED_"}
)
MIN_SCOPE_CONFIRMATION_LENGTH = 20


def canonical_json_sha256(value: Any) -> str:
    payload = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_json_object(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"{path} is not a JSON object")
    return payload


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def repo_path(value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else REPO_ROOT / path


def is_placeholder(value: Any) -> bool:
    if not isinstance(value, str):
        return False
    text = value.strip()
    if not text:
        return True
    upper = text.upper()
    return any(token.upper() in upper for token in PLACEHOLDER_TOKENS)


def is_scope_attestation_placeholder(value: Any) -> bool:
    if not isinstance(value, str):
        return False
    text = value.strip()
    if not text:
        return True
    upper = text.upper()
    return any(token.upper() in upper for token in SCOPE_ATTESTATION_PLACEHOLDER_TOKENS)


def require_concrete_string(
    errors: list[str],
    requirement: str,
    payload: dict[str, Any],
    field: str,
) -> str:
    value = payload.get(field)
    if not isinstance(value, str) or is_placeholder(value):
        errors.append(f"{requirement}.{field} must be a concrete non-placeholder string")
        return ""
    return value.strip()


def parse_aware_timestamp(value: Any) -> bool:
    if not isinstance(value, str) or is_placeholder(value):
        return False
    text = value.strip()
    if text.endswith("Z"):
        text = text[:-1] + "+00:00"
    try:
        parsed = datetime.fromisoformat(text)
    except ValueError:
        return False
    return parsed.tzinfo is not None and parsed.utcoffset() is not None


def parse_budget_usd(value: Any) -> float | None:
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return float(value)
    if isinstance(value, str) and not is_placeholder(value):
        text = value.strip().replace(",", "")
        text = text.replace("$", "").replace("USD", "").replace("usd", "").strip()
        try:
            return float(text)
        except ValueError:
            return None
    return None


def expected_reviewer_fields(requirement: str, *, required: bool) -> list[str]:
    if not required:
        return []
    fields = list(COMMON_REVIEWER_FIELDS)
    if requirement == "paid_api":
        fields.extend(["approved_budget_usd", "approved_model_scope"])
    elif requirement == "wandb_write":
        fields.extend(["approved_wandb_entity", "approved_wandb_project"])
    elif requirement == "third_party_acceptance":
        fields.append("third_party_terms_reviewed")
    elif requirement == "nemoclaw_install":
        fields.extend(["installer_lock_json", "installer_sha256", "sandbox"])
    elif requirement == "scope_confirmation":
        fields.append("scope_attestation_json")
    return fields


def expected_requirement_base(
    *,
    requirement: str,
    label: str,
    count: int,
    checklist: dict[str, Any],
) -> dict[str, Any]:
    gates = [
        item.get("gate")
        for item in checklist.get("items", [])
        if isinstance(item, dict)
        and requirement in (item.get("requirements") if isinstance(item.get("requirements"), list) else [])
    ]
    required = count > 0
    return {
        "requirement": requirement,
        "label": label,
        "count": count,
        "required": required,
        "required_before_gates": gates,
        "reviewer_fields": expected_reviewer_fields(requirement, required=required),
    }


def validate_common_approval_fields(
    *,
    requirement: str,
    item: dict[str, Any],
    errors: list[str],
) -> None:
    require_concrete_string(errors, requirement, item, "approved_by")
    require_concrete_string(errors, requirement, item, "approval_reference")
    if not parse_aware_timestamp(item.get("approved_at")):
        errors.append(f"{requirement}.approved_at must be a timezone-aware ISO 8601 timestamp")


def validate_paid_api(item: dict[str, Any], errors: list[str]) -> None:
    budget = parse_budget_usd(item.get("approved_budget_usd"))
    if budget is None or budget <= 0:
        errors.append("paid_api.approved_budget_usd must be a positive USD number")
    require_concrete_string(errors, "paid_api", item, "approved_model_scope")


def validate_wandb_write(item: dict[str, Any], errors: list[str]) -> None:
    require_concrete_string(errors, "wandb_write", item, "approved_wandb_entity")
    require_concrete_string(errors, "wandb_write", item, "approved_wandb_project")


def validate_third_party_acceptance(item: dict[str, Any], errors: list[str]) -> None:
    if item.get("third_party_terms_reviewed") is not True:
        errors.append("third_party_acceptance.third_party_terms_reviewed must be true")


def validate_nemoclaw_install(item: dict[str, Any], errors: list[str]) -> None:
    lock_json = require_concrete_string(errors, "nemoclaw_install", item, "installer_lock_json")
    installer_sha = require_concrete_string(errors, "nemoclaw_install", item, "installer_sha256").lower()
    require_concrete_string(errors, "nemoclaw_install", item, "sandbox")
    if installer_sha and SHA256_RE.fullmatch(installer_sha) is None:
        errors.append("nemoclaw_install.installer_sha256 must be 64 lowercase hex characters")
    if not lock_json:
        return
    lock_path = repo_path(lock_json)
    try:
        lock = read_json_object(lock_path)
    except (OSError, json.JSONDecodeError, ValueError) as exc:
        errors.append(f"nemoclaw_install.installer_lock_json is not readable: {exc}")
        return
    lock_sha = str(lock.get("sha256") or "").strip().lower()
    if SHA256_RE.fullmatch(lock_sha) is None:
        errors.append("nemoclaw_install installer lock sha256 is missing or invalid")
    elif installer_sha and lock_sha != installer_sha:
        errors.append("nemoclaw_install.installer_sha256 does not match installer lock sha256")


def validate_scope_confirmation(item: dict[str, Any], errors: list[str]) -> None:
    attestation_json = require_concrete_string(
        errors,
        "scope_confirmation",
        item,
        "scope_attestation_json",
    )
    if not attestation_json:
        return
    attestation_path = repo_path(attestation_json)
    try:
        attestation = read_json_object(attestation_path)
    except (OSError, json.JSONDecodeError, ValueError) as exc:
        errors.append(f"scope_confirmation.scope_attestation_json is not readable: {exc}")
        return
    if attestation.get("schema_version") != 1:
        errors.append("scope_confirmation scope_attestation_json schema_version must be 1")
    if attestation.get("confirmed") is not True:
        errors.append("scope_confirmation scope_attestation_json confirmed must be true")
    for field in (
        "confirmed_by",
        "confirmed_at",
        "confirmation",
        "actual_cost_estimate",
        "provider_bill_reference",
        "benchmark",
        "entity",
        "project",
        "run_id",
        "completion_path",
        "completion_sha256",
        "review_path",
    ):
        value = attestation.get(field)
        if not isinstance(value, str) or is_scope_attestation_placeholder(value):
            errors.append(f"scope_confirmation scope_attestation_json {field} must be concrete")
    if not parse_aware_timestamp(attestation.get("confirmed_at")):
        errors.append(
            "scope_confirmation scope_attestation_json confirmed_at must be a "
            "timezone-aware ISO 8601 timestamp"
        )
    confirmation = attestation.get("confirmation")
    if (
        not isinstance(confirmation, str)
        or is_scope_attestation_placeholder(confirmation)
        or len(confirmation.strip()) < MIN_SCOPE_CONFIRMATION_LENGTH
    ):
        errors.append(
            "scope_confirmation scope_attestation_json confirmation must be a "
            f"concrete sentence with at least {MIN_SCOPE_CONFIRMATION_LENGTH} characters"
        )
    completion_path_value = attestation.get("completion_path")
    completion_sha_value = attestation.get("completion_sha256")
    review_path_value = attestation.get("review_path")
    completion_payload: dict[str, Any] | None = None
    if isinstance(completion_path_value, str) and not is_scope_attestation_placeholder(completion_path_value):
        completion_path = repo_path(completion_path_value)
        try:
            completion_payload = read_json_object(completion_path)
        except (OSError, json.JSONDecodeError, ValueError) as exc:
            errors.append(f"scope_confirmation completion_path is not readable: {exc}")
        else:
            observed_sha = sha256_file(completion_path)
            if completion_sha_value != observed_sha:
                errors.append(
                    "scope_confirmation scope_attestation_json completion_sha256 "
                    "does not match completion_path"
                )
    if isinstance(review_path_value, str) and not is_scope_attestation_placeholder(review_path_value):
        try:
            read_json_object(repo_path(review_path_value))
        except (OSError, json.JSONDecodeError, ValueError) as exc:
            errors.append(f"scope_confirmation review_path is not readable: {exc}")
    if completion_payload is not None:
        for field in ("benchmark", "entity", "project", "run_id"):
            if attestation.get(field) != completion_payload.get(field):
                errors.append(
                    f"scope_confirmation scope_attestation_json {field} "
                    "does not match completion JSON"
                )


def validate_requirement(
    *,
    item: dict[str, Any],
    expected: dict[str, Any],
) -> dict[str, Any]:
    requirement = str(expected["requirement"])
    errors: list[str] = []
    for field in ("label", "count", "required", "required_before_gates", "reviewer_fields"):
        if item.get(field) != expected.get(field):
            errors.append(f"{requirement}.{field} does not match external_action_checklist")
    required = bool(expected.get("required"))
    status = item.get("approval_status")
    if not required:
        if status != "not_required":
            errors.append(f"{requirement}.approval_status must be not_required")
        return {
            "requirement": requirement,
            "required": False,
            "approved": status == "not_required" and not errors,
            "errors": errors,
        }

    if status != "granted":
        errors.append(f"{requirement}.approval_status must be granted")
    validate_common_approval_fields(requirement=requirement, item=item, errors=errors)
    if requirement == "paid_api":
        validate_paid_api(item, errors)
    elif requirement == "wandb_write":
        validate_wandb_write(item, errors)
    elif requirement == "third_party_acceptance":
        validate_third_party_acceptance(item, errors)
    elif requirement == "nemoclaw_install":
        validate_nemoclaw_install(item, errors)
    elif requirement == "scope_confirmation":
        validate_scope_confirmation(item, errors)
    result = {
        "requirement": requirement,
        "required": True,
        "approved": status == "granted" and not errors,
        "errors": errors,
    }
    if requirement == "paid_api":
        result["approved_budget_usd"] = parse_budget_usd(item.get("approved_budget_usd"))
        result["approved_model_scope"] = (
            item.get("approved_model_scope").strip()
            if isinstance(item.get("approved_model_scope"), str)
            else ""
        )
    return result


def approval_requirement_base(item: dict[str, Any]) -> dict[str, Any]:
    return {field: item.get(field) for field in APPROVAL_REQUIREMENT_BASE_FIELDS}


def approval_requirement_base_by_name(packet: dict[str, Any]) -> dict[str, dict[str, Any]]:
    requirements = packet.get("approval_requirements")
    if not isinstance(requirements, list):
        return {}
    return {
        str(item.get("requirement")): approval_requirement_base(item)
        for item in requirements
        if isinstance(item, dict) and isinstance(item.get("requirement"), str)
    }


def validate_source_packet_binding(
    *,
    reviewed_packet: dict[str, Any],
    reviewed_packet_path: Path,
    source_packet_path: Path,
) -> dict[str, Any]:
    errors: list[str] = []
    try:
        source_packet = read_json_object(source_packet_path)
        source_sha256 = sha256_file(source_packet_path)
    except (OSError, json.JSONDecodeError, ValueError) as exc:
        return {
            "source_packet_json": str(source_packet_path),
            "source_packet_readable": False,
            "source_approval_packet_sha256": "",
            "reviewed_packet_json": str(reviewed_packet_path),
            "bound": False,
            "errors": [f"source approval packet is not readable: {exc}"],
        }

    template = (
        reviewed_packet.get("approval_template")
        if isinstance(reviewed_packet.get("approval_template"), dict)
        else None
    )
    if template is None:
        errors.append("approval_template must be present when --source-packet-json is used")
    else:
        if template.get("source_approval_packet_sha256") != source_sha256:
            errors.append("approval_template.source_approval_packet_sha256 mismatch")
        if template.get("source_external_action_checklist_sha256") != source_packet.get(
            "external_action_checklist_sha256"
        ):
            errors.append("approval_template.source_external_action_checklist_sha256 mismatch")

    for field in SOURCE_BOUND_TOP_LEVEL_FIELDS:
        if reviewed_packet.get(field) != source_packet.get(field):
            errors.append(f"{field} does not match source approval packet")

    source_requirements = approval_requirement_base_by_name(source_packet)
    reviewed_requirements = approval_requirement_base_by_name(reviewed_packet)
    if source_requirements != reviewed_requirements:
        errors.append("approval_requirements base fields do not match source approval packet")

    return {
        "source_packet_json": str(source_packet_path),
        "source_packet_readable": True,
        "source_approval_packet_sha256": source_sha256,
        "reviewed_packet_json": str(reviewed_packet_path),
        "bound": not errors,
        "checked_top_level_fields": list(SOURCE_BOUND_TOP_LEVEL_FIELDS),
        "checked_approval_requirement_base_fields": list(APPROVAL_REQUIREMENT_BASE_FIELDS),
        "errors": errors,
    }


def verify_approval_packet(
    packet_path: Path,
    *,
    source_packet_path: Path | None = None,
) -> dict[str, Any]:
    errors: list[str] = []
    requirement_results: list[dict[str, Any]] = []
    source_binding: dict[str, Any] | None = None
    try:
        packet = read_json_object(packet_path)
    except (OSError, json.JSONDecodeError, ValueError) as exc:
        return {
            "schema_version": SCHEMA_VERSION,
            "ok": False,
            "status": "validation_failed",
            "generated_at": time.time(),
            "approval_packet_json": str(packet_path),
            "errors": [str(exc)],
            "approval_results": [],
            "will_execute_external_actions": False,
        }

    if source_packet_path is not None:
        source_binding = validate_source_packet_binding(
            reviewed_packet=packet,
            reviewed_packet_path=packet_path,
            source_packet_path=source_packet_path,
        )
        errors.extend(source_binding.get("errors") if isinstance(source_binding.get("errors"), list) else [])

    checklist = (
        packet.get("external_action_checklist")
        if isinstance(packet.get("external_action_checklist"), dict)
        else {}
    )
    requirement_counts = (
        checklist.get("requirement_counts")
        if isinstance(checklist.get("requirement_counts"), dict)
        else {}
    )
    if packet.get("schema_version") != SCHEMA_VERSION:
        errors.append(f"schema_version must be {SCHEMA_VERSION}")
    if not checklist:
        errors.append("external_action_checklist must be an object")
    expected_hash = canonical_json_sha256(checklist)
    if packet.get("external_action_checklist_sha256") != expected_hash:
        errors.append("external_action_checklist_sha256 mismatch")

    expected_by_name = {
        name: expected_requirement_base(
            requirement=name,
            label=label,
            count=int(requirement_counts.get(name) or 0),
            checklist=checklist,
        )
        for name, label in EXTERNAL_ACTION_REQUIREMENTS
    }
    approval_requirements = packet.get("approval_requirements")
    if not isinstance(approval_requirements, list):
        approval_requirements = []
        errors.append("approval_requirements must be a list")
    items_by_name = {
        item.get("requirement"): item
        for item in approval_requirements
        if isinstance(item, dict) and isinstance(item.get("requirement"), str)
    }
    if len(items_by_name) != len(approval_requirements):
        errors.append("approval_requirements contains duplicate or invalid requirement entries")

    for name, expected in expected_by_name.items():
        item = items_by_name.get(name)
        if not isinstance(item, dict):
            requirement_results.append(
                {
                    "requirement": name,
                    "required": bool(expected.get("required")),
                    "approved": False,
                    "errors": [f"{name} approval requirement is missing"],
                }
            )
            continue
        requirement_results.append(validate_requirement(item=item, expected=expected))

    required_count = sum(1 for item in expected_by_name.values() if item["required"])
    granted_count = sum(
        1 for item in requirement_results if item.get("required") and item.get("approved")
    )
    all_required_granted = required_count == granted_count
    if packet.get("approval_requirement_count") != len(expected_by_name):
        errors.append("approval_requirement_count mismatch")
    if packet.get("required_approval_count") != required_count:
        errors.append("required_approval_count mismatch")
    if bool(packet.get("all_required_approvals_granted")) is not all_required_granted:
        errors.append("all_required_approvals_granted mismatch")
    expected_status = "approved" if all_required_granted else "pending_approval"
    if packet.get("status") != expected_status:
        errors.append(f"status must be {expected_status}")

    for result in requirement_results:
        errors.extend(result.get("errors") if isinstance(result.get("errors"), list) else [])

    ok = not errors and all_required_granted
    return {
        "schema_version": SCHEMA_VERSION,
        "ok": ok,
        "status": "approved" if ok else "validation_failed",
        "generated_at": time.time(),
        "approval_packet_json": str(packet_path),
        "external_action_checklist_sha256": packet.get("external_action_checklist_sha256"),
        "expected_external_action_checklist_sha256": expected_hash,
        "required_approval_count": required_count,
        "granted_approval_count": granted_count,
        "all_required_approvals_granted": all_required_granted,
        "approval_results": requirement_results,
        "source_binding": source_binding or {},
        "will_execute_external_actions": False,
        "errors": errors,
    }


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--approval-packet-json", type=Path, required=True)
    parser.add_argument(
        "--source-packet-json",
        type=Path,
        help=(
            "Original release-bundle external_action_approval_packet.json. "
            "When supplied, immutable source fields and source SHA-256 are checked."
        ),
    )
    parser.add_argument(
        "--require-approved",
        action="store_true",
        help="Explicitly require all approval requirements to be granted. This is the default.",
    )
    parser.add_argument("--json", type=Path, help="Write verifier report JSON.")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    report = verify_approval_packet(
        repo_path(args.approval_packet_json),
        source_packet_path=repo_path(args.source_packet_json) if args.source_packet_json else None,
    )
    if args.json:
        write_json(repo_path(args.json), report)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    if report.get("ok") is not True:
        raise SystemExit(1)


if __name__ == "__main__":
    main(sys.argv[1:])
