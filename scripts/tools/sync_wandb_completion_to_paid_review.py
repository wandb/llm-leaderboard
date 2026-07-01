#!/usr/bin/env python3
"""Sync W&B completion verifier JSONs into a paid-run review record.

This is an offline bookkeeping tool. It does not query W&B and does not run
model inference. It copies already-generated verifier JSON paths into the
review record so production-readiness gates can verify the exact files.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import re
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[2]
WANDB_COMPLETION_SCHEMA_VERSION = 1
WANDB_COMPLETION_QUERY_SOURCE_KIND = "wandb_sdk"
WANDB_COMPLETION_API_TIMEOUT_SECONDS = 60
SCOPE_ATTESTATION_SCHEMA_VERSION = 1
MIN_SCOPE_CONFIRMATION_LENGTH = 20
NEMOCLAW_AUDIT_REQUIRED_BENCHMARKS = {"agentic_math", "agentic_swe"}
PLACEHOLDER_ACCOUNTING_VALUES = {
    "あとで",
    "仮",
    "仮置き",
    "不明",
    "未確認",
    "未定",
    "要確認",
    "actual or billing estimate",
    "actualorbillingestimate",
    "bill or dashboard reference",
    "billordashboardreference",
    "dummy",
    "fill me",
    "fill in",
    "n a",
    "na",
    "none",
    "pending",
    "placeholder",
    "tbd",
    "todo",
    "unknown",
}
PLACEHOLDER_ACCOUNTING_PREFIXES = (
    "あとで",
    "仮",
    "不明",
    "未確認",
    "未定",
    "要確認",
    "dummy",
    "fill me",
    "fill in",
    "pending",
    "placeholder",
    "tbd",
    "todo",
    "unknown",
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


def read_json_object(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"{path} is not a JSON object")
    return payload


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def normalized_display_path(value: str | Path) -> str:
    return path_display(repo_path(value))


def accounting_value_placeholder(value: str) -> bool:
    normalized = value.strip().casefold()
    collapsed = re.sub(r"[\s_:/\\|.,;\-]+", " ", normalized).strip()
    compact = re.sub(r"[^0-9a-zぁ-んァ-ン一-龥]+", "", normalized).strip()
    return (
        collapsed in PLACEHOLDER_ACCOUNTING_VALUES
        or compact in PLACEHOLDER_ACCOUNTING_VALUES
        or any(
            collapsed == prefix or collapsed.startswith(prefix + " ")
            for prefix in PLACEHOLDER_ACCOUNTING_PREFIXES
        )
    )


def scope_confirmed_at_valid(value: Any) -> bool:
    if not isinstance(value, str) or not value.strip():
        return False
    text = value.strip()
    if any(marker in text for marker in ("YYYY", "MM", "DD", "HH", "SS")):
        return False
    normalized = text[:-1] + "+00:00" if text.endswith("Z") else text
    try:
        parsed = datetime.fromisoformat(normalized)
    except ValueError:
        return False
    return parsed.tzinfo is not None and parsed.utcoffset() is not None


def scope_confirmation_valid(value: Any) -> bool:
    if not isinstance(value, str):
        return False
    stripped = value.strip()
    if len(stripped) < MIN_SCOPE_CONFIRMATION_LENGTH:
        return False
    return not accounting_value_placeholder(stripped)


def validate_accounting_value(value: str | None, *, field: str) -> str | None:
    if value is None:
        return None
    stripped = value.strip()
    if not stripped:
        raise SystemExit(f"--{field.replace('_', '-')} must not be empty")
    if accounting_value_placeholder(stripped):
        raise SystemExit(f"--{field.replace('_', '-')} must not be a placeholder")
    return stripped


def observed_evidence_valid(value: Any) -> bool:
    if not isinstance(value, dict):
        return False
    if value.get("run_state") != "finished":
        return False
    evidence_keys = (
        "summary_metrics",
        "tables",
        "artifacts",
        "taxonomy_tables",
        "aggregate_tables",
    )
    return any(bool(value.get(key)) for key in evidence_keys)


def int_like(value: Any) -> int | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, int):
        return value
    if isinstance(value, float) and math.isfinite(value) and value.is_integer():
        return int(value)
    return None


def nemoclaw_session_audit_current(payload: dict[str, Any]) -> tuple[bool, list[str]]:
    benchmark = payload.get("benchmark")
    if benchmark not in NEMOCLAW_AUDIT_REQUIRED_BENCHMARKS:
        return True, []

    issues: list[str] = []
    required = payload.get("required_evidence")
    if not isinstance(required, dict):
        issues.append("required_evidence must be an object for NeMoClaw session audit proof")
        required = {}
    required_audit = required.get("nemoclaw_session_audit")
    if not isinstance(required_audit, dict):
        issues.append("required_evidence.nemoclaw_session_audit must be an object")
        required_audit = {}
    elif required_audit.get("required") is not True:
        issues.append("required_evidence.nemoclaw_session_audit.required must be true")

    expected_total = int_like(required.get("expected_total"))
    if expected_total is None or expected_total <= 0:
        issues.append("required_evidence.expected_total must be a positive integer")

    observed = payload.get("observed_evidence")
    if not isinstance(observed, dict):
        issues.append("observed_evidence must be an object for NeMoClaw session audit proof")
        observed = {}
    observed_audit = observed.get("nemoclaw_session_audit")
    if not isinstance(observed_audit, dict):
        issues.append("observed_evidence.nemoclaw_session_audit must be an object")
        return False, issues
    if observed_audit.get("ok") is not True:
        issues.append("observed_evidence.nemoclaw_session_audit.ok must be true")

    required_count = int_like(observed_audit.get("required"))
    passed_count = int_like(observed_audit.get("passed"))
    failed_count = int_like(observed_audit.get("failed"))
    for field, value in (
        ("required", required_count),
        ("passed", passed_count),
        ("failed", failed_count),
    ):
        if value is None:
            issues.append(f"observed_evidence.nemoclaw_session_audit.{field} must be an integer")

    observed_expected_total = int_like(observed_audit.get("expected_total"))
    if expected_total is not None and observed_expected_total is not None and observed_expected_total != expected_total:
        issues.append("observed_evidence.nemoclaw_session_audit.expected_total must match required_evidence.expected_total")
    if expected_total is not None and required_count is not None and required_count != expected_total:
        issues.append("observed_evidence.nemoclaw_session_audit.required must equal expected_total")
    if expected_total is not None and passed_count is not None and passed_count != expected_total:
        issues.append("observed_evidence.nemoclaw_session_audit.passed must equal expected_total")
    if required_count is not None and passed_count is not None and required_count != passed_count:
        issues.append("observed_evidence.nemoclaw_session_audit.required must equal passed")
    if failed_count is not None and failed_count != 0:
        issues.append("observed_evidence.nemoclaw_session_audit.failed must be 0")
    return not issues, issues


def wandb_query_source_current(payload: dict[str, Any]) -> tuple[bool, list[str]]:
    issues: list[str] = []
    query_source = payload.get("query_source")
    if not isinstance(query_source, dict):
        return False, ["query_source must be an object"]
    expected = {
        "kind": WANDB_COMPLETION_QUERY_SOURCE_KIND,
        "api": "wandb.Api",
        "entity": payload.get("entity"),
        "project": payload.get("project"),
        "run_id": payload.get("run_id"),
        "run_path": (
            f"{payload.get('entity')}/{payload.get('project')}/{payload.get('run_id')}"
            if payload.get("entity") and payload.get("project") and payload.get("run_id")
            else None
        ),
        "benchmark": payload.get("benchmark"),
        "summary_source": "run.summary_metrics",
        "artifact_source": "run.logged_artifacts",
        "history_scanned": False,
    }
    for key, expected_value in expected.items():
        if expected_value is not None and query_source.get(key) != expected_value:
            issues.append(
                f"query_source.{key} must be {expected_value!r}, got {query_source.get(key)!r}"
            )
    if query_source.get("timeout_seconds") != WANDB_COMPLETION_API_TIMEOUT_SECONDS:
        issues.append(
            "query_source.timeout_seconds must be "
            f"{WANDB_COMPLETION_API_TIMEOUT_SECONDS}"
        )
    return not issues, issues


def observed_run_config_by_key(observed_metadata: dict[str, Any]) -> dict[str, dict[str, Any]]:
    rows = observed_metadata.get("config")
    if not isinstance(rows, list):
        return {}
    return {
        str(row.get("key")): row
        for row in rows
        if isinstance(row, dict) and isinstance(row.get("key"), str) and row.get("key")
    }


def wandb_run_metadata_current(payload: dict[str, Any]) -> tuple[bool, list[str]]:
    issues: list[str] = []
    required = payload.get("required_evidence")
    if not isinstance(required, dict):
        issues.append("required_evidence.run_metadata must be an object")
        required = {}
    required_metadata = required.get("run_metadata")
    if not isinstance(required_metadata, dict) or not required_metadata:
        issues.append("required_evidence.run_metadata must be an object")
        required_metadata = {}

    observed = payload.get("observed_evidence")
    observed_metadata = (
        observed.get("run_metadata")
        if isinstance(observed, dict)
        else None
    )
    if not isinstance(observed_metadata, dict) or not observed_metadata:
        issues.append("observed_evidence.run_metadata must be an object")
        observed_metadata = {}

    has_requirement = False
    observed_config = observed_run_config_by_key(observed_metadata)
    config_rows = required_metadata.get("config")
    if isinstance(config_rows, list):
        for row in config_rows:
            if not isinstance(row, dict):
                continue
            key = row.get("key")
            if not isinstance(key, str) or not key:
                continue
            if "expected" not in row:
                continue
            has_requirement = True
            expected = row.get("expected")
            observed_row = observed_config.get(key)
            if not isinstance(observed_row, dict):
                issues.append(f"observed_evidence.run_metadata.config missing {key}")
                continue
            if observed_row.get("present") is not True:
                issues.append(f"observed_evidence.run_metadata.config {key} is not present")
            if observed_row.get("value") != expected:
                issues.append(f"observed_evidence.run_metadata.config {key} value mismatch")

    observed_tags = observed_metadata.get("tags")
    if not isinstance(observed_tags, list):
        observed_tags = []
    tags = required_metadata.get("tags")
    if isinstance(tags, list):
        for tag in tags:
            if not isinstance(tag, str) or not tag:
                continue
            has_requirement = True
            if tag not in observed_tags:
                issues.append(f"observed_evidence.run_metadata.tags missing {tag}")

    for field in ("group", "job_type"):
        expected = required_metadata.get(field)
        if not isinstance(expected, str) or not expected:
            continue
        has_requirement = True
        if observed_metadata.get(field) != expected:
            issues.append(f"observed_evidence.run_metadata.{field} mismatch")

    if not has_requirement:
        issues.append(
            "required_evidence.run_metadata must specify at least one config, tag, group, or job_type requirement"
        )
    return not issues, issues


def completion_payload_current(
    path: Path,
    payload: dict[str, Any],
    *,
    require_query_source: bool,
) -> tuple[bool, list[str]]:
    issues: list[str] = []
    if payload.get("verification_schema_version") != WANDB_COMPLETION_SCHEMA_VERSION:
        issues.append(
            f"verification_schema_version must be {WANDB_COMPLETION_SCHEMA_VERSION}"
        )
    if not isinstance(payload.get("generated_at"), (int, float)):
        issues.append("generated_at must be numeric")
    if not observed_evidence_valid(payload.get("observed_evidence")):
        issues.append("observed_evidence must prove a finished W&B run with logged evidence")
    for field in ("entity", "project"):
        if not isinstance(payload.get(field), str) or not payload.get(field).strip():
            issues.append(f"{field} is required")
    if payload.get("returncode_ok") is False:
        issues.append("verifier subprocess returncode was nonzero")
    if payload.get("payload_ok") is False:
        issues.append("verifier payload_ok is false")
    if require_query_source:
        query_source_ok, query_source_issues = wandb_query_source_current(payload)
        if not query_source_ok:
            issues.extend(query_source_issues)
    run_metadata_ok, run_metadata_issues = wandb_run_metadata_current(payload)
    if not run_metadata_ok:
        issues.extend(run_metadata_issues)
    audit_ok, audit_issues = nemoclaw_session_audit_current(payload)
    if not audit_ok:
        issues.extend(audit_issues)
    return not issues, issues


def completion_entry(
    path: Path,
    payload: dict[str, Any],
    *,
    allow_failed: bool,
    require_query_source: bool = False,
) -> dict[str, Any]:
    benchmark = payload.get("benchmark")
    entity = payload.get("entity")
    project = payload.get("project")
    run_id = payload.get("run_id")
    if not isinstance(benchmark, str) or not benchmark:
        raise ValueError(f"{path} is missing benchmark")
    if not isinstance(entity, str) or not entity.strip():
        raise ValueError(f"{path} is missing entity")
    if not isinstance(project, str) or not project.strip():
        raise ValueError(f"{path} is missing project")
    if not isinstance(run_id, str) or not run_id:
        raise ValueError(f"{path} is missing run_id")
    if payload.get("ok") is not True and not allow_failed:
        raise ValueError(f"{path} is not a passing W&B completion verifier JSON")
    current, issues = completion_payload_current(
        path,
        payload,
        require_query_source=require_query_source,
    )
    if not current and not allow_failed:
        raise ValueError(
            f"{path} is not a current W&B completion verifier JSON: "
            + "; ".join(issues)
        )
    entry = {
        "benchmark": benchmark,
        "ok": bool(payload.get("ok")),
        "entity": entity.strip(),
        "project": project.strip(),
        "run_id": run_id,
        "path": path_display(path),
        "sha256": sha256_file(path),
        "verification_schema_version": payload.get("verification_schema_version"),
        "observed_evidence_valid": observed_evidence_valid(payload.get("observed_evidence")),
        "run_metadata_valid": wandb_run_metadata_current(payload)[0],
        "nemoclaw_session_audit_valid": nemoclaw_session_audit_current(payload)[0],
    }
    run_metadata_ok, run_metadata_issues = wandb_run_metadata_current(payload)
    entry["run_metadata_valid"] = run_metadata_ok
    if run_metadata_issues:
        entry["run_metadata_errors"] = run_metadata_issues
    audit_ok, audit_issues = nemoclaw_session_audit_current(payload)
    entry["nemoclaw_session_audit_valid"] = audit_ok
    if audit_issues:
        entry["nemoclaw_session_audit_errors"] = audit_issues
    query_source = payload.get("query_source")
    if isinstance(query_source, dict):
        entry["query_source_kind"] = query_source.get("kind")
        entry["query_source_run_path"] = query_source.get("run_path")
    return entry


def source_audit_record_matches_entry(record: dict[str, Any], entry: dict[str, Any]) -> bool:
    if record.get("benchmark") != entry.get("benchmark"):
        return False
    completion = record.get("wandb_completion")
    if not isinstance(completion, dict):
        return False
    comparisons = (
        ("entity", "entity"),
        ("project", "project"),
        ("run_id", "run_id"),
    )
    for completion_key, entry_key in comparisons:
        value = completion.get(completion_key)
        if isinstance(value, str) and value and value != entry.get(entry_key):
            return False
    completion_path = completion.get("path")
    if not isinstance(completion_path, str) or not completion_path.strip():
        return False
    if normalized_display_path(completion_path) != normalized_display_path(entry["path"]):
        return False
    verification_schema_version = completion.get("verification_schema_version")
    if (
        verification_schema_version is not None
        and verification_schema_version != entry.get("verification_schema_version")
    ):
        return False
    return True


def validate_source_audit_binding(
    *,
    payload: dict[str, Any],
    entry: dict[str, Any],
    issues: list[str],
) -> dict[str, str]:
    source_audit_json = payload.get("source_audit_json")
    source_audit_sha256 = payload.get("source_audit_sha256")
    if source_audit_json is None and source_audit_sha256 is None:
        return {}
    if not isinstance(source_audit_json, str) or not source_audit_json.strip():
        issues.append("source_audit_json is required when source_audit_sha256 is present")
        return {}
    if not isinstance(source_audit_sha256, str) or not re.fullmatch(
        r"[0-9a-f]{64}",
        source_audit_sha256,
    ):
        issues.append("source_audit_sha256 must be a 64-character lowercase hex digest")
        return {}

    audit_path = repo_path(source_audit_json)
    try:
        actual_sha256 = sha256_file(audit_path)
    except OSError as exc:
        issues.append(f"source_audit_json is not readable: {exc}")
        return {
            "source_audit_json": path_display(audit_path),
            "source_audit_sha256": source_audit_sha256,
        }
    if actual_sha256 != source_audit_sha256:
        issues.append("source_audit_sha256 does not match source_audit_json")

    try:
        audit_payload = read_json_object(audit_path)
    except (OSError, json.JSONDecodeError, ValueError) as exc:
        issues.append(f"source_audit_json is not a readable JSON object: {exc}")
        return {
            "source_audit_json": path_display(audit_path),
            "source_audit_sha256": source_audit_sha256,
        }

    formalized = audit_payload.get("formalized_records")
    if not isinstance(formalized, list):
        issues.append("source_audit_json formalized_records must be a list")
    elif not any(
        isinstance(record, dict) and source_audit_record_matches_entry(record, entry)
        for record in formalized
    ):
        issues.append("source_audit_json formalized_records does not include completion entry")

    return {
        "source_audit_json": path_display(audit_path),
        "source_audit_sha256": source_audit_sha256,
    }


def validate_adoption_args(args: argparse.Namespace) -> None:
    manual_scope_args = [
        name
        for name, value in (
            ("--scope-confirmed-by", args.scope_confirmed_by),
            ("--scope-confirmed-at", args.scope_confirmed_at),
            ("--scope-confirmation", args.scope_confirmation),
        )
        if isinstance(value, str) and value.strip()
    ]
    if manual_scope_args:
        raise SystemExit(
            "Manual scope-attestation args are no longer supported for release "
            "adoption: use --scope-attestation-json instead of "
            + ", ".join(manual_scope_args)
        )
    if not args.adopt_existing_result:
        if args.scope_attestation_json:
            raise SystemExit("--scope-attestation-json requires --adopt-existing-result")
        return
    if args.scope_attestation_json:
        return
    raise SystemExit("--adopt-existing-result requires --scope-attestation-json")


def validate_scope_attestation_json(
    *,
    path: Path,
    payload: dict[str, Any],
    entry: dict[str, Any],
    review_path: Path,
) -> dict[str, Any]:
    issues: list[str] = []
    if payload.get("schema_version") != SCOPE_ATTESTATION_SCHEMA_VERSION:
        issues.append(
            f"schema_version must be {SCOPE_ATTESTATION_SCHEMA_VERSION}"
        )
    if payload.get("confirmed") is not True:
        issues.append("confirmed must be true")
    for key in (
        "confirmed_by",
        "confirmed_at",
        "confirmation",
        "actual_cost_estimate",
        "provider_bill_reference",
    ):
        value = payload.get(key)
        if not isinstance(value, str) or not value.strip():
            issues.append(f"{key} is required")
    for key in ("actual_cost_estimate", "provider_bill_reference"):
        value = payload.get(key)
        if isinstance(value, str) and value.strip() and accounting_value_placeholder(value):
            issues.append(f"{key} must not be a placeholder")
    if not scope_confirmed_at_valid(payload.get("confirmed_at")):
        issues.append("confirmed_at must be a timezone-aware ISO 8601 timestamp")
    if not scope_confirmation_valid(payload.get("confirmation")):
        issues.append(
            "confirmation must be a concrete non-placeholder sentence "
            f"with at least {MIN_SCOPE_CONFIRMATION_LENGTH} characters"
        )
    if payload.get("benchmark") != entry.get("benchmark"):
        issues.append("benchmark does not match completion entry")
    if payload.get("entity") != entry.get("entity"):
        issues.append("entity does not match completion entry")
    if payload.get("project") != entry.get("project"):
        issues.append("project does not match completion entry")
    if payload.get("run_id") != entry.get("run_id"):
        issues.append("run_id does not match completion entry")

    completion_path = payload.get("completion_path")
    if not isinstance(completion_path, str) or not completion_path.strip():
        issues.append("completion_path is required")
    elif normalized_display_path(completion_path) != normalized_display_path(entry["path"]):
        issues.append("completion_path does not match completion entry")

    completion_sha256 = payload.get("completion_sha256")
    if not isinstance(completion_sha256, str) or not completion_sha256.strip():
        issues.append("completion_sha256 is required")
    elif completion_sha256 != entry.get("sha256"):
        issues.append("completion_sha256 does not match completion entry")

    attested_review = payload.get("review_path")
    if not isinstance(attested_review, str) or not attested_review.strip():
        issues.append("review_path is required")
    elif normalized_display_path(attested_review) != path_display(review_path):
        issues.append("review_path does not match --review-json")

    source_audit_binding = validate_source_audit_binding(
        payload=payload,
        entry=entry,
        issues=issues,
    )

    if issues:
        raise ValueError(
            f"{path} is not a valid scope attestation JSON: " + "; ".join(issues)
        )
    scope_attestation = {
        "schema_version": SCOPE_ATTESTATION_SCHEMA_VERSION,
        "confirmed": True,
        "confirmed_by": payload["confirmed_by"].strip(),
        "confirmed_at": payload["confirmed_at"].strip(),
        "confirmation": payload["confirmation"].strip(),
        "review_path": path_display(review_path),
        "completion_path": entry["path"],
        "completion_sha256": entry["sha256"],
        "benchmark": entry["benchmark"],
        "entity": entry["entity"],
        "project": entry["project"],
        "run_id": entry["run_id"],
        "actual_cost_estimate": payload["actual_cost_estimate"].strip(),
        "provider_bill_reference": payload["provider_bill_reference"].strip(),
        "source_attestation_json": path_display(path),
        "source_attestation_sha256": sha256_file(path),
    }
    scope_attestation.update(source_audit_binding)
    return scope_attestation


def attach_scope_attestation(
    entry: dict[str, Any],
    *,
    args: argparse.Namespace,
    review_path: Path,
    scope_attestation_path: Path | None = None,
    scope_attestation_payload: dict[str, Any] | None = None,
) -> dict[str, Any]:
    if not args.adopt_existing_result:
        return entry
    updated = dict(entry)
    if scope_attestation_path is not None and scope_attestation_payload is not None:
        updated["adopted_existing_result"] = True
        updated["scope_attestation"] = validate_scope_attestation_json(
            path=scope_attestation_path,
            payload=scope_attestation_payload,
            entry=entry,
            review_path=review_path,
        )
        return updated
    raise ValueError("--adopt-existing-result requires --scope-attestation-json")


def _same_completion(left: dict[str, Any], right: dict[str, Any]) -> bool:
    return all(
        left.get(key) == right.get(key)
        for key in ("benchmark", "entity", "project", "run_id")
    )


def merge_completion_entries(
    rows: Any,
    entry: dict[str, Any],
    *,
    replace: bool,
) -> tuple[list[dict[str, Any]], str]:
    existing = [row for row in rows if isinstance(row, dict)] if isinstance(rows, list) else []
    for index, row in enumerate(existing):
        if _same_completion(row, entry):
            if replace:
                existing[index] = entry
                return existing, "replaced"
            return existing, "kept_existing"
    existing.append(entry)
    return existing, "added"


def _run_identity_value(run: dict[str, Any], *keys: str) -> str:
    for key in keys:
        value = run.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    return ""


def _run_matches_completion_entry(run: dict[str, Any], entry: dict[str, Any]) -> bool:
    if run.get("wandb_run_id") != entry["run_id"]:
        return False
    run_entity = _run_identity_value(run, "wandb_entity", "entity")
    run_project = _run_identity_value(run, "wandb_project", "project")
    if run_entity and run_entity != entry.get("entity"):
        return False
    if run_project and run_project != entry.get("project"):
        return False
    return True


def _run_has_wandb_identity(run: dict[str, Any]) -> bool:
    return bool(
        _run_identity_value(run, "wandb_entity", "entity")
        and _run_identity_value(run, "wandb_project", "project")
    )


def matching_review_runs(runs: list[Any], entry: dict[str, Any]) -> tuple[list[dict[str, Any]], str]:
    candidates = [
        run
        for run in runs
        if isinstance(run, dict) and _run_matches_completion_entry(run, entry)
    ]
    exact = [run for run in candidates if _run_has_wandb_identity(run)]
    if exact:
        return exact, "matched_wandb_identity"
    if len(candidates) == 1:
        return candidates, "matched_run_id_only"
    if candidates:
        return [], "ambiguous_run_id_without_wandb_identity"
    return [], "unmatched_run_identity"


def sync_review(
    review: dict[str, Any],
    entries: list[dict[str, Any]],
    *,
    top_level: bool,
    replace: bool,
    set_verify_wandb_completion: bool,
    actual_cost_estimate: str | None,
    provider_bill_reference: str | None,
) -> tuple[dict[str, Any], list[dict[str, Any]], list[dict[str, Any]]]:
    updated = copy.deepcopy(review)
    changes: list[dict[str, Any]] = []
    unmatched: list[dict[str, Any]] = []

    if set_verify_wandb_completion:
        updated["verify_wandb_completion"] = True
    if actual_cost_estimate is not None:
        updated["actual_cost_estimate"] = actual_cost_estimate
    if provider_bill_reference is not None:
        updated["provider_bill_reference"] = provider_bill_reference

    if top_level:
        rows = updated.get("wandb_completion")
        for entry in entries:
            merged, action = merge_completion_entries(rows, entry, replace=replace)
            rows = merged
            changes.append(
                {
                    "target": "top_level",
                    "action": action,
                    "benchmark": entry["benchmark"],
                    "run_id": entry["run_id"],
                }
            )
        updated["wandb_completion"] = rows or []
        return updated, changes, unmatched

    runs = updated.get("runs")
    if not isinstance(runs, list):
        runs = []
        updated["runs"] = runs

    for entry in entries:
        matches, match_status = matching_review_runs(runs, entry)
        if not matches:
            unmatched.append({**entry, "match_status": match_status})
            continue
        for run in matches:
            merged, action = merge_completion_entries(
                run.get("wandb_completion"),
                entry,
                replace=replace,
            )
            run["wandb_completion"] = merged
            changes.append(
                {
                    "target": "run",
                    "action": action,
                    "benchmark": entry["benchmark"],
                    "entity": entry.get("entity"),
                    "project": entry.get("project"),
                    "run_id": entry["run_id"],
                    "match_status": match_status,
                    "config": run.get("config"),
                }
            )
    return updated, changes, unmatched


def compact_entry_for_report(entry: dict[str, Any]) -> dict[str, Any]:
    compact = {
        "benchmark": entry.get("benchmark"),
        "ok": entry.get("ok"),
        "entity": entry.get("entity"),
        "project": entry.get("project"),
        "run_id": entry.get("run_id"),
        "path": entry.get("path"),
        "sha256": entry.get("sha256"),
        "verification_schema_version": entry.get("verification_schema_version"),
        "observed_evidence_valid": entry.get("observed_evidence_valid"),
        "run_metadata_valid": entry.get("run_metadata_valid"),
        "nemoclaw_session_audit_valid": entry.get("nemoclaw_session_audit_valid"),
        "adopted_existing_result": bool(entry.get("adopted_existing_result")),
    }
    if isinstance(entry.get("run_metadata_errors"), list):
        compact["run_metadata_errors"] = entry.get("run_metadata_errors")
    if isinstance(entry.get("nemoclaw_session_audit_errors"), list):
        compact["nemoclaw_session_audit_errors"] = entry.get("nemoclaw_session_audit_errors")
    query_source_kind = entry.get("query_source_kind")
    query_source_run_path = entry.get("query_source_run_path")
    if query_source_kind is not None:
        compact["query_source_kind"] = query_source_kind
    if query_source_run_path is not None:
        compact["query_source_run_path"] = query_source_run_path
    scope = entry.get("scope_attestation")
    if isinstance(scope, dict):
        compact["scope_attestation"] = {
            key: scope.get(key)
            for key in (
                "schema_version",
                "confirmed",
                "confirmed_by",
                "confirmed_at",
                "confirmation",
                "review_path",
                "completion_path",
                "completion_sha256",
                "benchmark",
                "entity",
                "project",
                "run_id",
                "actual_cost_estimate",
                "provider_bill_reference",
                "source_attestation_json",
                "source_attestation_sha256",
                "source_audit_json",
                "source_audit_sha256",
            )
        }
    return compact


def build_report(
    *,
    review_path: Path,
    output_path: Path | None,
    review: dict[str, Any],
    updated: dict[str, Any],
    entries: list[dict[str, Any]],
    changes: list[dict[str, Any]],
    unmatched: list[dict[str, Any]],
    in_place: bool,
    dry_run: bool,
) -> dict[str, Any]:
    adopted_entries = [entry for entry in entries if entry.get("adopted_existing_result")]
    return {
        "ok": not unmatched,
        "status": "synced" if not unmatched else "unmatched_run_id",
        "generated_at": time.time(),
        "review_path": path_display(review_path),
        "source_review_sha256": sha256_file(review_path),
        "output_path": path_display(output_path) if output_path else "",
        "in_place": in_place,
        "dry_run": dry_run,
        "entry_count": len(entries),
        "adopted_existing_result_count": len(adopted_entries),
        "entries": [compact_entry_for_report(entry) for entry in entries],
        "change_count": len(changes),
        "unmatched_count": len(unmatched),
        "changes": changes,
        "unmatched_entries": unmatched,
        "before_status": review.get("status"),
        "after_status": updated.get("status"),
        "verify_wandb_completion": bool(updated.get("verify_wandb_completion")),
    }


def build_validation_failure_report(
    *,
    args: argparse.Namespace,
    review_path: Path,
    error: Exception,
) -> dict[str, Any]:
    output_path = (
        review_path
        if args.in_place
        else repo_path(args.output_json)
        if args.output_json
        else None
    )
    return {
        "ok": False,
        "status": "validation_failed",
        "generated_at": time.time(),
        "review_path": path_display(review_path),
        "output_path": path_display(output_path) if output_path else "",
        "in_place": bool(args.in_place),
        "dry_run": output_path is None,
        "entry_count": 0,
        "adopted_existing_result_count": 0,
        "entries": [],
        "change_count": 0,
        "unmatched_count": 0,
        "changes": [],
        "unmatched_entries": [],
        "verify_wandb_completion": False,
        "errors": [str(error)],
    }


def validate_validated_dry_run_report(
    *,
    path: Path,
    payload: dict[str, Any],
    current_report: dict[str, Any],
) -> None:
    issues: list[str] = []
    if payload.get("ok") is not True:
        issues.append("ok must be true")
    if payload.get("status") != "synced":
        issues.append("status must be synced")
    generated_at = payload.get("generated_at")
    if (
        not isinstance(generated_at, (int, float))
        or not math.isfinite(float(generated_at))
        or generated_at <= 0
    ):
        issues.append("generated_at must be a positive number")
    if payload.get("dry_run") is not True:
        issues.append("dry_run must be true")
    if payload.get("in_place") is not False:
        issues.append("in_place must be false")
    if payload.get("output_path") not in ("", None):
        issues.append("output_path must be empty")
    if payload.get("review_path") != current_report.get("review_path"):
        issues.append("review_path does not match current sync")
    if payload.get("source_review_sha256") != current_report.get("source_review_sha256"):
        issues.append("source_review_sha256 does not match current sync")
    if payload.get("verify_wandb_completion") != current_report.get("verify_wandb_completion"):
        issues.append("verify_wandb_completion does not match current sync")
    if payload.get("unmatched_count") != 0:
        issues.append("unmatched_count must be 0")
    for field in ("before_status", "after_status"):
        if field not in payload:
            issues.append(f"{field} is missing")
        elif payload.get(field) != current_report.get(field):
            issues.append(f"{field} does not match current sync")
    for field in (
        "entry_count",
        "adopted_existing_result_count",
        "entries",
        "change_count",
        "changes",
    ):
        if payload.get(field) != current_report.get(field):
            issues.append(f"{field} does not match current sync")
    if issues:
        raise SystemExit(
            f"{path_display(path)} is not a valid matching dry-run report: "
            + "; ".join(issues)
        )


def attach_validated_dry_run_report_path(
    updated: dict[str, Any],
    entries: list[dict[str, Any]],
    *,
    top_level: bool,
    path: Path,
    review_path: Path,
) -> None:
    display = path_display(path)
    source_review_display = path_display(review_path)
    source_review_sha256 = sha256_file(review_path)

    def attach_to_rows(rows: Any) -> None:
        if not isinstance(rows, list):
            return
        for entry in entries:
            for row in rows:
                if not isinstance(row, dict) or not _same_completion(row, entry):
                    continue
                row["sync_dry_run_report_json"] = display
                row["sync_dry_run_source_review_json"] = source_review_display
                row["sync_dry_run_source_review_sha256"] = source_review_sha256

    if top_level:
        attach_to_rows(updated.get("wandb_completion"))
        return

    runs = updated.get("runs")
    if not isinstance(runs, list):
        return
    for run in runs:
        if isinstance(run, dict):
            attach_to_rows(run.get("wandb_completion"))


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--review-json", type=Path, required=True)
    parser.add_argument("--completion-json", type=Path, action="append", required=True)
    parser.add_argument("--output-json", type=Path)
    parser.add_argument("--report-json", type=Path)
    parser.add_argument("--in-place", action="store_true")
    parser.add_argument("--top-level", action="store_true")
    parser.add_argument("--no-replace", action="store_true")
    parser.add_argument("--allow-failed", action="store_true")
    parser.add_argument("--set-verify-wandb-completion", action="store_true")
    parser.add_argument("--actual-cost-estimate")
    parser.add_argument("--provider-bill-reference")
    parser.add_argument("--allow-unmatched", action="store_true")
    parser.add_argument(
        "--adopt-existing-result",
        action="store_true",
        help=(
            "Mark synced W&B completion entries as adopted existing results. "
            "This requires --scope-attestation-json."
        ),
    )
    parser.add_argument(
        "--scope-confirmed-by",
        help="Deprecated. Use --scope-attestation-json instead.",
    )
    parser.add_argument(
        "--scope-confirmed-at",
        help="Deprecated. Use --scope-attestation-json instead.",
    )
    parser.add_argument(
        "--scope-confirmation",
        help="Deprecated. Use --scope-attestation-json instead.",
    )
    parser.add_argument(
        "--scope-attestation-json",
        type=Path,
        help=(
            "Machine-readable human approval record for adopting an existing "
            "W&B result. Requires --adopt-existing-result."
        ),
    )
    parser.add_argument(
        "--validated-dry-run-report-json",
        type=Path,
        help=(
            "Previously generated dry-run report from the exact same sync inputs. "
            "When provided, the current non-dry-run sync must match its entries "
            "and changes before any output JSON is written."
        ),
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    if args.in_place and args.output_json:
        raise SystemExit("--in-place and --output-json are mutually exclusive")
    if args.in_place and not args.validated_dry_run_report_json:
        raise SystemExit("--in-place requires --validated-dry-run-report-json")
    validate_adoption_args(args)
    review_path = repo_path(args.review_json)
    try:
        review = read_json_object(review_path)
        scope_attestation_path = (
            repo_path(args.scope_attestation_json)
            if args.scope_attestation_json
            else None
        )
        scope_attestation_payload = (
            read_json_object(scope_attestation_path) if scope_attestation_path else None
        )
        entries = [
            attach_scope_attestation(
                completion_entry(
                    repo_path(path),
                    read_json_object(repo_path(path)),
                    allow_failed=args.allow_failed,
                    require_query_source=not args.allow_failed
                    and not args.adopt_existing_result,
                ),
                args=args,
                review_path=review_path,
                scope_attestation_path=scope_attestation_path,
                scope_attestation_payload=scope_attestation_payload,
            )
            for path in args.completion_json
        ]
        actual_cost_estimate = validate_accounting_value(
            args.actual_cost_estimate,
            field="actual_cost_estimate",
        )
        provider_bill_reference = validate_accounting_value(
            args.provider_bill_reference,
            field="provider_bill_reference",
        )
        if scope_attestation_payload is not None:
            actual_cost_estimate = str(scope_attestation_payload["actual_cost_estimate"]).strip()
            provider_bill_reference = str(scope_attestation_payload["provider_bill_reference"]).strip()
    except (OSError, json.JSONDecodeError, ValueError) as exc:
        if args.report_json:
            report = build_validation_failure_report(
                args=args,
                review_path=review_path,
                error=exc,
            )
            write_json(repo_path(args.report_json), report)
            print(json.dumps(report, ensure_ascii=False, indent=2))
        raise SystemExit(f"error: {exc}")
    updated, changes, unmatched = sync_review(
        review,
        entries,
        top_level=bool(args.top_level),
        replace=not bool(args.no_replace),
        set_verify_wandb_completion=bool(args.set_verify_wandb_completion),
        actual_cost_estimate=actual_cost_estimate,
        provider_bill_reference=provider_bill_reference,
    )
    output_path = review_path if args.in_place else repo_path(args.output_json) if args.output_json else None
    dry_run = output_path is None
    report = build_report(
        review_path=review_path,
        output_path=output_path,
        review=review,
        updated=updated,
        entries=entries,
        changes=changes,
        unmatched=unmatched,
        in_place=bool(args.in_place),
        dry_run=dry_run,
    )
    if args.validated_dry_run_report_json:
        if dry_run:
            raise SystemExit(
                "--validated-dry-run-report-json requires --in-place or --output-json"
            )
        validated_dry_run_path = repo_path(args.validated_dry_run_report_json)
        validate_validated_dry_run_report(
            path=validated_dry_run_path,
            payload=read_json_object(validated_dry_run_path),
            current_report=report,
        )
        attach_validated_dry_run_report_path(
            updated,
            entries,
            top_level=bool(args.top_level),
            path=validated_dry_run_path,
            review_path=review_path,
        )
        report["validated_dry_run_report_json"] = path_display(validated_dry_run_path)
    if unmatched and not args.allow_unmatched:
        if args.report_json:
            write_json(repo_path(args.report_json), report)
        print(json.dumps(report, ensure_ascii=False, indent=2))
        raise SystemExit(1)
    if output_path is not None:
        write_json(output_path, updated)
    if args.report_json:
        write_json(repo_path(args.report_json), report)
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main(sys.argv[1:])
