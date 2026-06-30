#!/usr/bin/env python3
"""Draft a human review packet for adopting existing W&B completion evidence.

This tool is offline and read-only. It does not query W&B, write W&B, mutate
paid-run review JSONs, or run model inference. It converts existing-results
audit records into a reviewer checklist and exact sync commands with explicit
scope-attestation placeholders.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_REVIEW_JSON = "outputs/taiwan_full_eval/PHASE_paid_run_review.json"
SCOPE_ATTESTATION_SCHEMA_VERSION = 1
SCOPE_ATTESTATION_REQUIRED_HUMAN_FIELDS = [
    "scope_attestation_json.confirmed",
    "scope_attestation_json.confirmed_by",
    "scope_attestation_json.confirmed_at",
    "scope_attestation_json.confirmation",
    "scope_attestation_json.completion_sha256",
    "scope_attestation_json.actual_cost_estimate",
    "scope_attestation_json.provider_bill_reference",
]
BENCHMARK_REVIEW_PHASES = {
    "agentic_math": "canary_agentic",
    "agentic_swe": "canary_agentic",
    "taiwan_full": "canary_agentic_aggregate",
}


def repo_path(path: Path | str) -> Path:
    value = Path(path).expanduser()
    if value.is_absolute():
        return value
    return REPO_ROOT / value


def path_display(path: Path | str) -> str:
    value = Path(path)
    try:
        return str(value.resolve().relative_to(REPO_ROOT))
    except (OSError, ValueError):
        return str(value)


def read_json_object(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"{path} is not a JSON object")
    return payload


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def write_text(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def shell_quote(value: Any) -> str:
    text = str(value)
    if text and all(char.isalnum() or char in "/._:-=+" for char in text):
        return text
    return "'" + text.replace("'", "'\"'\"'") + "'"


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
    observed_metadata = observed.get("run_metadata") if isinstance(observed, dict) else None
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


def refresh_wandb_completion_command(candidate: dict[str, Any]) -> str:
    parts = [
        "uv run python scripts/tools/verify_taiwan_wandb_completion.py",
        "--entity",
        shell_quote(candidate.get("wandb_entity") or "WANDB_ENTITY"),
        "--project",
        shell_quote(candidate.get("wandb_project") or "WANDB_PROJECT"),
        "--run-id",
        shell_quote(candidate.get("wandb_run_id") or "RUN_ID"),
        "--benchmark",
        shell_quote(candidate.get("benchmark") or "BENCHMARK"),
    ]
    expected_total = candidate.get("metrics", {}).get("expected_total")
    if isinstance(expected_total, int):
        parts.extend(["--expected-total", str(expected_total)])
    model = candidate.get("model")
    if isinstance(model, str) and model:
        parts.extend(
            [
                "--expected-run-config",
                shell_quote(f"model.pretrained_model_name_or_path={model}"),
            ]
        )
    run_name = candidate.get("wandb_run_name")
    if isinstance(run_name, str) and "relog" in run_name:
        parts.extend(["--expected-run-job-type", "evaluation-relog"])
    if not any(part.startswith("--expected-run-") for part in parts):
        parts.extend(["--expected-run-tag", "REVIEWER_SELECTED_SCOPE_TAG"])
    parts.extend(["--json", shell_quote(candidate.get("wandb_completion_json") or "VERIFIER_JSON")])
    return " ".join(parts)


def sync_command(
    *,
    review_json: str,
    completion_json: str,
    benchmark: str,
    scope_attestation_json: str | None = None,
    in_place: bool = True,
    report_json: str | None = None,
    validated_dry_run_report_json: str | None = None,
) -> str:
    if not scope_attestation_json:
        return ""
    parts = [
        "uv run python scripts/tools/sync_wandb_completion_to_paid_review.py",
        "--review-json",
        shell_quote(review_json),
        "--completion-json",
        shell_quote(completion_json),
    ]
    if in_place:
        parts.append("--in-place")
    parts.extend(
        [
            "--set-verify-wandb-completion",
            "--top-level",
            "--adopt-existing-result",
            "--scope-attestation-json",
            shell_quote(scope_attestation_json),
        ]
    )
    if report_json:
        parts.extend(["--report-json", shell_quote(report_json)])
    if validated_dry_run_report_json:
        parts.extend(
            [
                "--validated-dry-run-report-json",
                shell_quote(validated_dry_run_report_json),
            ]
        )
    return " ".join(parts)


def scope_attestation_preflight_command(
    *,
    review_json: str,
    completion_json: str,
    scope_attestation_json: str,
    report_json: str,
) -> str:
    parts = [
        "uv run python scripts/tools/verify_wandb_scope_attestation.py",
        "--review-json",
        shell_quote(review_json),
        "--completion-json",
        shell_quote(completion_json),
        "--scope-attestation-json",
        shell_quote(scope_attestation_json),
        "--json",
        shell_quote(report_json),
    ]
    return " ".join(parts)


def scope_attestation_render_command(
    *,
    template_json: str,
    output_json: str,
    report_json: str,
    markdown: str,
    preflight_report_json: str,
    sync_dry_run_report_json: str,
) -> str:
    parts = [
        "uv run python scripts/tools/render_wandb_scope_attestation.py",
        "--template-json",
        shell_quote(template_json),
        "--output-json",
        shell_quote(output_json),
        "--confirmed-by",
        "REVIEWER_NAME",
        "--confirmed-at",
        "YYYY-MM-DDTHH:MM:SS+09:00",
        "--confirmation",
        shell_quote("CONFIRM_THIS_WANDB_RUN_IS_THE_REVIEWED_SCOPE"),
        "--actual-cost-estimate",
        "ACTUAL_COST_USD",
        "--provider-bill-reference",
        "PROVIDER_BILL_REFERENCE",
        "--report-json",
        shell_quote(report_json),
        "--markdown",
        shell_quote(markdown),
        "--preflight-report-json",
        shell_quote(preflight_report_json),
        "--sync-dry-run-report-json",
        shell_quote(sync_dry_run_report_json),
    ]
    return " ".join(parts)


def review_json_for_benchmark(review_json: str, benchmark: str) -> str:
    phase = BENCHMARK_REVIEW_PHASES.get(benchmark, "canary_full")
    return review_json.replace("PHASE", phase)


def slug_part(value: Any) -> str:
    text = str(value or "unknown")
    return "".join(char if char.isalnum() or char in "-_" else "_" for char in text)


def scope_attestation_template(candidate: dict[str, Any]) -> dict[str, Any]:
    benchmark = candidate.get("benchmark")
    return {
        "schema_version": SCOPE_ATTESTATION_SCHEMA_VERSION,
        "confirmed": False,
        "confirmed_by": "REVIEWER",
        "confirmed_at": "YYYY-MM-DDTHH:MM:SS+09:00",
        "confirmation": f"This W&B run is the reviewed canary scope for {benchmark}.",
        "review_path": candidate.get("target_review_json"),
        "completion_path": candidate.get("wandb_completion_json"),
        "completion_sha256": candidate.get("wandb_completion_sha256"),
        "source_audit_json": candidate.get("source_audit_json"),
        "source_audit_sha256": candidate.get("source_audit_sha256"),
        "benchmark": benchmark,
        "entity": candidate.get("wandb_entity"),
        "project": candidate.get("wandb_project"),
        "run_id": candidate.get("wandb_run_id"),
        "actual_cost_estimate": "$ACTUAL_OR_BILLING_ESTIMATE",
        "provider_bill_reference": "BILL_OR_DASHBOARD_REFERENCE",
        "instructions": [
            "Review the W&B run, verifier JSON, model, benchmark scope, and billing reference before changing confirmed to true.",
            "Keep source_audit_json and source_audit_sha256 unchanged; sync and release evidence checks use them to bind this attestation back to the existing-results audit.",
            "Replace $ACTUAL_OR_BILLING_ESTIMATE and BILL_OR_DASHBOARD_REFERENCE with concrete values; placeholder accounting strings are rejected by the sync and readiness gates.",
            "Do not use this template if the W&B run is outside the reviewed paid-run or agreed canary scope.",
            "After confirmation, pass this file to sync_wandb_completion_to_paid_review.py with --scope-attestation-json.",
        ],
    }


def record_candidate(
    record: dict[str, Any],
    *,
    review_json: str,
    source_audit_json: str = "",
    source_audit_sha256: str = "",
) -> dict[str, Any] | None:
    completion = record.get("wandb_completion")
    if not isinstance(completion, dict):
        return None
    completion_path = completion.get("path")
    entity = completion.get("entity")
    project = completion.get("project")
    run_id = completion.get("run_id")
    benchmark = record.get("benchmark")
    if not all(
        isinstance(value, str) and value
        for value in (completion_path, entity, project, run_id, benchmark)
    ):
        return None
    target_review_json = review_json_for_benchmark(review_json, benchmark)
    completion_sha256 = ""
    completion_payload: dict[str, Any] | None = None
    completion_read_error = ""
    try:
        completion_file = repo_path(completion_path)
        completion_sha256 = sha256_file(completion_file)
        completion_payload = read_json_object(completion_file)
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        completion_sha256 = ""
        completion_read_error = str(exc)
    metrics = {
        key: record.get(key)
        for key in (
            "expected_total",
            "row_count",
            "answered_instances",
            "correct_instances",
            "incorrect_instances",
            "accuracy",
            "resolved_instances",
            "pass_at_1",
        )
        if record.get(key) is not None
    }
    if completion_payload is None:
        run_metadata_valid = False
        run_metadata_errors = [
            "wandb_completion_json could not be read as a JSON object"
            + (f": {completion_read_error}" if completion_read_error else "")
        ]
    else:
        run_metadata_valid, run_metadata_errors = wandb_run_metadata_current(completion_payload)
    sync_ready = bool(
        completion_sha256
        and completion_payload is not None
        and completion.get("schema_current")
        and completion.get("observed_evidence_present")
        and run_metadata_valid
    )
    warnings = [
        "Do not attach this W&B run to a paid-run review unless it belongs to the reviewed paid run or agreed canary scope.",
        "If the reviewed scope is different, rerun the benchmark in the agreed one-model canary instead of adopting this existing result.",
    ]
    sync_command_blocked_reason = (
        "Run this draft command with --attestation-template-dir, review the generated "
        "scope-attestation JSON, set confirmed=true plus cost/bill fields, then sync "
        "with --scope-attestation-json. Run the generated dry-run command before the "
        "apply command."
    )
    if not run_metadata_valid:
        sync_command_blocked_reason = (
            "W&B completion verifier JSON is not sync-ready because run metadata is "
            "missing or invalid: "
            + "; ".join(run_metadata_errors)
            + ". Refresh the verifier JSON with --expected-run-config, "
            "--expected-run-tag, --expected-run-group, or --expected-run-job-type, "
            "then regenerate this adoption draft."
        )
        warnings.append(
            "The current verifier JSON lacks required run metadata; scope attestation alone is not enough for paid-review/release adoption."
        )
    elif not completion.get("schema_current"):
        sync_command_blocked_reason = (
            "W&B completion verifier JSON is not sync-ready because the verifier "
            "schema is not current. Regenerate the verifier JSON before adoption."
        )
        warnings.append("The current verifier JSON schema is not current.")
    elif not completion.get("observed_evidence_present"):
        sync_command_blocked_reason = (
            "W&B completion verifier JSON is not sync-ready because observed W&B "
            "evidence is missing. Regenerate or rerun the verifier before adoption."
        )
        warnings.append("The current verifier JSON does not include observed W&B completion evidence.")
    candidate = {
        "benchmark": benchmark,
        "model_slug": record.get("model_slug"),
        "model": record.get("model"),
        "run_kind": record.get("run_kind"),
        "result_dir": record.get("result_dir"),
        "wandb_entity": entity,
        "wandb_project": project,
        "wandb_run_id": run_id,
        "wandb_run_name": completion.get("run_name"),
        "wandb_completion_json": completion_path,
        "wandb_completion_sha256": completion_sha256,
        "source_audit_json": source_audit_json,
        "source_audit_sha256": source_audit_sha256,
        "verification_schema_version": completion.get("verification_schema_version"),
        "schema_current": bool(completion.get("schema_current")),
        "observed_evidence_present": bool(completion.get("observed_evidence_present")),
        "completion_json_readable": completion_payload is not None,
        "run_metadata_valid": run_metadata_valid,
        "run_metadata_errors": run_metadata_errors,
        "sync_ready": sync_ready,
        "metrics": metrics,
        "review_json_template": review_json,
        "target_review_json": target_review_json,
        "scope_attestation_required": True,
        "scope_attestation_schema_version": SCOPE_ATTESTATION_SCHEMA_VERSION,
        "required_human_fields": list(SCOPE_ATTESTATION_REQUIRED_HUMAN_FIELDS),
        "sync_command": sync_command(
            review_json=target_review_json,
            completion_json=completion_path,
            benchmark=benchmark,
        ),
        "sync_dry_run_command": "",
        "sync_apply_command": "",
        "sync_dry_run_report_json": "",
        "scope_attestation_render_command": "",
        "scope_attestation_render_report_json": "",
        "scope_attestation_render_markdown": "",
        "scope_attestation_preflight_command": "",
        "scope_attestation_preflight_report_json": "",
        "sync_command_blocked_reason": sync_command_blocked_reason,
        "scope_attestation_template": {
            "schema_version": SCOPE_ATTESTATION_SCHEMA_VERSION,
            "review_path": target_review_json,
            "completion_path": completion_path,
            "completion_sha256": completion_sha256,
            "source_audit_json": source_audit_json,
            "source_audit_sha256": source_audit_sha256,
            "benchmark": benchmark,
            "entity": entity,
            "project": project,
            "run_id": run_id,
            "confirmed": False,
        },
        "warnings": warnings,
    }
    if not run_metadata_valid:
        candidate["refresh_wandb_completion_command"] = refresh_wandb_completion_command(candidate)
    return candidate


def build_draft(
    *,
    audit_json: Path,
    review_json: str,
) -> dict[str, Any]:
    audit = read_json_object(audit_json)
    audit_json_display = path_display(audit_json)
    audit_sha256 = sha256_file(audit_json)
    formalized = audit.get("formalized_records")
    candidates: list[dict[str, Any]] = []
    if isinstance(formalized, list):
        for record in formalized:
            if not isinstance(record, dict):
                continue
            candidate = record_candidate(
                record,
                review_json=review_json,
                source_audit_json=audit_json_display,
                source_audit_sha256=audit_sha256,
            )
            if candidate is not None:
                candidates.append(candidate)
    return {
        "schema_version": 1,
        "ok": True,
        "status": "candidates_pending_scope_confirmation" if candidates else "no_candidates",
        "generated_at": time.time(),
        "source_audit_json": audit_json_display,
        "source_audit_sha256": audit_sha256,
        "review_json": review_json,
        "candidate_count": len(candidates),
        "requires_human_scope_confirmation": bool(candidates),
        "required_human_fields": list(SCOPE_ATTESTATION_REQUIRED_HUMAN_FIELDS)
        if candidates
        else [],
        "candidates": candidates,
    }


def write_scope_attestation_templates(
    *,
    draft: dict[str, Any],
    output_dir: Path,
) -> list[dict[str, Any]]:
    output_dir.mkdir(parents=True, exist_ok=True)
    written: list[dict[str, Any]] = []
    candidates = draft.get("candidates")
    if not isinstance(candidates, list):
        return written
    for candidate in candidates:
        if not isinstance(candidate, dict):
            continue
        filename = (
            slug_part(candidate.get("benchmark"))
            + "-"
            + slug_part(candidate.get("wandb_run_id"))
            + ".scope_attestation.json"
        )
        path = output_dir / filename
        template = scope_attestation_template(candidate)
        write_json(path, template)
        display = path_display(path)
        dry_run_report = path.with_name(
            filename.replace(".scope_attestation.json", ".sync_dry_run.json")
        )
        preflight_report = path.with_name(
            filename.replace(".scope_attestation.json", ".scope_preflight.json")
        )
        render_report = path.with_name(
            filename.replace(".scope_attestation.json", ".scope_attestation.render.json")
        )
        render_markdown = path.with_name(
            filename.replace(".scope_attestation.json", ".scope_attestation.render.md")
        )
        dry_run_report_display = path_display(dry_run_report)
        preflight_report_display = path_display(preflight_report)
        render_report_display = path_display(render_report)
        render_markdown_display = path_display(render_markdown)
        candidate["scope_attestation_template_json"] = display
        candidate["scope_attestation_render_report_json"] = render_report_display
        candidate["scope_attestation_render_markdown"] = render_markdown_display
        candidate["scope_attestation_preflight_report_json"] = preflight_report_display
        candidate["sync_dry_run_report_json"] = dry_run_report_display
        if candidate.get("sync_ready") is not True:
            candidate["sync_dry_run_command"] = ""
            candidate["sync_apply_command"] = ""
            candidate["scope_attestation_render_command"] = ""
            candidate["scope_attestation_preflight_command"] = ""
            candidate["sync_command"] = ""
            written.append(
                {
                    "benchmark": candidate.get("benchmark"),
                    "run_id": candidate.get("wandb_run_id"),
                    "path": display,
                    "render_report_json": render_report_display,
                    "render_markdown": render_markdown_display,
                    "preflight_report_json": preflight_report_display,
                    "dry_run_report_json": dry_run_report_display,
                    "sync_ready": False,
                }
            )
            continue
        candidate["scope_attestation_render_command"] = scope_attestation_render_command(
            template_json=display,
            output_json=display,
            report_json=render_report_display,
            markdown=render_markdown_display,
            preflight_report_json=preflight_report_display,
            sync_dry_run_report_json=dry_run_report_display,
        )
        candidate["scope_attestation_preflight_command"] = scope_attestation_preflight_command(
            review_json=str(candidate.get("target_review_json") or ""),
            completion_json=str(candidate.get("wandb_completion_json") or ""),
            scope_attestation_json=display,
            report_json=preflight_report_display,
        )
        candidate["sync_dry_run_command"] = sync_command(
            review_json=str(candidate.get("target_review_json") or ""),
            completion_json=str(candidate.get("wandb_completion_json") or ""),
            benchmark=str(candidate.get("benchmark") or ""),
            scope_attestation_json=display,
            in_place=False,
            report_json=dry_run_report_display,
        )
        candidate["sync_apply_command"] = sync_command(
            review_json=str(candidate.get("target_review_json") or ""),
            completion_json=str(candidate.get("wandb_completion_json") or ""),
            benchmark=str(candidate.get("benchmark") or ""),
            scope_attestation_json=display,
            in_place=True,
            validated_dry_run_report_json=dry_run_report_display,
        )
        candidate["sync_command"] = candidate["sync_apply_command"]
        candidate["sync_command_blocked_reason"] = ""
        written.append(
            {
                "benchmark": candidate.get("benchmark"),
                "run_id": candidate.get("wandb_run_id"),
                "path": display,
                "render_report_json": render_report_display,
                "render_markdown": render_markdown_display,
                "preflight_report_json": preflight_report_display,
                "dry_run_report_json": dry_run_report_display,
                "sync_ready": True,
            }
        )
    draft["scope_attestation_template_count"] = len(written)
    draft["scope_attestation_templates"] = written
    return written


def candidate_operator_handoff(candidate: dict[str, Any]) -> dict[str, Any]:
    required_human_fields = (
        candidate.get("required_human_fields")
        if isinstance(candidate.get("required_human_fields"), list)
        else []
    )
    sync_ready = candidate.get("sync_ready") is True
    blocked_reason = str(candidate.get("sync_command_blocked_reason") or "")
    if not sync_ready:
        refresh_command = candidate.get("refresh_wandb_completion_command")
        steps = []
        if isinstance(refresh_command, str) and refresh_command.strip():
            steps.append(
                {
                    "step": "refresh_wandb_completion_verifier",
                    "command": refresh_command,
                    "expected_evidence_paths": [candidate.get("wandb_completion_json")]
                    if isinstance(candidate.get("wandb_completion_json"), str)
                    and candidate.get("wandb_completion_json")
                    else [],
                    "requires_external_action": False,
                    "requires_scope_confirmation": False,
                    "mutates_review_json": False,
                    "required": True,
                }
            )
        evidence_paths = [
            path
            for step in steps
            for path in step.get("expected_evidence_paths", [])
            if isinstance(path, str) and path.strip()
        ]
        return {
            "available": False,
            "blocked_reason": blocked_reason,
            "requires_scope_confirmation": bool(candidate.get("scope_attestation_required")),
            "required_human_fields": required_human_fields,
            "pending_human_field_count": len(required_human_fields),
            "step_count": len(steps),
            "required_step_count": sum(1 for step in steps if step.get("required") is True),
            "command_count": sum(
                1 for step in steps if isinstance(step.get("command"), str) and step.get("command")
            ),
            "external_action_step_count": sum(
                1 for step in steps if step.get("requires_external_action") is True
            ),
            "scope_confirmation_step_count": sum(
                1 for step in steps if step.get("requires_scope_confirmation") is True
            ),
            "review_mutation_step_count": sum(
                1 for step in steps if step.get("mutates_review_json") is True
            ),
            "evidence_path_count": len(evidence_paths),
            "expected_evidence_paths": evidence_paths,
            "steps": steps,
        }

    steps = [
        {
            "step": "confirm_scope_attestation",
            "command": None,
            "expected_evidence_paths": [candidate.get("scope_attestation_template_json")],
            "requires_external_action": True,
            "requires_scope_confirmation": True,
            "mutates_review_json": False,
            "required": True,
        },
        {
            "step": "render_confirmed_attestation",
            "command": candidate.get("scope_attestation_render_command"),
            "expected_evidence_paths": [
                candidate.get("scope_attestation_template_json"),
                candidate.get("scope_attestation_render_report_json"),
                candidate.get("scope_attestation_render_markdown"),
            ],
            "requires_external_action": False,
            "requires_scope_confirmation": False,
            "mutates_review_json": False,
            "required": True,
        },
        {
            "step": "preflight_scope_attestation",
            "command": candidate.get("scope_attestation_preflight_command"),
            "expected_evidence_paths": [
                candidate.get("scope_attestation_preflight_report_json"),
            ],
            "requires_external_action": False,
            "requires_scope_confirmation": False,
            "mutates_review_json": False,
            "required": True,
        },
        {
            "step": "sync_paid_review_dry_run",
            "command": candidate.get("sync_dry_run_command"),
            "expected_evidence_paths": [candidate.get("sync_dry_run_report_json")],
            "requires_external_action": False,
            "requires_scope_confirmation": False,
            "mutates_review_json": False,
            "required": True,
        },
        {
            "step": "sync_paid_review_apply",
            "command": candidate.get("sync_apply_command") or candidate.get("sync_command"),
            "expected_evidence_paths": [candidate.get("target_review_json")],
            "requires_external_action": False,
            "requires_scope_confirmation": False,
            "mutates_review_json": True,
            "required": True,
        },
    ]
    normalized_steps: list[dict[str, Any]] = []
    for step in steps:
        paths = [
            path
            for path in step.get("expected_evidence_paths", [])
            if isinstance(path, str) and path.strip()
        ]
        normalized_steps.append({**step, "expected_evidence_paths": paths})
    evidence_paths: list[str] = []
    for step in normalized_steps:
        for path in step.get("expected_evidence_paths", []):
            if path not in evidence_paths:
                evidence_paths.append(path)
    return {
        "available": all(
            step.get("command") or step["step"] == "confirm_scope_attestation"
            for step in normalized_steps
        ),
        "blocked_reason": "",
        "requires_scope_confirmation": bool(candidate.get("scope_attestation_required")),
        "required_human_fields": required_human_fields,
        "pending_human_field_count": len(required_human_fields),
        "step_count": len(normalized_steps),
        "required_step_count": sum(
            1 for step in normalized_steps if step.get("required") is True
        ),
        "command_count": sum(
            1
            for step in normalized_steps
            if isinstance(step.get("command"), str) and step.get("command")
        ),
        "external_action_step_count": sum(
            1 for step in normalized_steps if step.get("requires_external_action") is True
        ),
        "scope_confirmation_step_count": sum(
            1
            for step in normalized_steps
            if step.get("requires_scope_confirmation") is True
        ),
        "review_mutation_step_count": sum(
            1 for step in normalized_steps if step.get("mutates_review_json") is True
        ),
        "evidence_path_count": len(evidence_paths),
        "expected_evidence_paths": evidence_paths,
        "steps": normalized_steps,
    }


def attach_operator_handoffs(draft: dict[str, Any]) -> None:
    candidates = draft.get("candidates")
    if not isinstance(candidates, list):
        draft["operator_handoff"] = {
            "candidate_count": 0,
            "available_candidate_count": 0,
            "step_count": 0,
            "command_count": 0,
            "external_action_step_count": 0,
            "scope_confirmation_step_count": 0,
            "review_mutation_step_count": 0,
            "evidence_path_count": 0,
        }
        return
    aggregate = {
        "candidate_count": 0,
        "available_candidate_count": 0,
        "step_count": 0,
        "command_count": 0,
        "external_action_step_count": 0,
        "scope_confirmation_step_count": 0,
        "review_mutation_step_count": 0,
        "evidence_path_count": 0,
    }
    evidence_paths: list[str] = []
    for candidate in candidates:
        if not isinstance(candidate, dict):
            continue
        handoff = candidate_operator_handoff(candidate)
        candidate["operator_handoff"] = handoff
        aggregate["candidate_count"] += 1
        if handoff.get("available") is True:
            aggregate["available_candidate_count"] += 1
        for field in (
            "step_count",
            "command_count",
            "external_action_step_count",
            "scope_confirmation_step_count",
            "review_mutation_step_count",
        ):
            value = handoff.get(field)
            if isinstance(value, int):
                aggregate[field] += value
        for path in handoff.get("expected_evidence_paths", []):
            if isinstance(path, str) and path.strip() and path not in evidence_paths:
                evidence_paths.append(path)
    aggregate["evidence_path_count"] = len(evidence_paths)
    aggregate["expected_evidence_paths"] = evidence_paths
    draft["operator_handoff"] = aggregate


def md_cell(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, bool):
        return str(value).lower()
    if isinstance(value, (list, dict)):
        text = json.dumps(value, ensure_ascii=False, sort_keys=True)
    else:
        text = str(value)
    return text.replace("\n", " ").replace("|", "\\|")


def markdown(draft: dict[str, Any]) -> str:
    lines = [
        "# Existing W&B Adoption Review Draft",
        "",
        f"Status: `{draft.get('status')}`",
        f"Candidates: `{draft.get('candidate_count')}`",
        f"Source audit: `{draft.get('source_audit_json')}`",
        f"Source audit SHA256: `{draft.get('source_audit_sha256')}`",
        "",
        "## Required Human Fields",
        "",
    ]
    fields = draft.get("required_human_fields")
    if isinstance(fields, list) and fields:
        lines.extend(f"- `{field}`" for field in fields)
    else:
        lines.append("- none")
    lines.extend(
        [
            "",
            "## Candidates",
            "",
            "| Benchmark | Model | Run ID | Schema | Observed evidence | Run metadata | Sync-ready | Target review | Completion JSON | Attestation template | Metrics |",
            "|---|---|---|---:|---:|---:|---:|---|---|---|---|",
        ]
    )
    candidates = draft.get("candidates")
    if isinstance(candidates, list) and candidates:
        for row in candidates:
            if not isinstance(row, dict):
                continue
            lines.append(
                "| "
                + " | ".join(
                    [
                        md_cell(row.get("benchmark")),
                        md_cell(row.get("model_slug")),
                        md_cell(row.get("wandb_run_id")),
                        md_cell(row.get("verification_schema_version")),
                        md_cell(row.get("observed_evidence_present")),
                        md_cell(row.get("run_metadata_valid")),
                        md_cell(row.get("sync_ready")),
                        md_cell(row.get("target_review_json")),
                        md_cell(row.get("wandb_completion_json")),
                        md_cell(row.get("scope_attestation_template_json")),
                        md_cell(row.get("metrics")),
                    ]
                )
                + " |"
            )
    else:
        lines.append("| none |  |  |  |  |  |  |  |  |  |  |")
    lines.extend(["", "## Commands", ""])
    if isinstance(candidates, list) and candidates:
        for index, row in enumerate(candidates, start=1):
            if not isinstance(row, dict):
                continue
            handoff = (
                row.get("operator_handoff")
                if isinstance(row.get("operator_handoff"), dict)
                else {}
            )
            lines.extend(
                [
                    f"### Candidate {index}",
                    "",
                    f"- Operator handoff available: `{md_cell(handoff.get('available'))}`",
                    f"- Handoff steps: `{md_cell(handoff.get('step_count'))}`",
                    f"- Handoff commands: `{md_cell(handoff.get('command_count'))}`",
                    f"- Scope confirmation steps: `{md_cell(handoff.get('scope_confirmation_step_count'))}`",
                    f"- Evidence paths: `{md_cell(handoff.get('evidence_path_count'))}`",
                    "",
                ]
            )
            lines.extend(
                [
                    f"#### Candidate {index} Handoff Steps",
                    "",
                    "| Step | Required | External action | Scope confirmation | Review mutation | Command | Expected evidence paths |",
                    "|---|---:|---:|---:|---:|---|---|",
                ]
            )
            steps = handoff.get("steps") if isinstance(handoff.get("steps"), list) else []
            if steps:
                for step in steps:
                    if not isinstance(step, dict):
                        continue
                    lines.append(
                        "| "
                        + " | ".join(
                            [
                                md_cell(step.get("step")),
                                md_cell(step.get("required")),
                                md_cell(step.get("requires_external_action")),
                                md_cell(step.get("requires_scope_confirmation")),
                                md_cell(step.get("mutates_review_json")),
                                md_cell(step.get("command")),
                                md_cell(step.get("expected_evidence_paths")),
                            ]
                        )
                        + " |"
                    )
            else:
                lines.append("| none |  |  |  |  |  |  |")
            lines.append("")
            dry_run_command = str(row.get("sync_dry_run_command") or "")
            render_command = str(row.get("scope_attestation_render_command") or "")
            preflight_command = str(row.get("scope_attestation_preflight_command") or "")
            apply_command = str(row.get("sync_apply_command") or row.get("sync_command") or "")
            if render_command or preflight_command or dry_run_command or apply_command:
                if render_command:
                    lines.extend(["Render confirmed attestation:", "", "```bash", render_command, "```", ""])
                if preflight_command:
                    lines.extend(["Preflight:", "", "```bash", preflight_command, "```", ""])
                if dry_run_command:
                    lines.extend(["Dry-run:", "", "```bash", dry_run_command, "```", ""])
                if apply_command:
                    lines.extend(["Apply:", "", "```bash", apply_command, "```"])
            else:
                lines.append(
                    "Sync command is withheld until a scope-attestation template is generated."
                )
                reason = row.get("sync_command_blocked_reason")
                if reason:
                    lines.append(str(reason))
                refresh_command = str(row.get("refresh_wandb_completion_command") or "")
                if refresh_command:
                    lines.extend(["", "Refresh verifier:", "", "```bash", refresh_command, "```"])
            lines.extend(
                [
                    "",
                    "Warnings:",
                    "",
                ]
            )
            warnings = row.get("warnings")
            if isinstance(warnings, list) and warnings:
                lines.extend(f"- {warning}" for warning in warnings)
            else:
                lines.append("- none")
            lines.append("")
    else:
        lines.append("- none")
    return "\n".join(lines)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--audit-json", type=Path, required=True)
    parser.add_argument("--review-json", default=DEFAULT_REVIEW_JSON)
    parser.add_argument("--json", type=Path)
    parser.add_argument("--markdown", type=Path)
    parser.add_argument(
        "--attestation-template-dir",
        type=Path,
        help="Write editable scope-attestation JSON templates for each candidate.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    draft = build_draft(
        audit_json=repo_path(args.audit_json),
        review_json=args.review_json,
    )
    if args.attestation_template_dir:
        write_scope_attestation_templates(
            draft=draft,
            output_dir=repo_path(args.attestation_template_dir),
        )
    attach_operator_handoffs(draft)
    if args.json:
        draft["path"] = str(repo_path(args.json))
    if args.markdown:
        draft["markdown_path"] = str(repo_path(args.markdown))
    if args.json:
        write_json(repo_path(args.json), draft)
    if args.markdown:
        write_text(repo_path(args.markdown), markdown(draft))
    print(json.dumps(draft, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
