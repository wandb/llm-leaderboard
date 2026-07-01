#!/usr/bin/env python3
"""Verify the stable pointer to the latest Taiwan release gate.

This command is offline and read-only. It validates that
temp/latest_taiwan_release_gate.json points to an existing formal timestamped
release-gate JSON and that the pointer summary matches the gate payload.
"""

from __future__ import annotations

import argparse
import json
import re
import time
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_POINTER_JSON = Path("temp") / "latest_taiwan_release_gate.json"
GATE_NAME_RE = re.compile(r"^taiwan_release_gate_(\d{8}T\d{6}Z)\.json$")
RELEASE_GATE_POINTER_PROOF_JSON = "release_gate_pointer_proof.json"
POINTER_GATE_SUMMARY_FIELDS = (
    "readiness_ok",
    "readiness_report_schema_version",
    "gate_count",
    "blocker_count",
    "blockers",
    "required_next_actions",
    "benchmark_completion",
    "weave_agents_completion",
    "existing_results_formalization",
    "wandb_adoption_draft",
    "paid_run_review_package",
    "wandb_completion_contract",
    "benchmark_progress_matrix",
    "nemoclaw_adoption",
    "operator_next_steps",
    "operator_plan",
    "external_action_checklist",
    "external_budget",
    "external_action_approval_packet",
    "bundle_file_count",
    "bundle_missing_file_count",
)
MANIFEST_CURRENT_GATE_SUMMARY_FIELDS = (
    "readiness_report_schema_version",
    "status",
    "readiness_status",
    "readiness_ok",
    "gate_count",
    "blocker_count",
    "blocking_gates",
    "required_next_actions",
    "benchmark_completion",
    "weave_agents_completion",
    "existing_results_formalization",
    "wandb_adoption_draft",
    "paid_run_review_package",
    "wandb_completion_contract",
    "benchmark_progress_matrix",
    "nemoclaw_adoption",
    "operator_next_steps",
    "external_action_checklist",
    "external_budget",
)


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


def same_path(left: str | Path | None, right: str | Path | None) -> bool:
    if left is None or right is None:
        return False
    try:
        return repo_path(left).resolve() == repo_path(right).resolve()
    except OSError:
        return repo_path(left) == repo_path(right)


def file_record_for_bundle_path(
    manifest: dict[str, Any],
    bundle_path_value: Any,
) -> dict[str, Any] | None:
    if not isinstance(bundle_path_value, str) or not bundle_path_value.strip():
        return None
    files = manifest.get("files")
    if not isinstance(files, list):
        return None
    for record in files:
        if not isinstance(record, dict):
            continue
        if record.get("bundle_path") == bundle_path_value:
            return record
    return None


def validate_pointer(pointer_path: Path) -> dict[str, Any]:
    issues: list[str] = []
    pointer: dict[str, Any] | None = None
    gate: dict[str, Any] | None = None
    gate_path: Path | None = None
    manifest: dict[str, Any] | None = None
    manifest_path: Path | None = None
    verification_payload: dict[str, Any] | None = None
    verification_path: Path | None = None
    pointer_verification_payload: dict[str, Any] | None = None
    pointer_verification_path: Path | None = None
    pointer_proof_payload: dict[str, Any] | None = None
    pointer_proof_path: Path | None = None
    operator_plan_payload: dict[str, Any] | None = None
    operator_plan_json_path: Path | None = None
    operator_plan_markdown_path: Path | None = None
    operator_plan_markdown_text: str | None = None

    if not pointer_path.exists():
        issues.append(f"pointer JSON does not exist: {path_display(pointer_path)}")
    else:
        try:
            pointer = read_json_object(pointer_path)
        except (OSError, json.JSONDecodeError, ValueError) as exc:
            issues.append(f"pointer JSON is not readable: {exc}")

    if pointer is not None:
        if pointer.get("schema_version") != 1:
            issues.append("pointer schema_version must be 1")
        timestamp = pointer.get("timestamp")
        if not isinstance(timestamp, str) or not timestamp:
            issues.append("pointer timestamp is required")

        latest_pointer_json = pointer.get("latest_pointer_json")
        if latest_pointer_json is None:
            issues.append("pointer latest_pointer_json is required")
        elif not isinstance(latest_pointer_json, str) or not latest_pointer_json:
            issues.append("pointer latest_pointer_json must be a non-empty string")
        elif not same_path(latest_pointer_json, pointer_path):
            issues.append("pointer latest_pointer_json does not point to this pointer")

        release_gate_json = pointer.get("release_gate_json")
        if not isinstance(release_gate_json, str) or not release_gate_json:
            issues.append("pointer release_gate_json is required")
        else:
            gate_path = repo_path(release_gate_json)
            match = GATE_NAME_RE.match(gate_path.name)
            if not match:
                issues.append(
                    "pointer release_gate_json must point to a formal "
                    "taiwan_release_gate_YYYYMMDDTHHMMSSZ.json file"
                )
            elif isinstance(timestamp, str) and timestamp and match.group(1) != timestamp:
                issues.append("pointer timestamp does not match release gate filename")
            if not gate_path.exists():
                issues.append(f"release gate JSON does not exist: {path_display(gate_path)}")
            else:
                try:
                    gate = read_json_object(gate_path)
                except (OSError, json.JSONDecodeError, ValueError) as exc:
                    issues.append(f"release gate JSON is not readable: {exc}")

        bundle_manifest = pointer.get("bundle_manifest")
        if not isinstance(bundle_manifest, str) or not bundle_manifest:
            issues.append("pointer bundle_manifest is required")
        else:
            manifest_path = repo_path(bundle_manifest)
            if not manifest_path.exists():
                issues.append(f"bundle manifest does not exist: {path_display(manifest_path)}")
            else:
                try:
                    manifest = read_json_object(manifest_path)
                except (OSError, json.JSONDecodeError, ValueError) as exc:
                    issues.append(f"bundle manifest is not readable: {exc}")

        bundle_verification_json = pointer.get("bundle_verification_json")
        if not isinstance(bundle_verification_json, str) or not bundle_verification_json:
            issues.append("pointer bundle_verification_json is required")
        else:
            verification_path = repo_path(bundle_verification_json)
            if not verification_path.exists():
                issues.append(
                    f"bundle verification JSON does not exist: {path_display(verification_path)}"
                )
            else:
                try:
                    verification_payload = read_json_object(verification_path)
                except (OSError, json.JSONDecodeError, ValueError) as exc:
                    issues.append(f"bundle verification JSON is not readable: {exc}")

        latest_pointer_verification_json = pointer.get(
            "latest_pointer_verification_json"
        )
        if latest_pointer_verification_json is not None:
            if (
                not isinstance(latest_pointer_verification_json, str)
                or not latest_pointer_verification_json
            ):
                issues.append(
                    "pointer latest_pointer_verification_json must be a non-empty string"
                )
            else:
                pointer_verification_path = repo_path(
                    latest_pointer_verification_json
                )
                if not pointer_verification_path.exists():
                    issues.append(
                        "latest pointer verification JSON does not exist: "
                        f"{path_display(pointer_verification_path)}"
                    )
                else:
                    try:
                        pointer_verification_payload = read_json_object(
                            pointer_verification_path
                        )
                    except (OSError, json.JSONDecodeError, ValueError) as exc:
                        issues.append(
                            "latest pointer verification JSON is not readable: "
                            f"{exc}"
                        )

        operator_plan_summary = pointer.get("operator_plan")
        if not isinstance(operator_plan_summary, dict):
            issues.append("pointer operator_plan is required")
        else:
            if operator_plan_summary.get("schema_version") != 1:
                issues.append("pointer operator_plan schema_version must be 1")
            if not isinstance(operator_plan_summary.get("status"), str) or not operator_plan_summary.get("status"):
                issues.append("pointer operator_plan status is required")

            operator_plan_json = operator_plan_summary.get("json")
            if not isinstance(operator_plan_json, str) or not operator_plan_json:
                issues.append("pointer operator_plan json is required")
            else:
                operator_plan_json_path = repo_path(operator_plan_json)
                if not operator_plan_json_path.exists():
                    issues.append(
                        "operator plan JSON does not exist: "
                        f"{path_display(operator_plan_json_path)}"
                    )
                else:
                    try:
                        operator_plan_payload = read_json_object(operator_plan_json_path)
                    except (OSError, json.JSONDecodeError, ValueError) as exc:
                        issues.append(f"operator plan JSON is not readable: {exc}")

            operator_plan_markdown = operator_plan_summary.get("markdown")
            if not isinstance(operator_plan_markdown, str) or not operator_plan_markdown:
                issues.append("pointer operator_plan markdown is required")
            else:
                operator_plan_markdown_path = repo_path(operator_plan_markdown)
                if not operator_plan_markdown_path.exists():
                    issues.append(
                        "operator plan Markdown does not exist: "
                        f"{path_display(operator_plan_markdown_path)}"
                    )
                else:
                    try:
                        operator_plan_markdown_text = operator_plan_markdown_path.read_text(
                            encoding="utf-8"
                        )
                    except OSError as exc:
                        issues.append(f"operator plan Markdown is not readable: {exc}")
                    else:
                        if not operator_plan_markdown_text.strip():
                            issues.append("operator plan Markdown is empty")

    if pointer is not None and gate is not None:
        if gate.get("schema_version") != 1:
            issues.append("release gate schema_version must be 1")
        if pointer.get("timestamp") != gate.get("timestamp"):
            issues.append("pointer timestamp does not match release gate payload")
        if pointer.get("status") != gate.get("status"):
            issues.append("pointer status does not match release gate payload")
        if bool(pointer.get("ok")) != bool(gate.get("ok")):
            issues.append("pointer ok does not match release gate payload")
        if bool(pointer.get("release_ready")) != bool(gate.get("release_ready")):
            issues.append("pointer release_ready does not match release gate payload")
        if pointer.get("readiness_status") != gate.get("readiness_status"):
            issues.append("pointer readiness_status does not match release gate payload")
        if pointer.get("readiness_report_source") != (
            gate.get("readiness_report_source") or gate.get("readiness_report")
        ):
            issues.append("pointer readiness_report_source does not match release gate payload")
        if pointer.get("blocking_gates") != (gate.get("blocking_gates") or []):
            issues.append("pointer blocking_gates do not match release gate payload")
        for field in POINTER_GATE_SUMMARY_FIELDS:
            if pointer.get(field) != gate.get(field):
                issues.append(f"pointer {field} does not match release gate payload")
        if not same_path(pointer.get("release_gate_json"), gate_path):
            issues.append("pointer release_gate_json does not resolve to the gate path")
        if not same_path(gate.get("latest_pointer_json"), pointer_path):
            issues.append("release gate latest_pointer_json does not point back to this pointer")
        if gate.get("latest_pointer_verification_json") is not None:
            if not same_path(
                pointer.get("latest_pointer_verification_json"),
                gate.get("latest_pointer_verification_json"),
            ):
                issues.append(
                    "pointer latest_pointer_verification_json does not match release gate payload"
                )
            for pointer_field, gate_field in (
                ("latest_pointer_verification_ok", "latest_pointer_verification_ok"),
                (
                    "latest_pointer_verification_status",
                    "latest_pointer_verification_status",
                ),
                (
                    "latest_pointer_verification_issue_count",
                    "latest_pointer_verification_issue_count",
                ),
            ):
                if pointer.get(pointer_field) != gate.get(gate_field):
                    issues.append(
                        f"pointer {pointer_field} does not match release gate payload"
                    )
            pointer_verification_summary = (
                pointer.get("latest_pointer_verification")
                if isinstance(pointer.get("latest_pointer_verification"), dict)
                else {}
            )
            gate_verification_summary = (
                gate.get("latest_pointer_verification")
                if isinstance(gate.get("latest_pointer_verification"), dict)
                else {}
            )
            if pointer_verification_summary != gate_verification_summary:
                issues.append(
                    "pointer latest_pointer_verification does not match release gate payload"
                )

        operator_plan_summary = (
            pointer.get("operator_plan")
            if isinstance(pointer.get("operator_plan"), dict)
            else {}
        )
        gate_operator_plan_summary = (
            gate.get("operator_plan")
            if isinstance(gate.get("operator_plan"), dict)
            else {}
        )
        if operator_plan_payload is not None:
            if operator_plan_payload.get("schema_version") != 1:
                issues.append("operator plan JSON schema_version must be 1")
            if operator_plan_payload.get("timestamp") != gate.get("timestamp"):
                issues.append("operator plan JSON timestamp does not match release gate payload")
            if operator_plan_payload.get("status") != operator_plan_summary.get("status"):
                issues.append("operator plan JSON status does not match pointer summary")
            if operator_plan_payload.get("status") != gate_operator_plan_summary.get("status"):
                issues.append("operator plan JSON status does not match release gate payload")
            if not same_path(
                operator_plan_payload.get("source_release_gate_json"),
                gate_path,
            ):
                issues.append(
                    "operator plan JSON source_release_gate_json does not match release gate"
                )
            if not same_path(operator_plan_payload.get("release_gate_json"), gate_path):
                issues.append("operator plan JSON release_gate_json does not match release gate")
            if operator_plan_payload.get("release_gate_status") != gate.get("status"):
                issues.append(
                    "operator plan JSON release_gate_status does not match release gate payload"
                )
            if bool(operator_plan_payload.get("release_ready")) != bool(
                gate.get("release_ready")
            ):
                issues.append(
                    "operator plan JSON release_ready does not match release gate payload"
                )
            if operator_plan_payload.get("readiness_status") != gate.get("readiness_status"):
                issues.append(
                    "operator plan JSON readiness_status does not match release gate payload"
                )
            if bool(operator_plan_payload.get("readiness_ok")) != bool(
                gate.get("readiness_ok")
            ):
                issues.append("operator plan JSON readiness_ok does not match release gate payload")
            if not same_path(
                operator_plan_payload.get("readiness_report_source"),
                gate.get("readiness_report_source") or gate.get("readiness_report"),
            ):
                issues.append(
                    "operator plan JSON readiness_report_source does not match release gate payload"
                )
            for field in (
                "blocking_gates",
                "operator_next_steps",
                "external_action_checklist",
                "wandb_adoption_draft",
                "paid_run_review_package",
                "wandb_completion_contract",
                "benchmark_progress_matrix",
                "nemoclaw_adoption",
            ):
                if operator_plan_payload.get(field) != gate.get(field):
                    issues.append(
                        f"operator plan JSON {field} does not match release gate payload"
                    )
            outputs = operator_plan_payload.get("outputs")
            if not isinstance(outputs, dict):
                issues.append("operator plan JSON outputs is required")
            else:
                if not same_path(outputs.get("json"), operator_plan_json_path):
                    issues.append("operator plan JSON outputs.json does not match operator plan path")
                if not same_path(outputs.get("markdown"), operator_plan_markdown_path):
                    issues.append(
                        "operator plan JSON outputs.markdown does not match operator plan Markdown"
                    )

        if operator_plan_markdown_text is not None:
            timestamp = gate.get("timestamp")
            if isinstance(timestamp, str) and timestamp:
                if f"Timestamp: `{timestamp}`" not in operator_plan_markdown_text:
                    issues.append("operator plan Markdown timestamp does not match release gate")
            if gate_path is not None and gate_path.name not in operator_plan_markdown_text:
                issues.append("operator plan Markdown does not mention release gate JSON")
            if gate.get("status") is not None and f"Release gate: `{gate.get('status')}`" not in operator_plan_markdown_text:
                issues.append("operator plan Markdown release gate status does not match")

        bundle = gate.get("bundle") if isinstance(gate.get("bundle"), dict) else {}
        verification = (
            gate.get("bundle_verification")
            if isinstance(gate.get("bundle_verification"), dict)
            else {}
        )
        if not same_path(pointer.get("bundle_manifest"), bundle.get("manifest")):
            issues.append("pointer bundle_manifest does not match release gate bundle manifest")
        if not same_path(pointer.get("bundle_verification_json"), verification.get("json")):
            issues.append("pointer bundle_verification_json does not match release gate verification JSON")
        if bool(pointer.get("bundle_integrity_ok")) != bool(gate.get("bundle_integrity_ok")):
            issues.append("pointer bundle_integrity_ok does not match release gate payload")
        if pointer.get("checked_file_count") != verification.get("checked_file_count"):
            issues.append("pointer checked_file_count does not match release gate verification")
        if pointer.get("verification_error_count") != len(verification.get("errors") or []):
            issues.append("pointer verification_error_count does not match release gate verification")

        if verification_payload is not None:
            if verification_payload.get("schema_version") != 1:
                issues.append("bundle verification JSON schema_version must be 1")
            expected_status = "passed" if verification_payload.get("ok") is True else "failed"
            if verification_payload.get("status") != expected_status:
                issues.append("bundle verification JSON status does not match ok")
            if verification_payload.get("error_count") != len(
                verification_payload.get("errors") or []
            ):
                issues.append("bundle verification JSON error_count does not match errors")
            if not same_path(verification_payload.get("manifest"), bundle.get("manifest")):
                issues.append(
                    "bundle verification JSON manifest does not match release gate bundle manifest"
                )
            if not same_path(verification_payload.get("manifest"), pointer.get("bundle_manifest")):
                issues.append("bundle verification JSON manifest does not match pointer")
            if not same_path(verification_payload.get("bundle_dir"), bundle.get("output_dir")):
                issues.append(
                    "bundle verification JSON bundle_dir does not match release gate bundle output_dir"
                )
            for field in (
                "ok",
                "integrity_ok",
                "readiness_ok",
                "readiness_status",
                "checked_file_count",
                "error_count",
                "errors",
                "require_ready",
            ):
                if verification_payload.get(field) != verification.get(field):
                    issues.append(
                        f"bundle verification JSON {field} does not match release gate payload"
                    )
            if bool(pointer.get("bundle_integrity_ok")) != bool(
                verification_payload.get("integrity_ok")
            ):
                issues.append(
                    "pointer bundle_integrity_ok does not match bundle verification JSON"
                )
            if pointer.get("checked_file_count") != verification_payload.get(
                "checked_file_count"
            ):
                issues.append(
                    "pointer checked_file_count does not match bundle verification JSON"
                )
            if pointer.get("verification_error_count") != len(
                verification_payload.get("errors") or []
            ):
                issues.append(
                    "pointer verification_error_count does not match bundle verification JSON"
                )

        if pointer_verification_payload is not None:
            if not same_path(
                pointer_verification_payload.get("pointer_json"),
                pointer_path,
            ):
                issues.append(
                    "latest pointer verification JSON pointer_json does not match this pointer"
                )
            if not same_path(
                pointer_verification_payload.get("release_gate_json"),
                gate_path,
            ):
                issues.append(
                    "latest pointer verification JSON release_gate_json does not match release gate"
                )
            for pointer_field, verification_field in (
                ("latest_pointer_verification_ok", "ok"),
                ("latest_pointer_verification_status", "status"),
                ("latest_pointer_verification_issue_count", "issue_count"),
            ):
                if pointer.get(pointer_field) != pointer_verification_payload.get(
                    verification_field
                ):
                    issues.append(
                        f"pointer {pointer_field} does not match latest pointer verification JSON"
                    )

    if pointer is not None and gate is not None and manifest is not None:
        if manifest.get("schema_version") != 1:
            issues.append("bundle manifest schema_version must be 1")
        if manifest.get("bundle_version") != 2:
            issues.append("bundle manifest bundle_version must be 2")
        if manifest.get("status") != gate.get("status"):
            issues.append("bundle manifest status does not match release gate payload")
        if manifest.get("readiness_status") != gate.get("readiness_status"):
            issues.append("bundle manifest readiness_status does not match release gate payload")
        if bool(manifest.get("readiness_ok")) != bool(gate.get("readiness_ok")):
            issues.append("bundle manifest readiness_ok does not match release gate payload")
        if manifest.get("gate_count") != gate.get("gate_count"):
            issues.append("bundle manifest gate_count does not match release gate payload")
        if manifest.get("blocker_count") != gate.get("blocker_count"):
            issues.append("bundle manifest blocker_count does not match release gate payload")
        if manifest.get("blocking_gates") != (gate.get("blocking_gates") or []):
            issues.append("bundle manifest blocking_gates do not match release gate payload")

        current_gate = manifest.get("current_gate")
        if not isinstance(current_gate, dict):
            issues.append("bundle manifest current_gate is required")
        else:
            for field in MANIFEST_CURRENT_GATE_SUMMARY_FIELDS:
                if current_gate.get(field) != gate.get(field):
                    issues.append(
                        "bundle manifest current_gate "
                        f"{field} does not match release gate payload"
                    )
            if not same_path(
                current_gate.get("readiness_report_source"),
                gate.get("readiness_report_source") or gate.get("readiness_report"),
            ):
                issues.append(
                    "bundle manifest current_gate readiness_report_source does not "
                    "match release gate payload"
                )
            runner_evidence = current_gate.get("runner_evidence")
            if not isinstance(runner_evidence, dict) or not runner_evidence:
                issues.append("bundle manifest current_gate runner_evidence is required")
            else:
                if runner_evidence.get("name") != "run_taiwan_release_gate.py":
                    issues.append("bundle manifest runner_evidence name is unexpected")
                if not same_path(
                    runner_evidence.get("report_json"),
                    gate.get("readiness_report_source") or gate.get("readiness_report"),
                ):
                    issues.append(
                        "bundle manifest runner_evidence report_json does not match "
                        "release gate readiness report"
                    )

        release_gate_pointer = manifest.get("release_gate_pointer")
        if not isinstance(release_gate_pointer, dict):
            issues.append("bundle manifest release_gate_pointer is required")
        else:
            if not same_path(
                release_gate_pointer.get("release_gate_json"),
                gate_path,
            ):
                issues.append(
                    "bundle manifest release_gate_pointer release_gate_json "
                    "does not match release gate"
                )
            if not same_path(
                release_gate_pointer.get("latest_pointer_json"),
                pointer_path,
            ):
                issues.append(
                    "bundle manifest release_gate_pointer latest_pointer_json "
                    "does not match this pointer"
                )
            if not same_path(
                release_gate_pointer.get("latest_pointer_verification_json"),
                pointer.get("latest_pointer_verification_json"),
            ):
                issues.append(
                    "bundle manifest release_gate_pointer "
                    "latest_pointer_verification_json does not match pointer"
                )
            for field in (
                "latest_pointer_verification_ok",
                "latest_pointer_verification_status",
                "latest_pointer_verification_issue_count",
            ):
                if release_gate_pointer.get(field) != pointer.get(field):
                    issues.append(
                        f"bundle manifest release_gate_pointer {field} "
                        "does not match pointer"
                    )

        pointer_proof_ref = manifest.get("release_gate_pointer_proof")
        if not isinstance(pointer_proof_ref, dict):
            issues.append("bundle manifest release_gate_pointer_proof is required")
        else:
            proof_json = pointer_proof_ref.get("json")
            if not isinstance(proof_json, str) or not proof_json:
                issues.append(
                    "bundle manifest release_gate_pointer_proof json is required"
                )
            elif Path(proof_json).name != RELEASE_GATE_POINTER_PROOF_JSON:
                issues.append(
                    "bundle manifest release_gate_pointer_proof json must be "
                    f"{RELEASE_GATE_POINTER_PROOF_JSON}"
                )
            if pointer_proof_ref.get("schema_version") != 1:
                issues.append(
                    "bundle manifest release_gate_pointer_proof schema_version must be 1"
                )
            if pointer_proof_ref.get("ok") is not True:
                issues.append("bundle manifest release_gate_pointer_proof ok must be true")
            if pointer_proof_ref.get("status") != "passed":
                issues.append(
                    "bundle manifest release_gate_pointer_proof status must be passed"
                )
            if isinstance(proof_json, str) and proof_json:
                record = file_record_for_bundle_path(manifest, proof_json)
                if not isinstance(record, dict):
                    issues.append(
                        "bundle manifest release_gate_pointer_proof json is not listed "
                        "in files"
                    )
                else:
                    roles = record.get("roles")
                    if (
                        not isinstance(roles, list)
                        or "release_gate_pointer:proof_json" not in roles
                    ):
                        issues.append(
                            "bundle manifest release_gate_pointer_proof json missing "
                            "role release_gate_pointer:proof_json"
                        )
                if manifest_path is not None:
                    pointer_proof_path = manifest_path.parent / proof_json
                    if not pointer_proof_path.exists():
                        issues.append(
                            "release gate pointer proof JSON does not exist: "
                            f"{path_display(pointer_proof_path)}"
                        )
                    else:
                        try:
                            pointer_proof_payload = read_json_object(
                                pointer_proof_path
                            )
                        except (OSError, json.JSONDecodeError, ValueError) as exc:
                            issues.append(
                                "release gate pointer proof JSON is not readable: "
                                f"{exc}"
                            )

        if pointer_proof_payload is not None:
            if pointer_proof_payload.get("schema_version") != 1:
                issues.append("release gate pointer proof JSON schema_version must be 1")
            if pointer_proof_payload.get("kind") != "release_gate_pointer_proof":
                issues.append(
                    "release gate pointer proof JSON kind must be release_gate_pointer_proof"
                )
            if pointer_proof_payload.get("ok") is not True:
                issues.append("release gate pointer proof JSON ok must be true")
            if pointer_proof_payload.get("status") != "passed":
                issues.append("release gate pointer proof JSON status must be passed")
            if pointer_proof_payload.get("release_gate_pointer") != release_gate_pointer:
                issues.append(
                    "release gate pointer proof JSON release_gate_pointer "
                    "does not match bundle manifest"
                )
            proof_verification = pointer_proof_payload.get(
                "latest_pointer_verification"
            )
            if not isinstance(proof_verification, dict):
                issues.append(
                    "release gate pointer proof JSON latest_pointer_verification "
                    "is required"
                )
            else:
                if not same_path(
                    proof_verification.get("json"),
                    pointer.get("latest_pointer_verification_json"),
                ):
                    issues.append(
                        "release gate pointer proof JSON latest_pointer_verification "
                        "json does not match pointer"
                    )
                if proof_verification.get("ok") is not True:
                    issues.append(
                        "release gate pointer proof JSON latest_pointer_verification "
                        "ok must be true"
                    )
                if proof_verification.get("status") != "passed":
                    issues.append(
                        "release gate pointer proof JSON latest_pointer_verification "
                        "status must be passed"
                    )
                if proof_verification.get("issue_count") != 0:
                    issues.append(
                        "release gate pointer proof JSON latest_pointer_verification "
                        "issue_count must be 0"
                    )

    return {
        "schema_version": 1,
        "ok": not issues,
        "status": "passed" if not issues else "failed",
        "generated_at": time.time(),
        "pointer_json": path_display(pointer_path),
        "release_gate_json": path_display(gate_path) if gate_path else None,
        "bundle_manifest": path_display(manifest_path) if manifest_path else None,
        "bundle_verification_json": (
            path_display(verification_path) if verification_path else None
        ),
        "issue_count": len(issues),
        "issues": issues,
        "pointer": pointer or {},
    }


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pointer-json", type=Path, default=DEFAULT_POINTER_JSON)
    parser.add_argument("--json", type=Path)
    parser.add_argument("--fail-on-invalid", action="store_true")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    result = validate_pointer(repo_path(args.pointer_json))
    if args.json:
        write_json(repo_path(args.json), result)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    if args.fail_on_invalid and not result["ok"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
