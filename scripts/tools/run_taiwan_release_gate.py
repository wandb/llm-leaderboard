#!/usr/bin/env python3
"""Run the Taiwan production gate and package release evidence.

This command is intentionally offline with respect to model providers. It
refreshes local readiness evidence, writes the production-readiness report,
builds a portable evidence bundle, and verifies the bundle hashes.
"""

from __future__ import annotations

import argparse
import glob
import json
import shutil
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import build_taiwan_release_evidence_bundle as bundle_builder
import run_taiwan_production_readiness_gate as readiness_gate
import verify_taiwan_release_gate_pointer as pointer_verifier
import verify_taiwan_release_evidence_bundle as bundle_verifier


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUTPUT_DIR = Path("temp")
DEFAULT_BUNDLE_OUTPUT_ROOT = Path("outputs") / "taiwan_release_evidence"


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


def latest_weave_agents_adoption_validation_failures(limit: int = 20) -> dict[str, Any]:
    records: list[dict[str, Any]] = []
    paths = sorted(
        (Path(value) for value in glob.glob(str(repo_path("temp/weave_agents_sync_*.validation_failed.json")))),
        key=lambda path: (path.stat().st_mtime if path.exists() else -1.0, str(path)),
        reverse=True,
    )
    for path in paths[:limit]:
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        if not isinstance(payload, dict) or payload.get("status") != "validation_failed":
            continue
        records.append(
            {
                "path": path_display(path),
                "ok": payload.get("ok"),
                "status": payload.get("status"),
                "generated_at": payload.get("generated_at"),
                "review_path": payload.get("review_path"),
                "source_review_sha256": payload.get("source_review_sha256"),
                "completion_paths": (
                    payload.get("completion_paths")
                    if isinstance(payload.get("completion_paths"), list)
                    else []
                ),
                "validation_errors": (
                    payload.get("validation_errors")
                    if isinstance(payload.get("validation_errors"), list)
                    else []
                ),
                "dry_run": payload.get("dry_run"),
                "in_place": payload.get("in_place"),
                "entry_count": payload.get("entry_count"),
                "change_count": payload.get("change_count"),
            }
        )
    return {
        "ok": not records,
        "status": "none" if not records else "validation_failed_reports_present",
        "record_count": len(records),
        "glob": "temp/weave_agents_sync_*.validation_failed.json",
        "records": records,
    }


def run_readiness_gate(
    args: argparse.Namespace,
    forwarded_args: list[str],
) -> tuple[Path, dict[str, Any]]:
    report_json = repo_path(
        args.readiness_report_json
        or args.output_dir / f"taiwan_production_readiness_report_{args.timestamp}.json"
    )
    gate_argv = [
        "--output-dir",
        str(args.output_dir),
        "--timestamp",
        args.timestamp,
        "--report-json",
        str(report_json),
        "--quiet",
        *forwarded_args,
    ]
    gate_args, report_args = readiness_gate.parse_args(gate_argv)
    nemoclaw_check = readiness_gate.run_nemoclaw_check(gate_args, gate_args.timestamp)
    existing_results_check = readiness_gate.run_existing_results_audit(
        gate_args,
        gate_args.timestamp,
    )
    wandb_adoption_draft = readiness_gate.run_wandb_adoption_draft(
        gate_args,
        gate_args.timestamp,
        existing_results_check,
    )
    wandb_adoption_unconfirmed_checks = (
        readiness_gate.run_wandb_adoption_unconfirmed_checks(
            gate_args,
            gate_args.timestamp,
            wandb_adoption_draft,
        )
    )
    nemoclaw_operator_docs_check = (
        readiness_gate.run_nemoclaw_operator_docs_verification(
            gate_args,
            gate_args.timestamp,
        )
    )
    nemoclaw_post_install_verification = (
        readiness_gate.run_nemoclaw_post_install_verification(
            gate_args,
            gate_args.timestamp,
        )
    )
    parsed_report_args = readiness_gate.build_report_args(
        gate_args,
        report_args,
        report_json=report_json,
        nemoclaw_check=nemoclaw_check,
        existing_results_check=existing_results_check,
        nemoclaw_operator_docs_check=nemoclaw_operator_docs_check,
    )
    report = readiness_gate.readiness.build_report(parsed_report_args)
    paid_review_check = readiness_gate.run_paid_review_check(
        gate_args,
        gate_args.timestamp,
        parsed_report_args,
    )
    nemoclaw_adoption_check = readiness_gate.run_nemoclaw_adoption_check(
        gate_args,
        gate_args.timestamp,
        parsed_report_args,
    )
    report["runner"] = {
        "name": "run_taiwan_release_gate.py",
        "generated_at": time.time(),
        "report_json": str(report_json),
        "nemoclaw_check": nemoclaw_check,
        "existing_results_audit": existing_results_check,
        "wandb_adoption_draft": wandb_adoption_draft,
        "wandb_adoption_unconfirmed_checks": wandb_adoption_unconfirmed_checks,
        "nemoclaw_operator_docs_verification": nemoclaw_operator_docs_check,
        "paid_run_review_check": paid_review_check,
        "weave_agents_adoption_validation_failures": (
            latest_weave_agents_adoption_validation_failures()
        ),
        "nemoclaw_adoption_check": nemoclaw_adoption_check,
        "nemoclaw_post_install_verification": nemoclaw_post_install_verification,
        "forwarded_report_args": report_args,
    }
    if parsed_report_args.json:
        readiness_gate.write_report(repo_path(parsed_report_args.json), report)
    return report_json, report


def build_bundle(
    *,
    report_path: Path,
    output_dir: Path,
) -> dict[str, Any]:
    report = bundle_builder.read_json(report_path)
    output_dir.mkdir(parents=True, exist_ok=True)
    evidence = bundle_builder.collect_evidence(report_path, report)
    bundle_builder.add_computed_operator_command_script_evidence(
        evidence,
        report_path=report_path,
        report=report,
    )
    evidence_files = bundle_builder.copy_evidence_files(
        evidence,
        output_dir=output_dir,
    )
    manifest = bundle_builder.build_manifest(
        report_path=report_path,
        report=report,
        evidence_files=evidence_files,
    )
    bundle_builder.add_operator_plan_files(output_dir, manifest)
    bundle_builder.add_external_action_approval_packet_files(output_dir, manifest)
    manifest_path = output_dir / "manifest.json"
    summary_path = output_dir / "summary.md"
    summary_path.write_text(bundle_builder.summary_markdown(manifest), encoding="utf-8")
    files = manifest.get("files")
    if not isinstance(files, list):
        files = []
        manifest["files"] = files
    files.append(
        bundle_builder.generated_bundle_file_record(
            summary_path,
            bundle_path=Path("summary.md"),
            roles=["release_summary", "release_summary_markdown"],
        )
    )
    bundle_builder.write_json(manifest_path, manifest)
    return {
        "output_dir": str(output_dir),
        "manifest": str(manifest_path),
        "summary": str(summary_path),
        "operator_plan": manifest.get("operator_plan"),
        "external_action_approval_packet": manifest.get("external_action_approval_packet"),
        "readiness_ok": bool(manifest.get("readiness_ok")),
        "readiness_status": manifest.get("readiness_status"),
        "file_count": len(manifest.get("files") or []),
        "missing_file_count": sum(
            1 for record in manifest.get("files") or [] if not record.get("exists")
        ),
    }


def verify_bundle(
    *,
    manifest_path: Path,
    verification_json: Path,
    require_ready: bool,
) -> dict[str, Any]:
    result = bundle_verifier.verify_bundle(
        manifest_path=manifest_path,
        require_ready=require_ready,
    )
    bundle_verifier.write_json(verification_json, result)
    result["json"] = str(verification_json)
    return result


def format_summary(result: dict[str, Any]) -> str:
    lines = [
        f"Taiwan release gate: {result.get('status')}",
        f"Readiness report: {result.get('readiness_report')}",
        f"Evidence bundle: {result.get('bundle', {}).get('manifest')}",
        f"Bundle verification: {result.get('bundle_verification', {}).get('json')}",
        f"Latest pointer verification: {result.get('latest_pointer_verification_json')}",
        f"Bundle integrity: {result.get('bundle_integrity_ok')}",
        f"Release ready: {result.get('release_ready')}",
    ]
    blockers = result.get("blocking_gates")
    if isinstance(blockers, list) and blockers:
        lines.append("Blocking gates: " + ", ".join(str(item) for item in blockers))
    else:
        lines.append("Blocking gates: none")
    lines.extend(format_operator_next_steps(result.get("operator_next_steps")))
    return "\n".join(lines)


def format_operator_next_steps(operator_steps: object) -> list[str]:
    if not isinstance(operator_steps, dict):
        return []

    status = operator_steps.get("status", "unknown")
    step_count = operator_steps.get("step_count")
    summary = f"Operator next steps: {status}"
    if isinstance(step_count, int):
        summary += f" ({step_count} steps)"
    lines = [summary]

    requirement_counts = [
        ("paid_api", "paid_api_step_count"),
        ("wandb_access", "wandb_access_step_count"),
        ("wandb_write", "wandb_write_step_count"),
        ("third_party_acceptance", "third_party_acceptance_step_count"),
        ("scope_confirmation", "scope_confirmation_step_count"),
    ]
    requirements = []
    for label, key in requirement_counts:
        value = operator_steps.get(key)
        if isinstance(value, int) and value > 0:
            requirements.append(f"{label}={value}")
    if requirements:
        lines.append("Operator requirements: " + ", ".join(requirements))
    template_count = operator_steps.get("command_template_step_count")
    if isinstance(template_count, int) and template_count > 0:
        lines.append(f"Operator command templates: {template_count}")
    placeholder_tokens = operator_steps.get("unresolved_placeholder_tokens")
    if isinstance(placeholder_tokens, list) and placeholder_tokens:
        lines.append(
            "Operator unresolved placeholders: "
            + ", ".join(str(token) for token in placeholder_tokens)
        )

    gates = []
    for step in operator_steps.get("steps") or []:
        if not isinstance(step, dict):
            continue
        gate = step.get("gate")
        if isinstance(gate, str) and gate and gate not in gates:
            gates.append(gate)
    if gates:
        lines.append("Operator gates: " + ", ".join(gates))

    warnings = operator_steps.get("warnings")
    if isinstance(warnings, list) and warnings:
        lines.append(f"Operator warnings: {len(warnings)}")

    return lines


def build_operator_plan(
    result: dict[str, Any],
    *,
    json_path: Path,
    markdown_path: Path,
) -> dict[str, Any]:
    operator_steps = result.get("operator_next_steps")
    if not isinstance(operator_steps, dict):
        operator_steps = {}
    blocking_gates = result.get("blocking_gates") or []
    if not isinstance(blocking_gates, list):
        blocking_gates = []
    external_action_checklist = bundle_builder.build_external_action_checklist(
        operator_steps=operator_steps,
        blocking_gates=blocking_gates,
        paid_run_review_package=(
            result.get("paid_run_review_package")
            if isinstance(result.get("paid_run_review_package"), dict)
            else {}
        ),
    )
    approval_source_packet_json = ""
    bundle = result.get("bundle") if isinstance(result.get("bundle"), dict) else {}
    manifest_value = bundle.get("manifest") if isinstance(bundle, dict) else None
    packet_ref = (
        bundle.get("external_action_approval_packet")
        if isinstance(bundle.get("external_action_approval_packet"), dict)
        else {}
    )
    packet_json = packet_ref.get("json")
    if isinstance(manifest_value, str) and isinstance(packet_json, str) and packet_json:
        packet_path = Path(packet_json)
        if not packet_path.is_absolute():
            packet_path = repo_path(manifest_value).parent / packet_path
        approval_source_packet_json = str(packet_path)
    return {
        "schema_version": 1,
        "generated_at": time.time(),
        "timestamp": result.get("timestamp"),
        "status": operator_steps.get("status", "unknown"),
        "source_release_gate_json": result.get("release_gate_json"),
        "release_gate_json": result.get("release_gate_json"),
        "release_gate_status": result.get("status"),
        "release_ready": bool(result.get("release_ready")),
        "readiness_status": result.get("readiness_status"),
        "readiness_ok": bool(result.get("readiness_ok")),
        "readiness_report_source": result.get("readiness_report_source")
        or result.get("readiness_report"),
        "blocking_gates": blocking_gates,
        "operator_next_steps": operator_steps,
        "operator_execution_plan_renderer": (
            bundle_builder.operator_execution_plan_renderer_summary(
                operator_plan_json=str(json_path),
                operator_steps=operator_steps,
                approval_source_packet_json=approval_source_packet_json
                or "external_action_approval_packet.json",
                release_gate_json=str(result.get("release_gate_json") or ""),
            )
        ),
        "external_action_checklist": external_action_checklist,
        "wandb_completion_contract": result.get("wandb_completion_contract") or {},
        "benchmark_progress_matrix": result.get("benchmark_progress_matrix") or [],
        "wandb_adoption_draft": result.get("wandb_adoption_draft") or {},
        "paid_run_review_package": result.get("paid_run_review_package") or {},
        "nemoclaw_adoption": result.get("nemoclaw_adoption") or {},
        "outputs": {
            "json": str(json_path),
            "markdown": str(markdown_path),
            "release_gate_json": result.get("release_gate_json"),
            "latest_pointer_json": result.get("latest_pointer_json"),
            "latest_pointer_verification_json": result.get(
                "latest_pointer_verification_json"
            ),
            "readiness_report": result.get("readiness_report"),
            "evidence_bundle": (result.get("bundle") or {}).get("manifest"),
            "bundle_verification": (result.get("bundle_verification") or {}).get("json"),
        },
    }


def format_bool(value: object) -> str:
    return "true" if bool(value) else "false"


def operator_plan_markdown(plan: dict[str, Any]) -> str:
    operator_steps = plan.get("operator_next_steps")
    if not isinstance(operator_steps, dict):
        operator_steps = {}
    lines = [
        "# Taiwan Release Operator Plan",
        "",
        f"Status: `{plan.get('status', 'unknown')}`",
        f"Timestamp: `{plan.get('timestamp') or ''}`",
        f"Source release gate JSON: `{plan.get('source_release_gate_json') or ''}`",
        f"Release gate: `{plan.get('release_gate_status', 'unknown')}`",
        f"Release ready: `{format_bool(plan.get('release_ready'))}`",
        f"Readiness status: `{plan.get('readiness_status', 'unknown')}`",
        f"Readiness OK: `{format_bool(plan.get('readiness_ok'))}`",
        f"Readiness report: `{plan.get('readiness_report_source') or ''}`",
        "",
        "## Blocking Gates",
        "",
    ]
    blockers = plan.get("blocking_gates")
    if isinstance(blockers, list) and blockers:
        lines.extend(f"- `{blocker}`" for blocker in blockers)
    else:
        lines.append("- none")

    lines.extend(
        [
            "",
            "## Requirement Counts",
            "",
            "| Requirement | Count |",
            "|---|---:|",
        ]
    )
    for label, key in [
        ("Paid API", "paid_api_step_count"),
        ("W&B access", "wandb_access_step_count"),
        ("W&B write", "wandb_write_step_count"),
        ("Third-party acceptance", "third_party_acceptance_step_count"),
        ("Scope confirmation", "scope_confirmation_step_count"),
        ("Command-template steps", "command_template_step_count"),
    ]:
        value = operator_steps.get(key)
        lines.append(f"| {label} | {value if isinstance(value, int) else 0} |")

    placeholder_tokens = operator_steps.get("unresolved_placeholder_tokens")
    if isinstance(placeholder_tokens, list) and placeholder_tokens:
        lines.extend(
            [
                "",
                "## Unresolved Placeholders",
                "",
            ]
        )
        lines.extend(f"- `{token}`" for token in placeholder_tokens)

    warnings = operator_steps.get("warnings")
    if isinstance(warnings, list) and warnings:
        lines.extend(["", "## Warnings", ""])
        lines.extend(f"- {warning}" for warning in warnings)

    renderer = plan.get("operator_execution_plan_renderer")
    if isinstance(renderer, dict):
        lines.extend(
            [
                "",
                "## Operator Execution Plan Renderer",
                "",
                f"Status: `{renderer.get('status', 'unknown')}`",
                f"Required before external action: `{format_bool(renderer.get('required_before_external_action'))}`",
                f"Script: `{renderer.get('script', '')}`",
                "",
                "Placeholder tokens:",
                "",
            ]
        )
        tokens = renderer.get("placeholder_tokens")
        if isinstance(tokens, list) and tokens:
            lines.extend(f"- `{token}`" for token in tokens)
        else:
            lines.append("- none")
        lines.extend(
            [
                "",
                "Commands:",
                "",
                f"- Review: `{renderer.get('review_command_template', '')}`",
                f"- Placeholder-ready shell: `{renderer.get('require_ready_command_template', '')}`",
                "",
                "Expected outputs:",
                "",
            ]
        )
        outputs = renderer.get("expected_outputs")
        if isinstance(outputs, dict):
            for key in ("json", "markdown", "shell_script"):
                value = outputs.get(key)
                if value:
                    lines.append(f"- {key}: `{value}`")

    lines.extend(
        bundle_builder.benchmark_progress_matrix_markdown(
            plan.get("benchmark_progress_matrix")
        )
    )

    lines.extend(
        bundle_builder.external_action_checklist_markdown(
            plan.get("external_action_checklist")
        )
    )

    lines.extend(["", "## Steps", ""])
    steps = operator_steps.get("steps")
    if not isinstance(steps, list) or not steps:
        lines.append("- none")
    else:
        for step in steps:
            if not isinstance(step, dict):
                continue
            required_labels = [
                label
                for label, key in [
                    ("paid API", "requires_paid_api"),
                    ("W&B access", "requires_wandb_access"),
                    ("W&B write", "requires_wandb_write"),
                    (
                        "third-party acceptance",
                        "requires_third_party_acceptance",
                    ),
                    ("NeMoClaw install", "requires_nemoclaw_install"),
                    ("scope confirmation", "requires_scope_confirmation"),
                ]
                if step.get(key)
            ]
            lines.extend(
                [
                    f"### {step.get('order', '?')}. {step.get('gate', 'unknown')}",
                    "",
                    f"- Status: `{step.get('status', 'unknown')}`",
                    f"- Next action: {step.get('next_action', '')}",
                    "- Requires: " + (", ".join(required_labels) or "none"),
                    f"- Commands: `{step.get('command_count', 0)}` total",
                    f"- Evidence paths: `{step.get('evidence_path_count', 0)}` total",
                    f"- Template commands: `{step.get('command_template_count', 0)}`",
                    f"- Evidence templates: `{step.get('evidence_template_count', 0)}`",
                    "- Placeholder tokens: "
                    + (
                        ", ".join(
                            f"`{token}`"
                            for token in step.get("unresolved_placeholder_tokens", [])
                            if isinstance(token, str)
                        )
                        or "none"
                    ),
                    f"- Ready without placeholder edit: `{format_bool(step.get('ready_to_execute_without_placeholder'))}`",
                    "",
                    "Commands:",
                    "",
                ]
            )
            commands = step.get("commands")
            if isinstance(commands, list) and commands:
                for command in commands:
                    lines.append(f"- `{command}`")
            else:
                lines.append("- none")
            evidence = step.get("evidence_to_produce")
            if isinstance(evidence, list) and evidence:
                lines.extend(["", "Evidence to produce:", ""])
                lines.extend(f"- `{path}`" for path in evidence)
            step_warnings = step.get("warnings")
            if isinstance(step_warnings, list) and step_warnings:
                lines.extend(["", "Step warnings:", ""])
                lines.extend(f"- {warning}" for warning in step_warnings)
            lines.append("")

    outputs = plan.get("outputs")
    if isinstance(outputs, dict):
        lines.extend(
            [
                "## Evidence Files",
                "",
                "| Type | Path |",
                "|---|---|",
            ]
        )
        for key in [
            "json",
            "markdown",
            "release_gate_json",
            "latest_pointer_json",
            "latest_pointer_verification_json",
            "readiness_report",
            "evidence_bundle",
            "bundle_verification",
        ]:
            value = outputs.get(key)
            if value:
                lines.append(f"| {key} | `{value}` |")
    lines.append("")
    return "\n".join(lines)


def write_text(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


def build_result(
    *,
    report_path: Path,
    report: dict[str, Any],
    bundle: dict[str, Any],
    verification: dict[str, Any],
    require_ready: bool,
    forwarded_args: list[str],
) -> dict[str, Any]:
    summary = report.get("summary") if isinstance(report.get("summary"), dict) else {}
    gates = report.get("gates") if isinstance(report.get("gates"), list) else []
    blocking_gates = summary.get("blockers", [])
    required_next_actions = bundle_builder.required_next_actions(gates)
    remediation = report.get("remediation_plan")
    if not isinstance(remediation, list):
        remediation = []
    benchmark_completion = bundle_builder.benchmark_completion_summary(report)
    weave_agents_completion = bundle_builder.weave_agents_completion_summary(report)
    existing_results_formalization = (
        bundle_builder.existing_results_formalization_summary(report)
    )
    wandb_adoption_draft = bundle_builder.wandb_adoption_draft_summary(report)
    paid_run_review_package = bundle_builder.paid_run_review_package_summary(report)
    wandb_completion_contract = bundle_builder.wandb_completion_contract_summary(
        benchmark_completion=benchmark_completion,
        existing_results_formalization=existing_results_formalization,
        wandb_adoption_draft=wandb_adoption_draft,
        paid_run_review_package=paid_run_review_package,
    )
    benchmark_progress_matrix = bundle_builder.benchmark_progress_matrix_summary(
        wandb_completion_contract=wandb_completion_contract,
        paid_run_review_package=paid_run_review_package,
        weave_agents_completion=weave_agents_completion,
    )
    nemoclaw_adoption = bundle_builder.nemoclaw_adoption_summary(report)
    operator_next_steps = bundle_builder.operator_next_steps_summary(
        remediation_plan=remediation,
        wandb_completion_contract=wandb_completion_contract,
        paid_run_review_package=paid_run_review_package,
        nemoclaw_adoption=nemoclaw_adoption,
    )
    external_action_checklist = bundle_builder.build_external_action_checklist(
        operator_steps=operator_next_steps,
        blocking_gates=blocking_gates if isinstance(blocking_gates, list) else [],
        paid_run_review_package=paid_run_review_package,
    )
    external_budget = bundle_builder.external_budget_summary(
        external_action_checklist
    )
    readiness_ok = bool(report.get("ok"))
    bundle_integrity_ok = bool(verification.get("integrity_ok"))
    release_ready = readiness_ok and bundle_integrity_ok
    if not bundle_integrity_ok:
        status = "invalid_bundle"
    elif release_ready:
        status = "ready"
    else:
        status = "not_ready"
    return {
        "schema_version": 1,
        "ok": release_ready,
        "status": status,
        "release_ready": release_ready,
        "readiness_report_schema_version": report.get("schema_version"),
        "readiness_status": report.get("status"),
        "readiness_ok": readiness_ok,
        "gate_count": summary.get("gate_count"),
        "blocker_count": summary.get("blocker_count"),
        "blocking_gates": blocking_gates,
        "blockers": blocking_gates,
        "required_next_actions": required_next_actions,
        "remediation_plan": remediation,
        "benchmark_completion": benchmark_completion,
        "weave_agents_completion": weave_agents_completion,
        "existing_results_formalization": existing_results_formalization,
        "wandb_adoption_draft": wandb_adoption_draft,
        "paid_run_review_package": paid_run_review_package,
        "wandb_completion_contract": wandb_completion_contract,
        "benchmark_progress_matrix": benchmark_progress_matrix,
        "nemoclaw_adoption": nemoclaw_adoption,
        "operator_next_steps": operator_next_steps,
        "external_action_checklist": external_action_checklist,
        "external_budget": external_budget,
        "external_action_approval_packet": (
            bundle.get("external_action_approval_packet")
            if isinstance(bundle.get("external_action_approval_packet"), dict)
            else {}
        ),
        "bundle_integrity_ok": bundle_integrity_ok,
        "bundle_verification_ok": bool(verification.get("ok")),
        "bundle_file_count": bundle.get("file_count"),
        "bundle_missing_file_count": bundle.get("missing_file_count"),
        "checked_file_count": verification.get("checked_file_count"),
        "verification_error_count": len(verification.get("errors") or []),
        "require_ready": require_ready,
        "generated_at": time.time(),
        "readiness_report_source": str(report_path),
        "readiness_report": str(report_path),
        "bundle": bundle,
        "bundle_verification": verification,
        "forwarded_readiness_args": forwarded_args,
    }


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def read_existing_latest_pointer_timestamp(path: Path) -> str | None:
    if not path.exists():
        return None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    if not isinstance(payload, dict):
        return None
    timestamp = payload.get("timestamp")
    if not isinstance(timestamp, str) or not timestamp.strip():
        return None
    return timestamp.strip()


def is_formal_release_timestamp(value: Any) -> bool:
    return isinstance(value, str) and bool(
        pointer_verifier.GATE_NAME_RE.match(f"taiwan_release_gate_{value}.json")
    )


def should_skip_latest_pointer_regression(
    *,
    latest_pointer_json: Path,
    new_timestamp: Any,
) -> tuple[bool, str | None]:
    existing_timestamp = read_existing_latest_pointer_timestamp(latest_pointer_json)
    if (
        existing_timestamp
        and is_formal_release_timestamp(existing_timestamp)
        and is_formal_release_timestamp(new_timestamp)
        and str(new_timestamp) < existing_timestamp
    ):
        return True, existing_timestamp
    return False, existing_timestamp


def build_latest_pointer(result: dict[str, Any]) -> dict[str, Any]:
    verification = result.get("bundle_verification")
    if not isinstance(verification, dict):
        verification = {}
    bundle = result.get("bundle")
    if not isinstance(bundle, dict):
        bundle = {}
    pointer = {
        "schema_version": 1,
        "generated_at": time.time(),
        "timestamp": result.get("timestamp"),
        "status": result.get("status"),
        "ok": bool(result.get("ok")),
        "release_ready": bool(result.get("release_ready")),
        "readiness_report_schema_version": result.get("readiness_report_schema_version"),
        "readiness_report_source": result.get("readiness_report_source")
        or result.get("readiness_report"),
        "readiness_status": result.get("readiness_status"),
        "readiness_ok": bool(result.get("readiness_ok")),
        "gate_count": result.get("gate_count"),
        "blocker_count": result.get("blocker_count"),
        "blocking_gates": result.get("blocking_gates") or [],
        "blockers": result.get("blockers") or result.get("blocking_gates") or [],
        "required_next_actions": result.get("required_next_actions") or [],
        "benchmark_completion": result.get("benchmark_completion") or [],
        "weave_agents_completion": result.get("weave_agents_completion") or [],
        "existing_results_formalization": result.get("existing_results_formalization") or {},
        "wandb_adoption_draft": result.get("wandb_adoption_draft") or {},
        "paid_run_review_package": result.get("paid_run_review_package") or {},
        "wandb_completion_contract": result.get("wandb_completion_contract") or {},
        "benchmark_progress_matrix": result.get("benchmark_progress_matrix") or [],
        "nemoclaw_adoption": result.get("nemoclaw_adoption") or {},
        "operator_next_steps": result.get("operator_next_steps") or {},
        "operator_plan": result.get("operator_plan") or {},
        "external_action_checklist": result.get("external_action_checklist") or {},
        "external_budget": result.get("external_budget") or {},
        "external_action_approval_packet": result.get("external_action_approval_packet") or {},
        "release_gate_json": result.get("release_gate_json"),
        "latest_pointer_json": result.get("latest_pointer_json"),
        "latest_pointer_verification_json": result.get(
            "latest_pointer_verification_json"
        ),
        "latest_pointer_verification_ok": result.get(
            "latest_pointer_verification_ok"
        ),
        "latest_pointer_verification_status": result.get(
            "latest_pointer_verification_status"
        ),
        "latest_pointer_verification_issue_count": result.get(
            "latest_pointer_verification_issue_count"
        ),
        "latest_pointer_verification": result.get("latest_pointer_verification")
        or {},
        "readiness_report": result.get("readiness_report"),
        "bundle_manifest": bundle.get("manifest"),
        "bundle_output_dir": bundle.get("output_dir"),
        "bundle_verification_json": verification.get("json"),
        "bundle_integrity_ok": bool(result.get("bundle_integrity_ok")),
        "bundle_file_count": result.get("bundle_file_count"),
        "bundle_missing_file_count": result.get("bundle_missing_file_count"),
        "checked_file_count": verification.get("checked_file_count"),
        "verification_error_count": len(verification.get("errors") or []),
    }
    return pointer


def release_gate_pointer_summary(result: dict[str, Any]) -> dict[str, Any]:
    return {
        "release_gate_json": result.get("release_gate_json"),
        "latest_pointer_json": result.get("latest_pointer_json"),
        "latest_pointer_verification_json": result.get(
            "latest_pointer_verification_json"
        ),
        "latest_pointer_verification_ok": result.get(
            "latest_pointer_verification_ok"
        ),
        "latest_pointer_verification_status": result.get(
            "latest_pointer_verification_status"
        ),
        "latest_pointer_verification_issue_count": result.get(
            "latest_pointer_verification_issue_count"
        ),
        "operator_plan": result.get("operator_plan") or {},
        "external_budget": result.get("external_budget") or {},
    }


def build_release_gate_pointer_proof(result: dict[str, Any]) -> dict[str, Any]:
    latest_pointer_verification = result.get("latest_pointer_verification")
    if not isinstance(latest_pointer_verification, dict):
        latest_pointer_verification = {}
    proof_ok = (
        latest_pointer_verification.get("ok") is True
        and latest_pointer_verification.get("status") == "passed"
        and latest_pointer_verification.get("issue_count") == 0
    )
    return {
        "schema_version": 1,
        "kind": "release_gate_pointer_proof",
        "generated_at": time.time(),
        "ok": proof_ok,
        "status": "passed" if proof_ok else "failed",
        "timestamp": result.get("timestamp"),
        "release_gate_status": result.get("status"),
        "release_ready": bool(result.get("release_ready")),
        "readiness_status": result.get("readiness_status"),
        "readiness_ok": bool(result.get("readiness_ok")),
        "blocking_gates": result.get("blocking_gates") or [],
        "release_gate_pointer": release_gate_pointer_summary(result),
        "latest_pointer_verification": latest_pointer_verification,
    }


def upsert_generated_bundle_record(
    manifest: dict[str, Any],
    path: Path,
    *,
    bundle_path: Path,
    roles: list[str],
) -> None:
    files = manifest.get("files")
    if not isinstance(files, list):
        files = []
        manifest["files"] = files
    record = bundle_builder.generated_bundle_file_record(
        path,
        bundle_path=bundle_path,
        roles=roles,
    )
    for index, existing in enumerate(files):
        if isinstance(existing, dict) and existing.get("bundle_path") == str(bundle_path):
            files[index] = record
            return
    files.append(record)


def upsert_copied_bundle_record(
    manifest: dict[str, Any],
    bundle_dir: Path,
    *,
    path_value: Any,
    roles: list[str],
) -> None:
    if not isinstance(path_value, str) or not path_value.strip():
        return
    source_path = repo_path(path_value)
    bundle_path = bundle_builder.safe_bundle_path(source_path)
    destination = bundle_dir / bundle_path
    if source_path.exists() and source_path.is_file():
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source_path, destination)
    upsert_generated_bundle_record(
        manifest,
        source_path,
        bundle_path=bundle_path,
        roles=roles,
    )


def refresh_bundle_release_gate_pointer(result: dict[str, Any]) -> None:
    bundle = result.get("bundle")
    if not isinstance(bundle, dict):
        return
    manifest_value = bundle.get("manifest")
    if not isinstance(manifest_value, str) or not manifest_value:
        return
    manifest_path = repo_path(manifest_value)
    if not manifest_path.exists():
        return

    manifest = bundle_builder.read_json(manifest_path)
    manifest["timestamp"] = result.get("timestamp")
    manifest["release_gate_pointer"] = release_gate_pointer_summary(result)
    bundle_dir = manifest_path.parent
    release_gate_pointer = (
        manifest.get("release_gate_pointer")
        if isinstance(manifest.get("release_gate_pointer"), dict)
        else {}
    )
    upsert_copied_bundle_record(
        manifest,
        bundle_dir,
        path_value=release_gate_pointer.get("release_gate_json"),
        roles=[
            "release_gate_pointer",
            "release_gate_pointer:release_gate_json",
        ],
    )
    upsert_copied_bundle_record(
        manifest,
        bundle_dir,
        path_value=release_gate_pointer.get("latest_pointer_json"),
        roles=[
            "release_gate_pointer",
            "release_gate_pointer:latest_pointer_json",
        ],
    )
    upsert_copied_bundle_record(
        manifest,
        bundle_dir,
        path_value=release_gate_pointer.get("latest_pointer_verification_json"),
        roles=[
            "release_gate_pointer",
            "release_gate_pointer:latest_pointer_verification_json",
        ],
    )
    proof_bundle_path = Path("release_gate_pointer_proof.json")
    proof_path = bundle_dir / proof_bundle_path
    proof = build_release_gate_pointer_proof(result)
    write_json(proof_path, proof)
    manifest["release_gate_pointer_proof"] = {
        "json": str(proof_bundle_path),
        "schema_version": proof["schema_version"],
        "status": proof["status"],
        "ok": proof["ok"],
    }
    upsert_generated_bundle_record(
        manifest,
        proof_path,
        bundle_path=proof_bundle_path,
        roles=[
            "release_gate_pointer",
            "release_gate_pointer:proof_json",
        ],
    )

    operator_plan_ref = (
        manifest.get("operator_plan")
        if isinstance(manifest.get("operator_plan"), dict)
        else {}
    )
    operator_plan_json = Path(operator_plan_ref.get("json") or "operator_plan.json")
    operator_plan_md = Path(operator_plan_ref.get("markdown") or "operator_plan.md")
    operator_plan = bundle_builder.build_operator_plan(
        manifest,
        json_bundle_path=operator_plan_json,
        markdown_bundle_path=operator_plan_md,
    )
    write_json(bundle_dir / operator_plan_json, operator_plan)
    write_text(
        bundle_dir / operator_plan_md,
        bundle_builder.operator_plan_markdown(operator_plan),
    )
    manifest["operator_plan"] = {
        "json": str(operator_plan_json),
        "markdown": str(operator_plan_md),
        "schema_version": operator_plan["schema_version"],
        "status": operator_plan["status"],
    }
    upsert_generated_bundle_record(
        manifest,
        bundle_dir / operator_plan_json,
        bundle_path=operator_plan_json,
        roles=["release_operator_plan", "release_operator_plan_json"],
    )
    upsert_generated_bundle_record(
        manifest,
        bundle_dir / operator_plan_md,
        bundle_path=operator_plan_md,
        roles=["release_operator_plan", "release_operator_plan_markdown"],
    )

    approval_ref = (
        manifest.get("external_action_approval_packet")
        if isinstance(manifest.get("external_action_approval_packet"), dict)
        else {}
    )
    approval_json = Path(approval_ref.get("json") or "external_action_approval_packet.json")
    approval_md = Path(approval_ref.get("markdown") or "external_action_approval_packet.md")
    approval_packet = bundle_builder.build_external_action_approval_packet(
        manifest,
        json_bundle_path=approval_json,
        markdown_bundle_path=approval_md,
        bundle_dir_template=path_display(bundle_dir),
    )
    write_json(bundle_dir / approval_json, approval_packet)
    write_text(
        bundle_dir / approval_md,
        bundle_builder.external_action_approval_packet_markdown(approval_packet),
    )
    manifest["external_action_approval_packet"] = {
        "json": str(approval_json),
        "markdown": str(approval_md),
        "schema_version": approval_packet["schema_version"],
        "status": approval_packet["status"],
        "external_action_checklist_sha256": approval_packet["external_action_checklist_sha256"],
        "required_approval_count": approval_packet["required_approval_count"],
        "all_required_approvals_granted": approval_packet["all_required_approvals_granted"],
        "approval_verifier": approval_packet["approval_verifier"],
        "approval_template_renderer": approval_packet["approval_template_renderer"],
        "approval_handoff_preparer": approval_packet["approval_handoff_preparer"],
    }
    upsert_generated_bundle_record(
        manifest,
        bundle_dir / approval_json,
        bundle_path=approval_json,
        roles=[
            "external_action_approval_packet",
            "external_action_approval_packet_json",
        ],
    )
    upsert_generated_bundle_record(
        manifest,
        bundle_dir / approval_md,
        bundle_path=approval_md,
        roles=[
            "external_action_approval_packet",
            "external_action_approval_packet_markdown",
        ],
    )
    approval_verifier_script = Path(
        bundle_builder.EXTERNAL_ACTION_APPROVAL_PACKET_VERIFIER_SCRIPT
    )
    upsert_generated_bundle_record(
        manifest,
        repo_path(approval_verifier_script),
        bundle_path=Path("evidence") / approval_verifier_script,
        roles=[
            "external_action_approval_packet",
            "external_action_approval_packet_verifier",
            "external_action_approval_packet_verifier:script",
        ],
    )
    approval_template_renderer_script = Path(
        bundle_builder.EXTERNAL_ACTION_APPROVAL_TEMPLATE_RENDERER_SCRIPT
    )
    approval_template_renderer_destination = (
        bundle_dir / "evidence" / approval_template_renderer_script
    )
    approval_template_renderer_source = repo_path(approval_template_renderer_script)
    if approval_template_renderer_source.exists() and approval_template_renderer_source.is_file():
        approval_template_renderer_destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(approval_template_renderer_source, approval_template_renderer_destination)
    upsert_generated_bundle_record(
        manifest,
        approval_template_renderer_source,
        bundle_path=Path("evidence") / approval_template_renderer_script,
        roles=[
            "external_action_approval_packet",
            "external_action_approval_template_renderer",
            "external_action_approval_template_renderer:script",
        ],
    )

    summary_path = bundle_dir / "summary.md"
    write_text(summary_path, bundle_builder.summary_markdown(manifest))
    upsert_generated_bundle_record(
        manifest,
        summary_path,
        bundle_path=Path("summary.md"),
        roles=["release_summary", "release_summary_markdown"],
    )
    write_json(manifest_path, manifest)


def should_verify_latest_pointer(args: argparse.Namespace) -> bool:
    if args.no_latest_pointer or args.no_latest_pointer_verification:
        return False
    if args.latest_pointer_verification_json:
        return True
    return pointer_verifier.GATE_NAME_RE.match(repo_path(args.release_gate_json).name) is not None


def write_latest_pointer_evidence(
    args: argparse.Namespace,
    result: dict[str, Any],
) -> None:
    if args.no_latest_pointer:
        return
    latest_pointer_json = repo_path(
        args.latest_pointer_json
        or args.output_dir / "latest_taiwan_release_gate.json"
    )
    result["latest_pointer_json"] = str(latest_pointer_json)
    skip_regression, existing_timestamp = should_skip_latest_pointer_regression(
        latest_pointer_json=latest_pointer_json,
        new_timestamp=result.get("timestamp"),
    )
    result["latest_pointer_update"] = {
        "status": "skipped_older_timestamp" if skip_regression else "will_update",
        "existing_timestamp": existing_timestamp,
        "new_timestamp": result.get("timestamp"),
        "latest_pointer_json": str(latest_pointer_json),
    }
    if skip_regression:
        result["latest_pointer_update"]["reason"] = (
            "existing latest pointer has a newer formal release timestamp"
        )
        if args.release_gate_json:
            write_json(repo_path(args.release_gate_json), result)
        return
    latest_pointer_verification_json: Path | None = None
    if should_verify_latest_pointer(args):
        latest_pointer_verification_json = repo_path(
            args.latest_pointer_verification_json
            or args.output_dir / f"latest_taiwan_release_gate_verify_{args.timestamp}.json"
        )
        result["latest_pointer_verification_json"] = str(
            latest_pointer_verification_json
        )
        result["latest_pointer_verification_ok"] = True
        result["latest_pointer_verification_status"] = "passed"
        result["latest_pointer_verification_issue_count"] = 0
        result["latest_pointer_verification"] = {
            "json": str(latest_pointer_verification_json),
            "ok": True,
            "status": "passed",
            "issue_count": 0,
        }
    write_json(latest_pointer_json, build_latest_pointer(result))
    if args.release_gate_json:
        write_json(repo_path(args.release_gate_json), result)
    if latest_pointer_verification_json is not None:
        provisional_verification = {
            "schema_version": 1,
            "ok": True,
            "status": "passed",
            "generated_at": time.time(),
            "pointer_json": path_display(latest_pointer_json),
            "release_gate_json": (
                path_display(repo_path(args.release_gate_json))
                if args.release_gate_json
                else None
            ),
            "bundle_manifest": path_display(
                repo_path((result.get("bundle") or {}).get("manifest", ""))
            )
            if isinstance(result.get("bundle"), dict)
            and (result.get("bundle") or {}).get("manifest")
            else None,
            "bundle_verification_json": path_display(
                repo_path((result.get("bundle_verification") or {}).get("json", ""))
            )
            if isinstance(result.get("bundle_verification"), dict)
            and (result.get("bundle_verification") or {}).get("json")
            else None,
            "issue_count": 0,
            "issues": [],
            "pointer": build_latest_pointer(result),
        }
        pointer_verifier.write_json(
            latest_pointer_verification_json,
            provisional_verification,
        )
        refresh_bundle_release_gate_pointer(result)
        latest_pointer_verification = pointer_verifier.validate_pointer(
            latest_pointer_json
        )
        pointer_verifier.write_json(
            latest_pointer_verification_json,
            latest_pointer_verification,
        )
        result["latest_pointer_verification_json"] = str(
            latest_pointer_verification_json
        )
        result["latest_pointer_verification_ok"] = bool(
            latest_pointer_verification.get("ok")
        )
        result["latest_pointer_verification_status"] = (
            latest_pointer_verification.get("status")
        )
        result["latest_pointer_verification_issue_count"] = (
            latest_pointer_verification.get("issue_count")
        )
        result["latest_pointer_verification"] = {
            "json": str(latest_pointer_verification_json),
            "ok": result["latest_pointer_verification_ok"],
            "status": result["latest_pointer_verification_status"],
            "issue_count": result["latest_pointer_verification_issue_count"],
        }
        if args.release_gate_json:
            write_json(repo_path(args.release_gate_json), result)
        write_json(latest_pointer_json, build_latest_pointer(result))
        refresh_bundle_release_gate_pointer(result)
        latest_pointer_verification = pointer_verifier.validate_pointer(
            latest_pointer_json
        )
        pointer_verifier.write_json(
            latest_pointer_verification_json,
            latest_pointer_verification,
        )
        result["latest_pointer_verification_ok"] = bool(
            latest_pointer_verification.get("ok")
        )
        result["latest_pointer_verification_status"] = (
            latest_pointer_verification.get("status")
        )
        result["latest_pointer_verification_issue_count"] = (
            latest_pointer_verification.get("issue_count")
        )
        result["latest_pointer_verification"] = {
            "json": str(latest_pointer_verification_json),
            "ok": result["latest_pointer_verification_ok"],
            "status": result["latest_pointer_verification_status"],
            "issue_count": result["latest_pointer_verification_issue_count"],
        }
        if args.release_gate_json:
            write_json(repo_path(args.release_gate_json), result)
        write_json(latest_pointer_json, build_latest_pointer(result))
        refresh_bundle_release_gate_pointer(result)


def refresh_bundle_counts(result: dict[str, Any]) -> None:
    bundle = result.get("bundle")
    if not isinstance(bundle, dict):
        return
    manifest_value = bundle.get("manifest")
    if not isinstance(manifest_value, str) or not manifest_value:
        return
    manifest_path = repo_path(manifest_value)
    if not manifest_path.exists():
        return
    manifest = bundle_builder.read_json(manifest_path)
    files = manifest.get("files")
    if isinstance(files, list):
        bundle["file_count"] = len(files)
        bundle["missing_file_count"] = sum(
            1
            for record in files
            if isinstance(record, dict) and not record.get("exists")
        )
        result["bundle_file_count"] = bundle["file_count"]
        result["bundle_missing_file_count"] = bundle["missing_file_count"]


def refresh_bundle_generated_summaries(result: dict[str, Any]) -> None:
    bundle = result.get("bundle")
    if not isinstance(bundle, dict):
        return
    manifest_value = bundle.get("manifest")
    if not isinstance(manifest_value, str) or not manifest_value:
        return
    manifest_path = repo_path(manifest_value)
    if not manifest_path.exists():
        return
    manifest = bundle_builder.read_json(manifest_path)
    for field in ("operator_plan", "external_action_approval_packet"):
        value = manifest.get(field)
        if isinstance(value, dict):
            bundle[field] = value
            if not isinstance(result.get(field), dict) or not result.get(field):
                result[field] = value


def apply_bundle_verification_result(
    result: dict[str, Any],
    verification: dict[str, Any],
) -> None:
    result["bundle_verification"] = verification
    result["bundle_integrity_ok"] = bool(verification.get("integrity_ok"))
    result["bundle_verification_ok"] = bool(verification.get("ok"))
    result["checked_file_count"] = verification.get("checked_file_count")
    result["verification_error_count"] = len(verification.get("errors") or [])
    refresh_bundle_counts(result)
    refresh_bundle_generated_summaries(result)


def parse_args(argv: list[str] | None = None) -> tuple[argparse.Namespace, list[str]]:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Unknown arguments are forwarded to "
            "run_taiwan_production_readiness_gate.py / "
            "build_taiwan_production_readiness_report.py. Example:\n"
            "  --no-require-nemoclaw --required-wandb-benchmark agentic_math"
        ),
    )
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--timestamp", default=utc_timestamp())
    parser.add_argument("--readiness-report-json", type=Path)
    parser.add_argument("--bundle-output-dir", type=Path)
    parser.add_argument("--bundle-verification-json", type=Path)
    parser.add_argument(
        "--release-gate-json",
        type=Path,
        help=(
            "Write the release gate JSON. Defaults to "
            "OUTPUT_DIR/taiwan_release_gate_TIMESTAMP.json."
        ),
    )
    parser.add_argument("--operator-plan-json", type=Path)
    parser.add_argument("--operator-plan-markdown", type=Path)
    parser.add_argument(
        "--latest-pointer-json",
        type=Path,
        help=(
            "Write a stable pointer to the latest formal timestamped release gate. "
            "Defaults to OUTPUT_DIR/latest_taiwan_release_gate.json."
        ),
    )
    parser.add_argument(
        "--no-latest-pointer",
        action="store_true",
        help="Do not write the stable latest release-gate pointer JSON.",
    )
    parser.add_argument(
        "--latest-pointer-verification-json",
        type=Path,
        help=(
            "Write pointer verification evidence. Defaults to "
            "OUTPUT_DIR/latest_taiwan_release_gate_verify_TIMESTAMP.json for formal "
            "taiwan_release_gate_YYYYMMDDTHHMMSSZ.json release gates."
        ),
    )
    parser.add_argument(
        "--no-latest-pointer-verification",
        action="store_true",
        help="Do not verify the latest release-gate pointer from this command.",
    )
    parser.add_argument(
        "--no-operator-plan",
        action="store_true",
        help="Do not write standalone operator-plan JSON/Markdown evidence.",
    )
    parser.add_argument(
        "--require-ready",
        action="store_true",
        help="Require readiness_ok=true in the bundle verifier and exit nonzero if not ready.",
    )
    parser.add_argument(
        "--fail-on-not-ready",
        action="store_true",
        help="Alias for --require-ready, kept for CI/readability.",
    )
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument(
        "--no-summary",
        action="store_true",
        help="Do not print the human-readable release summary to stderr.",
    )
    return parser.parse_known_args(argv)


def main(argv: list[str] | None = None) -> None:
    args, forwarded_args = parse_args(argv)
    if args.release_gate_json is None:
        args.release_gate_json = args.output_dir / f"taiwan_release_gate_{args.timestamp}.json"
    require_ready = bool(args.require_ready or args.fail_on_not_ready)
    report_path, report = run_readiness_gate(args, forwarded_args)
    bundle_output_dir = repo_path(
        args.bundle_output_dir
        or DEFAULT_BUNDLE_OUTPUT_ROOT / f"bundle_{args.timestamp}"
    )
    verification_json = repo_path(
        args.bundle_verification_json
        or args.output_dir / f"taiwan_release_evidence_bundle_verify_{args.timestamp}.json"
    )

    bundle = build_bundle(
        report_path=report_path,
        output_dir=bundle_output_dir,
    )
    verification = verify_bundle(
        manifest_path=repo_path(bundle["manifest"]),
        verification_json=verification_json,
        require_ready=require_ready,
    )
    result = build_result(
        report_path=report_path,
        report=report,
        bundle=bundle,
        verification=verification,
        require_ready=require_ready,
        forwarded_args=forwarded_args,
    )
    result["timestamp"] = args.timestamp
    if args.release_gate_json:
        result["release_gate_json"] = str(repo_path(args.release_gate_json))
    if args.release_gate_json:
        write_json(repo_path(args.release_gate_json), result)
    write_latest_pointer_evidence(args, result)
    if result.get("latest_pointer_verification_json"):
        refresh_bundle_release_gate_pointer(result)
        refresh_bundle_generated_summaries(result)
        verification = verify_bundle(
            manifest_path=repo_path(result["bundle"]["manifest"]),
            verification_json=verification_json,
            require_ready=require_ready,
        )
        apply_bundle_verification_result(result, verification)
        if args.release_gate_json:
            write_json(repo_path(args.release_gate_json), result)
        write_latest_pointer_evidence(args, result)
    if not args.no_operator_plan:
        operator_plan_json = repo_path(
            args.operator_plan_json
            or args.output_dir / f"taiwan_release_operator_plan_{args.timestamp}.json"
        )
        operator_plan_md = repo_path(
            args.operator_plan_markdown
            or args.output_dir / f"taiwan_release_operator_plan_{args.timestamp}.md"
        )
        operator_plan = build_operator_plan(
            result,
            json_path=operator_plan_json,
            markdown_path=operator_plan_md,
        )
        write_json(operator_plan_json, operator_plan)
        write_text(operator_plan_md, operator_plan_markdown(operator_plan))
        result["operator_plan"] = {
            "json": str(operator_plan_json),
            "markdown": str(operator_plan_md),
            "schema_version": operator_plan["schema_version"],
            "status": operator_plan["status"],
        }
        if args.release_gate_json:
            write_json(repo_path(args.release_gate_json), result)
        write_latest_pointer_evidence(args, result)
        if result.get("latest_pointer_verification_json"):
            refresh_bundle_release_gate_pointer(result)
            refresh_bundle_generated_summaries(result)
            verification = verify_bundle(
                manifest_path=repo_path(result["bundle"]["manifest"]),
                verification_json=verification_json,
                require_ready=require_ready,
            )
            apply_bundle_verification_result(result, verification)
            if args.release_gate_json:
                write_json(repo_path(args.release_gate_json), result)
            write_latest_pointer_evidence(args, result)
    if result.get("latest_pointer_verification_json"):
        refresh_bundle_release_gate_pointer(result)
        refresh_bundle_generated_summaries(result)
        verification = verify_bundle(
            manifest_path=repo_path(result["bundle"]["manifest"]),
            verification_json=verification_json,
            require_ready=require_ready,
        )
        apply_bundle_verification_result(result, verification)
        if args.release_gate_json:
            write_json(repo_path(args.release_gate_json), result)
        write_latest_pointer_evidence(args, result)
        refresh_bundle_release_gate_pointer(result)
        refresh_bundle_generated_summaries(result)
        verification = verify_bundle(
            manifest_path=repo_path(result["bundle"]["manifest"]),
            verification_json=verification_json,
            require_ready=require_ready,
        )
        apply_bundle_verification_result(result, verification)
        if args.release_gate_json:
            write_json(repo_path(args.release_gate_json), result)
        write_latest_pointer_evidence(args, result)
    if not args.quiet:
        if not args.no_summary:
            print(format_summary(result), file=sys.stderr)
        print(json.dumps(result, ensure_ascii=False, indent=2))
    latest_pointer_verification = result.get("latest_pointer_verification")
    if (
        isinstance(latest_pointer_verification, dict)
        and not latest_pointer_verification.get("ok")
    ):
        raise SystemExit(1)
    if require_ready and not result["release_ready"]:
        raise SystemExit(1)
    if not result["bundle_integrity_ok"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main(sys.argv[1:])
