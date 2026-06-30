#!/usr/bin/env python3
"""Render a reviewer-fillable external-action approval packet copy.

This tool is offline. It does not query W&B, write W&B, install NeMoClaw, or
launch model/provider calls. It copies an external_action_approval_packet.json
into a reviewer-editable JSON template and keeps every required approval
ungranted until a human fills concrete fields and runs the approval verifier.
"""

from __future__ import annotations

import argparse
import copy
import json
import sys
import time
from pathlib import Path
from typing import Any

from verify_external_action_approval_packet import (
    EXTERNAL_ACTION_REQUIREMENTS,
    SCHEMA_VERSION,
    canonical_json_sha256,
    expected_requirement_base,
    read_json_object,
    repo_path,
    sha256_file,
    write_json,
)


COMMON_PLACEHOLDERS = {
    "approved_by": "APPROVER_NAME",
    "approved_at": "YYYY-MM-DDTHH:MM:SS+09:00",
    "approval_reference": "APPROVAL_REFERENCE",
}
FIELD_PLACEHOLDERS = {
    "approved_budget_usd": "APPROVED_BUDGET_USD",
    "approved_model_scope": "APPROVED_MODEL_SCOPE",
    "approved_wandb_entity": "WANDB_ENTITY",
    "approved_wandb_project": "WANDB_PROJECT",
    "third_party_terms_reviewed": False,
    "installer_lock_json": "scripts/setup/nemoclaw_installer_lock.json",
    "installer_sha256": "REVIEWED_INSTALLER_SHA256",
    "sandbox": "REVIEWED_SANDBOX_NAME",
    "scope_attestation_json": (
        "temp/taiwan_wandb_adoption_attestations_YYYYMMDDTHHMM/"
        "BENCHMARK-RUN_ID.scope_attestation.json"
    ),
}
SAFETY_FLAGS = {
    "executes_external_action": False,
    "queries_wandb": False,
    "writes_wandb": False,
    "installs_third_party": False,
    "launches_model_inference": False,
}


def render_requirement_template(requirement: dict[str, Any]) -> dict[str, Any]:
    item = copy.deepcopy(requirement)
    if item.get("required") is not True:
        item["approval_status"] = "not_required"
        return item
    item["approval_status"] = "not_granted"
    reviewer_fields = (
        item.get("reviewer_fields")
        if isinstance(item.get("reviewer_fields"), list)
        else []
    )
    for field in reviewer_fields:
        if field in COMMON_PLACEHOLDERS:
            item[field] = COMMON_PLACEHOLDERS[field]
        elif field in FIELD_PLACEHOLDERS:
            item[field] = FIELD_PLACEHOLDERS[field]
    return item


def expected_requirements_from_packet(packet: dict[str, Any]) -> list[dict[str, Any]]:
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
    return [
        expected_requirement_base(
            requirement=name,
            label=label,
            count=int(requirement_counts.get(name) or 0),
            checklist=checklist,
        )
        for name, label in EXTERNAL_ACTION_REQUIREMENTS
    ]


def render_template(
    *,
    packet_path: Path,
    output_json: Path,
    markdown_path: Path | None,
) -> dict[str, Any]:
    packet = read_json_object(packet_path)
    checklist = packet.get("external_action_checklist")
    if not isinstance(checklist, dict):
        raise ValueError("external_action_checklist must be an object")
    expected_hash = canonical_json_sha256(checklist)
    if packet.get("external_action_checklist_sha256") != expected_hash:
        raise ValueError("external_action_checklist_sha256 mismatch")
    source_packet_sha256 = sha256_file(packet_path)

    expected_requirements = expected_requirements_from_packet(packet)
    required_count = sum(1 for item in expected_requirements if item.get("required"))
    template = copy.deepcopy(packet)
    template["schema_version"] = SCHEMA_VERSION
    template["status"] = "pending_approval" if required_count else "no_external_action_required"
    template["approval_requirement_count"] = len(expected_requirements)
    template["required_approval_count"] = required_count
    template["all_required_approvals_granted"] = required_count == 0
    template["approval_requirements"] = [
        render_requirement_template(item) for item in expected_requirements
    ]
    template["approval_template"] = {
        "schema_version": 1,
        "generated_at": time.time(),
        "source_approval_packet_json": str(packet_path),
        "source_approval_packet_sha256": source_packet_sha256,
        "source_external_action_checklist_sha256": expected_hash,
        "output_json": str(output_json),
        "output_markdown": str(markdown_path) if markdown_path else "",
        "reviewer_must_set_each_required_approval_status_to_granted": required_count > 0,
        "will_execute_external_actions": False,
        "safety": dict(SAFETY_FLAGS),
    }
    write_json(output_json, template)
    if markdown_path is not None:
        markdown_path.parent.mkdir(parents=True, exist_ok=True)
        markdown_path.write_text(
            approval_template_markdown(
                template,
                packet_path=packet_path,
                output_json=output_json,
            ),
            encoding="utf-8",
        )
    return {
        "schema_version": 1,
        "ok": True,
        "status": "template_rendered",
        "generated_at": time.time(),
        "source_approval_packet_json": str(packet_path),
        "source_approval_packet_sha256": source_packet_sha256,
        "output_json": str(output_json),
        "output_markdown": str(markdown_path) if markdown_path else "",
        "required_approval_count": required_count,
        "all_required_approvals_granted": required_count == 0,
        "will_execute_external_actions": False,
        "safety": dict(SAFETY_FLAGS),
    }


def md_cell(value: Any) -> str:
    if value is None:
        text = ""
    elif isinstance(value, bool):
        text = str(value)
    elif isinstance(value, (list, tuple)):
        text = ", ".join(str(item) for item in value)
    else:
        text = str(value)
    return text.replace("\n", "<br>").replace("|", "\\|")


def approval_template_markdown(
    template: dict[str, Any],
    *,
    packet_path: Path,
    output_json: Path,
) -> str:
    verifier = (
        template.get("approval_verifier")
        if isinstance(template.get("approval_verifier"), dict)
        else {}
    )
    verifier_command = str(verifier.get("command_template") or "")
    if verifier_command:
        verifier_command = verifier_command.replace(
            str(verifier.get("reviewed_packet_json_template") or ""),
            str(output_json),
        )
    lines = [
        "# Taiwan External Action Approval Reviewed Template",
        "",
        f"Source approval packet: `{packet_path}`",
        f"Source approval packet SHA-256: `{(template.get('approval_template') or {}).get('source_approval_packet_sha256')}`",
        f"Reviewed copy JSON: `{output_json}`",
        f"Status: `{template.get('status')}`",
        f"All required approvals granted: `{template.get('all_required_approvals_granted')}`",
        f"Will execute external actions: `{False}`",
        "",
        "## Review Instructions",
        "",
        "1. Fill every placeholder with concrete reviewed values.",
        "2. For each required row, set `approval_status` to `granted` only after approval.",
        "3. Set top-level `status` to `approved` and `all_required_approvals_granted` to `true` only after all required rows are approved.",
        "4. Run the approval verifier before any paid API, W&B write, third-party install, or scope-adoption action.",
        "",
        "## Approval Requirements",
        "",
        "| Requirement | Required | Approval status | Reviewer fields |",
        "|---|---:|---|---|",
    ]
    for item in template.get("approval_requirements") or []:
        if not isinstance(item, dict):
            continue
        lines.append(
            "| "
            f"{md_cell(item.get('label'))} | "
            f"{md_cell(item.get('required'))} | "
            f"{md_cell(item.get('approval_status'))} | "
            f"{md_cell(item.get('reviewer_fields'))} |"
        )
    lines.extend(
        [
            "",
            "## Approval Verifier",
            "",
            "```bash",
            verifier_command,
            "```",
            "",
            "## Safety",
            "",
            "| Field | Value |",
            "|---|---|",
        ]
    )
    for key, value in SAFETY_FLAGS.items():
        lines.append(f"| {key} | {md_cell(value)} |")
    lines.append("")
    return "\n".join(lines)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--approval-packet-json", type=Path, required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--markdown", type=Path)
    parser.add_argument("--report-json", type=Path)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    report = render_template(
        packet_path=repo_path(args.approval_packet_json),
        output_json=repo_path(args.output_json),
        markdown_path=repo_path(args.markdown) if args.markdown else None,
    )
    if args.report_json:
        write_json(repo_path(args.report_json), report)
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main(sys.argv[1:])
