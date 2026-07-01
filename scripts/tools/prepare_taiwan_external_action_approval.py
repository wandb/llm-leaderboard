#!/usr/bin/env python3
"""Prepare an offline external-action approval handoff for Taiwan release gates.

This tool is intentionally non-executing. It renders a reviewer-fillable
external_action_approval_packet.json copy, runs the offline verifier against
that copy, and writes a handoff summary. It does not query W&B, write W&B,
install NeMoClaw, or launch model/provider calls.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import time
from pathlib import Path
from typing import Any

from render_external_action_approval_template import render_template
from verify_external_action_approval_packet import (
    read_json_object,
    repo_path,
    sha256_file,
    verify_approval_packet,
    write_json,
)


DEFAULT_LATEST_POINTER_JSON = Path("temp/latest_taiwan_release_gate.json")
DEFAULT_OUTPUT_DIR = Path("temp")
SAFETY = {
    "executes_external_action": False,
    "queries_wandb": False,
    "writes_wandb": False,
    "installs_third_party": False,
    "launches_model_inference": False,
}


def safe_token(value: str) -> str:
    token = re.sub(r"[^A-Za-z0-9_.-]+", "_", value.strip())
    return token or time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())


def resolve_bundle_dir(
    *,
    bundle_dir: Path | None,
    latest_pointer_json: Path,
) -> tuple[Path, dict[str, Any]]:
    if bundle_dir is not None:
        return repo_path(bundle_dir), {}
    latest = read_json_object(repo_path(latest_pointer_json))
    bundle_value = latest.get("bundle_output_dir")
    if not isinstance(bundle_value, str) or not bundle_value.strip():
        raise ValueError(
            f"{latest_pointer_json} does not contain a concrete bundle_output_dir"
        )
    return repo_path(bundle_value), latest


def default_output_paths(
    *,
    output_dir: Path,
    timestamp: str,
) -> dict[str, Path]:
    stem = f"taiwan_external_action_approval_REVIEWED_{safe_token(timestamp)}"
    return {
        "reviewed_json": output_dir / f"{stem}.json",
        "reviewed_markdown": output_dir / f"{stem}.md",
        "render_report_json": output_dir / f"{stem}.render.json",
        "verify_report_json": output_dir / f"{stem}.verify.json",
        "handoff_json": output_dir / f"{stem}.handoff.json",
        "handoff_markdown": output_dir / f"{stem}.handoff.md",
    }


def count_granted_requirements(report: dict[str, Any]) -> int:
    results = report.get("approval_results")
    if not isinstance(results, list):
        return 0
    return sum(
        1
        for item in results
        if isinstance(item, dict)
        and item.get("required") is True
        and item.get("approved") is True
    )


def approval_requirement_summary(requirements: Any) -> list[dict[str, Any]]:
    if not isinstance(requirements, list):
        return []
    rows: list[dict[str, Any]] = []
    for item in requirements:
        if not isinstance(item, dict):
            continue
        rows.append(
            {
                "requirement": item.get("requirement"),
                "label": item.get("label"),
                "required": item.get("required"),
                "approval_status": item.get("approval_status"),
                "minimum_approved_budget_usd": item.get(
                    "minimum_approved_budget_usd"
                ),
                "minimum_approved_budget_source": item.get(
                    "minimum_approved_budget_source"
                ),
                "reviewer_fields": (
                    item.get("reviewer_fields")
                    if isinstance(item.get("reviewer_fields"), list)
                    else []
                ),
            }
        )
    return rows


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


def handoff_markdown(handoff: dict[str, Any]) -> str:
    verifier = handoff.get("verifier") if isinstance(handoff.get("verifier"), dict) else {}
    paths = handoff.get("paths") if isinstance(handoff.get("paths"), dict) else {}
    source = handoff.get("source") if isinstance(handoff.get("source"), dict) else {}
    safety = handoff.get("safety") if isinstance(handoff.get("safety"), dict) else {}
    verify_command = verifier.get("command")
    lines = [
        "# Taiwan External Action Approval Handoff",
        "",
        f"Status: `{handoff.get('status')}`",
        f"OK: `{handoff.get('ok')}`",
        f"Source packet: `{source.get('approval_packet_json')}`",
        f"Source packet SHA-256: `{source.get('approval_packet_sha256')}`",
        f"Required approvals: `{handoff.get('required_approval_count')}`",
        f"Granted approvals: `{handoff.get('granted_approval_count')}`",
        f"All required approvals granted: `{handoff.get('all_required_approvals_granted')}`",
        f"Will execute external actions: `{handoff.get('will_execute_external_actions')}`",
        "",
        "## Files",
        "",
        "| File | Path |",
        "|---|---|",
    ]
    for key in (
        "reviewed_json",
        "reviewed_markdown",
        "render_report_json",
        "verify_report_json",
        "handoff_json",
        "handoff_markdown",
    ):
        if paths.get(key):
            lines.append(f"| {key} | `{paths[key]}` |")
    lines.extend(
        [
            "",
            "## Approval Requirements",
            "",
            "| Requirement | Required | Approval status | Minimum approved budget USD | Reviewer fields |",
            "|---|---:|---|---:|---|",
        ]
    )
    requirements = handoff.get("approval_requirements")
    if isinstance(requirements, list) and requirements:
        for item in requirements:
            if not isinstance(item, dict):
                continue
            lines.append(
                "| "
                f"{md_cell(item.get('label') or item.get('requirement'))} | "
                f"{md_cell(item.get('required'))} | "
                f"{md_cell(item.get('approval_status'))} | "
                f"{md_cell(item.get('minimum_approved_budget_usd'))} | "
                f"{md_cell(item.get('reviewer_fields'))} |"
            )
    else:
        lines.append("| none | False | not_required |  |  |")
    lines.extend(
        [
            "",
            "## Verifier",
            "",
            f"Verifier status: `{verifier.get('status')}`",
            f"Verifier ok: `{verifier.get('ok')}`",
            f"Source binding ok: `{verifier.get('source_binding_bound')}`",
            "",
            "```bash",
            str(verify_command or ""),
            "```",
            "",
            "## Required Human Actions",
            "",
            "1. Review and edit the generated reviewed JSON file.",
            "2. Set each required approval row to `granted` only after approval.",
            "3. Re-run the verifier command above.",
            "4. Use the verifier report only if it returns `ok=true` and `status=approved`.",
            "",
            "## Safety",
            "",
            "| Field | Value |",
            "|---|---|",
        ]
    )
    for key, value in safety.items():
        lines.append(f"| {key} | `{value}` |")
    lines.append("")
    return "\n".join(lines)


def prepare_handoff(
    *,
    bundle_dir: Path | None,
    latest_pointer_json: Path,
    timestamp: str | None,
    output_dir: Path,
    reviewed_json: Path | None,
    reviewed_markdown: Path | None,
    render_report_json: Path | None,
    verify_report_json: Path | None,
    handoff_json: Path | None,
    handoff_markdown_path: Path | None,
) -> dict[str, Any]:
    resolved_bundle_dir, latest = resolve_bundle_dir(
        bundle_dir=bundle_dir,
        latest_pointer_json=latest_pointer_json,
    )
    packet_path = resolved_bundle_dir / "external_action_approval_packet.json"
    if not packet_path.exists():
        raise FileNotFoundError(f"{packet_path} does not exist")
    packet_sha = sha256_file(packet_path)
    resolved_timestamp = (
        timestamp
        or str(latest.get("timestamp") or "")
        or resolved_bundle_dir.name.replace("bundle_", "")
    )
    defaults = default_output_paths(
        output_dir=repo_path(output_dir),
        timestamp=resolved_timestamp,
    )
    paths = {
        "reviewed_json": repo_path(reviewed_json) if reviewed_json else defaults["reviewed_json"],
        "reviewed_markdown": (
            repo_path(reviewed_markdown) if reviewed_markdown else defaults["reviewed_markdown"]
        ),
        "render_report_json": (
            repo_path(render_report_json)
            if render_report_json
            else defaults["render_report_json"]
        ),
        "verify_report_json": (
            repo_path(verify_report_json)
            if verify_report_json
            else defaults["verify_report_json"]
        ),
        "handoff_json": repo_path(handoff_json) if handoff_json else defaults["handoff_json"],
        "handoff_markdown": (
            repo_path(handoff_markdown_path)
            if handoff_markdown_path
            else defaults["handoff_markdown"]
        ),
    }

    render_report = render_template(
        packet_path=packet_path,
        output_json=paths["reviewed_json"],
        markdown_path=paths["reviewed_markdown"],
    )
    write_json(paths["render_report_json"], render_report)
    reviewed_template = read_json_object(paths["reviewed_json"])
    verify_report = verify_approval_packet(
        paths["reviewed_json"],
        source_packet_path=packet_path,
    )
    write_json(paths["verify_report_json"], verify_report)
    source_binding = (
        verify_report.get("source_binding")
        if isinstance(verify_report.get("source_binding"), dict)
        else {}
    )
    source_binding_bound = source_binding.get("bound") is True
    verifier_ok = verify_report.get("ok") is True
    status = (
        "approved"
        if verifier_ok
        else "pending_human_approval"
        if source_binding_bound
        else "invalid_source_binding"
    )
    ok = source_binding_bound and render_report.get("ok") is True
    handoff = {
        "schema_version": 1,
        "ok": ok,
        "status": status,
        "generated_at": time.time(),
        "will_execute_external_actions": False,
        "source": {
            "bundle_dir": str(resolved_bundle_dir),
            "latest_pointer_json": str(repo_path(latest_pointer_json)),
            "latest_pointer_timestamp": latest.get("timestamp"),
            "approval_packet_json": str(packet_path),
            "approval_packet_sha256": packet_sha,
        },
        "paths": {key: str(value) for key, value in paths.items()},
        "render": {
            "ok": render_report.get("ok") is True,
            "status": render_report.get("status"),
            "report_json": str(paths["render_report_json"]),
        },
        "verifier": {
            "ok": verifier_ok,
            "status": verify_report.get("status"),
            "report_json": str(paths["verify_report_json"]),
            "source_binding_bound": source_binding_bound,
            "command": (
                "uv run python scripts/tools/verify_external_action_approval_packet.py "
                f"--approval-packet-json {paths['reviewed_json']} "
                f"--source-packet-json {packet_path} "
                "--require-approved "
                f"--json {paths['verify_report_json']}"
            ),
        },
        "required_approval_count": int(verify_report.get("required_approval_count") or 0),
        "granted_approval_count": count_granted_requirements(verify_report),
        "approval_requirements": approval_requirement_summary(
            reviewed_template.get("approval_requirements")
        ),
        "all_required_approvals_granted": verify_report.get(
            "all_required_approvals_granted"
        )
        is True,
        "safety": dict(SAFETY),
        "next_steps": [
            "Human reviewer fills concrete approval fields in reviewed_json.",
            "Human reviewer reruns verifier command before any external action.",
            "Operator uses the verifier report only when ok=true and status=approved.",
        ],
    }
    write_json(paths["handoff_json"], handoff)
    paths["handoff_markdown"].parent.mkdir(parents=True, exist_ok=True)
    paths["handoff_markdown"].write_text(handoff_markdown(handoff), encoding="utf-8")
    return handoff


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle-dir", type=Path)
    parser.add_argument(
        "--latest-pointer-json",
        type=Path,
        default=DEFAULT_LATEST_POINTER_JSON,
    )
    parser.add_argument("--timestamp")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--reviewed-json", type=Path)
    parser.add_argument("--reviewed-markdown", type=Path)
    parser.add_argument("--render-report-json", type=Path)
    parser.add_argument("--verify-report-json", type=Path)
    parser.add_argument("--handoff-json", type=Path)
    parser.add_argument("--handoff-markdown", type=Path)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    handoff = prepare_handoff(
        bundle_dir=args.bundle_dir,
        latest_pointer_json=args.latest_pointer_json,
        timestamp=args.timestamp,
        output_dir=args.output_dir,
        reviewed_json=args.reviewed_json,
        reviewed_markdown=args.reviewed_markdown,
        render_report_json=args.render_report_json,
        verify_report_json=args.verify_report_json,
        handoff_json=args.handoff_json,
        handoff_markdown_path=args.handoff_markdown,
    )
    print(json.dumps(handoff, ensure_ascii=False, indent=2))
    if handoff.get("ok") is not True:
        raise SystemExit(1)


if __name__ == "__main__":
    main(sys.argv[1:])
