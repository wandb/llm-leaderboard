#!/usr/bin/env python3
"""Check Taiwan paid-run review JSONs before release gating.

This is an offline doctor for paid-run review packages. It does not query W&B,
run model inference, install NeMoClaw, or mutate files unless --json/--markdown
output paths are supplied.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

import build_taiwan_production_readiness_report as readiness


REPO_ROOT = Path(__file__).resolve().parents[2]


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


def discover_review_paths(paths: list[Path] | None) -> list[Path]:
    if paths:
        return [repo_path(path) for path in paths]
    return readiness.discover_paths((readiness.DEFAULT_REVIEW_GLOB,))


def build_report(args: argparse.Namespace) -> dict[str, Any]:
    review_paths = discover_review_paths(args.review_json)
    required_benchmarks = args.required_wandb_benchmark or []
    required_run_ids = readiness.parse_required_wandb_run_ids(
        args.required_wandb_run_id,
        required_benchmarks=required_benchmarks,
        all_run_id=args.required_wandb_run_id_all,
    )
    gates = [
        readiness.evaluate_paid_run_review_package(
            review_paths,
            require=True,
            wandb_completion_max_age_seconds=args.wandb_completion_max_age_seconds,
            weave_agents_completion_max_age_seconds=args.weave_agents_completion_max_age_seconds,
        )
    ]
    if args.require_one_model_canary or required_benchmarks or required_run_ids:
        gates.append(
            readiness.evaluate_one_model_canary(
                review_paths,
                require=True,
                required_run_ids=required_run_ids,
                required_benchmarks=required_benchmarks,
                wandb_completion_max_age_seconds=args.wandb_completion_max_age_seconds,
            )
        )
    blockers = [gate for gate in gates if gate.get("blocking") and not gate.get("ok")]
    review_completion_requirements = readiness.paid_run_review_completion_requirements(
        wandb_completion_max_age_seconds=args.wandb_completion_max_age_seconds,
        weave_agents_completion_max_age_seconds=args.weave_agents_completion_max_age_seconds,
    )
    return {
        "ok": not blockers,
        "status": "passed" if not blockers else "not_ready",
        "generated_at": time.time(),
        "review_paths": [path_display(path) for path in review_paths],
        "review_completion_requirements": review_completion_requirements,
        "requirements": {
            "one_model_canary": bool(args.require_one_model_canary),
            "required_wandb_benchmarks": required_benchmarks,
            "required_wandb_run_ids": required_run_ids,
            "wandb_completion_max_age_seconds": args.wandb_completion_max_age_seconds,
            "weave_agents_completion_max_age_seconds": args.weave_agents_completion_max_age_seconds,
        },
        "summary": {
            "gate_count": len(gates),
            "blocker_count": len(blockers),
            "blockers": [str(gate.get("name")) for gate in blockers],
        },
        "gates": gates,
    }


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


def collect_wandb_completion_entries(gates: list[Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for gate in gates:
        if not isinstance(gate, dict):
            continue
        records = gate.get("records")
        if not isinstance(records, list):
            continue
        for record in records:
            if not isinstance(record, dict):
                continue
            entries = record.get("wandb_completion_entries")
            if not isinstance(entries, list):
                continue
            for entry in entries:
                if not isinstance(entry, dict):
                    continue
                scope = (
                    entry.get("scope_attestation")
                    if isinstance(entry.get("scope_attestation"), dict)
                    else {}
                )
                rows.append(
                    {
                        "gate": gate.get("name"),
                        "review_path": record.get("path"),
                        "phase": record.get("phase"),
                        "benchmark": entry.get("benchmark"),
                        "run_id": entry.get("run_id"),
                        "verified": entry.get("verified"),
                        "query_source_required": entry.get("query_source_required"),
                        "query_source_valid": entry.get("query_source_valid"),
                        "adopted_existing_result": entry.get("adopted_existing_result"),
                        "scope_attestation_valid": entry.get("scope_attestation_valid"),
                        "sync_dry_run_report_ok": entry.get("sync_dry_run_report_ok"),
                        "sync_dry_run_report_json": entry.get("sync_dry_run_report_json"),
                        "sync_dry_run_source_review_json": entry.get(
                            "sync_dry_run_source_review_json"
                        ),
                        "sync_dry_run_source_review_sha256": entry.get(
                            "sync_dry_run_source_review_sha256"
                        ),
                        "scope_confirmed_by": scope.get("confirmed_by"),
                        "scope_confirmed_at": scope.get("confirmed_at"),
                        "source_attestation_json": scope.get("source_attestation_json"),
                        "source_attestation_sha256": scope.get("source_attestation_sha256"),
                        "source_audit_json": scope.get("source_audit_json"),
                        "source_audit_sha256": scope.get("source_audit_sha256"),
                        "source_audit_entry_matches": entry.get(
                            "source_audit_completion_entry_matches"
                        ),
                        "path": entry.get("path"),
                        "verification_error": entry.get("verification_error"),
                    }
                )
    return rows


def collect_pre_run_budget_rows(gates: list[Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for gate in gates:
        if not isinstance(gate, dict):
            continue
        records = gate.get("records")
        if not isinstance(records, list):
            continue
        for record in records:
            if not isinstance(record, dict):
                continue
            budget = record.get("pre_run_budget_estimate")
            if not isinstance(budget, dict):
                continue
            alignment = (
                record.get("budget_approval_alignment")
                if isinstance(record.get("budget_approval_alignment"), dict)
                else {}
            )
            rows.append(
                {
                    "gate": gate.get("name"),
                    "review_path": record.get("path"),
                    "phase": record.get("phase"),
                    "required": budget.get("required_before_paid_execution"),
                    "present": budget.get("present"),
                    "valid": budget.get("valid"),
                    "sha256_matches": budget.get("sha256_matches"),
                    "target_model": budget.get("target_model"),
                    "target_model_matches_selected_config": budget.get(
                        "target_model_matches_selected_config"
                    ),
                    "selected_model_identifiers": budget.get("selected_model_identifiers"),
                    "estimated_total_usd": budget.get("estimated_total_usd"),
                    "budget_alignment_valid": alignment.get("valid"),
                    "approved_budget_usd": alignment.get("approved_budget_usd"),
                    "approved_budget_covers_high": alignment.get(
                        "approved_budget_covers_estimate_high"
                    ),
                    "path": budget.get("path"),
                    "errors": {
                        "budget": budget.get("errors"),
                        "alignment": alignment.get("errors"),
                    },
                }
            )
    return rows


def build_markdown(report: dict[str, Any]) -> str:
    lines = [
        "# Taiwan Paid Run Review Check",
        "",
        f"Status: `{report['status']}`",
        f"OK: `{str(report['ok']).lower()}`",
        "",
        "## Review Files",
        "",
    ]
    review_paths = report.get("review_paths")
    if isinstance(review_paths, list) and review_paths:
        lines.extend(f"- `{path}`" for path in review_paths)
    else:
        lines.append("- none")
    lines.extend(
        [
            "",
            "## Gates",
            "",
            "| Gate | OK | Status | Detail |",
            "|---|---:|---|---|",
        ]
    )
    gates = report.get("gates") if isinstance(report.get("gates"), list) else []
    for gate in gates:
        if not isinstance(gate, dict):
            continue
        lines.append(
            "| "
            + " | ".join(
                [
                    str(gate.get("name", "")),
                    str(bool(gate.get("ok"))).lower(),
                    str(gate.get("status", "")),
                    str(gate.get("detail", "")).replace("|", "\\|"),
                ]
            )
            + " |"
        )
    requirements = (
        report.get("review_completion_requirements")
        if isinstance(report.get("review_completion_requirements"), dict)
        else {}
    )
    lines.extend(
        [
            "",
            "## Completion Requirements",
            "",
            "| Area | Requirement |",
            "|---|---|",
            "| Review fields | "
            + ", ".join(map(str, requirements.get("review_required_fields") or []))
            + " |",
            "| Completed review fields | "
            + ", ".join(map(str, requirements.get("completed_review_required_fields") or []))
            + " |",
            "| Run fields | "
            + ", ".join(map(str, requirements.get("run_required_fields") or []))
            + " |",
            "| W&B completion | "
            + str(requirements.get("wandb_completion_required_when", ""))
            + " |",
            "| Weave Agents completion | "
            + str(requirements.get("weave_agents_completion_required_when", ""))
            + " |",
            "| Accounting | "
            + ", ".join(map(str, requirements.get("accounting_required_fields") or []))
            + " |",
        ]
    )
    budget_rows = collect_pre_run_budget_rows(gates)
    lines.extend(
        [
            "",
            "## Pre-Run Budget Estimates",
            "",
            "| Gate | Review | Phase | Required | Present | Valid | SHA256 matches | Target model | Matches selected config | Selected model identifiers | Estimated total USD | Alignment valid | Approved USD | Approved covers high | Budget JSON | Errors |",
            "|---|---|---|---:|---:|---:|---:|---|---:|---|---|---:|---:|---:|---|---|",
        ]
    )
    if budget_rows:
        for row in budget_rows:
            lines.append(
                "| "
                + " | ".join(
                    [
                        md_cell(row.get("gate")),
                        md_cell(row.get("review_path")),
                        md_cell(row.get("phase")),
                        md_cell(row.get("required")),
                        md_cell(row.get("present")),
                        md_cell(row.get("valid")),
                        md_cell(row.get("sha256_matches")),
                        md_cell(row.get("target_model")),
                        md_cell(row.get("target_model_matches_selected_config")),
                        md_cell(row.get("selected_model_identifiers")),
                        md_cell(row.get("estimated_total_usd")),
                        md_cell(row.get("budget_alignment_valid")),
                        md_cell(row.get("approved_budget_usd")),
                        md_cell(row.get("approved_budget_covers_high")),
                        md_cell(row.get("path")),
                        md_cell(row.get("errors")),
                    ]
                )
                + " |"
            )
    else:
        lines.append("| none |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |")
    wandb_entries = collect_wandb_completion_entries(gates)
    lines.extend(
        [
            "",
            "## W&B Completion Entries",
            "",
            "| Gate | Review | Phase | Benchmark | Run ID | Verified | Query source required | Query source valid | Adopted existing | Scope attestation | Sync dry-run | Confirmed by | Confirmed at | Verifier JSON | Sync dry-run JSON | Source review JSON | Source review SHA | Source attestation | Source attestation SHA | Source audit | Source audit SHA | Source audit entry | Error |",
            "|---|---|---|---|---|---:|---:|---:|---:|---:|---:|---|---|---|---|---|---|---|---|---|---|---:|---|",
        ]
    )
    if wandb_entries:
        for entry in wandb_entries:
            lines.append(
                "| "
                + " | ".join(
                    [
                        md_cell(entry.get("gate")),
                        md_cell(entry.get("review_path")),
                        md_cell(entry.get("phase")),
                        md_cell(entry.get("benchmark")),
                        md_cell(entry.get("run_id")),
                        md_cell(entry.get("verified")),
                        md_cell(entry.get("query_source_required")),
                        md_cell(entry.get("query_source_valid")),
                        md_cell(entry.get("adopted_existing_result")),
                        md_cell(entry.get("scope_attestation_valid")),
                        md_cell(entry.get("sync_dry_run_report_ok")),
                        md_cell(entry.get("scope_confirmed_by")),
                        md_cell(entry.get("scope_confirmed_at")),
                        md_cell(entry.get("path")),
                        md_cell(entry.get("sync_dry_run_report_json")),
                        md_cell(entry.get("sync_dry_run_source_review_json")),
                        md_cell(entry.get("sync_dry_run_source_review_sha256")),
                        md_cell(entry.get("source_attestation_json")),
                        md_cell(entry.get("source_attestation_sha256")),
                        md_cell(entry.get("source_audit_json")),
                        md_cell(entry.get("source_audit_sha256")),
                        md_cell(entry.get("source_audit_entry_matches")),
                        md_cell(entry.get("verification_error")),
                    ]
                )
                + " |"
            )
    else:
        lines.append("| none |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |")
    lines.extend(["", "## Blocking Errors", ""])
    error_count = 0
    for gate in gates:
        if not isinstance(gate, dict) or gate.get("ok"):
            continue
        records = gate.get("records")
        if not isinstance(records, list):
            continue
        for record in records:
            if not isinstance(record, dict):
                continue
            errors = record.get("errors")
            if not isinstance(errors, list):
                continue
            for error in errors:
                error_count += 1
                lines.append(
                    f"- `{record.get('path', 'unknown')}`: {str(error)}"
                )
    if error_count == 0:
        lines.append("- none")
    lines.append("")
    return "\n".join(lines)


def write_text(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--review-json", type=Path, action="append")
    parser.add_argument(
        "--required-wandb-benchmark",
        action="append",
        choices=["agentic_math", "agentic_swe", "taiwan_full"],
    )
    parser.add_argument(
        "--required-wandb-run-id",
        action="append",
        metavar="BENCHMARK=RUN_ID",
    )
    parser.add_argument("--required-wandb-run-id-all", metavar="RUN_ID")
    parser.add_argument("--require-one-model-canary", action="store_true")
    parser.add_argument(
        "--wandb-completion-max-age-seconds",
        type=int,
        default=readiness.DEFAULT_WANDB_COMPLETION_MAX_AGE_SECONDS,
    )
    parser.add_argument(
        "--weave-agents-completion-max-age-seconds",
        type=int,
        default=readiness.DEFAULT_WANDB_COMPLETION_MAX_AGE_SECONDS,
    )
    parser.add_argument("--json", type=Path)
    parser.add_argument("--markdown", type=Path)
    parser.add_argument("--fail-on-invalid", action="store_true")
    args = parser.parse_args(argv)
    if args.wandb_completion_max_age_seconds < 0:
        args.wandb_completion_max_age_seconds = None
    if args.weave_agents_completion_max_age_seconds < 0:
        args.weave_agents_completion_max_age_seconds = None
    return args


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    report = build_report(args)
    if args.json:
        output = repo_path(args.json)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    if args.markdown:
        write_text(repo_path(args.markdown), build_markdown(report))
    print(json.dumps(report, ensure_ascii=False, indent=2))
    if args.fail_on_invalid and not report["ok"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
