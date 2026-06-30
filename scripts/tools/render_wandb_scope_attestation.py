#!/usr/bin/env python3
"""Render a confirmed W&B scope-attestation JSON from a draft template.

This tool is offline. It does not query W&B, write W&B, mutate paid-run review
JSONs, install NeMoClaw, or launch model inference. It exists to replace
hand-editing of W&B adoption scope-attestation templates with an auditable,
source-bound rendering step before verify_wandb_scope_attestation.py and
sync_wandb_completion_to_paid_review.py are run.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import re
import shlex
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[2]
SCHEMA_VERSION = 1
SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
PLACEHOLDER_TOKENS = (
    "TODO",
    "TBD",
    "PLACEHOLDER",
    "PENDING",
    "YYYY",
    "RUN_ID",
    "MODEL_SLUG",
    "ACTUAL_OR_BILLING",
    "$ACTUAL",
    "BILL_OR_DASHBOARD",
    "APPROVER",
    "REVIEWER",
    "未定",
    "不明",
    "仮",
)
MIN_CONFIRMATION_LENGTH = 20
SAFETY_FLAGS = {
    "executes_external_action": False,
    "queries_wandb": False,
    "writes_wandb": False,
    "installs_third_party": False,
    "launches_model_inference": False,
    "mutates_paid_review": False,
}


def repo_path(value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else REPO_ROOT / path


def path_display(value: str | Path) -> str:
    path = Path(value)
    try:
        return str(path.resolve().relative_to(REPO_ROOT))
    except (OSError, ValueError):
        return str(path)


def read_json_object(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"{path} is not a JSON object")
    return payload


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def shell_quote(value: str | Path | None) -> str:
    return shlex.quote(str(value or ""))


def build_preflight_command(
    *,
    review_path: Any,
    completion_path: Any,
    scope_attestation_json: Path,
    preflight_report_json: Path | None,
) -> str:
    parts = [
        "uv",
        "run",
        "python",
        "scripts/tools/verify_wandb_scope_attestation.py",
        "--review-json",
        str(review_path or ""),
        "--completion-json",
        str(completion_path or ""),
        "--scope-attestation-json",
        path_display(scope_attestation_json),
    ]
    if preflight_report_json is not None:
        parts.extend(["--json", path_display(preflight_report_json)])
    return " ".join(shell_quote(part) for part in parts)


def build_sync_dry_run_command(
    *,
    review_path: Any,
    completion_path: Any,
    scope_attestation_json: Path,
    sync_dry_run_report_json: Path | None,
) -> str:
    parts = [
        "uv",
        "run",
        "python",
        "scripts/tools/sync_wandb_completion_to_paid_review.py",
        "--review-json",
        str(review_path or ""),
        "--completion-json",
        str(completion_path or ""),
        "--set-verify-wandb-completion",
        "--top-level",
        "--adopt-existing-result",
        "--scope-attestation-json",
        path_display(scope_attestation_json),
    ]
    if sync_dry_run_report_json is not None:
        parts.extend(["--report-json", path_display(sync_dry_run_report_json)])
    return " ".join(shell_quote(part) for part in parts)


def is_placeholder(value: Any) -> bool:
    if not isinstance(value, str):
        return False
    text = value.strip()
    if not text:
        return True
    upper = text.upper()
    return any(token.upper() in upper for token in PLACEHOLDER_TOKENS)


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


def parse_positive_money(value: str) -> bool:
    text = value.strip().replace(",", "")
    text = text.replace("$", "").replace("USD", "").replace("usd", "").strip()
    try:
        amount = float(text)
    except ValueError:
        return False
    return math.isfinite(amount) and amount > 0


def require_concrete_string(errors: list[str], payload: dict[str, Any], field: str) -> str:
    value = payload.get(field)
    if not isinstance(value, str) or is_placeholder(value):
        errors.append(f"{field} must be a concrete non-placeholder string")
        return ""
    return value.strip()


def validate_source_binding(template: dict[str, Any]) -> list[str]:
    errors: list[str] = []
    completion_path_value = template.get("completion_path")
    completion_sha256 = template.get("completion_sha256")
    if not isinstance(completion_path_value, str) or is_placeholder(completion_path_value):
        errors.append("completion_path must be concrete")
    elif not isinstance(completion_sha256, str) or SHA256_RE.fullmatch(completion_sha256) is None:
        errors.append("completion_sha256 must be a 64-character lowercase hex digest")
    else:
        completion_path = repo_path(completion_path_value)
        try:
            actual_completion_sha = sha256_file(completion_path)
            completion_payload = read_json_object(completion_path)
        except (OSError, json.JSONDecodeError, ValueError) as exc:
            errors.append(f"completion_path is not readable: {exc}")
        else:
            if actual_completion_sha != completion_sha256:
                errors.append("completion_sha256 does not match completion_path")
            for field, completion_field in (
                ("benchmark", "benchmark"),
                ("entity", "entity"),
                ("project", "project"),
                ("run_id", "run_id"),
            ):
                if template.get(field) != completion_payload.get(completion_field):
                    errors.append(f"{field} does not match completion JSON")

    source_audit_json = template.get("source_audit_json")
    source_audit_sha256 = template.get("source_audit_sha256")
    if source_audit_json is not None or source_audit_sha256 is not None:
        if not isinstance(source_audit_json, str) or is_placeholder(source_audit_json):
            errors.append("source_audit_json must be concrete when source_audit_sha256 is present")
        elif not isinstance(source_audit_sha256, str) or SHA256_RE.fullmatch(source_audit_sha256) is None:
            errors.append("source_audit_sha256 must be a 64-character lowercase hex digest")
        else:
            source_audit_path = repo_path(source_audit_json)
            try:
                actual_source_audit_sha = sha256_file(source_audit_path)
                read_json_object(source_audit_path)
            except (OSError, json.JSONDecodeError, ValueError) as exc:
                errors.append(f"source_audit_json is not readable: {exc}")
            else:
                if actual_source_audit_sha != source_audit_sha256:
                    errors.append("source_audit_sha256 does not match source_audit_json")

    review_path = template.get("review_path")
    if not isinstance(review_path, str) or is_placeholder(review_path):
        errors.append("review_path must be concrete")
    else:
        try:
            read_json_object(repo_path(review_path))
        except (OSError, json.JSONDecodeError, ValueError) as exc:
            errors.append(f"review_path is not readable: {exc}")

    for field in ("benchmark", "entity", "project", "run_id"):
        require_concrete_string(errors, template, field)
    return errors


def render_attestation(
    *,
    template_json: Path,
    output_json: Path,
    confirmed_by: str,
    confirmed_at: str,
    confirmation: str,
    actual_cost_estimate: str,
    provider_bill_reference: str,
    report_json: Path | None = None,
    markdown: Path | None = None,
    preflight_report_json: Path | None = None,
    sync_dry_run_report_json: Path | None = None,
) -> dict[str, Any]:
    template = read_json_object(template_json)
    source_template_sha256 = sha256_file(template_json)
    errors: list[str] = []
    if template.get("schema_version") != SCHEMA_VERSION:
        errors.append(f"schema_version must be {SCHEMA_VERSION}")
    errors.extend(validate_source_binding(template))

    reviewer_payload = {
        "confirmed_by": confirmed_by,
        "confirmed_at": confirmed_at,
        "confirmation": confirmation,
        "actual_cost_estimate": actual_cost_estimate,
        "provider_bill_reference": provider_bill_reference,
    }
    for field in reviewer_payload:
        require_concrete_string(errors, reviewer_payload, field)
    if not parse_aware_timestamp(confirmed_at):
        errors.append("confirmed_at must be a timezone-aware ISO 8601 timestamp")
    if len(confirmation.strip()) < MIN_CONFIRMATION_LENGTH:
        errors.append(
            "confirmation must be a concrete sentence with at least "
            f"{MIN_CONFIRMATION_LENGTH} characters"
        )
    if isinstance(actual_cost_estimate, str) and not is_placeholder(actual_cost_estimate):
        if not parse_positive_money(actual_cost_estimate):
            errors.append("actual_cost_estimate must include a positive numeric amount")

    if errors:
        report = {
            "schema_version": 1,
            "ok": False,
            "status": "validation_failed",
            "generated_at": time.time(),
            "template_json": path_display(template_json),
            "template_sha256": source_template_sha256,
            "output_json": path_display(output_json),
            "errors": errors,
            "safety": dict(SAFETY_FLAGS),
        }
        if report_json is not None:
            write_json(report_json, report)
        print(json.dumps(report, ensure_ascii=False, indent=2))
        raise SystemExit(1)

    output = copy.deepcopy(template)
    output["confirmed"] = True
    output["confirmed_by"] = confirmed_by.strip()
    output["confirmed_at"] = confirmed_at.strip()
    output["confirmation"] = confirmation.strip()
    output["actual_cost_estimate"] = actual_cost_estimate.strip()
    output["provider_bill_reference"] = provider_bill_reference.strip()
    output["rendered_scope_attestation"] = {
        "schema_version": 1,
        "generated_at": time.time(),
        "source_template_json": path_display(template_json),
        "source_template_sha256": source_template_sha256,
        "output_json": path_display(output_json),
        "will_execute_external_actions": False,
        "safety": dict(SAFETY_FLAGS),
    }
    write_json(output_json, output)
    output_sha256 = sha256_file(output_json)
    report = {
        "schema_version": 1,
        "ok": True,
        "status": "rendered",
        "generated_at": time.time(),
        "template_json": path_display(template_json),
        "template_sha256": source_template_sha256,
        "output_json": path_display(output_json),
        "output_sha256": output_sha256,
        "review_path": output.get("review_path"),
        "completion_path": output.get("completion_path"),
        "completion_sha256": output.get("completion_sha256"),
        "source_audit_json": output.get("source_audit_json"),
        "source_audit_sha256": output.get("source_audit_sha256"),
        "benchmark": output.get("benchmark"),
        "entity": output.get("entity"),
        "project": output.get("project"),
        "run_id": output.get("run_id"),
        "will_execute_external_actions": False,
        "safety": dict(SAFETY_FLAGS),
        "next_commands": {
            "preflight": build_preflight_command(
                review_path=output.get("review_path"),
                completion_path=output.get("completion_path"),
                scope_attestation_json=output_json,
                preflight_report_json=preflight_report_json,
            ),
            "sync_dry_run": build_sync_dry_run_command(
                review_path=output.get("review_path"),
                completion_path=output.get("completion_path"),
                scope_attestation_json=output_json,
                sync_dry_run_report_json=sync_dry_run_report_json,
            ),
        },
        "errors": [],
    }
    if report_json is not None:
        write_json(report_json, report)
    if markdown is not None:
        markdown.parent.mkdir(parents=True, exist_ok=True)
        markdown.write_text(render_markdown(report), encoding="utf-8")
    return report


def md_cell(value: Any) -> str:
    if value is None:
        text = ""
    elif isinstance(value, bool):
        text = str(value).lower()
    else:
        text = str(value)
    return text.replace("\n", "<br>").replace("|", "\\|")


def render_markdown(report: dict[str, Any]) -> str:
    lines = [
        "# W&B Scope Attestation Render Report",
        "",
        f"Status: `{report.get('status')}`",
        f"OK: `{report.get('ok')}`",
        f"Will execute external actions: `{report.get('will_execute_external_actions')}`",
        "",
        "| Field | Value |",
        "|---|---|",
    ]
    for field in (
        "template_json",
        "template_sha256",
        "output_json",
        "output_sha256",
        "review_path",
        "completion_path",
        "completion_sha256",
        "source_audit_json",
        "source_audit_sha256",
        "benchmark",
        "entity",
        "project",
        "run_id",
    ):
        lines.append(f"| {field} | {md_cell(report.get(field))} |")
    lines.extend(["", "## Next Commands", ""])
    next_commands = report.get("next_commands")
    if isinstance(next_commands, dict):
        for label, command in next_commands.items():
            lines.extend([f"### {label}", "", "```bash", str(command), "```", ""])
    return "\n".join(lines)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--template-json", type=Path, required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--confirmed-by", required=True)
    parser.add_argument("--confirmed-at", required=True)
    parser.add_argument("--confirmation", required=True)
    parser.add_argument("--actual-cost-estimate", required=True)
    parser.add_argument("--provider-bill-reference", required=True)
    parser.add_argument("--report-json", type=Path)
    parser.add_argument("--markdown", type=Path)
    parser.add_argument("--preflight-report-json", type=Path)
    parser.add_argument("--sync-dry-run-report-json", type=Path)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    report = render_attestation(
        template_json=repo_path(args.template_json),
        output_json=repo_path(args.output_json),
        confirmed_by=args.confirmed_by,
        confirmed_at=args.confirmed_at,
        confirmation=args.confirmation,
        actual_cost_estimate=args.actual_cost_estimate,
        provider_bill_reference=args.provider_bill_reference,
        report_json=repo_path(args.report_json) if args.report_json else None,
        markdown=repo_path(args.markdown) if args.markdown else None,
        preflight_report_json=(
            repo_path(args.preflight_report_json)
            if args.preflight_report_json
            else None
        ),
        sync_dry_run_report_json=(
            repo_path(args.sync_dry_run_report_json)
            if args.sync_dry_run_report_json
            else None
        ),
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main(sys.argv[1:])
