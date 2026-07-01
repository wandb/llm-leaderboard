#!/usr/bin/env python3
"""Audit local Taiwan evaluation results against W&B completion evidence.

This tool is read-only. It does not query W&B and does not run inference. It
scans local result files, classifies complete local outputs, and checks whether
each complete output is backed by a passing W&B completion verifier JSON.
"""

from __future__ import annotations

import argparse
import csv
import json
import time
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUTPUT_ROOT = Path("outputs") / "taiwan_full_eval"
DEFAULT_WANDB_COMPLETION_DIR = DEFAULT_OUTPUT_ROOT / "wandb_completion"
DEFAULT_PROVISIONAL_LEADERBOARD_DIR = DEFAULT_OUTPUT_ROOT / "provisional_leaderboard"
AGENTIC_MATH_EXPECTED_TOTAL = 100
AGENTIC_SWE_EXPECTED_TOTAL = 80
WANDB_COMPLETION_SCHEMA_VERSION = 1
WANDB_COMPLETION_QUERY_SOURCE_KIND = "wandb_sdk"
FULL_BENCHMARK_ID = "taiwan_full"
NEMOCLAW_AUDIT_REQUIRED_BENCHMARKS = {"agentic_math", "agentic_swe"}
LOCAL_COMPLETE_NEEDS_WANDB_RELOG = "local_complete_needs_wandb_relog"
LOCAL_COMPLETE_MISSING_NEMOCLAW_AUDIT = "local_complete_missing_nemoclaw_audit"
AGENTIC_MATH_NEMOCLAW_SUMMARY_KEYS = (
    "nemoclaw_session_audit_required_instances",
    "nemoclaw_session_audit_passed_instances",
    "nemoclaw_session_audit_failed_instances",
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


def read_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"{path} is not a JSON object")
    return payload


def read_jsonl_rows(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            payload = json.loads(line)
            if not isinstance(payload, dict):
                raise ValueError(f"{path}:{line_no} is not a JSON object")
            rows.append(payload)
    return rows


def read_csv_rows(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        if not reader.fieldnames:
            return []
        return [dict(row) for row in reader]


def jsonl_count(path: Path) -> int:
    if not path.exists():
        return 0
    with path.open(encoding="utf-8") as handle:
        return sum(1 for line in handle if line.strip())


def metric_from_checks(completion: dict[str, Any], name: str) -> Any:
    checks = completion.get("checks")
    if not isinstance(checks, list):
        return None
    for check in checks:
        if isinstance(check, dict) and check.get("name") == name:
            return check.get("value", check.get("nrows"))
    return None


def metric_from_observed_evidence(completion: dict[str, Any], metric_name: str) -> Any:
    observed = completion.get("observed_evidence")
    if not isinstance(observed, dict):
        return None
    metrics = observed.get("summary_metrics")
    if not isinstance(metrics, dict):
        return None
    metric = metrics.get(metric_name)
    if not isinstance(metric, dict):
        return None
    return metric.get("value")


def table_rows_from_observed_evidence(completion: dict[str, Any], table_name: str) -> Any:
    observed = completion.get("observed_evidence")
    if not isinstance(observed, dict):
        return None
    tables = observed.get("tables")
    if not isinstance(tables, list):
        return None
    for table in tables:
        if isinstance(table, dict) and table.get("name") == table_name:
            return table.get("nrows")
    return None


def int_like(value: Any) -> int | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, int):
        return value
    if isinstance(value, float) and value.is_integer():
        return int(value)
    return None


def nemoclaw_session_audit_issues(completion: dict[str, Any]) -> list[str]:
    benchmark = completion.get("benchmark")
    if benchmark not in NEMOCLAW_AUDIT_REQUIRED_BENCHMARKS:
        return []
    issues: list[str] = []
    required = completion.get("required_evidence")
    if not isinstance(required, dict):
        issues.append("required_evidence must be an object for NeMoClaw session audit proof")
        required = {}
    required_audit = required.get("nemoclaw_session_audit")
    if not isinstance(required_audit, dict):
        issues.append("required_evidence.nemoclaw_session_audit must be an object")
    elif required_audit.get("required") is not True:
        issues.append("required_evidence.nemoclaw_session_audit.required must be true")
    expected_total = int_like(required.get("expected_total"))
    if expected_total is None or expected_total <= 0:
        issues.append("required_evidence.expected_total must be a positive integer")

    observed = completion.get("observed_evidence")
    if not isinstance(observed, dict):
        issues.append("observed_evidence must be an object for NeMoClaw session audit proof")
        observed = {}
    observed_audit = observed.get("nemoclaw_session_audit")
    if not isinstance(observed_audit, dict):
        issues.append("observed_evidence.nemoclaw_session_audit must be an object")
        return issues
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
    return issues


def agentic_math_local_nemoclaw_audit_issues(
    *,
    summary: dict[str, Any],
    rows: list[dict[str, Any]],
    expected_total: int,
) -> list[str]:
    issues: list[str] = []
    missing = [key for key in AGENTIC_MATH_NEMOCLAW_SUMMARY_KEYS if key not in summary]
    if missing:
        issues.append(f"summary.json missing NeMoClaw audit keys: {missing}")
        return issues

    required = int_like(summary.get("nemoclaw_session_audit_required_instances"))
    passed = int_like(summary.get("nemoclaw_session_audit_passed_instances"))
    failed = int_like(summary.get("nemoclaw_session_audit_failed_instances"))
    if required is None:
        issues.append("nemoclaw_session_audit_required_instances must be an integer")
    if passed is None:
        issues.append("nemoclaw_session_audit_passed_instances must be an integer")
    if failed is None:
        issues.append("nemoclaw_session_audit_failed_instances must be an integer")

    row_required = sum(
        1
        for row in rows
        if isinstance(row.get("nemoclaw_session_audit"), dict)
        and row["nemoclaw_session_audit"].get("required") is True
    )
    row_passed = sum(1 for row in rows if row.get("nemoclaw_session_audit_ok") is True)
    row_failed = sum(
        1
        for row in rows
        if isinstance(row.get("nemoclaw_session_audit"), dict)
        and row["nemoclaw_session_audit"].get("required") is True
        and row.get("nemoclaw_session_audit_ok") is False
    )
    if required is not None and required != expected_total:
        issues.append("nemoclaw_session_audit_required_instances must equal expected_total")
    if passed is not None and passed != expected_total:
        issues.append("nemoclaw_session_audit_passed_instances must equal expected_total")
    if failed is not None and failed != 0:
        issues.append("nemoclaw_session_audit_failed_instances must be 0")
    if required is not None and required != row_required:
        issues.append("summary NeMoClaw required count does not match row audit required count")
    if passed is not None and passed != row_passed:
        issues.append("summary NeMoClaw passed count does not match row audit passed count")
    if failed is not None and failed != row_failed:
        issues.append("summary NeMoClaw failed count does not match row audit failed count")
    if row_required != expected_total:
        issues.append("every result row must require NeMoClaw session audit")
    if row_passed != expected_total:
        issues.append("every result row must pass NeMoClaw session audit")
    if row_failed != 0:
        issues.append("no result row may have failed NeMoClaw session audit")
    return issues


def agentic_swe_local_nemoclaw_audit_issues(
    *,
    official: dict[str, Any],
    patch_rows: list[Any],
    expected_total: int,
) -> list[str]:
    issues: list[str] = []
    resolved_ids = official.get("resolved_ids")
    unresolved_ids = official.get("unresolved_ids")
    if not isinstance(resolved_ids, list) or not isinstance(unresolved_ids, list):
        issues.append("official summary must include resolved_ids and unresolved_ids for NeMoClaw audit")
        expected_ids: set[str] = set()
    else:
        expected_ids = {str(value) for value in [*resolved_ids, *unresolved_ids]}
        if len(expected_ids) != expected_total:
            issues.append("resolved_ids plus unresolved_ids must equal expected_total")

    typed_rows = [row for row in patch_rows if isinstance(row, dict)]
    patch_by_instance = {
        str(row.get("instance_id")): row
        for row in typed_rows
        if row.get("instance_id") is not None
    }
    missing_patch_rows = sorted(expected_ids - set(patch_by_instance))
    if missing_patch_rows:
        issues.append(f"patches.json missing official instance ids: {missing_patch_rows[:5]}")

    rows_for_audit = [
        patch_by_instance[instance_id]
        for instance_id in sorted(expected_ids)
        if instance_id in patch_by_instance
    ] or typed_rows
    required = [
        row
        for row in rows_for_audit
        if isinstance(row.get("nemoclaw_session_audit"), dict)
        and row["nemoclaw_session_audit"].get("required") is True
    ]
    passed = [row for row in required if row.get("nemoclaw_session_audit_ok") is True]
    failed = [row for row in required if row.get("nemoclaw_session_audit_ok") is False]
    if len(required) != expected_total:
        issues.append("agentic_swe NeMoClaw audit required patch count must equal expected_total")
    if len(passed) != expected_total:
        issues.append("agentic_swe NeMoClaw audit passed patch count must equal expected_total")
    if failed:
        issues.append("agentic_swe NeMoClaw audit failed patch count must be 0")
    return issues


def completion_schema_issues(completion: dict[str, Any]) -> list[str]:
    issues: list[str] = []
    if completion.get("schema_version") != WANDB_COMPLETION_SCHEMA_VERSION:
        issues.append("schema_version must be 1")
    if completion.get("verification_schema_version") != WANDB_COMPLETION_SCHEMA_VERSION:
        issues.append("verification_schema_version must be 1")
    if completion.get("status") != "passed":
        issues.append("status must be passed")
    if completion.get("ok") is not True:
        issues.append("ok must be true")
    for field in ("entity", "project", "run_id"):
        if not isinstance(completion.get(field), str) or not completion.get(field).strip():
            issues.append(f"{field} must be a non-empty string")
    if not isinstance(completion.get("generated_at"), (int, float)):
        issues.append("generated_at must be numeric")
    checks = completion.get("checks")
    if not isinstance(checks, list) or not checks:
        issues.append("checks must be a non-empty list")
    else:
        failed_checks = [
            str(check.get("name") or index)
            for index, check in enumerate(checks, start=1)
            if not isinstance(check, dict) or check.get("ok") is not True
        ]
        if failed_checks:
            issues.append(
                "all checks must be ok=true: " + ", ".join(failed_checks[:10])
            )
    query_source = completion.get("query_source")
    if not isinstance(query_source, dict):
        issues.append("query_source must be an object")
        query_source = {}
    else:
        if query_source.get("kind") != WANDB_COMPLETION_QUERY_SOURCE_KIND:
            issues.append("query_source.kind must be wandb_sdk")
        for field in ("entity", "project", "run_id", "benchmark"):
            if query_source.get(field) != completion.get(field):
                issues.append(f"query_source.{field} must match top-level {field}")
    if not isinstance(completion.get("required_evidence"), dict) or not completion.get("required_evidence"):
        issues.append("required_evidence must be a non-empty object")
    observed = completion.get("observed_evidence")
    if not isinstance(observed, dict) or not observed:
        issues.append("observed_evidence must be a non-empty object")
        observed = {}
    if observed.get("run_state") != "finished":
        issues.append("observed_evidence.run_state must be finished")
    if completion.get("benchmark") == FULL_BENCHMARK_ID:
        required = completion.get("required_evidence")
        if not isinstance(required, dict):
            issues.append("required_evidence must be an object for taiwan_full")
            required = {}
        if not isinstance(required.get("taxonomy_tables"), list):
            issues.append("required_evidence.taxonomy_tables must be a list")
        if not isinstance(required.get("aggregate_tables"), list):
            issues.append("required_evidence.aggregate_tables must be a list")
        if not isinstance(observed.get("taxonomy_tables"), list):
            issues.append("observed_evidence.taxonomy_tables must be a list")
        if not isinstance(observed.get("aggregate_tables"), list):
            issues.append("observed_evidence.aggregate_tables must be a list")
        return issues
    if not isinstance(observed.get("summary_metrics"), dict):
        issues.append("observed_evidence.summary_metrics must be an object")
    if not isinstance(observed.get("tables"), list):
        issues.append("observed_evidence.tables must be a list")
    issues.extend(nemoclaw_session_audit_issues(completion))
    return issues


def current_completion_schema(completion: dict[str, Any]) -> bool:
    return not completion_schema_issues(completion)


def first_not_none(*values: Any) -> Any:
    for value in values:
        if value is not None:
            return value
    return None


def normalize_text(value: Any) -> str:
    return "".join(ch for ch in str(value).lower() if ch.isalnum())


def completion_search_text(completion: dict[str, Any]) -> str:
    values: list[str] = [
        str(completion.get("run_name") or ""),
        str(completion.get("run_id") or ""),
    ]
    checks = completion.get("checks")
    if isinstance(checks, list):
        for check in checks:
            if not isinstance(check, dict):
                continue
            artifacts = check.get("artifacts")
            if isinstance(artifacts, list):
                for artifact in artifacts:
                    if isinstance(artifact, dict):
                        values.append(str(artifact.get("name") or ""))
    return normalize_text(" ".join(values))


def local_model_tokens(record: dict[str, Any]) -> list[str]:
    tokens = [
        record.get("model_slug"),
        record.get("model"),
    ]
    model = str(record.get("model") or "")
    if "/" in model:
        tokens.append(model.rsplit("/", 1)[-1])
    return [
        normalize_text(token)
        for token in tokens
        if isinstance(token, str) and token.strip()
    ]


def completion_metrics(completion: dict[str, Any]) -> dict[str, Any]:
    benchmark = completion.get("benchmark")
    if benchmark == "agentic_math":
        return {
            "total": first_not_none(
                metric_from_observed_evidence(completion, "agentic_math/total_instances"),
                metric_from_checks(completion, "total_metric"),
            ),
            "answered": first_not_none(
                metric_from_observed_evidence(completion, "agentic_math/answered_instances"),
                metric_from_checks(completion, "answered_metric"),
            ),
            "correct": first_not_none(
                metric_from_observed_evidence(completion, "agentic_math/correct_instances"),
                metric_from_checks(completion, "correct_metric"),
            ),
            "accuracy": first_not_none(
                metric_from_observed_evidence(completion, "agentic_math/accuracy"),
                metric_from_checks(completion, "accuracy_metric"),
            ),
            "output_rows": first_not_none(
                table_rows_from_observed_evidence(completion, "agentic_math_output_table"),
                metric_from_checks(completion, "output_table"),
            ),
        }
    if benchmark == "agentic_swe":
        return {
            "total": first_not_none(
                metric_from_observed_evidence(completion, "agentic_swe/total_instances"),
                metric_from_checks(completion, "total_metric"),
            ),
            "correct": first_not_none(
                metric_from_observed_evidence(completion, "agentic_swe/resolved_instances"),
                metric_from_checks(completion, "correct_metric"),
            ),
            "accuracy": first_not_none(
                metric_from_observed_evidence(completion, "agentic_swe/pass_at_1"),
                metric_from_checks(completion, "accuracy_metric"),
            ),
            "output_rows": first_not_none(
                table_rows_from_observed_evidence(completion, "agentic_swe_output_table"),
                metric_from_checks(completion, "output_table"),
            ),
        }
    if benchmark == FULL_BENCHMARK_ID:
        observed = completion.get("observed_evidence")
        if not isinstance(observed, dict):
            return {}
        taxonomy_tables = observed.get("taxonomy_tables")
        aggregate_tables = observed.get("aggregate_tables")
        return {
            "taxonomy_table_count": len(taxonomy_tables)
            if isinstance(taxonomy_tables, list)
            else 0,
            "aggregate_table_count": len(aggregate_tables)
            if isinstance(aggregate_tables, list)
            else 0,
        }
    return {}


def numbers_equal(left: Any, right: Any, *, tolerance: float = 1e-12) -> bool:
    try:
        return abs(float(left) - float(right)) <= tolerance
    except (TypeError, ValueError):
        return left == right


def metrics_match(record: dict[str, Any], completion: dict[str, Any]) -> bool:
    if record.get("benchmark") == FULL_BENCHMARK_ID:
        run_id = record.get("run_id")
        return bool(
            isinstance(run_id, str)
            and run_id.strip()
            and completion.get("run_id") == run_id
        )
    metrics = completion_metrics(completion)
    checks = [
        ("total", "total_instances"),
        ("correct", "correct_instances"),
        ("accuracy", "accuracy"),
    ]
    if record["benchmark"] == "agentic_math":
        checks.append(("answered", "answered_instances"))
    if record["benchmark"] == "agentic_swe":
        checks = [
            ("total", "total_instances"),
            ("correct", "resolved_instances"),
            ("accuracy", "pass_at_1"),
        ]
    for completion_key, record_key in checks:
        if completion_key not in metrics:
            return False
        if not numbers_equal(metrics.get(completion_key), record.get(record_key)):
            return False
    output_rows = metrics.get("output_rows")
    if output_rows is not None and not numbers_equal(output_rows, record.get("row_count")):
        return False
    return True


def model_matches(record: dict[str, Any], completion: dict[str, Any]) -> bool:
    if record.get("benchmark") == FULL_BENCHMARK_ID:
        run_id = record.get("run_id")
        return bool(
            isinstance(run_id, str)
            and run_id.strip()
            and completion.get("run_id") == run_id
        )
    text = completion_search_text(completion)
    return any(token and token in text for token in local_model_tokens(record))


def shell_quote(value: Any) -> str:
    text = str(value)
    if text and all(char.isalnum() or char in "/._:-" for char in text):
        return text
    return "'" + text.replace("'", "'\"'\"'") + "'"


def safe_plan_component(value: Any) -> str:
    text = str(value or "unknown").strip().lower()
    result = "".join(char if char.isalnum() else "-" for char in text)
    result = "-".join(part for part in result.split("-") if part)
    return result or "unknown"


def relog_dry_run_plan_json(record: dict[str, Any]) -> str:
    benchmark = safe_plan_component(record.get("benchmark"))
    model_slug = safe_plan_component(record.get("model_slug"))
    components = [benchmark, model_slug]
    run_kind = str(record.get("run_kind") or "").strip()
    if run_kind and run_kind != "final":
        components.append(safe_plan_component(run_kind))
        result_dir = str(record.get("result_dir") or "").strip()
        if result_dir:
            components.append(safe_plan_component(Path(result_dir).name))
    return f"temp/wandb_relog_plans/{'-'.join(components)}.plan.json"


def relog_command(record: dict[str, Any], *, include_validated_plan: bool = True) -> str:
    model_name = record.get("model") or record.get("model_slug") or "MODEL_NAME"
    validated_plan_arg = (
        " --validated-dry-run-plan-json "
        f"{shell_quote(relog_dry_run_plan_json(record))}"
        if include_validated_plan
        else ""
    )
    approval_arg = (
        " --external-action-approval-source-packet-json "
        "EXTERNAL_ACTION_APPROVAL_SOURCE_PACKET_JSON"
        " --external-action-approval-report-json EXTERNAL_ACTION_APPROVAL_REPORT_JSON"
        if include_validated_plan
        else ""
    )
    if record.get("benchmark") == "agentic_math":
        return (
            "uv run python scripts/tools/log_agentic_math_results_to_wandb.py "
            f"--results-dir {shell_quote(record.get('result_dir'))} "
            f"--model-name {shell_quote(model_name)}"
            f"{validated_plan_arg}"
            f"{approval_arg}"
        )
    if record.get("benchmark") == "agentic_swe":
        official_summary_path = str(record.get("official_summary_path") or "")
        official_eval_dir = str(Path(official_summary_path).parent) if official_summary_path else "OFFICIAL_EVAL_DIR"
        patch_path = record.get("patches_path") or "PATCHES_JSON"
        expected_total = record.get("expected_total") or AGENTIC_SWE_EXPECTED_TOTAL
        return (
            "uv run python scripts/tools/log_agentic_swe_results_to_wandb.py "
            f"--official-eval-dir {shell_quote(official_eval_dir)} "
            f"--patch-path {shell_quote(patch_path)} "
            f"--model-name {shell_quote(model_name)} "
            f"--expected-total {shell_quote(expected_total)}"
            f"{validated_plan_arg}"
            f"{approval_arg}"
        )
    if record.get("benchmark") == FULL_BENCHMARK_ID:
        return ""
    return "UNSUPPORTED_BENCHMARK_RELOG"


def relog_inputs_present(record: dict[str, Any]) -> bool:
    if record.get("formalization_status") == LOCAL_COMPLETE_MISSING_NEMOCLAW_AUDIT:
        return False
    if record.get("benchmark") == "agentic_math":
        return all(
            isinstance(record.get(field), str) and bool(record.get(field))
            for field in ("result_dir", "summary_path", "results_path")
        )
    if record.get("benchmark") == "agentic_swe":
        return all(
            isinstance(record.get(field), str) and bool(record.get(field))
            for field in ("official_summary_path", "patches_path")
        )
    return False


def relog_dry_run_command(record: dict[str, Any]) -> str:
    if not relog_inputs_present(record):
        return ""
    command = relog_command(record, include_validated_plan=False)
    if not command or command.startswith("UNSUPPORTED_"):
        return ""
    return (
        f"{command} --dry-run --plan-json "
        f"{shell_quote(relog_dry_run_plan_json(record))}"
    )


def verifier_command(record: dict[str, Any], *, run_id: str = "RUN_ID") -> str:
    if record.get("benchmark") == "agentic_math":
        return (
            "uv run python scripts/tools/verify_taiwan_wandb_completion.py "
            f"--run-id {run_id} --benchmark agentic_math --expected-total 100 "
            "--require-nemoclaw-session-audit "
            "--env-file .env "
            f"--json outputs/taiwan_full_eval/wandb_completion/agentic_math-{run_id}.json"
        )
    if record.get("benchmark") == "agentic_swe":
        return (
            "uv run python scripts/tools/verify_taiwan_wandb_completion.py "
            f"--run-id {run_id} --benchmark agentic_swe --expected-total 80 "
            "--require-nemoclaw-session-audit "
            "--env-file .env "
            f"--json outputs/taiwan_full_eval/wandb_completion/agentic_swe-{run_id}.json"
        )
    if record.get("benchmark") == FULL_BENCHMARK_ID:
        return (
            "uv run python scripts/tools/verify_taiwan_wandb_completion.py "
            f"--run-id {run_id} --benchmark taiwan_full --env-file .env "
            f"--json outputs/taiwan_full_eval/wandb_completion/taiwan_full-{run_id}.json"
        )
    return "UNSUPPORTED_BENCHMARK_VERIFY"


def safe_read_json(path: Path) -> tuple[dict[str, Any] | None, str | None]:
    if not path.exists():
        return None, "missing"
    try:
        return read_json(path), None
    except (OSError, json.JSONDecodeError, ValueError) as exc:
        return None, str(exc)


def agentic_math_record(model_dir: Path, result_dir: Path, *, run_kind: str) -> dict[str, Any]:
    summary_path = result_dir / "summary.json"
    results_path = result_dir / "results.jsonl"
    partial_path = result_dir / "results.partial.jsonl"
    summary, summary_error = safe_read_json(summary_path)
    rows: list[dict[str, Any]] = []
    row_error = None
    if results_path.exists():
        try:
            rows = read_jsonl_rows(results_path)
        except (OSError, json.JSONDecodeError, ValueError) as exc:
            row_error = str(exc)

    record: dict[str, Any] = {
        "benchmark": "agentic_math",
        "model_slug": model_dir.name,
        "run_kind": run_kind,
        "result_dir": path_display(result_dir),
        "summary_path": path_display(summary_path) if summary_path.exists() else "",
        "results_path": path_display(results_path) if results_path.exists() else "",
        "partial_results_path": path_display(partial_path) if partial_path.exists() else "",
        "expected_total": AGENTIC_MATH_EXPECTED_TOTAL,
        "row_count": len(rows),
        "partial_row_count": jsonl_count(partial_path),
        "complete_local": False,
        "formalization_status": "missing_results",
        "reloggable_to_wandb": False,
        "nemoclaw_session_audit_present": False,
        "nemoclaw_session_audit_valid": False,
        "nemoclaw_session_audit_errors": [],
        "warnings": [],
        "errors": [],
    }
    if summary_error and summary_error != "missing":
        record["errors"].append(f"summary_json_error: {summary_error}")
    if row_error:
        record["errors"].append(f"results_jsonl_error: {row_error}")
    if isinstance(summary, dict):
        record.update(
            {
                "total_instances": summary.get("total_instances"),
                "answered_instances": summary.get("answered_instances"),
                "correct_instances": summary.get("correct_instances"),
                "incorrect_instances": summary.get("incorrect_instances"),
                "accuracy": summary.get("accuracy"),
                "model": summary.get("model"),
                "thinking": summary.get("thinking"),
                "dry_run": bool(summary.get("dry_run")),
            }
        )
        row_correct = sum(1 for row in rows if row.get("correct") is True)
        row_incorrect = sum(1 for row in rows if row.get("correct") is not True)
        row_answered = sum(1 for row in rows if row.get("predicted_answer") not in (None, ""))
        consistency_errors: list[str] = []
        if summary.get("total_instances") != len(rows):
            consistency_errors.append("total_instances does not match results.jsonl rows")
        if summary.get("correct_instances") != row_correct:
            consistency_errors.append("correct_instances does not match row correct count")
        if summary.get("incorrect_instances") != row_incorrect:
            consistency_errors.append("incorrect_instances does not match row incorrect count")
        if summary.get("answered_instances") != row_answered:
            consistency_errors.append("answered_instances does not match row answered count")
        expected_accuracy = row_correct / len(rows) if rows else 0.0
        if rows and not numbers_equal(summary.get("accuracy"), expected_accuracy):
            consistency_errors.append("accuracy does not match row correct/total")
        record["errors"].extend(consistency_errors)
        record["complete_local"] = bool(
            run_kind == "final"
            and not record["dry_run"]
            and not consistency_errors
            and summary.get("total_instances") == AGENTIC_MATH_EXPECTED_TOTAL
            and len(rows) == AGENTIC_MATH_EXPECTED_TOTAL
        )
        audit_issues = agentic_math_local_nemoclaw_audit_issues(
            summary=summary,
            rows=rows,
            expected_total=AGENTIC_MATH_EXPECTED_TOTAL,
        )
        record["nemoclaw_session_audit_errors"] = audit_issues
        record["nemoclaw_session_audit_present"] = not any(
            issue.startswith("summary.json missing NeMoClaw audit keys")
            for issue in audit_issues
        )
        record["nemoclaw_session_audit_valid"] = not audit_issues
    if record["complete_local"]:
        if record["nemoclaw_session_audit_valid"]:
            record["formalization_status"] = LOCAL_COMPLETE_NEEDS_WANDB_RELOG
            record["reloggable_to_wandb"] = True
        else:
            record["formalization_status"] = LOCAL_COMPLETE_MISSING_NEMOCLAW_AUDIT
            record["relog_blocked_reason"] = "missing_or_invalid_nemoclaw_session_audit"
    elif record["partial_row_count"] or record["row_count"]:
        record["formalization_status"] = (
            "probe_not_reloggable" if run_kind != "final" else "partial_not_reloggable"
        )
    return record


def discover_agentic_math(output_root: Path) -> list[dict[str, Any]]:
    root = output_root / "agentic_math"
    if not root.exists():
        return []
    records: list[dict[str, Any]] = []
    for model_dir in sorted(path for path in root.iterdir() if path.is_dir()):
        final_dir = model_dir / "openclaw"
        if final_dir.exists():
            records.append(agentic_math_record(model_dir, final_dir, run_kind="final"))
        for result_dir in sorted(path for path in model_dir.iterdir() if path.is_dir()):
            if result_dir.name in {"openclaw", "logs", "failed_runs"}:
                continue
            if (result_dir / "summary.json").exists() or (result_dir / "results.jsonl").exists():
                records.append(agentic_math_record(model_dir, result_dir, run_kind="probe"))
    return records


def swe_official_summary(model_dir: Path) -> tuple[dict[str, Any] | None, Path | None]:
    candidates = sorted(model_dir.glob("official_eval*/summary.json"))
    for path in reversed(candidates):
        payload, error = safe_read_json(path)
        if payload is not None:
            return payload, path
    return None, None


def discover_agentic_swe(output_root: Path) -> list[dict[str, Any]]:
    root = output_root / "swebench_pro"
    if not root.exists():
        return []
    records: list[dict[str, Any]] = []
    for model_dir in sorted(path for path in root.iterdir() if path.is_dir()):
        openclaw = model_dir / "openclaw"
        patches_path = openclaw / "patches.json"
        patches: list[Any] = []
        patch_error = None
        if patches_path.exists():
            try:
                loaded = json.loads(patches_path.read_text(encoding="utf-8"))
                if isinstance(loaded, list):
                    patches = loaded
                else:
                    patch_error = "patches.json is not a list"
            except (OSError, json.JSONDecodeError) as exc:
                patch_error = str(exc)
        patch_records = list(openclaw.glob("*/patch_record.json")) if openclaw.exists() else []
        official, official_path = swe_official_summary(model_dir)
        record: dict[str, Any] = {
            "benchmark": "agentic_swe",
            "model_slug": model_dir.name,
            "run_kind": "final",
            "result_dir": path_display(model_dir),
            "patches_path": path_display(patches_path) if patches_path.exists() else "",
            "official_summary_path": path_display(official_path) if official_path else "",
            "expected_total": AGENTIC_SWE_EXPECTED_TOTAL,
            "patch_count": len(patches) if patches else len(patch_records),
            "patch_record_count": len(patch_records),
            "row_count": official.get("total_instances") if isinstance(official, dict) else len(patches),
            "complete_local": False,
            "formalization_status": "missing_results",
            "reloggable_to_wandb": False,
            "nemoclaw_session_audit_present": False,
            "nemoclaw_session_audit_valid": False,
            "nemoclaw_session_audit_errors": [],
            "warnings": [],
            "errors": [],
        }
        if patch_error:
            record["errors"].append(f"patches_json_error: {patch_error}")
        if isinstance(official, dict):
            record.update(
                {
                    "total_instances": official.get("total_instances"),
                    "resolved_instances": official.get("resolved_instances"),
                    "unresolved_instances": official.get("unresolved_instances"),
                    "pass_at_1": official.get("pass_at_1"),
                }
            )
            record["complete_local"] = bool(
                official.get("total_instances") == AGENTIC_SWE_EXPECTED_TOTAL
            )
            audit_issues = agentic_swe_local_nemoclaw_audit_issues(
                official=official,
                patch_rows=patches,
                expected_total=AGENTIC_SWE_EXPECTED_TOTAL,
            )
            record["nemoclaw_session_audit_errors"] = audit_issues
            record["nemoclaw_session_audit_present"] = any(
                isinstance(row, dict) and isinstance(row.get("nemoclaw_session_audit"), dict)
                for row in patches
            )
            record["nemoclaw_session_audit_valid"] = not audit_issues
        if record["complete_local"]:
            if record["nemoclaw_session_audit_valid"]:
                record["formalization_status"] = LOCAL_COMPLETE_NEEDS_WANDB_RELOG
                record["reloggable_to_wandb"] = True
            else:
                record["formalization_status"] = LOCAL_COMPLETE_MISSING_NEMOCLAW_AUDIT
                record["relog_blocked_reason"] = "missing_or_invalid_nemoclaw_session_audit"
        elif record["patch_count"]:
            record["formalization_status"] = "partial_not_reloggable"
            if record["patch_count"] >= AGENTIC_SWE_EXPECTED_TOTAL and not official:
                record["warnings"].append("patches exist but full official evaluation summary is missing")
        records.append(record)
    return records


def discover_taiwan_full(output_root: Path) -> list[dict[str, Any]]:
    root = output_root / "provisional_leaderboard"
    if not root.exists():
        return []

    leaderboard_path = root / "leaderboard.csv"
    unit_scores_path = root / "unit_scores.csv"
    run_status_path = root / "run_status.csv"
    leaderboard_rows = read_csv_rows(leaderboard_path)
    unit_rows = read_csv_rows(unit_scores_path)
    status_rows = read_csv_rows(run_status_path)
    status_by_run_id = {
        row.get("run_id", ""): row
        for row in status_rows
        if isinstance(row.get("run_id"), str) and row.get("run_id")
    }
    status_by_slug = {
        row.get("slug", ""): row
        for row in status_rows
        if isinstance(row.get("slug"), str) and row.get("slug")
    }

    records: list[dict[str, Any]] = []
    for row in leaderboard_rows:
        run_id = str(row.get("source_run_id") or row.get("run_id") or "").strip()
        slug = str(row.get("slug") or row.get("model_name") or run_id or "taiwan_full").strip()
        status = status_by_run_id.get(run_id) or status_by_slug.get(slug) or {}
        status_value = status.get("status") or ("ok" if run_id else "missing_run")
        complete = bool(run_id and status_value == "ok")
        record: dict[str, Any] = {
            "benchmark": FULL_BENCHMARK_ID,
            "model_slug": slug,
            "run_kind": "provisional_wandb_full",
            "result_dir": path_display(root),
            "leaderboard_path": path_display(leaderboard_path) if leaderboard_path.exists() else "",
            "unit_scores_path": path_display(unit_scores_path) if unit_scores_path.exists() else "",
            "run_status_path": path_display(run_status_path) if run_status_path.exists() else "",
            "run_id": run_id,
            "run_name": row.get("source_run_name") or status.get("run_name"),
            "run_url": row.get("source_run_url") or status.get("url"),
            "row_count": 1,
            "unit_row_count": len(unit_rows),
            "status": status_value,
            "complete_local": complete,
            "formalization_status": "local_complete_needs_wandb_relog"
            if complete
            else "partial_not_reloggable",
            "warnings": [],
            "errors": [],
        }
        records.append(record)

    if not records and (leaderboard_path.exists() or run_status_path.exists()):
        records.append(
            {
                "benchmark": FULL_BENCHMARK_ID,
                "model_slug": "provisional_leaderboard",
                "run_kind": "provisional_wandb_full",
                "result_dir": path_display(root),
                "leaderboard_path": path_display(leaderboard_path)
                if leaderboard_path.exists()
                else "",
                "unit_scores_path": path_display(unit_scores_path)
                if unit_scores_path.exists()
                else "",
                "run_status_path": path_display(run_status_path)
                if run_status_path.exists()
                else "",
                "row_count": 0,
                "unit_row_count": len(unit_rows),
                "run_status_count": len(status_rows),
                "complete_local": False,
                "formalization_status": "partial_not_reloggable",
                "warnings": ["no completed Taiwan full W&B run rows in provisional leaderboard"],
                "errors": [],
            }
        )
    return records


def load_completions(completion_dir: Path) -> list[dict[str, Any]]:
    if not completion_dir.exists():
        return []
    completions: list[dict[str, Any]] = []
    for path in sorted(completion_dir.glob("*.json")):
        payload, error = safe_read_json(path)
        if payload is None:
            completions.append(
                {
                    "path": path_display(path),
                    "ok": False,
                    "error": error or "invalid JSON",
                }
            )
            continue
        payload = dict(payload)
        payload["path"] = path_display(path)
        payload["metrics"] = completion_metrics(payload)
        schema_issues = completion_schema_issues(payload)
        payload["schema_current"] = not schema_issues
        payload["schema_current_issues"] = schema_issues
        payload["verification_schema_version"] = payload.get("verification_schema_version")
        payload["generated_at_present"] = isinstance(payload.get("generated_at"), (int, float))
        payload["observed_evidence_present"] = isinstance(payload.get("observed_evidence"), dict) and bool(
            payload.get("observed_evidence")
        )
        completions.append(payload)
    return completions


def attach_wandb_formalization(
    records: list[dict[str, Any]],
    completions: list[dict[str, Any]],
) -> None:
    def completion_identity(completion: dict[str, Any]) -> tuple[str, str, str]:
        return (
            str(completion.get("entity") or "").strip(),
            str(completion.get("project") or "").strip(),
            str(completion.get("run_id") or "").strip(),
        )

    def select_single_completion(
        candidates: list[dict[str, Any]],
    ) -> tuple[dict[str, Any] | None, list[str]]:
        if len(candidates) == 1:
            return candidates[0], []

        identities = {completion_identity(completion) for completion in candidates}
        if len(identities) == 1 and all(identities.pop()):
            sorted_candidates = sorted(
                candidates,
                key=lambda completion: (
                    float(completion.get("generated_at") or 0),
                    str(completion.get("path") or ""),
                ),
                reverse=True,
            )
            return sorted_candidates[0], [
                str(completion.get("path") or "")
                for completion in sorted_candidates
                if completion is not sorted_candidates[0]
            ]

        return None, []

    complete_records = [record for record in records if record.get("complete_local")]
    for record in complete_records:
        benchmark_completions = [
            completion
            for completion in completions
            if completion.get("ok") is True
            and completion.get("schema_current") is True
            and completion.get("benchmark") == record.get("benchmark")
            and metrics_match(record, completion)
        ]
        model_matched = [
            completion for completion in benchmark_completions if model_matches(record, completion)
        ]
        selected = model_matched
        if not selected and len(benchmark_completions) == 1:
            selected = benchmark_completions
            record["warnings"].append(
                "W&B completion matched by benchmark/metrics only; completion JSON has no source path"
            )
        completion, duplicate_paths = select_single_completion(selected)
        if completion is not None:
            record["formalization_status"] = "formalized_wandb_complete"
            record["wandb_completion"] = {
                "path": completion.get("path"),
                "entity": completion.get("entity"),
                "project": completion.get("project"),
                "run_id": completion.get("run_id"),
                "run_name": completion.get("run_name"),
                "generated_at": completion.get("generated_at"),
                "verification_schema_version": completion.get("verification_schema_version"),
                "schema_current": completion.get("schema_current"),
                "observed_evidence_present": completion.get("observed_evidence_present"),
            }
            if len(selected) > 1:
                record["matching_wandb_completion_paths"] = [
                    completion.get("path") for completion in selected
                ]
                record["deduped_wandb_completion_paths"] = duplicate_paths
                record["warnings"].append(
                    "multiple W&B completion JSONs matched the same W&B entity/project/run_id; selected newest generated_at"
                )
        elif len(selected) > 1:
            record["formalization_status"] = "ambiguous_wandb_completion"
            record["errors"].append("multiple W&B completion JSONs match this local result")
            record["matching_wandb_completion_paths"] = [
                completion.get("path") for completion in selected
            ]


def build_audit(
    *,
    output_root: Path,
    completion_dir: Path,
) -> dict[str, Any]:
    records = [
        *discover_agentic_math(output_root),
        *discover_agentic_swe(output_root),
        *discover_taiwan_full(output_root),
    ]
    completions = load_completions(completion_dir)
    attach_wandb_formalization(records, completions)
    for record in records:
        dry_run_command = relog_dry_run_command(record)
        if dry_run_command:
            record["relog_dry_run_plan_json"] = relog_dry_run_plan_json(record)
            record["relog_dry_run_command"] = dry_run_command
    formalized = [
        record for record in records if record.get("formalization_status") == "formalized_wandb_complete"
    ]
    unformalized = [
        record
        for record in records
        if record.get("complete_local")
        and record.get("formalization_status") != "formalized_wandb_complete"
    ]
    for record in unformalized:
        if record.get("formalization_status") == LOCAL_COMPLETE_MISSING_NEMOCLAW_AUDIT:
            record["relog_dry_run_plan_json"] = ""
            record["relog_dry_run_command"] = ""
            record["relog_command"] = ""
            record["verify_command"] = ""
            record["rerun_required"] = True
            record["rerun_required_reason"] = (
                "Local complete result is missing valid NeMoClaw session audit evidence; "
                "rerun the benchmark with NeMoClaw-native tracing before W&B formalization."
            )
        else:
            record["relog_command"] = relog_command(record)
            record["verify_command"] = verifier_command(
                record,
                run_id=str(record.get("run_id") or "RUN_ID"),
            )
        dry_run_command = relog_dry_run_command(record)
        if dry_run_command:
            record["relog_dry_run_plan_json"] = relog_dry_run_plan_json(record)
            record["relog_dry_run_command"] = dry_run_command
    partial = [
        record for record in records if record.get("formalization_status") in {"partial_not_reloggable", "probe_not_reloggable"}
    ]
    return {
        "ok": not unformalized,
        "status": "passed" if not unformalized else "unformalized_complete_results",
        "generated_at": time.time(),
        "output_root": path_display(output_root),
        "completion_dir": path_display(completion_dir),
        "summary": {
            "record_count": len(records),
            "complete_local_count": len([record for record in records if record.get("complete_local")]),
            "formalized_wandb_complete_count": len(formalized),
            "unformalized_complete_count": len(unformalized),
            "nemoclaw_audit_blocked_complete_count": len(
                [
                    record
                    for record in unformalized
                    if record.get("formalization_status") == LOCAL_COMPLETE_MISSING_NEMOCLAW_AUDIT
                ]
            ),
            "partial_or_probe_count": len(partial),
            "wandb_completion_json_count": len(completions),
        },
        "remediation_commands": [
            command
            for record in unformalized
            for command in (
                record.get("relog_dry_run_command"),
                record.get("relog_command"),
                record.get("verify_command"),
            )
            if isinstance(command, str) and command
        ],
        "formalized_records": formalized,
        "unformalized_complete_records": unformalized,
        "partial_records": partial,
        "records": records,
        "wandb_completion_records": [
            {
                "path": completion.get("path"),
                "ok": bool(completion.get("ok")),
                "benchmark": completion.get("benchmark"),
                "entity": completion.get("entity"),
                "project": completion.get("project"),
                "run_id": completion.get("run_id"),
                "run_name": completion.get("run_name"),
                "generated_at": completion.get("generated_at"),
                "verification_schema_version": completion.get("verification_schema_version"),
                "schema_current": completion.get("schema_current"),
                "schema_current_issues": completion.get("schema_current_issues"),
                "observed_evidence_present": completion.get("observed_evidence_present"),
                "metrics": completion.get("metrics"),
                "error": completion.get("error"),
            }
            for completion in completions
        ],
    }


def summary_markdown(audit: dict[str, Any]) -> str:
    summary = audit.get("summary") if isinstance(audit.get("summary"), dict) else {}
    lines = [
        "# Taiwan Existing Results Audit",
        "",
        f"- Status: `{audit.get('status')}`",
        f"- OK: `{audit.get('ok')}`",
        f"- Records: `{summary.get('record_count', 0)}`",
        f"- Complete local: `{summary.get('complete_local_count', 0)}`",
        f"- Formalized in W&B: `{summary.get('formalized_wandb_complete_count', 0)}`",
        f"- Unformalized complete: `{summary.get('unformalized_complete_count', 0)}`",
        f"- NeMoClaw-audit blocked complete: `{summary.get('nemoclaw_audit_blocked_complete_count', 0)}`",
        f"- Partial/probe: `{summary.get('partial_or_probe_count', 0)}`",
        "",
        "## Complete Local Records",
        "",
        "| Benchmark | Model | Status | W&B run | Result dir |",
        "| --- | --- | --- | --- | --- |",
    ]
    complete_records = [
        record for record in audit.get("records", []) if isinstance(record, dict) and record.get("complete_local")
    ]
    for record in complete_records:
        wandb_completion = record.get("wandb_completion")
        run_id = wandb_completion.get("run_id") if isinstance(wandb_completion, dict) else ""
        lines.append(
            "| {benchmark} | {model} | {status} | {run_id} | `{result_dir}` |".format(
                benchmark=record.get("benchmark"),
                model=record.get("model_slug"),
                status=record.get("formalization_status"),
                run_id=run_id or "",
                result_dir=record.get("result_dir"),
            )
        )
    lines.extend(
        [
            "",
            "## Unformalized Complete Records",
            "",
        "| Benchmark | Model | Status | Action | Dry-run command | Relog command | Verify command |",
        "| --- | --- | --- | --- | --- | --- | --- |",
        ]
    )
    for record in audit.get("unformalized_complete_records", []):
        if not isinstance(record, dict):
            continue
        lines.append(
            "| {benchmark} | {model} | {status} | {action} | `{dry_run}` | `{relog}` | `{verify}` |".format(
                benchmark=record.get("benchmark"),
                model=record.get("model_slug"),
                status=record.get("formalization_status"),
                action=record.get("rerun_required_reason", ""),
                dry_run=record.get("relog_dry_run_command", ""),
                relog=record.get("relog_command", ""),
                verify=record.get("verify_command", ""),
            )
        )
    lines.extend(
        [
            "",
            "## Partial / Probe Records",
            "",
            "| Benchmark | Model | Status | Count | Result dir |",
            "| --- | --- | --- | --- | --- |",
        ]
    )
    for record in audit.get("partial_records", []):
        if not isinstance(record, dict):
            continue
        count = record.get("partial_row_count", record.get("patch_count", record.get("row_count", "")))
        lines.append(
            "| {benchmark} | {model} | {status} | {count} | `{result_dir}` |".format(
                benchmark=record.get("benchmark"),
                model=record.get("model_slug"),
                status=record.get("formalization_status"),
                count=count,
                result_dir=record.get("result_dir"),
            )
        )
    return "\n".join(lines) + "\n"


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--wandb-completion-dir", type=Path, default=DEFAULT_WANDB_COMPLETION_DIR)
    parser.add_argument("--json", type=Path)
    parser.add_argument("--markdown", type=Path)
    parser.add_argument("--fail-on-unformalized", action="store_true")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    audit = build_audit(
        output_root=repo_path(args.output_root),
        completion_dir=repo_path(args.wandb_completion_dir),
    )
    if args.json:
        write_json(repo_path(args.json), audit)
    if args.markdown:
        markdown_path = repo_path(args.markdown)
        markdown_path.parent.mkdir(parents=True, exist_ok=True)
        markdown_path.write_text(summary_markdown(audit), encoding="utf-8")
    print(json.dumps(audit, ensure_ascii=False, indent=2))
    if args.fail_on_unformalized and not audit["ok"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
