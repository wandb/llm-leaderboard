#!/usr/bin/env python3
"""
Verify that a Taiwan leaderboard benchmark is complete in W&B.

This script is intentionally read-only. It checks the W&B run state, required
summary metrics, required W&B tables, and result artifacts for benchmark runs
where "complete" must mean "logged to W&B", not just "files exist locally".
"""

from __future__ import annotations

import argparse
import json
import math
import os
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import wandb
from omegaconf import OmegaConf


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_ENV_FILE = REPO_ROOT / ".env"
VERIFICATION_SCHEMA_VERSION = 1
WANDB_API_TIMEOUT_SECONDS = 60
WANDB_QUERY_SOURCE_KIND = "wandb_sdk"
AGENTIC_MATH_OUTPUT_TABLE_REQUIRED_COLUMNS = (
    "nemoclaw_session_audit_ok",
    "nemoclaw_session_audit",
    "conversation_order_ok",
    "conversation_order",
    "tool_policy_ok",
    "tool_policy_violations",
    "weave_sidecar_ok",
    "weave_sidecar",
    "openclaw_result_path",
    "openclaw_invocation_path",
    "openclaw_invocation_sha256",
    "openclaw_command_sha256",
)
AGENTIC_SWE_OUTPUT_TABLE_REQUIRED_COLUMNS = (
    "nemoclaw_session_audit_ok",
    "nemoclaw_session_audit_required",
    "conversation_order_ok",
    "conversation_order",
    "tool_policy_ok",
    "tool_policy_violations",
    "weave_sidecar_ok",
    "weave_sidecar",
    "openclaw_result_path",
    "openclaw_invocation_path",
    "openclaw_invocation_sha256",
    "openclaw_command_sha256",
)
OPENCLAW_INVOCATION_EVIDENCE_COLUMNS = (
    "openclaw_result_path",
    "openclaw_invocation_path",
    "openclaw_invocation_sha256",
    "openclaw_command_sha256",
)
OPENCLAW_INVOCATION_HASH_COLUMNS = (
    "openclaw_invocation_sha256",
    "openclaw_command_sha256",
)
AGENTIC_ROW_TRUE_COLUMNS = (
    "nemoclaw_session_audit_ok",
    "conversation_order_ok",
    "tool_policy_ok",
    "weave_sidecar_ok",
)
AGENTIC_ROW_EMPTY_LIST_COLUMNS = (
    "tool_policy_violations",
)
AGENTIC_ROW_DICT_OK_COLUMNS = (
    "nemoclaw_session_audit",
    "conversation_order",
    "weave_sidecar",
)


@dataclass(frozen=True)
class BenchmarkSpec:
    id: str
    leaderboard_table: str
    output_table: str
    total_metric: str
    answered_metric: str | None = None
    correct_metric: str | None = None
    accuracy_metric: str | None = None
    result_artifact_type: str | None = None
    production_artifact_alias_required: bool = False
    nemoclaw_audit_required_metric: str | None = None
    nemoclaw_audit_passed_metric: str | None = None
    nemoclaw_audit_failed_metric: str | None = None
    output_table_required_columns: tuple[str, ...] = ()


BENCHMARK_SPECS: dict[str, BenchmarkSpec] = {
    "agentic_math": BenchmarkSpec(
        id="agentic_math",
        leaderboard_table="agentic_math_leaderboard_table",
        output_table="agentic_math_output_table",
        total_metric="agentic_math/total_instances",
        answered_metric="agentic_math/answered_instances",
        correct_metric="agentic_math/correct_instances",
        accuracy_metric="agentic_math/accuracy",
        result_artifact_type="evaluation-results",
        production_artifact_alias_required=True,
        nemoclaw_audit_required_metric="agentic_math/nemoclaw_session_audit_required_instances",
        nemoclaw_audit_passed_metric="agentic_math/nemoclaw_session_audit_passed_instances",
        nemoclaw_audit_failed_metric="agentic_math/nemoclaw_session_audit_failed_instances",
        output_table_required_columns=AGENTIC_MATH_OUTPUT_TABLE_REQUIRED_COLUMNS,
    ),
    "agentic_swe": BenchmarkSpec(
        id="agentic_swe",
        leaderboard_table="agentic_swe_leaderboard_table",
        output_table="agentic_swe_output_table",
        total_metric="agentic_swe/total_instances",
        correct_metric="agentic_swe/resolved_instances",
        accuracy_metric="agentic_swe/pass_at_1",
        result_artifact_type="evaluation-results",
        production_artifact_alias_required=True,
        nemoclaw_audit_required_metric="agentic_swe/nemoclaw_session_audit_required_patches",
        nemoclaw_audit_passed_metric="agentic_swe/nemoclaw_session_audit_passed_patches",
        nemoclaw_audit_failed_metric="agentic_swe/nemoclaw_session_audit_failed_patches",
        output_table_required_columns=AGENTIC_SWE_OUTPUT_TABLE_REQUIRED_COLUMNS,
    ),
}

FULL_BENCHMARK_ID = "taiwan_full"
DEFAULT_TAXONOMY_PATH = Path("taxonomies/nejumi45_taiwan.yaml")
AGGREGATE_TABLES = (
    "taiwan_leaderboard_table",
    "taiwan_unit_scores_table",
    "taiwan_glp_radar_table",
    "taiwan_alt_radar_table",
)


def load_env_file(env: dict[str, str], path: Path | None) -> tuple[dict[str, str], bool]:
    if path is None:
        return env, False
    expanded = path.expanduser()
    if not expanded.exists():
        return env, False
    loaded = False
    for raw in expanded.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        if key and key not in env:
            env[key] = value.strip().strip('"').strip("'")
            loaded = True
    return env, loaded


def _as_summary_dict(run: Any) -> dict[str, Any]:
    summary = getattr(run, "summary_metrics", None)
    if summary is None:
        summary = getattr(run, "summary", {})
    return dict(summary)


def _as_config_dict(run: Any) -> dict[str, Any]:
    config = getattr(run, "config", None)
    if config is None:
        return {}
    try:
        return dict(config)
    except (TypeError, ValueError):
        return {}


def _metric(summary: dict[str, Any], key: str) -> Any:
    if key in summary:
        return summary[key]
    dotted_key = key.replace("/", ".")
    if dotted_key in summary:
        return summary[dotted_key]
    return None


def _table_nrows(summary: dict[str, Any], key: str) -> int | None:
    value = summary.get(key)
    if not isinstance(value, dict):
        return None
    nrows = value.get("nrows")
    if nrows is None:
        return None
    try:
        return int(nrows)
    except (TypeError, ValueError):
        return None


def _table_check(summary: dict[str, Any], key: str, *, name: str) -> dict[str, Any]:
    nrows = _table_nrows(summary, key)
    if nrows is None:
        return _fail_check(
            name,
            f"missing or invalid W&B table summary for {key}",
            table_name=key,
        )
    if nrows < 1:
        return _fail_check(name, f"{key} has no rows", table_name=key, nrows=nrows)
    return _ok_check(name, f"{key} has rows", table_name=key, nrows=nrows)


def _string_list(value: Any) -> list[str] | None:
    if not isinstance(value, list):
        return None
    result = [str(item) for item in value if isinstance(item, str) and item]
    return result if len(result) == len(value) else None


def _table_columns_from_payload(payload: Any) -> list[str] | None:
    if not isinstance(payload, dict):
        return None
    return _string_list(payload.get("columns"))


def _table_data_rows_from_payload(payload: Any) -> tuple[list[dict[str, Any]] | None, str | None]:
    if not isinstance(payload, dict):
        return None, "W&B table payload is not a JSON object"
    columns = _table_columns_from_payload(payload)
    data = payload.get("data")
    if columns is None:
        return None, "W&B table payload has no valid columns"
    if not isinstance(data, list):
        return None, "W&B table payload has no valid data rows"
    rows: list[dict[str, Any]] = []
    for index, raw_row in enumerate(data, start=1):
        if isinstance(raw_row, dict):
            rows.append(raw_row)
            continue
        if isinstance(raw_row, list):
            rows.append({column: raw_row[i] if i < len(raw_row) else None for i, column in enumerate(columns)})
            continue
        return None, f"W&B table data row {index} is not a list or object"
    return rows, None


def _is_sha256(value: Any) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(char in "0123456789abcdefABCDEF" for char in value)
    )


def _download_wandb_table_json(run: Any, table_path: str) -> Any:
    file_getter = getattr(run, "file", None)
    if not callable(file_getter):
        raise RuntimeError("run.file API is not available")
    with tempfile.TemporaryDirectory(prefix="taiwan-wandb-table-") as tmpdir:
        file_ref = file_getter(table_path)
        downloaded = file_ref.download(root=tmpdir, replace=True)
        downloaded_name = getattr(downloaded, "name", None)
        local_path = Path(downloaded_name) if downloaded_name else Path(tmpdir) / table_path
        return json.loads(local_path.read_text(encoding="utf-8"))


def _download_table_payload_from_summary(
    run: Any,
    summary: dict[str, Any],
    key: str,
) -> tuple[Any | None, str, str | None]:
    value = summary.get(key)
    if not isinstance(value, dict):
        return None, "summary", f"missing or invalid W&B table summary for {key}"
    table_path = value.get("path")
    if not isinstance(table_path, str) or not table_path:
        return None, "summary", f"{key} table summary has no path for row-level evidence"
    try:
        return _download_wandb_table_json(run, table_path), "wandb_file", None
    except (OSError, json.JSONDecodeError, RuntimeError, AttributeError) as exc:
        return None, "wandb_file", f"could not read W&B table file for {key}: {exc}"


def _table_columns(
    run: Any,
    summary: dict[str, Any],
    key: str,
) -> tuple[list[str] | None, str, str | None]:
    value = summary.get(key)
    if not isinstance(value, dict):
        return None, "summary", f"missing or invalid W&B table summary for {key}"
    columns = _table_columns_from_payload(value)
    if columns is not None:
        return columns, "summary", None
    table_path = value.get("path")
    if not isinstance(table_path, str) or not table_path:
        return None, "summary", f"{key} table summary has no columns or path"
    try:
        payload = _download_wandb_table_json(run, table_path)
    except (OSError, json.JSONDecodeError, RuntimeError, AttributeError) as exc:
        return None, "wandb_file", f"could not read W&B table file for {key}: {exc}"
    columns = _table_columns_from_payload(payload)
    if columns is None:
        return None, "wandb_file", f"W&B table file for {key} has no valid columns"
    return columns, "wandb_file", None


def _output_table_columns_check(
    run: Any,
    summary: dict[str, Any],
    spec: BenchmarkSpec,
) -> dict[str, Any] | None:
    required_columns = list(spec.output_table_required_columns)
    if not required_columns:
        return None
    columns, source, error = _table_columns(run, summary, spec.output_table)
    if columns is None:
        return _fail_check(
            "output_table_columns",
            error or f"could not inspect columns for {spec.output_table}",
            table_name=spec.output_table,
            required_columns=required_columns,
            columns=None,
            missing_columns=required_columns,
            source=source,
        )
    missing_columns = [column for column in required_columns if column not in columns]
    if missing_columns:
        return _fail_check(
            "output_table_columns",
            f"{spec.output_table} is missing required observability columns",
            table_name=spec.output_table,
            required_columns=required_columns,
            columns=columns,
            missing_columns=missing_columns,
            source=source,
        )
    return _ok_check(
        "output_table_columns",
        f"{spec.output_table} contains required observability columns",
        table_name=spec.output_table,
        required_columns=required_columns,
        columns=columns,
        missing_columns=[],
        source=source,
    )


def _output_table_invocation_evidence_check(
    run: Any,
    summary: dict[str, Any],
    spec: BenchmarkSpec,
    *,
    expected_rows: int | None,
) -> dict[str, Any] | None:
    required = list(OPENCLAW_INVOCATION_EVIDENCE_COLUMNS)
    if not all(column in spec.output_table_required_columns for column in required):
        return None
    payload, source, error = _download_table_payload_from_summary(run, summary, spec.output_table)
    if payload is None:
        return _fail_check(
            "output_table_invocation_evidence",
            error or f"could not inspect row-level evidence for {spec.output_table}",
            table_name=spec.output_table,
            required_columns=required,
            checked_rows=0,
            invalid_row_count=None,
            invalid_examples=[],
            source=source,
        )
    rows, rows_error = _table_data_rows_from_payload(payload)
    if rows is None:
        return _fail_check(
            "output_table_invocation_evidence",
            rows_error or f"{spec.output_table} has no inspectable rows",
            table_name=spec.output_table,
            required_columns=required,
            checked_rows=0,
            invalid_row_count=None,
            invalid_examples=[],
            source=source,
        )
    invalid_examples: list[dict[str, Any]] = []
    invalid_count = 0
    for index, row in enumerate(rows, start=1):
        missing_or_empty = [
            column
            for column in required
            if not isinstance(row.get(column), str) or not row.get(column)
        ]
        invalid_hashes = [
            column
            for column in OPENCLAW_INVOCATION_HASH_COLUMNS
            if column in row and not _is_sha256(row.get(column))
        ]
        if missing_or_empty or invalid_hashes:
            invalid_count += 1
            if len(invalid_examples) < 5:
                invalid_examples.append(
                    {
                        "row_index": index,
                        "missing_or_empty": missing_or_empty,
                        "invalid_hashes": invalid_hashes,
                    }
                )
    if expected_rows is not None and len(rows) != expected_rows:
        return _fail_check(
            "output_table_invocation_evidence",
            f"{spec.output_table} table JSON row count does not match summary",
            table_name=spec.output_table,
            required_columns=required,
            checked_rows=len(rows),
            expected_rows=expected_rows,
            invalid_row_count=invalid_count,
            invalid_examples=invalid_examples,
            source=source,
        )
    if invalid_count:
        return _fail_check(
            "output_table_invocation_evidence",
            f"{spec.output_table} has rows without valid OpenClaw invocation evidence",
            table_name=spec.output_table,
            required_columns=required,
            checked_rows=len(rows),
            expected_rows=expected_rows,
            invalid_row_count=invalid_count,
            invalid_examples=invalid_examples,
            source=source,
        )
    return _ok_check(
        "output_table_invocation_evidence",
        f"{spec.output_table} rows contain OpenClaw invocation evidence",
        table_name=spec.output_table,
        required_columns=required,
        checked_rows=len(rows),
        expected_rows=expected_rows,
        invalid_row_count=0,
        invalid_examples=[],
        source=source,
    )


def _output_table_row_observability_check(
    run: Any,
    summary: dict[str, Any],
    spec: BenchmarkSpec,
    *,
    expected_rows: int | None,
) -> dict[str, Any] | None:
    required_true = [
        column
        for column in AGENTIC_ROW_TRUE_COLUMNS
        if column in spec.output_table_required_columns
    ]
    required_empty = [
        column
        for column in AGENTIC_ROW_EMPTY_LIST_COLUMNS
        if column in spec.output_table_required_columns
    ]
    required_dict_ok = [
        column
        for column in AGENTIC_ROW_DICT_OK_COLUMNS
        if column in spec.output_table_required_columns
    ]
    if not required_true and not required_empty and not required_dict_ok:
        return None
    payload, source, error = _download_table_payload_from_summary(run, summary, spec.output_table)
    required_columns = required_true + required_empty + required_dict_ok
    if payload is None:
        return _fail_check(
            "output_table_row_observability",
            error or f"could not inspect row observability for {spec.output_table}",
            table_name=spec.output_table,
            required_true_columns=required_true,
            required_empty_list_columns=required_empty,
            required_dict_ok_columns=required_dict_ok,
            required_columns=required_columns,
            checked_rows=0,
            invalid_row_count=None,
            invalid_examples=[],
            source=source,
        )
    rows, rows_error = _table_data_rows_from_payload(payload)
    if rows is None:
        return _fail_check(
            "output_table_row_observability",
            rows_error or f"{spec.output_table} has no inspectable rows",
            table_name=spec.output_table,
            required_true_columns=required_true,
            required_empty_list_columns=required_empty,
            required_dict_ok_columns=required_dict_ok,
            required_columns=required_columns,
            checked_rows=0,
            invalid_row_count=None,
            invalid_examples=[],
            source=source,
        )
    invalid_examples: list[dict[str, Any]] = []
    invalid_count = 0
    for index, row in enumerate(rows, start=1):
        not_true = [column for column in required_true if row.get(column) is not True]
        non_empty_lists = [
            column
            for column in required_empty
            if not isinstance(row.get(column), list) or row.get(column)
        ]
        dict_not_ok = [
            column
            for column in required_dict_ok
            if not isinstance(row.get(column), dict) or row.get(column, {}).get("ok") is not True
        ]
        if not_true or non_empty_lists or dict_not_ok:
            invalid_count += 1
            if len(invalid_examples) < 5:
                invalid_examples.append(
                    {
                        "row_index": index,
                        "not_true": not_true,
                        "non_empty_lists": non_empty_lists,
                        "dict_not_ok": dict_not_ok,
                    }
                )
    if expected_rows is not None and len(rows) != expected_rows:
        return _fail_check(
            "output_table_row_observability",
            f"{spec.output_table} table JSON row count does not match summary",
            table_name=spec.output_table,
            required_true_columns=required_true,
            required_empty_list_columns=required_empty,
            required_dict_ok_columns=required_dict_ok,
            required_columns=required_columns,
            checked_rows=len(rows),
            expected_rows=expected_rows,
            invalid_row_count=invalid_count,
            invalid_examples=invalid_examples,
            source=source,
        )
    if invalid_count:
        return _fail_check(
            "output_table_row_observability",
            f"{spec.output_table} has rows with failed Agentic observability checks",
            table_name=spec.output_table,
            required_true_columns=required_true,
            required_empty_list_columns=required_empty,
            required_dict_ok_columns=required_dict_ok,
            required_columns=required_columns,
            checked_rows=len(rows),
            expected_rows=expected_rows,
            invalid_row_count=invalid_count,
            invalid_examples=invalid_examples,
            source=source,
        )
    return _ok_check(
        "output_table_row_observability",
        f"{spec.output_table} rows passed Agentic observability checks",
        table_name=spec.output_table,
        required_true_columns=required_true,
        required_empty_list_columns=required_empty,
        required_dict_ok_columns=required_dict_ok,
        required_columns=required_columns,
        checked_rows=len(rows),
        expected_rows=expected_rows,
        invalid_row_count=0,
        invalid_examples=[],
        source=source,
    )


def _int_metric(summary: dict[str, Any], key: str | None) -> tuple[int | None, Any]:
    if not key:
        return None, None
    raw = _metric(summary, key)
    try:
        return int(raw), raw
    except (TypeError, ValueError):
        return None, raw


def _artifact_summaries(run: Any) -> list[dict[str, Any]]:
    artifacts = []
    for artifact in run.logged_artifacts():
        artifacts.append(
            {
                "name": getattr(artifact, "name", ""),
                "type": getattr(artifact, "type", ""),
                "aliases": list(getattr(artifact, "aliases", []) or []),
            }
        )
    return artifacts


def _ok_check(name: str, detail: str, **extra: Any) -> dict[str, Any]:
    return {"name": name, "ok": True, "detail": detail, **extra}


def _fail_check(name: str, detail: str, **extra: Any) -> dict[str, Any]:
    return {"name": name, "ok": False, "detail": detail, **extra}


def _verification_header(ok: bool) -> dict[str, Any]:
    status = "passed" if ok else "failed"
    return {
        "schema_version": VERIFICATION_SCHEMA_VERSION,
        "verification_schema_version": VERIFICATION_SCHEMA_VERSION,
        "status": status,
    }


def _config_lookup(config: dict[str, Any], key: str) -> tuple[bool, Any]:
    if key in config:
        return True, config[key]
    current: Any = config
    for part in key.split("."):
        if not isinstance(current, dict) or part not in current:
            return False, None
        current = current[part]
    return True, current


def _values_match(actual: Any, expected: Any) -> bool:
    if actual == expected:
        return True
    if isinstance(expected, str):
        return str(actual) == expected
    return False


def _run_metadata_required_evidence(
    *,
    expected_config: dict[str, Any] | None = None,
    expected_tags: list[str] | None = None,
    expected_group: str | None = None,
    expected_job_type: str | None = None,
) -> dict[str, Any]:
    return {
        "config": [
            {"key": key, "expected": value}
            for key, value in sorted((expected_config or {}).items())
        ],
        "tags": sorted(expected_tags or []),
        "group": expected_group or "",
        "job_type": expected_job_type or "",
    }


def _run_metadata_observed_evidence(
    *,
    run: Any,
    expected_config: dict[str, Any] | None = None,
) -> dict[str, Any]:
    config = _as_config_dict(run)
    observed_config = []
    for key in sorted((expected_config or {}).keys()):
        present, value = _config_lookup(config, key)
        observed_config.append({"key": key, "present": present, "value": value})
    return {
        "config": observed_config,
        "tags": sorted(list(getattr(run, "tags", []) or [])),
        "group": getattr(run, "group", None) or "",
        "job_type": getattr(run, "job_type", None) or "",
    }


def run_metadata_checks(
    run: Any,
    *,
    expected_config: dict[str, Any] | None = None,
    expected_tags: list[str] | None = None,
    expected_group: str | None = None,
    expected_job_type: str | None = None,
) -> list[dict[str, Any]]:
    checks: list[dict[str, Any]] = []
    config = _as_config_dict(run)
    for key, expected in sorted((expected_config or {}).items()):
        present, actual = _config_lookup(config, key)
        if not present:
            checks.append(
                _fail_check(
                    "run_config",
                    f"run.config missing expected key {key}",
                    key=key,
                    expected=expected,
                    present=False,
                )
            )
        elif not _values_match(actual, expected):
            checks.append(
                _fail_check(
                    "run_config",
                    f"run.config {key} does not match expected value",
                    key=key,
                    expected=expected,
                    value=actual,
                    present=True,
                )
            )
        else:
            checks.append(
                _ok_check(
                    "run_config",
                    f"run.config {key} matches expected value",
                    key=key,
                    expected=expected,
                    value=actual,
                    present=True,
                )
            )
    tags = set(getattr(run, "tags", []) or [])
    for tag in sorted(expected_tags or []):
        if tag in tags:
            checks.append(_ok_check("run_tag", "run tag is present", tag=tag))
        else:
            checks.append(_fail_check("run_tag", "run tag is missing", tag=tag))
    if expected_group is not None:
        group = getattr(run, "group", None) or ""
        if group == expected_group:
            checks.append(
                _ok_check("run_group", "run group matches expected value", value=group)
            )
        else:
            checks.append(
                _fail_check(
                    "run_group",
                    "run group does not match expected value",
                    value=group,
                    expected=expected_group,
                )
            )
    if expected_job_type is not None:
        job_type = getattr(run, "job_type", None) or ""
        if job_type == expected_job_type:
            checks.append(
                _ok_check("run_job_type", "run job_type matches expected value", value=job_type)
            )
        else:
            checks.append(
                _fail_check(
                    "run_job_type",
                    "run job_type does not match expected value",
                    value=job_type,
                    expected=expected_job_type,
                )
            )
    return checks


def _merge_run_metadata_evidence(
    result: dict[str, Any],
    run: Any,
    *,
    expected_config: dict[str, Any] | None = None,
    expected_tags: list[str] | None = None,
    expected_group: str | None = None,
    expected_job_type: str | None = None,
) -> dict[str, Any]:
    expected_config = expected_config or {}
    expected_tags = expected_tags or []
    if not expected_config and not expected_tags and expected_group is None and expected_job_type is None:
        return result
    checks = run_metadata_checks(
        run,
        expected_config=expected_config,
        expected_tags=expected_tags,
        expected_group=expected_group,
        expected_job_type=expected_job_type,
    )
    result["checks"].extend(checks)
    result["ok"] = bool(result["ok"]) and all(check["ok"] for check in checks)
    result["required_evidence"]["run_metadata"] = _run_metadata_required_evidence(
        expected_config=expected_config,
        expected_tags=expected_tags,
        expected_group=expected_group,
        expected_job_type=expected_job_type,
    )
    result["observed_evidence"]["run_metadata"] = _run_metadata_observed_evidence(
        run=run,
        expected_config=expected_config,
    )
    return result


def _benchmark_required_evidence(
    spec: BenchmarkSpec,
    *,
    expected_total: int | None,
    require_nemoclaw_session_audit: bool,
) -> dict[str, Any]:
    metrics = [spec.total_metric]
    if spec.answered_metric:
        metrics.append(spec.answered_metric)
    if spec.correct_metric:
        metrics.append(spec.correct_metric)
    if spec.accuracy_metric:
        metrics.append(spec.accuracy_metric)
    artifacts = []
    if spec.result_artifact_type:
        artifacts.append(
            {
                "type": spec.result_artifact_type,
                "required_aliases": ["production"]
                if spec.production_artifact_alias_required
                else [],
            }
        )
    required_evidence = {
        "run_state": "finished",
        "expected_total": expected_total,
        "summary_metrics": metrics,
        "tables": [
            {
                "name": spec.leaderboard_table,
                "row_count": ">=1",
            },
            {
                "name": spec.output_table,
                "row_count": "must equal total metric",
                "required_columns": list(spec.output_table_required_columns),
                "row_observability": "all required audit/tool/order/sidecar row checks must pass",
            },
        ],
        "artifacts": artifacts,
    }
    if require_nemoclaw_session_audit:
        required_evidence["nemoclaw_session_audit"] = {
            "required": True,
            "required_metric": spec.nemoclaw_audit_required_metric,
            "passed_metric": spec.nemoclaw_audit_passed_metric,
            "failed_metric": spec.nemoclaw_audit_failed_metric,
            "expected_required_equals_total": True,
            "expected_passed_equals_total": True,
            "expected_failed": 0,
        }
    else:
        required_evidence["nemoclaw_session_audit"] = {"required": False}
    return required_evidence


def _full_required_evidence(
    taxonomy: dict[str, Any],
    *,
    taxonomy_path: Path,
    num_few_shots: int,
    include_pending: bool,
    require_aggregate: bool,
) -> dict[str, Any]:
    taxonomy_tables: list[dict[str, Any]] = []
    skipped_pending_units: list[str] = []
    for unit in taxonomy["units"]:
        if not unit.get("required", True):
            continue
        if unit.get("pending", False) and not include_pending:
            skipped_pending_units.append(str(unit["id"]))
            continue
        for table_name in _taxonomy_unit_sources(unit, num_few_shots=num_few_shots):
            taxonomy_tables.append(
                {
                    "unit_id": unit["id"],
                    "display_name": unit.get("display_name", unit["id"]),
                    "table_name": table_name,
                    "row_count": ">=1",
                }
            )
    return {
        "run_state": "finished",
        "taxonomy_path": str(taxonomy_path),
        "taxonomy_version": taxonomy.get("version"),
        "num_few_shots": num_few_shots,
        "include_pending": include_pending,
        "skipped_pending_units": skipped_pending_units,
        "taxonomy_tables": taxonomy_tables,
        "aggregate_tables": [
            {"name": table_name, "row_count": ">=1"}
            for table_name in AGGREGATE_TABLES
        ]
        if require_aggregate
        else [],
    }


def _observed_benchmark_evidence(
    checks: list[dict[str, Any]],
    spec: BenchmarkSpec,
    *,
    expected_total: int | None,
) -> dict[str, Any]:
    metric_names = {
        "total_metric": spec.total_metric,
        "answered_metric": spec.answered_metric,
        "correct_metric": spec.correct_metric,
        "accuracy_metric": spec.accuracy_metric,
    }
    table_names = {
        "leaderboard_table": spec.leaderboard_table,
        "output_table": spec.output_table,
    }
    observed: dict[str, Any] = {
        "run_state": None,
        "expected_total": expected_total,
        "summary_metrics": {},
        "tables": [],
        "artifacts": [],
        "nemoclaw_session_audit": {},
    }
    for check in checks:
        name = check.get("name")
        if name == "run_state":
            observed["run_state"] = check.get("state")
            continue
        metric_name = metric_names.get(str(name))
        if metric_name:
            observed["summary_metrics"][metric_name] = {
                "ok": bool(check.get("ok")),
                "value": check.get("value"),
            }
            if "expected" in check:
                observed["summary_metrics"][metric_name]["expected"] = check.get("expected")
            continue
        table_name = table_names.get(str(name))
        if table_name:
            observed["tables"].append(
                {
                    "name": table_name,
                    "ok": bool(check.get("ok")),
                    "nrows": check.get("nrows"),
                    "expected": check.get("expected"),
                }
            )
            continue
        if name == "output_table_columns":
            output_table_row = next(
                (
                    row
                    for row in observed["tables"]
                    if row.get("name") == spec.output_table
                ),
                None,
            )
            if output_table_row is None:
                output_table_row = {
                    "name": spec.output_table,
                    "ok": bool(check.get("ok")),
                    "nrows": None,
                    "expected": None,
                }
                observed["tables"].append(output_table_row)
            output_table_row.update(
                {
                    "columns_ok": bool(check.get("ok")),
                    "columns": check.get("columns"),
                    "required_columns": check.get("required_columns"),
                    "missing_columns": check.get("missing_columns"),
                    "columns_source": check.get("source"),
                }
            )
            continue
        if name == "output_table_invocation_evidence":
            output_table_row = next(
                (
                    row
                    for row in observed["tables"]
                    if row.get("name") == spec.output_table
                ),
                None,
            )
            if output_table_row is None:
                output_table_row = {
                    "name": spec.output_table,
                    "ok": bool(check.get("ok")),
                    "nrows": None,
                    "expected": None,
                }
                observed["tables"].append(output_table_row)
            output_table_row.update(
                {
                    "invocation_evidence_ok": bool(check.get("ok")),
                    "invocation_evidence_source": check.get("source"),
                    "invocation_checked_rows": check.get("checked_rows"),
                    "invocation_expected_rows": check.get("expected_rows"),
                    "invocation_invalid_row_count": check.get("invalid_row_count"),
                    "invocation_invalid_examples": check.get("invalid_examples"),
                }
            )
            continue
        if name == "output_table_row_observability":
            output_table_row = next(
                (
                    row
                    for row in observed["tables"]
                    if row.get("name") == spec.output_table
                ),
                None,
            )
            if output_table_row is None:
                output_table_row = {
                    "name": spec.output_table,
                    "ok": bool(check.get("ok")),
                    "nrows": None,
                    "expected": None,
                }
                observed["tables"].append(output_table_row)
            output_table_row.update(
                {
                    "row_observability_ok": bool(check.get("ok")),
                    "row_observability_source": check.get("source"),
                    "row_observability_checked_rows": check.get("checked_rows"),
                    "row_observability_expected_rows": check.get("expected_rows"),
                    "row_observability_invalid_row_count": check.get("invalid_row_count"),
                    "row_observability_invalid_examples": check.get("invalid_examples"),
                    "row_observability_required_true_columns": check.get("required_true_columns"),
                    "row_observability_required_empty_list_columns": check.get(
                        "required_empty_list_columns"
                    ),
                    "row_observability_required_dict_ok_columns": check.get(
                        "required_dict_ok_columns"
                    ),
                }
            )
            continue
        if name == "result_artifact":
            observed["artifacts"] = check.get("artifacts", [])
            continue
        if name == "nemoclaw_session_audit":
            observed["nemoclaw_session_audit"] = {
                "ok": bool(check.get("ok")),
                "required": check.get("required"),
                "passed": check.get("passed"),
                "failed": check.get("failed"),
                "expected_total": check.get("expected_total"),
                "required_metric": spec.nemoclaw_audit_required_metric,
                "passed_metric": spec.nemoclaw_audit_passed_metric,
                "failed_metric": spec.nemoclaw_audit_failed_metric,
            }
    return observed


def _nemoclaw_session_audit_check(
    summary: dict[str, Any],
    spec: BenchmarkSpec,
    *,
    total: int | None,
    required: bool,
) -> dict[str, Any] | None:
    if not required:
        return None
    metric_names = [
        spec.nemoclaw_audit_required_metric,
        spec.nemoclaw_audit_passed_metric,
        spec.nemoclaw_audit_failed_metric,
    ]
    if not all(metric_names):
        return _fail_check(
            "nemoclaw_session_audit",
            f"{spec.id} does not define required NeMoClaw session audit metrics",
            expected_total=total,
        )
    required_count, required_raw = _int_metric(summary, spec.nemoclaw_audit_required_metric)
    passed_count, passed_raw = _int_metric(summary, spec.nemoclaw_audit_passed_metric)
    failed_count, failed_raw = _int_metric(summary, spec.nemoclaw_audit_failed_metric)
    if required_count is None or passed_count is None or failed_count is None:
        return _fail_check(
            "nemoclaw_session_audit",
            "missing or invalid NeMoClaw session audit summary metrics",
            required=required_raw,
            passed=passed_raw,
            failed=failed_raw,
            expected_total=total,
        )
    if total is not None and required_count != total:
        return _fail_check(
            "nemoclaw_session_audit",
            "NeMoClaw session audit required count does not match total metric",
            required=required_count,
            passed=passed_count,
            failed=failed_count,
            expected_total=total,
        )
    if total is not None and passed_count != total:
        return _fail_check(
            "nemoclaw_session_audit",
            "NeMoClaw session audit passed count does not match total metric",
            required=required_count,
            passed=passed_count,
            failed=failed_count,
            expected_total=total,
        )
    if failed_count != 0:
        return _fail_check(
            "nemoclaw_session_audit",
            "NeMoClaw session audit has failed tasks",
            required=required_count,
            passed=passed_count,
            failed=failed_count,
            expected_total=total,
        )
    return _ok_check(
        "nemoclaw_session_audit",
        "NeMoClaw session audit metrics prove every evaluated task passed",
        required=required_count,
        passed=passed_count,
        failed=failed_count,
        expected_total=total,
    )


def _observed_full_evidence(checks: list[dict[str, Any]]) -> dict[str, Any]:
    observed: dict[str, Any] = {
        "run_state": None,
        "taxonomy_tables": [],
        "aggregate_tables": [],
        "skipped_pending_units": [],
    }
    for check in checks:
        name = check.get("name")
        if name == "run_state":
            observed["run_state"] = check.get("state")
        elif name == "taxonomy_table":
            observed["taxonomy_tables"].append(
                {
                    "unit_id": check.get("unit_id"),
                    "display_name": check.get("display_name"),
                    "table_name": check.get("table_name"),
                    "ok": bool(check.get("ok")),
                    "nrows": check.get("nrows"),
                }
            )
        elif name == "aggregate_table":
            observed["aggregate_tables"].append(
                {
                    "name": check.get("table_name"),
                    "ok": bool(check.get("ok")),
                    "nrows": check.get("nrows"),
                }
            )
        elif name == "taxonomy_unit_pending_skipped":
            observed["skipped_pending_units"].append(
                {
                    "unit_id": check.get("unit_id"),
                    "display_name": check.get("display_name"),
                }
            )
    return observed


def verify_run(
    run: Any,
    spec: BenchmarkSpec,
    *,
    expected_total: int | None = None,
    require_nemoclaw_session_audit: bool = False,
    expected_config: dict[str, Any] | None = None,
    expected_tags: list[str] | None = None,
    expected_group: str | None = None,
    expected_job_type: str | None = None,
) -> dict[str, Any]:
    summary = _as_summary_dict(run)
    checks: list[dict[str, Any]] = []

    state = getattr(run, "state", None)
    if state == "finished":
        checks.append(_ok_check("run_state", "run state is finished", state=state))
    else:
        checks.append(_fail_check("run_state", "run state is not finished", state=state))

    total_raw = _metric(summary, spec.total_metric)
    try:
        total = int(total_raw)
    except (TypeError, ValueError):
        total = None
    if total is None:
        checks.append(
            _fail_check("total_metric", f"missing or invalid {spec.total_metric}", value=total_raw)
        )
    elif expected_total is not None and total != expected_total:
        checks.append(
            _fail_check(
                "total_metric",
                f"{spec.total_metric} does not match expected total",
                value=total,
                expected=expected_total,
            )
        )
    else:
        checks.append(
            _ok_check("total_metric", f"{spec.total_metric} is present", value=total)
        )

    leaderboard_rows = _table_nrows(summary, spec.leaderboard_table)
    if leaderboard_rows is None:
        checks.append(
            _fail_check(
                "leaderboard_table",
                f"missing or invalid W&B table summary for {spec.leaderboard_table}",
            )
        )
    elif leaderboard_rows < 1:
        checks.append(
            _fail_check(
                "leaderboard_table",
                f"{spec.leaderboard_table} has no rows",
                nrows=leaderboard_rows,
            )
        )
    else:
        checks.append(
            _ok_check(
                "leaderboard_table",
                f"{spec.leaderboard_table} has rows",
                nrows=leaderboard_rows,
            )
        )

    output_rows = _table_nrows(summary, spec.output_table)
    if output_rows is None:
        checks.append(
            _fail_check(
                "output_table",
                f"missing or invalid W&B table summary for {spec.output_table}",
            )
        )
    elif total is not None and output_rows != total:
        checks.append(
            _fail_check(
                "output_table",
                f"{spec.output_table} row count does not match total metric",
                nrows=output_rows,
                expected=total,
            )
        )
    else:
        checks.append(
            _ok_check(
                "output_table",
                f"{spec.output_table} row count matches total metric",
                nrows=output_rows,
            )
        )
    output_table_columns_check = _output_table_columns_check(run, summary, spec)
    if output_table_columns_check is not None:
        checks.append(output_table_columns_check)
    output_table_invocation_evidence_check = _output_table_invocation_evidence_check(
        run,
        summary,
        spec,
        expected_rows=output_rows,
    )
    if output_table_invocation_evidence_check is not None:
        checks.append(output_table_invocation_evidence_check)
    output_table_row_observability_check = _output_table_row_observability_check(
        run,
        summary,
        spec,
        expected_rows=output_rows,
    )
    if output_table_row_observability_check is not None:
        checks.append(output_table_row_observability_check)

    if spec.answered_metric:
        answered = _metric(summary, spec.answered_metric)
        try:
            answered_int = int(answered)
        except (TypeError, ValueError):
            answered_int = None
        if answered_int is None:
            checks.append(
                _fail_check(
                    "answered_metric",
                    f"missing or invalid {spec.answered_metric}",
                    value=answered,
                )
            )
        elif total is not None and not 0 <= answered_int <= total:
            checks.append(
                _fail_check(
                    "answered_metric",
                    f"{spec.answered_metric} is outside [0, total]",
                    value=answered_int,
                    total=total,
                )
            )
        else:
            checks.append(
                _ok_check(
                    "answered_metric",
                    f"{spec.answered_metric} is present",
                    value=answered_int,
                )
            )

    correct_int: int | None = None
    if spec.correct_metric:
        correct = _metric(summary, spec.correct_metric)
        try:
            correct_int = int(correct)
        except (TypeError, ValueError):
            correct_int = None
        if correct_int is None:
            checks.append(
                _fail_check(
                    "correct_metric",
                    f"missing or invalid {spec.correct_metric}",
                    value=correct,
                )
            )
        elif total is not None and not 0 <= correct_int <= total:
            checks.append(
                _fail_check(
                    "correct_metric",
                    f"{spec.correct_metric} is outside [0, total]",
                    value=correct_int,
                    total=total,
                )
            )
        else:
            checks.append(
                _ok_check(
                    "correct_metric",
                    f"{spec.correct_metric} is present",
                    value=correct_int,
                )
            )

    if spec.accuracy_metric:
        accuracy_raw = _metric(summary, spec.accuracy_metric)
        try:
            accuracy = float(accuracy_raw)
        except (TypeError, ValueError):
            accuracy = math.nan
        if not math.isfinite(accuracy):
            checks.append(
                _fail_check(
                    "accuracy_metric",
                    f"missing or invalid {spec.accuracy_metric}",
                    value=accuracy_raw,
                )
            )
        elif total is not None and correct_int is not None:
            expected_accuracy = correct_int / total if total else 0.0
            if abs(accuracy - expected_accuracy) > 1e-12:
                checks.append(
                    _fail_check(
                        "accuracy_metric",
                        f"{spec.accuracy_metric} does not equal correct/total",
                        value=accuracy,
                        expected=expected_accuracy,
                    )
                )
            else:
                checks.append(
                    _ok_check(
                        "accuracy_metric",
                        f"{spec.accuracy_metric} equals correct/total",
                        value=accuracy,
                    )
                )
        else:
            checks.append(
                _ok_check(
                    "accuracy_metric",
                    f"{spec.accuracy_metric} is present",
                    value=accuracy,
                )
            )

    nemoclaw_audit_check = _nemoclaw_session_audit_check(
        summary,
        spec,
        total=total,
        required=require_nemoclaw_session_audit,
    )
    if nemoclaw_audit_check is not None:
        checks.append(nemoclaw_audit_check)

    artifacts = _artifact_summaries(run)
    if spec.result_artifact_type:
        matching = [
            artifact
            for artifact in artifacts
            if artifact["type"] == spec.result_artifact_type
        ]
        if not matching:
            checks.append(
                _fail_check(
                    "result_artifact",
                    f"no logged artifact with type {spec.result_artifact_type}",
                    artifact_type=spec.result_artifact_type,
                )
            )
        elif spec.production_artifact_alias_required and not any(
            "production" in artifact["aliases"] for artifact in matching
        ):
            checks.append(
                _fail_check(
                    "result_artifact",
                    "result artifact exists but lacks production alias",
                    artifacts=matching,
                )
            )
        else:
            checks.append(
                _ok_check(
                    "result_artifact",
                    "result artifact is logged",
                    artifacts=matching,
                )
            )

    ok = all(check["ok"] for check in checks)
    result = {
        **_verification_header(ok),
        "ok": ok,
        "benchmark": spec.id,
        "run_id": getattr(run, "id", None),
        "run_name": getattr(run, "name", None),
        "required_evidence": _benchmark_required_evidence(
            spec,
            expected_total=expected_total,
            require_nemoclaw_session_audit=require_nemoclaw_session_audit,
        ),
        "observed_evidence": _observed_benchmark_evidence(
            checks,
            spec,
            expected_total=expected_total,
        ),
        "checks": checks,
    }
    return _merge_run_metadata_evidence(
        result,
        run,
        expected_config=expected_config,
        expected_tags=expected_tags,
        expected_group=expected_group,
        expected_job_type=expected_job_type,
    )


def _format_table_name(template: str, *, num_few_shots: int) -> str:
    return template.format(num_few_shots=num_few_shots)


def _taxonomy_unit_sources(unit: dict[str, Any], *, num_few_shots: int) -> list[str]:
    if "sources" in unit:
        return [
            _format_table_name(str(source["table_name"]), num_few_shots=num_few_shots)
            for source in unit["sources"]
        ]
    return [_format_table_name(str(unit["table_name"]), num_few_shots=num_few_shots)]


def _load_taxonomy(path: Path) -> dict[str, Any]:
    return OmegaConf.to_container(OmegaConf.load(path), resolve=True)


def verify_full_taiwan_run(
    run: Any,
    *,
    taxonomy_path: Path = DEFAULT_TAXONOMY_PATH,
    num_few_shots: int = 2,
    include_pending: bool = False,
    require_aggregate: bool = True,
    expected_config: dict[str, Any] | None = None,
    expected_tags: list[str] | None = None,
    expected_group: str | None = None,
    expected_job_type: str | None = None,
) -> dict[str, Any]:
    summary = _as_summary_dict(run)
    taxonomy = _load_taxonomy(taxonomy_path)
    checks: list[dict[str, Any]] = []

    state = getattr(run, "state", None)
    if state == "finished":
        checks.append(_ok_check("run_state", "run state is finished", state=state))
    else:
        checks.append(_fail_check("run_state", "run state is not finished", state=state))

    for unit in taxonomy["units"]:
        if not unit.get("required", True):
            continue
        if unit.get("pending", False) and not include_pending:
            checks.append(
                _ok_check(
                    "taxonomy_unit_pending_skipped",
                    f"{unit['id']} is pending and excluded from completion gate",
                    unit_id=unit["id"],
                    display_name=unit.get("display_name", unit["id"]),
                )
            )
            continue
        for table_name in _taxonomy_unit_sources(unit, num_few_shots=num_few_shots):
            check = _table_check(
                summary,
                table_name,
                name="taxonomy_table",
            )
            check["unit_id"] = unit["id"]
            check["display_name"] = unit.get("display_name", unit["id"])
            checks.append(check)

    if require_aggregate:
        for table_name in AGGREGATE_TABLES:
            checks.append(_table_check(summary, table_name, name="aggregate_table"))

    ok = all(check["ok"] for check in checks)
    result = {
        **_verification_header(ok),
        "ok": ok,
        "benchmark": FULL_BENCHMARK_ID,
        "taxonomy_path": str(taxonomy_path),
        "taxonomy_version": taxonomy.get("version"),
        "num_few_shots": num_few_shots,
        "include_pending": include_pending,
        "require_aggregate": require_aggregate,
        "run_id": getattr(run, "id", None),
        "run_name": getattr(run, "name", None),
        "required_evidence": _full_required_evidence(
            taxonomy,
            taxonomy_path=taxonomy_path,
            num_few_shots=num_few_shots,
            include_pending=include_pending,
            require_aggregate=require_aggregate,
        ),
        "observed_evidence": _observed_full_evidence(checks),
        "checks": checks,
    }
    return _merge_run_metadata_evidence(
        result,
        run,
        expected_config=expected_config,
        expected_tags=expected_tags,
        expected_group=expected_group,
        expected_job_type=expected_job_type,
    )


def load_run(entity: str, project: str, run_id: str) -> Any:
    api = wandb.Api(timeout=WANDB_API_TIMEOUT_SECONDS)
    return api.run(f"{entity}/{project}/{run_id}")


def query_source(
    *,
    entity: str,
    project: str,
    run_id: str,
    benchmark: str,
) -> dict[str, Any]:
    return {
        "kind": WANDB_QUERY_SOURCE_KIND,
        "api": "wandb.Api",
        "timeout_seconds": WANDB_API_TIMEOUT_SECONDS,
        "entity": entity,
        "project": project,
        "run_id": run_id,
        "run_path": f"{entity}/{project}/{run_id}",
        "benchmark": benchmark,
        "summary_source": "run.summary_metrics",
        "artifact_source": "run.logged_artifacts",
        "history_scanned": False,
    }


def parse_expected_config(items: list[str] | None) -> dict[str, Any]:
    parsed: dict[str, Any] = {}
    for item in items or []:
        if "=" not in item:
            raise SystemExit("--expected-run-config values must be KEY=JSON_OR_STRING")
        key, raw_value = item.split("=", 1)
        key = key.strip()
        if not key:
            raise SystemExit("--expected-run-config key must not be empty")
        try:
            value = json.loads(raw_value)
        except json.JSONDecodeError:
            value = raw_value
        parsed[key] = value
    return parsed


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--entity")
    parser.add_argument("--project")
    parser.add_argument("--run-id", required=True)
    parser.add_argument(
        "--benchmark",
        choices=sorted([*BENCHMARK_SPECS, FULL_BENCHMARK_ID]),
        default="agentic_math",
    )
    parser.add_argument(
        "--expected-total",
        type=int,
        default=None,
        help=(
            "Expected number of evaluated instances. Agentic Math full is 100; "
            "the current SWE-Bench Pro leaderboard subset is 80."
        ),
    )
    parser.add_argument(
        "--taxonomy-path",
        type=Path,
        default=DEFAULT_TAXONOMY_PATH,
        help="Taiwan taxonomy YAML path used by --benchmark taiwan_full.",
    )
    parser.add_argument(
        "--num-few-shots",
        type=int,
        default=2,
        help="Value used to format taxonomy table names containing {num_few_shots}.",
    )
    parser.add_argument(
        "--include-pending",
        action="store_true",
        help="Require taxonomy units marked pending=true when checking taiwan_full.",
    )
    parser.add_argument(
        "--no-require-aggregate",
        action="store_true",
        help="For taiwan_full, do not require taiwan_leaderboard/unit/radar tables.",
    )
    parser.add_argument(
        "--expected-run-config",
        action="append",
        default=[],
        metavar="KEY=JSON_OR_STRING",
        help=(
            "Require a W&B run.config value. Repeatable. Values are parsed as JSON "
            "when possible, otherwise compared as strings. Only expected keys are "
            "included in the verifier JSON."
        ),
    )
    parser.add_argument(
        "--expected-run-tag",
        action="append",
        default=[],
        help="Require a W&B run tag. Repeatable.",
    )
    parser.add_argument("--expected-run-group", help="Require a W&B run group.")
    parser.add_argument("--expected-run-job-type", help="Require a W&B run job_type.")
    parser.add_argument(
        "--require-nemoclaw-session-audit",
        action="store_true",
        help=(
            "For Agentic Math/SWE completion, require W&B summary metrics proving "
            "that every evaluated NeMoClaw task copied and passed its OpenClaw "
            "session audit."
        ),
    )
    parser.add_argument("--env-file", type=Path, default=DEFAULT_ENV_FILE)
    parser.add_argument("--json", type=Path, help="Optional path to write verifier JSON.")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    env, env_loaded = load_env_file(os.environ.copy(), args.env_file)
    os.environ.update(env)
    entity = args.entity or env.get("WANDB_ENTITY", "llm-leaderboard")
    project = args.project or env.get("WANDB_PROJECT", "tc-leaderboard")
    run = load_run(entity, project, args.run_id)
    expected_config = parse_expected_config(args.expected_run_config)
    if args.benchmark == FULL_BENCHMARK_ID:
        result = verify_full_taiwan_run(
            run,
            taxonomy_path=args.taxonomy_path,
            num_few_shots=args.num_few_shots,
            include_pending=args.include_pending,
            require_aggregate=not args.no_require_aggregate,
            expected_config=expected_config,
            expected_tags=args.expected_run_tag,
            expected_group=args.expected_run_group,
            expected_job_type=args.expected_run_job_type,
        )
    else:
        result = verify_run(
            run,
            BENCHMARK_SPECS[args.benchmark],
            expected_total=args.expected_total,
            require_nemoclaw_session_audit=bool(args.require_nemoclaw_session_audit),
            expected_config=expected_config,
            expected_tags=args.expected_run_tag,
            expected_group=args.expected_run_group,
            expected_job_type=args.expected_run_job_type,
        )
    result["entity"] = entity
    result["project"] = project
    result["query_source"] = query_source(
        entity=entity,
        project=project,
        run_id=args.run_id,
        benchmark=args.benchmark,
    )
    result["env_file"] = str(args.env_file.expanduser()) if args.env_file else None
    result["env_file_loaded"] = env_loaded
    result["generated_at"] = time.time()
    if args.json:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))
    if not result["ok"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
