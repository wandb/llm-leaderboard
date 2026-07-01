#!/usr/bin/env python3
"""Sync Weave Agents verifier JSONs into a paid-run review record.

This is an offline bookkeeping tool. It does not query W&B, run model
inference, or rewrite trace contents. It attaches already-generated
verify_taiwan_weave_agents.py JSONs to a paid-run review so release gates can
verify the exact native Agents evidence files.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[2]
WEAVE_AGENTS_COMPLETION_SCHEMA_VERSION = 1
REQUIRED_WEAVE_AGENTS_CHECK_NAMES = {
    "input_message_capture",
    "message_content_capture",
    "request_model",
    "trace_timestamp_quality",
    "trace_order",
    "trace_user_message_order",
    "trace_final_answer_order",
    "tool_content_capture",
    "tool_span_count",
    "trace_errors",
    "usage",
}
WEAVE_AGENTS_QUERY_SOURCE_KIND = "wandb_agents_api"
WEAVE_AGENTS_API_BASE_URL = "https://trace.wandb.ai"
WEAVE_AGENTS_QUERY_ENDPOINT = "/agents/query"
WEAVE_AGENTS_SPANS_QUERY_ENDPOINT = "/agents/spans/query"
WEAVE_AGENTS_QUERY_COUNT_FIELDS = (
    "agents_count",
    "spans_count",
    "matching_span_count",
    "latest_trace_span_count",
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


def _checks_all_ok(value: Any) -> bool:
    if not isinstance(value, list) or not value:
        return False
    return all(isinstance(row, dict) and row.get("ok") is True for row in value)


def _missing_required_check_names(value: Any) -> list[str]:
    if not isinstance(value, list):
        return sorted(REQUIRED_WEAVE_AGENTS_CHECK_NAMES)
    names = {
        row.get("name")
        for row in value
        if isinstance(row, dict) and isinstance(row.get("name"), str)
    }
    return sorted(REQUIRED_WEAVE_AGENTS_CHECK_NAMES - names)


def _check_names(value: Any) -> set[str]:
    if not isinstance(value, list):
        return set()
    return {
        row.get("name")
        for row in value
        if isinstance(row, dict) and isinstance(row.get("name"), str)
    }


def _non_empty_string_list(value: Any) -> list[str]:
    if not isinstance(value, list):
        return []
    result: list[str] = []
    for item in value:
        if isinstance(item, str) and item.strip():
            result.append(item.strip())
    return result


def _check_by_name(value: Any, name: str) -> dict[str, Any] | None:
    if not isinstance(value, list):
        return None
    for row in value:
        if isinstance(row, dict) and row.get("name") == name:
            return row
    return None


def _request_model_evidence(
    payload: dict[str, Any],
    *,
    required: dict[str, Any],
    health: dict[str, Any],
) -> tuple[bool, dict[str, Any], list[str]]:
    issues: list[str] = []
    checks = payload.get("checks")
    expected = _non_empty_string_list(required.get("expected_request_models"))
    if not expected:
        issues.append(
            "required_evidence.expected_request_models must be a non-empty list of strings"
        )

    check = _check_by_name(checks, "request_model")
    check_expected: list[str] = []
    check_observed: list[str] = []
    if check is None:
        issues.append("checks missing required check(s): request_model")
    else:
        if check.get("ok") is not True:
            issues.append("request_model check must be ok=true")
        check_expected = _non_empty_string_list(check.get("expected_request_models"))
        check_observed = _non_empty_string_list(check.get("observed_request_models"))
        if not check_expected:
            issues.append(
                "checks.request_model.expected_request_models must be a non-empty list of strings"
            )
        elif expected and set(check_expected) != set(expected):
            issues.append(
                "checks.request_model.expected_request_models must match "
                "required_evidence.expected_request_models"
            )
        if not check_observed:
            issues.append(
                "checks.request_model.observed_request_models must be a non-empty list of strings"
            )
        elif expected and set(expected).isdisjoint(check_observed):
            issues.append(
                "checks.request_model.observed_request_models must include an expected model alias"
            )

    spans = payload.get("latest_trace_spans_chronological")
    span_rows = spans if isinstance(spans, list) else []
    span_models = sorted(
        {
            str(span.get("request_model")).strip()
            for span in span_rows
            if isinstance(span, dict)
            and isinstance(span.get("request_model"), str)
            and span.get("request_model").strip()
        }
    )
    if not span_models:
        issues.append("latest_trace_spans_chronological must expose request_model")
    elif expected and set(expected).isdisjoint(span_models):
        issues.append(
            "latest_trace_spans_chronological request_model values must include an expected model alias"
        )
    if check_observed and span_models and set(check_observed) != set(span_models):
        issues.append(
            "checks.request_model.observed_request_models must match "
            "latest_trace_spans_chronological request_model values"
        )

    request_model_count = health.get("request_model_count")
    if not isinstance(request_model_count, int) or request_model_count <= 0:
        issues.append("content_capture_health.request_model_count must be a positive integer")
    elif span_models and request_model_count != len(span_models):
        issues.append(
            "content_capture_health.request_model_count must match unique request_model values"
        )

    evidence = {
        "expected_request_models": expected,
        "observed_request_models": check_observed,
        "span_request_models": span_models,
        "request_model_count": request_model_count,
    }
    return not issues, evidence, issues


def _parse_span_timestamp(value: Any) -> float | None:
    if not isinstance(value, str) or not value.strip():
        return None
    text = value.strip()
    if text.endswith("Z"):
        text = f"{text[:-1]}+00:00"
    try:
        parsed = datetime.fromisoformat(text)
    except ValueError:
        return None
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed.timestamp()


def _span_timestamps(span: dict[str, Any]) -> tuple[float | None, float | None]:
    return (
        _parse_span_timestamp(span.get("started_at")),
        _parse_span_timestamp(span.get("ended_at")),
    )


def _timestamp_order_key(span: dict[str, Any]) -> tuple[float, float, str] | None:
    started_ts, ended_ts = _span_timestamps(span)
    if started_ts is None or ended_ts is None:
        return None
    return (started_ts, ended_ts, str(span.get("span_id") or ""))


def _required_texts(value: Any, issues: list[str]) -> list[str]:
    if value in (None, ""):
        return []
    if not isinstance(value, list):
        issues.append("required_evidence.required_texts must be a list")
        return []
    result: list[str] = []
    for index, item in enumerate(value, start=1):
        if not isinstance(item, str):
            issues.append(f"required_evidence.required_texts[{index}] must be a string")
            continue
        if item:
            result.append(item)
    return result


def _query_source_issues(
    payload: dict[str, Any],
    *,
    required: dict[str, Any],
    spans: Any,
) -> list[str]:
    issues: list[str] = []
    query_source = payload.get("query_source")
    if not isinstance(query_source, dict):
        return ["query_source must be an object"]

    expected_query_fields = {
        "kind": WEAVE_AGENTS_QUERY_SOURCE_KIND,
        "api_base_url": WEAVE_AGENTS_API_BASE_URL,
        "agents_endpoint": WEAVE_AGENTS_QUERY_ENDPOINT,
        "spans_endpoint": WEAVE_AGENTS_SPANS_QUERY_ENDPOINT,
        "project_id": payload.get("project_id"),
        "agent_name": payload.get("agent_name"),
    }
    for field, expected in expected_query_fields.items():
        if query_source.get(field) != expected:
            issues.append(
                f"query_source.{field} must be {expected}; got {query_source.get(field)}"
            )

    for field in WEAVE_AGENTS_QUERY_COUNT_FIELDS:
        value = query_source.get(field)
        if not isinstance(value, int) or value < 0:
            issues.append(f"query_source.{field} must be a non-negative integer")

    for field in ("conversation_id", "conversation_id_contains"):
        expected = required.get(field) if isinstance(required.get(field), str) else ""
        if query_source.get(field) != expected:
            issues.append(f"query_source.{field} must match required_evidence")

    if isinstance(spans, list):
        valid_span_count = sum(1 for span in spans if isinstance(span, dict))
        latest_count = query_source.get("latest_trace_span_count")
        if isinstance(latest_count, int) and latest_count != valid_span_count:
            issues.append(
                "query_source.latest_trace_span_count must match "
                "latest_trace_spans_chronological"
            )
    return issues


def _weave_run_scope(payload: dict[str, Any], run_id: str) -> tuple[bool, dict[str, Any], str]:
    required = payload.get("required_evidence")
    if not isinstance(required, dict):
        required = {}
    conversation_id = required.get("conversation_id")
    if not isinstance(conversation_id, str):
        conversation_id = ""
    conversation_id_contains = required.get("conversation_id_contains")
    if not isinstance(conversation_id_contains, str):
        conversation_id_contains = ""

    spans = payload.get("latest_trace_spans_chronological")
    span_conversation_ids: list[str] = []
    if isinstance(spans, list):
        for span in spans:
            if not isinstance(span, dict):
                continue
            value = span.get("conversation_id")
            if isinstance(value, str) and value and value not in span_conversation_ids:
                span_conversation_ids.append(value)

    evidence = {
        "run_id": run_id,
        "conversation_id": conversation_id,
        "conversation_id_contains": conversation_id_contains,
        "span_conversation_ids": span_conversation_ids,
    }
    if not run_id:
        return False, evidence, "--run-id is required for run-scoped Weave Agents evidence"
    if run_id in conversation_id or run_id in conversation_id_contains:
        return True, evidence, ""
    return (
        False,
        evidence,
        "Weave Agents verifier JSON does not prove W&B run scope: "
        "required_evidence.conversation_id or conversation_id_contains must include "
        f"{run_id}",
    )


def _numeric_minimum_check(
    health: dict[str, Any],
    *,
    field: str,
    expected_min: Any,
    issues: list[str],
) -> None:
    if expected_min is None:
        return
    if not isinstance(expected_min, int):
        issues.append(f"required_evidence.{field} must be an integer")
        return
    value = health.get(field)
    if not isinstance(value, int):
        issues.append(f"content_capture_health.{field} must be an integer")
        return
    if value < expected_min:
        issues.append(
            f"content_capture_health.{field} must be at least {expected_min}; got {value}"
        )


def weave_payload_current(payload: dict[str, Any]) -> tuple[bool, list[str]]:
    issues: list[str] = []
    if payload.get("verification_schema_version") != WEAVE_AGENTS_COMPLETION_SCHEMA_VERSION:
        issues.append(
            f"verification_schema_version must be {WEAVE_AGENTS_COMPLETION_SCHEMA_VERSION}"
        )
    if not isinstance(payload.get("generated_at"), (int, float)):
        issues.append("generated_at must be numeric")
    if not isinstance(payload.get("agent_name"), str) or not payload.get("agent_name"):
        issues.append("agent_name must be a non-empty string")
    if not isinstance(payload.get("latest_trace_id"), str) or not payload.get("latest_trace_id"):
        issues.append("latest_trace_id must be a non-empty string")
    if not _checks_all_ok(payload.get("checks")):
        issues.append("checks must be a non-empty list with every check ok=true")
    missing_checks = _missing_required_check_names(payload.get("checks"))
    if missing_checks:
        issues.append(
            "checks missing required check(s): " + ", ".join(missing_checks)
        )

    required = payload.get("required_evidence")
    if not isinstance(required, dict):
        issues.append("required_evidence must be an object")
        required = {}
    else:
        if required.get("content_required") is not True:
            issues.append("required_evidence.content_required must be true")
        if required.get("input_message_required") is not True:
            issues.append("required_evidence.input_message_required must be true")
        if required.get("tool_span_required") is not True:
            issues.append("required_evidence.tool_span_required must be true")
        if required.get("tool_content_required") is not True:
            issues.append("required_evidence.tool_content_required must be true")
        if required.get("trace_timestamp_quality_required") is not True:
            issues.append("required_evidence.trace_timestamp_quality_required must be true")
        if required.get("trace_final_answer_order_required") is not True:
            issues.append("required_evidence.trace_final_answer_order_required must be true")
        if required.get("usage_required") is not True:
            issues.append("required_evidence.usage_required must be true")
        if required.get("no_error_spans_required") is not True:
            issues.append("required_evidence.no_error_spans_required must be true")
    required_texts = _required_texts(required.get("required_texts"), issues)
    issues.extend(
        _query_source_issues(
            payload,
            required=required,
            spans=payload.get("latest_trace_spans_chronological"),
        )
    )
    health = payload.get("content_capture_health")
    if not isinstance(health, dict):
        issues.append("content_capture_health must be an object")
        health = {}
    check_names = _check_names(payload.get("checks"))
    request_model_proven, _, request_model_issues = _request_model_evidence(
        payload,
        required=required,
        health=health,
    )
    if not request_model_proven:
        issues.extend(request_model_issues)

    _numeric_minimum_check(
        health,
        field="span_count_checked",
        expected_min=required.get("min_trace_spans"),
        issues=issues,
    )
    _numeric_minimum_check(
        health,
        field="message_span_count",
        expected_min=required.get("min_message_spans"),
        issues=issues,
    )
    if required.get("content_required") is True:
        value = health.get("message_spans_with_content")
        if not isinstance(value, int) or value <= 0:
            issues.append("required message content is not visible")
    if required.get("input_message_required") is True:
        value = health.get("message_spans_with_input")
        if not isinstance(value, int) or value <= 0:
            issues.append("required user/problem input message is not visible")
    if required.get("tool_span_required") is True:
        value = health.get("tool_span_count")
        if not isinstance(value, int) or value <= 0:
            issues.append("required tool span is missing")
    if required.get("tool_content_required") is True:
        tool_count = health.get("tool_span_count")
        with_content = health.get("tool_spans_with_content")
        if not isinstance(tool_count, int) or tool_count <= 0:
            issues.append("required tool content has no tool spans")
        elif not isinstance(with_content, int) or with_content < tool_count:
            issues.append("required tool content is not visible for every tool span")
    if required.get("usage_required") is True:
        usage_check = _check_by_name(payload.get("checks"), "usage")
        if usage_check is None:
            issues.append("checks missing required check(s): usage")
        elif usage_check.get("ok") is not True:
            issues.append("usage check must be ok=true")
        else:
            total_tokens = 0
            for field in (
                "agent_input_tokens",
                "agent_output_tokens",
                "trace_input_tokens",
                "trace_output_tokens",
            ):
                value = usage_check.get(field)
                if isinstance(value, int):
                    total_tokens += value
            if total_tokens <= 0:
                issues.append("usage check must expose positive token usage")
    if required_texts:
        if "required_text_capture" not in check_names:
            issues.append("checks missing required check(s): required_text_capture")
        required_text_count = health.get("required_text_count")
        if not isinstance(required_text_count, int) or required_text_count < len(required_texts):
            issues.append(
                "content_capture_health.required_text_count must be at least "
                f"{len(required_texts)}; got {required_text_count}"
            )

    spans = payload.get("latest_trace_spans_chronological")
    if not isinstance(spans, list) or not spans:
        issues.append("latest_trace_spans_chronological must be a non-empty list")
        spans = []
    chronological_keys: list[tuple[str, str, str]] = []
    timestamp_order_keys: list[tuple[float, float, str]] = []
    message_started_at: list[str] = []
    message_started_ts: list[float] = []
    input_message_started_at: list[str] = []
    input_message_started_ts: list[float] = []
    tool_started_at: list[str] = []
    tool_started_ts: list[float] = []
    final_answer_ended_ts: list[float] = []
    latest_trace_id = payload.get("latest_trace_id")
    for index, span in enumerate(spans, start=1):
        if not isinstance(span, dict):
            issues.append(f"latest_trace_spans_chronological[{index}] must be an object")
            continue
        timestamp_key = _timestamp_order_key(span)
        if timestamp_key is None:
            issues.append(
                f"latest_trace_spans_chronological[{index}] has missing or invalid timestamps"
            )
        else:
            started_ts, ended_ts, _ = timestamp_key
            if ended_ts < started_ts:
                issues.append(
                    f"latest_trace_spans_chronological[{index}] ends before it starts"
                )
            timestamp_order_keys.append(timestamp_key)
        chronological_keys.append(
            (
                str(span.get("started_at") or ""),
                str(span.get("ended_at") or ""),
                str(span.get("span_id") or ""),
            )
        )
        if latest_trace_id and span.get("trace_id") != latest_trace_id:
            issues.append("latest_trace_spans_chronological contains spans from another trace")
        if span.get("operation_name") in {"chat", "invoke_agent"}:
            message_started_at.append(str(span.get("started_at") or ""))
            started_ts, ended_ts = _span_timestamps(span)
            if started_ts is not None:
                message_started_ts.append(started_ts)
            output_text = json.dumps(span.get("output_messages"), ensure_ascii=False)
            if ended_ts is not None and any(
                marker in output_text
                for marker in ("ANSWER:", "FINAL ANSWER", "Final answer", "\\boxed", "CANARY_RESULT")
            ):
                final_answer_ended_ts.append(ended_ts)
            if span.get("input_messages") or span.get("has_input_messages") is True:
                input_message_started_at.append(str(span.get("started_at") or ""))
                if started_ts is not None:
                    input_message_started_ts.append(started_ts)
        if span.get("operation_name") == "execute_tool":
            tool_started_at.append(str(span.get("started_at") or ""))
            started_ts, _ = _span_timestamps(span)
            if started_ts is not None:
                tool_started_ts.append(started_ts)
        if required.get("no_error_spans_required") is True and span.get("error_type"):
            issues.append("latest trace contains error spans")
    if chronological_keys and chronological_keys != sorted(chronological_keys):
        issues.append("latest_trace_spans_chronological must be sorted by span time")
    if timestamp_order_keys and timestamp_order_keys != sorted(timestamp_order_keys):
        issues.append("latest_trace_spans_chronological must be sorted by parsed span time")
    health_valid = health.get("spans_with_valid_timestamps")
    health_invalid = health.get("spans_with_invalid_timestamps")
    if not isinstance(health_valid, int):
        issues.append("content_capture_health.spans_with_valid_timestamps must be an integer")
    elif health_valid != len(timestamp_order_keys):
        issues.append("content_capture_health.spans_with_valid_timestamps does not match spans")
    expected_invalid = len(spans) - len(timestamp_order_keys)
    if not isinstance(health_invalid, int):
        issues.append("content_capture_health.spans_with_invalid_timestamps must be an integer")
    elif health_invalid != expected_invalid:
        issues.append("content_capture_health.spans_with_invalid_timestamps does not match spans")
    if health_invalid not in (0, None):
        issues.append("latest trace has invalid timestamps")
    if message_started_ts and tool_started_ts and min(tool_started_ts) <= min(message_started_ts):
        issues.append("tool span starts before the first message span")
    if (
        required.get("input_message_required") is True
        and tool_started_ts
        and not input_message_started_ts
    ):
        issues.append("tool span is present but no visible user/problem input span exists")
    if (
        input_message_started_ts
        and tool_started_ts
        and min(tool_started_ts) <= min(input_message_started_ts)
    ):
        issues.append("tool span starts before the first visible user/problem input span")
    if (
        final_answer_ended_ts
        and tool_started_ts
        and max(tool_started_ts) >= min(final_answer_ended_ts)
    ):
        issues.append("tool span starts after a final-answer message span")
    return not issues, issues


def weave_completion_entry(
    path: Path,
    payload: dict[str, Any],
    *,
    run_id: str,
    allow_failed: bool,
    agent_name: str | None,
) -> dict[str, Any]:
    if payload.get("ok") is not True and not allow_failed:
        raise ValueError(f"{path} is not a passing Weave Agents verifier JSON")
    current, issues = weave_payload_current(payload)
    if not current and not allow_failed:
        raise ValueError(
            f"{path} is not a current Weave Agents verifier JSON: "
            + "; ".join(issues)
        )
    payload_agent_name = payload.get("agent_name")
    if agent_name and payload_agent_name != agent_name:
        raise ValueError(
            f"{path} agent_name mismatch: expected {agent_name}, got {payload_agent_name}"
        )
    if not isinstance(payload_agent_name, str) or not payload_agent_name:
        raise ValueError(f"{path} is missing agent_name")
    if not isinstance(run_id, str) or not run_id:
        raise ValueError("--run-id is required for run-level Weave Agents sync")
    run_scope_proven, run_scope, run_scope_error = _weave_run_scope(payload, run_id)
    if not run_scope_proven and not allow_failed:
        raise ValueError(f"{path} {run_scope_error}")
    query_source = payload.get("query_source")
    if not isinstance(query_source, dict):
        query_source = {}
    required = payload.get("required_evidence")
    if not isinstance(required, dict):
        required = {}
    health = payload.get("content_capture_health")
    if not isinstance(health, dict):
        health = {}
    request_model_proven, request_model_evidence, _ = _request_model_evidence(
        payload,
        required=required,
        health=health,
    )
    checks = payload.get("checks")
    usage_check = _check_by_name(checks, "usage")
    trace_errors_check = _check_by_name(checks, "trace_errors")
    tool_count = health.get("tool_span_count")
    tool_spans_with_content = health.get("tool_spans_with_content")
    message_content_proven = (
        required.get("content_required") is True
        and isinstance(health.get("message_spans_with_content"), int)
        and health.get("message_spans_with_content") > 0
    )
    input_message_proven = (
        required.get("input_message_required") is True
        and isinstance(health.get("message_spans_with_input"), int)
        and health.get("message_spans_with_input") > 0
    )
    tool_span_proven = (
        required.get("tool_span_required") is True
        and isinstance(tool_count, int)
        and tool_count > 0
    )
    tool_content_proven = (
        required.get("tool_content_required") is True
        and tool_span_proven
        and isinstance(tool_spans_with_content, int)
        and tool_spans_with_content >= tool_count
    )
    usage_proven = (
        required.get("usage_required") is True
        and isinstance(usage_check, dict)
        and usage_check.get("ok") is True
        and sum(
            value
            for value in (
                usage_check.get("agent_input_tokens"),
                usage_check.get("agent_output_tokens"),
                usage_check.get("trace_input_tokens"),
                usage_check.get("trace_output_tokens"),
            )
            if isinstance(value, int)
        )
        > 0
    )
    no_error_spans_proven = (
        required.get("no_error_spans_required") is True
        and isinstance(trace_errors_check, dict)
        and trace_errors_check.get("ok") is True
    )
    return {
        "ok": bool(payload.get("ok")),
        "run_id": run_id,
        "path": path_display(path),
        "agent_name": payload_agent_name,
        "verification_schema_version": payload.get("verification_schema_version"),
        "latest_trace_id": payload.get("latest_trace_id"),
        "checks_valid": _checks_all_ok(payload.get("checks")),
        "trace_present": isinstance(payload.get("latest_trace_id"), str)
        and bool(payload.get("latest_trace_id")),
        "run_scope_proven": run_scope_proven,
        "request_model_proven": request_model_proven,
        "message_content_proven": message_content_proven,
        "input_message_proven": input_message_proven,
        "tool_span_proven": tool_span_proven,
        "tool_content_proven": tool_content_proven,
        "usage_proven": usage_proven,
        "no_error_spans_proven": no_error_spans_proven,
        "expected_request_models": request_model_evidence["expected_request_models"],
        "observed_request_models": request_model_evidence["observed_request_models"],
        "span_request_models": request_model_evidence["span_request_models"],
        "conversation_id": run_scope["conversation_id"],
        "conversation_id_contains": run_scope["conversation_id_contains"],
        "query_source_kind": query_source.get("kind"),
        "query_source_api_base_url": query_source.get("api_base_url"),
        "query_source_agents_endpoint": query_source.get("agents_endpoint"),
        "query_source_spans_endpoint": query_source.get("spans_endpoint"),
        "query_source_project_id": query_source.get("project_id"),
    }


def merge_weave_entry(
    current: Any,
    entry: dict[str, Any],
    *,
    replace: bool,
) -> tuple[dict[str, Any], str]:
    if not isinstance(current, dict) or not current:
        return copy.deepcopy(entry), "added"
    same = (
        current.get("run_id") == entry.get("run_id")
        and current.get("agent_name") == entry.get("agent_name")
    )
    if same and not replace:
        return current, "kept_existing"
    return copy.deepcopy(entry), "replaced" if same else "replaced_different_entry"


def sync_review(
    review: dict[str, Any],
    entries: list[dict[str, Any]],
    *,
    top_level: bool,
    replace: bool,
    set_verify_weave_agents: bool,
) -> tuple[dict[str, Any], list[dict[str, Any]], list[dict[str, Any]]]:
    updated = copy.deepcopy(review)
    changes: list[dict[str, Any]] = []
    unmatched: list[dict[str, Any]] = []

    if set_verify_weave_agents:
        updated["verify_weave_agents"] = True

    if top_level:
        if len(entries) > 1:
            raise ValueError("--top-level supports exactly one Weave Agents completion entry")
        if entries:
            merged, action = merge_weave_entry(
                updated.get("weave_agents_completion"),
                entries[0],
                replace=replace,
            )
            updated["weave_agents_completion"] = merged
            changes.append(
                {
                    "target": "top_level",
                    "action": action,
                    "run_id": entries[0].get("run_id"),
                    "agent_name": entries[0].get("agent_name"),
                }
            )
        return updated, changes, unmatched

    runs = updated.get("runs")
    if not isinstance(runs, list):
        runs = []
        updated["runs"] = runs

    for entry in entries:
        matches = [
            run
            for run in runs
            if isinstance(run, dict) and run.get("wandb_run_id") == entry.get("run_id")
        ]
        if not matches:
            unmatched.append(entry)
            continue
        for run in matches:
            merged, action = merge_weave_entry(
                run.get("weave_agents_completion"),
                entry,
                replace=replace,
            )
            run["weave_agents_completion"] = merged
            changes.append(
                {
                    "target": "run",
                    "action": action,
                    "run_id": entry.get("run_id"),
                    "agent_name": entry.get("agent_name"),
                    "config": run.get("config"),
                }
            )
    return updated, changes, unmatched


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
        "entries": entries,
        "change_count": len(changes),
        "unmatched_count": len(unmatched),
        "changes": changes,
        "unmatched_entries": unmatched,
        "before_status": review.get("status"),
        "after_status": updated.get("status"),
        "verify_weave_agents": bool(updated.get("verify_weave_agents")),
    }


def build_validation_failed_report(
    *,
    review_path: Path,
    output_path: Path | None,
    completion_paths: list[Path],
    error: str,
    in_place: bool,
    dry_run: bool,
) -> dict[str, Any]:
    source_review_sha256 = ""
    if review_path.exists():
        source_review_sha256 = sha256_file(review_path)
    return {
        "ok": False,
        "status": "validation_failed",
        "generated_at": time.time(),
        "review_path": path_display(review_path),
        "source_review_sha256": source_review_sha256,
        "output_path": path_display(output_path) if output_path else "",
        "in_place": in_place,
        "dry_run": dry_run,
        "entry_count": 0,
        "entries": [],
        "change_count": 0,
        "unmatched_count": 0,
        "changes": [],
        "unmatched_entries": [],
        "completion_paths": [path_display(path) for path in completion_paths],
        "validation_errors": [error],
        "verify_weave_agents": False,
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
    source_review_sha256 = payload.get("source_review_sha256")
    if (
        not isinstance(source_review_sha256, str)
        or len(source_review_sha256) != 64
        or any(char not in "0123456789abcdef" for char in source_review_sha256)
    ):
        issues.append("source_review_sha256 must be a 64-character lowercase hex digest")
    elif source_review_sha256 != current_report.get("source_review_sha256"):
        issues.append("source_review_sha256 does not match current sync")
    if payload.get("verify_weave_agents") != current_report.get("verify_weave_agents"):
        issues.append("verify_weave_agents does not match current sync")
    if payload.get("unmatched_count") != 0:
        issues.append("unmatched_count must be 0")
    for field in ("before_status", "after_status"):
        if field not in payload:
            issues.append(f"{field} is missing")
        elif payload.get(field) != current_report.get(field):
            issues.append(f"{field} does not match current sync")
    for field in ("entry_count", "entries", "change_count", "changes"):
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
    if top_level:
        current = updated.get("weave_agents_completion")
        if isinstance(current, dict):
            current["sync_dry_run_report_json"] = display
            current["sync_dry_run_source_review_json"] = source_review_display
            current["sync_dry_run_source_review_sha256"] = source_review_sha256
        return

    runs = updated.get("runs")
    if not isinstance(runs, list):
        return
    for entry in entries:
        run_id = entry.get("run_id")
        agent_name = entry.get("agent_name")
        for run in runs:
            if not isinstance(run, dict) or run.get("wandb_run_id") != run_id:
                continue
            current = run.get("weave_agents_completion")
            if not isinstance(current, dict):
                continue
            if current.get("run_id") == run_id and current.get("agent_name") == agent_name:
                current["sync_dry_run_report_json"] = display
                current["sync_dry_run_source_review_json"] = source_review_display
                current["sync_dry_run_source_review_sha256"] = source_review_sha256


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--review-json", type=Path, required=True)
    parser.add_argument("--completion-json", type=Path, action="append", required=True)
    parser.add_argument("--run-id", required=True, help="W&B run id whose run row should receive the verifier entry.")
    parser.add_argument("--agent-name", help="Require verifier JSON agent_name to match this value.")
    parser.add_argument("--output-json", type=Path)
    parser.add_argument("--report-json", type=Path)
    parser.add_argument("--in-place", action="store_true")
    parser.add_argument("--top-level", action="store_true")
    parser.add_argument("--no-replace", action="store_true")
    parser.add_argument("--allow-failed", action="store_true")
    parser.add_argument("--allow-unmatched", action="store_true")
    parser.add_argument("--set-verify-weave-agents", action="store_true")
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
    review_path = repo_path(args.review_json)
    output_path = review_path if args.in_place else repo_path(args.output_json) if args.output_json else None
    dry_run = output_path is None
    completion_paths = [repo_path(path) for path in args.completion_json]
    try:
        review = read_json_object(review_path)
        entries = [
            weave_completion_entry(
                path,
                read_json_object(path),
                run_id=args.run_id,
                allow_failed=bool(args.allow_failed),
                agent_name=args.agent_name,
            )
            for path in completion_paths
        ]
    except (OSError, json.JSONDecodeError, ValueError) as exc:
        message = str(exc)
        if args.report_json:
            report = build_validation_failed_report(
                review_path=review_path,
                output_path=output_path,
                completion_paths=completion_paths,
                error=message,
                in_place=bool(args.in_place),
                dry_run=dry_run,
            )
            write_json(repo_path(args.report_json), report)
            print(json.dumps(report, ensure_ascii=False, indent=2))
            raise SystemExit(1)
        raise SystemExit(message) from exc
    updated, changes, unmatched = sync_review(
        review,
        entries,
        top_level=bool(args.top_level),
        replace=not bool(args.no_replace),
        set_verify_weave_agents=bool(args.set_verify_weave_agents),
    )
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
