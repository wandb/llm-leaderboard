#!/usr/bin/env python3
"""Shared contract checks for passed Weave content-canary gate JSON."""

from __future__ import annotations

from typing import Any


WEAVE_CONTENT_CANARY_GATE_NAME = "weave_agents_content_canary"
WEAVE_AGENTS_VERIFIER_SCHEMA_VERSION = 1
AGENTS_DIAGNOSTIC_SCHEMA_VERSION = 1


def nonempty_string(value: Any) -> bool:
    return isinstance(value, str) and bool(value.strip())


def int_at_least(value: Any, minimum: int) -> bool:
    return isinstance(value, int) and value >= minimum


def empty_list(value: Any) -> bool:
    return isinstance(value, list) and not value


def weave_content_canary_gate_contract_issues(payload: dict[str, Any]) -> list[str]:
    """Validate native Weave proof carried by a passed content-canary gate."""

    issues: list[str] = []
    if payload.get("gate") != WEAVE_CONTENT_CANARY_GATE_NAME:
        issues.append(f"gate must be {WEAVE_CONTENT_CANARY_GATE_NAME!r}")
    if payload.get("command_ok") is not True:
        issues.append("command_ok must be true")
    if payload.get("command_returncode") != 0:
        issues.append("command_returncode must be 0")
    if payload.get("paid_api_attempted") is not True:
        issues.append("paid_api_attempted must be true")
    if payload.get("weave_verifier_ok") is not True:
        issues.append("weave_verifier_ok must be true")
    if payload.get("weave_verifier_schema_version") != WEAVE_AGENTS_VERIFIER_SCHEMA_VERSION:
        issues.append(
            f"weave_verifier_schema_version must be {WEAVE_AGENTS_VERIFIER_SCHEMA_VERSION}"
        )
    verifier_trace_id = payload.get("weave_verifier_latest_trace_id")
    if not nonempty_string(verifier_trace_id):
        issues.append("weave_verifier_latest_trace_id must be a non-empty string")
    if not empty_list(payload.get("weave_verifier_validation_issues")):
        issues.append("weave_verifier_validation_issues must be an empty list")
    if payload.get("agents_diagnostic_ok") is not True:
        issues.append("agents_diagnostic_ok must be true")
    if payload.get("agents_diagnostic_schema_version") != AGENTS_DIAGNOSTIC_SCHEMA_VERSION:
        issues.append(
            f"agents_diagnostic_schema_version must be {AGENTS_DIAGNOSTIC_SCHEMA_VERSION}"
        )
    diagnostic_trace_id = payload.get("agents_diagnostic_latest_trace_id")
    if not nonempty_string(diagnostic_trace_id):
        issues.append("agents_diagnostic_latest_trace_id must be a non-empty string")
    elif nonempty_string(verifier_trace_id) and diagnostic_trace_id != verifier_trace_id:
        issues.append(
            "agents_diagnostic_latest_trace_id must match weave_verifier_latest_trace_id"
        )
    if not empty_list(payload.get("agents_diagnostic_validation_issues")):
        issues.append("agents_diagnostic_validation_issues must be an empty list")
    if not empty_list(payload.get("failed_checks")):
        issues.append("failed_checks must be an empty list")

    task_id = payload.get("task_id")
    if not nonempty_string(task_id) or not str(task_id).startswith(
        WEAVE_CONTENT_CANARY_GATE_NAME + "_"
    ):
        issues.append("task_id must be a weave_agents_content_canary_* string")
    for field in ("entity", "project", "agent_name", "canary_id", "model"):
        if not nonempty_string(payload.get(field)):
            issues.append(f"{field} must be a non-empty string")

    health = payload.get("content_capture_health")
    if not isinstance(health, dict):
        issues.append("content_capture_health must be an object")
    else:
        if not int_at_least(health.get("message_spans_with_input"), 1):
            issues.append("content_capture_health.message_spans_with_input must be >= 1")
        if not int_at_least(health.get("tool_spans_with_content"), 1):
            issues.append("content_capture_health.tool_spans_with_content must be >= 1")
        if not int_at_least(health.get("spans_with_valid_timestamps"), 1):
            issues.append("content_capture_health.spans_with_valid_timestamps must be >= 1")
        if health.get("spans_with_invalid_timestamps") != 0:
            issues.append("content_capture_health.spans_with_invalid_timestamps must be 0")

    paths = payload.get("paths")
    if not isinstance(paths, dict):
        issues.append("paths must be an object")
    else:
        for field in (
            "plan_file",
            "command_result_file",
            "verifier_json",
            "agents_diagnostic_json",
            "expected_sidecar",
            "prompt_file",
        ):
            if not nonempty_string(paths.get(field)):
                issues.append(f"paths.{field} must be a non-empty string")
        for field in (
            "command_result_exists",
            "verifier_json_exists",
            "agents_diagnostic_json_exists",
        ):
            if paths.get(field) is not True:
                issues.append(f"paths.{field} must be true")
    return issues
