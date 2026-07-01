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


def nonempty_string_list(value: Any) -> list[str]:
    if not isinstance(value, list):
        return []
    return [str(item).strip() for item in value if isinstance(item, str) and item.strip()]


def expected_required_texts(payload: dict[str, Any]) -> list[str]:
    texts: list[str] = []
    canary_id = payload.get("canary_id")
    if nonempty_string(canary_id):
        texts.append(str(canary_id))
        texts.append(f"CANARY_RESULT {canary_id} 91")
    nemoclaw = payload.get("nemoclaw")
    preflight = payload.get("nemoclaw_openclaw_config_preflight")
    if isinstance(nemoclaw, dict) and nemoclaw.get("enabled") is True:
        if isinstance(preflight, dict) and nonempty_string(preflight.get("config_path")):
            texts.append(f"openclaw_config_source: {preflight.get('config_path')}")
    return list(dict.fromkeys(texts))


def weave_content_canary_gate_contract_issues(payload: dict[str, Any]) -> list[str]:
    """Validate native Weave proof carried by a passed content-canary gate."""

    issues: list[str] = []
    if payload.get("gate") != WEAVE_CONTENT_CANARY_GATE_NAME:
        issues.append(f"gate must be {WEAVE_CONTENT_CANARY_GATE_NAME!r}")
    if payload.get("command_ok") is not True:
        issues.append("command_ok must be true")
    if payload.get("command_returncode") != 0:
        issues.append("command_returncode must be 0")
    if not empty_list(payload.get("command_result_contract_issues")):
        issues.append("command_result_contract_issues must be an empty list")
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
    if not empty_list(payload.get("plan_required_text_validation_issues")):
        issues.append("plan_required_text_validation_issues must be an empty list")
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

    expected_request_models = nonempty_string_list(payload.get("expected_request_models"))
    observed_request_models = nonempty_string_list(payload.get("observed_request_models"))
    span_request_models = nonempty_string_list(payload.get("span_request_models"))
    if not expected_request_models:
        issues.append("expected_request_models must be a non-empty list")
    if payload.get("request_model_proven") is not True:
        issues.append("request_model_proven must be true")
    if not observed_request_models:
        issues.append("observed_request_models must be a non-empty list")
    elif expected_request_models and set(expected_request_models).isdisjoint(
        observed_request_models
    ):
        issues.append("observed_request_models must include an expected model alias")
    if not span_request_models:
        issues.append("span_request_models must be a non-empty list")
    elif expected_request_models and set(expected_request_models).isdisjoint(
        span_request_models
    ):
        issues.append("span_request_models must include an expected model alias")

    gate_required_texts = nonempty_string_list(payload.get("expected_required_texts"))
    for required_text in expected_required_texts(payload):
        if required_text not in gate_required_texts:
            issues.append(
                f"expected_required_texts must include {required_text!r}"
            )

    nemoclaw = payload.get("nemoclaw")
    if not isinstance(nemoclaw, dict):
        issues.append("nemoclaw must be an object")
    else:
        if nemoclaw.get("required") is not True:
            issues.append("nemoclaw.required must be true")
        if nemoclaw.get("enabled") is not True:
            issues.append("nemoclaw.enabled must be true")
        for field in ("sandbox", "bin", "workdir"):
            if not nonempty_string(nemoclaw.get(field)):
                issues.append(f"nemoclaw.{field} must be a non-empty string")

    preflight = payload.get("nemoclaw_openclaw_config_preflight")
    if not isinstance(preflight, dict):
        issues.append("nemoclaw_openclaw_config_preflight must be an object")
    else:
        if preflight.get("required_before_openclaw") is not True:
            issues.append(
                "nemoclaw_openclaw_config_preflight.required_before_openclaw must be true"
            )
        if preflight.get("ran") is not True:
            issues.append("nemoclaw_openclaw_config_preflight.ran must be true")
        if preflight.get("ok") is not True:
            issues.append("nemoclaw_openclaw_config_preflight.ok must be true")
        if preflight.get("returncode") != 0:
            issues.append("nemoclaw_openclaw_config_preflight.returncode must be 0")
        if preflight.get("model") != payload.get("model"):
            issues.append("nemoclaw_openclaw_config_preflight.model must match model")
        if not nonempty_string(preflight.get("config_path")):
            issues.append(
                "nemoclaw_openclaw_config_preflight.config_path must be a non-empty string"
            )
        if not empty_list(preflight.get("errors")):
            issues.append("nemoclaw_openclaw_config_preflight.errors must be an empty list")
        checks = preflight.get("checks")
        if not isinstance(checks, list) or not checks:
            issues.append("nemoclaw_openclaw_config_preflight.checks must be a non-empty list")
        elif any(not isinstance(check, dict) or check.get("ok") is not True for check in checks):
            issues.append("nemoclaw_openclaw_config_preflight.checks must all have ok=true")

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
        if not int_at_least(health.get("request_model_count"), 1):
            issues.append("content_capture_health.request_model_count must be >= 1")

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
