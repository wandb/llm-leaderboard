#!/usr/bin/env python3
"""Summarize a fresh Weave Agents content canary as a production gate.

This script is intentionally offline: it reads the canary plan, the local
OpenClaw command result, and an already-written Weave Agents verifier JSON. It
does not query W&B or call a model.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUTPUT_DIR = Path("outputs/weave_agents_content_canary")
GATE_NAME = "weave_agents_content_canary"
WEAVE_AGENTS_VERIFIER_SCHEMA_VERSION = 1
AGENTS_DIAGNOSTIC_SCHEMA_VERSION = 1
WEAVE_AGENTS_API_BASE_URL = "https://trace.wandb.ai"
WEAVE_AGENTS_QUERY_ENDPOINT = "/agents/query"
WEAVE_AGENTS_SPANS_ENDPOINT = "/agents/spans/query"
REQUIRED_PASSING_VERIFIER_CHECKS = (
    "trace_timestamp_quality",
    "trace_order",
    "trace_user_message_order",
    "trace_final_answer_order",
)


def safe_canary_id(canary_id: str) -> str:
    return "".join(char if char.isalnum() else "_" for char in canary_id)


def task_id_from_canary_id(canary_id: str) -> str:
    return f"weave_agents_content_canary_{safe_canary_id(canary_id)}"


def read_json(path: Path | None) -> tuple[dict[str, Any] | None, str | None]:
    if path is None:
        return None, "path is missing"
    if not path.exists():
        return None, f"{path} does not exist"
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        return None, f"{path} is not readable JSON: {exc}"
    if not isinstance(payload, dict):
        return None, f"{path} is not a JSON object"
    return payload, None


def repo_relative_path(path_value: str) -> Path:
    path = Path(path_value).expanduser()
    if path.is_absolute():
        return path
    return REPO_ROOT / path


def resolve_plan_file(args: argparse.Namespace) -> Path:
    if args.plan_file:
        return args.plan_file
    if not args.canary_id:
        raise SystemExit("pass --plan-file or --canary-id")
    return args.output_dir / "plans" / f"{task_id_from_canary_id(args.canary_id)}.json"


def task_id_from_plan(plan: dict[str, Any], plan_file: Path) -> str:
    value = plan.get("task_id")
    if isinstance(value, str) and value:
        return value
    return plan_file.stem


def default_command_result_file(plan_file: Path, task_id: str) -> Path:
    return plan_file.with_name(f"{task_id}.command_result.json")


def output_dir_from_plan_file(plan_file: Path) -> Path:
    if plan_file.parent.name == "plans":
        return plan_file.parent.parent
    return plan_file.parent


def default_verifier_dir(plan_file: Path, task_id: str) -> Path:
    return output_dir_from_plan_file(plan_file) / "verifier" / task_id


def find_latest_verifier_json(plan_file: Path, task_id: str) -> Path | None:
    verifier_dir = default_verifier_dir(plan_file, task_id)
    candidates = sorted(verifier_dir.glob("attempt_*.json"))
    if not candidates:
        return None
    return max(candidates, key=lambda path: (path.stat().st_mtime, path.name))


def failed_check_names(verifier: dict[str, Any] | None) -> set[str]:
    if not verifier:
        return set()
    checks = verifier.get("checks")
    if not isinstance(checks, list):
        return set()
    names: set[str] = set()
    for check in checks:
        if isinstance(check, dict) and not check.get("ok"):
            name = check.get("name")
            if isinstance(name, str) and name:
                names.add(name)
    return names


def failed_checks(verifier: dict[str, Any] | None) -> list[dict[str, Any]]:
    if not verifier:
        return []
    checks = verifier.get("checks")
    if not isinstance(checks, list):
        return []
    return [check for check in checks if isinstance(check, dict) and not check.get("ok")]


def verifier_check_names(verifier: dict[str, Any] | None) -> set[str]:
    if not verifier:
        return set()
    checks = verifier.get("checks")
    if not isinstance(checks, list):
        return set()
    names: set[str] = set()
    for check in checks:
        if isinstance(check, dict):
            name = check.get("name")
            if isinstance(name, str) and name:
                names.add(name)
    return names


def required_texts_from_verifier(
    verifier: dict[str, Any] | None,
    issues: list[str],
) -> list[str]:
    if not verifier:
        return []
    required_evidence = verifier.get("required_evidence")
    if not isinstance(required_evidence, dict):
        return []
    value = required_evidence.get("required_texts")
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


def _int_at_least(value: Any, minimum: int) -> bool:
    return isinstance(value, int) and value >= minimum


def _non_empty_string_list(value: Any) -> list[str]:
    if not isinstance(value, list):
        return []
    return [str(item).strip() for item in value if isinstance(item, str) and item.strip()]


def _span_time_key(span: dict[str, Any]) -> tuple[str, str, str]:
    return (
        str(span.get("started_at") or ""),
        str(span.get("ended_at") or ""),
        str(span.get("span_id") or ""),
    )


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


def _parsed_span_time_key(span: dict[str, Any]) -> tuple[float, float, str] | None:
    started_ts = _parse_span_timestamp(span.get("started_at"))
    ended_ts = _parse_span_timestamp(span.get("ended_at"))
    if started_ts is None or ended_ts is None:
        return None
    return (started_ts, ended_ts, str(span.get("span_id") or ""))


def _conversation_scope_values(verifier: dict[str, Any]) -> list[str]:
    values: list[str] = []
    for container_name in ("query_source", "required_evidence"):
        container = verifier.get(container_name)
        if not isinstance(container, dict):
            continue
        for key in ("conversation_id", "conversation_id_contains"):
            value = container.get(key)
            if isinstance(value, str) and value:
                values.append(value)
    return values


def query_source_validation_issues(
    verifier: dict[str, Any],
    *,
    expected_project_id: str | None,
    expected_agent_name: str | None,
    expected_task_id: str | None,
) -> list[str]:
    issues: list[str] = []
    query_source = verifier.get("query_source")
    if not isinstance(query_source, dict):
        return ["query_source must be an object from the native W&B Agents API verifier"]

    expected_pairs = {
        "kind": "wandb_agents_api",
        "api_base_url": WEAVE_AGENTS_API_BASE_URL,
        "agents_endpoint": WEAVE_AGENTS_QUERY_ENDPOINT,
        "spans_endpoint": WEAVE_AGENTS_SPANS_ENDPOINT,
    }
    if expected_project_id:
        expected_pairs["project_id"] = expected_project_id
    if expected_agent_name:
        expected_pairs["agent_name"] = expected_agent_name
    for key, expected in expected_pairs.items():
        if query_source.get(key) != expected:
            issues.append(f"query_source.{key} must be {expected!r}")

    for key in ("agents_count", "spans_count", "matching_span_count", "latest_trace_span_count"):
        if not _int_at_least(query_source.get(key), 1):
            issues.append(f"query_source.{key} must be an integer >= 1")

    if expected_task_id and not any(
        expected_task_id in value for value in _conversation_scope_values(verifier)
    ):
        issues.append(
            "query_source or required_evidence must scope the query to the canary task_id "
            f"{expected_task_id!r}"
        )
    return issues


def latest_trace_span_validation_issues(
    verifier: dict[str, Any],
    *,
    expected_agent_name: str | None,
    expected_task_id: str | None,
) -> list[str]:
    issues: list[str] = []
    spans = verifier.get("latest_trace_spans_chronological")
    if not isinstance(spans, list) or not spans:
        return ["latest_trace_spans_chronological must be a non-empty list"]

    latest_trace_id = verifier.get("latest_trace_id")
    previous_key: tuple[str, str, str] | None = None
    previous_parsed_key: tuple[float, float, str] | None = None
    for index, span in enumerate(spans, start=1):
        if not isinstance(span, dict):
            issues.append(f"latest_trace_spans_chronological[{index}] must be an object")
            continue
        parsed_key = _parsed_span_time_key(span)
        if parsed_key is None:
            issues.append(
                f"latest_trace_spans_chronological[{index}] has missing or invalid timestamps"
            )
        else:
            started_ts, ended_ts, _ = parsed_key
            if ended_ts < started_ts:
                issues.append(
                    f"latest_trace_spans_chronological[{index}] ends before it starts"
                )
            if previous_parsed_key is not None and parsed_key < previous_parsed_key:
                issues.append(
                    "latest_trace_spans_chronological must be sorted by parsed span time"
                )
            previous_parsed_key = parsed_key
        if expected_agent_name and span.get("agent_name") != expected_agent_name:
            issues.append(
                f"latest_trace_spans_chronological[{index}].agent_name must be "
                f"{expected_agent_name!r}"
            )
        if latest_trace_id and span.get("trace_id") != latest_trace_id:
            issues.append(
                f"latest_trace_spans_chronological[{index}].trace_id must match latest_trace_id"
            )
        conversation_id = span.get("conversation_id")
        if expected_task_id and (
            not isinstance(conversation_id, str) or expected_task_id not in conversation_id
        ):
            issues.append(
                f"latest_trace_spans_chronological[{index}].conversation_id must contain "
                f"{expected_task_id!r}"
            )
        key = _span_time_key(span)
        if previous_key is not None and key < previous_key:
            issues.append("latest_trace_spans_chronological must be sorted by span time")
        previous_key = key
    return issues


def request_model_aliases(model_id: str | None) -> list[str]:
    value = str(model_id or "").strip()
    if not value:
        return []
    aliases = [value]
    if value.startswith("openai-direct/"):
        aliases.append(value.removeprefix("openai-direct/"))
    if "/" in value:
        aliases.append(value.rsplit("/", 1)[-1])
    return list(dict.fromkeys(alias for alias in aliases if alias))


def expected_request_models_from_plan(plan: dict[str, Any]) -> list[str]:
    verification_requirements = plan.get("verification_requirements")
    if isinstance(verification_requirements, dict):
        expected = _non_empty_string_list(
            verification_requirements.get("expected_request_models")
        )
        if expected:
            return list(dict.fromkeys(expected))
    return request_model_aliases(
        plan.get("model") if isinstance(plan.get("model"), str) else None
    )


def request_model_evidence(verifier: dict[str, Any] | None) -> dict[str, Any]:
    if not isinstance(verifier, dict):
        return {
            "observed_request_models": [],
            "span_request_models": [],
            "request_model_count": None,
        }
    spans = verifier.get("latest_trace_spans_chronological")
    span_models = sorted(
        {
            str(span.get("request_model")).strip()
            for span in spans
            if isinstance(span, dict) and str(span.get("request_model") or "").strip()
        }
    ) if isinstance(spans, list) else []
    observed: list[str] = []
    checks = verifier.get("checks")
    if isinstance(checks, list):
        for check in checks:
            if isinstance(check, dict) and check.get("name") == "request_model":
                observed = _non_empty_string_list(check.get("observed_request_models"))
                break
    if not observed:
        observed = span_models
    health = verifier.get("content_capture_health")
    request_model_count = (
        health.get("request_model_count") if isinstance(health, dict) else None
    )
    return {
        "observed_request_models": observed,
        "span_request_models": span_models,
        "request_model_count": request_model_count,
    }


def request_model_validation_issues(
    verifier: dict[str, Any],
    *,
    expected_request_models: list[str],
) -> list[str]:
    issues: list[str] = []
    if not expected_request_models:
        return ["expected_request_models must be a non-empty list"]
    expected = set(expected_request_models)
    required_evidence = verifier.get("required_evidence")
    if not isinstance(required_evidence, dict):
        issues.append("required_evidence must be an object")
        required_expected: list[str] = []
    else:
        required_expected = _non_empty_string_list(
            required_evidence.get("expected_request_models")
        )
        if not required_expected:
            issues.append(
                "required_evidence.expected_request_models must be a non-empty list"
            )
        elif set(required_expected) != expected:
            issues.append(
                "required_evidence.expected_request_models must match the canary plan"
            )

    request_model_checks = [
        check
        for check in verifier.get("checks", [])
        if isinstance(check, dict) and check.get("name") == "request_model"
    ] if isinstance(verifier.get("checks"), list) else []
    if len(request_model_checks) != 1:
        issues.append("checks must include exactly one request_model check")
        observed_from_check: list[str] = []
    else:
        check = request_model_checks[0]
        if check.get("ok") is not True:
            issues.append("request_model check must have ok=true")
        check_expected = _non_empty_string_list(check.get("expected_request_models"))
        observed_from_check = _non_empty_string_list(check.get("observed_request_models"))
        if set(check_expected) != expected:
            issues.append(
                "checks.request_model.expected_request_models must match the canary plan"
            )
        if not observed_from_check:
            issues.append(
                "checks.request_model.observed_request_models must be a non-empty list"
            )
        elif expected.isdisjoint(observed_from_check):
            issues.append(
                "checks.request_model.observed_request_models must include an expected model alias"
            )

    evidence = request_model_evidence(verifier)
    span_models = evidence["span_request_models"]
    if not span_models:
        issues.append("latest_trace_spans_chronological must expose request_model")
    elif expected.isdisjoint(span_models):
        issues.append(
            "latest_trace_spans_chronological request_model values must include an expected model alias"
        )
    if observed_from_check and sorted(set(observed_from_check)) != sorted(set(span_models)):
        issues.append(
            "checks.request_model.observed_request_models must match latest_trace_spans_chronological request_model values"
        )
    request_model_count = evidence["request_model_count"]
    if not isinstance(request_model_count, int) or request_model_count <= 0:
        issues.append("content_capture_health.request_model_count must be a positive integer")
    elif span_models and request_model_count != len(set(span_models)):
        issues.append(
            "content_capture_health.request_model_count must match unique request_model values"
        )
    return issues


def nemoclaw_openclaw_config_preflight_validation_issues(
    plan: dict[str, Any],
) -> list[str]:
    if plan.get("will_call_paid_model_api") is not True:
        return []
    preflight = plan.get("nemoclaw_openclaw_config_preflight")
    if not isinstance(preflight, dict):
        return ["nemoclaw_openclaw_config_preflight must be an object"]
    issues: list[str] = []
    if preflight.get("required_before_openclaw") is not True:
        issues.append("nemoclaw_openclaw_config_preflight.required_before_openclaw must be true")
    if preflight.get("ran") is not True:
        issues.append("nemoclaw_openclaw_config_preflight.ran must be true")
    if preflight.get("ok") is not True:
        issues.append("nemoclaw_openclaw_config_preflight.ok must be true")
    if preflight.get("returncode") != 0:
        issues.append("nemoclaw_openclaw_config_preflight.returncode must be 0")
    model = plan.get("model")
    if isinstance(model, str) and model:
        if preflight.get("model") != model:
            issues.append("nemoclaw_openclaw_config_preflight.model must match plan model")
    else:
        issues.append("plan.model must be a non-empty string")
    config_path = preflight.get("config_path")
    if not isinstance(config_path, str) or not config_path:
        issues.append("nemoclaw_openclaw_config_preflight.config_path must be a non-empty string")
    checks = preflight.get("checks")
    if not isinstance(checks, list):
        issues.append("nemoclaw_openclaw_config_preflight.checks must be a list")
        return issues
    checks_by_name = {
        check.get("name"): check
        for check in checks
        if isinstance(check, dict) and isinstance(check.get("name"), str)
    }
    provider = preflight.get("provider")
    required_names = [
        f"NeMoClaw sandbox OpenClaw config is readable: {config_path}",
        f"NeMoClaw sandbox OpenClaw {provider} provider exists",
        f"NeMoClaw sandbox OpenClaw model is registered: {model}",
        "NeMoClaw sandbox OpenClaw Weave plugin is enabled",
    ]
    for name in required_names:
        check = checks_by_name.get(name)
        if not isinstance(check, dict):
            issues.append(f"nemoclaw_openclaw_config_preflight missing check: {name}")
        elif check.get("ok") is not True:
            issues.append(f"nemoclaw_openclaw_config_preflight check is not ok: {name}")
    errors = preflight.get("errors")
    if not isinstance(errors, list):
        issues.append("nemoclaw_openclaw_config_preflight.errors must be a list")
    elif errors:
        issues.append("nemoclaw_openclaw_config_preflight.errors must be empty")
    return issues


def passing_verifier_validation_issues(
    verifier: dict[str, Any] | None,
    *,
    expected_project_id: str | None = None,
    expected_agent_name: str | None = None,
    expected_task_id: str | None = None,
    expected_request_models: list[str] | None = None,
) -> list[str]:
    if verifier is None:
        return ["Weave verifier JSON is missing"]
    issues: list[str] = []
    if verifier.get("verification_schema_version") != WEAVE_AGENTS_VERIFIER_SCHEMA_VERSION:
        issues.append(
            f"verification_schema_version must be {WEAVE_AGENTS_VERIFIER_SCHEMA_VERSION}"
        )
    if not isinstance(verifier.get("generated_at"), (int, float)):
        issues.append("generated_at must be numeric")
    if not isinstance(verifier.get("latest_trace_id"), str) or not verifier.get("latest_trace_id"):
        issues.append("latest_trace_id must be a non-empty string")
    issues.extend(
        query_source_validation_issues(
            verifier,
            expected_project_id=expected_project_id,
            expected_agent_name=expected_agent_name,
            expected_task_id=expected_task_id,
        )
    )
    issues.extend(
        latest_trace_span_validation_issues(
            verifier,
            expected_agent_name=expected_agent_name,
            expected_task_id=expected_task_id,
        )
    )
    required_evidence = verifier.get("required_evidence")
    if not isinstance(required_evidence, dict):
        issues.append("required_evidence must be an object")
    else:
        if required_evidence.get("input_message_required") is not True:
            issues.append("required_evidence.input_message_required must be true")
        if required_evidence.get("trace_timestamp_quality_required") is not True:
            issues.append("required_evidence.trace_timestamp_quality_required must be true")
        if required_evidence.get("trace_final_answer_order_required") is not True:
            issues.append("required_evidence.trace_final_answer_order_required must be true")
    checks = verifier.get("checks")
    if not isinstance(checks, list) or not checks:
        issues.append("checks must be a non-empty list")
    elif any(not isinstance(check, dict) or check.get("ok") is not True for check in checks):
        issues.append("checks must all have ok=true")
    missing_checks = [
        name for name in REQUIRED_PASSING_VERIFIER_CHECKS if name not in verifier_check_names(verifier)
    ]
    if missing_checks:
        issues.append("checks missing required check(s): " + ", ".join(missing_checks))
    issues.extend(
        request_model_validation_issues(
            verifier,
            expected_request_models=expected_request_models or [],
        )
    )
    required_texts = required_texts_from_verifier(verifier, issues)
    if required_texts and "required_text_capture" not in verifier_check_names(verifier):
        issues.append("checks missing required check(s): required_text_capture")
    if required_texts:
        health = verifier.get("content_capture_health")
        if not isinstance(health, dict):
            issues.append("content_capture_health must be an object")
        else:
            count = health.get("required_text_count")
            if not isinstance(count, int) or count < len(required_texts):
                issues.append(
                    "content_capture_health.required_text_count must be at least "
                    f"{len(required_texts)}; got {count}"
                )
    health = verifier.get("content_capture_health")
    if not isinstance(health, dict):
        issues.append("content_capture_health must be an object")
    else:
        valid_timestamps = health.get("spans_with_valid_timestamps")
        invalid_timestamps = health.get("spans_with_invalid_timestamps")
        spans = verifier.get("latest_trace_spans_chronological")
        span_count = len(spans) if isinstance(spans, list) else 0
        if not isinstance(valid_timestamps, int) or valid_timestamps <= 0:
            issues.append("content_capture_health.spans_with_valid_timestamps must be positive")
        elif valid_timestamps != span_count:
            issues.append(
                "content_capture_health.spans_with_valid_timestamps must match latest_trace_spans_chronological"
            )
        if not isinstance(invalid_timestamps, int) or invalid_timestamps != 0:
            issues.append("content_capture_health.spans_with_invalid_timestamps must be zero")
    return issues


def agents_diagnostic_validation_issues(
    diagnostic: dict[str, Any] | None,
    *,
    expected_project_id: str | None,
    expected_agent_name: str | None,
    expected_task_id: str | None,
) -> list[str]:
    if diagnostic is None:
        return ["Agents diagnostic JSON is missing"]
    issues: list[str] = []
    if diagnostic.get("diagnostic_schema_version") != AGENTS_DIAGNOSTIC_SCHEMA_VERSION:
        issues.append(
            f"diagnostic_schema_version must be {AGENTS_DIAGNOSTIC_SCHEMA_VERSION}"
        )
    if not isinstance(diagnostic.get("generated_at"), (int, float)):
        issues.append("generated_at must be numeric")
    if expected_project_id and diagnostic.get("project_id") != expected_project_id:
        issues.append(f"project_id must be {expected_project_id!r}")
    if expected_agent_name and diagnostic.get("agent_name_filter") != expected_agent_name:
        issues.append(f"agent_name_filter must be {expected_agent_name!r}")

    query_source = diagnostic.get("query_source")
    if not isinstance(query_source, dict):
        issues.append("query_source must be an object from the native W&B Agents API")
        query_source = {}
    expected_pairs = {
        "kind": "wandb_agents_api",
        "api_base_url": WEAVE_AGENTS_API_BASE_URL,
        "agents_endpoint": WEAVE_AGENTS_QUERY_ENDPOINT,
        "spans_endpoint": WEAVE_AGENTS_SPANS_ENDPOINT,
    }
    if expected_project_id:
        expected_pairs["project_id"] = expected_project_id
    if expected_agent_name:
        expected_pairs["agent_name"] = expected_agent_name
    for key, expected in expected_pairs.items():
        if query_source.get(key) != expected:
            issues.append(f"query_source.{key} must be {expected!r}")
    for key in ("agents_count", "spans_count", "matching_span_count", "latest_trace_span_count"):
        if not _int_at_least(query_source.get(key), 1):
            issues.append(f"query_source.{key} must be an integer >= 1")

    if expected_task_id and not any(
        expected_task_id in value
        for value in (
            str(query_source.get("conversation_id") or ""),
            str(query_source.get("conversation_id_contains") or ""),
        )
    ):
        issues.append(
            "query_source conversation_id or conversation_id_contains must scope "
            f"the diagnostic to {expected_task_id!r}"
        )

    latest_trace_id = diagnostic.get("latest_trace_id")
    if not isinstance(latest_trace_id, str) or not latest_trace_id:
        issues.append("latest_trace_id must be a non-empty string")

    spans = diagnostic.get("latest_trace_spans_chronological")
    if isinstance(spans, list):
        latest_count = query_source.get("latest_trace_span_count")
        if isinstance(latest_count, int) and latest_count != len(spans):
            issues.append(
                "query_source.latest_trace_span_count must match "
                "latest_trace_spans_chronological"
            )
    issues.extend(
        latest_trace_span_validation_issues(
            diagnostic,
            expected_agent_name=expected_agent_name,
            expected_task_id=expected_task_id,
        )
    )

    health = diagnostic.get("content_capture_health")
    if not isinstance(health, dict):
        issues.append("content_capture_health must be an object")
        health = {}
    for key in (
        "trace_timestamp_quality_ok",
        "trace_order_ok",
        "trace_user_message_order_ok",
        "trace_final_answer_order_ok",
    ):
        if health.get(key) is not True:
            issues.append(f"content_capture_health.{key} must be true")
    if not _int_at_least(health.get("message_spans_with_input"), 1):
        issues.append("content_capture_health.message_spans_with_input must be >= 1")
    if not _int_at_least(health.get("tool_span_count"), 1):
        issues.append("content_capture_health.tool_span_count must be >= 1")
    if not _int_at_least(health.get("tool_spans_with_content"), 1):
        issues.append("content_capture_health.tool_spans_with_content must be >= 1")
    if not _int_at_least(health.get("final_answer_span_count"), 1):
        issues.append("content_capture_health.final_answer_span_count must be >= 1")
    if health.get("spans_with_invalid_timestamps") != 0:
        issues.append("content_capture_health.spans_with_invalid_timestamps must be zero")

    trace_order = diagnostic.get("trace_order_health")
    if not isinstance(trace_order, dict):
        issues.append("trace_order_health must be an object")
        trace_order = {}
    for key in (
        "timestamp_quality_ok",
        "trace_order_ok",
        "trace_user_message_order_ok",
        "trace_final_answer_order_ok",
    ):
        if trace_order.get(key) is not True:
            issues.append(f"trace_order_health.{key} must be true")
    order_issues = trace_order.get("order_issues")
    if not isinstance(order_issues, list):
        issues.append("trace_order_health.order_issues must be a list")
    elif order_issues:
        issues.append("trace_order_health.order_issues must be empty")
    return issues


def command_failure_kind(command_result: dict[str, Any] | None) -> str | None:
    if not command_result:
        return None
    failure = command_result.get("failure")
    if isinstance(failure, dict):
        kind = failure.get("kind")
        if isinstance(kind, str) and kind:
            return kind
    return None


def classify_failure_text(text: str) -> tuple[str, str] | None:
    lowered = text.lower()
    if "insufficient_quota" in lowered or "exceeded your current quota" in lowered:
        return (
            "provider_quota",
            "provider returned insufficient_quota before a scoreable canary trace was produced",
        )
    if "unknown model" in lowered or "model_not_found" in lowered:
        return ("model_not_found", "OpenClaw could not resolve the requested model id")
    if "authentication" in lowered or "unauthorized" in lowered or "401" in lowered:
        return ("provider_auth", "provider authentication failed before a scoreable canary trace was produced")
    if "rate_limit" in lowered or "429" in lowered:
        return ("provider_rate_limit", "provider rate limit failed the canary before trace verification")
    return None


def infer_failure_from_payloads(
    *,
    command_result: dict[str, Any] | None,
    sidecar: dict[str, Any] | None,
) -> tuple[str, str] | None:
    chunks: list[str] = []
    if isinstance(command_result, dict):
        for key in ("stdout_tail", "stderr_tail"):
            value = command_result.get(key)
            if isinstance(value, str):
                chunks.append(value)
    if isinstance(sidecar, dict):
        for key in ("stdout", "stderr"):
            value = sidecar.get(key)
            if isinstance(value, str):
                chunks.append(value)
    return classify_failure_text("\n".join(chunks))


def command_failure_detail(command_result: dict[str, Any] | None) -> str:
    if not command_result:
        return ""
    failure = command_result.get("failure")
    if isinstance(failure, dict):
        detail = failure.get("detail")
        if isinstance(detail, str):
            return detail
    return "OpenClaw command failed before the canary was scoreable"


def stable_command_sha256(command: Any) -> str:
    if not isinstance(command, list):
        return ""
    payload = json.dumps(command, ensure_ascii=False, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _same_path(left: Any, right: Any) -> bool:
    if not isinstance(left, str) or not isinstance(right, str) or not left or not right:
        return False
    try:
        return repo_relative_path(left).resolve() == repo_relative_path(right).resolve()
    except OSError:
        return left == right


def command_result_contract_issues(
    *,
    plan: dict[str, Any],
    command_result: dict[str, Any] | None,
    task_id: str,
    plan_file: Path,
) -> list[str]:
    if command_result is None:
        return []
    issues: list[str] = []
    expected_fields = {
        "task_id": task_id,
        "canary_id": plan.get("canary_id"),
        "model": plan.get("model"),
        "thinking": plan.get("thinking"),
    }
    for field, expected in expected_fields.items():
        if isinstance(expected, str) and expected:
            if command_result.get(field) != expected:
                issues.append(f"command_result.{field} must match plan.{field}")

    if not _same_path(command_result.get("plan_file"), str(plan_file)):
        issues.append("command_result.plan_file must match the gate plan_file")
    for field in ("prompt_file", "expected_sidecar"):
        expected_path = plan.get(field)
        if isinstance(expected_path, str) and expected_path:
            if not _same_path(command_result.get(field), expected_path):
                issues.append(f"command_result.{field} must match plan.{field}")

    expected_sha = plan.get("run_command_sha256")
    if not isinstance(expected_sha, str) or not expected_sha:
        expected_sha = stable_command_sha256(plan.get("run_command"))
    observed_sha = command_result.get("run_command_sha256")
    if not isinstance(observed_sha, str) or not observed_sha:
        issues.append("command_result.run_command_sha256 must be a non-empty string")
    elif expected_sha and observed_sha != expected_sha:
        issues.append("command_result.run_command_sha256 must match the plan run command")

    observed_command = command_result.get("run_command")
    if not isinstance(observed_command, list) or not observed_command:
        issues.append("command_result.run_command must be a non-empty list")
    elif stable_command_sha256(observed_command) != observed_sha:
        issues.append("command_result.run_command must hash to command_result.run_command_sha256")
    return issues


def status_from_verifier(
    verifier: dict[str, Any] | None,
    verifier_error: str | None,
    *,
    expected_project_id: str | None,
    expected_agent_name: str | None,
    expected_task_id: str | None,
    expected_request_models: list[str],
) -> tuple[str, str]:
    if verifier is None:
        return "trace_missing", verifier_error or "Weave verifier JSON is missing"
    if verifier.get("ok") is True:
        issues = passing_verifier_validation_issues(
            verifier,
            expected_project_id=expected_project_id,
            expected_agent_name=expected_agent_name,
            expected_task_id=expected_task_id,
            expected_request_models=expected_request_models,
        )
        if issues:
            return "weave_verifier_schema_invalid", "; ".join(issues)
        return "passed", "fresh Weave Agents canary has trace, content, tool content, and usage"

    names = failed_check_names(verifier)
    if names & {"spans_present", "latest_trace", "trace_span_count", "message_span_count"}:
        return "trace_missing", "fresh canary trace is absent or incomplete in the Agents API"
    if "message_content_capture" in names:
        return "content_missing", "message content is not visible in the Agents API"
    if "tool_span_count" in names:
        return "tool_trace_missing", "tool execution span is not visible in the Agents API"
    if "tool_content_capture" in names:
        return "tool_content_missing", "tool arguments/results are not visible in the Agents API"
    if "required_text_capture" in names:
        return "canary_text_missing", "required canary id or expected answer text is not visible in the Agents API"
    if "usage" in names:
        return "usage_missing", "token usage is missing from the Agents API"
    if "request_model" in names:
        return "request_model_missing", "request_model is missing or does not match the canary model"
    if "trace_timestamp_quality" in names:
        return "trace_order_invalid", "span timestamps are missing or invalid in the Agents API"
    if "trace_final_answer_order" in names:
        return "trace_order_invalid", "tool span appears after a final-answer message in the Agents API"
    if "trace_order" in names:
        return "trace_order_invalid", "tool span ordering is invalid in the Agents API"
    if "trace_errors" in names:
        return "trace_error", "latest canary trace contains error spans"
    return "weave_verification_failed", "Weave Agents verifier failed"


def status_from_command_failure(command_result: dict[str, Any], failure_kind: str | None, failure_detail: str) -> tuple[str, str]:
    kind = failure_kind
    if kind == "external_action_approval_missing":
        return "external_action_approval_missing", failure_detail
    if kind == "nemoclaw_config_preflight_failed":
        return "nemoclaw_config_preflight_failed", failure_detail
    if kind == "model_not_found":
        return "model_configuration_failure", failure_detail
    if kind and kind.startswith("provider_"):
        return "provider_failure", failure_detail
    return "execution_failure", failure_detail


def recommendation(status: str, failure_kind: str | None) -> str:
    if status == "passed":
        return "Gate passed; this tracing path can be used for the next production canary."
    if status == "not_run":
        return "Run the content canary with --execute when paid inference is intentionally approved."
    if status == "incomplete":
        return "Rerun the canary or inspect why OpenClaw exited before writing a command result."
    if status == "external_action_approval_missing":
        return "Review and approve the source-bound external action packet, then rerun the live content canary."
    if status == "nemoclaw_config_preflight_failed":
        return "Fix the sandbox OpenClaw provider/model/Weave plugin config, then rerun the live content canary."
    if status == "provider_failure":
        if failure_kind == "provider_quota":
            return "Use a provider/model with available quota, then rerun the same canary."
        if failure_kind == "provider_auth":
            return "Fix provider credentials visible to the OpenClaw gateway, then rerun."
        if failure_kind == "provider_rate_limit":
            return "Wait for rate-limit recovery or switch to an approved test model, then rerun."
        return "Resolve the provider failure and rerun the same canary."
    if status == "model_configuration_failure":
        return "Fix the OpenClaw model id/provider mapping, then rerun the canary."
    if status == "command_result_contract_invalid":
        return "Regenerate the canary command result from the matching plan, then rerun the gate."
    if status in {"trace_missing", "content_missing", "tool_trace_missing", "tool_content_missing", "canary_text_missing"}:
        return "Investigate native OpenClaw/Weave content capture and rerun a fresh canary before production runs."
    if status == "usage_missing":
        return "Confirm token usage is exported by the provider/OpenClaw integration, then rerun."
    if status == "trace_order_invalid":
        return "Inspect the native Weave span ordering; production traces must reflect actual conversation order."
    if status == "trace_error":
        return "Inspect the trace error spans and rerun after the runtime issue is fixed."
    if status == "weave_verifier_schema_invalid":
        return "Regenerate the Weave Agents verifier JSON with the current verifier, then rerun the content canary gate."
    if status == "agents_diagnostic_invalid":
        return "Regenerate the check-agents diagnostic JSON for the same canary task, then rerun the content canary gate."
    return "Inspect the command result and verifier JSON, then rerun the canary after fixing the failing gate."


def build_gate_summary(
    *,
    plan_file: Path,
    command_result_file: Path | None = None,
    verifier_json: Path | None = None,
    agents_diagnostic_json: Path | None = None,
) -> dict[str, Any]:
    plan, plan_error = read_json(plan_file)
    if plan is None:
        return {
            "ok": False,
            "gate": GATE_NAME,
            "status": "plan_missing",
            "detail": plan_error,
            "plan_file": str(plan_file),
            "generated_at": time.time(),
            "recommended_next_action": "Regenerate the canary plan before running this gate.",
        }

    task_id = task_id_from_plan(plan, plan_file)
    command_path = command_result_file or default_command_result_file(plan_file, task_id)
    verifier_path = verifier_json or find_latest_verifier_json(plan_file, task_id)
    diagnostic_path_value = plan.get("agents_diagnostic_file")
    diagnostic_path = (
        agents_diagnostic_json
        or (
            repo_relative_path(diagnostic_path_value)
            if isinstance(diagnostic_path_value, str) and diagnostic_path_value
            else None
        )
    )
    command_result, command_error = read_json(command_path)
    verifier, verifier_error = read_json(verifier_path)
    diagnostic, _diagnostic_error = read_json(diagnostic_path)
    expected_agent_name = plan.get("agent_name") if isinstance(plan.get("agent_name"), str) else None
    entity = plan.get("entity") if isinstance(plan.get("entity"), str) else None
    project = plan.get("project") if isinstance(plan.get("project"), str) else None
    expected_project_id = f"{entity}/{project}" if entity and project else None
    expected_request_models = expected_request_models_from_plan(plan)
    sidecar_path_value = plan.get("expected_sidecar")
    sidecar_path = repo_relative_path(sidecar_path_value) if isinstance(sidecar_path_value, str) else None
    sidecar, _sidecar_error = read_json(sidecar_path)
    command_contract_issues = command_result_contract_issues(
        plan=plan,
        command_result=command_result,
        task_id=task_id,
        plan_file=plan_file,
    )

    will_call_paid_model_api = bool(plan.get("will_call_paid_model_api"))
    if isinstance(command_result, dict) and "paid_api_attempted" in command_result:
        paid_api_attempted = bool(command_result.get("paid_api_attempted"))
    else:
        paid_api_attempted = command_result is not None
    failure_kind = command_failure_kind(command_result)
    failure_detail = command_failure_detail(command_result)
    if command_result is not None and command_result.get("ok") is not True and not failure_kind:
        inferred = infer_failure_from_payloads(command_result=command_result, sidecar=sidecar)
        if inferred:
            failure_kind, failure_detail = inferred

    if command_result is None:
        if will_call_paid_model_api:
            status = "incomplete"
            detail = command_error or "command result is missing"
        else:
            status = "not_run"
            detail = "prepare-only canary; no paid model call was attempted"
    elif command_result.get("ok") is not True:
        status, detail = status_from_command_failure(command_result, failure_kind, failure_detail)
    elif command_contract_issues:
        status = "command_result_contract_invalid"
        detail = "; ".join(command_contract_issues)
    else:
        status, detail = status_from_verifier(
            verifier,
            verifier_error,
            expected_project_id=expected_project_id,
            expected_agent_name=expected_agent_name,
            expected_task_id=task_id,
            expected_request_models=expected_request_models,
        )
        if status == "passed":
            diagnostic_issues = agents_diagnostic_validation_issues(
                diagnostic,
                expected_project_id=expected_project_id,
                expected_agent_name=expected_agent_name,
                expected_task_id=task_id,
            )
            if diagnostic_issues:
                status = "agents_diagnostic_invalid"
                detail = "; ".join(diagnostic_issues)
        if status == "passed":
            nemoclaw_preflight_issues = (
                nemoclaw_openclaw_config_preflight_validation_issues(plan)
            )
            if nemoclaw_preflight_issues:
                status = "nemoclaw_config_preflight_invalid"
                detail = "; ".join(nemoclaw_preflight_issues)

    ok = status == "passed"
    verifier_validation_issues = (
        passing_verifier_validation_issues(
            verifier,
            expected_project_id=expected_project_id,
            expected_agent_name=expected_agent_name,
            expected_task_id=task_id,
            expected_request_models=expected_request_models,
        )
        if isinstance(verifier, dict) and verifier.get("ok") is True
        else []
    )
    request_model = request_model_evidence(verifier)
    agents_diagnostic_validation = (
        agents_diagnostic_validation_issues(
            diagnostic,
            expected_project_id=expected_project_id,
            expected_agent_name=expected_agent_name,
            expected_task_id=task_id,
        )
        if isinstance(diagnostic, dict)
        else []
    )
    return {
        "ok": ok,
        "gate": GATE_NAME,
        "status": status,
        "detail": detail,
        "failure_kind": failure_kind,
        "model": plan.get("model"),
        "thinking": plan.get("thinking"),
        "canary_id": plan.get("canary_id"),
        "task_id": task_id,
        "agent_name": plan.get("agent_name"),
        "project": plan.get("project"),
        "entity": plan.get("entity"),
        "nemoclaw": plan.get("nemoclaw") if isinstance(plan.get("nemoclaw"), dict) else {},
        "nemoclaw_openclaw_config_preflight": (
            plan.get("nemoclaw_openclaw_config_preflight")
            if isinstance(plan.get("nemoclaw_openclaw_config_preflight"), dict)
            else {}
        ),
        "expected_request_models": expected_request_models,
        "observed_request_models": request_model["observed_request_models"],
        "span_request_models": request_model["span_request_models"],
        "request_model_proven": status == "passed"
        and bool(expected_request_models)
        and not any(
            issue
            for issue in verifier_validation_issues
            if "request_model" in issue
            or "expected_request_models" in issue
            or "latest_trace_spans_chronological request_model" in issue
        ),
        "will_call_paid_model_api": will_call_paid_model_api,
        "paid_api_attempted": paid_api_attempted,
        "command_ok": command_result.get("ok") if isinstance(command_result, dict) else None,
        "command_returncode": command_result.get("returncode") if isinstance(command_result, dict) else None,
        "command_result_contract_issues": command_contract_issues,
        "weave_verifier_ok": verifier.get("ok") if isinstance(verifier, dict) else None,
        "weave_verifier_schema_version": (
            verifier.get("verification_schema_version") if isinstance(verifier, dict) else None
        ),
        "weave_verifier_latest_trace_id": (
            verifier.get("latest_trace_id") if isinstance(verifier, dict) else None
        ),
        "weave_verifier_validation_issues": verifier_validation_issues,
        "agents_diagnostic_ok": (
            isinstance(diagnostic, dict) and not agents_diagnostic_validation
        ),
        "agents_diagnostic_schema_version": (
            diagnostic.get("diagnostic_schema_version") if isinstance(diagnostic, dict) else None
        ),
        "agents_diagnostic_latest_trace_id": (
            diagnostic.get("latest_trace_id") if isinstance(diagnostic, dict) else None
        ),
        "agents_diagnostic_validation_issues": agents_diagnostic_validation,
        "content_capture_health": verifier.get("content_capture_health") if isinstance(verifier, dict) else None,
        "failed_checks": failed_checks(verifier),
        "paths": {
            "plan_file": str(plan_file),
            "command_result_file": str(command_path),
            "command_result_exists": command_path.exists(),
            "verifier_json": str(verifier_path) if verifier_path else None,
            "verifier_json_exists": bool(verifier_path and verifier_path.exists()),
            "agents_diagnostic_json": str(diagnostic_path) if diagnostic_path else None,
            "agents_diagnostic_json_exists": bool(diagnostic_path and diagnostic_path.exists()),
            "expected_sidecar": plan.get("expected_sidecar"),
            "prompt_file": plan.get("prompt_file"),
        },
        "generated_at": time.time(),
        "recommended_next_action": recommendation(status, failure_kind),
    }


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan-file", type=Path)
    parser.add_argument("--canary-id")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--command-result-file", type=Path)
    parser.add_argument("--verifier-json", type=Path)
    parser.add_argument("--agents-diagnostic-json", type=Path)
    parser.add_argument("--json", type=Path, help="Optional path to write gate JSON.")
    parser.add_argument("--fail-on-not-ok", action="store_true")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    plan_file = resolve_plan_file(args)
    summary = build_gate_summary(
        plan_file=plan_file,
        command_result_file=args.command_result_file,
        verifier_json=args.verifier_json,
        agents_diagnostic_json=args.agents_diagnostic_json,
    )
    if args.json:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    if args.fail_on_not_ok and not summary["ok"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
