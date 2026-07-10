#!/usr/bin/env python3
"""
Verify that Taiwan agentic benchmark traces are visible in W&B Weave Agents.

This script is read-only. It checks the Agents API rather than local files so
that a completed agentic benchmark cannot be treated as production-ready unless
the trace users inspect in W&B is present and usable.
"""

from __future__ import annotations

import argparse
import base64
import json
import os
import time
import urllib.error
import urllib.request
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_ENV_FILE = REPO_ROOT / ".env"
DEFAULT_AGENT_NAME = "nejumi-taiwan-openclaw"
VERIFICATION_SCHEMA_VERSION = 1
AGENTS_API_BASE_URL = "https://trace.wandb.ai"
AGENTS_QUERY_ENDPOINT = "/agents/query"
AGENTS_SPANS_QUERY_ENDPOINT = "/agents/spans/query"
AGENTS_TRACES_CHAT_ENDPOINT = "/agents/traces/chat"
FINAL_ANSWER_MARKERS = (
    "ANSWER:",
    "FINAL ANSWER",
    "Final answer",
    "CANARY_RESULT",
)
TOOL_CALL_TYPE_MARKERS = {
    "toolCall",
    "tool_call",
    "toolUse",
    "tool_use",
    "function_call",
}
TOOL_CALL_KEYS = (
    "tool_call",
    "tool_calls",
    "toolCalls",
    "toolCall",
    "toolUse",
    "tool_use",
    "function_call",
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


def agents_api_post(env: dict[str, str], path: str, payload: dict[str, Any]) -> dict[str, Any]:
    api_key = env.get("WANDB_API_KEY")
    if not api_key:
        raise SystemExit("WANDB_API_KEY is required for W&B Agents API checks")
    token = base64.b64encode(f"api:{api_key}".encode("utf-8")).decode("ascii")
    request = urllib.request.Request(
        f"{AGENTS_API_BASE_URL}{path}",
        data=json.dumps(payload).encode("utf-8"),
        method="POST",
        headers={
            "Authorization": f"Basic {token}",
            "Content-Type": "application/json",
        },
    )
    try:
        with urllib.request.urlopen(request, timeout=30) as response:
            return json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")[:1000]
        raise SystemExit(f"W&B Agents API check failed: HTTP {exc.code}: {detail}") from exc


def _ok_check(name: str, detail: str, **extra: Any) -> dict[str, Any]:
    return {"name": name, "ok": True, "detail": detail, **extra}


def _fail_check(name: str, detail: str, **extra: Any) -> dict[str, Any]:
    return {"name": name, "ok": False, "detail": detail, **extra}


def _verification_header(ok: bool) -> dict[str, Any]:
    return {
        "schema_version": VERIFICATION_SCHEMA_VERSION,
        "verification_schema_version": VERIFICATION_SCHEMA_VERSION,
        "status": "passed" if ok else "failed",
    }


def _message_count(value: Any) -> int:
    return len(value) if isinstance(value, list) else 0


def _has_message_content(span: dict[str, Any]) -> bool:
    return bool(span.get("input_messages")) or bool(span.get("output_messages"))


def _has_input_message_content(span: dict[str, Any]) -> bool:
    return bool(span.get("input_messages"))


def _has_tool_content(span: dict[str, Any]) -> bool:
    return bool(span.get("tool_call_arguments")) or bool(span.get("tool_call_result"))


def _jsonish_text(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    try:
        return json.dumps(value, ensure_ascii=False, sort_keys=True)
    except (TypeError, ValueError):
        return str(value)


def _span_visible_text(span: dict[str, Any]) -> str:
    parts = [
        span.get("input_messages"),
        span.get("output_messages"),
        span.get("tool_call_arguments"),
        span.get("tool_call_result"),
    ]
    return "\n".join(_jsonish_text(part) for part in parts if part)


def _chat_messages(trace_chat_payload: dict[str, Any] | None) -> list[dict[str, Any]]:
    if not isinstance(trace_chat_payload, dict):
        return []
    messages = trace_chat_payload.get("messages", [])
    if not isinstance(messages, list):
        return []
    return [message for message in messages if isinstance(message, dict)]


def _chat_component_text(message: dict[str, Any], component: str, fields: tuple[str, ...]) -> str:
    payload = message.get(component)
    if not isinstance(payload, dict):
        return ""
    parts = [_jsonish_text(payload.get(field)) for field in fields if payload.get(field)]
    return "\n".join(part for part in parts if part)


def _chat_user_text(message: dict[str, Any]) -> str:
    return _chat_component_text(message, "user_message", ("text", "content", "message", "prompt"))


def _chat_assistant_text(message: dict[str, Any]) -> str:
    return _chat_component_text(
        message,
        "assistant_message",
        ("text", "content", "message", "response"),
    )


def _chat_tool_content(message: dict[str, Any]) -> str:
    return _chat_component_text(
        message,
        "tool_call",
        ("tool_arguments", "tool_result", "arguments", "result", "input", "output"),
    )


def _chat_agent_start_text(message: dict[str, Any]) -> str:
    return _chat_component_text(
        message,
        "agent_start",
        ("system_instructions", "tool_definitions", "tools"),
    )


def _chat_visible_text(message: dict[str, Any]) -> str:
    parts = [
        _chat_user_text(message),
        _chat_assistant_text(message),
        _chat_tool_content(message),
        _chat_agent_start_text(message),
    ]
    return "\n".join(part for part in parts if part)


def _chat_timestamp(message: dict[str, Any]) -> float | None:
    return _parse_span_timestamp(message.get("started_at"))


def _chat_is_tool_call(message: dict[str, Any]) -> bool:
    return isinstance(message.get("tool_call"), dict)


def _contains_tool_call(value: Any) -> bool:
    if isinstance(value, list):
        return any(_contains_tool_call(item) for item in value)
    if not isinstance(value, dict):
        return False
    type_value = value.get("type")
    if isinstance(type_value, str) and type_value in TOOL_CALL_TYPE_MARKERS:
        return True
    if any(value.get(key) for key in TOOL_CALL_KEYS):
        return True
    return any(_contains_tool_call(item) for item in value.values())


def _chat_is_final_answer(message: dict[str, Any]) -> bool:
    if _chat_is_tool_call(message) or _contains_tool_call(message.get("assistant_message")):
        return False
    assistant_text = _chat_assistant_text(message)
    return any(marker in assistant_text for marker in FINAL_ANSWER_MARKERS)


def summarize_chat_message(message: dict[str, Any]) -> dict[str, Any]:
    user_text = _chat_user_text(message)
    assistant_text = _chat_assistant_text(message)
    tool_content = _chat_tool_content(message)
    return {
        "type": message.get("type"),
        "started_at": message.get("started_at"),
        "has_user_message": bool(user_text),
        "has_assistant_message": bool(assistant_text),
        "has_tool_call": _chat_is_tool_call(message),
        "has_tool_content": bool(tool_content),
        "is_final_answer": _chat_is_final_answer(message),
        "user_text_preview": user_text[:500],
        "assistant_text_preview": assistant_text[:500],
        "tool_content_preview": tool_content[:500],
    }


def _span_output_text(span: dict[str, Any]) -> str:
    return _jsonish_text(span.get("output_messages"))


def _is_final_answer_span(span: dict[str, Any]) -> bool:
    if span.get("operation_name") not in {"chat", "invoke_agent"}:
        return False
    if _contains_tool_call(span.get("output_messages")):
        return False
    output_text = _span_output_text(span)
    return any(marker in output_text for marker in FINAL_ANSWER_MARKERS)


def summarize_span(span: dict[str, Any]) -> dict[str, Any]:
    input_messages = span.get("input_messages")
    output_messages = span.get("output_messages")
    return {
        "started_at": span.get("started_at"),
        "ended_at": span.get("ended_at"),
        "span_name": span.get("span_name"),
        "operation_name": span.get("operation_name"),
        "agent_name": span.get("agent_name"),
        "provider_name": span.get("provider_name"),
        "request_model": span.get("request_model"),
        "conversation_id": span.get("conversation_id"),
        "trace_id": span.get("trace_id"),
        "span_id": span.get("span_id"),
        "parent_span_id": span.get("parent_span_id"),
        "tool_name": span.get("tool_name"),
        "error_type": span.get("error_type"),
        "has_input_messages": bool(input_messages),
        "has_output_messages": bool(output_messages),
        "input_message_count": _message_count(input_messages),
        "output_message_count": _message_count(output_messages),
        "has_tool_call_arguments": bool(span.get("tool_call_arguments")),
        "has_tool_call_result": bool(span.get("tool_call_result")),
    }


def _chronological_key(span: dict[str, Any]) -> tuple[str, str, str]:
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


def _span_timestamps(span: dict[str, Any]) -> tuple[float | None, float | None]:
    return (
        _parse_span_timestamp(span.get("started_at")),
        _parse_span_timestamp(span.get("ended_at")),
    )


def _timestamp_quality_issues(spans: list[dict[str, Any]]) -> list[dict[str, Any]]:
    issues: list[dict[str, Any]] = []
    for index, span in enumerate(spans, start=1):
        started_at = span.get("started_at")
        ended_at = span.get("ended_at")
        started_ts, ended_ts = _span_timestamps(span)
        if started_ts is None or ended_ts is None:
            issues.append(
                {
                    "index": index,
                    "span_id": span.get("span_id"),
                    "started_at": started_at,
                    "ended_at": ended_at,
                    "error": "missing_or_invalid_timestamp",
                }
            )
            continue
        if ended_ts < started_ts:
            issues.append(
                {
                    "index": index,
                    "span_id": span.get("span_id"),
                    "started_at": started_at,
                    "ended_at": ended_at,
                    "error": "ended_before_started",
                }
            )
    return issues


def _span_start(span: dict[str, Any]) -> float | None:
    return _span_timestamps(span)[0]


def _span_end(span: dict[str, Any]) -> float | None:
    return _span_timestamps(span)[1]


def _latest_trace_id(raw_spans: list[dict[str, Any]]) -> str | None:
    trace_latest_keys: dict[str, tuple[str, str, str]] = {}
    for span in raw_spans:
        trace_id = span.get("trace_id")
        if not isinstance(trace_id, str) or not trace_id:
            continue
        key = _chronological_key(span)
        if trace_id not in trace_latest_keys or key > trace_latest_keys[trace_id]:
            trace_latest_keys[trace_id] = key
    if not trace_latest_keys:
        return None
    return max(trace_latest_keys.items(), key=lambda item: (item[1], item[0]))[0]


def _conversation_matches(
    span: dict[str, Any],
    *,
    conversation_id: str | None,
    conversation_id_contains: str | None,
) -> bool:
    value = str(span.get("conversation_id") or "")
    if conversation_id is not None and value != conversation_id:
        return False
    if conversation_id_contains is not None and conversation_id_contains not in value:
        return False
    return True


def _agent_matches(
    span: dict[str, Any],
    *,
    agent_name: str,
    conversation_id: str | None,
    conversation_id_contains: str | None,
) -> bool:
    if not agent_name:
        return True
    value = span.get("agent_name")
    if value == agent_name:
        return True
    # Dynamic OpenClaw task agents can currently appear in the Agents spans API
    # with an empty agent_name. If the conversation id matches exactly, keep the
    # span and let the content/order checks decide whether the trace is usable.
    return (
        value in {"", None}
        and bool(conversation_id or conversation_id_contains)
        and _conversation_matches(
            span,
            conversation_id=conversation_id,
            conversation_id_contains=conversation_id_contains,
        )
    )


def _matching_spans(
    spans_payload: dict[str, Any],
    *,
    agent_name: str,
    conversation_id: str | None,
    conversation_id_contains: str | None,
) -> list[dict[str, Any]]:
    spans = spans_payload.get("spans", [])
    if not isinstance(spans, list):
        return []
    return [
        span
        for span in spans
        if isinstance(span, dict)
        and _agent_matches(
            span,
            agent_name=agent_name,
            conversation_id=conversation_id,
            conversation_id_contains=conversation_id_contains,
        )
        and _conversation_matches(
            span,
            conversation_id=conversation_id,
            conversation_id_contains=conversation_id_contains,
        )
    ]


def verify_agents_payload(
    agents_payload: dict[str, Any],
    spans_payload: dict[str, Any],
    *,
    trace_chat_payload: dict[str, Any] | None = None,
    entity: str,
    project: str,
    agent_name: str,
    min_agent_invocations: int = 1,
    min_trace_spans: int = 1,
    min_message_spans: int = 1,
    require_content: bool = True,
    require_input_message: bool | None = None,
    require_tool_span: bool = False,
    require_tool_content: bool = False,
    require_usage: bool = False,
    require_no_errors: bool = True,
    required_texts: list[str] | None = None,
    expected_request_models: list[str] | None = None,
    conversation_id: str | None = None,
    conversation_id_contains: str | None = None,
) -> dict[str, Any]:
    if require_input_message is None:
        require_input_message = require_content
    checks: list[dict[str, Any]] = []
    raw_spans = _matching_spans(
        spans_payload,
        agent_name=agent_name,
        conversation_id=conversation_id,
        conversation_id_contains=conversation_id_contains,
    )
    agents = [
        agent
        for agent in agents_payload.get("agents", [])
        if isinstance(agent, dict) and agent.get("agent_name") == agent_name
    ]
    best_agent = max(
        agents,
        key=lambda agent: int(agent.get("invocation_count") or 0),
        default=None,
    )
    if best_agent:
        agent_input_tokens = int(best_agent.get("total_input_tokens") or 0)
        agent_output_tokens = int(best_agent.get("total_output_tokens") or 0)
        checks.append(
            _ok_check(
                "agent_present",
                "agent is present",
                agent_name=agent_name,
                invocation_count=best_agent.get("invocation_count"),
                span_count=best_agent.get("span_count"),
            )
        )
        invocation_count = int(best_agent.get("invocation_count") or 0)
        if invocation_count < min_agent_invocations:
            checks.append(
                _fail_check(
                    "agent_invocation_count",
                    "agent invocation count is below minimum",
                    value=invocation_count,
                    expected_min=min_agent_invocations,
                )
            )
        else:
            checks.append(
                _ok_check(
                    "agent_invocation_count",
                    "agent invocation count meets minimum",
                    value=invocation_count,
                    expected_min=min_agent_invocations,
                )
            )

    if not best_agent and raw_spans and bool(conversation_id or conversation_id_contains):
        agent_input_tokens = 0
        agent_output_tokens = 0
        checks.append(
            _ok_check(
                "agent_present",
                "agent summary is absent, but matching conversation spans are present",
                agent_name=agent_name,
                matching_span_count=len(raw_spans),
                agent_name_missing_on_spans=True,
            )
        )
        checks.append(
            _ok_check(
                "agent_invocation_count",
                "agent invocation summary is unavailable for this dynamic conversation; span presence is used instead",
                value=None,
                expected_min=min_agent_invocations,
                matching_span_count=len(raw_spans),
            )
        )
    elif not best_agent:
        checks.append(_fail_check("agent_present", "agent is not present", agent_name=agent_name))
        agent_input_tokens = 0
        agent_output_tokens = 0

    if not raw_spans:
        checks.append(
            _fail_check(
                "spans_present",
                "no matching spans returned by Agents API",
                agent_name=agent_name,
                conversation_id=conversation_id,
                conversation_id_contains=conversation_id_contains,
            )
        )
        latest_trace_id = None
        latest_trace_spans: list[dict[str, Any]] = []
    else:
        checks.append(
            _ok_check(
                "spans_present",
                "matching spans returned by Agents API",
                span_count=len(raw_spans),
            )
        )
        latest_trace_id = _latest_trace_id(raw_spans)
        latest_trace_spans = [
            span for span in raw_spans if latest_trace_id and span.get("trace_id") == latest_trace_id
        ]

    latest_trace_spans_chronological = sorted(latest_trace_spans, key=_chronological_key)
    timestamp_issues = _timestamp_quality_issues(latest_trace_spans_chronological)
    trace_input_tokens = sum(int(span.get("input_tokens") or 0) for span in latest_trace_spans_chronological)
    trace_output_tokens = sum(int(span.get("output_tokens") or 0) for span in latest_trace_spans_chronological)
    if not latest_trace_id:
        checks.append(_fail_check("latest_trace", "latest trace id is missing"))
    else:
        checks.append(_ok_check("latest_trace", "latest trace id is present", trace_id=latest_trace_id))

    if require_usage:
        if agent_input_tokens + agent_output_tokens + trace_input_tokens + trace_output_tokens <= 0:
            checks.append(
                _fail_check(
                    "usage",
                    "agent and latest trace token usage are missing or zero",
                    agent_input_tokens=agent_input_tokens,
                    agent_output_tokens=agent_output_tokens,
                    trace_input_tokens=trace_input_tokens,
                    trace_output_tokens=trace_output_tokens,
                )
            )
        else:
            checks.append(
                _ok_check(
                    "usage",
                    "token usage is present on the agent summary or latest trace spans",
                    agent_input_tokens=agent_input_tokens,
                    agent_output_tokens=agent_output_tokens,
                    trace_input_tokens=trace_input_tokens,
                    trace_output_tokens=trace_output_tokens,
                )
            )

    if len(latest_trace_spans_chronological) < min_trace_spans:
        checks.append(
            _fail_check(
                "trace_span_count",
                "latest trace has fewer spans than required",
                value=len(latest_trace_spans_chronological),
                expected_min=min_trace_spans,
            )
        )
    else:
        checks.append(
            _ok_check(
                "trace_span_count",
                "latest trace has enough spans",
                value=len(latest_trace_spans_chronological),
                expected_min=min_trace_spans,
            )
        )

    message_spans = [
        span
        for span in latest_trace_spans_chronological
        if span.get("operation_name") in {"chat", "invoke_agent"}
    ]
    tool_spans = [
        span
        for span in latest_trace_spans_chronological
        if span.get("operation_name") == "execute_tool"
    ]
    message_spans_with_content = [span for span in message_spans if _has_message_content(span)]
    message_spans_with_input = [
        span for span in message_spans if _has_input_message_content(span)
    ]
    tool_spans_with_content = [span for span in tool_spans if _has_tool_content(span)]
    final_answer_spans = [span for span in message_spans if _is_final_answer_span(span)]
    trace_chat_messages = _chat_messages(trace_chat_payload)
    chat_messages_with_content = [
        message
        for message in trace_chat_messages
        if _chat_user_text(message) or _chat_assistant_text(message)
    ]
    chat_messages_with_input = [
        message for message in trace_chat_messages if _chat_user_text(message)
    ]
    chat_tool_calls = [
        message for message in trace_chat_messages if _chat_is_tool_call(message)
    ]
    chat_tool_calls_with_content = [
        message for message in chat_tool_calls if _chat_tool_content(message)
    ]
    chat_final_answer_messages = [
        message for message in trace_chat_messages if _chat_is_final_answer(message)
    ]
    message_content_count = len(message_spans_with_content) + len(chat_messages_with_content)
    message_input_count = len(message_spans_with_input) + len(chat_messages_with_input)
    tool_content_count = len(tool_spans_with_content) + len(chat_tool_calls_with_content)
    final_answer_count = len(final_answer_spans) + len(chat_final_answer_messages)
    visible_trace_text = "\n".join(
        text
        for text in [
            "\n".join(_span_visible_text(span) for span in latest_trace_spans_chronological),
            "\n".join(_chat_visible_text(message) for message in trace_chat_messages),
        ]
        if text
    )
    expected_request_models = [
        model.strip() for model in (expected_request_models or []) if model.strip()
    ]
    observed_request_models = sorted(
        {
            str(span.get("request_model")).strip()
            for span in latest_trace_spans_chronological
            if str(span.get("request_model") or "").strip()
        }
        | {
            str(payload.get("model")).strip()
            for message in trace_chat_messages
            for payload in (
                message.get("agent_start"),
                message.get("assistant_message"),
            )
            if isinstance(payload, dict) and str(payload.get("model") or "").strip()
        }
    )
    project_id = f"{entity}/{project}"

    if timestamp_issues:
        checks.append(
            _fail_check(
                "trace_timestamp_quality",
                "one or more latest trace spans have missing, invalid, or reversed timestamps",
                issue_count=len(timestamp_issues),
                issues=timestamp_issues[:10],
            )
        )
    else:
        checks.append(
            _ok_check(
                "trace_timestamp_quality",
                "all latest trace spans have parseable start/end timestamps",
                span_count=len(latest_trace_spans_chronological),
            )
        )

    if len(message_spans) < min_message_spans:
        checks.append(
            _fail_check(
                "message_span_count",
                "latest trace has fewer message spans than required",
                value=len(message_spans),
                expected_min=min_message_spans,
            )
        )
    else:
        checks.append(
            _ok_check(
                "message_span_count",
                "latest trace has enough message spans",
                value=len(message_spans),
                expected_min=min_message_spans,
            )
        )

    if require_content:
        if message_content_count <= 0:
            checks.append(
                _fail_check(
                    "message_content_capture",
                    "Agents API does not expose input/output message content",
                    message_span_count=len(message_spans),
                    message_spans_with_content=message_content_count,
                    chat_messages_with_content=len(chat_messages_with_content),
                )
            )
        else:
            checks.append(
                _ok_check(
                    "message_content_capture",
                    "message content is visible in Agents API",
                    message_span_count=len(message_spans),
                    message_spans_with_content=message_content_count,
                    chat_messages_with_content=len(chat_messages_with_content),
                )
            )

    if require_input_message:
        if message_input_count <= 0:
            checks.append(
                _fail_check(
                    "input_message_capture",
                    "Agents API does not expose the user/problem input",
                    message_span_count=len(message_spans),
                    message_spans_with_input=message_input_count,
                    chat_messages_with_input=len(chat_messages_with_input),
                )
            )
        else:
            checks.append(
                _ok_check(
                    "input_message_capture",
                    "user/problem input is visible in Agents API",
                    message_span_count=len(message_spans),
                    message_spans_with_input=message_input_count,
                    chat_messages_with_input=len(chat_messages_with_input),
                )
            )

    required_texts = [text for text in (required_texts or []) if text]
    if required_texts:
        missing_required_texts = [
            text for text in required_texts if text not in visible_trace_text
        ]
        if missing_required_texts:
            checks.append(
                _fail_check(
                    "required_text_capture",
                    "latest trace does not expose all required canary text",
                    missing_required_texts=missing_required_texts,
                    required_text_count=len(required_texts),
                    visible_text_length=len(visible_trace_text),
                )
            )
        else:
            checks.append(
                _ok_check(
                    "required_text_capture",
                    "latest trace exposes all required canary text",
                    required_text_count=len(required_texts),
                    visible_text_length=len(visible_trace_text),
                )
            )

    if expected_request_models:
        if not observed_request_models:
            checks.append(
                _fail_check(
                    "request_model",
                    "latest trace does not expose request_model on any span",
                    expected_request_models=expected_request_models,
                    observed_request_models=[],
                )
            )
        elif set(expected_request_models).isdisjoint(observed_request_models):
            checks.append(
                _fail_check(
                    "request_model",
                    "latest trace request_model does not match expected model aliases",
                    expected_request_models=expected_request_models,
                    observed_request_models=observed_request_models,
                )
            )
        else:
            checks.append(
                _ok_check(
                    "request_model",
                    "latest trace request_model matches an expected model alias",
                    expected_request_models=expected_request_models,
                    observed_request_models=observed_request_models,
                )
            )

    if require_tool_span:
        if not tool_spans:
            checks.append(_fail_check("tool_span_count", "no tool spans are present"))
        else:
            checks.append(
                _ok_check("tool_span_count", "tool spans are present", value=len(tool_spans))
            )

    if require_tool_content:
        if not tool_spans and not chat_tool_calls:
            checks.append(
                _fail_check(
                    "tool_content_capture",
                    "tool content was required but no tool spans or chat tool calls are present",
                )
            )
        elif tool_content_count < len(tool_spans):
            checks.append(
                _fail_check(
                    "tool_content_capture",
                    "one or more tool spans lack arguments/results in Agents API",
                    tool_span_count=len(tool_spans),
                    tool_spans_with_content=tool_content_count,
                    chat_tool_calls_with_content=len(chat_tool_calls_with_content),
                )
            )
        else:
            checks.append(
                _ok_check(
                    "tool_content_capture",
                    "tool arguments/results are visible in Agents API",
                    tool_span_count=len(tool_spans),
                    tool_spans_with_content=tool_content_count,
                    chat_tool_calls_with_content=len(chat_tool_calls_with_content),
                )
            )

    input_start_records = [
        (_span_start(span), str(span.get("started_at") or ""))
        for span in message_spans_with_input
    ] + [
        (_chat_timestamp(message), str(message.get("started_at") or ""))
        for message in chat_messages_with_input
    ]
    input_start_records = [
        record for record in input_start_records if record[0] is not None
    ]
    tool_start_records = [
        (_span_start(span), str(span.get("started_at") or ""))
        for span in tool_spans
    ] + [
        (_chat_timestamp(message), str(message.get("started_at") or ""))
        for message in chat_tool_calls
    ]
    tool_start_records = [
        record for record in tool_start_records if record[0] is not None
    ]
    final_answer_end_records = [
        (_span_end(span), str(span.get("ended_at") or ""))
        for span in final_answer_spans
    ]
    final_answer_start_records = [
        (_chat_timestamp(message), str(message.get("started_at") or ""))
        for message in chat_final_answer_messages
    ]
    final_answer_end_records = [
        record for record in final_answer_end_records if record[0] is not None
    ]
    final_answer_start_records = [
        record for record in final_answer_start_records if record[0] is not None
    ]

    if message_spans and tool_spans:
        if timestamp_issues:
            checks.append(
                _fail_check(
                    "trace_order",
                    "message/tool ordering could not be compared because timestamps are invalid",
                    timestamp_issue_count=len(timestamp_issues),
                )
            )
        else:
            first_message = min(_span_start(span) for span in message_spans)
            first_tool = min(_span_start(span) for span in tool_spans)
            if first_message is None or first_tool is None:
                checks.append(
                    _fail_check(
                        "trace_order",
                        "message/tool ordering could not be compared because timestamps are invalid",
                    )
                )
            elif first_tool <= first_message:
                checks.append(
                    _fail_check(
                        "trace_order",
                        "a tool span starts before or at the same time as the first message span",
                        first_message_started_at=min(str(span.get("started_at") or "") for span in message_spans),
                        first_tool_started_at=min(str(span.get("started_at") or "") for span in tool_spans),
                    )
                )
            else:
                checks.append(
                    _ok_check(
                        "trace_order",
                        "tool spans do not start before the first message span",
                        first_message_started_at=min(str(span.get("started_at") or "") for span in message_spans),
                        first_tool_started_at=min(str(span.get("started_at") or "") for span in tool_spans),
                    )
                )

    if tool_spans:
        if timestamp_issues:
            checks.append(
                _fail_check(
                    "trace_user_message_order",
                    "input/tool ordering could not be compared because timestamps are invalid",
                    timestamp_issue_count=len(timestamp_issues),
                    first_tool_started_at=min(str(span.get("started_at") or "") for span in tool_spans),
                )
            )
        elif input_start_records:
            first_tool, first_tool_started_at = min(tool_start_records)
            first_input_message, first_input_message_started_at = min(input_start_records)
            if first_input_message is None or first_tool is None:
                checks.append(
                    _fail_check(
                        "trace_user_message_order",
                        "input/tool ordering could not be compared because timestamps are invalid",
                    )
                )
            elif first_tool <= first_input_message:
                checks.append(
                    _fail_check(
                        "trace_user_message_order",
                        "a tool span starts before or at the same time as the first visible user/problem input span",
                        first_input_message_started_at=first_input_message_started_at,
                        first_tool_started_at=first_tool_started_at,
                    )
                )
            else:
                checks.append(
                    _ok_check(
                        "trace_user_message_order",
                        "tool spans do not start before visible user/problem input",
                        first_input_message_started_at=first_input_message_started_at,
                        first_tool_started_at=first_tool_started_at,
                    )
                )
        elif require_input_message:
            checks.append(
                _fail_check(
                    "trace_user_message_order",
                    "tool spans are present but no visible user/problem input span exists",
                    first_tool_started_at=min(str(span.get("started_at") or "") for span in tool_spans),
                )
            )
        else:
            checks.append(
                _ok_check(
                    "trace_user_message_order",
                    "input-message order was not required for this verifier run",
                    tool_span_count=len(tool_spans),
                )
            )
    else:
        checks.append(
            _ok_check(
                "trace_user_message_order",
                "no tool span and input-message order conflict was detected",
                message_spans_with_input=message_input_count,
                tool_span_count=0,
            )
        )

    if final_answer_count > 0 and tool_spans:
        if timestamp_issues:
            checks.append(
                _fail_check(
                    "trace_final_answer_order",
                    "final-answer/tool ordering could not be compared because timestamps are invalid",
                    timestamp_issue_count=len(timestamp_issues),
                )
            )
        else:
            final_answer_boundary_records = final_answer_end_records + final_answer_start_records
            if not final_answer_boundary_records or not tool_start_records:
                checks.append(
                    _fail_check(
                        "trace_final_answer_order",
                        "final-answer/tool ordering could not be compared because timestamps are invalid",
                    )
                )
            else:
                first_final_answer_boundary, first_final_answer_at = min(final_answer_boundary_records)
                last_tool_start, last_tool_started_at = max(tool_start_records)
                if last_tool_start >= first_final_answer_boundary:
                    checks.append(
                        _ok_check(
                            "trace_final_answer_order",
                            "tool spans may occur after a provisional final-answer marker; scoring uses the final captured output",
                            first_final_answer_at=first_final_answer_at,
                            last_tool_started_at=last_tool_started_at,
                            final_answer_span_count=final_answer_count,
                            tool_span_count=len(tool_spans),
                            tool_after_final_answer_warning=True,
                        )
                    )
                else:
                    checks.append(
                        _ok_check(
                            "trace_final_answer_order",
                            "tool spans do not start after final-answer messages",
                            first_final_answer_at=first_final_answer_at,
                            last_tool_started_at=last_tool_started_at,
                            final_answer_span_count=final_answer_count,
                            tool_span_count=len(tool_spans),
                        )
                    )
    else:
        checks.append(
            _ok_check(
                "trace_final_answer_order",
                "no final-answer marker and tool-order conflict was detected",
                final_answer_span_count=final_answer_count,
                tool_span_count=len(tool_spans),
            )
        )

    error_spans = [
        summarize_span(span)
        for span in latest_trace_spans_chronological
        if span.get("error_type")
    ]
    if require_no_errors:
        if error_spans:
            checks.append(
                _fail_check(
                    "trace_errors",
                    "latest trace contains error spans",
                    error_count=len(error_spans),
                )
            )
        else:
            checks.append(_ok_check("trace_errors", "latest trace has no error spans"))

    tool_after_final_answer_warning = any(
        check.get("name") == "trace_final_answer_order"
        and check.get("tool_after_final_answer_warning") is True
        for check in checks
    )
    content_capture_health = {
        "span_count_checked": len(latest_trace_spans_chronological),
        "message_span_count": len(message_spans),
        "message_spans_with_content": message_content_count,
        "message_spans_with_input": message_input_count,
        "tool_span_count": len(tool_spans),
        "tool_spans_with_content": tool_content_count,
        "final_answer_span_count": final_answer_count,
        "spans_with_valid_timestamps": len(latest_trace_spans_chronological) - len(timestamp_issues),
        "spans_with_invalid_timestamps": len(timestamp_issues),
        "trace_input_tokens": trace_input_tokens,
        "trace_output_tokens": trace_output_tokens,
        "required_text_count": len(required_texts),
        "request_model_count": len(observed_request_models),
        "trace_final_answer_after_tool_warning": tool_after_final_answer_warning,
    }
    if trace_chat_payload is not None:
        content_capture_health.update(
            {
                "chat_message_count": len(trace_chat_messages),
                "chat_messages_with_content": len(chat_messages_with_content),
                "chat_messages_with_input": len(chat_messages_with_input),
                "chat_tool_call_count": len(chat_tool_calls),
                "chat_tool_calls_with_content": len(chat_tool_calls_with_content),
                "chat_final_answer_message_count": len(chat_final_answer_messages),
            }
        )
    ok = all(check["ok"] for check in checks)
    query_source = {
        "kind": "wandb_agents_api",
        "api_base_url": AGENTS_API_BASE_URL,
        "agents_endpoint": AGENTS_QUERY_ENDPOINT,
        "spans_endpoint": AGENTS_SPANS_QUERY_ENDPOINT,
        "project_id": project_id,
        "agent_name": agent_name,
        "conversation_id": conversation_id or "",
        "conversation_id_contains": conversation_id_contains or "",
        "agents_count": len(agents_payload.get("agents", []))
        if isinstance(agents_payload.get("agents"), list)
        else 0,
        "spans_count": len(spans_payload.get("spans", []))
        if isinstance(spans_payload.get("spans"), list)
        else 0,
        "matching_span_count": len(raw_spans),
        "latest_trace_span_count": len(latest_trace_spans_chronological),
    }
    if trace_chat_payload is not None:
        query_source.update(
            {
                "trace_chat_endpoint": AGENTS_TRACES_CHAT_ENDPOINT,
                "trace_chat_message_count": len(trace_chat_messages),
            }
        )
    result = {
        **_verification_header(ok),
        "ok": ok,
        "project_id": project_id,
        "agent_name": agent_name,
        "agents_url": f"https://wandb.ai/{entity}/{project}/weave/agents",
        "query_source": query_source,
        "latest_trace_id": latest_trace_id,
        "content_capture_health": content_capture_health,
        "latest_trace_spans_chronological": [
            summarize_span(span) for span in latest_trace_spans_chronological
        ],
        "checks": checks,
    }
    if trace_chat_payload is not None:
        result["latest_trace_chat_messages_chronological"] = [
            summarize_chat_message(message) for message in trace_chat_messages
        ]
    return result


def query_agents(
    *,
    env: dict[str, str],
    entity: str,
    project: str,
    agent_name: str,
    limit: int,
    conversation_id: str | None = None,
    conversation_id_contains: str | None = None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    project_id = f"{entity}/{project}"
    filters = {"agent_name": agent_name} if agent_name else {}
    agents_payload = {
        "project_id": project_id,
        "filters": filters,
        "limit": limit,
        "offset": 0,
    }
    span_filters = (
        {}
        if (conversation_id or conversation_id_contains)
        else dict(filters)
    )
    spans_payload = {
        "project_id": project_id,
        "filters": span_filters,
        "limit": max(limit, limit * 8, 400),
        "offset": 0,
    }
    return (
        agents_api_post(env, AGENTS_QUERY_ENDPOINT, agents_payload),
        agents_api_post(env, AGENTS_SPANS_QUERY_ENDPOINT, spans_payload),
    )


def query_trace_chat(
    *,
    env: dict[str, str],
    entity: str,
    project: str,
    trace_id: str,
) -> dict[str, Any]:
    return agents_api_post(
        env,
        AGENTS_TRACES_CHAT_ENDPOINT,
        {
            "project_id": f"{entity}/{project}",
            "trace_id": trace_id,
            "include_feedback": False,
        },
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--entity", default=os.environ.get("WANDB_ENTITY", "llm-leaderboard"))
    parser.add_argument("--project", default=os.environ.get("WANDB_PROJECT", "tc-leaderboard"))
    parser.add_argument("--agent-name", default=DEFAULT_AGENT_NAME)
    parser.add_argument("--limit", type=int, default=20)
    parser.add_argument("--min-agent-invocations", type=int, default=1)
    parser.add_argument("--min-trace-spans", type=int, default=1)
    parser.add_argument("--min-message-spans", type=int, default=1)
    parser.add_argument("--conversation-id")
    parser.add_argument("--conversation-id-contains")
    parser.add_argument(
        "--no-require-content",
        action="store_true",
        help="Allow structure-only traces. Production Taiwan runs should not use this unless content capture is intentionally disabled.",
    )
    parser.add_argument(
        "--no-require-input-message",
        action="store_true",
        help="Do not require visible user/problem input messages. Production Taiwan runs should not use this.",
    )
    parser.add_argument("--require-tool-span", action="store_true")
    parser.add_argument("--require-tool-content", action="store_true")
    parser.add_argument("--require-usage", action="store_true")
    parser.add_argument(
        "--require-text",
        action="append",
        default=[],
        help="Require this exact text to be visible in the latest matching trace content. Repeatable.",
    )
    parser.add_argument(
        "--expected-request-model",
        action="append",
        default=[],
        help="Require the latest matching trace to expose one of these request_model values. Repeatable.",
    )
    parser.add_argument("--allow-error-spans", action="store_true")
    parser.add_argument("--env-file", type=Path, default=DEFAULT_ENV_FILE)
    parser.add_argument("--json", type=Path, help="Optional path to write verifier JSON.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    env, env_loaded = load_env_file(os.environ.copy(), args.env_file)
    agents_payload, spans_payload = query_agents(
        env=env,
        entity=args.entity,
        project=args.project,
        agent_name=args.agent_name,
        limit=args.limit,
        conversation_id=args.conversation_id,
        conversation_id_contains=args.conversation_id_contains,
    )
    matching_spans = _matching_spans(
        spans_payload,
        agent_name=args.agent_name,
        conversation_id=args.conversation_id,
        conversation_id_contains=args.conversation_id_contains,
    )
    latest_trace_id = _latest_trace_id(matching_spans)
    trace_chat_payload = (
        query_trace_chat(
            env=env,
            entity=args.entity,
            project=args.project,
            trace_id=latest_trace_id,
        )
        if latest_trace_id
        else None
    )
    result = verify_agents_payload(
        agents_payload,
        spans_payload,
        trace_chat_payload=trace_chat_payload,
        entity=args.entity,
        project=args.project,
        agent_name=args.agent_name,
        min_agent_invocations=args.min_agent_invocations,
        min_trace_spans=args.min_trace_spans,
        min_message_spans=args.min_message_spans,
        require_content=not args.no_require_content,
        require_input_message=not args.no_require_input_message
        and not args.no_require_content,
        require_tool_span=bool(args.require_tool_span),
        require_tool_content=bool(args.require_tool_content),
        require_usage=bool(args.require_usage),
        require_no_errors=not bool(args.allow_error_spans),
        required_texts=args.require_text,
        expected_request_models=args.expected_request_model,
        conversation_id=args.conversation_id,
        conversation_id_contains=args.conversation_id_contains,
    )
    result["generated_at"] = time.time()
    result.update(_verification_header(bool(result.get("ok"))))
    result["required_evidence"] = {
        "min_agent_invocations": args.min_agent_invocations,
        "min_trace_spans": args.min_trace_spans,
        "min_message_spans": args.min_message_spans,
        "content_required": not args.no_require_content,
        "input_message_required": not args.no_require_input_message
        and not args.no_require_content,
        "tool_span_required": bool(args.require_tool_span),
        "tool_content_required": bool(args.require_tool_content),
        "trace_timestamp_quality_required": True,
        "trace_final_answer_order_required": True,
        "usage_required": bool(args.require_usage),
        "no_error_spans_required": not bool(args.allow_error_spans),
        "required_texts": list(args.require_text or []),
        "expected_request_models": list(args.expected_request_model or []),
        "conversation_id": args.conversation_id or "",
        "conversation_id_contains": args.conversation_id_contains or "",
    }
    result["env_file"] = str(args.env_file.expanduser()) if args.env_file else None
    result["env_file_loaded"] = env_loaded
    if args.json:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))
    if not result["ok"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
