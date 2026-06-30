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
FINAL_ANSWER_MARKERS = (
    "ANSWER:",
    "FINAL ANSWER",
    "Final answer",
    "\\boxed",
    "CANARY_RESULT",
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


def _span_output_text(span: dict[str, Any]) -> str:
    return _jsonish_text(span.get("output_messages"))


def _is_final_answer_span(span: dict[str, Any]) -> bool:
    if span.get("operation_name") not in {"chat", "invoke_agent"}:
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
        and span.get("agent_name") == agent_name
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
    conversation_id: str | None = None,
    conversation_id_contains: str | None = None,
) -> dict[str, Any]:
    if require_input_message is None:
        require_input_message = require_content
    checks: list[dict[str, Any]] = []
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
    if not best_agent:
        checks.append(_fail_check("agent_present", "agent is not present", agent_name=agent_name))
        agent_input_tokens = 0
        agent_output_tokens = 0
    else:
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

    raw_spans = _matching_spans(
        spans_payload,
        agent_name=agent_name,
        conversation_id=conversation_id,
        conversation_id_contains=conversation_id_contains,
    )
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
    visible_trace_text = "\n".join(_span_visible_text(span) for span in latest_trace_spans_chronological)
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
        if not message_spans_with_content:
            checks.append(
                _fail_check(
                    "message_content_capture",
                    "message spans do not expose input/output content",
                    message_span_count=len(message_spans),
                    message_spans_with_content=0,
                )
            )
        else:
            checks.append(
                _ok_check(
                    "message_content_capture",
                    "message content is visible in Agents API",
                    message_span_count=len(message_spans),
                    message_spans_with_content=len(message_spans_with_content),
                )
            )

    if require_input_message:
        if not message_spans_with_input:
            checks.append(
                _fail_check(
                    "input_message_capture",
                    "message spans do not expose the user/problem input",
                    message_span_count=len(message_spans),
                    message_spans_with_input=0,
                )
            )
        else:
            checks.append(
                _ok_check(
                    "input_message_capture",
                    "user/problem input is visible in Agents API",
                    message_span_count=len(message_spans),
                    message_spans_with_input=len(message_spans_with_input),
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

    if require_tool_span:
        if not tool_spans:
            checks.append(_fail_check("tool_span_count", "no tool spans are present"))
        else:
            checks.append(
                _ok_check("tool_span_count", "tool spans are present", value=len(tool_spans))
            )

    if require_tool_content:
        if not tool_spans:
            checks.append(
                _fail_check(
                    "tool_content_capture",
                    "tool content was required but no tool spans are present",
                )
            )
        elif len(tool_spans_with_content) != len(tool_spans):
            checks.append(
                _fail_check(
                    "tool_content_capture",
                    "one or more tool spans lack arguments/results",
                    tool_span_count=len(tool_spans),
                    tool_spans_with_content=len(tool_spans_with_content),
                )
            )
        else:
            checks.append(
                _ok_check(
                    "tool_content_capture",
                    "tool arguments/results are visible in Agents API",
                    tool_span_count=len(tool_spans),
                )
            )

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
        elif message_spans_with_input:
            first_tool = min(_span_start(span) for span in tool_spans)
            first_input_message = min(_span_start(span) for span in message_spans_with_input)
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
                        first_input_message_started_at=min(
                            str(span.get("started_at") or "") for span in message_spans_with_input
                        ),
                        first_tool_started_at=min(str(span.get("started_at") or "") for span in tool_spans),
                    )
                )
            else:
                checks.append(
                    _ok_check(
                        "trace_user_message_order",
                        "tool spans do not start before visible user/problem input",
                        first_input_message_started_at=min(
                            str(span.get("started_at") or "") for span in message_spans_with_input
                        ),
                        first_tool_started_at=min(str(span.get("started_at") or "") for span in tool_spans),
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
                message_spans_with_input=len(message_spans_with_input),
                tool_span_count=0,
            )
        )

    if final_answer_spans and tool_spans:
        if timestamp_issues:
            checks.append(
                _fail_check(
                    "trace_final_answer_order",
                    "final-answer/tool ordering could not be compared because timestamps are invalid",
                    timestamp_issue_count=len(timestamp_issues),
                )
            )
        else:
            first_final_answer_end = min(_span_end(span) for span in final_answer_spans)
            last_tool_start = max(_span_start(span) for span in tool_spans)
            if first_final_answer_end is None or last_tool_start is None:
                checks.append(
                    _fail_check(
                        "trace_final_answer_order",
                        "final-answer/tool ordering could not be compared because timestamps are invalid",
                    )
                )
            elif last_tool_start >= first_final_answer_end:
                checks.append(
                    _fail_check(
                        "trace_final_answer_order",
                        "a tool span starts after or at the same time as a final-answer message span ended",
                        first_final_answer_ended_at=min(str(span.get("ended_at") or "") for span in final_answer_spans),
                        last_tool_started_at=max(str(span.get("started_at") or "") for span in tool_spans),
                        final_answer_span_count=len(final_answer_spans),
                        tool_span_count=len(tool_spans),
                    )
                )
            else:
                checks.append(
                    _ok_check(
                        "trace_final_answer_order",
                        "tool spans do not start after final-answer message spans",
                        first_final_answer_ended_at=min(str(span.get("ended_at") or "") for span in final_answer_spans),
                        last_tool_started_at=max(str(span.get("started_at") or "") for span in tool_spans),
                        final_answer_span_count=len(final_answer_spans),
                        tool_span_count=len(tool_spans),
                    )
                )
    else:
        checks.append(
            _ok_check(
                "trace_final_answer_order",
                "no final-answer marker and tool-order conflict was detected",
                final_answer_span_count=len(final_answer_spans),
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

    content_capture_health = {
        "span_count_checked": len(latest_trace_spans_chronological),
        "message_span_count": len(message_spans),
        "message_spans_with_content": len(message_spans_with_content),
        "message_spans_with_input": len(message_spans_with_input),
        "tool_span_count": len(tool_spans),
        "tool_spans_with_content": len(tool_spans_with_content),
        "final_answer_span_count": len(final_answer_spans),
        "spans_with_valid_timestamps": len(latest_trace_spans_chronological) - len(timestamp_issues),
        "spans_with_invalid_timestamps": len(timestamp_issues),
        "trace_input_tokens": trace_input_tokens,
        "trace_output_tokens": trace_output_tokens,
        "required_text_count": len(required_texts),
    }
    ok = all(check["ok"] for check in checks)
    result = {
        **_verification_header(ok),
        "ok": ok,
        "project_id": project_id,
        "agent_name": agent_name,
        "agents_url": f"https://wandb.ai/{entity}/{project}/weave/agents",
        "query_source": {
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
        },
        "latest_trace_id": latest_trace_id,
        "content_capture_health": content_capture_health,
        "latest_trace_spans_chronological": [
            summarize_span(span) for span in latest_trace_spans_chronological
        ],
        "checks": checks,
    }
    return result


def query_agents(
    *,
    env: dict[str, str],
    entity: str,
    project: str,
    agent_name: str,
    limit: int,
) -> tuple[dict[str, Any], dict[str, Any]]:
    project_id = f"{entity}/{project}"
    filters = {"agent_name": agent_name} if agent_name else {}
    agents_payload = {
        "project_id": project_id,
        "filters": filters,
        "limit": limit,
        "offset": 0,
    }
    spans_payload = dict(agents_payload)
    spans_payload["limit"] = max(limit, limit * 4)
    return (
        agents_api_post(env, AGENTS_QUERY_ENDPOINT, agents_payload),
        agents_api_post(env, AGENTS_SPANS_QUERY_ENDPOINT, spans_payload),
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
    )
    result = verify_agents_payload(
        agents_payload,
        spans_payload,
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
