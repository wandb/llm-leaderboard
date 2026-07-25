"""Normalize and aggregate OpenClaw usage without losing retry cost."""

from __future__ import annotations

from typing import Any, Iterable


TOKEN_KEYS = {
    "inputTokens": ("inputTokens", "input", "promptTokens", "prompt_tokens"),
    "outputTokens": ("outputTokens", "output", "completionTokens", "completion_tokens"),
    "cacheReadInputTokens": (
        "cacheReadInputTokens",
        "cacheRead",
        "cache_read_input_tokens",
        "cachedTokens",
        "cached_tokens",
    ),
    "cacheWriteInputTokens": (
        "cacheWriteInputTokens",
        "cacheWrite",
        "cache_write_input_tokens",
    ),
    "reasoningTokens": ("reasoningTokens", "reasoning", "reasoning_tokens"),
    "totalTokens": ("totalTokens", "total_tokens"),
}


def _number(value: Any) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return 0.0
    return float(value)


def _first_number(mapping: dict[str, Any], keys: Iterable[str]) -> float:
    for key in keys:
        value = _number(mapping.get(key))
        if value:
            return value
    return 0.0


def normalize_openclaw_usage(value: Any) -> dict[str, float]:
    """Return one canonical usage object from the known OpenClaw SDK shapes."""
    if not isinstance(value, dict):
        value = {}
    normalized = {
        target: _first_number(value, aliases)
        for target, aliases in TOKEN_KEYS.items()
    }
    cost = value.get("cost")
    if isinstance(cost, dict):
        normalized["costUsd"] = _first_number(cost, ("total", "usd", "costUsd", "cost_usd"))
    else:
        normalized["costUsd"] = _first_number(value, ("costUsd", "cost_usd", "cost"))
    if not normalized["totalTokens"]:
        normalized["totalTokens"] = (
            normalized["inputTokens"]
            + normalized["outputTokens"]
            + normalized["cacheReadInputTokens"]
            + normalized["cacheWriteInputTokens"]
        )
    return normalized


def aggregate_openclaw_usage(values: Iterable[Any]) -> dict[str, float]:
    totals = normalize_openclaw_usage({})
    for value in values:
        usage = normalize_openclaw_usage(value)
        for key in totals:
            totals[key] += usage[key]
    return totals


def attach_billable_openclaw_usage(
    record: dict[str, Any],
    attempts: list[dict[str, Any]],
    *,
    prior_record: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Attach retry-inclusive usage while retaining final-attempt scoring usage."""
    prior_attempts = []
    prior_usage: list[Any] = []
    prior_count = 0
    if isinstance(prior_record, dict):
        existing_attempts = prior_record.get("billable_openclaw_attempts")
        if isinstance(existing_attempts, list):
            prior_attempts = [item for item in existing_attempts if isinstance(item, dict)]
        existing_billable = prior_record.get("billable_openclaw_usage")
        has_prior_usage = False
        if isinstance(existing_billable, dict) and existing_billable:
            prior_usage.append(existing_billable)
            has_prior_usage = True
        elif isinstance(prior_record.get("openclaw_usage"), dict):
            prior_usage.append(prior_record["openclaw_usage"])
            has_prior_usage = bool(prior_record["openclaw_usage"])
        prior_count = int(
            prior_record.get("billable_openclaw_attempt_count")
            or len(prior_attempts)
            or (1 if has_prior_usage else 0)
        )

    compact_attempts = []
    current_usage = []
    for attempt in attempts:
        usage = attempt.get("usage") if isinstance(attempt, dict) else None
        current_usage.append(usage)
        compact_attempts.append(
            {
                "attempt_id": attempt.get("attempt_id"),
                "attempt_number": attempt.get("attempt_number"),
                "returncode": attempt.get("returncode"),
                "wall_clock_seconds": attempt.get("wall_clock_seconds"),
                "openclaw_result_path": attempt.get("openclaw_result_path"),
                "usage": usage if isinstance(usage, dict) else {},
            }
        )

    updated = dict(record)
    updated["billable_openclaw_usage"] = aggregate_openclaw_usage([*prior_usage, *current_usage])
    updated["billable_openclaw_attempt_count"] = prior_count + len(compact_attempts)
    updated["billable_openclaw_attempts"] = [*prior_attempts, *compact_attempts]
    updated["billable_openclaw_wall_seconds"] = sum(
        _number(item.get("wall_clock_seconds"))
        for item in updated["billable_openclaw_attempts"]
    )
    return updated


def ensure_billable_openclaw_usage(record: dict[str, Any]) -> dict[str, Any]:
    """Backfill retry-inclusive fields on legacy single-attempt cache records."""
    required = (
        "billable_openclaw_usage",
        "billable_openclaw_attempt_count",
        "billable_openclaw_attempts",
        "billable_openclaw_wall_seconds",
    )
    if all(key in record for key in required):
        return dict(record)
    return attach_billable_openclaw_usage(record, [], prior_record=record)


def merge_prior_billable_openclaw_usage(
    record: dict[str, Any],
    prior_record: dict[str, Any] | None,
) -> dict[str, Any]:
    """Prepend a prior same-cache run's billable evidence to a resumed result."""
    current = ensure_billable_openclaw_usage(record)
    if not isinstance(prior_record, dict):
        return current
    prior = ensure_billable_openclaw_usage(prior_record)
    prior_attempts = prior.get("billable_openclaw_attempts")
    current_attempts = current.get("billable_openclaw_attempts")
    prior_attempts = prior_attempts if isinstance(prior_attempts, list) else []
    current_attempts = current_attempts if isinstance(current_attempts, list) else []
    updated = dict(current)
    updated["billable_openclaw_usage"] = aggregate_openclaw_usage(
        [prior.get("billable_openclaw_usage"), current.get("billable_openclaw_usage")]
    )
    updated["billable_openclaw_attempt_count"] = max(
        0, int(prior.get("billable_openclaw_attempt_count") or 0)
    ) + max(0, int(current.get("billable_openclaw_attempt_count") or 0))
    updated["billable_openclaw_attempts"] = [*prior_attempts, *current_attempts]
    updated["billable_openclaw_wall_seconds"] = _number(
        prior.get("billable_openclaw_wall_seconds")
    ) + _number(current.get("billable_openclaw_wall_seconds"))
    return updated


def summarize_billable_openclaw_records(records: Iterable[Any]) -> dict[str, Any]:
    """Summarize retry-inclusive usage across task result records."""
    rows = [record for record in records if isinstance(record, dict)]
    attempts = sum(
        max(0, int(record.get("billable_openclaw_attempt_count") or 0))
        for record in rows
    )
    return {
        "usage": aggregate_openclaw_usage(
            record.get("billable_openclaw_usage")
            if isinstance(record.get("billable_openclaw_usage"), dict)
            else record.get("openclaw_usage")
            for record in rows
        ),
        "attempt_count": attempts,
        "retry_count": max(0, attempts - len(rows)),
        "wall_seconds": sum(
            _number(record.get("billable_openclaw_wall_seconds"))
            for record in rows
        ),
    }
