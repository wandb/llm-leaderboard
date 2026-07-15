#!/usr/bin/env python3
"""Low-cost OpenRouter model health probe before a paid full evaluation."""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import statistics
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import openai


DEFAULT_BASE_URL = "https://openrouter.ai/api/v1"


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _percentile(values: list[float], q: float) -> float | None:
    if not values:
        return None
    if len(values) == 1:
        return values[0]
    sorted_values = sorted(values)
    index = (len(sorted_values) - 1) * q
    lower = int(index)
    upper = min(lower + 1, len(sorted_values) - 1)
    weight = index - lower
    return sorted_values[lower] * (1 - weight) + sorted_values[upper] * weight


def _parse_json_object(value: str | None, *, label: str) -> dict[str, Any]:
    if not value:
        return {}
    try:
        parsed = json.loads(value)
    except json.JSONDecodeError as exc:
        raise SystemExit(f"{label} must be a JSON object: {exc}") from exc
    if not isinstance(parsed, dict):
        raise SystemExit(f"{label} must be a JSON object")
    return parsed


def _build_payload(
    model: str,
    index: int,
    max_tokens: int,
) -> dict[str, Any]:
    tool = {
        "type": "function",
        "function": {
            "name": "lookup_tax_rate",
            "description": "Return the sales tax rate for a city code.",
            "parameters": {
                "type": "object",
                "properties": {
                    "city_code": {
                        "type": "string",
                        "description": "Short city code such as TPE or KHH.",
                    }
                },
                "required": ["city_code"],
            },
        },
    }
    city = "TPE" if index % 2 == 0 else "KHH"
    payload = {
        "model": model,
        "messages": [
            {
                "role": "system",
                "content": "You are a concise function-calling probe. Use tools when useful.",
            },
            {
                "role": "user",
                "content": (
                    f"Probe {index}: call the tool for city code {city}, then answer in one short sentence."
                ),
            },
        ],
        "tools": [tool],
        "tool_choice": "auto",
        "max_tokens": max_tokens,
        "temperature": 0,
    }
    return payload


async def _one_request(
    client: openai.AsyncOpenAI,
    *,
    model: str,
    index: int,
    timeout: float,
    max_tokens: int,
    extra_body: dict[str, Any],
    semaphore: asyncio.Semaphore,
) -> dict[str, Any]:
    async with semaphore:
        started = time.monotonic()
        payload = _build_payload(model, index, max_tokens)
        request_kwargs = {"timeout": timeout}
        if extra_body:
            request_kwargs["extra_body"] = extra_body
        try:
            response = await client.chat.completions.create(
                **payload,
                **request_kwargs,
            )
            elapsed = time.monotonic() - started
            choice = response.choices[0] if response.choices else None
            message = getattr(choice, "message", None)
            tool_calls = getattr(message, "tool_calls", None) or []
            usage = getattr(response, "usage", None)
            return {
                "index": index,
                "ok": True,
                "elapsed_sec": elapsed,
                "finish_reason": getattr(choice, "finish_reason", None),
                "tool_call_count": len(tool_calls),
                "prompt_tokens": getattr(usage, "prompt_tokens", 0) if usage else 0,
                "completion_tokens": getattr(usage, "completion_tokens", 0) if usage else 0,
            }
        except Exception as exc:
            elapsed = time.monotonic() - started
            status_code = getattr(exc, "status_code", None)
            return {
                "index": index,
                "ok": False,
                "elapsed_sec": elapsed,
                "error_type": type(exc).__name__,
                "status_code": status_code,
                "error": str(exc)[:1000],
            }


def _summarize(results: list[dict[str, Any]], *, timeout: float) -> dict[str, Any]:
    latencies = [row["elapsed_sec"] for row in results if row.get("ok")]
    errors = [row for row in results if not row.get("ok")]
    rate_limited = [
        row for row in errors
        if row.get("status_code") == 429 or "rate limit" in row.get("error", "").lower()
    ]
    timed_out = [
        row for row in errors
        if row.get("error_type") in {"APITimeoutError", "TimeoutError"}
        or row.get("elapsed_sec", 0) >= timeout
    ]
    success_count = len(results) - len(errors)
    total = len(results)
    return {
        "total": total,
        "success_count": success_count,
        "error_count": len(errors),
        "rate_limit_count": len(rate_limited),
        "timeout_count": len(timed_out),
        "success_rate": success_count / total if total else 0,
        "rate_limit_rate": len(rate_limited) / total if total else 0,
        "timeout_rate": len(timed_out) / total if total else 0,
        "latency_p50_sec": statistics.median(latencies) if latencies else None,
        "latency_p95_sec": _percentile(latencies, 0.95),
        "prompt_tokens": sum(int(row.get("prompt_tokens") or 0) for row in results),
        "completion_tokens": sum(int(row.get("completion_tokens") or 0) for row in results),
    }


async def _run(args: argparse.Namespace) -> dict[str, Any]:
    extra_body = _parse_json_object(args.extra_body_json, label="--extra-body-json")
    api_key = os.environ.get(args.api_key_env)
    if not api_key and not args.dry_run:
        raise SystemExit(f"{args.api_key_env} is required unless --dry-run is set")
    if args.dry_run:
        return {
            "schema_version": 1,
            "generated_at": _now_iso(),
            "dry_run": True,
            "model": args.model,
            "request_count": args.requests,
            "timeout_sec": args.timeout,
            "max_tokens": args.max_tokens,
            "extra_body": extra_body,
            "payload_preview": _build_payload(args.model, 0, args.max_tokens),
            "sdk_extra_body_preview": extra_body,
        }

    client = openai.AsyncOpenAI(
        api_key=api_key,
        base_url=args.base_url,
        max_retries=args.max_retries,
    )
    semaphore = asyncio.Semaphore(max(1, args.concurrency))
    started = time.monotonic()
    results = await asyncio.gather(
        *[
            _one_request(
                client,
                model=args.model,
                index=index,
                timeout=args.timeout,
                max_tokens=args.max_tokens,
                extra_body=extra_body,
                semaphore=semaphore,
            )
            for index in range(args.requests)
        ]
    )
    summary = _summarize(results, timeout=args.timeout)
    gate_ok = (
        summary["error_count"] == 0
        and summary["rate_limit_rate"] <= args.max_rate_limit_rate
        and (summary["latency_p95_sec"] is None or summary["latency_p95_sec"] <= args.max_p95_sec)
        and summary["timeout_count"] == 0
    )
    return {
        "schema_version": 1,
        "generated_at": _now_iso(),
        "dry_run": False,
        "model": args.model,
        "base_url": args.base_url,
        "request_count": args.requests,
        "concurrency": args.concurrency,
        "timeout_sec": args.timeout,
        "max_tokens": args.max_tokens,
        "extra_body": extra_body,
        "elapsed_sec": time.monotonic() - started,
        "gate": {
            "ok": gate_ok,
            "max_rate_limit_rate": args.max_rate_limit_rate,
            "max_p95_sec": args.max_p95_sec,
            "requires_timeout_count": 0,
        },
        "summary": summary,
        "results": results,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--base-url", default=DEFAULT_BASE_URL)
    parser.add_argument("--api-key-env", default="OPENROUTER_API_KEY")
    parser.add_argument("--requests", type=int, default=12)
    parser.add_argument("--concurrency", type=int, default=2)
    parser.add_argument("--timeout", type=float, default=60)
    parser.add_argument("--max-retries", type=int, default=0)
    parser.add_argument("--max-tokens", type=int, default=64)
    parser.add_argument(
        "--extra-body-json",
        default="",
        help=(
            "JSON object merged into every OpenRouter request body, for example "
            "'{\"provider\":{\"order\":[\"provider-name\"],\"allow_fallbacks\":false}}'."
        ),
    )
    parser.add_argument("--max-rate-limit-rate", type=float, default=0.10)
    parser.add_argument("--max-p95-sec", type=float, default=120)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    report = asyncio.run(_run(args))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({
        "output": str(args.output),
        "gate_ok": report.get("gate", {}).get("ok"),
        "summary": report.get("summary"),
        "dry_run": report.get("dry_run"),
    }, ensure_ascii=False, indent=2))
    return 0 if report.get("dry_run") or report.get("gate", {}).get("ok") else 2


if __name__ == "__main__":
    raise SystemExit(main())
