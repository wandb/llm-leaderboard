#!/usr/bin/env python3
"""
Estimate agentic OpenClaw usage costs from local Nejumi output records.

This is an accountability helper, not a billing source of truth. Provider
dashboards remain authoritative. The script deduplicates records by
openclaw_result_path when present, so it can scan both per-task JSON files and
aggregate JSONL files without double-counting the same OpenClaw run.
"""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path
from typing import Any


PRICE_PER_MILLION: dict[str, dict[str, float]] = {
    # OpenAI GPT-4.1 mini API release pricing, per 1M tokens:
    # input $0.40, cached input $0.10, output $1.60.
    # Provider dashboards remain authoritative for billing.
    "openai-direct/gpt-4.1-mini-2025-04-14": {
        "input": 0.40,
        "output": 1.60,
        "cacheRead": 0.10,
        "cacheWrite": 0.0,
    },
    "openrouter-direct/z-ai/glm-5.2": {
        "input": 0.95,
        "output": 3.00,
        "cacheRead": 0.18,
        "cacheWrite": 0.0,
    },
    "openrouter-direct/google/gemini-3.1-pro-preview": {
        "input": 2.00,
        "output": 12.00,
        "cacheRead": 0.30,
        "cacheWrite": 0.0,
    },
    "openrouter-direct/qwen/qwen3.6-max-preview": {
        "input": 1.10,
        "output": 3.00,
        "cacheRead": 0.0,
        "cacheWrite": 0.0,
    },
    "openrouter-direct/anthropic/claude-sonnet-4.6": {
        "input": 3.00,
        "output": 15.00,
        "cacheRead": 0.30,
        "cacheWrite": 0.0,
    },
    "openrouter-direct/anthropic/claude-opus-4.7": {
        "input": 5.00,
        "output": 25.00,
        "cacheRead": 0.50,
        "cacheWrite": 0.0,
    },
    "deepseek/deepseek-v4-pro": {
        "input": 1.74,
        "output": 3.48,
        "cacheRead": 0.145,
        "cacheWrite": 0.0,
    },
}


TOKEN_KEYS = ("input", "output", "cacheRead", "cacheWrite", "reasoningTokens")
EXCLUDED_SCAN_DIR_NAMES = {
    ".git",
    ".pytest_cache",
    ".venv",
    "__pycache__",
    "node_modules",
    "swebench_pro_checkouts",
}


def iter_json_records(path: Path) -> Any:
    if path.suffix == ".jsonl":
        for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(record, dict):
                yield record, path
        return

    if path.suffix == ".json":
        try:
            record = json.loads(path.read_text(encoding="utf-8", errors="replace"))
        except json.JSONDecodeError:
            return
        if isinstance(record, dict):
            yield record, path
        elif isinstance(record, list):
            for item in record:
                if isinstance(item, dict):
                    yield item, path


def usage_from_record(record: dict[str, Any]) -> dict[str, int] | None:
    usage = record.get("openclaw_usage")
    if not isinstance(usage, dict):
        meta = record.get("metadata")
        if isinstance(meta, dict):
            usage = meta.get("usage")
    if not isinstance(usage, dict):
        return None

    normalized: dict[str, int] = {}
    for key in TOKEN_KEYS:
        value = usage.get(key)
        if isinstance(value, (int, float)):
            normalized[key] = int(value)
    return normalized or None


def model_from_record(record: dict[str, Any]) -> str:
    cache_key = record.get("cache_key")
    if isinstance(cache_key, dict) and cache_key.get("model"):
        return str(cache_key["model"])
    for key in ("model", "openclaw_model"):
        if record.get(key):
            return str(record[key])
    return "unknown"


def record_key(record: dict[str, Any], source: Path) -> str:
    for key in ("openclaw_result_path", "openclaw_invocation_path"):
        if record.get(key):
            return str(record[key])
    cache_key = record.get("cache_key")
    if isinstance(cache_key, dict):
        return json.dumps(cache_key, ensure_ascii=False, sort_keys=True)
    return f"{source}:{id(record)}"


def estimate_cost(model: str, usage: dict[str, int]) -> float | None:
    price = PRICE_PER_MILLION.get(model)
    if price is None:
        return None
    total = 0.0
    for key, token_count in usage.items():
        if key == "reasoningTokens":
            continue
        total += token_count / 1_000_000 * price.get(key, 0.0)
    return total


def scan(root: Path) -> list[dict[str, Any]]:
    seen: set[str] = set()
    rows: list[dict[str, Any]] = []
    pending = [root]
    while pending:
        path = pending.pop()
        if path.is_dir():
            if path.name in EXCLUDED_SCAN_DIR_NAMES:
                continue
            pending.extend(path.iterdir())
            continue
        if not path.is_file() or path.suffix not in {".json", ".jsonl"}:
            continue
        for record, source in iter_json_records(path):
            usage = usage_from_record(record)
            if not usage:
                continue
            key = record_key(record, source)
            if key in seen:
                continue
            seen.add(key)
            model = model_from_record(record)
            cost = estimate_cost(model, usage)
            row: dict[str, Any] = {
                "model": model,
                "source": str(source),
                "record_key": key,
                "estimated_cost_usd": cost,
            }
            for token_key in TOKEN_KEYS:
                row[token_key] = usage.get(token_key, 0)
            rows.append(row)
    return rows


def summarize(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    totals: dict[str, dict[str, Any]] = defaultdict(lambda: {"records": 0, "estimated_cost_usd": 0.0})
    for row in rows:
        model = str(row["model"])
        totals[model]["records"] += 1
        cost = row.get("estimated_cost_usd")
        if isinstance(cost, float):
            totals[model]["estimated_cost_usd"] += cost
        for token_key in TOKEN_KEYS:
            totals[model][token_key] = totals[model].get(token_key, 0) + int(row.get(token_key, 0))
    return [
        {"model": model, **values}
        for model, values in sorted(
            totals.items(),
            key=lambda item: item[1].get("estimated_cost_usd", 0.0),
            reverse=True,
        )
    ]


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "model",
        "records",
        "estimated_cost_usd",
        *TOKEN_KEYS,
    ]
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in fieldnames})


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path, help="Output directory to scan.")
    parser.add_argument("--csv", type=Path, help="Optional summary CSV path.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rows = scan(args.root)
    summary = summarize(rows)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    if args.csv:
        write_csv(args.csv, summary)


if __name__ == "__main__":
    main()
