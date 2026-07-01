#!/usr/bin/env python3
"""
Estimate a one-model Taiwan canary budget from local historical agentic usage.

The estimate intentionally focuses on Agentic Math and SWE-Bench Pro, because
those are the cost drivers with available local token evidence. Non-agentic
benchmarks and LLM judges must be budgeted separately as a buffer.
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from estimate_agentic_usage_costs import PRICE_PER_MILLION, TOKEN_KEYS, scan


DEFAULT_TARGET_MODEL = "openai-direct/gpt-4.1-mini-2025-04-14"
DEFAULT_MATH_TASKS = 100
DEFAULT_SWE_TASKS = 80


def category_from_row(row: dict[str, Any]) -> str | None:
    text = f"{row.get('source', '')}\n{row.get('record_key', '')}"
    if "agentic_math" in text:
        return "agentic_math"
    if "swebench_pro" in text:
        return "swebench_pro"
    return None


def cost_with_price(row: dict[str, Any], price: dict[str, float]) -> float:
    total = 0.0
    for key in TOKEN_KEYS:
        if key == "reasoningTokens":
            continue
        total += int(row.get(key, 0)) / 1_000_000 * price.get(key, 0.0)
    return total


def summarize_category(
    rows: list[dict[str, Any]],
    *,
    target_count: int,
    price: dict[str, float],
    require_complete_baseline: bool = False,
) -> dict[str, Any]:
    by_model: dict[str, list[float]] = {}
    for row in rows:
        by_model.setdefault(str(row["model"]), []).append(cost_with_price(row, price))

    model_summaries = []
    for model, costs in sorted(by_model.items()):
        total = sum(costs)
        count = len(costs)
        model_summaries.append(
            {
                "model": model,
                "records": count,
                "observed_cost_at_target_price": total,
                "extrapolated_target_count_cost": total / count * target_count if count else None,
                "avg_cost_per_record": total / count if count else None,
            }
        )

    complete = [item for item in model_summaries if item["records"] >= target_count * 0.95]
    if require_complete_baseline and complete:
        selected = min(complete, key=lambda item: item["extrapolated_target_count_cost"])
        estimate = {
            "method": "complete_baseline_model",
            "low": selected["extrapolated_target_count_cost"],
            "mid": selected["extrapolated_target_count_cost"],
            "high": selected["extrapolated_target_count_cost"],
            "baseline_model": selected["model"],
        }
    else:
        per_record_costs = [cost_with_price(row, price) for row in rows]
        if per_record_costs:
            sorted_costs = sorted(per_record_costs)
            p25 = sorted_costs[int((len(sorted_costs) - 1) * 0.25)]
            median = statistics.median(sorted_costs)
            p90 = sorted_costs[int((len(sorted_costs) - 1) * 0.90)]
            estimate = {
                "method": "per_record_distribution_extrapolation",
                "low": p25 * target_count,
                "mid": median * target_count,
                "high": p90 * target_count,
            }
        else:
            estimate = {"method": "no_local_evidence", "low": None, "mid": None, "high": None}

    return {
        "target_count": target_count,
        "historical_records": len(rows),
        "estimate_usd": estimate,
        "historical_model_summaries": model_summaries,
    }


def estimate_band_complete(summary: dict[str, Any]) -> bool:
    estimate = summary.get("estimate_usd")
    if not isinstance(estimate, dict):
        return False
    return all(isinstance(estimate.get(key), (int, float)) for key in ("low", "mid", "high"))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path, help="Local output root to scan.")
    parser.add_argument("--target-model", default=DEFAULT_TARGET_MODEL)
    parser.add_argument("--math-tasks", type=int, default=DEFAULT_MATH_TASKS)
    parser.add_argument("--swe-tasks", type=int, default=DEFAULT_SWE_TASKS)
    parser.add_argument(
        "--nonagentic-buffer-usd",
        type=float,
        default=75.0,
        help="Manual buffer for non-agentic generation and judge calls.",
    )
    parser.add_argument("--output", type=Path, help="Optional JSON output path.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.target_model not in PRICE_PER_MILLION:
        available = ", ".join(sorted(PRICE_PER_MILLION))
        raise SystemExit(f"unknown target model for pricing: {args.target_model}; available: {available}")
    price = PRICE_PER_MILLION[args.target_model]
    rows = scan(args.root)
    by_category: dict[str, list[dict[str, Any]]] = {"agentic_math": [], "swebench_pro": []}
    for row in rows:
        category = category_from_row(row)
        if category in by_category:
            by_category[category].append(row)

    math = summarize_category(
        by_category["agentic_math"],
        target_count=args.math_tasks,
        price=price,
        require_complete_baseline=True,
    )
    swe = summarize_category(
        by_category["swebench_pro"],
        target_count=args.swe_tasks,
        price=price,
    )
    missing_estimates = [
        name
        for name, summary in (("agentic_math", math), ("swebench_pro", swe))
        if not estimate_band_complete(summary)
    ]
    if missing_estimates:
        raise SystemExit(
            "cannot estimate paid canary budget without local token evidence for: "
            + ", ".join(missing_estimates)
            + "; rerun after representative Agentic Math/SWE usage exists or provide a reviewed budget JSON manually"
        )
    subtotal_low = sum(
        value
        for value in (
            math["estimate_usd"].get("low"),
            swe["estimate_usd"].get("low"),
        )
        if isinstance(value, (int, float))
    )
    subtotal_mid = sum(
        value
        for value in (
            math["estimate_usd"].get("mid"),
            swe["estimate_usd"].get("mid"),
        )
        if isinstance(value, (int, float))
    )
    subtotal_high = sum(
        value
        for value in (
            math["estimate_usd"].get("high"),
            swe["estimate_usd"].get("high"),
        )
        if isinstance(value, (int, float))
    )
    result = {
        "schema_version": 1,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "target_model": args.target_model,
        "price_per_million_tokens": price,
        "pricing_note": "OpenAI GPT-4.1 mini release pricing is used for the default OpenAI-direct canary estimate; provider dashboards are authoritative for billing.",
        "pricing_source_url": "https://openai.com/index/gpt-4-1/",
        "agentic_math": math,
        "swebench_pro": swe,
        "nonagentic_buffer_usd": args.nonagentic_buffer_usd,
        "estimated_total_usd": {
            "low": subtotal_low + args.nonagentic_buffer_usd,
            "mid": subtotal_mid + args.nonagentic_buffer_usd,
            "high": subtotal_high + args.nonagentic_buffer_usd,
        },
        "notes": [
            "Provider dashboards are authoritative for billing.",
            "Non-agentic and judge costs are represented by a manual buffer.",
            "SWE-Bench Pro estimate is extrapolated from partial historical runs.",
        ],
    }
    text = json.dumps(result, ensure_ascii=False, indent=2)
    print(text)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
