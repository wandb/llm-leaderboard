#!/usr/bin/env python3
"""Simulate SWE-Bench Pro runtime cap costs from observed local usage records.

This script does not call model APIs or W&B. It reads the local observed usage
distribution produced from OpenClaw outputs and estimates cost under input-token
and tool-call caps. The cap model is deliberately conservative: input and
cache-read tokens are scaled down, while output tokens are left unchanged.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Any


PRICE_PER_MILLION: dict[str, dict[str, float]] = {
    "openai-small-like": {"input": 0.40, "output": 1.60, "cacheRead": 0.10},
    "low-deepseek-like": {"input": 1.74, "output": 3.48, "cacheRead": 0.145},
    "mid-glm5.2-like": {"input": 0.95, "output": 3.00, "cacheRead": 0.18},
    "high-sonnet-like": {"input": 3.00, "output": 15.00, "cacheRead": 0.30},
    "very-high-opus-gpt5.5-like": {"input": 5.00, "output": 25.00, "cacheRead": 0.50},
    "gemini-pro-like": {"input": 2.00, "output": 12.00, "cacheRead": 0.30},
    "qwen-max-like": {"input": 1.10, "output": 3.00, "cacheRead": 0.00},
    "qwen-flash-lowcost-like": {"input": 0.065, "output": 0.260, "cacheRead": 0.00},
}

DEFAULT_GROUPS: dict[str, tuple[str, str]] = {
    "openai-small-like_on_deepseek_behavior": (
        "deepseek-v4-pro-thinking-max",
        "openai-small-like",
    ),
    "low-deepseek-like": ("deepseek-v4-pro-thinking-max", "low-deepseek-like"),
    "mid-glm5.2-like_on_deepseek_behavior": (
        "deepseek-v4-pro-thinking-max",
        "mid-glm5.2-like",
    ),
    "high-sonnet-like": ("claude-sonnet-4_6-openrouter-high", "high-sonnet-like"),
    "very-high-opus-gpt5.5-like": (
        "claude-opus-4_7-openrouter-xhigh",
        "very-high-opus-gpt5.5-like",
    ),
    "gemini-pro-like": ("gemini-3_1-pro-preview-openrouter", "gemini-pro-like"),
    "qwen-max-like": ("qwen3_6-max-preview-openrouter", "qwen-max-like"),
    "qwen-flash-lowcost-like_on_deepseek_behavior": (
        "deepseek-v4-pro-thinking-max",
        "qwen-flash-lowcost-like",
    ),
}

TOKEN_KEYS = ("input", "output", "cacheRead")


def percentile(values: list[float], p: float) -> float:
    if not values:
        return 0.0
    xs = sorted(values)
    k = (len(xs) - 1) * p / 100
    f = math.floor(k)
    c = math.ceil(k)
    if f == c:
        return xs[int(k)]
    return xs[f] * (c - k) + xs[c] * (k - f)


def parse_cap_list(value: str) -> list[int | None]:
    caps: list[int | None] = []
    for item in value.split(","):
        item = item.strip()
        if not item:
            continue
        if item.lower() in {"none", "null", "uncapped"}:
            caps.append(None)
        else:
            caps.append(int(item.replace("_", "")))
    return caps


def cap_label(value: int | None) -> str:
    return "none" if value is None else str(value)


def load_records(path: Path) -> list[dict[str, Any]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    records = payload.get("records")
    if not isinstance(records, list):
        raise ValueError(f"{path} does not contain a records list")
    return [record for record in records if isinstance(record, dict)]


def group_records(records: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        model = record.get("model")
        if isinstance(model, str):
            grouped[model].append(record)
    return grouped


def effective_scale(
    record: dict[str, Any],
    token_cap: int | None,
    cap_field: str,
    cap_value: int | None,
) -> float:
    scale = 1.0
    input_tokens = float(record.get("input") or 0)
    if token_cap is not None and input_tokens > 0:
        scale = min(scale, token_cap / input_tokens)
    event_count = float(record.get(cap_field) or 0)
    if cap_value is not None and event_count > 0:
        scale = min(scale, cap_value / event_count)
    return max(0.0, min(1.0, scale))


def cost_record(
    record: dict[str, Any],
    price: dict[str, float],
    scale: float,
    *,
    scale_output: bool,
) -> float:
    output_scale = scale if scale_output else 1.0
    return (
        float(record.get("input") or 0) * scale / 1_000_000 * price["input"]
        + float(record.get("cacheRead") or 0) * scale / 1_000_000 * price.get("cacheRead", 0.0)
        + float(record.get("output") or 0) * output_scale / 1_000_000 * price["output"]
    )


def distribution(rows: list[dict[str, Any]], key: str) -> dict[str, float]:
    values = [float(row.get(key) or 0) for row in rows]
    if not values:
        return {"p50": 0.0, "p75": 0.0, "p90": 0.0, "p95": 0.0, "max": 0.0, "mean": 0.0}
    return {
        "p50": percentile(values, 50),
        "p75": percentile(values, 75),
        "p90": percentile(values, 90),
        "p95": percentile(values, 95),
        "max": max(values),
        "mean": statistics.mean(values),
    }


def simulate_main(
    grouped: dict[str, list[dict[str, Any]]],
    *,
    token_caps: list[int | None],
    tool_caps: list[int | None],
    target_tasks: int,
    scale_output: bool,
) -> list[dict[str, Any]]:
    rows_out: list[dict[str, Any]] = []
    for tier, (observed_model, price_profile) in DEFAULT_GROUPS.items():
        rows = grouped.get(observed_model, [])
        if not rows:
            continue
        price = PRICE_PER_MILLION[price_profile]
        baseline = sum(cost_record(row, price, 1.0, scale_output=scale_output) for row in rows)
        baseline_projected = baseline / len(rows) * target_tasks
        for token_cap in token_caps:
            for tool_cap in tool_caps:
                sample_cost = 0.0
                token_affected = 0
                tool_affected = 0
                any_affected = 0
                original_input = 0.0
                scaled_input = 0.0
                original_cache = 0.0
                scaled_cache = 0.0
                scales: list[float] = []
                for row in rows:
                    scale = effective_scale(row, token_cap, "tool_calls", tool_cap)
                    scales.append(scale)
                    sample_cost += cost_record(row, price, scale, scale_output=scale_output)
                    input_tokens = float(row.get("input") or 0)
                    tool_calls = float(row.get("tool_calls") or 0)
                    token_affected += int(token_cap is not None and input_tokens > token_cap)
                    tool_affected += int(tool_cap is not None and tool_calls > tool_cap)
                    any_affected += int(scale < 0.999999)
                    original_input += input_tokens
                    scaled_input += input_tokens * scale
                    original_cache += float(row.get("cacheRead") or 0)
                    scaled_cache += float(row.get("cacheRead") or 0) * scale
                projected = sample_cost / len(rows) * target_tasks
                rows_out.append(
                    {
                        "scenario": "token_cap_plus_tool_call_cap",
                        "tier": tier,
                        "observed_behavior_model": observed_model,
                        "price_profile": price_profile,
                        "n_observed": len(rows),
                        "target_tasks": target_tasks,
                        "token_cap_input_tokens": cap_label(token_cap),
                        "tool_call_cap": cap_label(tool_cap),
                        "affected_token_records": token_affected,
                        "affected_tool_records": tool_affected,
                        "affected_any_records": any_affected,
                        "input_multiplier": scaled_input / original_input if original_input else "",
                        "cacheRead_multiplier": scaled_cache / original_cache if original_cache else "",
                        "cost_observed_sample_usd": sample_cost,
                        "cost_projected_usd": projected,
                        "cost_ratio_vs_uncapped": projected / baseline_projected if baseline_projected else "",
                        "savings_vs_uncapped_pct": (
                            (1 - projected / baseline_projected) * 100 if baseline_projected else ""
                        ),
                        "mean_scale": statistics.mean(scales) if scales else "",
                        "p50_scale": percentile(scales, 50) if scales else "",
                        "p10_scale": percentile(scales, 10) if scales else "",
                        "output_policy": "scaled" if scale_output else "unchanged_conservative",
                    }
                )
    return rows_out


def simulate_event_proxy(
    grouped: dict[str, list[dict[str, Any]]],
    *,
    token_caps: list[int | None],
    event_caps: list[int | None],
    target_tasks: int,
    scale_output: bool,
) -> list[dict[str, Any]]:
    rows_out: list[dict[str, Any]] = []
    for cap_field in ("tool_calls", "assistant_events", "timeline_events"):
        for tier, (observed_model, price_profile) in DEFAULT_GROUPS.items():
            rows = grouped.get(observed_model, [])
            if not rows:
                continue
            price = PRICE_PER_MILLION[price_profile]
            baseline = sum(cost_record(row, price, 1.0, scale_output=scale_output) for row in rows)
            baseline_projected = baseline / len(rows) * target_tasks
            for token_cap in token_caps:
                for event_cap in event_caps:
                    sample_cost = 0.0
                    affected = 0
                    original_input = 0.0
                    scaled_input = 0.0
                    for row in rows:
                        scale = effective_scale(row, token_cap, cap_field, event_cap)
                        sample_cost += cost_record(row, price, scale, scale_output=scale_output)
                        affected += int(scale < 0.999999)
                        input_tokens = float(row.get("input") or 0)
                        original_input += input_tokens
                        scaled_input += input_tokens * scale
                    projected = sample_cost / len(rows) * target_tasks
                    rows_out.append(
                        {
                            "cap_field": cap_field,
                            "tier": tier,
                            "observed_behavior_model": observed_model,
                            "price_profile": price_profile,
                            "n_observed": len(rows),
                            "target_tasks": target_tasks,
                            "token_cap_input_tokens": cap_label(token_cap),
                            "event_cap": cap_label(event_cap),
                            "affected_records": affected,
                            "input_multiplier": scaled_input / original_input if original_input else "",
                            "cost_projected_usd": projected,
                            "cost_ratio_vs_uncapped": (
                                projected / baseline_projected if baseline_projected else ""
                            ),
                            "savings_vs_uncapped_pct": (
                                (1 - projected / baseline_projected) * 100
                                if baseline_projected
                                else ""
                            ),
                            "output_policy": "scaled" if scale_output else "unchanged_conservative",
                        }
                    )
    return rows_out


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def selected_row(
    rows: list[dict[str, Any]],
    *,
    tier: str,
    token_cap: int | None,
    tool_cap: int | None,
) -> dict[str, Any]:
    for row in rows:
        if (
            row["tier"] == tier
            and row["token_cap_input_tokens"] == cap_label(token_cap)
            and row["tool_call_cap"] == cap_label(tool_cap)
        ):
            return row
    raise KeyError((tier, token_cap, tool_cap))


def render_markdown(
    path: Path,
    grouped: dict[str, list[dict[str, Any]]],
    main_rows: list[dict[str, Any]],
    *,
    records_path: Path,
    target_tasks: int,
) -> None:
    selected = [
        (None, None, "no cap"),
        (1_000_000, 40, "1M+40"),
        (1_000_000, 60, "1M+60"),
        (1_000_000, 80, "1M+80"),
        (2_000_000, 60, "2M+60"),
        (500_000, 60, "500k+60"),
        (500_000, 40, "500k+40"),
    ]
    tiers = [
        "openai-small-like_on_deepseek_behavior",
        "low-deepseek-like",
        "mid-glm5.2-like_on_deepseek_behavior",
        "high-sonnet-like",
        "very-high-opus-gpt5.5-like",
        "gemini-pro-like",
        "qwen-max-like",
    ]
    lines = [
        "# SWE-Bench Pro runtime cap cost simulation",
        "",
        f"Generated from `{records_path}`. No paid API calls are made.",
        "",
        "## Method",
        "",
        "- Main cap model: `scale = min(1, token_cap / input, tool_call_cap / tool_calls)`.",
        "- `input` and `cacheRead` are scaled; `output` is unchanged, so savings are conservative.",
        f"- Costs are extrapolated to {target_tasks} tasks by observed per-task mean.",
        "- Exact turn-by-turn cumulative token traces are not present in these local records.",
        "",
        "## Observed Distribution",
        "",
        "| observed behavior | n | input p50 | input p90 | input max | tool p50 | tool p90 | tool max |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for observed_model, rows in sorted(grouped.items()):
        if observed_model not in {value[0] for value in DEFAULT_GROUPS.values()}:
            continue
        input_stat = distribution(rows, "input")
        tool_stat = distribution(rows, "tool_calls")
        lines.append(
            "| "
            + f"{observed_model} | {len(rows)} | {input_stat['p50']:.0f} | "
            + f"{input_stat['p90']:.0f} | {input_stat['max']:.0f} | "
            + f"{tool_stat['p50']:.0f} | {tool_stat['p90']:.0f} | {tool_stat['max']:.0f} |"
        )
    lines.extend(
        [
            "",
            f"## Selected Projected {target_tasks}-Task Costs",
            "",
            "| tier | no cap | 1M+40 | 1M+60 | 1M+80 | 2M+60 | 500k+60 | 500k+40 |",
            "|---|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for tier in tiers:
        values = []
        for token_cap, tool_cap, _label in selected:
            row = selected_row(main_rows, tier=tier, token_cap=token_cap, tool_cap=tool_cap)
            values.append(f"${float(row['cost_projected_usd']):.2f}")
        lines.append(f"| {tier} | " + " | ".join(values) + " |")
    lines.extend(
        [
            "",
            "## Selected Savings Vs Uncapped",
            "",
            "| tier | 1M+40 | 1M+60 | 1M+80 | 2M+60 | 500k+60 | 500k+40 |",
            "|---|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for tier in tiers:
        values = []
        for token_cap, tool_cap, _label in selected[1:]:
            row = selected_row(main_rows, tier=tier, token_cap=token_cap, tool_cap=tool_cap)
            values.append(f"{float(row['savings_vs_uncapped_pct']):.1f}%")
        lines.append(f"| {tier} | " + " | ".join(values) + " |")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--records-json",
        type=Path,
        default=Path("temp/swebench_pro_observed_usage_distribution.json"),
    )
    parser.add_argument(
        "--main-csv",
        type=Path,
        default=Path("temp/swebench_pro_runtime_cap_simulation_detailed.csv"),
    )
    parser.add_argument(
        "--event-proxy-csv",
        type=Path,
        default=Path("temp/swebench_pro_event_proxy_cap_simulation.csv"),
    )
    parser.add_argument(
        "--summary-md",
        type=Path,
        default=Path("temp/swebench_pro_runtime_cap_simulation_summary.md"),
    )
    parser.add_argument("--target-tasks", type=int, default=80)
    parser.add_argument(
        "--token-caps",
        default="none,250000,500000,750000,1000000,1500000,2000000,3000000",
    )
    parser.add_argument("--tool-caps", default="none,20,40,60,80,100")
    parser.add_argument("--event-caps", default="none,20,40,60,80,100,150,200")
    parser.add_argument(
        "--scale-output",
        action="store_true",
        help="Also scale output tokens. Default leaves output unchanged.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    records = load_records(args.records_json)
    grouped = group_records(records)
    main_rows = simulate_main(
        grouped,
        token_caps=parse_cap_list(args.token_caps),
        tool_caps=parse_cap_list(args.tool_caps),
        target_tasks=args.target_tasks,
        scale_output=args.scale_output,
    )
    event_rows = simulate_event_proxy(
        grouped,
        token_caps=parse_cap_list("none,500000,1000000,2000000"),
        event_caps=parse_cap_list(args.event_caps),
        target_tasks=args.target_tasks,
        scale_output=args.scale_output,
    )
    write_csv(args.main_csv, main_rows)
    write_csv(args.event_proxy_csv, event_rows)
    render_markdown(
        args.summary_md,
        grouped,
        main_rows,
        records_path=args.records_json,
        target_tasks=args.target_tasks,
    )
    print(
        json.dumps(
            {
                "ok": True,
                "records": len(records),
                "main_rows": len(main_rows),
                "event_proxy_rows": len(event_rows),
                "main_csv": str(args.main_csv),
                "event_proxy_csv": str(args.event_proxy_csv),
                "summary_md": str(args.summary_md),
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
