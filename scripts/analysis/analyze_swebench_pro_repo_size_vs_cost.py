#!/usr/bin/env python3
"""Analyze SWE-Bench Pro repository size versus observed runtime cost.

This script is local/offline. It joins repository size metadata with observed
OpenClaw usage records and reports correlations, repo-level runtime summaries,
large-repo exclusion effects, and top input bursts.
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


OBSERVED_MODEL_PRICE_PER_MILLION: dict[str, dict[str, float]] = {
    "claude-opus-4_7-openrouter-xhigh": {"input": 5.00, "output": 25.00, "cacheRead": 0.50},
    "claude-sonnet-4_6-openrouter-high": {"input": 3.00, "output": 15.00, "cacheRead": 0.30},
    "deepseek-v4-pro-thinking-max": {"input": 1.74, "output": 3.48, "cacheRead": 0.145},
    "gemini-3_1-pro-preview-openrouter": {"input": 2.00, "output": 12.00, "cacheRead": 0.30},
    "qwen3_6-max-preview-openrouter": {"input": 1.10, "output": 3.00, "cacheRead": 0.00},
}

NUMERIC_SIZE_FIELDS = (
    "swebench_pro_instances",
    "tracked_files",
    "tracked_mb",
    "text_tokens_m_cl100k",
    "source_tokens_m_cl100k",
    "largest_file_mb",
    "skipped_large_files_gt_5mb",
    "binary_or_decode_skipped_files",
)


def as_float(value: Any) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0


def percentile(values: list[float], p: float) -> float:
    if not values:
        return 0.0
    xs = sorted(values)
    k = (len(xs) - 1) * p / 100
    low = math.floor(k)
    high = math.ceil(k)
    if low == high:
        return xs[low]
    return xs[low] * (high - k) + xs[high] * (k - low)


def rankdata(values: list[float]) -> list[float]:
    order = sorted(range(len(values)), key=lambda index: values[index])
    ranks = [0.0] * len(values)
    index = 0
    while index < len(order):
        next_index = index + 1
        while next_index < len(order) and values[order[next_index]] == values[order[index]]:
            next_index += 1
        average_rank = (index + 1 + next_index) / 2
        for ranked_index in range(index, next_index):
            ranks[order[ranked_index]] = average_rank
        index = next_index
    return ranks


def pearson(xs: list[float], ys: list[float]) -> float | None:
    if len(xs) < 2:
        return None
    mean_x = sum(xs) / len(xs)
    mean_y = sum(ys) / len(ys)
    var_x = sum((x - mean_x) ** 2 for x in xs)
    var_y = sum((y - mean_y) ** 2 for y in ys)
    if var_x == 0 or var_y == 0:
        return None
    return sum((x - mean_x) * (y - mean_y) for x, y in zip(xs, ys)) / math.sqrt(var_x * var_y)


def spearman(xs: list[float], ys: list[float]) -> float | None:
    return pearson(rankdata(xs), rankdata(ys))


def correlation_table(
    rows: list[dict[str, Any]],
    x_keys: list[str],
    y_keys: list[str],
) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    for x_key in x_keys:
        for y_key in y_keys:
            xs = [float(row[x_key]) for row in rows]
            ys = [float(row[y_key]) for row in rows]
            output.append(
                {
                    "x": x_key,
                    "y": y_key,
                    "n": len(rows),
                    "pearson": pearson(xs, ys),
                    "spearman": spearman(xs, ys),
                }
            )
    return output


def load_repo_sizes(path: Path) -> dict[str, dict[str, Any]]:
    sizes: dict[str, dict[str, Any]] = {}
    with path.open(encoding="utf-8", newline="") as f:
        for row in csv.DictReader(f):
            normalized = dict(row)
            for field in NUMERIC_SIZE_FIELDS:
                normalized[field] = as_float(row.get(field))
            sizes[str(row["repo"])] = normalized
    return sizes


def load_usage_records(path: Path) -> list[dict[str, Any]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    records = payload.get("records")
    if not isinstance(records, list):
        raise ValueError(f"{path} does not contain records list")
    return [record for record in records if isinstance(record, dict)]


def observed_cost(record: dict[str, Any]) -> float:
    price = OBSERVED_MODEL_PRICE_PER_MILLION.get(str(record.get("model")), {})
    return (
        as_float(record.get("input")) / 1_000_000 * price.get("input", 0.0)
        + as_float(record.get("output")) / 1_000_000 * price.get("output", 0.0)
        + as_float(record.get("cacheRead")) / 1_000_000 * price.get("cacheRead", 0.0)
    )


def join_records(
    repo_sizes: dict[str, dict[str, Any]],
    usage_records: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    joined: list[dict[str, Any]] = []
    for record in usage_records:
        repo = str(record.get("repo"))
        if repo not in repo_sizes:
            continue
        size = repo_sizes[repo]
        joined.append(
            {
                "model": record.get("model"),
                "repo": repo,
                "instance_id": record.get("instance_id"),
                "observed_input": as_float(record.get("input")),
                "observed_output": as_float(record.get("output")),
                "observed_cacheRead": as_float(record.get("cacheRead")),
                "observed_tool_calls": as_float(record.get("tool_calls")),
                "observed_cost_usd": observed_cost(record),
                "text_tokens_m_cl100k": float(size["text_tokens_m_cl100k"]),
                "source_tokens_m_cl100k": float(size["source_tokens_m_cl100k"]),
                "tracked_mb": float(size["tracked_mb"]),
                "tracked_files": float(size["tracked_files"]),
                "largest_file_mb": float(size["largest_file_mb"]),
                "swebench_pro_instances": float(size["swebench_pro_instances"]),
            }
        )
    return joined


def build_repo_rows(
    repo_sizes: dict[str, dict[str, Any]],
    joined: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    by_repo: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in joined:
        by_repo[str(row["repo"])].append(row)

    rows: list[dict[str, Any]] = []
    for repo, repo_records in by_repo.items():
        size = repo_sizes[repo]
        inputs = [float(row["observed_input"]) for row in repo_records]
        tools = [float(row["observed_tool_calls"]) for row in repo_records]
        costs = [float(row["observed_cost_usd"]) for row in repo_records]
        text_tokens = float(size["text_tokens_m_cl100k"])
        rows.append(
            {
                "repo": repo,
                "public_instances": int(size["swebench_pro_instances"]),
                "observed_runs": len(repo_records),
                "tracked_files": int(size["tracked_files"]),
                "tracked_mb": float(size["tracked_mb"]),
                "text_tokens_m_cl100k": text_tokens,
                "source_tokens_m_cl100k": float(size["source_tokens_m_cl100k"]),
                "largest_file_mb": float(size["largest_file_mb"]),
                "mean_input": statistics.mean(inputs),
                "median_input": percentile(inputs, 50),
                "p90_input": percentile(inputs, 90),
                "max_input": max(inputs),
                "mean_tool_calls": statistics.mean(tools),
                "p90_tool_calls": percentile(tools, 90),
                "max_tool_calls": max(tools),
                "mean_cost_usd_observed_models": statistics.mean(costs),
                "max_cost_usd_observed_models": max(costs),
                "input_per_text_repo_token": statistics.mean(inputs) / (text_tokens * 1_000_000)
                if text_tokens
                else 0,
            }
        )
    return sorted(rows, key=lambda row: float(row["text_tokens_m_cl100k"]), reverse=True)


def build_exclusion_rows(
    repo_sizes: dict[str, dict[str, Any]],
    joined: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    all_size_rows = [
        {
            "repo": repo,
            "text_tokens_m_cl100k": float(size["text_tokens_m_cl100k"]),
            "public_instances": int(size["swebench_pro_instances"]),
        }
        for repo, size in repo_sizes.items()
    ]
    sorted_by_text = sorted(all_size_rows, key=lambda row: row["text_tokens_m_cl100k"], reverse=True)
    scenarios = [
        ("exclude_top1_text_repo", [sorted_by_text[0]["repo"]]),
        ("exclude_top3_text_repos", [row["repo"] for row in sorted_by_text[:3]]),
        ("exclude_top4_text_repos_or_text_tokens_gt5m", [row["repo"] for row in sorted_by_text[:4]]),
        (
            "exclude_text_tokens_gt10m",
            [row["repo"] for row in sorted_by_text if row["text_tokens_m_cl100k"] > 10],
        ),
        (
            "exclude_text_tokens_gt2m",
            [row["repo"] for row in sorted_by_text if row["text_tokens_m_cl100k"] > 2],
        ),
    ]

    all_input = sum(float(row["observed_input"]) for row in joined)
    all_cost = sum(float(row["observed_cost_usd"]) for row in joined)
    all_public = sum(int(size["swebench_pro_instances"]) for size in repo_sizes.values())
    output: list[dict[str, Any]] = []
    for scenario, repos in scenarios:
        repo_set = set(repos)
        remaining = [row for row in joined if row["repo"] not in repo_set]
        removed_public = sum(int(repo_sizes[repo]["swebench_pro_instances"]) for repo in repos)
        output.append(
            {
                "scenario": scenario,
                "excluded_repos": ";".join(repos),
                "excluded_repo_count": len(repos),
                "remaining_observed_runs": len(remaining),
                "observed_input_remaining_pct": sum(float(row["observed_input"]) for row in remaining)
                / all_input
                * 100
                if all_input
                else 0,
                "observed_cost_remaining_pct": sum(float(row["observed_cost_usd"]) for row in remaining)
                / all_cost
                * 100
                if all_cost
                else 0,
                "public_instances_removed": removed_public,
                "public_instances_removed_pct": removed_public / all_public * 100 if all_public else 0,
                "public_instances_remaining": all_public - removed_public,
            }
        )
    return output


def find_corr(table: list[dict[str, Any]], x_key: str, y_key: str) -> dict[str, Any]:
    for row in table:
        if row["x"] == x_key and row["y"] == y_key:
            return row
    raise KeyError((x_key, y_key))


def format_corr(value: float | None) -> str:
    return "nan" if value is None else f"{value:.3f}"


def render_markdown(path: Path, payload: dict[str, Any]) -> None:
    run_corr = payload["run_level_correlations"]
    repo_corr = payload["repo_level_correlations"]
    repo_rows = payload["repo_rows"]
    exclusion_rows = payload["exclusion_scenarios"]
    top_bursts = payload["top_bursts"]

    selected_correlations = [
        ("run", run_corr, "text_tokens_m_cl100k", "observed_input"),
        ("run", run_corr, "source_tokens_m_cl100k", "observed_input"),
        ("run", run_corr, "tracked_mb", "observed_input"),
        ("run", run_corr, "tracked_files", "observed_input"),
        ("run", run_corr, "text_tokens_m_cl100k", "observed_tool_calls"),
        ("repo", repo_corr, "text_tokens_m_cl100k", "mean_input"),
        ("repo", repo_corr, "text_tokens_m_cl100k", "p90_input"),
        ("repo", repo_corr, "text_tokens_m_cl100k", "max_input"),
        ("repo", repo_corr, "tracked_files", "mean_input"),
        ("repo", repo_corr, "public_instances", "mean_input"),
    ]

    lines = [
        "# SWE-Bench Pro repo size vs observed runtime cost",
        "",
        "Inputs: repo size metadata and observed OpenClaw usage records. No paid API calls or W&B access were made.",
        "",
        "## Correlation summary",
        "",
        "| level | x | y | n | Pearson | Spearman |",
        "|---|---|---|---:|---:|---:|",
    ]
    for level, table, x_key, y_key in selected_correlations:
        row = find_corr(table, x_key, y_key)
        lines.append(
            f"| {level} | {x_key} | {y_key} | {row['n']} | "
            f"{format_corr(row['pearson'])} | {format_corr(row['spearman'])} |"
        )

    lines.extend(
        [
            "",
            "## Repo-level table",
            "",
            "| repo | public n | observed n | text tokens M | tracked MB | mean input | p90 input | max input | mean tools | max tools |",
            "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for row in repo_rows:
        lines.append(
            f"| {row['repo']} | {row['public_instances']} | {row['observed_runs']} | "
            f"{row['text_tokens_m_cl100k']:.3f} | {row['tracked_mb']:.1f} | "
            f"{row['mean_input']:.0f} | {row['p90_input']:.0f} | {row['max_input']:.0f} | "
            f"{row['mean_tool_calls']:.1f} | {row['max_tool_calls']:.0f} |"
        )

    lines.extend(
        [
            "",
            "## Large-repo exclusion scenarios on observed runs",
            "",
            "| scenario | excluded repos | public removed | public removed % | observed input remaining % | observed cost remaining % |",
            "|---|---|---:|---:|---:|---:|",
        ]
    )
    for row in exclusion_rows:
        lines.append(
            f"| {row['scenario']} | {row['excluded_repos']} | "
            f"{row['public_instances_removed']} | {row['public_instances_removed_pct']:.1f}% | "
            f"{row['observed_input_remaining_pct']:.1f}% | {row['observed_cost_remaining_pct']:.1f}% |"
        )

    lines.extend(
        [
            "",
            "## Top input bursts",
            "",
            "| model | repo | input | tools | repo text tokens M |",
            "|---|---|---:|---:|---:|",
        ]
    )
    for row in top_bursts[:10]:
        lines.append(
            f"| {row['model']} | {row['repo']} | {row['observed_input']:.0f} | "
            f"{row['observed_tool_calls']:.0f} | {row['text_tokens_m_cl100k']:.3f} |"
        )

    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            "- Repository size has weak to negative correlation with observed input tokens in the current 134-run sample.",
            "- The largest observed bursts are often in small or mid-size repositories such as flipt, ansible, navidrome, and vuls.",
            "- Large-repo exclusion is useful as a static sampling and operational-risk filter, but runtime input/tool caps are the primary guard against the observed long tail.",
        ]
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-size-csv", type=Path, default=Path("temp/swebench_pro_repo_size_report.csv"))
    parser.add_argument(
        "--usage-json",
        type=Path,
        default=Path("temp/swebench_pro_observed_usage_distribution.json"),
    )
    parser.add_argument(
        "--repo-summary-csv",
        type=Path,
        default=Path("temp/swebench_pro_repo_size_vs_cost_analysis.csv"),
    )
    parser.add_argument(
        "--analysis-json",
        type=Path,
        default=Path("temp/swebench_pro_repo_size_vs_cost_analysis.json"),
    )
    parser.add_argument(
        "--summary-md",
        type=Path,
        default=Path("temp/swebench_pro_repo_size_vs_cost_analysis.md"),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    repo_sizes = load_repo_sizes(args.repo_size_csv)
    usage_records = load_usage_records(args.usage_json)
    joined = join_records(repo_sizes, usage_records)
    repo_rows = build_repo_rows(repo_sizes, joined)

    run_x = [
        "text_tokens_m_cl100k",
        "source_tokens_m_cl100k",
        "tracked_mb",
        "tracked_files",
        "largest_file_mb",
        "swebench_pro_instances",
    ]
    repo_x = [
        "text_tokens_m_cl100k",
        "source_tokens_m_cl100k",
        "tracked_mb",
        "tracked_files",
        "largest_file_mb",
        "public_instances",
    ]
    run_corr = correlation_table(joined, run_x, ["observed_input", "observed_tool_calls", "observed_cost_usd"])
    repo_corr = correlation_table(
        repo_rows,
        repo_x,
        [
            "mean_input",
            "median_input",
            "p90_input",
            "max_input",
            "mean_tool_calls",
            "max_tool_calls",
            "mean_cost_usd_observed_models",
        ],
    )
    model_corr = {
        model: correlation_table(
            [row for row in joined if row["model"] == model],
            run_x,
            ["observed_input", "observed_tool_calls", "observed_cost_usd"],
        )
        for model in sorted({row["model"] for row in joined})
    }
    payload = {
        "joined_records": len(joined),
        "repo_size_rows": len(repo_sizes),
        "observed_repo_rows": len(repo_rows),
        "repo_rows": repo_rows,
        "run_level_correlations": run_corr,
        "repo_level_correlations": repo_corr,
        "model_level_correlations": model_corr,
        "exclusion_scenarios": build_exclusion_rows(repo_sizes, joined),
        "top_bursts": sorted(joined, key=lambda row: float(row["observed_input"]), reverse=True)[:15],
    }

    write_csv(args.repo_summary_csv, repo_rows)
    args.analysis_json.parent.mkdir(parents=True, exist_ok=True)
    args.analysis_json.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    render_markdown(args.summary_md, payload)
    print(
        json.dumps(
            {
                "ok": True,
                "joined_records": len(joined),
                "repo_size_rows": len(repo_sizes),
                "observed_repo_rows": len(repo_rows),
                "repo_summary_csv": str(args.repo_summary_csv),
                "analysis_json": str(args.analysis_json),
                "summary_md": str(args.summary_md),
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
