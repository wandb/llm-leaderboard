#!/usr/bin/env python3
"""Build SWE-rebench v2 vs DeepSWE scatter plots.

The source leaderboards use different naming conventions and sometimes expose
different configuration detail. This script writes both a strict exact-config
view and a broader semantic view:

- strict: same model generation and same explicit effort/configuration.
- semantic: same model generation, allowing effort to be missing on one side or
  mismatched, with the caveat recorded in the output CSV.

Cross-generation rows are kept in the audit file but excluded from plots.
"""

from __future__ import annotations

import csv
import json
import math
import re
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import requests
from bs4 import BeautifulSoup
from scipy import stats


SWE_REBENCH_URL = "https://swe-rebench.com/"
DEEPSWE_URL = "https://deepswe.datacurve.ai/"

OUT_DATA = Path("data/analysis")
OUT_FIGURES = Path("docs/figures")


@dataclass(frozen=True)
class SemanticMatch:
    canonical_model: str
    swerebench_model: str
    deepswe_model: str
    deepswe_effort: str | None
    decision: str
    include_in_strict: bool
    include_in_semantic: bool
    rationale: str


MATCH_AUDIT: list[SemanticMatch] = [
    SemanticMatch(
        "gpt-5.5[xhigh]",
        "gpt-5.5-2026-04-23-xhigh",
        "gpt-5-5",
        "xhigh",
        "strict_config_match",
        True,
        True,
        "same GPT-5.5 dated model generation and explicit xhigh effort",
    ),
    SemanticMatch(
        "gpt-5.5[medium]",
        "gpt-5.5-2026-04-23-medium",
        "gpt-5-5",
        "medium",
        "strict_config_match",
        True,
        True,
        "same GPT-5.5 dated model generation and explicit medium effort",
    ),
    SemanticMatch(
        "claude-opus-4.8[xhigh]",
        "Claude Opus 4.8-xhigh",
        "claude-opus-4-8",
        "xhigh",
        "strict_config_match",
        True,
        True,
        "same Claude Opus 4.8 generation and explicit xhigh effort",
    ),
    SemanticMatch(
        "gpt-5.4[swerebench medium / deepswe xhigh]",
        "gpt-5.4-2026-03-05-medium",
        "gpt-5-4",
        "xhigh",
        "same_generation_effort_mismatch",
        False,
        True,
        "same GPT-5.4 generation, but SWE-rebench is medium and DeepSWE only has xhigh",
    ),
    SemanticMatch(
        "claude-sonnet-4.6[deepswe high]",
        "Claude Sonnet 4.6",
        "claude-sonnet-4-6",
        "high",
        "same_generation_config_uncertain",
        False,
        True,
        "same Sonnet 4.6 generation, but SWE-rebench row does not expose an effort label",
    ),
    SemanticMatch(
        "gemini-3.1-pro-preview[deepswe high]",
        "Gemini 3.1 Pro Preview",
        "gemini-3-1-pro-preview",
        "high",
        "same_generation_config_uncertain",
        False,
        True,
        "same Gemini 3.1 Pro Preview generation, but SWE-rebench row does not expose an effort label",
    ),
    SemanticMatch(
        "gemini-3.5-flash[deepswe medium]",
        "Gemini 3.5 Flash",
        "gemini-3-5-flash",
        "medium",
        "same_generation_config_uncertain",
        False,
        True,
        "same Gemini 3.5 Flash generation, but SWE-rebench row does not expose an effort label",
    ),
    SemanticMatch(
        "glm",
        "GLM-5.1",
        "glm-5-2",
        "max",
        "different_generation",
        False,
        False,
        "GLM-5.1 and GLM-5.2 are different generations",
    ),
    SemanticMatch(
        "kimi-k2",
        "Kimi K2.6",
        "kimi-k2-7-code",
        None,
        "different_generation_or_variant",
        False,
        False,
        "Kimi K2.6 and Kimi K2.7 Code are different generation/variant labels",
    ),
]


def parse_percent(value: str) -> float | None:
    value = value.strip()
    if value == "N/A":
        return None
    return float(value.rstrip("%"))


def fetch_swerebench() -> tuple[list[dict], dict]:
    html = requests.get(SWE_REBENCH_URL, timeout=30).text
    soup = BeautifulSoup(html, "html.parser")
    text_lines = soup.get_text("\n", strip=True).splitlines()
    table = soup.find("table")
    if table is None:
        raise RuntimeError("Could not find SWE-rebench leaderboard table")

    rows = []
    for tr in table.find_all("tr")[1:]:
        cells = [cell.get_text(" ", strip=True) for cell in tr.find_all("td")]
        if not cells:
            continue
        resolved_rate = parse_percent(cells[2])
        row = {
            "rank": int(cells[0]),
            "model": cells[1],
            "resolved_rate": resolved_rate,
            "resolved_rate_sem": parse_percent(cells[3]),
            "pass_at_5": parse_percent(cells[4]),
            "cost_per_problem": cells[5],
            "tokens_per_problem": cells[6],
            "cached_tokens": cells[7],
        }
        if resolved_rate is not None:
            rows.append(row)

    meta = {
        "source_url": SWE_REBENCH_URL,
        "title": text_lines[0] if text_lines else "SWE-rebench Leaderboard",
        "date_from": None,
        "date_to": None,
        "problem_count": None,
        "repo_count": None,
    }
    # The current page renders the selected date range early in the text.
    dates = [line for line in text_lines[:80] if re.fullmatch(r"\d{2}/\d{2}/\d{4}", line)]
    if len(dates) >= 2:
        meta["date_from"] = dates[0]
        meta["date_to"] = dates[1]
    for idx, line in enumerate(text_lines):
        if line == "problems from" and idx > 0:
            meta["problem_count"] = int(text_lines[idx - 1])
        if line == "repositories selected within the current time window." and idx > 0:
            meta["repo_count"] = int(text_lines[idx - 1])
    return rows, meta


def fetch_deepswe() -> tuple[list[dict], dict]:
    html = requests.get(DEEPSWE_URL, timeout=30).text
    pattern = re.compile(
        r'model:"(?P<model>[^"]+)",'
        r'harness:"(?P<harness>[^"]+)",'
        r'reasoning_effort:(?P<effort>null|"[^"]+"),'
        r'config:"(?P<config>[^"]+)",'
        r'source:"(?P<source>[^"]+)",'
        r'pass_rate:(?P<pass_rate>[0-9.]+),'
        r'pass_at_1:(?P<pass_at_1>[0-9.]+),'
        r'pass_at_4:(?P<pass_at_4>[0-9.]+),'
        r'n_passed:(?P<n_passed>\d+),'
        r'n_attempted:(?P<n_attempted>\d+),.*?'
        r'ci_half:(?P<ci_half>[0-9.]+),.*?'
        r'mean_cost_usd:(?P<mean_cost_usd>[0-9.]+),.*?'
        r'mean_output_tokens:(?P<mean_output_tokens>[0-9.]+),.*?'
        r'mean_agent_steps:(?P<mean_agent_steps>[0-9.]+)',
        re.S,
    )

    rows = []
    for match in pattern.finditer(html):
        row = match.groupdict()
        row["reasoning_effort"] = None if row.pop("effort") == "null" else match.group("effort").strip('"')
        for key in [
            "pass_rate",
            "pass_at_1",
            "pass_at_4",
            "ci_half",
            "mean_cost_usd",
            "mean_output_tokens",
            "mean_agent_steps",
        ]:
            row[key] = float(row[key])
        for key in ["n_passed", "n_attempted"]:
            row[key] = int(row[key])
        rows.append(row)
    if not rows:
        raise RuntimeError("Could not parse DeepSWE leaderboard rows")

    generated_match = re.search(r'generated_at:"([^"]+)"', html)
    latest_job_match = re.search(
        r'latest_job:\$R\[\d+\]=\{name:"([^"]+)",finished_at:"([^"]+)"\}',
        html,
    )
    meta = {
        "source_url": DEEPSWE_URL,
        "generated_at": generated_match.group(1) if generated_match else None,
        "latest_job_name": latest_job_match.group(1) if latest_job_match else None,
        "latest_job_finished_at": latest_job_match.group(2) if latest_job_match else None,
    }
    return rows, meta


def pearson(x: list[float], y: list[float]) -> float | None:
    if len(x) < 2:
        return None
    return float(stats.pearsonr(x, y).statistic)


def spearman(x: list[float], y: list[float]) -> float | None:
    if len(x) < 2:
        return None
    return float(stats.spearmanr(x, y).statistic)


def kendall(x: list[float], y: list[float]) -> float | None:
    if len(x) < 2:
        return None
    return float(stats.kendalltau(x, y).statistic)


def score_row(spec: SemanticMatch, swe: dict, deep: dict) -> dict:
    return {
        "canonical_model": spec.canonical_model,
        "swerebench_model": spec.swerebench_model,
        "swerebench_rank": swe["rank"],
        "swerebench_resolved_rate": swe["resolved_rate"],
        "swerebench_resolved_rate_sem": swe["resolved_rate_sem"],
        "swerebench_pass_at_5": swe["pass_at_5"],
        "deepswe_model": deep["model"],
        "deepswe_reasoning_effort": deep["reasoning_effort"] or "",
        "deepswe_pass_at_1": deep["pass_at_1"] * 100,
        "deepswe_ci_half": deep["ci_half"] * 100,
        "deepswe_n_passed": deep["n_passed"],
        "deepswe_n_attempted": deep["n_attempted"],
        "deepswe_mean_cost_usd": deep["mean_cost_usd"],
        "deepswe_mean_output_tokens": deep["mean_output_tokens"],
        "deepswe_mean_agent_steps": deep["mean_agent_steps"],
        "match_decision": spec.decision,
        "match_rationale": spec.rationale,
        "source_url_swerebench": SWE_REBENCH_URL,
        "source_url_deepswe": DEEPSWE_URL,
    }


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        raise RuntimeError(f"No rows to write: {path}")
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def build_summary(
    fetched_at: str,
    score_rows: list[dict],
    swe_meta: dict,
    deep_meta: dict,
    caveat: str,
) -> dict:
    x = [row["deepswe_pass_at_1"] for row in score_rows]
    y = [row["swerebench_resolved_rate"] for row in score_rows]
    return {
        "fetched_at": fetched_at,
        "n_matches": len(score_rows),
        "pearson_r": pearson(x, y),
        "spearman_rho": spearman(x, y),
        "kendall_tau": kendall(x, y),
        "sources": {
            "swerebench_v2": swe_meta,
            "deepswe": deep_meta,
        },
        "included_models": [
            {
                "canonical_model": row["canonical_model"],
                "match_decision": row["match_decision"],
                "deepswe_pass_at_1": row["deepswe_pass_at_1"],
                "swerebench_resolved_rate": row["swerebench_resolved_rate"],
            }
            for row in score_rows
        ],
        "caveat": caveat,
    }


def plot_scatter(
    score_rows: list[dict],
    figure_path: Path,
    title: str,
    note: str,
) -> None:
    x = [row["deepswe_pass_at_1"] for row in score_rows]
    y = [row["swerebench_resolved_rate"] for row in score_rows]
    fig, ax = plt.subplots(figsize=(9.0, 6.6))
    colors = {
        "strict_config_match": "#2563eb",
        "same_generation_config_uncertain": "#f59e0b",
        "same_generation_effort_mismatch": "#dc2626",
    }
    markers = {
        "strict_config_match": "o",
        "same_generation_config_uncertain": "s",
        "same_generation_effort_mismatch": "^",
    }
    labels = {
        "strict_config_match": "same model + same explicit effort",
        "same_generation_config_uncertain": "same model, SWE-rebench effort not shown",
        "same_generation_effort_mismatch": "same model, effort differs",
    }
    seen: set[str] = set()
    for row in score_rows:
        decision = row["match_decision"]
        ax.errorbar(
            [row["deepswe_pass_at_1"]],
            [row["swerebench_resolved_rate"]],
            xerr=[row["deepswe_ci_half"]],
            yerr=[row["swerebench_resolved_rate_sem"]],
            fmt=markers.get(decision, "o"),
            color=colors.get(decision, "#334155"),
            ecolor="#94a3b8",
            elinewidth=1.1,
            capsize=3,
            markersize=7.5,
            markeredgecolor="#0f172a",
            markeredgewidth=0.75,
            label=labels.get(decision, decision) if decision not in seen else None,
        )
        seen.add(decision)

    label_offsets = {
        "gpt-5.5[xhigh]": (8, -16, "left"),
        "gpt-5.5[medium]": (-8, 10, "right"),
        "claude-opus-4.8[xhigh]": (10, 8, "left"),
        "gpt-5.4[swerebench medium / deepswe xhigh]": (-12, -18, "right"),
        "claude-sonnet-4.6[deepswe high]": (10, 10, "left"),
        "gemini-3.1-pro-preview[deepswe high]": (10, 10, "left"),
        "gemini-3.5-flash[deepswe medium]": (10, -18, "left"),
    }
    for idx, row in enumerate(score_rows):
        label = row["canonical_model"]
        dx, dy, ha = label_offsets.get(label, (8 if idx % 2 == 0 else -8, 8 if idx % 3 else -16, "left"))
        ax.annotate(
            label,
            (row["deepswe_pass_at_1"], row["swerebench_resolved_rate"]),
            xytext=(dx, dy),
            textcoords="offset points",
            fontsize=8.5,
            ha=ha,
            annotation_clip=False,
            arrowprops={"arrowstyle": "-", "color": "#64748b", "lw": 0.65},
        )

    if len(score_rows) >= 2:
        coef = np.polyfit(np.array(x), np.array(y), 1)
        xx = np.linspace(max(0, min(x) - 5), min(100, max(x) + 5), 100)
        yy = coef[0] * xx + coef[1]
        ax.plot(xx, yy, linestyle="--", color="#64748b", linewidth=1.0, label=f"linear fit (n={len(score_rows)})")

    ax.set_title(title)
    ax.set_xlabel("DeepSWE Pass@1 (%)")
    ax.set_ylabel("SWE-rebench v2 Resolved Rate (%)")
    ax.grid(True, color="#e2e8f0", linewidth=0.8)
    ax.set_axisbelow(True)
    ax.set_xlim(max(0, min(x) - 8), min(100, max(x) + 8))
    ax.set_ylim(max(0, min(y) - 5), min(100, max(y) + 5))
    ax.legend(frameon=False, loc="best", fontsize=8.5)
    fig.text(0.01, 0.012, note, ha="left", va="bottom", fontsize=8, color="#475569")
    fig.tight_layout(rect=[0, 0.11, 1, 1])
    fig.savefig(figure_path, dpi=180)
    plt.close(fig)


def main() -> None:
    fetched_at = datetime.now(timezone.utc).isoformat()
    OUT_DATA.mkdir(parents=True, exist_ok=True)
    OUT_FIGURES.mkdir(parents=True, exist_ok=True)

    swe_rows, swe_meta = fetch_swerebench()
    deep_rows, deep_meta = fetch_deepswe()
    swe_by_model = {row["model"]: row for row in swe_rows}
    deep_by_key = {(row["model"], row["reasoning_effort"]): row for row in deep_rows}

    audit_rows = []
    strict_rows = []
    semantic_rows = []
    for spec in MATCH_AUDIT:
        swe = swe_by_model.get(spec.swerebench_model)
        deep = deep_by_key.get((spec.deepswe_model, spec.deepswe_effort))
        present = swe is not None and deep is not None
        audit_rows.append(
            {
                "canonical_model": spec.canonical_model,
                "swerebench_model": spec.swerebench_model,
                "deepswe_model": spec.deepswe_model,
                "deepswe_effort": spec.deepswe_effort or "",
                "decision": spec.decision,
                "include_in_strict": spec.include_in_strict and present,
                "include_in_semantic": spec.include_in_semantic and present,
                "present_in_sources": present,
                "rationale": spec.rationale,
            }
        )
        if present:
            row = score_row(spec, swe, deep)
            if spec.include_in_strict:
                strict_rows.append(row)
            if spec.include_in_semantic:
                semantic_rows.append(row)

    strict_score_path = OUT_DATA / "swerebench_v2_deepswe_exact_config_scores.csv"
    semantic_score_path = OUT_DATA / "swerebench_v2_deepswe_semantic_scores.csv"
    audit_path = OUT_DATA / "swerebench_v2_deepswe_match_audit.csv"
    strict_summary_path = OUT_DATA / "swerebench_v2_deepswe_exact_config_summary.json"
    semantic_summary_path = OUT_DATA / "swerebench_v2_deepswe_semantic_summary.json"
    strict_figure_path = OUT_FIGURES / "swerebench_v2_vs_deepswe_exact_config_scatter.png"
    semantic_figure_path = OUT_FIGURES / "swerebench_v2_vs_deepswe_semantic_scatter.png"

    write_csv(strict_score_path, strict_rows)
    write_csv(semantic_score_path, semantic_rows)
    write_csv(audit_path, audit_rows)

    strict_summary = build_summary(
        fetched_at,
        strict_rows,
        swe_meta,
        deep_meta,
        "Only strict same-generation plus same-explicit-effort matches are included.",
    )
    semantic_summary = build_summary(
        fetched_at,
        semantic_rows,
        swe_meta,
        deep_meta,
        "Same model generation matches are included. Rows where SWE-rebench effort is missing or differs from DeepSWE are marked in match_decision.",
    )
    strict_summary["n_exact_config_matches"] = len(strict_rows)
    semantic_summary["n_semantic_matches"] = len(semantic_rows)
    strict_summary_path.write_text(json.dumps(strict_summary, indent=2) + "\n")
    semantic_summary_path.write_text(json.dumps(semantic_summary, indent=2) + "\n")

    common_note = (
        "Excluded from main plot: cross-generation or variant changes such as GLM-5.1 vs 5.2 and Kimi K2.6 vs K2.7-code.\n"
        f"SWE-rebench v2 window: {swe_meta.get('date_from')} to {swe_meta.get('date_to')} "
        f"({swe_meta.get('problem_count')} tasks, {swe_meta.get('repo_count')} repos)."
    )
    plot_scatter(
        strict_rows,
        strict_figure_path,
        "SWE-rebench v2 vs DeepSWE\nstrict same-model and same-effort matches only",
        "Only rows with explicit matching effort are included.\n" + common_note,
    )
    plot_scatter(
        semantic_rows,
        semantic_figure_path,
        "SWE-rebench v2 vs DeepSWE\nsame model generation, effort caveats marked",
        "Colors distinguish exact effort, missing effort label, and effort mismatch.\n" + common_note,
    )

    print(f"Wrote {strict_score_path}")
    print(f"Wrote {semantic_score_path}")
    print(f"Wrote {audit_path}")
    print(f"Wrote {strict_summary_path}")
    print(f"Wrote {semantic_summary_path}")
    print(f"Wrote {strict_figure_path}")
    print(f"Wrote {semantic_figure_path}")


if __name__ == "__main__":
    main()
