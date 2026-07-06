#!/usr/bin/env python3
"""Build SWE-Bench Pro vs SWE-rebench correlation tables and plots.

The comparison has two subtly different time axes:

- Current SWE-Bench Pro leaderboard on Hugging Face vs the latest default
  SWE-rebench time window.
- SWE-Bench Pro paper Table 1, i.e. release-time public-set scores, vs the
  September 2025 SWE-rebench window around the benchmark release.

SWE-rebench is a date-windowed leaderboard. A model is considered active for a
window only when the selected task-date window is fully covered by that model's
evaluation range, matching the page's N/A behavior.
"""

from __future__ import annotations

import csv
import html
import json
import re
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import requests
from scipy import stats


SWE_REBENCH_URL = "https://swe-rebench.com/"
SWE_BENCH_PRO_HF_URL = "https://huggingface.co/datasets/ScaleAI/SWE-bench_Pro"
SWE_BENCH_PRO_PAPER_URL = "https://arxiv.org/html/2509.16941"

OUT_DATA = Path("data/analysis")
OUT_FIGURES = Path("docs/figures")


@dataclass(frozen=True)
class MatchSpec:
    pro_model_id: str
    pro_display_name: str
    swerebench_model: str | None
    match_kind: str
    include_in_exact: bool
    rationale: str


@dataclass(frozen=True)
class ReleaseScore:
    pro_display_name: str
    pro_score: float
    swerebench_model: str | None
    match_kind: str
    include_in_exact: bool
    rationale: str


CURRENT_PRO_MATCHES: list[MatchSpec] = [
    MatchSpec(
        "zai-org/GLM-5.2",
        "GLM-5.2",
        None,
        "different_generation_available_only",
        False,
        "SWE-rebench latest has GLM-5.1, not GLM-5.2",
    ),
    MatchSpec("MiniMaxAI/MiniMax-M3", "MiniMax M3", "MiniMax M3", "exact_model", True, "same model label"),
    MatchSpec("moonshotai/Kimi-K2.6", "Kimi K2.6", "Kimi K2.6", "exact_model", True, "same model label"),
    MatchSpec("zai-org/GLM-5.1", "GLM-5.1", "GLM-5.1", "exact_model", True, "same model label"),
    MatchSpec(
        "XiaomiMiMo/MiMo-V2.5-Pro",
        "MiMo-V2.5-Pro",
        None,
        "no_swerebench_match",
        False,
        "no matching SWE-rebench model label",
    ),
    MatchSpec(
        "stepfun-ai/Step-3.7-Flash",
        "Step-3.7-Flash",
        None,
        "different_generation_available_only",
        False,
        "SWE-rebench has Step-3.5-Flash, not Step-3.7-Flash",
    ),
    MatchSpec("MiniMaxAI/MiniMax-M2.7", "MiniMax M2.7", "MiniMax M2.7", "exact_model", True, "same model label"),
    MatchSpec(
        "XiaomiMiMo/MiMo-V2.5",
        "MiMo-V2.5",
        None,
        "no_swerebench_match",
        False,
        "no matching SWE-rebench model label",
    ),
    MatchSpec("MiniMaxAI/MiniMax-M2.5", "MiniMax M2.5", "MiniMax M2.5", "exact_model", True, "same model label"),
    MatchSpec(
        "deepseek-ai/DeepSeek-V4-Pro",
        "DeepSeek-V4-Pro",
        None,
        "different_generation_available_only",
        False,
        "SWE-rebench has DeepSeek-V3.2/V3.x, not DeepSeek-V4-Pro",
    ),
    MatchSpec(
        "Qwen/Qwen3.6-27B",
        "Qwen3.6-27B",
        None,
        "different_generation_available_only",
        False,
        "SWE-rebench has Qwen3.5 rows, not Qwen3.6",
    ),
    MatchSpec("moonshotai/Kimi-K2.5", "Kimi K2.5", "Kimi K2.5", "exact_model", True, "same model label"),
    MatchSpec(
        "Qwen/Qwen3.6-35B-A3B",
        "Qwen3.6-35B-A3B",
        None,
        "different_generation_available_only",
        False,
        "SWE-rebench has Qwen3.5 rows, not Qwen3.6",
    ),
    MatchSpec(
        "poolside/Laguna-M.1",
        "Laguna-M.1",
        None,
        "no_swerebench_match",
        False,
        "no matching SWE-rebench model label",
    ),
    MatchSpec(
        "poolside/Laguna-XS.2",
        "Laguna-XS.2",
        None,
        "no_swerebench_match",
        False,
        "no matching SWE-rebench model label",
    ),
    MatchSpec(
        "MuVeraAI/Laguna-XS.2",
        "Laguna-XS.2",
        None,
        "no_swerebench_match",
        False,
        "no matching SWE-rebench model label",
    ),
    MatchSpec("Qwen/Qwen3-Coder-Next", "Qwen3-Coder-Next", "Qwen3-Coder-Next", "exact_model", True, "same model label"),
    MatchSpec(
        "CohereLabs/North-Mini-Code-1.0",
        "North-Mini-Code-1.0",
        None,
        "no_swerebench_match",
        False,
        "no matching SWE-rebench model label",
    ),
    MatchSpec(
        "Qwen/Qwen3-Coder-480B-A35B-Instruct",
        "Qwen3-Coder-480B-A35B-Instruct",
        "Qwen3-Coder-480B-A35B-Instruct",
        "exact_model",
        True,
        "same model label",
    ),
    MatchSpec("MiniMaxAI/MiniMax-M2.1", "MiniMax M2.1", "MiniMax M2.1", "exact_model", True, "same model label"),
    MatchSpec(
        "moonshotai/Kimi-K2-Instruct",
        "Kimi K2 Instruct",
        "Kimi K2 Instruct 0905",
        "same_model_snapshot_label",
        True,
        "SWE-rebench exposes the 0905 snapshot label; SWE-Bench Pro paper was run as of 2025-09-18",
    ),
    MatchSpec("Qwen/Qwen3-235B-A22B", "Qwen3-235B-A22B", "Qwen3-235B-A22B", "exact_model", True, "same model label"),
    MatchSpec("openai/gpt-oss-120b", "gpt-oss-120b", "gpt-oss-120b", "exact_model", True, "same model label"),
    MatchSpec("zai-org/GLM-4.6", "GLM-4.6", "GLM-4.6", "exact_model", True, "same model label"),
]


RELEASE_TABLE1_SCORES: list[ReleaseScore] = [
    ReleaseScore("Claude Sonnet 4.5", 43.6, "Claude Sonnet 4.5", "exact_model", True, "paper Table 1 model label"),
    ReleaseScore("Claude Sonnet 4", 42.7, "Claude Sonnet 4", "exact_model", True, "paper Table 1 model label"),
    ReleaseScore("OpenAI GPT-5 high", 41.8, "gpt-5-2025-08-07-high", "exact_model_effort", True, "GPT-5 high effort"),
    ReleaseScore(
        "Claude Haiku 4.5",
        39.5,
        None,
        "no_swerebench_match",
        False,
        "not present in SWE-rebench state",
    ),
    ReleaseScore(
        "Kimi K2 Instruct",
        27.7,
        "Kimi K2 Instruct 0905",
        "same_model_snapshot_label",
        True,
        "SWE-rebench exposes the 0905 snapshot label",
    ),
    ReleaseScore("OpenAI GPT-OSS 120B", 16.2, "gpt-oss-120b", "exact_model", True, "same model label"),
]


def fetch_text(url: str) -> str:
    response = requests.get(url, timeout=45)
    response.raise_for_status()
    return response.text


def extract_next_flight_text(page_html: str) -> str:
    chunks = []
    for match in re.finditer(r"self\.__next_f\.push\((.*?)\)</script>", page_html, re.S):
        chunks.append(json.loads(match.group(1)))
    return "".join(chunk[1] for chunk in chunks if len(chunk) > 1 and isinstance(chunk[1], str))


def extract_balanced_json(text: str, start_pattern: str) -> dict[str, Any]:
    start = text.find(start_pattern)
    if start < 0:
        raise RuntimeError(f"Could not find JSON object starting with {start_pattern!r}")
    depth = 0
    in_string = False
    escaped = False
    for idx, char in enumerate(text[start:], start):
        if in_string:
            if escaped:
                escaped = False
            elif char == "\\":
                escaped = True
            elif char == '"':
                in_string = False
            continue
        if char == '"':
            in_string = True
        elif char == "{":
            depth += 1
        elif char == "}":
            depth -= 1
            if depth == 0:
                return json.loads(text[start : idx + 1])
    raise RuntimeError("Unterminated JSON object")


def load_swerebench_state() -> dict[str, Any]:
    page_html = fetch_text(SWE_REBENCH_URL)
    flight_text = extract_next_flight_text(page_html)
    return extract_balanced_json(flight_text, '{"problems":')


def load_swebench_pro_hf_rows() -> list[dict[str, Any]]:
    page_html = html.unescape(fetch_text(SWE_BENCH_PRO_HF_URL))
    rows = []
    seen = set()
    for match in re.finditer(r'\{"rank":\d+,', page_html):
        obj = extract_json_object_at(page_html, match.start())
        key = (obj["rank"], obj["modelId"], obj["value"])
        if key in seen:
            continue
        seen.add(key)
        rows.append(obj)
    return sorted(rows, key=lambda row: row["rank"])


def extract_json_object_at(text: str, start: int) -> dict[str, Any]:
    depth = 0
    in_string = False
    escaped = False
    for idx, char in enumerate(text[start:], start):
        if in_string:
            if escaped:
                escaped = False
            elif char == "\\":
                escaped = True
            elif char == '"':
                in_string = False
            continue
        if char == '"':
            in_string = True
        elif char == "{":
            depth += 1
        elif char == "}":
            depth -= 1
            if depth == 0:
                return json.loads(text[start : idx + 1])
    raise RuntimeError("Unterminated leaderboard JSON object")


def fmt_date(timestamp_ms: int) -> str:
    return datetime.fromtimestamp(timestamp_ms / 1000, timezone.utc).date().isoformat()


def window_label(dates: list[int], start_index: int, end_index: int) -> str:
    return f"{fmt_date(dates[start_index])}..{fmt_date(dates[end_index])}"


def item_map(state: dict[str, Any]) -> dict[str, list[dict[str, Any]]]:
    mapping: dict[str, list[dict[str, Any]]] = {}
    for item in state["items"]:
        mapping.setdefault(item["modelName"], []).append(item)
    return mapping


def choose_active_item(
    candidates: list[dict[str, Any]],
    start_ts: int,
    end_ts: int,
) -> tuple[dict[str, Any] | None, str]:
    active = [
        item
        for item in candidates
        if item["taskRangeTimestamp"]["from"] <= start_ts and item["taskRangeTimestamp"]["to"] >= end_ts
    ]
    if not active:
        return None, "inactive_for_window"
    return max(active, key=lambda item: item["taskRangeTimestamp"]["to"]), "active"


def choose_latest_available_item(candidates: list[dict[str, Any]]) -> dict[str, Any] | None:
    if not candidates:
        return None
    return max(candidates, key=lambda item: (item["taskRangeTimestamp"]["to"], item["taskRangeTimestamp"]["from"]))


def score_for_window(item: dict[str, Any], start_ts: int, end_ts: int) -> dict[str, Any] | None:
    return item["rangeStats"].get(f"{start_ts}:{end_ts}")


def score_for_item_full_range(item: dict[str, Any]) -> tuple[dict[str, Any] | None, int, int]:
    start_ts = item["taskRangeTimestamp"]["from"]
    end_ts = item["taskRangeTimestamp"]["to"]
    return score_for_window(item, start_ts, end_ts), start_ts, end_ts


def correlations(rows: list[dict[str, Any]]) -> dict[str, float | None]:
    if len(rows) < 2:
        return {"pearson": None, "spearman": None, "kendall": None}
    x = [float(row["swebench_pro_score"]) for row in rows]
    y = [float(row["swerebench_resolved_rate"]) for row in rows]
    return {
        "pearson": float(stats.pearsonr(x, y).statistic),
        "spearman": float(stats.spearmanr(x, y).statistic),
        "kendall": float(stats.kendalltau(x, y).statistic),
    }


def build_current_pro_window_rows(
    pro_rows: list[dict[str, Any]],
    state: dict[str, Any],
    start_index: int,
    end_index: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    dates = state["dates"]["dates"]
    start_ts = dates[start_index]
    end_ts = dates[end_index]
    by_id = {row["modelId"]: row for row in pro_rows}
    by_name = item_map(state)
    rows = []
    audit = []
    for spec in CURRENT_PRO_MATCHES:
        pro_row = by_id.get(spec.pro_model_id)
        if pro_row is None:
            audit.append(audit_row(spec, None, "missing_from_hf_leaderboard", None, None, None))
            continue
        if not spec.swerebench_model:
            audit.append(audit_row(spec, pro_row, spec.match_kind, None, None, None))
            continue
        candidates = by_name.get(spec.swerebench_model, [])
        item, status = choose_active_item(candidates, start_ts, end_ts)
        if item is None:
            audit.append(audit_row(spec, pro_row, status, None, start_ts, end_ts))
            continue
        stat = score_for_window(item, start_ts, end_ts)
        if stat is None or stat.get("resolvedRate") is None:
            audit.append(audit_row(spec, pro_row, "no_score_for_window", item, start_ts, end_ts))
            continue
        row = paired_row(
            pro_display_name=spec.pro_display_name,
            pro_model_id=spec.pro_model_id,
            pro_score=pro_row["value"],
            swerebench_model=item["modelName"],
            swerebench_agent_version=item.get("agentVersion"),
            swerebench_stat=stat,
            swerebench_window_from=start_ts,
            swerebench_window_to=end_ts,
            match_kind=spec.match_kind,
            rationale=spec.rationale,
        )
        rows.append(row)
        audit.append(audit_row(spec, pro_row, "included", item, start_ts, end_ts))
    return rows, audit


def build_current_pro_model_latest_rows(
    pro_rows: list[dict[str, Any]],
    state: dict[str, Any],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    by_id = {row["modelId"]: row for row in pro_rows}
    by_name = item_map(state)
    rows = []
    audit = []
    for spec in CURRENT_PRO_MATCHES:
        pro_row = by_id.get(spec.pro_model_id)
        if pro_row is None:
            audit.append(audit_row(spec, None, "missing_from_hf_leaderboard", None, None, None, mode="model_latest"))
            continue
        if not spec.swerebench_model:
            audit.append(audit_row(spec, pro_row, spec.match_kind, None, None, None, mode="model_latest"))
            continue
        item = choose_latest_available_item(by_name.get(spec.swerebench_model, []))
        if item is None:
            audit.append(audit_row(spec, pro_row, "no_swerebench_match", None, None, None, mode="model_latest"))
            continue
        stat, start_ts, end_ts = score_for_item_full_range(item)
        if stat is None or stat.get("resolvedRate") is None:
            audit.append(audit_row(spec, pro_row, "no_score_for_model_latest_range", item, start_ts, end_ts, mode="model_latest"))
            continue
        rows.append(
            paired_row(
                pro_display_name=spec.pro_display_name,
                pro_model_id=spec.pro_model_id,
                pro_score=pro_row["value"],
                swerebench_model=item["modelName"],
                swerebench_agent_version=item.get("agentVersion"),
                swerebench_stat=stat,
                swerebench_window_from=start_ts,
                swerebench_window_to=end_ts,
                match_kind=spec.match_kind,
                rationale=spec.rationale,
            )
        )
        audit.append(audit_row(spec, pro_row, "included", item, start_ts, end_ts, mode="model_latest"))
    return rows, audit


def build_release_table_rows(
    state: dict[str, Any],
    start_index: int,
    end_index: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    dates = state["dates"]["dates"]
    start_ts = dates[start_index]
    end_ts = dates[end_index]
    by_name = item_map(state)
    rows = []
    audit = []
    for spec in RELEASE_TABLE1_SCORES:
        if not spec.swerebench_model:
            audit.append(release_audit_row(spec, "no_swerebench_match", None, None, None))
            continue
        item, status = choose_active_item(by_name.get(spec.swerebench_model, []), start_ts, end_ts)
        if item is None:
            audit.append(release_audit_row(spec, status, None, start_ts, end_ts))
            continue
        stat = score_for_window(item, start_ts, end_ts)
        if stat is None or stat.get("resolvedRate") is None:
            audit.append(release_audit_row(spec, "no_score_for_window", item, start_ts, end_ts))
            continue
        rows.append(
            paired_row(
                pro_display_name=spec.pro_display_name,
                pro_model_id=spec.pro_display_name,
                pro_score=spec.pro_score,
                swerebench_model=item["modelName"],
                swerebench_agent_version=item.get("agentVersion"),
                swerebench_stat=stat,
                swerebench_window_from=start_ts,
                swerebench_window_to=end_ts,
                match_kind=spec.match_kind,
                rationale=spec.rationale,
            )
        )
        audit.append(release_audit_row(spec, "included", item, start_ts, end_ts))
    return rows, audit


def paired_row(
    *,
    pro_display_name: str,
    pro_model_id: str,
    pro_score: float,
    swerebench_model: str,
    swerebench_agent_version: str | None,
    swerebench_stat: dict[str, Any],
    swerebench_window_from: int,
    swerebench_window_to: int,
    match_kind: str,
    rationale: str,
) -> dict[str, Any]:
    return {
        "model": pro_display_name,
        "swebench_pro_model_id": pro_model_id,
        "swebench_pro_score": float(pro_score),
        "swerebench_model": swerebench_model,
        "swerebench_agent_version": swerebench_agent_version,
        "swerebench_window_from": fmt_date(swerebench_window_from),
        "swerebench_window_to": fmt_date(swerebench_window_to),
        "swerebench_resolved_rate": float(swerebench_stat["resolvedRate"]),
        "swerebench_resolved_rate_sem": swerebench_stat.get("sem"),
        "swerebench_pass_at_5": swerebench_stat.get("passN"),
        "match_kind": match_kind,
        "rationale": rationale,
    }


def audit_row(
    spec: MatchSpec,
    pro_row: dict[str, Any] | None,
    decision: str,
    item: dict[str, Any] | None,
    start_ts: int | None,
    end_ts: int | None,
    mode: str = "fixed_window",
) -> dict[str, Any]:
    task_range = item["taskRangeTimestamp"] if item else None
    return {
        "source": "hf_current",
        "mode": mode,
        "swebench_pro_model_id": spec.pro_model_id,
        "swebench_pro_model": spec.pro_display_name,
        "swebench_pro_score": pro_row.get("value") if pro_row else None,
        "swerebench_model": spec.swerebench_model,
        "match_kind": spec.match_kind,
        "decision": decision,
        "requested_window_from": fmt_date(start_ts) if start_ts else None,
        "requested_window_to": fmt_date(end_ts) if end_ts else None,
        "swerebench_task_range_from": fmt_date(task_range["from"]) if task_range else None,
        "swerebench_task_range_to": fmt_date(task_range["to"]) if task_range else None,
        "rationale": spec.rationale,
    }


def release_audit_row(
    spec: ReleaseScore,
    decision: str,
    item: dict[str, Any] | None,
    start_ts: int | None,
    end_ts: int | None,
) -> dict[str, Any]:
    task_range = item["taskRangeTimestamp"] if item else None
    return {
        "source": "paper_table1_release",
        "mode": "fixed_window",
        "swebench_pro_model_id": spec.pro_display_name,
        "swebench_pro_model": spec.pro_display_name,
        "swebench_pro_score": spec.pro_score,
        "swerebench_model": spec.swerebench_model,
        "match_kind": spec.match_kind,
        "decision": decision,
        "requested_window_from": fmt_date(start_ts) if start_ts else None,
        "requested_window_to": fmt_date(end_ts) if end_ts else None,
        "swerebench_task_range_from": fmt_date(task_range["from"]) if task_range else None,
        "swerebench_task_range_to": fmt_date(task_range["to"]) if task_range else None,
        "rationale": spec.rationale,
    }


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def plot_scatter(path: Path, rows: list[dict[str, Any]], title: str, subtitle: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(8.4, 6.2), dpi=180)
    if rows:
        x = [float(row["swebench_pro_score"]) for row in rows]
        y = [float(row["swerebench_resolved_rate"]) for row in rows]
        ax.scatter(x, y, s=58, color="#2563eb", edgecolor="#0f172a", linewidth=0.6)
        for row in rows:
            ax.annotate(
                short_label(row["model"]),
                (float(row["swebench_pro_score"]), float(row["swerebench_resolved_rate"])),
                xytext=(5, 4),
                textcoords="offset points",
                fontsize=7.2,
            )
        if len(rows) >= 2:
            min_x, max_x = min(x), max(x)
            if min_x != max_x:
                slope, intercept, *_ = stats.linregress(x, y)
                xs = [min_x, max_x]
                ys = [intercept + slope * value for value in xs]
                ax.plot(xs, ys, color="#ef4444", linewidth=1.4, alpha=0.75)
    corr = correlations(rows)
    corr_text = "n={n}".format(n=len(rows))
    if corr["pearson"] is not None:
        corr_text += "  Pearson r={:.3f}  Spearman rho={:.3f}".format(corr["pearson"], corr["spearman"])
    fig.suptitle(title, fontsize=12.5, y=0.985)
    ax.set_title(subtitle + "\n" + corr_text, fontsize=8.5, loc="left", pad=10)
    ax.set_xlabel("SWE-Bench Pro resolve (%)")
    ax.set_ylabel("SWE-rebench resolved rate (%)")
    ax.grid(True, color="#e5e7eb", linewidth=0.8)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.savefig(path)
    plt.close(fig)


def short_label(label: str) -> str:
    replacements = {
        "OpenAI ": "",
        "Claude ": "",
        "Qwen3-Coder-480B-A35B-Instruct": "Qwen3 Coder 480B",
        "Qwen3-Coder-Next": "Qwen3 Coder Next",
        "Kimi K2 Instruct": "Kimi K2",
    }
    for old, new in replacements.items():
        label = label.replace(old, new)
    return label


def main() -> None:
    OUT_DATA.mkdir(parents=True, exist_ok=True)
    OUT_FIGURES.mkdir(parents=True, exist_ok=True)

    state = load_swerebench_state()
    pro_rows = load_swebench_pro_hf_rows()
    dates = state["dates"]["dates"]
    latest_start = state["dates"]["initialStartIndex"]
    latest_end = state["dates"]["initialEndIndex"]

    current_latest_rows, current_latest_audit = build_current_pro_window_rows(
        pro_rows,
        state,
        latest_start,
        latest_end,
    )
    current_sep_rows, current_sep_audit = build_current_pro_window_rows(pro_rows, state, 16, 18)
    current_model_latest_rows, current_model_latest_audit = build_current_pro_model_latest_rows(pro_rows, state)
    release_sep_rows, release_sep_audit = build_release_table_rows(state, 16, 18)
    release_post_rows, release_post_audit = build_release_table_rows(state, 17, 18)

    outputs = {
        "current_hf_vs_latest_default": {
            "rows": current_latest_rows,
            "csv": OUT_DATA / "swebench_pro_current_vs_swerebench_latest_default_scores.csv",
            "figure": OUT_FIGURES / "swebench_pro_current_vs_swerebench_latest_default_scatter.png",
            "title": "Current SWE-Bench Pro vs Latest SWE-rebench",
            "subtitle": f"SWE-rebench default window {window_label(dates, latest_start, latest_end)}",
        },
        "current_hf_vs_sep_2025": {
            "rows": current_sep_rows,
            "csv": OUT_DATA / "swebench_pro_current_vs_swerebench_sep_2025_scores.csv",
            "figure": OUT_FIGURES / "swebench_pro_current_vs_swerebench_sep_2025_scatter.png",
            "title": "Current SWE-Bench Pro vs September 2025 SWE-rebench",
            "subtitle": f"SWE-rebench window {window_label(dates, 16, 18)}",
        },
        "current_hf_vs_model_latest_available": {
            "rows": current_model_latest_rows,
            "csv": OUT_DATA / "swebench_pro_current_vs_swerebench_model_latest_scores.csv",
            "figure": OUT_FIGURES / "swebench_pro_current_vs_swerebench_model_latest_scatter.png",
            "title": "Current SWE-Bench Pro vs Model-Latest SWE-rebench",
            "subtitle": "Each model uses its own latest active SWE-rebench task window",
        },
        "release_table1_vs_sep_2025": {
            "rows": release_sep_rows,
            "csv": OUT_DATA / "swebench_pro_release_vs_swerebench_sep_2025_scores.csv",
            "figure": OUT_FIGURES / "swebench_pro_release_vs_swerebench_sep_2025_scatter.png",
            "title": "Release SWE-Bench Pro vs September 2025 SWE-rebench",
            "subtitle": f"SWE-rebench window {window_label(dates, 16, 18)}",
        },
        "release_table1_vs_post_release": {
            "rows": release_post_rows,
            "csv": OUT_DATA / "swebench_pro_release_vs_swerebench_post_release_scores.csv",
            "figure": OUT_FIGURES / "swebench_pro_release_vs_swerebench_post_release_scatter.png",
            "title": "Release SWE-Bench Pro vs Post-Release SWE-rebench",
            "subtitle": f"SWE-rebench window {window_label(dates, 17, 18)}",
        },
    }

    summary: dict[str, Any] = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "sources": {
            "swe_rebench": SWE_REBENCH_URL,
            "swebench_pro_hf": SWE_BENCH_PRO_HF_URL,
            "swebench_pro_paper": SWE_BENCH_PRO_PAPER_URL,
        },
        "swe_rebench_latest_default_window": {
            "from": fmt_date(dates[latest_start]),
            "to": fmt_date(dates[latest_end]),
            "start_index": latest_start,
            "end_index": latest_end,
        },
        "comparisons": {},
    }
    for key, payload in outputs.items():
        rows = payload["rows"]
        write_csv(payload["csv"], rows)
        plot_scatter(payload["figure"], rows, payload["title"], payload["subtitle"])
        summary["comparisons"][key] = {
            "n": len(rows),
            **correlations(rows),
            "csv": str(payload["csv"]),
            "figure": str(payload["figure"]),
        }

    audit_rows = (
        current_latest_audit
        + current_sep_audit
        + current_model_latest_audit
        + release_sep_audit
        + release_post_audit
    )
    write_csv(OUT_DATA / "swebench_pro_swerebench_match_audit.csv", audit_rows)
    summary["match_audit_csv"] = str(OUT_DATA / "swebench_pro_swerebench_match_audit.csv")
    (OUT_DATA / "swebench_pro_swerebench_correlation_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )

    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
