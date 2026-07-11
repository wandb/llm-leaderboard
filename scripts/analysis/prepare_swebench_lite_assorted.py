#!/usr/bin/env python3
"""Build SWE-bench Lite Low/Middle slices for Agentic SWE-Assorted.

The legacy slices are selected from static task features only. The v2 slices add
public SWE-bench Lite per-instance outcomes as a difficulty prior plus local
pilot-run cost-risk signals. Public outcomes are used only for tier calibration,
not for scoring any model.
"""

from __future__ import annotations

import argparse
import ast
import csv
import json
import math
import re
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import requests


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUTPUT_DIR = REPO_ROOT / "data" / "taiwan" / "swebench_lite_assorted"
DATASET = "princeton-nlp/SWE-bench_Lite"
CONFIG = "default"
SPLIT = "test"
SOURCE_BENCHMARK = "SWE-bench Lite"
ROWS_URL = "https://datasets-server.huggingface.co/rows"
SWEBENCH_EXPERIMENTS_REPO = "SWE-bench/experiments"
SWEBENCH_EXPERIMENTS_BRANCH = "main"
PUBLIC_LITE_SUBMISSIONS_URL = (
    "https://api.github.com/repos/"
    f"{SWEBENCH_EXPERIMENTS_REPO}/contents/evaluation/lite"
)
PUBLIC_LITE_RESULTS_URL = (
    "https://raw.githubusercontent.com/"
    f"{SWEBENCH_EXPERIMENTS_REPO}/{SWEBENCH_EXPERIMENTS_BRANCH}"
    "/evaluation/lite/{submission}/results/results.json"
)
DEFAULT_PILOT_OUTPUT_TABLE = (
    REPO_ROOT
    / "outputs"
    / "agentic_swe_assorted_runs"
    / "glm52_lm12m12_20260712_002247"
    / "output_table.jsonl"
)
DEFAULT_PILOT_SUMMARY = (
    DEFAULT_OUTPUT_DIR / "source" / "glm52_lm12m12_20260712_pilot_summary.json"
)
V2_SELECTION_VERSION = "v2_public_lite_prior_glm52_pilot_20260712"


DIFF_FILE_RE = re.compile(r"^diff --git a/(.*?) b/(.*?)$", re.MULTILINE)
DIFF_HUNK_RE = re.compile(r"^@@", re.MULTILINE)


def read_json_list(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, list):
        return [str(item) for item in value]
    text = str(value).strip()
    if not text:
        return []
    for loader in (json.loads, ast.literal_eval):
        try:
            parsed = loader(text)
        except Exception:
            continue
        if isinstance(parsed, list):
            return [str(item) for item in parsed]
    return [text]


def diff_stats(diff_text: str) -> dict[str, int]:
    files = DIFF_FILE_RE.findall(diff_text or "")
    hunks = len(DIFF_HUNK_RE.findall(diff_text or ""))
    added = 0
    removed = 0
    for line in (diff_text or "").splitlines():
        if line.startswith("+++") or line.startswith("---"):
            continue
        if line.startswith("+"):
            added += 1
        elif line.startswith("-"):
            removed += 1
    return {
        "files": len({new or old for old, new in files}),
        "hunks": hunks,
        "added_lines": added,
        "removed_lines": removed,
        "changed_lines": added + removed,
    }


def selected_test_paths(*items: str) -> list[str]:
    paths: set[str] = set()
    for item in items:
        path = str(item).split("::", 1)[0].strip().lstrip("./")
        if path:
            paths.add(path)
    return sorted(paths)


def static_difficulty(row: dict[str, Any]) -> dict[str, Any]:
    patch = diff_stats(str(row.get("patch") or ""))
    test_patch = diff_stats(str(row.get("test_patch") or ""))
    fail_to_pass = read_json_list(row.get("FAIL_TO_PASS"))
    pass_to_pass = read_json_list(row.get("PASS_TO_PASS"))
    problem_words = len(str(row.get("problem_statement") or "").split())
    problem_chars = len(str(row.get("problem_statement") or ""))
    score = (
        2.0 * math.log1p(patch["changed_lines"])
        + 1.7 * patch["files"]
        + 1.2 * patch["hunks"]
        + 0.65 * math.log1p(test_patch["changed_lines"])
        + 0.35 * math.log1p(problem_words)
        + 0.25 * len(fail_to_pass)
        + 0.035 * len(pass_to_pass)
    )
    return {
        "static_difficulty_score": round(score, 6),
        "gold_patch_files": patch["files"],
        "gold_patch_hunks": patch["hunks"],
        "gold_patch_changed_lines": patch["changed_lines"],
        "gold_patch_added_lines": patch["added_lines"],
        "gold_patch_removed_lines": patch["removed_lines"],
        "test_patch_files": test_patch["files"],
        "test_patch_hunks": test_patch["hunks"],
        "test_patch_changed_lines": test_patch["changed_lines"],
        "fail_to_pass_count": len(fail_to_pass),
        "pass_to_pass_count": len(pass_to_pass),
        "problem_words": problem_words,
        "problem_chars": problem_chars,
        "selected_test_files_to_run": selected_test_paths(*(fail_to_pass + pass_to_pass)),
    }


def fetch_rows(*, refresh_cache: bool, cache_path: Path) -> list[dict[str, Any]]:
    if cache_path.exists() and not refresh_cache:
        return [
            json.loads(line)
            for line in cache_path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
    rows: list[dict[str, Any]] = []
    offset = 0
    while True:
        response = requests.get(
            ROWS_URL,
            params={
                "dataset": DATASET,
                "config": CONFIG,
                "split": SPLIT,
                "offset": offset,
                "length": 100,
            },
            timeout=120,
        )
        response.raise_for_status()
        payload = response.json()
        page = [item["row"] for item in payload.get("rows", [])]
        rows.extend(page)
        total = int(payload.get("num_rows_total") or len(rows))
        offset += len(page)
        if not page or offset >= total:
            break
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    with cache_path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    return rows


def result_instance_ids(payload: dict[str, Any]) -> set[str]:
    ids: set[str] = set()
    for value in payload.values():
        if not isinstance(value, list):
            continue
        ids.update(str(item) for item in value if isinstance(item, str))
    return ids


def fetch_public_lite_results(*, refresh_cache: bool, cache_path: Path) -> dict[str, Any]:
    """Fetch public SWE-bench Lite per-instance outcomes from SWE-bench/experiments."""

    if cache_path.exists() and not refresh_cache:
        return json.loads(cache_path.read_text(encoding="utf-8"))

    response = requests.get(
        PUBLIC_LITE_SUBMISSIONS_URL,
        params={"ref": SWEBENCH_EXPERIMENTS_BRANCH},
        timeout=120,
    )
    response.raise_for_status()
    submissions = [
        item["name"]
        for item in response.json()
        if item.get("type") == "dir" and isinstance(item.get("name"), str)
    ]

    counts: dict[str, dict[str, int]] = {}
    fetched_submissions: list[str] = []
    skipped_submissions: list[dict[str, str]] = []
    for submission in submissions:
        url = PUBLIC_LITE_RESULTS_URL.format(submission=submission)
        try:
            result_response = requests.get(url, timeout=120)
            result_response.raise_for_status()
            payload = result_response.json()
        except Exception as exc:
            skipped_submissions.append({"submission": submission, "error": str(exc)})
            continue
        fetched_submissions.append(submission)
        seen_ids = result_instance_ids(payload)
        resolved_ids = {
            str(item)
            for item in payload.get("resolved", [])
            if isinstance(item, str)
        }
        for instance_id in seen_ids:
            entry = counts.setdefault(
                instance_id,
                {"public_lite_seen_count": 0, "public_lite_resolved_count": 0},
            )
            entry["public_lite_seen_count"] += 1
            if instance_id in resolved_ids:
                entry["public_lite_resolved_count"] += 1

    instances = {}
    for instance_id, entry in sorted(counts.items()):
        seen = entry["public_lite_seen_count"]
        resolved = entry["public_lite_resolved_count"]
        instances[instance_id] = {
            **entry,
            "public_lite_resolve_rate": round(resolved / seen, 6) if seen else None,
        }

    payload = {
        "source_repo": SWEBENCH_EXPERIMENTS_REPO,
        "source_branch": SWEBENCH_EXPERIMENTS_BRANCH,
        "source_split": "evaluation/lite",
        "fetched_at": datetime.now(timezone.utc).isoformat(),
        "submission_count": len(fetched_submissions),
        "submissions": fetched_submissions,
        "skipped_submissions": skipped_submissions,
        "instances": instances,
    }
    write_json(cache_path, payload)
    return payload


def pilot_row_summary(row: dict[str, Any]) -> dict[str, Any]:
    runtime_budget = row.get("runtime_budget") if isinstance(row.get("runtime_budget"), dict) else {}
    violations = runtime_budget.get("violations") if isinstance(runtime_budget.get("violations"), list) else []
    disqualified_reason = str(row.get("openclaw_disqualified_reason") or "")
    runtime_budget_exceeded = disqualified_reason == "runtime_budget_exceeded" or any(
        isinstance(item, dict) and "budget" in str(item.get("type", ""))
        for item in violations
    )
    return {
        "resolved": row.get("resolved") is True,
        "patch_empty": row.get("patch_empty") is True,
        "openclaw_disqualified_reason": disqualified_reason,
        "runtime_budget_exceeded": runtime_budget_exceeded,
        "provider_transient_exhausted": disqualified_reason == "provider_transient_exhausted",
        "tool_policy_violation": disqualified_reason == "tool_policy_violation",
        "openclaw_tool_call_count": row.get("openclaw_tool_call_count"),
        "openclaw_tool_error_count": row.get("openclaw_tool_error_count"),
        "weave_agents_ok": row.get("weave_agents_ok"),
    }


def build_pilot_summary_from_output_table(path: Path, *, run_label: str) -> dict[str, Any]:
    rows = [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    return {
        "run_label": run_label,
        "source_output_table": str(path),
        "total_instances": len(rows),
        "instances": {
            str(row["instance_id"]): pilot_row_summary(row)
            for row in rows
            if row.get("instance_id")
        },
    }


def load_or_build_pilot_summary(
    *,
    summary_path: Path | None,
    output_table_path: Path | None,
    run_label: str,
) -> dict[str, Any]:
    if summary_path is not None and summary_path.exists():
        return json.loads(summary_path.read_text(encoding="utf-8"))
    if output_table_path is not None and output_table_path.exists():
        payload = build_pilot_summary_from_output_table(output_table_path, run_label=run_label)
        if summary_path is not None:
            write_json(summary_path, payload)
        return payload
    return {"run_label": run_label, "total_instances": 0, "instances": {}}


def public_metrics(public_results: dict[str, Any], instance_id: str) -> dict[str, Any]:
    entry = (public_results.get("instances") or {}).get(instance_id) or {}
    seen = int(entry.get("public_lite_seen_count") or 0)
    resolved = int(entry.get("public_lite_resolved_count") or 0)
    rate = entry.get("public_lite_resolve_rate")
    if rate is None and seen:
        rate = resolved / seen
    return {
        "public_lite_seen_count": seen,
        "public_lite_resolved_count": resolved,
        "public_lite_resolve_rate": round(float(rate), 6) if rate is not None else None,
    }


def annotate_empirical_signals(
    row: dict[str, Any],
    *,
    public_results: dict[str, Any],
    pilot_summary: dict[str, Any],
) -> dict[str, Any]:
    instance_id = str(row.get("instance_id") or "")
    public = public_metrics(public_results, instance_id)
    pilot = (pilot_summary.get("instances") or {}).get(instance_id) or {}
    annotated = {
        **row,
        **public,
        "_public_seen_count": public["public_lite_seen_count"],
        "_public_resolve_rate": public["public_lite_resolve_rate"],
        "_pilot": pilot,
    }
    if pilot:
        annotated.update(
            {
                "pilot_glm52_lm12m12_resolved": pilot.get("resolved"),
                "pilot_glm52_lm12m12_patch_empty": pilot.get("patch_empty"),
                "pilot_glm52_lm12m12_runtime_budget_exceeded": pilot.get(
                    "runtime_budget_exceeded"
                ),
                "pilot_glm52_lm12m12_provider_transient_exhausted": pilot.get(
                    "provider_transient_exhausted"
                ),
                "pilot_glm52_lm12m12_tool_policy_violation": pilot.get(
                    "tool_policy_violation"
                ),
                "pilot_glm52_lm12m12_disqualified_reason": pilot.get(
                    "openclaw_disqualified_reason"
                ),
                "pilot_glm52_lm12m12_tool_call_count": pilot.get(
                    "openclaw_tool_call_count"
                ),
            }
        )
    return annotated


def add_common_fields(
    row: dict[str, Any],
    *,
    tier: str,
    rank: int,
    score_percentile: float,
    source_subset: str,
) -> dict[str, Any]:
    fail_to_pass = read_json_list(row.get("FAIL_TO_PASS"))
    pass_to_pass = read_json_list(row.get("PASS_TO_PASS"))
    stats = static_difficulty(row)
    return {
        **row,
        "benchmark_id": "swebench_lite",
        "benchmark_name": SOURCE_BENCHMARK,
        "agentic_swe_tier": tier,
        "agentic_swe_subset_rank": rank,
        "source_benchmark": SOURCE_BENCHMARK,
        "source_dataset": DATASET,
        "source_config": CONFIG,
        "source_split": SPLIT,
        "source_subset": source_subset,
        "source_instance_id": str(row.get("instance_id") or ""),
        "repo_language": "python",
        "issue_specificity": "swebench_lite",
        "issue_categories": ["python", "repo_repair", tier],
        "agentic_swe_selection_version": row.get("agentic_swe_selection_version"),
        "agentic_swe_selection_basis": row.get("agentic_swe_selection_basis"),
        "requirements": (
            "Resolve the issue so the fail-to-pass tests pass while preserving existing "
            "pass-to-pass behavior. Keep the fix minimal."
        ),
        "interface": "",
        "dockerhub_tag": None,
        "fail_to_pass": fail_to_pass,
        "pass_to_pass": pass_to_pass,
        "selected_test_files_to_run": stats["selected_test_files_to_run"],
        "static_difficulty_percentile": round(score_percentile, 6),
        "public_lite_seen_count": row.get("public_lite_seen_count", 0),
        "public_lite_resolved_count": row.get("public_lite_resolved_count", 0),
        "public_lite_resolve_rate": row.get("public_lite_resolve_rate"),
        "pilot_glm52_lm12m12_resolved": row.get("pilot_glm52_lm12m12_resolved"),
        "pilot_glm52_lm12m12_patch_empty": row.get("pilot_glm52_lm12m12_patch_empty"),
        "pilot_glm52_lm12m12_runtime_budget_exceeded": row.get(
            "pilot_glm52_lm12m12_runtime_budget_exceeded"
        ),
        "pilot_glm52_lm12m12_provider_transient_exhausted": row.get(
            "pilot_glm52_lm12m12_provider_transient_exhausted"
        ),
        "pilot_glm52_lm12m12_tool_policy_violation": row.get(
            "pilot_glm52_lm12m12_tool_policy_violation"
        ),
        "pilot_glm52_lm12m12_disqualified_reason": row.get(
            "pilot_glm52_lm12m12_disqualified_reason"
        ),
        "pilot_glm52_lm12m12_tool_call_count": row.get(
            "pilot_glm52_lm12m12_tool_call_count"
        ),
        **{key: value for key, value in stats.items() if key != "selected_test_files_to_run"},
    }


def stratified_pick(
    candidates: list[dict[str, Any]],
    *,
    size: int,
    repo_cap: int,
    target_quantile: float | None = None,
) -> list[dict[str, Any]]:
    selected: list[dict[str, Any]] = []
    repo_counts: Counter[str] = Counter()
    ordered = candidates
    if target_quantile is not None:
        ordered = sorted(candidates, key=lambda row: abs(row["_percentile"] - target_quantile))
    for row in ordered:
        repo = str(row.get("repo") or "")
        if repo_counts[repo] >= repo_cap:
            continue
        selected.append(row)
        repo_counts[repo] += 1
        if len(selected) >= size:
            return selected
    for row in ordered:
        if row in selected:
            continue
        selected.append(row)
        if len(selected) >= size:
            return selected
    raise RuntimeError(f"Only selected {len(selected)} rows out of requested {size}")


def stratified_pick_ordered(
    candidates: list[dict[str, Any]],
    *,
    size: int,
    repo_cap: int,
    sort_key,
    excluded_ids: set[str] | None = None,
) -> list[dict[str, Any]]:
    selected: list[dict[str, Any]] = []
    selected_ids: set[str] = set(excluded_ids or set())
    repo_counts: Counter[str] = Counter()
    for row in sorted(candidates, key=sort_key):
        instance_id = str(row.get("instance_id") or "")
        repo = str(row.get("repo") or "")
        if instance_id in selected_ids or repo_counts[repo] >= repo_cap:
            continue
        selected.append(row)
        selected_ids.add(instance_id)
        repo_counts[repo] += 1
        if len(selected) >= size:
            return selected
    raise RuntimeError(
        f"Only selected {len(selected)} rows out of requested {size}; "
        f"repo_counts={dict(repo_counts)}"
    )


def pilot_cost_risk(row: dict[str, Any]) -> bool:
    pilot = row.get("_pilot")
    if not isinstance(pilot, dict) or not pilot:
        return False
    return bool(
        pilot.get("runtime_budget_exceeded")
        or pilot.get("provider_transient_exhausted")
        or pilot.get("tool_policy_violation")
        or (pilot.get("patch_empty") and not pilot.get("resolved"))
    )


def pilot_unresolved(row: dict[str, Any]) -> bool:
    pilot = row.get("_pilot")
    return isinstance(pilot, dict) and bool(pilot) and pilot.get("resolved") is not True


def public_rate(row: dict[str, Any]) -> float:
    value = row.get("_public_resolve_rate")
    return float(value) if isinstance(value, (int, float)) else 0.0


def public_seen(row: dict[str, Any]) -> int:
    return int(row.get("_public_seen_count") or 0)


def v2_low_candidates(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [
        row
        for row in rows
        if public_seen(row) >= 20
        and public_rate(row) >= 0.65
        and float(row.get("_percentile") or 0.0) <= 0.85
        and not pilot_cost_risk(row)
        and not pilot_unresolved(row)
    ]


def v2_middle_candidates(rows: list[dict[str, Any]], *, excluded_ids: set[str]) -> list[dict[str, Any]]:
    return [
        row
        for row in rows
        if str(row.get("instance_id") or "") not in excluded_ids
        and public_seen(row) >= 20
        and 0.25 <= public_rate(row) <= 0.72
        and float(row.get("_percentile") or 0.0) <= 0.95
        and not pilot_cost_risk(row)
    ]


def add_selection_metadata(
    row: dict[str, Any],
    *,
    selection_version: str,
    selection_basis: str,
) -> dict[str, Any]:
    return {
        **row,
        "agentic_swe_selection_version": selection_version,
        "agentic_swe_selection_basis": selection_basis,
    }


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames: list[str] = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)

    def csv_cell(value: Any) -> Any:
        if isinstance(value, (list, dict)):
            value = json.dumps(value, ensure_ascii=False)
        if isinstance(value, str):
            return value.replace("\r\n", "\n").replace("\r", "\n").replace("\n", "\\n").rstrip()
        return value

    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        for row in rows:
            writer.writerow({key: csv_cell(value) for key, value in row.items()})


def relative_display_path(path: Path, *, base_dir: Path) -> str:
    resolved = path.resolve()
    for base in (base_dir.resolve(), REPO_ROOT.resolve()):
        try:
            return str(resolved.relative_to(base))
        except ValueError:
            continue
    return str(path)


def summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "count": len(rows),
        "repo_distribution": dict(Counter(str(row.get("repo")) for row in rows)),
        "static_difficulty": {
            "min": min(row["static_difficulty_score"] for row in rows),
            "mean": sum(row["static_difficulty_score"] for row in rows) / len(rows),
            "max": max(row["static_difficulty_score"] for row in rows),
        },
        "gold_patch_changed_lines_mean": sum(row["gold_patch_changed_lines"] for row in rows)
        / len(rows),
        "gold_patch_files_mean": sum(row["gold_patch_files"] for row in rows) / len(rows),
        "fail_to_pass_count_mean": sum(row["fail_to_pass_count"] for row in rows) / len(rows),
        "pass_to_pass_count_mean": sum(row["pass_to_pass_count"] for row in rows) / len(rows),
    }


def build(
    output_dir: Path,
    *,
    low_size: int,
    middle_size: int,
    refresh_cache: bool,
    refresh_public_results: bool,
    public_results_cache: Path | None,
    pilot_summary_path: Path | None,
    pilot_output_table: Path | None,
) -> dict[str, Any]:
    cache_path = output_dir / "source" / "swebench_lite_test.jsonl"
    source_rows = fetch_rows(refresh_cache=refresh_cache, cache_path=cache_path)
    public_results = fetch_public_lite_results(
        refresh_cache=refresh_public_results,
        cache_path=public_results_cache or output_dir / "source" / "swebench_lite_public_results.json",
    )
    pilot_summary = load_or_build_pilot_summary(
        summary_path=pilot_summary_path,
        output_table_path=pilot_output_table,
        run_label="glm52_lm12m12_20260712",
    )
    scored = []
    for row in source_rows:
        stats = static_difficulty(row)
        scored.append(
            annotate_empirical_signals(
                {**row, "_score": stats["static_difficulty_score"]},
                public_results=public_results,
                pilot_summary=pilot_summary,
            )
        )
    scored = sorted(scored, key=lambda row: (row["_score"], str(row.get("instance_id"))))
    total = len(scored)
    for index, row in enumerate(scored):
        row["_percentile"] = index / max(total - 1, 1)

    low_candidates = [row for row in scored if row["_percentile"] <= 0.38]
    middle_candidates = [row for row in scored if 0.40 <= row["_percentile"] <= 0.72]
    low_raw = stratified_pick(low_candidates, size=low_size, repo_cap=6)
    middle_raw = stratified_pick(middle_candidates, size=middle_size, repo_cap=6, target_quantile=0.56)
    low_v2_raw = stratified_pick_ordered(
        v2_low_candidates(scored),
        size=low_size,
        repo_cap=6,
        sort_key=lambda row: (-public_rate(row), row["_score"], str(row.get("instance_id"))),
    )
    low_v2_ids = {str(row.get("instance_id") or "") for row in low_v2_raw}
    middle_v2_raw = stratified_pick_ordered(
        v2_middle_candidates(scored, excluded_ids=low_v2_ids),
        size=middle_size,
        repo_cap=6,
        sort_key=lambda row: (
            abs(public_rate(row) - 0.50),
            abs(float(row.get("_percentile") or 0.0) - 0.56),
            row["_score"],
            str(row.get("instance_id")),
        ),
    )

    subsets = {
        "low_36": [
            add_common_fields(
                row,
                tier="low",
                rank=rank,
                score_percentile=row["_percentile"],
                source_subset="low_36",
            )
            for rank, row in enumerate(low_raw, start=1)
        ],
        "middle_36": [
            add_common_fields(
                row,
                tier="middle",
                rank=rank,
                score_percentile=row["_percentile"],
                source_subset="middle_36",
            )
            for rank, row in enumerate(middle_raw, start=1)
        ],
        "low_v2_36": [
            add_common_fields(
                add_selection_metadata(
                    row,
                    selection_version=V2_SELECTION_VERSION,
                    selection_basis=(
                        "public_lite_resolve_rate>=0.65, public_lite_seen_count>=20, "
                        "static_percentile<=0.85, no local GLM-5.2 pilot unresolved/cost-risk signal"
                    ),
                ),
                tier="low",
                rank=rank,
                score_percentile=row["_percentile"],
                source_subset="low_v2_36",
            )
            for rank, row in enumerate(low_v2_raw, start=1)
        ],
        "middle_v2_36": [
            add_common_fields(
                add_selection_metadata(
                    row,
                    selection_version=V2_SELECTION_VERSION,
                    selection_basis=(
                        "0.25<=public_lite_resolve_rate<=0.72, public_lite_seen_count>=20, "
                        "static_percentile<=0.95, no local GLM-5.2 pilot cost-risk signal"
                    ),
                ),
                tier="middle",
                rank=rank,
                score_percentile=row["_percentile"],
                source_subset="middle_v2_36",
            )
            for rank, row in enumerate(middle_v2_raw, start=1)
        ],
    }
    subsets["low_middle_72"] = subsets["low_36"] + subsets["middle_36"]
    subsets["low_middle_v2_72"] = subsets["low_v2_36"] + subsets["middle_v2_36"]

    for name, rows in subsets.items():
        write_jsonl(output_dir / "subsets" / f"{name}.jsonl", rows)
        write_csv(output_dir / "subsets" / f"{name}.csv", rows)
        write_json(
            output_dir / "subsets" / f"{name}_instance_ids.json",
            [str(row["instance_id"]) for row in rows],
        )

    manifest = {
        "benchmark": "Agentic SWE-Assorted Low/Middle",
        "source_benchmark": SOURCE_BENCHMARK,
        "source_dataset": DATASET,
        "source_config": CONFIG,
        "source_split": SPLIT,
        "source_count": len(source_rows),
        "selection_method": (
            "Legacy low_36/middle_36 use static-only SWE-bench Lite slicing. "
            "V2 low_v2_36/middle_v2_36 add public SWE-bench Lite per-instance "
            "resolution rates from SWE-bench/experiments as a difficulty prior and "
            "local GLM-5.2 pilot signals only to remove cost-risk outliers. Public "
            "outcomes are used for subset calibration, not scoring."
        ),
        "selection_versions": {
            "legacy_static": {
                "subsets": ["low_36", "middle_36", "low_middle_72"],
                "method": (
                    "Static-only lower/middle difficulty bands using gold patch files/hunks/"
                    "changed lines, test patch size, issue length, and F2P/P2P counts."
                ),
            },
            V2_SELECTION_VERSION: {
                "subsets": ["low_v2_36", "middle_v2_36", "low_middle_v2_72"],
                "public_results_cache": str(
                    relative_display_path(
                        public_results_cache
                        or output_dir / "source" / "swebench_lite_public_results.json",
                        base_dir=output_dir,
                    )
                ),
                "public_submission_count": public_results.get("submission_count"),
                "pilot_summary_path": (
                    relative_display_path(pilot_summary_path, base_dir=output_dir)
                    if pilot_summary_path
                    else None
                ),
                "pilot_total_instances": pilot_summary.get("total_instances"),
                "low_rule": (
                    "public_lite_resolve_rate>=0.65, public_lite_seen_count>=20, "
                    "static_percentile<=0.85, no local GLM-5.2 pilot unresolved/cost-risk signal"
                ),
                "middle_rule": (
                    "0.25<=public_lite_resolve_rate<=0.72, public_lite_seen_count>=20, "
                    "static_percentile<=0.95, no local GLM-5.2 pilot cost-risk signal"
                ),
                "repo_cap": 6,
            },
        },
        "subsets": {
            name: {
                "count": len(rows),
                "jsonl_path": f"subsets/{name}.jsonl",
                "csv_path": f"subsets/{name}.csv",
                "instance_ids_path": f"subsets/{name}_instance_ids.json",
                **summarize(rows),
            }
            for name, rows in subsets.items()
        },
        "assorted_80_plan": {
            "low": 36,
            "middle": 36,
            "high": 8,
            "low_middle_jsonl_path": "subsets/low_middle_v2_72.jsonl",
            "legacy_low_middle_jsonl_path": "subsets/low_middle_72.jsonl",
            "high_subset": "data/taiwan/deepswe/subsets/essential_8.jsonl",
        },
    }
    write_json(output_dir / "manifest.json", manifest)
    return manifest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--low-size", type=int, default=36)
    parser.add_argument("--middle-size", type=int, default=36)
    parser.add_argument("--refresh-cache", action="store_true")
    parser.add_argument("--refresh-public-results", action="store_true")
    parser.add_argument("--public-results-cache", type=Path)
    parser.add_argument("--pilot-summary-path", type=Path, default=DEFAULT_PILOT_SUMMARY)
    parser.add_argument("--pilot-output-table", type=Path, default=DEFAULT_PILOT_OUTPUT_TABLE)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    manifest = build(
        args.output_dir,
        low_size=args.low_size,
        middle_size=args.middle_size,
        refresh_cache=args.refresh_cache,
        refresh_public_results=args.refresh_public_results,
        public_results_cache=args.public_results_cache,
        pilot_summary_path=args.pilot_summary_path,
        pilot_output_table=args.pilot_output_table,
    )
    for name, entry in manifest["subsets"].items():
        difficulty = entry["static_difficulty"]
        print(
            f"{name}: count={entry['count']} "
            f"score_mean={difficulty['mean']:.2f} "
            f"score_range={difficulty['min']:.2f}-{difficulty['max']:.2f}"
        )
    print(f"Wrote {args.output_dir / 'manifest.json'}")


if __name__ == "__main__":
    main()
