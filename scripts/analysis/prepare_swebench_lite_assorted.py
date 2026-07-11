#!/usr/bin/env python3
"""Build SWE-bench Lite Low/Middle slices for Agentic SWE-Assorted.

The slices are selected from static task features only. Model outcomes are not
used. This keeps the split auditable and avoids fitting the subset to any model
family.
"""

from __future__ import annotations

import argparse
import ast
import csv
import json
import math
import re
from collections import Counter
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
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    key: json.dumps(value, ensure_ascii=False)
                    if isinstance(value, (list, dict))
                    else value
                    for key, value in row.items()
                }
            )


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


def build(output_dir: Path, *, low_size: int, middle_size: int, refresh_cache: bool) -> dict[str, Any]:
    cache_path = output_dir / "source" / "swebench_lite_test.jsonl"
    source_rows = fetch_rows(refresh_cache=refresh_cache, cache_path=cache_path)
    scored = []
    for row in source_rows:
        stats = static_difficulty(row)
        scored.append({**row, "_score": stats["static_difficulty_score"]})
    scored = sorted(scored, key=lambda row: (row["_score"], str(row.get("instance_id"))))
    total = len(scored)
    for index, row in enumerate(scored):
        row["_percentile"] = index / max(total - 1, 1)

    low_candidates = [row for row in scored if row["_percentile"] <= 0.38]
    middle_candidates = [row for row in scored if 0.40 <= row["_percentile"] <= 0.72]
    low_raw = stratified_pick(low_candidates, size=low_size, repo_cap=6)
    middle_raw = stratified_pick(middle_candidates, size=middle_size, repo_cap=6, target_quantile=0.56)

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
    }
    subsets["low_middle_72"] = subsets["low_36"] + subsets["middle_36"]

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
            "Static-only SWE-bench Lite slicing. Low is selected from the lower "
            "difficulty band; Middle is selected near the middle difficulty band. "
            "Difficulty uses gold patch files/hunks/changed lines, test patch size, "
            "issue length, and F2P/P2P test counts. No model outcomes are used."
        ),
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
            "low_middle_jsonl_path": "subsets/low_middle_72.jsonl",
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
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    manifest = build(
        args.output_dir,
        low_size=args.low_size,
        middle_size=args.middle_size,
        refresh_cache=args.refresh_cache,
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
