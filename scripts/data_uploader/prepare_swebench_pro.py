#!/usr/bin/env python3
"""
Materialize SWE-bench Pro public data for Nejumi Taiwan agentic SWE runs.

This script keeps data preparation separate from model execution:

  python3 scripts/data_uploader/prepare_swebench_pro.py \
    --output-dir data/taiwan --leaderboard-size 80 --smoke-size 10

  python3 scripts/data_uploader/prepare_swebench_pro.py \
    --output-dir data/taiwan --upload --entity llm-leaderboard --project tc-leaderboard

The W&B artifact layout is:

  swebench_pro_public/
    test.jsonl
    test.csv
    manifest.json
    subsets/
      smoke.jsonl
      leaderboard_80.jsonl
      full_public.jsonl
      smoke.csv
      leaderboard_80.csv
      full_public.csv
"""

from __future__ import annotations

import argparse
import ast
import csv
import hashlib
import json
import math
import random
import shlex
import subprocess
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable

import wandb


DEFAULT_DATASET = "ScaleAI/SWE-bench_Pro"
DEFAULT_SPLIT = "test"
DEFAULT_ARTIFACT_NAME = "swebench-pro-public"
DEFAULT_DATASET_DIR_NAME = "swebench_pro_public"
DEFAULT_SEED = 45
REPO_METADATA_SCHEMA_VERSION = 1
REPO_TREE_METADATA_METHOD = "git-ls-tree-v1"
DEFAULT_COMPACT_MAX_COST_PERCENTILE = 0.70
TAIWAN_RECOMMENDED_SUBSET = "leaderboard_compact_80"
TAIWAN_RECOMMENDED_MAX_INPUT_TOKENS = 1_000_000
TAIWAN_RECOMMENDED_MAX_TOOL_CALLS = 60

CSV_FIELDS = [
    "repo",
    "instance_id",
    "base_commit",
    "problem_statement",
    "requirements",
    "interface",
    "repo_language",
    "issue_specificity",
    "issue_categories",
    "before_repo_set_cmd",
    "selected_test_files_to_run",
    "dockerhub_tag",
    "fail_to_pass",
    "pass_to_pass",
    "patch",
    "test_patch",
]

SOURCE_EXTENSIONS = {
    ".c",
    ".cc",
    ".cfg",
    ".cpp",
    ".cs",
    ".css",
    ".go",
    ".h",
    ".hpp",
    ".html",
    ".java",
    ".js",
    ".jsx",
    ".json",
    ".kt",
    ".kts",
    ".m",
    ".md",
    ".mm",
    ".php",
    ".py",
    ".pyi",
    ".rb",
    ".rs",
    ".scala",
    ".scss",
    ".sh",
    ".sql",
    ".svelte",
    ".swift",
    ".toml",
    ".ts",
    ".tsx",
    ".vue",
    ".xml",
    ".yaml",
    ".yml",
}

SOURCE_FILENAMES = {
    "Dockerfile",
    "Makefile",
    "Pipfile",
    "pyproject.toml",
    "package.json",
    "pnpm-lock.yaml",
    "requirements.txt",
}


def jsonable(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [jsonable(v) for v in value]
    if hasattr(value, "item"):
        try:
            return value.item()
        except Exception:
            pass
    return value


def as_list(value: Any) -> list[Any]:
    if value is None:
        return []
    if isinstance(value, list):
        return _flatten_stringified_singleton_list(value)
    if isinstance(value, tuple):
        return _flatten_stringified_singleton_list(list(value))
    if isinstance(value, str):
        stripped = value.strip()
        if not stripped:
            return []
        parsed: Any
        try:
            parsed = json.loads(stripped)
        except json.JSONDecodeError:
            try:
                parsed = ast.literal_eval(stripped)
            except (SyntaxError, ValueError):
                return [stripped]
        if isinstance(parsed, (list, tuple)):
            return _flatten_stringified_singleton_list(list(parsed))
        return [] if parsed is None else [parsed]
    return [value]


def _flatten_stringified_singleton_list(values: list[Any]) -> list[Any]:
    if len(values) != 1 or not isinstance(values[0], str):
        return values
    parsed = as_list(values[0])
    if len(parsed) == 1 and parsed[0] == values[0]:
        return values
    return parsed


def stable_key(row: dict[str, Any], seed: int) -> str:
    text = f"{seed}:{row.get('instance_id', '')}"
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def safe_path_component(value: str) -> str:
    return "".join(ch if ch.isalnum() or ch in "._-" else "_" for ch in value)


def repo_url(repo: str) -> str:
    if repo.startswith("http://") or repo.startswith("https://") or repo.endswith(".git"):
        return repo
    return f"https://github.com/{repo}.git"


def run_command(
    command: list[str],
    cwd: Path | None = None,
    check: bool = True,
) -> subprocess.CompletedProcess[str]:
    result = subprocess.run(
        command,
        cwd=str(cwd) if cwd else None,
        text=True,
        capture_output=True,
        check=False,
    )
    if check and result.returncode != 0:
        raise RuntimeError(
            "Command failed\n"
            f"cmd: {' '.join(shlex.quote(part) for part in command)}\n"
            f"cwd: {cwd}\n"
            f"returncode: {result.returncode}\n"
            f"stdout:\n{result.stdout}\n"
            f"stderr:\n{result.stderr}"
        )
    return result


def issue_category_key(value: Any) -> str:
    categories = as_list(value)
    if not categories:
        return "unknown_category"
    return str(categories[0]) or "unknown_category"


def selected_test_paths(row: dict[str, Any]) -> list[str]:
    paths: list[str] = []
    for item in as_list(row.get("selected_test_files_to_run")):
        path = str(item).split("::", 1)[0].strip().lstrip("./")
        if path:
            paths.append(path)
    return sorted(set(paths))


def is_source_path(path: str) -> bool:
    normalized = path.rsplit("/", 1)[-1]
    if normalized in SOURCE_FILENAMES:
        return True
    return Path(normalized).suffix.lower() in SOURCE_EXTENSIONS


def parse_ls_tree(stdout: str) -> list[dict[str, Any]]:
    files: list[dict[str, Any]] = []
    for record in stdout.split("\0"):
        if not record:
            continue
        try:
            metadata, path = record.split("\t", 1)
        except ValueError:
            continue
        parts = metadata.split()
        if len(parts) < 4 or parts[1] != "blob":
            continue
        size_text = parts[3]
        try:
            size = int(size_text)
        except ValueError:
            size = 0
        files.append({"path": path, "bytes": max(0, size)})
    return files


def static_cost_proxy_score(metrics: dict[str, Any]) -> float:
    """Model-independent proxy for expected agentic repository exploration cost."""
    tracked_files = float(metrics.get("tracked_file_count") or 0)
    tracked_kib = float(metrics.get("tracked_total_bytes") or 0) / 1024.0
    source_files = float(metrics.get("source_file_count") or 0)
    source_kib = float(metrics.get("source_total_bytes") or 0) / 1024.0
    max_file_kib = float(metrics.get("max_file_bytes") or 0) / 1024.0
    selected_test_kib = float(metrics.get("selected_test_total_bytes") or 0) / 1024.0
    prompt_chars = float(metrics.get("task_text_chars") or 0)
    score = (
        4.0 * math.log1p(tracked_files)
        + 3.0 * math.log1p(tracked_kib)
        + 3.0 * math.log1p(source_files)
        + 3.0 * math.log1p(source_kib)
        + 2.0 * math.log1p(max_file_kib)
        + 2.0 * math.log1p(selected_test_kib)
        + 0.5 * math.log1p(prompt_chars)
    )
    return round(score, 6)


def repo_tree_metrics(row: dict[str, Any], tree_files: list[dict[str, Any]]) -> dict[str, Any]:
    selected_paths = set(selected_test_paths(row))
    tracked_total_bytes = sum(int(item["bytes"]) for item in tree_files)
    source_files = [item for item in tree_files if is_source_path(str(item["path"]))]
    selected_test_files = [
        item for item in tree_files if str(item["path"]).lstrip("./") in selected_paths
    ]
    task_text_chars = sum(
        len(str(row.get(field) or ""))
        for field in ("problem_statement", "requirements", "interface")
    )
    metrics: dict[str, Any] = {
        "metadata_schema_version": REPO_METADATA_SCHEMA_VERSION,
        "measurement_method": REPO_TREE_METADATA_METHOD,
        "instance_id": str(row.get("instance_id") or ""),
        "repo": str(row.get("repo") or ""),
        "base_commit": str(row.get("base_commit") or ""),
        "tracked_file_count": len(tree_files),
        "tracked_total_bytes": tracked_total_bytes,
        "source_file_count": len(source_files),
        "source_total_bytes": sum(int(item["bytes"]) for item in source_files),
        "max_file_bytes": max((int(item["bytes"]) for item in tree_files), default=0),
        "selected_test_file_count": len(selected_test_files),
        "selected_test_total_bytes": sum(int(item["bytes"]) for item in selected_test_files),
        "selected_test_path_count": len(selected_paths),
        "task_text_chars": task_text_chars,
    }
    metrics["static_cost_proxy_score"] = static_cost_proxy_score(metrics)
    return metrics


def repo_mirror_path(repo: str, mirror_root: Path) -> Path:
    return mirror_root / f"{safe_path_component(repo)}.git"


def ensure_repo_mirror(repo: str, mirror_root: Path, refresh: bool) -> Path:
    mirror_root.mkdir(parents=True, exist_ok=True)
    mirror = repo_mirror_path(repo, mirror_root)
    if mirror.exists():
        if refresh:
            run_command(["git", "remote", "update", "--prune"], cwd=mirror)
        return mirror
    run_command(["git", "clone", "--mirror", repo_url(repo), str(mirror)])
    return mirror


def collect_repo_metadata(rows: list[dict[str, Any]], mirror_root: Path, refresh: bool) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    mirrors: dict[str, Path] = {}
    for index, row in enumerate(rows, start=1):
        repo = str(row.get("repo") or "")
        if not repo:
            raise ValueError(f"Row {row.get('instance_id')} is missing repo")
        mirror = mirrors.get(repo)
        if mirror is None:
            mirror = ensure_repo_mirror(repo, mirror_root, refresh)
            mirrors[repo] = mirror
        base_commit = str(row.get("base_commit") or "")
        if not base_commit:
            raise ValueError(f"Row {row.get('instance_id')} is missing base_commit")
        tree = run_command(["git", "ls-tree", "-r", "-l", "-z", base_commit], cwd=mirror)
        records.append(repo_tree_metrics(row, parse_ls_tree(tree.stdout)))
        if index % 50 == 0:
            print(f"Collected SWE-bench Pro repo metadata: {index}/{len(rows)}")
    return records


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def load_repo_metadata(path: Path) -> dict[str, dict[str, Any]]:
    records = read_jsonl(path)
    by_instance: dict[str, dict[str, Any]] = {}
    for record in records:
        instance_id = str(record.get("instance_id") or "")
        if not instance_id:
            raise ValueError(f"Repo metadata record in {path} is missing instance_id")
        if int(record.get("metadata_schema_version") or 0) != REPO_METADATA_SCHEMA_VERSION:
            raise ValueError(
                f"Repo metadata record {instance_id} has unsupported schema "
                f"{record.get('metadata_schema_version')}"
            )
        by_instance[instance_id] = record
    return by_instance


def attach_repo_metadata(
    rows: list[dict[str, Any]],
    metadata_by_instance: dict[str, dict[str, Any]],
) -> list[dict[str, Any]]:
    enriched: list[dict[str, Any]] = []
    missing: list[str] = []
    for row in rows:
        instance_id = str(row.get("instance_id") or "")
        metadata = metadata_by_instance.get(instance_id)
        if metadata is None:
            missing.append(instance_id)
            continue
        if metadata.get("repo") != row.get("repo") or metadata.get("base_commit") != row.get("base_commit"):
            raise ValueError(
                f"Repo metadata mismatch for {instance_id}: "
                f"metadata repo/base={metadata.get('repo')}@{metadata.get('base_commit')} "
                f"row repo/base={row.get('repo')}@{row.get('base_commit')}"
            )
        new_row = dict(row)
        new_row["repo_static_cost"] = {
            key: metadata[key]
            for key in (
                "metadata_schema_version",
                "measurement_method",
                "tracked_file_count",
                "tracked_total_bytes",
                "source_file_count",
                "source_total_bytes",
                "max_file_bytes",
                "selected_test_file_count",
                "selected_test_total_bytes",
                "selected_test_path_count",
                "task_text_chars",
                "static_cost_proxy_score",
            )
            if key in metadata
        }
        new_row["static_cost_proxy_score"] = metadata.get("static_cost_proxy_score")
        enriched.append(new_row)
    if missing:
        preview = ", ".join(missing[:5])
        raise ValueError(f"Missing repo metadata for {len(missing)} instances: {preview}")
    return add_cost_percentiles(enriched)


def add_cost_percentiles(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    if not rows:
        return []
    ranked = sorted(
        rows,
        key=lambda row: (
            float(row.get("static_cost_proxy_score") or 0.0),
            stable_key(row, DEFAULT_SEED),
        ),
    )
    total = len(ranked)
    percentiles = {
        str(row.get("instance_id")): round((index + 1) / total, 6)
        for index, row in enumerate(ranked)
    }
    output: list[dict[str, Any]] = []
    for row in rows:
        new_row = dict(row)
        percentile = percentiles[str(row.get("instance_id"))]
        new_row["static_cost_proxy_percentile"] = percentile
        repo_static_cost = dict(new_row.get("repo_static_cost") or {})
        repo_static_cost["static_cost_proxy_percentile"] = percentile
        new_row["repo_static_cost"] = repo_static_cost
        output.append(new_row)
    return output


def stratum_key(row: dict[str, Any]) -> tuple[str, str, str]:
    return (
        str(row.get("repo_language") or "unknown_language"),
        str(row.get("issue_specificity") or "unknown_specificity"),
        issue_category_key(row.get("issue_categories")),
    )


def stratified_sample(rows: list[dict[str, Any]], size: int, seed: int) -> list[dict[str, Any]]:
    if size >= len(rows):
        return sorted(rows, key=lambda row: stable_key(row, seed))

    language_groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        language_groups[str(row.get("repo_language") or "unknown_language")].append(row)

    language_allocations: dict[str, int] = {}
    remainders: list[tuple[float, str]] = []
    for language, language_rows in language_groups.items():
        raw = size * len(language_rows) / len(rows)
        take = min(len(language_rows), int(raw))
        language_allocations[language] = take
        remainders.append((raw - take, language))

    # Preserve minority languages whenever the requested subset is large enough.
    if size >= len(language_groups):
        for language, language_rows in language_groups.items():
            if language_rows and language_allocations[language] == 0:
                language_allocations[language] = 1

    selected_count = sum(language_allocations.values())
    while selected_count > size:
        candidates = [
            language
            for language, take in language_allocations.items()
            if take > 1
        ]
        if not candidates:
            break
        language = max(candidates, key=lambda key: language_allocations[key])
        language_allocations[language] -= 1
        selected_count -= 1

    for _, language in sorted(remainders, reverse=True):
        if selected_count >= size:
            break
        if language_allocations[language] < len(language_groups[language]):
            language_allocations[language] += 1
            selected_count += 1

    sampled_by_language: list[dict[str, Any]] = []
    for language, language_rows in language_groups.items():
        take = language_allocations[language]
        if take <= 0:
            continue
        sampled_by_language.extend(_sample_within_language(language_rows, take, seed))

    return sorted(sampled_by_language[:size], key=lambda row: stable_key(row, seed))


def auto_repo_cap(rows: list[dict[str, Any]], size: int) -> int:
    repo_count = len({str(row.get("repo") or "unknown_repo") for row in rows}) or 1
    return max(1, math.ceil((size / repo_count) * 1.5))


def enforce_repo_cap(
    selected: list[dict[str, Any]],
    candidates: list[dict[str, Any]],
    *,
    size: int,
    seed: int,
    max_per_repo: int,
) -> list[dict[str, Any]]:
    if max_per_repo <= 0:
        return sorted(selected[:size], key=lambda row: stable_key(row, seed))

    selected_ids = {str(row.get("instance_id")) for row in selected}
    repo_counts: dict[str, int] = defaultdict(int)
    capped: list[dict[str, Any]] = []
    for row in sorted(selected, key=lambda item: stable_key(item, seed)):
        repo = str(row.get("repo") or "unknown_repo")
        if repo_counts[repo] >= max_per_repo:
            continue
        capped.append(row)
        repo_counts[repo] += 1

    if len(capped) >= size:
        return sorted(capped[:size], key=lambda row: stable_key(row, seed))

    for row in sorted(candidates, key=lambda item: stable_key(item, seed + 17)):
        instance_id = str(row.get("instance_id"))
        if instance_id in selected_ids:
            continue
        repo = str(row.get("repo") or "unknown_repo")
        if repo_counts[repo] >= max_per_repo:
            continue
        capped.append(row)
        selected_ids.add(instance_id)
        repo_counts[repo] += 1
        if len(capped) >= size:
            break

    if len(capped) < size:
        raise ValueError(
            f"Repo cap {max_per_repo} left only {len(capped)} rows for requested compact subset size {size}"
        )
    return sorted(capped[:size], key=lambda row: stable_key(row, seed))


def compact_candidate_pool(
    rows: list[dict[str, Any]],
    *,
    max_cost_percentile: float,
    max_tracked_total_bytes: int | None = None,
    max_tracked_file_count: int | None = None,
) -> list[dict[str, Any]]:
    if not 0 < max_cost_percentile <= 1:
        raise ValueError("--compact-max-cost-percentile must be in (0, 1]")
    candidates: list[dict[str, Any]] = []
    for row in rows:
        percentile = float(row.get("static_cost_proxy_percentile") or 0.0)
        cost = row.get("repo_static_cost") if isinstance(row.get("repo_static_cost"), dict) else {}
        tracked_total_bytes = int(cost.get("tracked_total_bytes") or 0)
        tracked_file_count = int(cost.get("tracked_file_count") or 0)
        if percentile > max_cost_percentile:
            continue
        if max_tracked_total_bytes is not None and tracked_total_bytes > max_tracked_total_bytes:
            continue
        if max_tracked_file_count is not None and tracked_file_count > max_tracked_file_count:
            continue
        candidates.append(row)
    return candidates


def compact_sample(
    rows: list[dict[str, Any]],
    *,
    size: int,
    seed: int,
    max_cost_percentile: float,
    max_per_repo: int | None,
    max_tracked_total_bytes: int | None = None,
    max_tracked_file_count: int | None = None,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    candidates = compact_candidate_pool(
        rows,
        max_cost_percentile=max_cost_percentile,
        max_tracked_total_bytes=max_tracked_total_bytes,
        max_tracked_file_count=max_tracked_file_count,
    )
    if len(candidates) < size:
        raise ValueError(
            f"Compact candidate pool has {len(candidates)} rows, fewer than requested size {size}"
        )
    repo_cap = max_per_repo if max_per_repo is not None and max_per_repo > 0 else auto_repo_cap(candidates, size)
    initial = stratified_sample(candidates, size, seed)
    sampled = enforce_repo_cap(
        initial,
        candidates,
        size=size,
        seed=seed,
        max_per_repo=repo_cap,
    )
    scores = [float(row.get("static_cost_proxy_score") or 0.0) for row in sampled]
    metadata = {
        "method": "deterministic cost-capped stratified sample",
        "size": len(sampled),
        "candidate_pool_size": len(candidates),
        "max_cost_percentile": max_cost_percentile,
        "max_per_repo": repo_cap,
        "max_tracked_total_bytes": max_tracked_total_bytes,
        "max_tracked_file_count": max_tracked_file_count,
        "strata": ["repo_language", "issue_specificity", "first issue_categories"],
        "cost_proxy": "static repository tree metadata; model-independent",
        "score_min": min(scores) if scores else None,
        "score_median": sorted(scores)[len(scores) // 2] if scores else None,
        "score_max": max(scores) if scores else None,
    }
    return sampled, metadata


def _sample_within_language(
    rows: list[dict[str, Any]], size: int, seed: int
) -> list[dict[str, Any]]:
    if size >= len(rows):
        return sorted(rows, key=lambda row: stable_key(row, seed))

    groups: dict[tuple[str, str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[stratum_key(row)].append(row)
    for group_rows in groups.values():
        group_rows.sort(key=lambda row: stable_key(row, seed))

    allocations: dict[tuple[str, str, str], int] = {}
    remainders: list[tuple[float, tuple[str, str, str]]] = []
    for key, group_rows in groups.items():
        raw = size * len(group_rows) / len(rows)
        take = min(len(group_rows), int(raw))
        allocations[key] = take
        remainders.append((raw - take, key))

    selected_count = sum(allocations.values())
    for _, key in sorted(remainders, reverse=True):
        if selected_count >= size:
            break
        if allocations[key] < len(groups[key]):
            allocations[key] += 1
            selected_count += 1

    if selected_count < size:
        for key in sorted(groups, key=lambda k: len(groups[k]), reverse=True):
            while selected_count < size and allocations[key] < len(groups[key]):
                allocations[key] += 1
                selected_count += 1

    sampled: list[dict[str, Any]] = []
    for key, take in allocations.items():
        sampled.extend(groups[key][:take])
    return sorted(sampled[:size], key=lambda row: stable_key(row, seed))


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def csv_value(value: Any) -> str:
    if isinstance(value, (dict, list, tuple)):
        return json.dumps(value, ensure_ascii=False)
    if value is None:
        return ""
    return str(value)


def write_csv(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_FIELDS)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: csv_value(row.get(field)) for field in CSV_FIELDS})


def normalize_row(row: dict[str, Any]) -> dict[str, Any]:
    normalized = {str(key): jsonable(value) for key, value in row.items()}
    if "FAIL_TO_PASS" in normalized and "fail_to_pass" not in normalized:
        normalized["fail_to_pass"] = normalized["FAIL_TO_PASS"]
    if "PASS_TO_PASS" in normalized and "pass_to_pass" not in normalized:
        normalized["pass_to_pass"] = normalized["PASS_TO_PASS"]
    normalized["fail_to_pass"] = as_list(normalized.get("fail_to_pass"))
    normalized["pass_to_pass"] = as_list(normalized.get("pass_to_pass"))
    normalized["selected_test_files_to_run"] = as_list(
        normalized.get("selected_test_files_to_run")
    )
    normalized["issue_categories"] = as_list(normalized.get("issue_categories"))
    return normalized


def metadata_block(row: dict[str, Any]) -> str:
    metadata_lines = [
        f"repo: {row.get('repo')}",
        f"base_commit: {row.get('base_commit')}",
        f"language: {row.get('repo_language')}",
        f"issue_specificity: {row.get('issue_specificity')}",
    ]
    return "<metadata>\n" + "\n".join(metadata_lines) + "\n</metadata>"


def extra_task_blocks(row: dict[str, Any]) -> list[str]:
    extra_blocks: list[str] = []
    requirements = str(row.get("requirements") or "").strip()
    if requirements:
        extra_blocks.append(f"<requirements>\n{requirements}\n</requirements>")
    interface = str(row.get("interface") or "").strip()
    if interface:
        extra_blocks.append(f"<interface>\n{interface}\n</interface>")
    return extra_blocks


def build_agentic_prompt(row: dict[str, Any]) -> str:
    """Build a code-free task prompt for agentic repository exploration."""
    parts = [
        "# SWE-bench Pro Task",
        "",
        "You are running inside a checkout of the target repository at the base commit.",
        "Inspect the repository, edit files as needed, and leave the working tree with the minimal fix.",
        "Do not create commits. The harness will collect `git diff --binary` after you finish.",
        "",
        metadata_block(row),
        "",
        f"<issue>\n{str(row.get('problem_statement') or '').strip()}\n</issue>",
    ]
    parts.extend(extra_task_blocks(row))
    return "\n\n".join(parts) + "\n"


def enrich_row(row: dict[str, Any], args: argparse.Namespace) -> dict[str, Any]:
    enriched = dict(row)
    agentic_prompt = build_agentic_prompt(row)
    enriched["agentic_prompt"] = agentic_prompt
    # Keep `text` code-free so generic readers do not silently turn this into a one-shot benchmark.
    enriched["text"] = agentic_prompt
    return enriched


def load_rows(dataset_name: str, split: str, limit: int | None) -> list[dict[str, Any]]:
    from datasets import load_dataset

    dataset = load_dataset(dataset_name, split=split)
    rows = [normalize_row(dict(row)) for row in dataset]
    if limit is not None:
        rows = rows[:limit]
    return rows


def write_readme(
    path: Path,
    rows: list[dict[str, Any]],
    leaderboard_size: int,
    compact_sizes: list[int],
) -> None:
    compact_lines = [
        f"- `subsets/leaderboard_compact_{size}.*`: deterministic cost-capped stratified subset"
        for size in compact_sizes
    ]
    path.write_text(
        "\n".join(
            [
                "# SWE-bench Pro Public for Nejumi Taiwan",
                "",
                "This artifact contains the public `ScaleAI/SWE-bench_Pro` test split",
                "materialized for Agentic SWE evaluation.",
                "",
                "## Files",
                "",
                "- `test.jsonl` / `test.csv`: full public split",
                "- `subsets/smoke.*`: small harness check subset",
                f"- `subsets/leaderboard_{leaderboard_size}.*`: deterministic stratified leaderboard subset",
                *compact_lines,
                "- `subsets/full_public.*`: full 731-instance public split",
                "- `manifest.json`: counts, hashes, source, and sampling metadata",
                "- `repo_metadata.jsonl`: optional model-independent repo tree size metadata when compact subsets are built",
                "",
                "JSONL and Hugging Face dataset rows also include:",
                "",
                "- `agentic_prompt` / `text`: code-free task prompt for real checkout exploration",
                "",
                "## Evaluation",
                "",
                "Use `scripts/tools/run_swebench_pro_openclaw.py` to generate patches",
                "and `scripts/tools/evaluate_swebench_pro_patches.py` to run the Scale",
                "official evaluator with Docker or Modal.",
                "",
                "Offline/on-prem support should mirror target repositories as separate",
                "checkout or bare-repo artifacts instead of embedding source in prompts.",
                "",
                "Compact subsets exclude high static-cost instances before stratified",
                "sampling. They are separate benchmark variants and must not be reported",
                "as the full SWE-bench Pro public split.",
                "",
                "Taiwan leaderboard default recommendation:",
                "",
                f"- subset: `{TAIWAN_RECOMMENDED_SUBSET}`",
                f"- runtime cap: {TAIWAN_RECOMMENDED_MAX_INPUT_TOKENS:,} input tokens per task",
                f"- runtime cap: {TAIWAN_RECOMMENDED_MAX_TOOL_CALLS} tool calls per task",
                "- large-repo filtering is a static risk reducer; runtime caps are the primary cost guard",
                "",
                f"Rows: {len(rows)}",
            ]
        )
        + "\n",
        encoding="utf-8",
    )


def materialize(args: argparse.Namespace) -> Path:
    rows = load_rows(args.dataset, args.split, args.limit)
    if not rows:
        raise RuntimeError("No SWE-bench Pro rows were loaded")

    artifact_root = args.output_dir / args.dataset_dir_name
    subsets_dir = artifact_root / "subsets"
    artifact_root.mkdir(parents=True, exist_ok=True)
    subsets_dir.mkdir(parents=True, exist_ok=True)

    metadata_path = args.repo_metadata_jsonl or (artifact_root / "repo_metadata.jsonl")
    needs_compact = args.compact_leaderboard_size > 0 or args.compact_pilot_size > 0
    repo_metadata_records: list[dict[str, Any]] = []
    if args.collect_repo_metadata:
        repo_metadata_records = collect_repo_metadata(
            rows,
            args.metadata_mirror_root,
            args.metadata_refresh,
        )
        write_jsonl(metadata_path, repo_metadata_records)
    elif metadata_path.exists():
        repo_metadata_records = read_jsonl(metadata_path)
    elif needs_compact:
        raise RuntimeError(
            "Compact SWE-bench Pro subsets require repo metadata. "
            "Pass --collect-repo-metadata or --repo-metadata-jsonl."
        )

    if repo_metadata_records:
        artifact_metadata_path = artifact_root / "repo_metadata.jsonl"
        if metadata_path.resolve() != artifact_metadata_path.resolve():
            write_jsonl(artifact_metadata_path, repo_metadata_records)
        metadata_by_instance = {
            str(record["instance_id"]): record for record in repo_metadata_records
        }
        rows = attach_repo_metadata(rows, metadata_by_instance)

    rows = [enrich_row(row, args) for row in rows]

    full_rows = sorted(rows, key=lambda row: str(row["instance_id"]))
    leaderboard_rows = stratified_sample(full_rows, args.leaderboard_size, args.seed)
    smoke_rows = stratified_sample(full_rows, args.smoke_size, args.seed + 1)
    compact_subsets: dict[str, dict[str, Any]] = {}

    write_jsonl(artifact_root / "test.jsonl", full_rows)
    write_csv(artifact_root / "test.csv", full_rows)
    write_jsonl(subsets_dir / "full_public.jsonl", full_rows)
    write_csv(subsets_dir / "full_public.csv", full_rows)
    write_jsonl(subsets_dir / f"leaderboard_{args.leaderboard_size}.jsonl", leaderboard_rows)
    write_csv(subsets_dir / f"leaderboard_{args.leaderboard_size}.csv", leaderboard_rows)
    write_jsonl(subsets_dir / "smoke.jsonl", smoke_rows)
    write_csv(subsets_dir / "smoke.csv", smoke_rows)

    for subset_name, subset_size, seed_offset in (
        (f"leaderboard_compact_{args.compact_leaderboard_size}", args.compact_leaderboard_size, 101),
        (f"leaderboard_compact_{args.compact_pilot_size}", args.compact_pilot_size, 211),
    ):
        if subset_size <= 0:
            continue
        compact_rows, compact_metadata = compact_sample(
            full_rows,
            size=subset_size,
            seed=args.seed + seed_offset,
            max_cost_percentile=args.compact_max_cost_percentile,
            max_per_repo=args.compact_max_per_repo,
            max_tracked_total_bytes=args.compact_max_tracked_total_bytes,
            max_tracked_file_count=args.compact_max_tracked_file_count,
        )
        write_jsonl(subsets_dir / f"{subset_name}.jsonl", compact_rows)
        write_csv(subsets_dir / f"{subset_name}.csv", compact_rows)
        compact_subsets[subset_name] = compact_metadata

    try:
        from datasets import Dataset

        Dataset.from_list(full_rows).save_to_disk(str(artifact_root / "hf_dataset"))
    except Exception as exc:
        print(f"Warning: failed to save HF dataset directory: {exc}")

    manifest = {
        "dataset": args.dataset,
        "split": args.split,
        "dataset_dir_name": args.dataset_dir_name,
        "total_instances": len(full_rows),
        "leaderboard_size": len(leaderboard_rows),
        "smoke_size": len(smoke_rows),
        "compact_subset_sizes": {
            name: metadata["size"] for name, metadata in compact_subsets.items()
        },
        "seed": args.seed,
        "sampling": {
            "method": "deterministic stratified sample",
            "strata": ["repo_language", "issue_specificity", "first issue_categories"],
        },
        "compact_sampling": compact_subsets,
        "taiwan_recommended_runtime": {
            "subset": TAIWAN_RECOMMENDED_SUBSET,
            "max_input_tokens": TAIWAN_RECOMMENDED_MAX_INPUT_TOKENS,
            "max_tool_calls": TAIWAN_RECOMMENDED_MAX_TOOL_CALLS,
            "static_risk_control": "cost-capped compact subset with per-repo cap",
            "primary_cost_control": "runtime input-token and tool-call budgets",
        },
        "repo_static_cost_metadata": {
            "included": bool(repo_metadata_records),
            "schema_version": REPO_METADATA_SCHEMA_VERSION if repo_metadata_records else None,
            "measurement_method": REPO_TREE_METADATA_METHOD if repo_metadata_records else None,
            "path": "repo_metadata.jsonl" if (artifact_root / "repo_metadata.jsonl").exists() else None,
            "record_count": len(repo_metadata_records),
        },
        "agentic_prompt": {
            "code_embedded": False,
            "field": "agentic_prompt",
            "text_alias_code_embedded": False,
        },
        "offline_repo_artifact": {
            "included": False,
            "note": "Future on-prem mode should store repository checkouts or bare mirrors, not prompt-embedded source.",
        },
        "license_note": (
            "The source GitHub repository scaleapi/SWE-bench_Pro-os is MIT licensed; "
            "verify downstream use against the current dataset card before release."
        ),
        "source_urls": {
            "dataset": "https://huggingface.co/datasets/ScaleAI/SWE-bench_Pro",
            "code": "https://github.com/scaleapi/SWE-bench_Pro-os",
        },
        "files": {},
    }
    for path in sorted(artifact_root.rglob("*")):
        if path.is_file():
            manifest["files"][str(path.relative_to(artifact_root))] = {
                "bytes": path.stat().st_size,
                "sha256": sha256_file(path),
            }
    (artifact_root / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    write_readme(
        artifact_root / "README.md",
        full_rows,
        args.leaderboard_size,
        [metadata["size"] for metadata in compact_subsets.values()],
    )
    return artifact_root


def upload_artifact(args: argparse.Namespace, artifact_root: Path) -> None:
    if not args.entity or not args.project:
        raise ValueError("--entity and --project are required with --upload")
    wandb.login()
    with wandb.init(
        entity=args.entity,
        project=args.project,
        job_type="data-upload",
        name=f"{args.artifact_name}-upload",
    ) as run:
        artifact = wandb.Artifact(
            args.artifact_name,
            type="dataset",
            metadata={
                "dataset": args.dataset,
                "split": args.split,
                "dataset_dir_name": args.dataset_dir_name,
                "leaderboard_size": args.leaderboard_size,
                "smoke_size": args.smoke_size,
                "compact_leaderboard_size": args.compact_leaderboard_size,
                "compact_pilot_size": args.compact_pilot_size,
                "compact_max_cost_percentile": args.compact_max_cost_percentile,
                "repo_metadata_schema_version": REPO_METADATA_SCHEMA_VERSION
                if (artifact_root / "repo_metadata.jsonl").exists()
                else None,
                "seed": args.seed,
            },
        )
        artifact.add_dir(str(artifact_root), name=args.dataset_dir_name)
        run.log_artifact(artifact, aliases=["latest", "production"])


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", default=DEFAULT_DATASET)
    parser.add_argument("--split", default=DEFAULT_SPLIT)
    parser.add_argument("--output-dir", type=Path, default=Path("data/taiwan"))
    parser.add_argument("--dataset-dir-name", default=DEFAULT_DATASET_DIR_NAME)
    parser.add_argument("--artifact-name", default=DEFAULT_ARTIFACT_NAME)
    parser.add_argument("--leaderboard-size", type=int, default=80)
    parser.add_argument("--smoke-size", type=int, default=10)
    parser.add_argument(
        "--compact-leaderboard-size",
        type=int,
        default=0,
        help="Write subsets/leaderboard_compact_N using static repo cost metadata. 0 disables it.",
    )
    parser.add_argument(
        "--compact-pilot-size",
        type=int,
        default=0,
        help="Write a smaller compact pilot subset, typically 40. 0 disables it.",
    )
    parser.add_argument(
        "--compact-max-cost-percentile",
        type=float,
        default=DEFAULT_COMPACT_MAX_COST_PERCENTILE,
        help="Keep only instances at or below this static-cost percentile before compact sampling.",
    )
    parser.add_argument(
        "--compact-max-per-repo",
        type=int,
        default=0,
        help="Maximum instances per repo in compact subsets. 0 uses an automatic cap.",
    )
    parser.add_argument("--compact-max-tracked-total-bytes", type=int)
    parser.add_argument("--compact-max-tracked-file-count", type=int)
    parser.add_argument(
        "--repo-metadata-jsonl",
        type=Path,
        help="Existing repo metadata JSONL keyed by instance_id. Defaults to the artifact-local repo_metadata.jsonl.",
    )
    parser.add_argument(
        "--collect-repo-metadata",
        action="store_true",
        help="Clone/fetch repo mirrors and compute static repo tree metadata for all loaded rows.",
    )
    parser.add_argument(
        "--metadata-mirror-root",
        type=Path,
        default=Path("data/taiwan/swebench_pro_repo_mirrors"),
        help="Directory for bare Git mirrors used by --collect-repo-metadata.",
    )
    parser.add_argument(
        "--metadata-refresh",
        action="store_true",
        help="Run git remote update --prune for existing metadata mirrors.",
    )
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--limit", type=int, help="Debug-only row limit before sampling")
    parser.add_argument("--upload", action="store_true")
    parser.add_argument("--entity")
    parser.add_argument("--project")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.leaderboard_size <= 0:
        raise ValueError("--leaderboard-size must be positive")
    if args.smoke_size <= 0:
        raise ValueError("--smoke-size must be positive")
    if args.compact_leaderboard_size < 0:
        raise ValueError("--compact-leaderboard-size must be non-negative")
    if args.compact_pilot_size < 0:
        raise ValueError("--compact-pilot-size must be non-negative")
    if not 0 < args.compact_max_cost_percentile <= 1:
        raise ValueError("--compact-max-cost-percentile must be in (0, 1]")
    artifact_root = materialize(args)
    print(artifact_root)
    if args.upload:
        upload_artifact(args, artifact_root)


if __name__ == "__main__":
    main()
