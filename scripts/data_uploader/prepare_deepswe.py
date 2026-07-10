#!/usr/bin/env python3
"""Prepare fixed DeepSWE subset manifests for the Taiwan leaderboard."""

from __future__ import annotations

import argparse
import hashlib
import json
import random
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

try:
    import tomllib  # type: ignore
except ModuleNotFoundError:
    import tomli as tomllib  # type: ignore


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_TASKS_ROOT = REPO_ROOT / "external" / "deep-swe" / "tasks"
DEFAULT_OUTPUT_DIR = REPO_ROOT / "data" / "taiwan" / "deepswe"
DEFAULT_LEADERBOARD_COUNTS = {
    "typescript": 16,
    "go": 15,
    "python": 15,
    "rust": 2,
    "javascript": 2,
}
DEFAULT_PILOT_COUNTS = {
    "typescript": 5,
    "go": 5,
    "python": 4,
    "rust": 1,
    "javascript": 1,
}


def read_task(path: Path) -> dict[str, Any]:
    with (path / "task.toml").open("rb") as f:
        payload = tomllib.load(f)
    metadata = payload.get("metadata") if isinstance(payload.get("metadata"), dict) else {}
    environment = (
        payload.get("environment") if isinstance(payload.get("environment"), dict) else {}
    )
    task_id = str(metadata.get("task_id") or path.name)
    return {
        "task_name": path.name,
        "task_id": task_id,
        "language": str(metadata.get("language") or ""),
        "repo_name": str(metadata.get("repo_name") or ""),
        "repository_url": str(metadata.get("repo_url") or ""),
        "base_commit": str(metadata.get("base_commit") or ""),
        "docker_image": str(environment.get("docker_image") or ""),
        "task_path": str(path.relative_to(REPO_ROOT)),
    }


def stable_key(row: dict[str, Any], seed: int) -> str:
    material = f"{seed}\n{row['task_name']}\n{row['repository_url']}\n{row['base_commit']}"
    return hashlib.sha256(material.encode("utf-8")).hexdigest()


def sample_by_language(
    rows: list[dict[str, Any]],
    counts: dict[str, int],
    *,
    seed: int,
) -> list[dict[str, Any]]:
    by_language: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_language[row["language"]].append(row)

    selected: list[dict[str, Any]] = []
    for language, count in counts.items():
        candidates = sorted(by_language.get(language, []), key=lambda row: stable_key(row, seed))
        if len(candidates) < count:
            raise ValueError(
                f"Not enough DeepSWE tasks for language={language}: "
                f"need {count}, found {len(candidates)}"
            )

        repo_buckets: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for row in candidates:
            repo_buckets[row["repo_name"] or row["repository_url"]].append(row)
        for bucket in repo_buckets.values():
            bucket.sort(key=lambda row: stable_key(row, seed))

        # Round-robin repositories to avoid an accidental single-repo-heavy sample.
        repo_order = sorted(repo_buckets, key=lambda repo: stable_key({"task_name": repo, "repository_url": repo, "base_commit": ""}, seed))
        language_rows: list[dict[str, Any]] = []
        while len(language_rows) < count:
            made_progress = False
            for repo in repo_order:
                if repo_buckets[repo]:
                    language_rows.append(repo_buckets[repo].pop(0))
                    made_progress = True
                    if len(language_rows) >= count:
                        break
            if not made_progress:
                break
        if len(language_rows) != count:
            raise ValueError(f"Failed to sample {count} tasks for {language}")
        selected.extend(language_rows)

    selected.sort(key=lambda row: stable_key(row, seed + 1))
    return selected


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tasks-root", type=Path, default=DEFAULT_TASKS_ROOT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--seed", type=int, default=20260711)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    tasks_root = args.tasks_root.resolve()
    output_dir = args.output_dir.resolve()
    task_dirs = sorted(path for path in tasks_root.iterdir() if (path / "task.toml").exists())
    rows = [read_task(path) for path in task_dirs]
    if not rows:
        raise SystemExit(f"No DeepSWE tasks found under {tasks_root}")

    leaderboard = sample_by_language(rows, DEFAULT_LEADERBOARD_COUNTS, seed=args.seed)
    pilot = sample_by_language(rows, DEFAULT_PILOT_COUNTS, seed=args.seed)

    subsets = {
        "full": rows,
        "leaderboard_50": leaderboard,
        "pilot_16": pilot,
    }
    for name, subset_rows in subsets.items():
        write_jsonl(output_dir / "subsets" / f"{name}.jsonl", subset_rows)
        write_json(
            output_dir / "subsets" / f"{name}_task_names.json",
            [row["task_name"] for row in subset_rows],
        )

    manifest = {
        "benchmark": "DeepSWE",
        "source": str(tasks_root.relative_to(REPO_ROOT)),
        "source_task_count": len(rows),
        "seed": args.seed,
        "language_distribution": dict(sorted(Counter(row["language"] for row in rows).items())),
        "subsets": {
            name: {
                "count": len(subset_rows),
                "language_distribution": dict(
                    sorted(Counter(row["language"] for row in subset_rows).items())
                ),
                "task_names_path": f"subsets/{name}_task_names.json",
                "metadata_jsonl_path": f"subsets/{name}.jsonl",
            }
            for name, subset_rows in subsets.items()
        },
    }
    write_json(output_dir / "manifest.json", manifest)
    (output_dir / "README.md").write_text(
        "# DeepSWE Taiwan subset manifest\n\n"
        "This directory contains fixed DeepSWE task-name subsets for the Taiwan "
        "leaderboard. It does not copy task source files; Pier reads the canonical "
        "`external/deep-swe/tasks` tree at evaluation time.\n",
        encoding="utf-8",
    )
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
