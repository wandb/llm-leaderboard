#!/usr/bin/env python3
"""
Materialize Agentic Math data for Nejumi Taiwan.

Default source:

  opencompass/AIME2025

The output artifact layout is:

  agentic_math_aime2025/
    test.jsonl
    manifest.json
    subsets/
      smoke.jsonl
      leaderboard.jsonl
      full.jsonl
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Any, Iterable

import wandb


DEFAULT_DATASET = "opencompass/AIME2025"
DEFAULT_CONFIGS = ["AIME2025-I", "AIME2025-II"]
DEFAULT_SPLIT = "test"
DEFAULT_ARTIFACT_NAME = "agentic-math-aime2025"
DEFAULT_DATASET_DIR_NAME = "agentic_math_aime2025"
DEFAULT_SMOKE_SIZE = 3
DATASETS_SERVER = "https://datasets-server.huggingface.co"
CSV_FIELDS = [
    "task_id",
    "benchmark",
    "source_dataset",
    "source_config",
    "split",
    "problem_index",
    "question",
    "answer_raw",
    "answer",
    "answer_int",
]


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


def load_hf_rows(dataset: str, config: str, split: str, max_retries: int = 5) -> list[dict[str, Any]]:
    try:
        from datasets import load_dataset

        loaded = load_dataset(dataset, config, split=split)
        return [dict(row) for row in loaded]
    except Exception as exc:
        print(f"Falling back to Hugging Face Dataset Viewer for {dataset}/{config}/{split}: {exc}")

    rows: list[dict[str, Any]] = []
    offset = 0
    length = 100
    while True:
        query = urllib.parse.urlencode(
            {
                "dataset": dataset,
                "config": config,
                "split": split,
                "offset": offset,
                "length": length,
            }
        )
        url = f"{DATASETS_SERVER}/rows?{query}"
        payload = None
        for attempt in range(max_retries):
            try:
                request = urllib.request.Request(
                    url,
                    headers={"User-Agent": "nejumi-agentic-math-builder/0.1"},
                )
                with urllib.request.urlopen(request, timeout=60) as response:
                    payload = json.load(response)
                break
            except urllib.error.HTTPError as exc:
                if exc.code not in {429, 500, 502, 503, 504} or attempt == max_retries - 1:
                    raise
                sleep_seconds = min(2**attempt, 20)
                time.sleep(sleep_seconds)
        if payload is None:
            raise RuntimeError(f"Failed to fetch rows for {dataset}/{config}/{split}")
        batch = [item["row"] for item in payload.get("rows", [])]
        rows.extend(batch)
        total = payload.get("num_rows_total")
        if not batch or total is None or len(rows) >= total:
            break
        offset += len(batch)
    return rows


def normalize_answer(value: Any) -> tuple[str, int]:
    text = str(value).strip()
    match = re.search(r"\b([0-9]{1,3})\b", text)
    if not match:
        raise ValueError(f"AIME answer must contain an integer 0..999, got {value!r}")
    answer_int = int(match.group(1))
    if not 0 <= answer_int <= 999:
        raise ValueError(f"AIME answer out of range 0..999: {answer_int}")
    return str(answer_int), answer_int


def task_prefix(config: str) -> str:
    if config.endswith("-I"):
        return "aime2025_i"
    if config.endswith("-II"):
        return "aime2025_ii"
    safe = "".join(char.lower() if char.isalnum() else "_" for char in config)
    return safe.strip("_")


def build_rows(args: argparse.Namespace) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for config in args.configs:
        source_rows = load_hf_rows(args.dataset, config, args.split)
        prefix = task_prefix(config)
        for index, row in enumerate(source_rows, start=1):
            answer, answer_int = normalize_answer(row["answer"])
            rows.append(
                {
                    "task_id": f"{prefix}_{index:02d}",
                    "benchmark": "AIME2025",
                    "source_dataset": args.dataset,
                    "source_config": config,
                    "split": args.split,
                    "problem_index": index,
                    "question": str(row["question"]).strip(),
                    "answer_raw": str(row["answer"]).strip(),
                    "answer": answer,
                    "answer_int": answer_int,
                }
            )
    rows.sort(key=lambda row: row["task_id"])
    if args.limit is not None:
        rows = rows[: args.limit]
    return rows


def write_readme(path: Path, rows: list[dict[str, Any]], smoke_size: int) -> None:
    path.write_text(
        "\n".join(
            [
                "# Agentic Math AIME 2025 for Nejumi Taiwan",
                "",
                "This artifact contains AIME 2025 I and II from `opencompass/AIME2025`",
                "for agentic mathematical reasoning evaluation.",
                "",
                "## Files",
                "",
                "- `test.jsonl`: full 30-problem set",
                "- `subsets/smoke.jsonl`: small harness check subset",
                "- `subsets/leaderboard.jsonl`: full leaderboard subset",
                "- `manifest.json`: source, license, counts, and hashes",
                "",
                "## Scoring",
                "",
                "Each answer is scored by exact integer equality after extracting",
                "`ANSWER: <integer>` from the OpenClaw response.",
                "",
                f"Rows: {len(rows)}",
                f"Smoke rows: {min(smoke_size, len(rows))}",
            ]
        )
        + "\n",
        encoding="utf-8",
    )


def materialize(args: argparse.Namespace) -> Path:
    rows = build_rows(args)
    if not rows:
        raise RuntimeError("No Agentic Math rows were loaded")

    artifact_root = args.output_dir / args.dataset_dir_name
    subsets_dir = artifact_root / "subsets"
    artifact_root.mkdir(parents=True, exist_ok=True)
    subsets_dir.mkdir(parents=True, exist_ok=True)

    smoke_rows = rows[: min(args.smoke_size, len(rows))]
    write_jsonl(artifact_root / "test.jsonl", rows)
    write_csv(artifact_root / "test.csv", rows)
    write_jsonl(subsets_dir / "full.jsonl", rows)
    write_csv(subsets_dir / "full.csv", rows)
    write_jsonl(subsets_dir / "leaderboard.jsonl", rows)
    write_csv(subsets_dir / "leaderboard.csv", rows)
    write_jsonl(subsets_dir / "smoke.jsonl", smoke_rows)
    write_csv(subsets_dir / "smoke.csv", smoke_rows)

    manifest: dict[str, Any] = {
        "dataset": args.dataset,
        "configs": args.configs,
        "split": args.split,
        "dataset_dir_name": args.dataset_dir_name,
        "total_instances": len(rows),
        "smoke_size": len(smoke_rows),
        "license": "mit",
        "source_urls": {
            "dataset": "https://huggingface.co/datasets/opencompass/AIME2025",
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
    write_readme(artifact_root / "README.md", rows, args.smoke_size)
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
                "configs": args.configs,
                "split": args.split,
                "dataset_dir_name": args.dataset_dir_name,
                "smoke_size": args.smoke_size,
            },
        )
        artifact.add_dir(str(artifact_root), name=args.dataset_dir_name)
        run.log_artifact(artifact, aliases=["latest", "production"])


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", default=DEFAULT_DATASET)
    parser.add_argument("--configs", nargs="+", default=DEFAULT_CONFIGS)
    parser.add_argument("--split", default=DEFAULT_SPLIT)
    parser.add_argument("--output-dir", type=Path, default=Path("data/taiwan"))
    parser.add_argument("--dataset-dir-name", default=DEFAULT_DATASET_DIR_NAME)
    parser.add_argument("--artifact-name", default=DEFAULT_ARTIFACT_NAME)
    parser.add_argument("--smoke-size", type=int, default=DEFAULT_SMOKE_SIZE)
    parser.add_argument("--limit", type=int, help="Debug-only row limit before writing")
    parser.add_argument("--upload", action="store_true")
    parser.add_argument("--entity")
    parser.add_argument("--project")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.smoke_size <= 0:
        raise ValueError("--smoke-size must be positive")
    artifact_root = materialize(args)
    print(artifact_root)
    if args.upload:
        upload_artifact(args, artifact_root)


if __name__ == "__main__":
    main()
