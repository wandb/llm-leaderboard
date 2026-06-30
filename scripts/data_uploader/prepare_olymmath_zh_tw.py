#!/usr/bin/env python3
"""
Materialize OlymMATH-HARD zh-TW data for Nejumi Taiwan Agentic Math.

The output artifact layout is:

  agentic_math_olymmath_hard_zh_tw/
    test.jsonl
    manifest.json
    subsets/
      smoke.jsonl
      leaderboard.jsonl
      leaderboard_40.jsonl
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


DATASETS_SERVER = "https://datasets-server.huggingface.co"
DEFAULT_DATASET = "RUC-AIBOX/OlymMATH"
DEFAULT_ZH_CONFIG = "zh-hard"
DEFAULT_EN_CONFIG = "en-hard"
DEFAULT_SPLIT = "test"
DEFAULT_ARTIFACT_NAME = "agentic-math-olymmath-hard-zh-tw"
DEFAULT_DATASET_DIR_NAME = "agentic_math_olymmath_hard_zh_tw"
DEFAULT_SMOKE_SIZE = 4
DEFAULT_LEADERBOARD40_SIZE = 40
CSV_FIELDS = [
    "task_id",
    "benchmark",
    "source_dataset",
    "source_config",
    "split",
    "problem_index",
    "unique_id",
    "subject",
    "subject_zh_hans",
    "subject_en",
    "question",
    "question_zh_hans",
    "question_en",
    "answer_raw",
    "answer",
    "answer_format",
    "scoring",
]
SUBJECT_ZH_TW_TO_EN = {
    "代數": "Algebra",
    "幾何": "Geometry",
    "組合": "Combinatorics",
    "數論": "Number Theory",
}


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
                    headers={"User-Agent": "nejumi-olymmath-zh-tw-builder/0.1"},
                )
                with urllib.request.urlopen(request, timeout=60) as response:
                    payload = json.load(response)
                break
            except urllib.error.HTTPError as exc:
                if exc.code not in {429, 500, 502, 503, 504} or attempt == max_retries - 1:
                    raise
                time.sleep(min(2**attempt, 20))
        if payload is None:
            raise RuntimeError(f"Failed to fetch rows for {dataset}/{config}/{split}")
        batch = [item["row"] for item in payload.get("rows", [])]
        rows.extend(batch)
        total = payload.get("num_rows_total")
        if not batch or total is None or len(rows) >= total:
            break
        offset += len(batch)
    return rows


def opencc_converter() -> Any:
    try:
        from opencc import OpenCC
    except Exception as exc:
        raise RuntimeError(
            "opencc-python-reimplemented is required to build OlymMATH zh-TW. "
            "Install the project dependencies with `uv sync`."
        ) from exc

    for config_name in ("s2twp", "s2tw", "s2t"):
        try:
            return OpenCC(config_name)
        except Exception:
            continue
    raise RuntimeError("OpenCC is installed but no s2twp/s2tw/s2t converter is available")


def to_task_id(unique_id: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", unique_id.lower()).strip("_")


def pair_key(unique_id: str) -> str:
    return re.sub(r"-(ZH|EN)$", "", unique_id, flags=re.IGNORECASE)


def stratified_rows(rows: list[dict[str, Any]], size: int) -> list[dict[str, Any]]:
    if size >= len(rows):
        return list(rows)
    by_subject: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        by_subject.setdefault(str(row.get("subject", "")), []).append(row)
    selected: list[dict[str, Any]] = []
    subjects = sorted(by_subject)
    index = 0
    while len(selected) < size:
        made_progress = False
        for subject in subjects:
            bucket = by_subject[subject]
            if index < len(bucket):
                selected.append(bucket[index])
                made_progress = True
                if len(selected) >= size:
                    break
        if not made_progress:
            break
        index += 1
    return sorted(selected, key=lambda row: int(row["problem_index"]))


def build_rows(args: argparse.Namespace) -> list[dict[str, Any]]:
    converter = opencc_converter()
    zh_rows = load_hf_rows(args.dataset, args.zh_config, args.split)
    en_rows = load_hf_rows(args.dataset, args.en_config, args.split) if args.include_english else []
    en_by_key = {pair_key(str(row["unique_id"])): row for row in en_rows if row.get("unique_id")}
    rows: list[dict[str, Any]] = []
    for index, zh_row in enumerate(zh_rows, start=1):
        unique_id = str(zh_row["unique_id"])
        english = en_by_key.get(pair_key(unique_id), {})
        question_zh_hans = str(zh_row["problem"]).strip()
        subject_zh_hans = str(zh_row["subject"]).strip()
        subject_zh_tw = converter.convert(subject_zh_hans)
        answer = str(zh_row["answer"]).strip()
        rows.append(
            {
                "task_id": to_task_id(unique_id),
                "benchmark": "OlymMATH-HARD",
                "source_dataset": args.dataset,
                "source_config": args.zh_config,
                "split": args.split,
                "problem_index": index,
                "unique_id": unique_id,
                "subject": subject_zh_tw,
                "subject_zh_hans": subject_zh_hans,
                "subject_en": str(english.get("subject") or SUBJECT_ZH_TW_TO_EN.get(subject_zh_tw, "")),
                "question": converter.convert(question_zh_hans),
                "question_zh_hans": question_zh_hans,
                "question_en": str(english.get("problem", "")).strip(),
                "answer_raw": answer,
                "answer": answer,
                "answer_format": "math_expression",
                "scoring": "symbolic_equivalence",
            }
        )
    rows.sort(key=lambda row: int(row["problem_index"]))
    if args.limit is not None:
        rows = rows[: args.limit]
    return rows


def write_readme(path: Path, rows: list[dict[str, Any]], args: argparse.Namespace) -> None:
    path.write_text(
        "\n".join(
            [
                "# Agentic Math OlymMATH-HARD zh-TW for Nejumi Taiwan",
                "",
                "This artifact contains the OlymMATH-HARD Chinese subset from",
                "`RUC-AIBOX/OlymMATH`, converted from Simplified Chinese to",
                "Traditional Chinese with OpenCC (`s2twp`/`s2tw` fallback).",
                "",
                "AIME 2025 remains useful as an OpenClaw smoke test, but this",
                "OlymMATH-HARD artifact is the stronger leaderboard candidate",
                "because it has harder symbolic final-answer math problems.",
                "",
                "## Files",
                "",
                "- `test.jsonl`: full 100-problem OlymMATH-HARD zh-TW set",
                "- `subsets/smoke.jsonl`: stratified harness check subset",
                "- `subsets/leaderboard.jsonl`: full leaderboard subset",
                "- `subsets/leaderboard_40.jsonl`: balanced lower-cost pilot subset",
                "- `subsets/full.jsonl`: alias for the full set",
                "- `manifest.json`: source, license, counts, and hashes",
                "",
                "## Scoring",
                "",
                "The OpenClaw runner extracts an `ANSWER:` line and scores it",
                "with the official OlymMATH evaluator's Math-Verify direction:",
                "`math_verify.parse` / `math_verify.verify` first, followed by",
                "exact text normalization, SymPy, and conservative string",
                "comparison fallbacks for API/OpenClaw output format differences.",
                "",
                f"Rows: {len(rows)}",
                f"Smoke rows: {min(args.smoke_size, len(rows))}",
                f"Leaderboard-40 rows: {min(args.leaderboard40_size, len(rows))}",
            ]
        )
        + "\n",
        encoding="utf-8",
    )


def materialize(args: argparse.Namespace) -> Path:
    rows = build_rows(args)
    if not rows:
        raise RuntimeError("No OlymMATH rows were loaded")

    artifact_root = args.output_dir / args.dataset_dir_name
    subsets_dir = artifact_root / "subsets"
    artifact_root.mkdir(parents=True, exist_ok=True)
    subsets_dir.mkdir(parents=True, exist_ok=True)

    smoke_rows = stratified_rows(rows, min(args.smoke_size, len(rows)))
    leaderboard40_rows = stratified_rows(rows, min(args.leaderboard40_size, len(rows)))
    write_jsonl(artifact_root / "test.jsonl", rows)
    write_csv(artifact_root / "test.csv", rows)
    write_jsonl(subsets_dir / "full.jsonl", rows)
    write_csv(subsets_dir / "full.csv", rows)
    write_jsonl(subsets_dir / "leaderboard.jsonl", rows)
    write_csv(subsets_dir / "leaderboard.csv", rows)
    write_jsonl(subsets_dir / "leaderboard_40.jsonl", leaderboard40_rows)
    write_csv(subsets_dir / "leaderboard_40.csv", leaderboard40_rows)
    write_jsonl(subsets_dir / "smoke.jsonl", smoke_rows)
    write_csv(subsets_dir / "smoke.csv", smoke_rows)

    manifest: dict[str, Any] = {
        "dataset": args.dataset,
        "zh_config": args.zh_config,
        "en_config": args.en_config if args.include_english else None,
        "split": args.split,
        "dataset_dir_name": args.dataset_dir_name,
        "total_instances": len(rows),
        "smoke_size": len(smoke_rows),
        "leaderboard40_size": len(leaderboard40_rows),
        "license": "mit",
        "language": "zh-TW",
        "script_conversion": "OpenCC s2twp with s2tw/s2t fallback",
        "source_urls": {
            "dataset": "https://huggingface.co/datasets/RUC-AIBOX/OlymMATH",
            "paper": "https://arxiv.org/abs/2503.21380",
            "repository": "https://github.com/RUCAIBox/OlymMATH",
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
    write_readme(artifact_root / "README.md", rows, args)
    return artifact_root


def upload_artifact(args: argparse.Namespace, artifact_root: Path) -> None:
    if not args.entity or not args.project:
        raise ValueError("--entity and --project are required with --upload")
    with (artifact_root / "test.jsonl").open(encoding="utf-8") as f:
        total_instances = sum(1 for line in f if line.strip())
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
                "zh_config": args.zh_config,
                "split": args.split,
                "dataset_dir_name": args.dataset_dir_name,
                "total_instances": total_instances,
                "license": "mit",
            },
        )
        artifact.add_dir(str(artifact_root), name=args.dataset_dir_name)
        run.log_artifact(artifact, aliases=["latest", "production"])


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", default=DEFAULT_DATASET)
    parser.add_argument("--zh-config", default=DEFAULT_ZH_CONFIG)
    parser.add_argument("--en-config", default=DEFAULT_EN_CONFIG)
    parser.add_argument("--split", default=DEFAULT_SPLIT)
    parser.add_argument("--output-dir", type=Path, default=Path("data/taiwan"))
    parser.add_argument("--dataset-dir-name", default=DEFAULT_DATASET_DIR_NAME)
    parser.add_argument("--artifact-name", default=DEFAULT_ARTIFACT_NAME)
    parser.add_argument("--smoke-size", type=int, default=DEFAULT_SMOKE_SIZE)
    parser.add_argument("--leaderboard40-size", type=int, default=DEFAULT_LEADERBOARD40_SIZE)
    parser.add_argument("--limit", type=int, help="Debug-only row limit before writing")
    parser.add_argument("--include-english", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--upload", action="store_true")
    parser.add_argument("--entity")
    parser.add_argument("--project")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.smoke_size <= 0:
        raise ValueError("--smoke-size must be positive")
    if args.leaderboard40_size <= 0:
        raise ValueError("--leaderboard40-size must be positive")
    artifact_root = materialize(args)
    print(artifact_root)
    if args.upload:
        upload_artifact(args, artifact_root)


if __name__ == "__main__":
    main()
