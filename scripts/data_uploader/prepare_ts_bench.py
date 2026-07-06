#!/usr/bin/env python3
"""Materialize the official TS-Bench dataset for Nejumi Taiwan."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import subprocess
from pathlib import Path
from typing import Any, Iterable

import wandb


DEFAULT_ARTIFACT_NAME = "ts-bench"
DEFAULT_DATASET_DIR_NAME = "ts_bench"
DEFAULT_SOURCE_DIR = Path("external/TS-Bench")


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


def source_commit(source_dir: Path) -> str | None:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=source_dir,
            text=True,
            capture_output=True,
            check=False,
        )
    except FileNotFoundError:
        return None
    return result.stdout.strip() if result.returncode == 0 else None


def read_rows(csv_path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with csv_path.open(encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        required = {"id", "message", "label", "split"}
        if not required.issubset(reader.fieldnames or []):
            raise ValueError(f"{csv_path} must contain columns {sorted(required)}")
        for row in reader:
            rows.append(
                {
                    "id": int(row["id"]),
                    "message": row["message"],
                    "label": int(row["label"]),
                    "split": row["split"],
                    "source": "MediaTek-Research/TS-Bench",
                }
            )
    rows.sort(key=lambda row: row["id"])
    return rows


def materialize(args: argparse.Namespace) -> Path:
    source_dir = args.source_dir.expanduser()
    source_csv = source_dir / "data" / "TSB400.csv"
    source_license = source_dir / "LICENSE"
    if not source_csv.exists():
        raise FileNotFoundError(source_csv)
    if not source_license.exists():
        raise FileNotFoundError(source_license)

    artifact_root = args.output_dir / args.dataset_dir_name
    artifact_root.mkdir(parents=True, exist_ok=True)

    rows = read_rows(source_csv)
    csv_out = artifact_root / "TSB400.csv"
    jsonl_out = artifact_root / "ts_bench.jsonl"
    license_out = artifact_root / "LICENSE"
    csv_out.write_bytes(source_csv.read_bytes())
    license_out.write_text(source_license.read_text(encoding="utf-8"), encoding="utf-8")
    write_jsonl(jsonl_out, rows)

    manifest = {
        "dataset": "TS-Bench",
        "dataset_dir_name": args.dataset_dir_name,
        "total_instances": len(rows),
        "unsafe_count": sum(1 for row in rows if row["label"] == 1),
        "safe_count": sum(1 for row in rows if row["label"] == 0),
        "source_repo": "https://github.com/mtkresearch/TS-Bench",
        "source_commit": source_commit(source_dir),
        "paper": "https://arxiv.org/abs/2603.07286",
        "license": "Apache-2.0",
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
    (artifact_root / "README.md").write_text(
        "\n".join(
            [
                "# TS-Bench",
                "",
                "Official Taiwan Safety Benchmark artifact for Nejumi Taiwan.",
                "",
                "Task: binary safety classification, where `label=1` means unsafe",
                "and `label=0` means safe.",
                "",
                f"Rows: {len(rows)}",
                "License: Apache-2.0",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    return artifact_root


def upload_artifact(args: argparse.Namespace, artifact_root: Path) -> None:
    run = wandb.init(
        entity=args.entity,
        project=args.project,
        job_type="upload-ts-bench",
    )
    artifact = wandb.Artifact(
        args.artifact_name,
        type="dataset",
        description="TS-Bench Taiwan Safety Benchmark for Nejumi Taiwan",
        metadata={
            "source_repo": "https://github.com/mtkresearch/TS-Bench",
            "source_commit": source_commit(args.source_dir.expanduser()),
            "license": "Apache-2.0",
            "dataset_dir": args.dataset_dir_name,
        },
    )
    artifact.add_dir(str(artifact_root), name=args.dataset_dir_name)
    run.log_artifact(artifact, aliases=["production"])
    run.finish()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Prepare TS-Bench W&B artifact")
    parser.add_argument("--source-dir", type=Path, default=DEFAULT_SOURCE_DIR)
    parser.add_argument("--output-dir", type=Path, default=Path("data/taiwan"))
    parser.add_argument("--dataset-dir-name", default=DEFAULT_DATASET_DIR_NAME)
    parser.add_argument("--artifact-name", default=DEFAULT_ARTIFACT_NAME)
    parser.add_argument("--entity", default="llm-leaderboard")
    parser.add_argument("--project", default="tc-leaderboard")
    parser.add_argument("--upload", action="store_true")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    artifact_root = materialize(args)
    print(artifact_root)
    if args.upload:
        upload_artifact(args, artifact_root)


if __name__ == "__main__":
    main()
