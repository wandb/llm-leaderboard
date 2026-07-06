#!/usr/bin/env python3
"""Materialize TWBias data for Nejumi Taiwan.

TWBias upstream currently ships data in the GitHub repository, but no license
file is present. The artifact records that status explicitly so release
packaging can decide whether to publish or keep the dataset internal.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import subprocess
from pathlib import Path

import wandb


DEFAULT_ARTIFACT_NAME = "twbias"
DEFAULT_DATASET_DIR_NAME = "twbias"
DEFAULT_SOURCE_DIR = Path("external/TWBias")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


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


def copy_tree(src: Path, dst: Path) -> None:
    if dst.exists():
        shutil.rmtree(dst)
    shutil.copytree(src, dst)


def materialize(args: argparse.Namespace) -> Path:
    source_dir = args.source_dir.expanduser()
    data_dir = source_dir / "data"
    prompts_path = source_dir / "prompts.json"
    readme_path = source_dir / "README.md"
    if not data_dir.exists():
        raise FileNotFoundError(data_dir)
    if not prompts_path.exists():
        raise FileNotFoundError(prompts_path)

    artifact_root = args.output_dir / args.dataset_dir_name
    artifact_root.mkdir(parents=True, exist_ok=True)
    copy_tree(data_dir, artifact_root / "data")
    shutil.copy2(prompts_path, artifact_root / "prompts.json")
    if readme_path.exists():
        shutil.copy2(readme_path, artifact_root / "UPSTREAM_README.md")

    files = {}
    for path in sorted(artifact_root.rglob("*")):
        if path.is_file():
            files[str(path.relative_to(artifact_root))] = {
                "bytes": path.stat().st_size,
                "sha256": sha256_file(path),
            }

    manifest = {
        "dataset": "TWBias",
        "dataset_dir_name": args.dataset_dir_name,
        "source_repo": "https://github.com/hsinmosyi/TWBias",
        "source_commit": source_commit(source_dir),
        "paper": "https://aclanthology.org/2024.findings-emnlp.507/",
        "license": "unknown",
        "license_note": "No LICENSE file was present in hsinmosyi/TWBias at materialization time.",
        "files": files,
    }
    (artifact_root / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    (artifact_root / "README.md").write_text(
        "\n".join(
            [
                "# TWBias",
                "",
                "Official TWBias data materialized for Nejumi Taiwan.",
                "",
                "Important: upstream license was not found in the GitHub repository.",
                "Keep this artifact internal until distribution rights are confirmed.",
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
        job_type="upload-twbias",
    )
    artifact = wandb.Artifact(
        args.artifact_name,
        type="dataset",
        description="TWBias data for Nejumi Taiwan",
        metadata={
            "source_repo": "https://github.com/hsinmosyi/TWBias",
            "source_commit": source_commit(args.source_dir.expanduser()),
            "license": "unknown",
            "dataset_dir": args.dataset_dir_name,
        },
    )
    artifact.add_dir(str(artifact_root), name=args.dataset_dir_name)
    run.log_artifact(artifact, aliases=["production"])
    run.finish()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Prepare TWBias W&B artifact")
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
