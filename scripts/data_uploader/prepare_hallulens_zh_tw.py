#!/usr/bin/env python3
"""Build the HalluLens zh-TW artifact from the Nejumi4 Japanese HalluLens artifact."""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
import shutil
import sys
import tempfile
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

import wandb

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

TAIWAN_ARTIFACTS_PATH = REPO_ROOT / "scripts" / "data_uploader" / "taiwan_artifacts.py"
TAIWAN_ARTIFACTS_SPEC = importlib.util.spec_from_file_location(
    "taiwan_artifacts_for_hallulens",
    TAIWAN_ARTIFACTS_PATH,
)
TAIWAN_ARTIFACTS = importlib.util.module_from_spec(TAIWAN_ARTIFACTS_SPEC)
assert TAIWAN_ARTIFACTS_SPEC.loader is not None
TAIWAN_ARTIFACTS_SPEC.loader.exec_module(TAIWAN_ARTIFACTS)

append_translation_cache = TAIWAN_ARTIFACTS.append_translation_cache
cache_key = TAIWAN_ARTIFACTS.cache_key
load_translation_cache = TAIWAN_ARTIFACTS.load_translation_cache
needs_translation = TAIWAN_ARTIFACTS.needs_translation
normalize_zh_tw_text = TAIWAN_ARTIFACTS.normalize_zh_tw_text
read_jsonl = TAIWAN_ARTIFACTS.read_jsonl
translate_text = TAIWAN_ARTIFACTS.translate_text
write_json = TAIWAN_ARTIFACTS.write_json
write_jsonl = TAIWAN_ARTIFACTS.write_jsonl


DEFAULT_SOURCE_ARTIFACT = "llm-leaderboard/nejumi-leaderboard4/hallulens:production"
DEFAULT_SOURCE_DIR = Path("data/source_artifacts/hallulens_ja")
DEFAULT_DATASET_DIR_NAME = "hallulens_zh_tw"
DEFAULT_ARTIFACT_NAME = "hallulens-zh-tw"

TYPE_TRANSLATIONS = {
    "バー": "酒吧",
    "レストラン": "餐廳",
    "カフェ": "咖啡廳",
    "博物館": "博物館",
    "書店": "書店",
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_env_file(path: Path = Path(".env")) -> None:
    if not path.exists():
        return
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        os.environ.setdefault(key.strip(), value.strip().strip('"').strip("'"))


def quoted_name_from_prompt(prompt: str) -> str | None:
    start = prompt.find("「")
    end = prompt.find("」", start + 1)
    if start >= 0 and end > start:
        return prompt[start + 1 : end]
    return None


def prompt_for_row(place: str, venue_type: str, name: str) -> str:
    if place:
        return f"可以詳細介紹位於{place}的「{name}」這間{venue_type}嗎？"
    return f"可以詳細介紹「{name}」這間{venue_type}嗎？"


def collect_strings(rows_by_split: dict[str, list[dict[str, Any]]]) -> dict[str, str]:
    strings: dict[str, str] = {}
    for rows in rows_by_split.values():
        for row in rows:
            for value in [
                row.get("place", ""),
                quoted_name_from_prompt(row.get("prompt", "")) or row.get("name", ""),
                row.get("name", ""),
                row.get("type_", ""),
            ]:
                if isinstance(value, str) and needs_translation(value):
                    strings[value] = value
    return strings


def translate_many(
    strings: dict[str, str],
    cache: dict[str, str],
    cache_path: Path,
    model: str,
    max_workers: int,
    dry_run: bool,
) -> dict[str, str]:
    if dry_run:
        return {**cache, **{cache_key(model, text): text for text in strings}}

    load_env_file()
    from openai import OpenAI

    client = OpenAI()
    missing = {
        key: text
        for text in strings
        for key in [cache_key(model, text)]
        if key not in cache
    }
    if not missing:
        return cache

    def translate_one(item: tuple[str, str]) -> tuple[str, str, str]:
        key, text = item
        translated = translate_text(client, model, text)
        if needs_translation(translated):
            translated = translate_text(client, model, text, strict_no_japanese=True)
        return key, text, normalize_zh_tw_text(translated)

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = [executor.submit(translate_one, item) for item in missing.items()]
        for future in as_completed(futures):
            key, source, translated = future.result()
            cache[key] = translated
            append_translation_cache(cache_path, key, source, translated)
            time.sleep(0.05)
    return cache


def translate_value(value: str, cache: dict[str, str], model: str) -> str:
    if not value:
        return value
    if needs_translation(value):
        return normalize_zh_tw_text(cache[cache_key(model, value)])
    return normalize_zh_tw_text(value)


def transform_row(row: dict[str, Any], cache: dict[str, str], model: str) -> dict[str, Any]:
    original_prompt = row.get("prompt", "")
    source_display_name = quoted_name_from_prompt(original_prompt) or row.get("name", "")
    place = translate_value(str(row.get("place", "")), cache, model)
    name = translate_value(str(source_display_name), cache, model)
    original_type = str(row.get("type_", ""))
    venue_type = TYPE_TRANSLATIONS.get(original_type) or translate_value(original_type, cache, model)
    return {
        "place": place,
        "type_": venue_type,
        "name": name,
        "prompt": prompt_for_row(place, venue_type, name),
        "place_original": row.get("place", ""),
        "type_original": original_type,
        "name_original": row.get("name", ""),
        "prompt_original": original_prompt,
        "source_display_name": source_display_name,
    }


def resolve_source_dir(args: argparse.Namespace) -> tuple[Path, dict[str, Any]]:
    if args.source_dir and args.source_dir.exists():
        return args.source_dir, {"source_dir": str(args.source_dir)}
    if DEFAULT_SOURCE_DIR.exists() and not args.force_download:
        return DEFAULT_SOURCE_DIR, {"source_dir": str(DEFAULT_SOURCE_DIR)}

    load_env_file()
    api = wandb.Api(timeout=60)
    artifact = api.artifact(args.source_artifact, type="dataset")
    root = Path(tempfile.mkdtemp(prefix="hallulens_ja_"))
    source_dir = Path(artifact.download(root=str(root)))
    return source_dir, {
        "source_artifact": args.source_artifact,
        "source_version": artifact.version,
        "source_aliases": list(artifact.aliases),
    }


def materialize(args: argparse.Namespace) -> Path:
    source_dir, source_meta = resolve_source_dir(args)
    rows_by_split = {
        split: read_jsonl(source_dir / split / "generation.jsonl")
        for split in args.splits
    }
    cache_path = Path(args.cache_path) if args.cache_path else args.output_dir / ".cache" / "hallulens_zh_tw.jsonl"
    cache = load_translation_cache(cache_path)
    strings = collect_strings(rows_by_split)
    cache = translate_many(
        strings,
        cache,
        cache_path,
        args.model,
        args.max_workers,
        args.dry_run,
    )

    artifact_root = args.output_dir / args.dataset_dir_name
    if artifact_root.exists():
        shutil.rmtree(artifact_root)
    for split, rows in rows_by_split.items():
        transformed = [transform_row(row, cache, args.model) for row in rows]
        write_jsonl(artifact_root / split / "generation.jsonl", transformed)

    files = {}
    for path in sorted(artifact_root.rglob("*")):
        if path.is_file():
            files[str(path.relative_to(artifact_root))] = {
                "bytes": path.stat().st_size,
                "sha256": sha256_file(path),
            }

    manifest = {
        "dataset": "HalluLens zh-TW",
        "dataset_dir_name": args.dataset_dir_name,
        "source": source_meta,
        "source_artifact_default": DEFAULT_SOURCE_ARTIFACT,
        "translation_model": args.model,
        "translation_cache": str(cache_path),
        "splits": {split: len(rows) for split, rows in rows_by_split.items()},
        "type_translations": TYPE_TRANSLATIONS,
        "prompt_template": "可以詳細介紹位於{place}的「{name}」這間{type_}嗎？",
        "files": files,
    }
    write_json(artifact_root / "manifest.json", manifest)
    (artifact_root / "README.md").write_text(
        "\n".join(
            [
                "# HalluLens zh-TW",
                "",
                "Traditional Chinese HalluLens artifact for the Nejumi Taiwan leaderboard.",
                "",
                "Rows are transformed from the Nejumi4 Japanese HalluLens artifact.",
                "Japanese prompts are reconstructed in Traditional Chinese while preserving original fields with *_original columns.",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    return artifact_root


def upload_artifact(args: argparse.Namespace, artifact_root: Path) -> None:
    load_env_file()
    run = wandb.init(entity=args.entity, project=args.project, job_type="upload-hallulens-zh-tw")
    manifest = json.loads((artifact_root / "manifest.json").read_text(encoding="utf-8"))
    artifact = wandb.Artifact(
        args.artifact_name,
        type="dataset",
        description="HalluLens zh-TW data for Nejumi Taiwan",
        metadata=manifest,
    )
    artifact.add_dir(str(artifact_root), name=args.dataset_dir_name)
    run.log_artifact(artifact, aliases=["production"])
    run.finish()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-artifact", default=DEFAULT_SOURCE_ARTIFACT)
    parser.add_argument("--source-dir", type=Path, default=DEFAULT_SOURCE_DIR)
    parser.add_argument("--force-download", action="store_true")
    parser.add_argument("--output-dir", type=Path, default=Path("data/taiwan"))
    parser.add_argument("--dataset-dir-name", default=DEFAULT_DATASET_DIR_NAME)
    parser.add_argument("--artifact-name", default=DEFAULT_ARTIFACT_NAME)
    parser.add_argument("--model", default="gpt-5.5")
    parser.add_argument("--splits", nargs="+", default=["dev", "test"])
    parser.add_argument("--max-workers", type=int, default=4)
    parser.add_argument("--cache-path", default=None)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--upload", action="store_true")
    parser.add_argument("--entity", default="llm-leaderboard")
    parser.add_argument("--project", default="tc-leaderboard")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    artifact_root = materialize(args)
    print(artifact_root)
    if args.upload:
        upload_artifact(args, artifact_root)


if __name__ == "__main__":
    main()
