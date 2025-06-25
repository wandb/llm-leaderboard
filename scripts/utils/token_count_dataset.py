#!/usr/bin/env python3
"""token_count_dataset.py

指定した HuggingFace Datasets (Arrow 形式) ディレクトリを読み込み、
各インスタンスの text / problem_statement / hints_text / patch (模範回答) の
トークン数を算出して CSV に出力するユーティリティ。

使い方:
    python scripts/utils/token_count_dataset.py --dataset_dir /path/to/dataset \
        --output_csv token_stats.csv [--split test]

トークンは単純に空白分割した長さ (len(str.split())) で計測する。
必要に応じて tiktoken 等に差し替え可。
"""
from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import List, Dict, Any

from datasets import load_from_disk


def count_tokens(s: str | None) -> int:
    if not s:
        return 0
    return len(s.split())


def process_dataset(ds, output_path: Path, split: str | None = None):
    if split and split in ds:
        ds = ds[split]

    fields = [
        ("text", "text_tokens"),
        ("problem_statement", "problem_tokens"),
        ("hints_text", "hints_tokens"),
        ("patch", "gold_patch_tokens"),  # gold patch のフィールド名 (SWE-bench では 'patch')
    ]

    with output_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        header = ["instance_id"] + [name for _, name in fields]
        writer.writerow(header)

        for sample in ds:
            row: List[Any] = [sample.get("instance_id", "")]
            for field_name, _ in fields:
                row.append(count_tokens(sample.get(field_name)))
            writer.writerow(row)

    print(f"✅ Saved token statistics to {output_path} (total {len(ds)} rows)")


def main():
    parser = argparse.ArgumentParser(description="Count tokens of dataset fields and save as CSV")
    parser.add_argument("--dataset_dir", required=True, help="Path to dataset directory (load_from_disk)")
    parser.add_argument("--output_csv", required=True, help="Output CSV filename")
    parser.add_argument("--split", default=None, help="Dataset split to use (e.g., test)")
    args = parser.parse_args()

    ds = load_from_disk(args.dataset_dir)
    output_path = Path(args.output_csv)
    process_dataset(ds, output_path, args.split)


if __name__ == "__main__":
    main() 