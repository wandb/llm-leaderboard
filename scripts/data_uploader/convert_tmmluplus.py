"""
Convert TMMLU+ dataset from HuggingFace to Nejumi jaster format.

Matches JMMLU's exact format convention:
  input: "問題：{question}\n選項：A.{A},B.{B},C.{C},D.{D}"
  output: "A"

Usage:
    python scripts/data_uploader/convert_tmmluplus.py \
        --output_dir ./data/tmmluplus_jaster

Then upload:
    python scripts/data_uploader/upload_dataset.py \
        -e <entity> -p <project> -n tmmluplus -d ./data/tmmluplus_jaster -m "TMMLU+ TC benchmark"
"""

import json
from argparse import ArgumentParser
from pathlib import Path

from datasets import load_dataset, get_dataset_config_names


def format_question(row):
    """Format a TMMLU+ row to match JMMLU input convention."""
    # JMMLU format: 質問：{q}\n選択肢：A.{a},B.{b},C.{c},D.{d}
    # TC equivalent: 問題：{q}\n選項：A.{a},B.{b},C.{c},D.{d}
    q = row["question"]
    choices = f"A.{row['A']},B.{row['B']},C.{row['C']},D.{row['D']}"
    return f"問題：{q}\n選項：{choices}", row["answer"]


def convert_split(ds_split, max_samples=None):
    """Convert a HuggingFace dataset split to jaster samples list."""
    samples = []
    for i, row in enumerate(ds_split):
        if max_samples and i >= max_samples:
            break
        input_text, output_text = format_question(row)
        samples.append({"input": input_text, "output": output_text})
    return samples


def write_task_json(out_path, samples, instruction):
    """Write a jaster-format JSON file."""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    task_json = {
        "instruction": instruction,
        "metrics": ["exact_match"],
        "output_length": 5,
        "samples": samples,
    }
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(task_json, f, ensure_ascii=False, indent=2)


INSTRUCTION = "從給定的問題和選項中，選擇最適當的答案。回答僅包含選項的字母（例：A），不得包含其他任何內容。"


def main():
    parser = ArgumentParser()
    parser.add_argument("--output_dir", type=str, required=True,
                        help="Output directory (will contain tmmluplus/{test,dev,train}/)")
    args = parser.parse_args()

    output_dir = Path(args.output_dir) / "tmmluplus"

    # Discover all TMMLU+ subjects
    print("Discovering TMMLU+ subjects...")
    subjects = get_dataset_config_names("ikala/tmmluplus")
    print(f"Found {len(subjects)} subjects")

    # Aggregate all samples across subjects
    all_samples = {"test": [], "dev": [], "train": []}
    split_map = {"test": "test", "validation": "dev", "train": "train"}

    for subject in subjects:
        print(f"  {subject}...", end=" ")
        try:
            ds = load_dataset("ikala/tmmluplus", name=subject)
        except Exception as e:
            print(f"SKIP ({e})")
            continue

        counts = []
        for hf_split, jaster_split in split_map.items():
            if hf_split not in ds:
                continue
            samples = convert_split(ds[hf_split])
            all_samples[jaster_split].extend(samples)
            counts.append(f"{jaster_split}={len(samples)}")
        print(", ".join(counts))

    # Write aggregate tmmluplus.json for each split
    for split_name, samples in all_samples.items():
        if not samples:
            continue
        out_path = output_dir / split_name / "tmmluplus.json"
        write_task_json(out_path, samples, INSTRUCTION)
        print(f"\n{split_name}/tmmluplus.json: {len(samples)} samples")

    print(f"\nDone. Output: {output_dir}")
    print(f"\nNext: upload to W&B:")
    print(f"  python scripts/data_uploader/upload_dataset.py \\")
    print(f"    -e <entity> -p <project> \\")
    print(f"    -n tmmluplus -d {output_dir} \\")
    print(f'    -m "TMMLU+ TC benchmark (ikala/tmmluplus, {len(subjects)} subjects)"')


if __name__ == "__main__":
    main()
