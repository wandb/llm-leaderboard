"""
Build Taiwan leaderboard dataset artifacts.

This script intentionally separates local materialization from W&B upload:

  python scripts/data_uploader/taiwan_artifacts.py build-tceval-v2 \
    --output-dir data/taiwan

  python scripts/data_uploader/taiwan_artifacts.py translate-hle-ja \
    --output-dir data/taiwan --limit 3 --dry-run

Add --upload with --entity/--project after inspecting local outputs.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shutil
import string
import tempfile
import time
import urllib.error
import urllib.parse
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

import wandb


TCEVAL_V2_DATASET = "MediaTek-Research/TCEval-v2"
DATASETS_SERVER = "https://datasets-server.huggingface.co"
MTBENCH_TW_CONFIGS = [
    "mt_bench_tw-coding",
    "mt_bench_tw-extraction",
    "mt_bench_tw-humanities",
    "mt_bench_tw-math",
    "mt_bench_tw-reasoning",
    "mt_bench_tw-roleplay",
    "mt_bench_tw-stem",
    "mt_bench_tw-writing",
]
JP_CHAR_RE = re.compile(r"[\u3040-\u30ff]")
CHOICE_SYMBOLS = ("$", "&", "#", "@", "%", "!", "?", "~", "^", "*", "+", "=")
ZH_TW_COMPAT_TRANSLATION = str.maketrans({"・": "·", "ー": "-"})

BFCL_V3_TAIWAN_CATEGORIES = [
    "java",
    "javascript",
    "live_irrelevance",
    "live_multiple",
    "live_relevance",
    "live_simple",
    "multi_turn_base",
    "multi_turn_miss_func",
    "multi_turn_miss_param",
    "simple",
    "multiple",
    "irrelevance",
]
BFCL_PROMPT_REL_DIR = Path("bfcl")
BFCL_POSSIBLE_ANSWER_REL_DIR = Path("bfcl") / "possible_answer"
BFCL_TRANSLATABLE_MESSAGE_ROLES = {"user", "system"}

ZH_TW_HLE_SYSTEM_PROMPT = """請用以下格式回答：
說明: {你對所選答案的說明}
答案: {你選擇的答案}
信心: {你對答案的 0% 到 100% 信心分數}"""

ZH_TW_HLE_JUDGE_PROMPT = """請判斷以下[問題]的[回答]是否依據明確且不含歧義的[正解]作答正確。

[問題]: {question}

[回答]: {response}

請依照以下格式與標準判定：

extracted_final_answer: 從[回答]中抽取出的最終精確答案。若無法抽取精確且最終的答案，請寫 'None'。

[正解]: {correct_answer}

reasoning: 只根據[正解]說明 extracted_final_answer 正確或錯誤。請只聚焦在[正解]與 extracted_final_answer 是否存在有意義差異；不要評論問題背景，也不要嘗試重新解題。

correct: 若 extracted_final_answer 與上述[正解]一致，或數值題在合理小誤差範圍內，請回答 'yes'。否則請回答 'no'。

confidence: 從[回答]中抽取 0% 到 100% 的信心分數。若沒有信心分數，請填 100。"""


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)


def normalize_zh_tw_text(text: str) -> str:
    return text.translate(ZH_TW_COMPAT_TRANSLATION)


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def hf_rows(
    dataset: str,
    config: str,
    split: str,
    request_interval: float = 0.25,
    max_retries: int = 6,
) -> list[dict[str, Any]]:
    try:
        from datasets import load_dataset

        loaded = load_dataset(dataset, config, split=split)
        return [dict(row) for row in loaded]
    except Exception as exc:
        print(
            f"Falling back to Hugging Face Dataset Viewer for "
            f"{dataset}/{config}/{split}: {exc}"
        )

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
        for attempt in range(max_retries):
            try:
                request = urllib.request.Request(
                    url,
                    headers={"User-Agent": "nejumi-taiwan-artifact-builder/0.1"},
                )
                with urllib.request.urlopen(request, timeout=60) as response:
                    payload = json.load(response)
                break
            except urllib.error.HTTPError as exc:
                if exc.code not in {429, 500, 502, 503, 504} or attempt == max_retries - 1:
                    raise
                retry_after = exc.headers.get("Retry-After")
                sleep_seconds = float(retry_after) if retry_after else min(2 ** attempt, 30)
                print(
                    f"Retrying HF rows request after HTTP {exc.code}: "
                    f"{config}/{split} offset={offset} sleep={sleep_seconds}s"
                )
                time.sleep(sleep_seconds)
        batch = [item["row"] for item in payload.get("rows", [])]
        rows.extend(batch)
        total = payload.get("num_rows_total")
        if not batch or total is None or len(rows) >= total:
            break
        offset += len(batch)
        if request_interval > 0:
            time.sleep(request_interval)
    return rows


def build_mtbench_tw(output_dir: Path, request_interval: float) -> dict[str, Path]:
    question_rows: list[dict[str, Any]] = []
    reference_rows: list[dict[str, Any]] = []

    for config in MTBENCH_TW_CONFIGS:
        for row in hf_rows(
            TCEVAL_V2_DATASET,
            config,
            "test",
            request_interval=request_interval,
        ):
            question_id = int(row["id"])
            question_rows.append(
                {
                    "question_id": question_id,
                    "category": row["category"],
                    "turns": row["turns"],
                }
            )
            references = row.get("reference")
            if references:
                reference_rows.append(
                    {
                        "question_id": question_id,
                        "answer_id": f"tceval-v2-ref-{question_id}",
                        "model_id": "tceval-v2-reference",
                        "choices": [{"index": 0, "turns": references}],
                    }
                )

    question_rows.sort(key=lambda row: row["question_id"])
    reference_rows.sort(key=lambda row: row["question_id"])

    question_path = output_dir / "mtbench_tw_question" / "question.jsonl"
    reference_path = (
        output_dir
        / "mtbench_tw_referenceanswer"
        / "tceval-v2-reference.jsonl"
    )
    write_jsonl(question_path, question_rows)
    write_jsonl(reference_path, reference_rows)
    return {"question": question_path, "referenceanswer": reference_path}


def build_tceval_v2_selected(
    output_dir: Path,
    request_interval: float,
) -> dict[str, Path]:
    artifact_dir = output_dir / "tceval_v2_selected" / "tceval_v2_selected"
    raw_dir = artifact_dir / "raw"
    jaster_dir = artifact_dir / "jaster"

    written: dict[str, Path] = {}
    for split in ("dev", "test"):
        drcd_rows = hf_rows(
            TCEVAL_V2_DATASET,
            "drcd",
            split,
            request_interval=request_interval,
        )
        raw_path = raw_dir / "drcd" / f"{split}.jsonl"
        write_jsonl(raw_path, drcd_rows)
        samples = [
            {
                "input": f"文章：{row['paragraph']}\n問題：{row['question']}",
                "output": row["references"][0] if row.get("references") else "",
            }
            for row in drcd_rows
        ]
        write_json(
            jaster_dir / split / "drcd.json",
            {
                "instruction": "請根據文章回答問題。只輸出最精簡且正確的答案。",
                "metrics": ["char_f1"],
                "output_length": 64,
                "samples": samples,
            },
        )
        written[f"drcd_{split}"] = raw_path

    for split in ("dev", "test"):
        penguin_rows = hf_rows(
            TCEVAL_V2_DATASET,
            "penguin_table",
            split,
            request_interval=request_interval,
        )
        raw_path = raw_dir / "penguin_table" / f"{split}.jsonl"
        write_jsonl(raw_path, penguin_rows)
        samples = []
        for row in penguin_rows:
            choices = ",".join(
                f"{label}.{row[label]}"
                for label in ("A", "B", "C", "D", "E")
                if row.get(label) not in (None, "")
            )
            samples.append(
                {
                    "input": f"問題：{row['question']}\n選項：{choices}",
                    "output": row["answer"],
                }
            )
        write_json(
            jaster_dir / split / "penguin_table.json",
            {
                "instruction": "請根據表格與問題，選擇最適當的答案。回答僅包含選項字母。",
                "metrics": ["exact_match"],
                "output_length": 5,
                "samples": samples,
            },
        )
        written[f"penguin_table_{split}"] = raw_path

    return written


def upload_file_artifact(
    entity: str,
    project: str,
    artifact_name: str,
    file_path: Path,
    metadata: dict[str, Any],
    aliases: list[str] | None = None,
) -> None:
    with wandb.init(entity=entity, project=project, job_type="upload_data") as run:
        artifact = wandb.Artifact(
            name=artifact_name,
            type="dataset",
            metadata={**metadata, "sha256": sha256_file(file_path)},
        )
        artifact.add_file(str(file_path), name=file_path.name)
        run.log_artifact(artifact, aliases=aliases or ["production"])


def upload_dir_artifact(
    entity: str,
    project: str,
    artifact_name: str,
    dir_path: Path,
    artifact_dir_name: str,
    metadata: dict[str, Any],
    aliases: list[str] | None = None,
) -> None:
    with wandb.init(entity=entity, project=project, job_type="upload_data") as run:
        artifact = wandb.Artifact(name=artifact_name, type="dataset", metadata=metadata)
        artifact.add_dir(str(dir_path), name=artifact_dir_name)
        run.log_artifact(artifact, aliases=aliases or ["production"])


def build_tceval_v2(args: argparse.Namespace) -> None:
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    mtbench_paths = build_mtbench_tw(output_dir, request_interval=args.request_interval)
    selected_paths = build_tceval_v2_selected(
        output_dir,
        request_interval=args.request_interval,
    )

    manifest = {
        "source_dataset": TCEVAL_V2_DATASET,
        "generated_at": int(time.time()),
        "mtbench_tw_configs": MTBENCH_TW_CONFIGS,
        "selected_configs": ["drcd", "penguin_table"],
        "files": {
            **{k: str(v) for k, v in mtbench_paths.items()},
            **{k: str(v) for k, v in selected_paths.items()},
        },
    }
    manifest_path = output_dir / "tceval_v2_manifest.json"
    write_json(manifest_path, manifest)
    print(json.dumps(manifest, ensure_ascii=False, indent=2))

    if args.upload:
        require_wandb_destination(args)
        common_metadata = {
            "source_dataset": TCEVAL_V2_DATASET,
            "source_url": "https://huggingface.co/datasets/MediaTek-Research/TCEval-v2",
            "generated_at": manifest["generated_at"],
        }
        upload_file_artifact(
            args.entity,
            args.project,
            "mtbench_tw_question",
            mtbench_paths["question"],
            {**common_metadata, "configs": MTBENCH_TW_CONFIGS},
        )
        upload_file_artifact(
            args.entity,
            args.project,
            "mtbench_tw_referenceanswer",
            mtbench_paths["referenceanswer"],
            {**common_metadata, "configs": MTBENCH_TW_CONFIGS},
        )
        upload_dir_artifact(
            args.entity,
            args.project,
            "tceval_v2_selected",
            output_dir / "tceval_v2_selected" / "tceval_v2_selected",
            "tceval_v2_selected",
            {**common_metadata, "configs": ["drcd", "penguin_table"]},
        )


def _option_label_markers(text: str) -> list[tuple[str, int]]:
    markers: list[tuple[str, int]] = []
    option_start = text.find("選項：")
    if option_start >= 0:
        block = text[option_start + len("選項：") :]
        base = option_start + len("選項：")
    else:
        block = text
        base = 0

    for match in re.finditer(r"(?:^|,)([A-Z])\.", block):
        markers.append((match.group(1), base + match.start(1)))
    return markers


def choice_label_positions_from_input(text: str) -> list[tuple[str, int]]:
    """Infer contiguous option labels while ignoring initials inside choices."""
    markers = _option_label_markers(text)
    selected: list[tuple[str, int]] = []
    search_from = 0
    for expected in string.ascii_uppercase:
        found_at = None
        for idx in range(search_from, len(markers)):
            if markers[idx][0] == expected:
                found_at = idx
                break
        if found_at is None:
            break
        selected.append(markers[found_at])
        search_from = found_at + 1
    return selected


def choice_labels_from_input(text: str) -> list[str]:
    return [label for label, _ in choice_label_positions_from_input(text)]


def choice_labels_from_outputs(samples: list[dict[str, Any]]) -> list[str]:
    max_index = -1
    for sample in samples:
        output = str(sample.get("output", "")).strip().upper()
        if output in string.ascii_uppercase:
            max_index = max(max_index, string.ascii_uppercase.index(output))
    if max_index >= 0:
        return list(string.ascii_uppercase[: max_index + 1])
    return []


def choice_labels_for_sample(
    sample: dict[str, Any],
    task_data: dict[str, Any],
    fallback_labels: list[str] | None = None,
) -> list[str]:
    for source in (sample.get("label_list"), task_data.get("label_list")):
        if isinstance(source, list) and source:
            return [str(label).strip().upper() for label in source]

    choices = sample.get("choices") or sample.get("choice")
    if isinstance(choices, list) and choices:
        return list(string.ascii_uppercase[: len(choices)])

    labels = choice_labels_from_input(sample.get("input", ""))
    output = str(sample.get("output", "")).strip().upper()
    if output in string.ascii_uppercase and output not in labels:
        output_index = string.ascii_uppercase.index(output) + 1
        labels = list(string.ascii_uppercase[: max(output_index, len(labels))])
    if fallback_labels and len(labels) < len(fallback_labels):
        labels = fallback_labels
    return labels or list("ABCD")


def get_incorrect_choices(correct_answer: str, labels: list[str]) -> str:
    correct_answer = str(correct_answer).strip().upper()
    return ",".join(choice for choice in labels if choice != correct_answer)


def replace_tmmlu_choice_labels(text: str, mapping: dict[str, str]) -> str:
    replacements = [
        (pos, mapping[label])
        for label, pos in choice_label_positions_from_input(text)
        if label in mapping
    ]
    chars = list(text)
    for pos, replacement in sorted(replacements, reverse=True):
        chars[pos : pos + 1] = replacement
    return "".join(chars)


def build_tmmluplus_robust(args: argparse.Namespace) -> None:
    api = wandb.Api(timeout=60)
    source_artifact = api.artifact(args.source_artifact, type="dataset")
    source_root = Path(tempfile.mkdtemp(prefix="tmmluplus_source_"))
    source_dir = (
        Path(source_artifact.download(root=str(source_root)))
        / args.source_dataset_dir
    )
    if not source_dir.exists():
        raise FileNotFoundError(f"source dataset dir not found: {source_dir}")

    output_root = Path(args.output_dir)
    output_dir = output_root / args.artifact_dir_name
    for split in ("dev", "test", "train"):
        source_file = source_dir / split / "tmmluplus.json"
        if not source_file.exists():
            continue
        data = json.loads(source_file.read_text(encoding="utf-8"))
        fallback_labels = choice_labels_from_outputs(data["samples"])
        max_choice_count = max(
            len(choice_labels_for_sample(sample, data, fallback_labels))
            for sample in data["samples"]
        )
        if max_choice_count > len(CHOICE_SYMBOLS):
            raise ValueError(
                f"{split} requires {max_choice_count} symbols, "
                f"but only {len(CHOICE_SYMBOLS)} are configured"
            )
        (output_dir / split).mkdir(parents=True, exist_ok=True)
        write_json(output_dir / split / "tmmluplus.json", data)

        incorrect_data = dict(data)
        incorrect_data["instruction"] = (
            "從給定的問題和選項中，選出所有不正確的答案。"
            "回答僅包含選項字母，並以逗號分隔（例：B,C,D），不得包含其他任何內容。"
        )
        incorrect_samples = []
        for sample in data["samples"]:
            labels = choice_labels_for_sample(sample, data, fallback_labels)
            incorrect_samples.append({
                **sample,
                "robust_choice_labels": labels,
                "output": get_incorrect_choices(
                    sample["output"],
                    labels,
                ),
            })
        incorrect_data["samples"] = incorrect_samples
        write_json(
            output_dir / split / "tmmluplus_IncorrectChoice.json",
            incorrect_data,
        )

        symbol_data = dict(data)
        symbol_data["instruction"] = (
            "從給定的問題和選項中，選擇最適當的答案。"
            "回答僅包含選項符號（例：$），不得包含其他任何內容。"
        )
        symbol_data["label_list"] = list(CHOICE_SYMBOLS[:max_choice_count])
        symbol_samples = []
        for sample in data["samples"]:
            labels = choice_labels_for_sample(sample, data, fallback_labels)
            label_mapping = {
                label: CHOICE_SYMBOLS[idx]
                for idx, label in enumerate(labels)
            }
            symbol_samples.append(
                {
                    **sample,
                    "robust_choice_labels": labels,
                    "robust_symbol_mapping": label_mapping,
                    "input": replace_tmmlu_choice_labels(sample["input"], label_mapping),
                    "output": label_mapping.get(sample["output"], sample["output"]),
                }
            )
        symbol_data["samples"] = symbol_samples
        write_json(output_dir / split / "tmmluplus_SymbolChoice.json", symbol_data)

    metadata = {
        "source_artifact": args.source_artifact,
        "source_version": source_artifact.version,
        "source_dataset_dir": args.source_dataset_dir,
        "variants": ["tmmluplus", "tmmluplus_IncorrectChoice", "tmmluplus_SymbolChoice"],
        "scoring": "Nejumi4-style three-way consistency robust score with variable choice-label support",
    }
    write_json(output_dir / "robust_metadata.json", metadata)
    print(json.dumps(metadata, ensure_ascii=False, indent=2))

    if args.upload:
        require_wandb_destination(args)
        upload_dir_artifact(
            args.entity,
            args.project,
            args.artifact_name,
            output_dir,
            args.artifact_dir_name,
            metadata,
        )


def needs_translation(value: Any) -> bool:
    return bool(JP_CHAR_RE.search(str(value or "")))


def cache_key(model: str, text: str) -> str:
    return hashlib.sha256(f"{model}\n{text}".encode("utf-8")).hexdigest()


def load_translation_cache(path: Path) -> dict[str, str]:
    cache: dict[str, str] = {}
    if not path.exists():
        return cache
    with path.open(encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            row = json.loads(line)
            cache[row["key"]] = normalize_zh_tw_text(row["translation"])
    return cache


def append_translation_cache(path: Path, key: str, source: str, translation: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(
            json.dumps(
                {
                    "key": key,
                    "source_sha256": hashlib.sha256(source.encode("utf-8")).hexdigest(),
                    "translation": translation,
                },
                ensure_ascii=False,
            )
            + "\n"
        )


def translate_text(
    client: Any,
    model: str,
    text: str,
    max_retries: int = 5,
    strict_no_japanese: bool = False,
) -> str:
    if not text or not needs_translation(text):
        return text
    strict_rule = ""
    if strict_no_japanese:
        strict_rule = (
            "The previous translation left Japanese text behind. Translate every "
            "Japanese phrase into Traditional Chinese. The output must contain no "
            "Hiragana or Katakana characters. Use Traditional Chinese transliteration "
            "for proper nouns when needed. Do not preserve Japanese words such as "
            "'次の', '正の', '以下に', 'とする', 'および', or Katakana names; translate them.\n"
        )
    prompt = (
        "Translate the following Japanese benchmark text into Traditional Chinese "
        "for Taiwanese readers. Preserve LaTeX, code blocks, option labels, numbers, "
        "citations, IDs, and technical symbols exactly. Return only the translation.\n"
        f"{strict_rule}\n{text}"
    )
    for attempt in range(max_retries):
        try:
            response = client.responses.create(
                model=model,
                input=[{"role": "user", "content": prompt}],
            )
            return normalize_zh_tw_text(response.output_text)
        except Exception:
            if attempt == max_retries - 1:
                raise
            time.sleep(min(2 ** attempt, 30))
    raise RuntimeError("unreachable")


def bfcl_prompt_file_paths(bfcl_dir: Path) -> list[Path]:
    return sorted(bfcl_dir.glob("BFCL_v3_*.json"))


def bfcl_possible_answer_file_paths(bfcl_dir: Path) -> list[Path]:
    possible_answer_dir = bfcl_dir / "possible_answer"
    if not possible_answer_dir.exists():
        return []
    return sorted(possible_answer_dir.glob("BFCL_v3_*.json"))


def bfcl_category_from_file(path: Path) -> str:
    name = path.stem
    prefix = "BFCL_v3_"
    if not name.startswith(prefix):
        raise ValueError(f"not a BFCL v3 file: {path}")
    return name[len(prefix) :]


def selected_bfcl_files(paths: list[Path], categories: list[str] | None) -> list[Path]:
    if not categories:
        return paths
    category_set = set(categories)
    return [path for path in paths if bfcl_category_from_file(path) in category_set]


def iter_bfcl_question_texts(row: dict[str, Any]) -> list[str]:
    texts: list[str] = []
    for turn in row.get("question", []) or []:
        if not isinstance(turn, list):
            continue
        for message in turn:
            if not isinstance(message, dict):
                continue
            if message.get("role") not in BFCL_TRANSLATABLE_MESSAGE_ROLES:
                continue
            content = message.get("content")
            if isinstance(content, str) and content.strip():
                texts.append(content)
    return texts


def translate_bfcl_question_row(
    row: dict[str, Any],
    translation_cache: dict[str, str],
    model: str,
    dry_run: bool = False,
    translate_all_messages: bool = True,
) -> tuple[dict[str, Any], int]:
    translated_count = 0
    out = dict(row)
    out["question"] = []
    for turn in row.get("question", []) or []:
        if not isinstance(turn, list):
            out["question"].append(turn)
            continue
        new_turn = []
        for message in turn:
            if not isinstance(message, dict):
                new_turn.append(message)
                continue
            new_message = dict(message)
            content = new_message.get("content")
            should_translate = (
                isinstance(content, str)
                and content.strip()
                and new_message.get("role") in BFCL_TRANSLATABLE_MESSAGE_ROLES
                and (translate_all_messages or needs_translation(content))
            )
            if should_translate:
                translated_count += 1
                if dry_run:
                    new_message["content_translation_needed"] = True
                else:
                    new_message["content"] = normalize_zh_tw_text(
                        translation_cache[cache_key(model, content)]
                    )
            new_turn.append(new_message)
        out["question"].append(new_turn)
    return out, translated_count


def collect_japanese_strings_from_obj(value: Any, found: dict[str, str], model: str) -> None:
    if isinstance(value, str):
        if needs_translation(value):
            found[cache_key(model, value)] = value
        return
    if isinstance(value, list):
        for item in value:
            collect_japanese_strings_from_obj(item, found, model)
        return
    if isinstance(value, dict):
        for item in value.values():
            collect_japanese_strings_from_obj(item, found, model)


def translate_japanese_strings_in_obj(
    value: Any,
    translation_cache: dict[str, str],
    model: str,
    dry_run: bool = False,
) -> tuple[Any, int]:
    if isinstance(value, str):
        if needs_translation(value):
            if dry_run:
                return value, 1
            return normalize_zh_tw_text(translation_cache[cache_key(model, value)]), 1
        return value, 0
    if isinstance(value, list):
        translated_items = []
        count = 0
        for item in value:
            translated_item, translated_count = translate_japanese_strings_in_obj(
                item,
                translation_cache,
                model,
                dry_run=dry_run,
            )
            translated_items.append(translated_item)
            count += translated_count
        return translated_items, count
    if isinstance(value, dict):
        translated_dict = {}
        count = 0
        for key, item in value.items():
            translated_item, translated_count = translate_japanese_strings_in_obj(
                item,
                translation_cache,
                model,
                dry_run=dry_run,
            )
            translated_dict[key] = translated_item
            count += translated_count
        return translated_dict, count
    return value, 0


def repair_remaining_japanese_in_obj(
    value: Any,
    client: Any,
    model: str,
    translation_cache: dict[str, str],
    cache_path: Path,
    max_attempts: int,
) -> tuple[Any, int]:
    current = value
    repaired_total = 0
    for _ in range(max_attempts):
        remaining: dict[str, str] = {}
        collect_japanese_strings_from_obj(current, remaining, model)
        if not remaining:
            break
        for key, text in remaining.items():
            cached = translation_cache.get(key)
            if cached is None or needs_translation(cached):
                repaired = translate_text(
                    client,
                    model,
                    text,
                    strict_no_japanese=True,
                )
                translation_cache[key] = repaired
                append_translation_cache(cache_path, key, text, repaired)
        current, repaired_count = translate_japanese_strings_in_obj(
            current,
            translation_cache,
            model,
            dry_run=False,
        )
        repaired_total += repaired_count
    return current, repaired_total


def count_japanese_chars_in_path(path: Path) -> int:
    return len(JP_CHAR_RE.findall(path.read_text(encoding="utf-8", errors="ignore")))


def audit_bfcl_dir(bfcl_dir: Path) -> dict[str, Any]:
    groups = {
        "prompt": bfcl_prompt_file_paths(bfcl_dir),
        "possible_answer": bfcl_possible_answer_file_paths(bfcl_dir),
        "multi_turn_func_doc": sorted((bfcl_dir / "multi_turn_func_doc").glob("*.json"))
        if (bfcl_dir / "multi_turn_func_doc").exists()
        else [],
    }
    group_stats: dict[str, Any] = {}
    total_japanese_chars = 0
    for group_name, files in groups.items():
        rows = 0
        japanese_chars = 0
        files_with_japanese = 0
        for path in files:
            text = path.read_text(encoding="utf-8", errors="ignore")
            rows += sum(1 for line in text.splitlines() if line.strip())
            file_japanese_chars = len(JP_CHAR_RE.findall(text))
            japanese_chars += file_japanese_chars
            if file_japanese_chars:
                files_with_japanese += 1
        total_japanese_chars += japanese_chars
        group_stats[group_name] = {
            "files": len(files),
            "rows": rows,
            "japanese_chars": japanese_chars,
            "files_with_japanese": files_with_japanese,
        }
    group_stats["total_japanese_chars"] = total_japanese_chars
    return group_stats


def translate_bfcl_v3_ja(args: argparse.Namespace) -> None:
    output_root = Path(args.output_dir)
    output_dir = output_root / args.dataset_dir_name
    bfcl_output_dir = output_dir / "bfcl"
    output_dir.mkdir(parents=True, exist_ok=True)

    api = wandb.Api(timeout=60)
    source_artifact = api.artifact(args.source_artifact, type="dataset")
    source_root = Path(tempfile.mkdtemp(prefix="bfcl_v3_ja_source_"))
    source_dir = Path(source_artifact.download(root=str(source_root)))
    source_bfcl_dir = source_dir / "bfcl"
    if not source_bfcl_dir.exists():
        raise FileNotFoundError(f"source BFCL dir not found: {source_bfcl_dir}")

    if bfcl_output_dir.exists() and not args.dry_run:
        shutil.rmtree(bfcl_output_dir)
    if not args.dry_run:
        shutil.copytree(source_bfcl_dir, bfcl_output_dir)

    categories = None if args.categories == ["all"] else args.categories
    prompt_files = selected_bfcl_files(bfcl_prompt_file_paths(source_bfcl_dir), categories)
    possible_answer_files = selected_bfcl_files(
        bfcl_possible_answer_file_paths(source_bfcl_dir),
        categories,
    )

    client = None
    if not args.dry_run:
        from openai import OpenAI

        if not os.environ.get("OPENAI_API_KEY"):
            raise ValueError("OPENAI_API_KEY is required unless --dry-run is set")
        client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))

    cache_path = Path(args.cache_path) if args.cache_path else output_root / "bfcl_zh_tw_translation_cache.jsonl"
    translation_cache = load_translation_cache(cache_path)
    texts_to_translate: dict[str, str] = {}

    prompt_rows_by_file: dict[str, list[dict[str, Any]]] = {}
    possible_rows_by_file: dict[str, list[dict[str, Any]]] = {}
    for path in prompt_files:
        rows = read_jsonl(path)
        if args.limit is not None:
            rows = rows[: args.limit]
        prompt_rows_by_file[path.name] = rows
        for row in rows:
            for text in iter_bfcl_question_texts(row):
                if args.translate_all_prompt_messages or needs_translation(text):
                    key = cache_key(args.model, text)
                    if args.dry_run or key not in translation_cache:
                        texts_to_translate[key] = text

    for path in possible_answer_files:
        rows = read_jsonl(path)
        if args.limit is not None:
            rows = rows[: args.limit]
        possible_rows_by_file[path.name] = rows
        for row in rows:
            found_possible_answer_texts: dict[str, str] = {}
            collect_japanese_strings_from_obj(
                row.get("ground_truth"),
                found_possible_answer_texts,
                args.model,
            )
            for key, text in found_possible_answer_texts.items():
                if args.dry_run or key not in translation_cache:
                    texts_to_translate[key] = text

    if texts_to_translate and not args.dry_run:
        print(
            f"Translating {len(texts_to_translate)} unique BFCL text fields "
            f"with {args.max_workers} workers..."
        )
        with ThreadPoolExecutor(max_workers=args.max_workers) as executor:
            future_to_key = {
                executor.submit(translate_text, client, args.model, text): key
                for key, text in texts_to_translate.items()
            }
            for index, future in enumerate(as_completed(future_to_key), start=1):
                key = future_to_key[future]
                source = texts_to_translate[key]
                translation = future.result()
                translation_cache[key] = translation
                append_translation_cache(cache_path, key, source, translation)
                if index % 25 == 0 or index == len(future_to_key):
                    print(f"Translated {index}/{len(future_to_key)} BFCL text fields")

    stats: dict[str, Any] = {
        "source_artifact": args.source_artifact,
        "source_version": source_artifact.version,
        "target_language": "zh-Hant-TW",
        "translation_model": args.model,
        "dry_run": args.dry_run,
        "categories": args.categories,
        "limit": args.limit,
        "translate_all_prompt_messages": args.translate_all_prompt_messages,
        "prompt_files": {},
        "possible_answer_files": {},
        "translation_cache_path": str(cache_path),
    }

    if not args.dry_run:
        for file_name, rows in prompt_rows_by_file.items():
            translated_rows = []
            translated_count = 0
            for row in rows:
                translated_row, count = translate_bfcl_question_row(
                    row,
                    translation_cache,
                    args.model,
                    dry_run=False,
                    translate_all_messages=args.translate_all_prompt_messages,
                )
                if args.repair_japanese:
                    translated_row, repaired_count = repair_remaining_japanese_in_obj(
                        translated_row,
                        client,
                        args.model,
                        translation_cache,
                        cache_path,
                        args.max_repair_attempts,
                    )
                    count += repaired_count
                translated_rows.append(translated_row)
                translated_count += count
            write_jsonl(bfcl_output_dir / file_name, translated_rows)
            stats["prompt_files"][file_name] = {
                "rows": len(translated_rows),
                "translated_message_count": translated_count,
                "japanese_chars_after": count_japanese_chars_in_path(bfcl_output_dir / file_name),
            }

        for file_name, rows in possible_rows_by_file.items():
            translated_rows = []
            translated_count = 0
            for row in rows:
                translated_row, count = translate_japanese_strings_in_obj(
                    row,
                    translation_cache,
                    args.model,
                    dry_run=False,
                )
                if args.repair_japanese:
                    translated_row, repaired_count = repair_remaining_japanese_in_obj(
                        translated_row,
                        client,
                        args.model,
                        translation_cache,
                        cache_path,
                        args.max_repair_attempts,
                    )
                    count += repaired_count
                translated_rows.append(translated_row)
                translated_count += count
            write_jsonl(bfcl_output_dir / "possible_answer" / file_name, translated_rows)
            stats["possible_answer_files"][file_name] = {
                "rows": len(translated_rows),
                "translated_string_count": translated_count,
                "japanese_chars_after": count_japanese_chars_in_path(
                    bfcl_output_dir / "possible_answer" / file_name
                ),
            }

        stats["audit_after"] = audit_bfcl_dir(bfcl_output_dir)
        write_json(bfcl_output_dir / "translation_metadata.json", stats)
        remaining_japanese_chars = stats["audit_after"]["total_japanese_chars"]
        if remaining_japanese_chars > args.allow_japanese_chars:
            raise ValueError(
                f"BFCL zh-TW artifact still contains {remaining_japanese_chars} "
                f"Japanese kana characters; allowed={args.allow_japanese_chars}"
            )
    else:
        stats["texts_to_translate"] = len(texts_to_translate)
        stats["source_audit"] = audit_bfcl_dir(source_bfcl_dir)

    print(json.dumps(stats, ensure_ascii=False, indent=2))

    if args.upload:
        if args.dry_run:
            raise ValueError("--upload cannot be used with --dry-run")
        require_wandb_destination(args)
        upload_dir_artifact(
            args.entity,
            args.project,
            args.artifact_name,
            bfcl_output_dir,
            "bfcl",
            stats,
            aliases=["production", "zh-tw", "tc"],
        )


def translate_hle_ja(args: argparse.Namespace) -> None:
    output_root = Path(args.output_dir)
    output_dir = output_root / args.dataset_dir_name
    output_dir.mkdir(parents=True, exist_ok=True)

    api = wandb.Api(timeout=60)
    source_artifact = api.artifact(args.source_artifact, type="dataset")
    source_root = Path(tempfile.mkdtemp(prefix="hle_ja_source_"))
    source_dir = Path(source_artifact.download(root=str(source_root))) / "hle-ja"

    client = None
    if not args.dry_run:
        from openai import OpenAI

        if not os.environ.get("OPENAI_API_KEY"):
            raise ValueError("OPENAI_API_KEY is required unless --dry-run is set")
        client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))

    stats: dict[str, Any] = {
        "source_artifact": args.source_artifact,
        "source_version": source_artifact.version,
        "target_language": "zh-Hant-TW",
        "translation_model": args.model,
        "fields": args.fields,
        "dry_run": args.dry_run,
        "limit": args.limit,
        "requested_splits": args.splits,
        "splits": {},
    }

    cache_path = Path(args.cache_path) if args.cache_path else output_root / "hle_zh_tw_translation_cache.jsonl"
    translation_cache = load_translation_cache(cache_path)

    source_rows_by_file: dict[str, list[dict[str, Any]]] = {}
    texts_to_translate: dict[str, str] = {}

    for source_file in sorted(source_dir.glob("*.jsonl")):
        if source_file.stem not in set(args.splits):
            continue
        rows: list[dict[str, Any]] = []
        total = 0
        for line in source_file.open(encoding="utf-8"):
            if not line.strip():
                continue
            row = json.loads(line)
            total += 1
            if args.limit is not None and total > args.limit:
                break
            rows.append(row)
            for field in args.fields:
                text = str(row.get(field) or "")
                if needs_translation(text):
                    key = cache_key(args.model, text)
                    if not args.dry_run and key not in translation_cache:
                        texts_to_translate[key] = text
        source_rows_by_file[source_file.name] = rows

    if texts_to_translate and not args.dry_run:
        print(
            f"Translating {len(texts_to_translate)} unique HLE text fields "
            f"with {args.max_workers} workers..."
        )
        with ThreadPoolExecutor(max_workers=args.max_workers) as executor:
            future_to_key = {
                executor.submit(translate_text, client, args.model, text): key
                for key, text in texts_to_translate.items()
            }
            for future in as_completed(future_to_key):
                key = future_to_key[future]
                source = texts_to_translate[key]
                translation = future.result()
                translation_cache[key] = translation
                append_translation_cache(cache_path, key, source, translation)

    for file_name, rows in source_rows_by_file.items():
        translated_rows: list[dict[str, Any]] = []
        translated_count = 0
        for row in rows:
            out = dict(row)
            for field in args.fields:
                text = str(out.get(field) or "")
                if needs_translation(text):
                    translated_count += 1
                    if args.dry_run:
                        out[f"{field}_translation_needed"] = True
                    else:
                        out[field] = translation_cache[cache_key(args.model, text)]
            if not args.dry_run and args.repair_japanese:
                for field in args.fields:
                    for _ in range(args.max_repair_attempts):
                        text = str(out.get(field) or "")
                        if not needs_translation(text):
                            break
                        repaired = translate_text(
                            client,
                            args.model,
                            text,
                            strict_no_japanese=True,
                        )
                        out[field] = repaired
                        key = cache_key(args.model, text)
                        translation_cache[key] = repaired
                        append_translation_cache(cache_path, key, text, repaired)
            translated_rows.append(out)
        source_file_name = file_name
        write_jsonl(output_dir / source_file_name, translated_rows)
        stats["splits"][source_file_name] = {
            "rows": len(translated_rows),
            "translated_field_count": translated_count,
        }

    metadata_path = output_dir / "translation_metadata.json"
    stats["translation_cache_path"] = str(cache_path)
    write_json(metadata_path, stats)
    print(json.dumps(stats, ensure_ascii=False, indent=2))

    if args.upload:
        if args.dry_run:
            raise ValueError("--upload cannot be used with --dry-run")
        require_wandb_destination(args)
        upload_dir_artifact(
            args.entity,
            args.project,
            args.artifact_name,
            output_dir,
            args.dataset_dir_name,
            stats,
        )


def audit_artifacts(args: argparse.Namespace) -> None:
    api = wandb.Api(timeout=60)
    for artifact_path in args.artifact:
        artifact = api.artifact(artifact_path, type="dataset")
        root = Path(tempfile.mkdtemp(prefix="artifact_audit_"))
        artifact_dir = Path(artifact.download(root=str(root)))
        print(f"=== {artifact_path} ({artifact.version}) ===")
        for path in sorted(p for p in artifact_dir.rglob("*") if p.is_file()):
            if path.suffix.lower() not in {".json", ".jsonl", ".txt", ".md"}:
                continue
            text = path.read_text(encoding="utf-8", errors="ignore")
            print(
                path.relative_to(artifact_dir),
                {
                    "bytes": path.stat().st_size,
                    "jp_chars": len(JP_CHAR_RE.findall(text)),
                    "sha256": sha256_file(path),
                },
            )


def require_wandb_destination(args: argparse.Namespace) -> None:
    if not args.entity or not args.project:
        raise ValueError("--entity and --project are required with --upload")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    build = subparsers.add_parser("build-tceval-v2")
    build.add_argument("--output-dir", default="data/taiwan")
    build.add_argument("--request-interval", type=float, default=0.5)
    build.add_argument("--upload", action="store_true")
    build.add_argument("--entity")
    build.add_argument("--project")
    build.set_defaults(func=build_tceval_v2)

    robust = subparsers.add_parser("build-tmmluplus-robust")
    robust.add_argument("--source-artifact", default="llm-leaderboard/tc-leaderboard/tmmluplus:production")
    robust.add_argument("--source-dataset-dir", default="tmmluplus/tmmluplus")
    robust.add_argument("--output-dir", default="data/taiwan")
    robust.add_argument("--artifact-name", default="tmmluplus_robust")
    robust.add_argument("--artifact-dir-name", default="tmmluplus_robust")
    robust.add_argument("--upload", action="store_true")
    robust.add_argument("--entity")
    robust.add_argument("--project")
    robust.set_defaults(func=build_tmmluplus_robust)

    hle = subparsers.add_parser("translate-hle-ja")
    hle.add_argument("--source-artifact", default="llm-leaderboard/nejumi-leaderboard4/hle-ja:production")
    hle.add_argument("--output-dir", default="data/taiwan")
    hle.add_argument("--dataset-dir-name", default="hle-zh-tw")
    hle.add_argument("--artifact-name", default="hle-zh-tw")
    hle.add_argument("--model", default="gpt-5.4-mini-2026-03-17")
    hle.add_argument("--splits", nargs="+", default=["dev", "test"])
    hle.add_argument("--fields", nargs="+", default=["question", "answer"])
    hle.add_argument("--max-workers", type=int, default=4)
    hle.add_argument("--cache-path", default=None)
    hle.add_argument("--repair-japanese", action=argparse.BooleanOptionalAction, default=True)
    hle.add_argument("--max-repair-attempts", type=int, default=3)
    hle.add_argument("--limit", type=int, default=None)
    hle.add_argument("--dry-run", action="store_true")
    hle.add_argument("--upload", action="store_true")
    hle.add_argument("--entity")
    hle.add_argument("--project")
    hle.set_defaults(func=translate_hle_ja)

    bfcl = subparsers.add_parser("translate-bfcl-v3-ja")
    bfcl.add_argument("--source-artifact", default="llm-leaderboard/nejumi-leaderboard4/bfcl:production")
    bfcl.add_argument("--output-dir", default="data/taiwan")
    bfcl.add_argument("--dataset-dir-name", default="bfcl-zh-tw")
    bfcl.add_argument("--artifact-name", default="bfcl-zh-tw")
    bfcl.add_argument("--model", default="gpt-5.4-mini-2026-03-17")
    bfcl.add_argument("--categories", nargs="+", default=BFCL_V3_TAIWAN_CATEGORIES)
    bfcl.add_argument("--translate-all-prompt-messages", action=argparse.BooleanOptionalAction, default=True)
    bfcl.add_argument("--max-workers", type=int, default=4)
    bfcl.add_argument("--cache-path", default=None)
    bfcl.add_argument("--repair-japanese", action=argparse.BooleanOptionalAction, default=True)
    bfcl.add_argument("--max-repair-attempts", type=int, default=3)
    bfcl.add_argument("--allow-japanese-chars", type=int, default=0)
    bfcl.add_argument("--limit", type=int, default=None)
    bfcl.add_argument("--dry-run", action="store_true")
    bfcl.add_argument("--upload", action="store_true")
    bfcl.add_argument("--entity")
    bfcl.add_argument("--project")
    bfcl.set_defaults(func=translate_bfcl_v3_ja)

    audit = subparsers.add_parser("audit-artifacts")
    audit.add_argument("artifact", nargs="+")
    audit.set_defaults(func=audit_artifacts)

    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
