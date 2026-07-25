#!/usr/bin/env python3
"""Build a deterministic Traditional-Chinese BFCL v4 subset.

The source directory must be the ``bfcl_eval/data`` directory from the
official Gorilla repository.  Selection is completed before translation so
that a translated artifact has a stable, auditable set of test IDs.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
import random
import shutil
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Iterable

import wandb


VERSION_PREFIX = "BFCL_v4"
DEFAULT_SEED = 20260724
DEFAULT_MAX_PER_CATEGORY = 30
MAX_MULTI_TURN_TURNS = 3

CORE_CATEGORIES = (
    "simple_python",
    "simple_java",
    "simple_javascript",
    "multiple",
    "irrelevance",
    "live_simple",
    "live_multiple",
    "live_irrelevance",
    "live_relevance",
    "multi_turn_base",
    "multi_turn_miss_func",
    "multi_turn_miss_param",
)
AGENTIC_CATEGORIES = (
    "memory_kv",
    "memory_vector",
    "memory_rec_sum",
    "web_search_base",
    "web_search_no_snippet",
)
EXCLUDED_CATEGORIES = (
    "parallel",
    "parallel_multiple",
    "live_parallel",
    "live_parallel_multiple",
    "multi_turn_long_context",
    "format_sensitivity",
)

SHARED_SOURCE_FILE = {
    "memory_kv": "memory",
    "memory_vector": "memory",
    "memory_rec_sum": "memory",
    "web_search_base": "web_search",
    "web_search_no_snippet": "web_search",
}

TRANSLATABLE_ROLES = {"system", "user"}
NATURAL_LANGUAGE_SCHEMA_KEYS = {"description", "title"}
TRANSLATION_CACHE_VERSION = "bfcl-v4-zh-tw-v1"
DEFAULT_TRANSLATION_BATCH_SIZE = 24
DEFAULT_TRANSLATION_BATCH_CHARS = 12_000


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                value = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"invalid JSONL at {path}:{line_number}") from exc
            if not isinstance(value, dict):
                raise ValueError(f"expected an object at {path}:{line_number}")
            rows.append(value)
    return rows


def write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def category_source_name(category: str) -> str:
    return SHARED_SOURCE_FILE.get(category, category)


def category_source_path(source_dir: Path, category: str) -> Path:
    return source_dir / f"{VERSION_PREFIX}_{category_source_name(category)}.json"


def load_category_rows(source_dir: Path, category: str) -> list[dict[str, Any]]:
    rows = read_jsonl(category_source_path(source_dir, category))
    if category.startswith("multi_turn_"):
        rows = [
            row
            for row in rows
            if isinstance(row.get("question"), list)
            and len(row["question"]) <= MAX_MULTI_TURN_TURNS
        ]
    return rows


def stable_sample(
    rows: list[dict[str, Any]],
    *,
    category: str,
    seed: int,
    max_per_category: int,
) -> list[dict[str, Any]]:
    rows = sorted(rows, key=lambda row: str(row["id"]))
    if len(rows) <= max_per_category:
        return rows
    rng = random.Random(f"{seed}:{category}")
    selected = rng.sample(rows, max_per_category)
    return sorted(selected, key=lambda row: str(row["id"]))


def select_rows(
    source_dir: Path,
    categories: tuple[str, ...],
    *,
    seed: int,
    max_per_category: int,
) -> tuple[dict[str, list[dict[str, Any]]], dict[str, list[str]]]:
    selected: dict[str, list[dict[str, Any]]] = {}
    selected_ids: dict[str, list[str]] = {}
    shared_selection: dict[str, list[dict[str, Any]]] = {}

    for category in categories:
        source_name = category_source_name(category)
        if source_name in shared_selection:
            rows = copy.deepcopy(shared_selection[source_name])
        else:
            rows = stable_sample(
                load_category_rows(source_dir, category),
                category=source_name,
                seed=seed,
                max_per_category=max_per_category,
            )
            if source_name in {"memory", "web_search"}:
                shared_selection[source_name] = copy.deepcopy(rows)
        selected[category] = rows
        selected_ids[category] = [str(row["id"]) for row in rows]

    return selected, selected_ids


def merge_rows_for_files(
    selected: dict[str, list[dict[str, Any]]],
) -> dict[str, list[dict[str, Any]]]:
    files: dict[str, dict[str, dict[str, Any]]] = {}
    for category, rows in selected.items():
        source_name = category_source_name(category)
        by_id = files.setdefault(source_name, {})
        for row in rows:
            by_id[str(row["id"])] = copy.deepcopy(row)
    return {
        source_name: [by_id[row_id] for row_id in sorted(by_id)]
        for source_name, by_id in sorted(files.items())
    }


def collect_ground_truth_rows(
    source_dir: Path,
    selected_files: dict[str, list[dict[str, Any]]],
) -> dict[str, list[dict[str, Any]]]:
    output: dict[str, list[dict[str, Any]]] = {}
    possible_dir = source_dir / "possible_answer"
    for source_name, rows in selected_files.items():
        source_path = possible_dir / f"{VERSION_PREFIX}_{source_name}.json"
        if not source_path.exists():
            continue
        selected_ids = {str(row["id"]) for row in rows}
        ground_truth = read_jsonl(source_path)
        filtered = [
            row for row in ground_truth if str(row.get("id")) in selected_ids
        ]
        found_ids = {str(row.get("id")) for row in filtered}
        missing = selected_ids - found_ids
        if missing:
            raise ValueError(
                f"{source_name}: {len(missing)} selected IDs have no ground truth: "
                f"{sorted(missing)[:5]}"
            )
        output[source_name] = sorted(filtered, key=lambda row: str(row["id"]))
    return output


def iter_message_contents(value: Any) -> Iterable[str]:
    if not isinstance(value, list):
        return
    for turn in value:
        if not isinstance(turn, list):
            continue
        for message in turn:
            if (
                isinstance(message, dict)
                and message.get("role") in TRANSLATABLE_ROLES
                and isinstance(message.get("content"), str)
                and message["content"].strip()
            ):
                yield message["content"]


def iter_schema_texts(value: Any) -> Iterable[str]:
    if isinstance(value, list):
        for item in value:
            yield from iter_schema_texts(item)
    elif isinstance(value, dict):
        for key, item in value.items():
            if (
                key in NATURAL_LANGUAGE_SCHEMA_KEYS
                and isinstance(item, str)
                and item.strip()
            ):
                yield item
            else:
                yield from iter_schema_texts(item)


def iter_agentic_answer_texts(
    ground_truth_files: dict[str, list[dict[str, Any]]],
) -> Iterable[str]:
    for source_name in ("memory", "web_search"):
        for row in ground_truth_files.get(source_name, []):
            for answer in row.get("ground_truth", []):
                if isinstance(answer, str) and answer.strip():
                    yield answer


def translation_key(model: str, text: str) -> str:
    payload = f"{TRANSLATION_CACHE_VERSION}\0{model}\0{text}".encode()
    return hashlib.sha256(payload).hexdigest()


def load_translation_cache(path: Path) -> dict[str, str]:
    if not path.exists():
        return {}
    cache: dict[str, str] = {}
    for row in read_jsonl(path):
        if isinstance(row.get("key"), str) and isinstance(row.get("translation"), str):
            cache[row["key"]] = row["translation"]
    return cache


def append_translation_cache(
    path: Path, *, key: str, source: str, translation: str
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(
            json.dumps(
                {
                    "key": key,
                    "source": source,
                    "translation": translation,
                },
                ensure_ascii=False,
                sort_keys=True,
            )
            + "\n"
        )


def translation_batches(
    missing: dict[str, str],
    *,
    max_items: int,
    max_chars: int,
) -> list[dict[str, str]]:
    if max_items < 1:
        raise ValueError("translation batch size must be positive")
    if max_chars < 1:
        raise ValueError("translation batch character limit must be positive")

    batches: list[dict[str, str]] = []
    current: dict[str, str] = {}
    current_chars = 0
    for key, source in sorted(missing.items()):
        source_chars = len(source)
        if current and (
            len(current) >= max_items or current_chars + source_chars > max_chars
        ):
            batches.append(current)
            current = {}
            current_chars = 0
        current[key] = source
        current_chars += source_chars
    if current:
        batches.append(current)
    return batches


def translate_batch(
    client: Any, model: str, batch: dict[str, str]
) -> dict[str, str]:
    items = [{"id": key, "text": text} for key, text in batch.items()]
    prompt = (
        "Translate every benchmark text below into natural Traditional Chinese for "
        "Taiwanese readers. Preserve code, JSON, LaTeX, URLs, identifiers, function "
        "names, parameter names, enum values, quoted literal values, numbers, units, "
        "and proper-name spellings when those spellings may be function arguments. "
        "Return only a JSON object with this exact shape: "
        '{"translations":[{"id":"the unchanged input id","text":"translation"}]}. '
        "Return exactly one item for every input id; do not reorder, omit, duplicate, "
        "or alter ids.\n\n"
        + json.dumps({"items": items}, ensure_ascii=False)
    )
    for attempt in range(4):
        try:
            response = client.responses.create(
                model=model,
                input=[{"role": "user", "content": prompt}],
                max_output_tokens=16_384,
                text={
                    "format": {
                        "type": "json_schema",
                        "name": "bfcl_translation_batch",
                        "strict": True,
                        "schema": {
                            "type": "object",
                            "properties": {
                                "translations": {
                                    "type": "array",
                                    "items": {
                                        "type": "object",
                                        "properties": {
                                            "id": {"type": "string"},
                                            "text": {"type": "string"},
                                        },
                                        "required": ["id", "text"],
                                        "additionalProperties": False,
                                    },
                                }
                            },
                            "required": ["translations"],
                            "additionalProperties": False,
                        },
                    }
                },
            )
            payload = json.loads(response.output_text.strip())
            translated_items = payload.get("translations")
            if not isinstance(translated_items, list):
                raise ValueError("translation response has no translations list")
            translated: dict[str, str] = {}
            for item in translated_items:
                if not isinstance(item, dict):
                    raise ValueError("translation response item is not an object")
                key = item.get("id")
                text = item.get("text")
                if (
                    not isinstance(key, str)
                    or key in translated
                    or not isinstance(text, str)
                    or not text.strip()
                ):
                    raise ValueError("invalid or duplicate translation response item")
                translated[key] = text.strip()
            if list(translated) != list(batch):
                raise ValueError(
                    "translation response IDs/order do not match the request"
                )
            return translated
        except Exception:
            if attempt == 3:
                if len(batch) == 1:
                    raise
                items = list(batch.items())
                midpoint = len(items) // 2
                left = translate_batch(client, model, dict(items[:midpoint]))
                right = translate_batch(client, model, dict(items[midpoint:]))
                return {**left, **right}
            time.sleep(min(2**attempt, 15))
    raise RuntimeError("unreachable")


def translate_messages(
    value: Any, translations: dict[str, str], model: str
) -> Any:
    output = copy.deepcopy(value)
    if not isinstance(output, list):
        return output
    for turn in output:
        if not isinstance(turn, list):
            continue
        for message in turn:
            if (
                isinstance(message, dict)
                and message.get("role") in TRANSLATABLE_ROLES
                and isinstance(message.get("content"), str)
                and message["content"].strip()
            ):
                key = translation_key(model, message["content"])
                message["content"] = translations[key]
    return output


def translate_schema(value: Any, translations: dict[str, str], model: str) -> Any:
    if isinstance(value, list):
        return [translate_schema(item, translations, model) for item in value]
    if isinstance(value, dict):
        translated: dict[str, Any] = {}
        for key, item in value.items():
            if (
                key in NATURAL_LANGUAGE_SCHEMA_KEYS
                and isinstance(item, str)
                and item.strip()
            ):
                translated[key] = translations[translation_key(model, item)]
            else:
                translated[key] = translate_schema(item, translations, model)
        return translated
    return value


def translate_prompt_rows(
    rows: list[dict[str, Any]], translations: dict[str, str], model: str
) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    for row in rows:
        translated = copy.deepcopy(row)
        translated["question"] = translate_messages(
            translated.get("question"), translations, model
        )
        if "function" in translated:
            translated["function"] = translate_schema(
                translated["function"], translations, model
            )
        output.append(translated)
    return output


def augment_agentic_ground_truth(
    rows: list[dict[str, Any]],
    translations: dict[str, str],
    model: str,
) -> list[dict[str, Any]]:
    output = copy.deepcopy(rows)
    for row in output:
        original_answers = row.get("ground_truth", [])
        if not isinstance(original_answers, list):
            continue
        augmented: list[Any] = []
        for answer in original_answers:
            if answer not in augmented:
                augmented.append(answer)
            if isinstance(answer, str) and answer.strip():
                translated = translations[translation_key(model, answer)].strip()
                if translated and translated not in augmented:
                    augmented.append(translated)
        row["ground_truth"] = augmented
    return output


def collect_translation_texts(
    selected_files: dict[str, list[dict[str, Any]]],
    ground_truth_files: dict[str, list[dict[str, Any]]],
    source_dir: Path,
    *,
    include_support_files: bool,
) -> set[str]:
    texts: set[str] = set()
    for rows in selected_files.values():
        for row in rows:
            texts.update(iter_message_contents(row.get("question")))
            texts.update(iter_schema_texts(row.get("function")))

    if include_support_files:
        texts.update(iter_agentic_answer_texts(ground_truth_files))
        for path in sorted((source_dir / "multi_turn_func_doc").glob("*.json")):
            texts.update(iter_schema_texts(read_jsonl(path)))
        for path in sorted(
            (source_dir / "memory_prereq_conversation").glob("*.json")
        ):
            for row in read_jsonl(path):
                texts.update(iter_message_contents(row.get("question")))
    return {text for text in texts if text.strip()}


def ensure_translations(
    texts: set[str],
    *,
    model: str,
    cache_path: Path,
    max_workers: int,
    batch_size: int,
    batch_chars: int,
) -> dict[str, str]:
    from openai import OpenAI

    if not os.environ.get("OPENAI_API_KEY"):
        raise ValueError("OPENAI_API_KEY is required for --translate")
    client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])
    cache = load_translation_cache(cache_path)
    missing = {
        translation_key(model, text): text
        for text in texts
        if translation_key(model, text) not in cache
    }
    if missing:
        batches = translation_batches(
            missing,
            max_items=batch_size,
            max_chars=batch_chars,
        )
        print(
            f"Translating {len(missing)} unique BFCL v4 fields in "
            f"{len(batches)} validated JSON batches..."
        )
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_to_batch = {
                executor.submit(translate_batch, client, model, batch): batch
                for batch in batches
            }
            completed_fields = 0
            for index, future in enumerate(as_completed(future_to_batch), start=1):
                translations = future.result()
                for key, translation in translations.items():
                    cache[key] = translation
                    append_translation_cache(
                        cache_path,
                        key=key,
                        source=missing[key],
                        translation=translation,
                    )
                completed_fields += len(translations)
                if index % 10 == 0 or index == len(future_to_batch):
                    print(
                        f"Translated {completed_fields}/{len(missing)} fields "
                        f"({index}/{len(batches)} batches)"
                    )
    return cache


def upload_artifact(
    output_dir: Path,
    metadata: dict[str, Any],
    *,
    entity: str,
    project: str,
    artifact_name: str,
    aliases: list[str],
) -> None:
    with wandb.init(entity=entity, project=project, job_type="upload_data") as run:
        artifact = wandb.Artifact(
            name=artifact_name,
            type="dataset",
            metadata=metadata,
        )
        artifact.add_dir(str(output_dir), name="bfcl")
        run.log_artifact(artifact, aliases=aliases)
        artifact.wait()
        print(
            f"Uploaded {entity}/{project}/{artifact_name} "
            f"with aliases {', '.join(aliases)}"
        )


def copy_support_files(
    source_dir: Path,
    output_dir: Path,
    *,
    translations: dict[str, str] | None,
    model: str,
    include_agentic: bool,
) -> None:
    func_doc_source = source_dir / "multi_turn_func_doc"
    func_doc_target = output_dir / "multi_turn_func_doc"
    func_doc_target.mkdir(parents=True, exist_ok=True)
    for path in sorted(func_doc_source.glob("*.json")):
        if not include_agentic and path.name in {
            "memory_kv.json",
            "memory_vector.json",
            "memory_rec_sum.json",
            "web_search.json",
        }:
            continue
        value = read_jsonl(path)
        if translations is not None:
            value = translate_schema(value, translations, model)
        write_jsonl(func_doc_target / path.name, value)

    if include_agentic:
        prereq_target = output_dir / "memory_prereq_conversation"
        prereq_target.mkdir(parents=True, exist_ok=True)
        for path in sorted(
            (source_dir / "memory_prereq_conversation").glob("*.json")
        ):
            rows = read_jsonl(path)
            if translations is not None:
                rows = [
                    {
                        **row,
                        "question": translate_messages(
                            row.get("question"), translations, model
                        ),
                    }
                    for row in rows
                ]
            write_jsonl(prereq_target / path.name, rows)


def _without_schema_natural_language(value: Any) -> Any:
    if isinstance(value, list):
        return [_without_schema_natural_language(item) for item in value]
    if isinstance(value, dict):
        return {
            key: _without_schema_natural_language(item)
            for key, item in value.items()
            if key not in NATURAL_LANGUAGE_SCHEMA_KEYS
        }
    return value


def _message_structure(value: Any) -> Any:
    if not isinstance(value, list):
        return None
    return [
        [
            {key: item for key, item in message.items() if key != "content"}
            for message in turn
        ]
        for turn in value
    ]


def audit_generated_output(
    *,
    source_dir: Path,
    output_dir: Path,
    selected_files: dict[str, list[dict[str, Any]]],
    translation_texts: set[str],
    translations: dict[str, str] | None,
    model: str,
    include_agentic: bool,
) -> dict[str, Any]:
    issues: list[str] = []
    prompt_rows = 0
    for source_name, source_rows in selected_files.items():
        output_path = output_dir / f"{VERSION_PREFIX}_{source_name}.json"
        output_rows = {
            str(row["id"]): row for row in read_jsonl(output_path)
        }
        if set(output_rows) != {str(row["id"]) for row in source_rows}:
            issues.append(f"{source_name}: translated prompt IDs changed")
            continue
        for source_row in source_rows:
            prompt_rows += 1
            row_id = str(source_row["id"])
            output_row = output_rows[row_id]
            if _message_structure(source_row.get("question")) != _message_structure(
                output_row.get("question")
            ):
                issues.append(f"{row_id}: message structure changed")
            if _without_schema_natural_language(
                source_row.get("function")
            ) != _without_schema_natural_language(output_row.get("function")):
                issues.append(f"{row_id}: executable schema fields changed")

    for output_path in sorted((output_dir / "multi_turn_func_doc").glob("*.json")):
        source_path = source_dir / "multi_turn_func_doc" / output_path.name
        if _without_schema_natural_language(
            read_jsonl(source_path)
        ) != _without_schema_natural_language(read_jsonl(output_path)):
            issues.append(
                f"multi_turn_func_doc/{output_path.name}: executable fields changed"
            )

    if include_agentic:
        for output_path in sorted(
            (output_dir / "memory_prereq_conversation").glob("*.json")
        ):
            source_path = (
                source_dir / "memory_prereq_conversation" / output_path.name
            )
            source_rows = read_jsonl(source_path)
            output_rows = read_jsonl(output_path)
            if len(source_rows) != len(output_rows):
                issues.append(
                    f"memory_prereq_conversation/{output_path.name}: row count changed"
                )
                continue
            for index, (source_row, output_row) in enumerate(
                zip(source_rows, output_rows)
            ):
                if _message_structure(
                    source_row.get("question")
                ) != _message_structure(output_row.get("question")):
                    issues.append(
                        "memory_prereq_conversation/"
                        f"{output_path.name}:{index}: message structure changed"
                    )

    translated_outputs = (
        [translations[translation_key(model, text)] for text in translation_texts]
        if translations is not None
        else []
    )
    source_with_letters = [
        text for text in translation_texts if any(char.isalpha() for char in text)
    ]
    changed_letter_fields = (
        sum(
            translations[translation_key(model, text)].strip() != text.strip()
            for text in source_with_letters
        )
        if translations is not None
        else 0
    )
    japanese_kana_chars = sum(
        1
        for text in translated_outputs
        for char in text
        if "\u3040" <= char <= "\u30ff"
    )
    if translations is not None and japanese_kana_chars:
        issues.append(
            f"translated fields contain {japanese_kana_chars} Japanese kana characters"
        )
    if issues:
        raise ValueError(
            "BFCL v4 translation audit failed: " + "; ".join(issues[:10])
        )

    return {
        "physical_prompt_rows": prompt_rows,
        "translation_field_count": len(translation_texts),
        "source_fields_with_letters": len(source_with_letters),
        "changed_letter_fields": changed_letter_fields,
        "changed_letter_rate": (
            changed_letter_fields / len(source_with_letters)
            if source_with_letters
            else 1.0
        ),
        "translated_fields_with_cjk": sum(
            any("\u3400" <= char <= "\u9fff" for char in text)
            for text in translated_outputs
        ),
        "japanese_kana_chars": japanese_kana_chars,
        "structural_issue_count": 0,
    }


def build(args: argparse.Namespace) -> dict[str, Any]:
    source_dir = args.source_dir.resolve()
    if not source_dir.exists():
        raise FileNotFoundError(source_dir)

    categories = CORE_CATEGORIES
    if args.profile == "full":
        categories += AGENTIC_CATEGORIES

    selected, selected_ids = select_rows(
        source_dir,
        categories,
        seed=args.seed,
        max_per_category=args.max_per_category,
    )
    selected_files = merge_rows_for_files(selected)
    ground_truth_files = collect_ground_truth_rows(source_dir, selected_files)
    scored_count = sum(len(selected[category]) for category in categories)

    metadata: dict[str, Any] = {
        "benchmark": "BFCL",
        "benchmark_version": "v4",
        "profile": args.profile,
        "locale": "zh-Hant-TW" if args.translate else "en",
        "upstream_repository": "https://github.com/ShishirPatil/gorilla",
        "upstream_commit": args.upstream_commit,
        "selection_seed": args.seed,
        "max_per_category": args.max_per_category,
        "max_multi_turn_turns": MAX_MULTI_TURN_TURNS,
        "categories": list(categories),
        "excluded_categories": list(EXCLUDED_CATEGORIES),
        "selected_ids": selected_ids,
        "category_counts": {
            category: len(selected[category]) for category in categories
        },
        "scored_case_count": scored_count,
        "translation_model": args.translation_model if args.translate else None,
    }

    if args.audit_only:
        return metadata

    output_dir = args.output_dir.resolve()
    if output_dir.exists():
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True)

    translations: dict[str, str] | None = None
    translation_texts: set[str] = set()
    if args.translate:
        translation_texts = collect_translation_texts(
            selected_files,
            ground_truth_files,
            source_dir,
            include_support_files=args.profile == "full",
        )
        translations = ensure_translations(
            translation_texts,
            model=args.translation_model,
            cache_path=args.cache_path.resolve(),
            max_workers=args.max_workers,
            batch_size=args.translation_batch_size,
            batch_chars=args.translation_batch_chars,
        )
        metadata["translated_unique_fields"] = len(translation_texts)

    for source_name, rows in selected_files.items():
        if translations is not None:
            rows = translate_prompt_rows(rows, translations, args.translation_model)
        write_jsonl(output_dir / f"{VERSION_PREFIX}_{source_name}.json", rows)

    for source_name, rows in ground_truth_files.items():
        # Tool arguments are intentionally not translated. They remain aligned
        # with enum values, identifiers, and executable backend state.
        if translations is not None and source_name in {"memory", "web_search"}:
            rows = augment_agentic_ground_truth(
                rows, translations, args.translation_model
            )
        write_jsonl(
            output_dir
            / "possible_answer"
            / f"{VERSION_PREFIX}_{source_name}.json",
            rows,
        )

    copy_support_files(
        source_dir,
        output_dir,
        translations=translations,
        model=args.translation_model,
        include_agentic=args.profile == "full",
    )
    metadata["translation_audit"] = audit_generated_output(
        source_dir=source_dir,
        output_dir=output_dir,
        selected_files=selected_files,
        translation_texts=translation_texts,
        translations=translations,
        model=args.translation_model,
        include_agentic=args.profile == "full",
    )

    metadata["files"] = {
        str(path.relative_to(output_dir)): {
            "sha256": sha256_file(path),
            "bytes": path.stat().st_size,
        }
        for path in sorted(output_dir.rglob("*"))
        if path.is_file()
    }
    write_json(output_dir / "selection_metadata.json", metadata)
    if args.upload:
        if not args.translate:
            raise ValueError("--upload requires --translate")
        if not args.entity or not args.project:
            raise ValueError("--upload requires --entity and --project")
        upload_artifact(
            output_dir,
            metadata,
            entity=args.entity,
            project=args.project,
            artifact_name=args.artifact_name,
            aliases=args.alias,
        )
    return metadata


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--source-dir",
        type=Path,
        required=True,
        help="Official gorilla/berkeley-function-call-leaderboard/bfcl_eval/data",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/taiwan/bfcl-v4-zh-tw/bfcl"),
    )
    parser.add_argument("--profile", choices=("core", "full"), default="full")
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument(
        "--max-per-category", type=int, default=DEFAULT_MAX_PER_CATEGORY
    )
    parser.add_argument("--upstream-commit", default="unknown")
    parser.add_argument(
        "--translation-model", default="gpt-5.4-mini-2026-03-17"
    )
    parser.add_argument(
        "--cache-path",
        type=Path,
        default=Path("data/taiwan/bfcl_v4_zh_tw_translation_cache.jsonl"),
    )
    parser.add_argument("--max-workers", type=int, default=4)
    parser.add_argument(
        "--translation-batch-size",
        type=int,
        default=DEFAULT_TRANSLATION_BATCH_SIZE,
    )
    parser.add_argument(
        "--translation-batch-chars",
        type=int,
        default=DEFAULT_TRANSLATION_BATCH_CHARS,
    )
    parser.add_argument(
        "--translate", action=argparse.BooleanOptionalAction, default=False
    )
    parser.add_argument("--audit-only", action="store_true")
    parser.add_argument("--upload", action="store_true")
    parser.add_argument("--entity")
    parser.add_argument("--project")
    parser.add_argument("--artifact-name", default="bfcl-v4-zh-tw")
    parser.add_argument(
        "--alias",
        action="append",
        default=["production"],
        help="Artifact alias; may be repeated.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    metadata = build(args)
    print(json.dumps(metadata, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
