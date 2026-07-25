#!/usr/bin/env python3
"""Prepare and merge targeted Agentic SWE Low/Middle budget-stop repairs."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

from openclaw_usage import merge_prior_billable_openclaw_usage


BUDGET_STOP_REASON = "runtime_budget_exceeded"
SCOREABLE_STOP_REASONS = {
    BUDGET_STOP_REASON,
    "conversation_order_violation",
    "time_up",
    "model_output_truncated",
    "openclaw_no_response",
}


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def unique_by_id(
    rows: list[dict[str, Any]],
    *,
    id_key: str,
    label: str,
) -> dict[str, dict[str, Any]]:
    by_id: dict[str, dict[str, Any]] = {}
    for row in rows:
        if not isinstance(row, dict):
            raise ValueError(f"{label} contains a non-object row")
        instance_id = str(row.get(id_key) or "")
        if not instance_id:
            raise ValueError(f"{label} contains a row without {id_key}")
        if instance_id in by_id:
            raise ValueError(f"{label} contains duplicate {id_key}: {instance_id}")
        by_id[instance_id] = row
    return by_id


def record_model(record: dict[str, Any]) -> str:
    cache_key = record.get("cache_key")
    return (
        str(cache_key.get("model") or "")
        if isinstance(cache_key, dict)
        else ""
    )


def budget_stop_ids(records: list[dict[str, Any]]) -> list[str]:
    return [
        str(record["instance_id"])
        for record in records
        if record.get("openclaw_disqualified_reason") == BUDGET_STOP_REASON
    ]


def prepare(args: argparse.Namespace) -> None:
    dataset_rows = read_jsonl(args.dataset_jsonl.resolve())
    source_records = read_json(args.source_patches_json.resolve())
    if not isinstance(source_records, list):
        raise ValueError("--source-patches-json must contain a JSON list")
    dataset_by_id = unique_by_id(
        dataset_rows,
        id_key="instance_id",
        label="dataset",
    )
    source_by_id = unique_by_id(
        source_records,
        id_key="instance_id",
        label="source patches",
    )
    missing_source = sorted(set(dataset_by_id) - set(source_by_id))
    unexpected_source = sorted(set(source_by_id) - set(dataset_by_id))
    if missing_source or unexpected_source:
        raise ValueError(
            "Source patch coverage does not match the dataset: "
            f"missing={missing_source}, unexpected={unexpected_source}"
        )
    selected_ids = budget_stop_ids(source_records)
    if not selected_ids:
        raise ValueError("Source patches contain no runtime budget stops to repair")
    selected_rows = [dataset_by_id[instance_id] for instance_id in selected_ids]
    output_dataset_path = args.output_dataset_jsonl.resolve()
    output_ids_path = (
        args.output_instance_ids_json.resolve()
        if args.output_instance_ids_json
        else output_dataset_path.with_name(
            f"{output_dataset_path.stem}_instance_ids.json"
        )
    )
    write_jsonl(output_dataset_path, selected_rows)
    write_json(output_ids_path, selected_ids)
    state_path = args.state_json or args.output_dataset_jsonl.with_suffix(".state.json")
    write_json(
        state_path.resolve(),
        {
            "status": "prepared",
            "created_at": time.time(),
            "source_dataset_jsonl": str(args.dataset_jsonl.resolve()),
            "source_patches_json": str(args.source_patches_json.resolve()),
            "output_dataset_jsonl": str(output_dataset_path),
            "output_instance_ids_json": str(output_ids_path),
            "source_count": len(source_records),
            "repair_count": len(selected_ids),
            "repair_instance_ids": selected_ids,
            "models": sorted(
                {record_model(record) for record in source_records if record_model(record)}
            ),
        },
    )
    print(
        f"Prepared {len(selected_ids)} budget-stop repairs at "
        f"{output_dataset_path}; IDs at {output_ids_path}"
    )


def validate_repair_record(
    repair: dict[str, Any],
    source: dict[str, Any],
) -> None:
    instance_id = str(repair["instance_id"])
    source_model = record_model(source)
    repair_model = record_model(repair)
    if not source_model or repair_model != source_model:
        raise ValueError(
            f"Model mismatch for repair {instance_id}: "
            f"source={source_model or 'missing'}, repair={repair_model or 'missing'}"
        )
    reason = str(repair.get("openclaw_disqualified_reason") or "")
    if reason and reason not in SCOREABLE_STOP_REASONS:
        raise ValueError(f"Repair remains unscoreable for {instance_id}: {reason}")
    if repair.get("weave_agents_ok") is not True:
        raise ValueError(f"Repair native trace verification failed for {instance_id}")
    if repair.get("nemoclaw_session_audit_ok") is not True:
        raise ValueError(f"Repair NeMoClaw session audit failed for {instance_id}")
    if "patch" not in repair:
        raise ValueError(f"Repair is missing patch field for {instance_id}")


def merge(args: argparse.Namespace) -> None:
    dataset_rows = read_jsonl(args.dataset_jsonl.resolve())
    source_records = read_json(args.source_patches_json.resolve())
    repair_records = read_json(args.repair_patches_json.resolve())
    if not isinstance(source_records, list) or not isinstance(repair_records, list):
        raise ValueError("Source and repair patch files must each contain a JSON list")
    dataset_by_id = unique_by_id(dataset_rows, id_key="instance_id", label="dataset")
    source_by_id = unique_by_id(
        source_records,
        id_key="instance_id",
        label="source patches",
    )
    repair_by_id = unique_by_id(
        repair_records,
        id_key="instance_id",
        label="repair patches",
    )
    expected_repair_ids = set(budget_stop_ids(source_records))
    if set(repair_by_id) != expected_repair_ids:
        raise ValueError(
            "Repair patch coverage must exactly match source budget stops: "
            f"missing={sorted(expected_repair_ids - set(repair_by_id))}, "
            f"unexpected={sorted(set(repair_by_id) - expected_repair_ids)}"
        )
    if set(source_by_id) != set(dataset_by_id):
        raise ValueError("Source patch coverage does not match the complete dataset")

    merged_by_id = dict(source_by_id)
    for instance_id in sorted(expected_repair_ids):
        source = source_by_id[instance_id]
        repair = repair_by_id[instance_id]
        validate_repair_record(repair, source)
        merged = merge_prior_billable_openclaw_usage(repair, source)
        merged["budget_repair"] = {
            "source_reason": source.get("openclaw_disqualified_reason"),
            "source_cache_key": source.get("cache_key"),
            "repair_cache_key": repair.get("cache_key"),
            "merged_at": time.time(),
        }
        merged_by_id[instance_id] = merged

    ordered = [merged_by_id[str(row["instance_id"])] for row in dataset_rows]
    output_path = args.output_patches_json.resolve()
    write_json(output_path, ordered)
    write_jsonl(output_path.with_suffix(".jsonl"), ordered)
    residual_budget_stops = budget_stop_ids(ordered)
    state_path = args.state_json or output_path.with_suffix(".repair.json")
    write_json(
        state_path.resolve(),
        {
            "status": "merged",
            "created_at": time.time(),
            "source_patches_json": str(args.source_patches_json.resolve()),
            "repair_patches_json": str(args.repair_patches_json.resolve()),
            "output_patches_json": str(output_path),
            "total_count": len(ordered),
            "repaired_count": len(expected_repair_ids),
            "repaired_instance_ids": sorted(expected_repair_ids),
            "residual_budget_stop_count": len(residual_budget_stops),
            "residual_budget_stop_instance_ids": residual_budget_stops,
        },
    )
    print(
        f"Merged {len(expected_repair_ids)} repairs into {len(ordered)} patches at "
        f"{output_path}; residual budget stops={len(residual_budget_stops)}"
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    prepare_parser = subparsers.add_parser(
        "prepare",
        help="Extract only runtime-budget-stopped instances into a repair dataset.",
    )
    prepare_parser.add_argument("--dataset-jsonl", type=Path, required=True)
    prepare_parser.add_argument("--source-patches-json", type=Path, required=True)
    prepare_parser.add_argument("--output-dataset-jsonl", type=Path, required=True)
    prepare_parser.add_argument("--output-instance-ids-json", type=Path)
    prepare_parser.add_argument("--state-json", type=Path)
    prepare_parser.set_defaults(handler=prepare)

    merge_parser = subparsers.add_parser(
        "merge",
        help="Merge repaired patches and retry-inclusive usage into the complete set.",
    )
    merge_parser.add_argument("--dataset-jsonl", type=Path, required=True)
    merge_parser.add_argument("--source-patches-json", type=Path, required=True)
    merge_parser.add_argument("--repair-patches-json", type=Path, required=True)
    merge_parser.add_argument("--output-patches-json", type=Path, required=True)
    merge_parser.add_argument("--state-json", type=Path)
    merge_parser.set_defaults(handler=merge)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.handler(args)


if __name__ == "__main__":
    main()
