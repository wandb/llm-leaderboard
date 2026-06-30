#!/usr/bin/env python3
"""Preflight-check a W&B scope attestation before paid-review sync.

This is an offline verifier. It does not query W&B, write W&B, mutate paid-run
review JSONs, or run model inference. It validates that a human-confirmed
scope-attestation JSON is concrete and bound to the exact W&B completion
verifier JSON before sync_wandb_completion_to_paid_review.py is allowed to
mutate a review record.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
import time
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[2]
SYNC_TOOL = REPO_ROOT / "scripts" / "tools" / "sync_wandb_completion_to_paid_review.py"


def load_sync_tool() -> Any:
    spec = importlib.util.spec_from_file_location("sync_wandb_completion_to_paid_review", SYNC_TOOL)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


_sync_tool = load_sync_tool()

compact_entry_for_report = _sync_tool.compact_entry_for_report
completion_entry = _sync_tool.completion_entry
path_display = _sync_tool.path_display
read_json_object = _sync_tool.read_json_object
repo_path = _sync_tool.repo_path
validate_scope_attestation_json = _sync_tool.validate_scope_attestation_json
write_json = _sync_tool.write_json
sha256_file = _sync_tool.sha256_file


def source_file_record(path: Path) -> dict[str, Any]:
    record: dict[str, Any] = {
        "path": path_display(path),
        "readable": False,
        "sha256": "",
    }
    try:
        record["sha256"] = sha256_file(path)
        record["readable"] = True
    except OSError as exc:
        record["error"] = str(exc)
    return record


def build_report(
    *,
    review_path: Path,
    completion_path: Path,
    attestation_path: Path,
    entry: dict[str, Any] | None,
    scope_attestation: dict[str, Any] | None,
    errors: list[str],
) -> dict[str, Any]:
    ok = not errors
    return {
        "ok": ok,
        "status": "passed" if ok else "validation_failed",
        "generated_at": time.time(),
        "review_path": path_display(review_path),
        "completion_json": path_display(completion_path),
        "scope_attestation_json": path_display(attestation_path),
        "source_files": {
            "review_json": source_file_record(review_path),
            "completion_json": source_file_record(completion_path),
            "scope_attestation_json": source_file_record(attestation_path),
        },
        "entry": compact_entry_for_report(entry) if isinstance(entry, dict) else None,
        "scope_attestation": scope_attestation,
        "errors": errors,
    }


def verify_scope_attestation(
    *,
    review_path: Path,
    completion_path: Path,
    attestation_path: Path,
    allow_failed: bool = False,
    require_query_source: bool = False,
) -> dict[str, Any]:
    entry: dict[str, Any] | None = None
    scope: dict[str, Any] | None = None
    errors: list[str] = []
    try:
        read_json_object(review_path)
        entry = completion_entry(
            completion_path,
            read_json_object(completion_path),
            allow_failed=allow_failed,
            require_query_source=require_query_source,
        )
        scope = validate_scope_attestation_json(
            path=attestation_path,
            payload=read_json_object(attestation_path),
            entry=entry,
            review_path=review_path,
        )
    except (OSError, json.JSONDecodeError, ValueError) as exc:
        errors.append(str(exc))
    return build_report(
        review_path=review_path,
        completion_path=completion_path,
        attestation_path=attestation_path,
        entry=entry,
        scope_attestation=scope,
        errors=errors,
    )


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--review-json", type=Path, required=True)
    parser.add_argument("--completion-json", type=Path, required=True)
    parser.add_argument("--scope-attestation-json", type=Path, required=True)
    parser.add_argument("--json", type=Path, help="Write the preflight report JSON.")
    parser.add_argument("--allow-failed", action="store_true")
    parser.add_argument(
        "--require-query-source",
        action="store_true",
        help="Also require current W&B query_source provenance in the completion JSON.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    report = verify_scope_attestation(
        review_path=repo_path(args.review_json),
        completion_path=repo_path(args.completion_json),
        attestation_path=repo_path(args.scope_attestation_json),
        allow_failed=bool(args.allow_failed),
        require_query_source=bool(args.require_query_source),
    )
    if args.json:
        write_json(repo_path(args.json), report)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    if report.get("ok") is not True:
        raise SystemExit(1)


if __name__ == "__main__":
    main(sys.argv[1:])
