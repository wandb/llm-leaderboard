from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def validate_agentic_math_completion(
    output_dir: Path,
    *,
    expected_count: int,
    expected_model: str,
    require_weave_agents: bool,
    require_nemoclaw_session_audit: bool,
) -> dict[str, Any]:
    runner_dir = output_dir / "openclaw"
    summary_path = runner_dir / "summary.json"
    results_path = runner_dir / "results.jsonl"
    errors: list[str] = []

    try:
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        return {
            "ok": False,
            "errors": [f"invalid or missing summary.json: {exc}"],
            "runner_dir": str(runner_dir),
        }

    rows = []
    try:
        for line in results_path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                rows.append(json.loads(line))
    except (OSError, json.JSONDecodeError) as exc:
        errors.append(f"invalid or missing results.jsonl: {exc}")

    summary_total = int(summary.get("total_instances") or -1)
    if summary_total != expected_count:
        errors.append(
            f"summary total_instances mismatch ({summary_total} != {expected_count})"
        )
    if len(rows) != expected_count:
        errors.append(f"result row count mismatch ({len(rows)} != {expected_count})")

    task_ids = [str(row.get("task_id") or "") for row in rows]
    if any(not task_id for task_id in task_ids):
        errors.append("one or more result rows have no task_id")
    if len(task_ids) != len(set(task_ids)):
        errors.append("result rows contain duplicate task_id values")

    mismatched_models = sorted(
        {
            str((row.get("cache_key") or {}).get("model") or "")
            for row in rows
            if str((row.get("cache_key") or {}).get("model") or "")
            != expected_model
        }
    )
    if mismatched_models:
        errors.append(
            "result model mismatch "
            f"(expected={expected_model!r}, found={mismatched_models})"
        )

    if require_weave_agents:
        required = int(summary.get("weave_agents_required_instances") or 0)
        passed = int(summary.get("weave_agents_passed_instances") or 0)
        failed = int(summary.get("weave_agents_failed_instances") or 0)
        if (required, passed, failed) != (expected_count, expected_count, 0):
            errors.append(
                "Weave evidence coverage mismatch "
                f"(required={required}, passed={passed}, failed={failed})"
            )
        if any(row.get("weave_agents_ok") is not True for row in rows):
            errors.append("one or more result rows failed Weave evidence validation")

    if require_nemoclaw_session_audit:
        required = int(
            summary.get("nemoclaw_session_audit_required_instances") or 0
        )
        passed = int(summary.get("nemoclaw_session_audit_passed_instances") or 0)
        failed = int(summary.get("nemoclaw_session_audit_failed_instances") or 0)
        if (required, passed, failed) != (expected_count, expected_count, 0):
            errors.append(
                "NeMoClaw session audit coverage mismatch "
                f"(required={required}, passed={passed}, failed={failed})"
            )
        if any(row.get("nemoclaw_session_audit_ok") is not True for row in rows):
            errors.append("one or more result rows failed NeMoClaw session audit")

    return {
        "ok": not errors,
        "errors": errors,
        "runner_dir": str(runner_dir),
        "summary_path": str(summary_path),
        "results_path": str(results_path),
        "result_count": len(rows),
        "expected_count": expected_count,
        "expected_model": expected_model,
    }
