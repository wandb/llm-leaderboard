#!/usr/bin/env python3
"""Log existing SWE-Bench Pro results to W&B without running inference.

This is a recovery/audit path for completed local SWE-Bench Pro official evals.
It uses the same W&B keys as scripts/evaluator/swebench_pro.py so
verify_taiwan_wandb_completion.py can prove completion afterward.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any

import pandas as pd
import wandb

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from relog_wandb_approval import validate_external_action_approval_for_wandb_write


REQUIRED_SUMMARY_KEYS = {
    "total_instances",
    "resolved_instances",
    "unresolved_instances",
    "pass_at_1",
    "resolved_ids",
    "unresolved_ids",
}


def sanitize_artifact_component(value: str) -> str:
    return (
        value.replace("/", "-")
        .replace(":", "-")
        .replace(" ", "-")
        .replace("_", "-")
        .lower()
    )


def read_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"{path} must contain a JSON object")
    return payload


def read_eval_results(path: Path) -> dict[str, bool]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"{path} must contain a JSON object")
    return {str(key): bool(value) for key, value in payload.items()}


def read_patch_rows(path: Path) -> list[dict[str, Any]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, list):
        raise ValueError(f"{path} must contain a JSON list")
    rows: list[dict[str, Any]] = []
    for index, row in enumerate(payload, start=1):
        if not isinstance(row, dict):
            raise ValueError(f"{path}:{index} must contain a JSON object")
        rows.append(row)
    return rows


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def source_sha256s(*, official_eval_dir: Path, patch_path: Path) -> dict[str, str]:
    result = {
        "summary_json": sha256_file(official_eval_dir / "summary.json"),
        "patch_path": sha256_file(patch_path),
    }
    eval_results_path = official_eval_dir / "eval_results.json"
    if eval_results_path.exists():
        result["eval_results_json"] = sha256_file(eval_results_path)
    return result


def validate_summary(
    summary: dict[str, Any],
    *,
    eval_results: dict[str, bool] | None,
    expected_total: int | None,
) -> None:
    missing = sorted(REQUIRED_SUMMARY_KEYS - set(summary))
    if missing:
        raise ValueError(f"summary.json is missing required keys: {missing}")

    total = int(summary["total_instances"])
    resolved = int(summary["resolved_instances"])
    unresolved = int(summary["unresolved_instances"])
    resolved_ids = sorted(map(str, summary["resolved_ids"]))
    unresolved_ids = sorted(map(str, summary["unresolved_ids"]))
    all_ids = sorted(resolved_ids + unresolved_ids)
    mismatches: list[str] = []

    if expected_total is not None and total != expected_total:
        mismatches.append(f"total_instances={total} but expected_total={expected_total}")
    if total != resolved + unresolved:
        mismatches.append("total_instances does not equal resolved_instances + unresolved_instances")
    if resolved != len(resolved_ids):
        mismatches.append("resolved_instances does not match resolved_ids length")
    if unresolved != len(unresolved_ids):
        mismatches.append("unresolved_instances does not match unresolved_ids length")
    if total != len(all_ids):
        mismatches.append("total_instances does not match resolved_ids + unresolved_ids length")
    expected_pass_at_1 = resolved / total if total else 0.0
    if abs(float(summary["pass_at_1"]) - expected_pass_at_1) > 1e-12:
        mismatches.append(f"pass_at_1={summary['pass_at_1']} but resolved/total={expected_pass_at_1}")

    if eval_results is not None:
        eval_resolved_ids = sorted(key for key, value in eval_results.items() if value)
        eval_unresolved_ids = sorted(key for key, value in eval_results.items() if not value)
        if sorted(eval_results) != all_ids:
            mismatches.append("eval_results.json instance ids do not match summary ids")
        if eval_resolved_ids != resolved_ids:
            mismatches.append("eval_results.json resolved ids do not match summary")
        if eval_unresolved_ids != unresolved_ids:
            mismatches.append("eval_results.json unresolved ids do not match summary")

    if mismatches:
        raise ValueError("; ".join(mismatches))


def build_leaderboard(
    *,
    model_name: str,
    summary: dict[str, Any],
    benchmark_name: str,
    result_source: str,
) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "model_name": model_name,
                "benchmark": benchmark_name,
                "total_samples": int(summary["total_instances"]),
                "issues_resolved": int(summary["resolved_instances"]),
                "pass_at_1": float(summary["pass_at_1"]),
                "result_source": result_source,
            }
        ]
    )


def build_output_table(summary: dict[str, Any], patch_rows: list[dict[str, Any]]) -> pd.DataFrame:
    resolved = set(map(str, summary["resolved_ids"]))
    output_rows: list[dict[str, Any]] = []
    patch_by_instance = {
        str(row.get("instance_id")): row
        for row in patch_rows
        if row.get("instance_id") is not None
    }
    for instance_id in sorted(map(str, summary["resolved_ids"] + summary["unresolved_ids"])):
        patch_row = patch_by_instance.get(instance_id, {})
        output_rows.append(
            {
                "instance_id": instance_id,
                "resolved": instance_id in resolved,
                "has_patch_record": bool(patch_row),
                "openclaw_returncode": patch_row.get("openclaw_returncode"),
                "tool_policy_ok": patch_row.get("tool_policy_ok"),
                "conversation_order_ok": patch_row.get("conversation_order_ok"),
                "nemoclaw_session_audit_ok": patch_row.get("nemoclaw_session_audit_ok"),
                "nemoclaw_session_audit_required": (
                    (patch_row.get("nemoclaw_session_audit") or {}).get("required")
                    if isinstance(patch_row.get("nemoclaw_session_audit"), dict)
                    else None
                ),
                "openclaw_tool_call_count": patch_row.get("openclaw_tool_call_count"),
                "openclaw_result_path": patch_row.get("openclaw_result_path"),
            }
        )
    return pd.DataFrame(output_rows)


def make_artifact(
    *,
    artifact_name: str,
    official_eval_dir: Path,
    patch_path: Path,
    summary: dict[str, Any],
    model_name: str,
    benchmark_name: str,
) -> wandb.Artifact:
    artifact = wandb.Artifact(
        artifact_name,
        type="evaluation-results",
        metadata={
            "model_name": model_name,
            "benchmark": benchmark_name,
            "total_instances": summary["total_instances"],
            "resolved_instances": summary["resolved_instances"],
            "pass_at_1": summary["pass_at_1"],
            "source_official_eval_dir": str(official_eval_dir),
            "patch_path": str(patch_path),
            "source_sha256": source_sha256s(
                official_eval_dir=official_eval_dir,
                patch_path=patch_path,
            ),
            "no_inference": True,
        },
    )
    if patch_path.exists():
        artifact.add_file(str(patch_path), name="patches.json")
    for filename in (
        "summary.json",
        "eval_results.json",
        "official_invocation.json",
        "official_stdout.log",
        "official_stderr.log",
    ):
        path = official_eval_dir / filename
        if path.exists():
            artifact.add_file(str(path), name=f"official_eval/{filename}")
    return artifact


def build_run_tags(extra_tags: list[str]) -> list[str]:
    return sorted(
        {
            "taiwan",
            "agentic_swe",
            "swebench-pro",
            "relog",
            "no-inference",
            *extra_tags,
        }
    )


def build_run_config(
    *,
    model_name: str,
    benchmark_name: str,
    official_eval_dir: Path,
    patch_path: Path,
) -> dict[str, Any]:
    return {
        "model": {"pretrained_model_name_or_path": model_name},
        "benchmark": benchmark_name,
        "agentic_swe": {
            "run_openclaw": False,
            "evaluate": False,
            "official_eval_dir": str(official_eval_dir),
            "patch_path": str(patch_path),
        },
        "relog": {
            "no_inference": True,
            "source_official_eval_dir": str(official_eval_dir),
            "patch_path": str(patch_path),
            "source_sha256": source_sha256s(
                official_eval_dir=official_eval_dir,
                patch_path=patch_path,
            ),
        },
    }


def external_action_approval_plan(*, entity: str, project: str) -> dict[str, Any]:
    return {
        "required_before_wandb_write": True,
        "required_report_option": "--external-action-approval-report-json",
        "required_source_packet_option": "--external-action-approval-source-packet-json",
        "required_report_status": "approved",
        "required_source_bound": True,
        "required_source_packet_sha256_match": True,
        "required_requirements": ["wandb_access", "wandb_write"],
        "target_entity": entity,
        "target_project": project,
    }


def build_dry_run_plan(
    *,
    entity: str,
    project: str,
    model_name: str,
    benchmark_name: str,
    run_name: str,
    artifact_name: str,
    official_eval_dir: Path,
    patch_path: Path,
    summary: dict[str, Any],
    row_count: int,
    expected_total: int | None,
    tags: list[str],
    config: dict[str, Any],
) -> dict[str, Any]:
    expected_total_arg = (
        f"--expected-total {expected_total} " if expected_total is not None else ""
    )
    source_hashes = source_sha256s(
        official_eval_dir=official_eval_dir,
        patch_path=patch_path,
    )
    source_hash_config_args = " ".join(
        f"--expected-run-config relog.source_sha256.{key}={value}"
        for key, value in sorted(source_hashes.items())
    )
    if source_hash_config_args:
        source_hash_config_args += " "
    return {
        "schema_version": 1,
        "ok": True,
        "will_write_wandb": False,
        "benchmark": "agentic_swe",
        "entity": entity,
        "project": project,
        "run_name": run_name,
        "job_type": "evaluation-relog",
        "tags": tags,
        "config": config,
        "source": {
            "official_eval_dir": str(official_eval_dir),
            "summary_json": str(official_eval_dir / "summary.json"),
            "eval_results_json": str(official_eval_dir / "eval_results.json"),
            "patch_path": str(patch_path),
            "source_sha256": source_hashes,
        },
        "would_log": {
            "tables": {
                "agentic_swe_leaderboard_table": 1,
                "agentic_swe_output_table": row_count,
            },
            "summary_metrics": {
                "agentic_swe/pass_at_1": float(summary["pass_at_1"]),
                "agentic_swe/resolved_instances": int(summary["resolved_instances"]),
                "agentic_swe/total_instances": int(summary["total_instances"]),
                "agentic_swe/unresolved_instances": int(summary["unresolved_instances"]),
            },
            "artifact": {
                "name": artifact_name,
                "type": "evaluation-results",
                "aliases": ["latest", "production"],
            },
        },
        "external_action_approval": external_action_approval_plan(
            entity=entity,
            project=project,
        ),
        "post_log_verifier_command_template": (
            "uv run python scripts/tools/verify_taiwan_wandb_completion.py "
            f"--entity {entity} --project {project} --run-id RUN_ID_AFTER_WANDB_LOG "
            "--benchmark agentic_swe "
            f"{expected_total_arg}"
            f"--expected-run-config model.pretrained_model_name_or_path={model_name} "
            f"{source_hash_config_args}"
            "--expected-run-job-type evaluation-relog "
            "--json outputs/taiwan_full_eval/wandb_completion/agentic_swe-RUN_ID_AFTER_WANDB_LOG.json"
        ),
    }


def build_validation_failed_plan(
    *,
    entity: str,
    project: str,
    model_name: str,
    benchmark_name: str,
    official_eval_dir: Path,
    patch_path: Path,
    expected_total: int | None,
    error: str,
) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "ok": False,
        "status": "validation_failed",
        "will_write_wandb": False,
        "benchmark": "agentic_swe",
        "entity": entity,
        "project": project,
        "model_name": model_name,
        "benchmark_name": benchmark_name,
        "expected_total": expected_total,
        "source": {
            "official_eval_dir": str(official_eval_dir),
            "summary_json": str(official_eval_dir / "summary.json"),
            "eval_results_json": str(official_eval_dir / "eval_results.json"),
            "patch_path": str(patch_path),
            "source_sha256": source_sha256s(
                official_eval_dir=official_eval_dir,
                patch_path=patch_path,
            ),
        },
        "external_action_approval": external_action_approval_plan(
            entity=entity,
            project=project,
        ),
        "errors": [error],
    }


def validate_dry_run_plan_for_write(
    *,
    plan_path: Path,
    expected_plan: dict[str, Any],
) -> None:
    observed = read_json(plan_path)
    mismatches: list[str] = []
    if observed.get("ok") is not True:
        mismatches.append("validated dry-run plan must have ok=true")
    if observed.get("will_write_wandb") is not False:
        mismatches.append("validated dry-run plan must have will_write_wandb=false")
    for key in (
        "schema_version",
        "benchmark",
        "entity",
        "project",
        "run_name",
        "job_type",
        "tags",
        "config",
        "source",
        "would_log",
        "external_action_approval",
        "post_log_verifier_command_template",
    ):
        if observed.get(key) != expected_plan.get(key):
            mismatches.append(f"validated dry-run plan {key} does not match current inputs")
    if mismatches:
        raise ValueError("; ".join(mismatches))


def emit_plan(payload: dict[str, Any], plan_json: Path | None) -> None:
    if plan_json:
        plan_json.parent.mkdir(parents=True, exist_ok=True)
        plan_json.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
    print(json.dumps(payload, ensure_ascii=False, indent=2))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--official-eval-dir", type=Path, required=True)
    parser.add_argument("--patch-path", type=Path, required=True)
    parser.add_argument("--entity", default="llm-leaderboard")
    parser.add_argument("--project", default="tc-leaderboard")
    parser.add_argument("--model-name", required=True)
    parser.add_argument("--benchmark-name", default="SWE-Bench Pro")
    parser.add_argument("--expected-total", type=int, default=80)
    parser.add_argument("--run-name")
    parser.add_argument("--artifact-name")
    parser.add_argument("--tag", action="append", default=[])
    parser.add_argument(
        "--notes",
        default=(
            "Recovery run: logs existing SWE-Bench Pro official eval results "
            "to W&B without model inference."
        ),
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate local inputs and print the planned W&B write without logging to W&B.",
    )
    parser.add_argument(
        "--plan-json",
        type=Path,
        help="Optional path for the dry-run plan JSON. Implies --dry-run.",
    )
    parser.add_argument(
        "--validated-dry-run-plan-json",
        type=Path,
        help=(
            "Required for W&B writes. Must point to a previously reviewed "
            "ok=true dry-run plan generated with the same inputs."
        ),
    )
    parser.add_argument(
        "--external-action-approval-report-json",
        type=Path,
        help=(
            "Required for W&B writes. Must point to a source-bound approved "
            "verify_external_action_approval_packet.py report whose W&B write "
            "scope matches --entity/--project."
        ),
    )
    parser.add_argument(
        "--external-action-approval-source-packet-json",
        type=Path,
        help=(
            "Required for W&B writes. Must point to the source approval packet "
            "that the approval report reviewed. The report source binding path "
            "and sha256 must match this file."
        ),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    official_eval_dir = args.official_eval_dir.resolve()
    patch_path = args.patch_path.resolve()
    summary_path = official_eval_dir / "summary.json"
    eval_results_path = official_eval_dir / "eval_results.json"
    if not summary_path.exists():
        raise FileNotFoundError(summary_path)
    if not patch_path.exists():
        raise FileNotFoundError(patch_path)

    summary = read_json(summary_path)
    eval_results = read_eval_results(eval_results_path) if eval_results_path.exists() else None
    patch_rows = read_patch_rows(patch_path)
    dry_run_requested = args.dry_run or args.plan_json
    try:
        validate_summary(
            summary,
            eval_results=eval_results,
            expected_total=args.expected_total,
        )
    except ValueError as exc:
        if dry_run_requested:
            emit_plan(
                build_validation_failed_plan(
                    entity=args.entity,
                    project=args.project,
                    model_name=args.model_name,
                    benchmark_name=args.benchmark_name,
                    official_eval_dir=official_eval_dir,
                    patch_path=patch_path,
                    expected_total=args.expected_total,
                    error=str(exc),
                ),
                args.plan_json,
            )
            raise SystemExit(1)
        raise

    run_name = args.run_name or (
        "taiwan-agentic-swe-relog-"
        + args.model_name.replace("/", "-").replace(":", "-")
    )
    artifact_name = args.artifact_name or (
        "agentic-swe-swebench-pro-"
        + sanitize_artifact_component(args.model_name)
        + "-results"
    )
    result_source = str(official_eval_dir)
    tags = build_run_tags(args.tag)
    config = build_run_config(
        model_name=args.model_name,
        benchmark_name=args.benchmark_name,
        official_eval_dir=official_eval_dir,
        patch_path=patch_path,
    )
    leaderboard = build_leaderboard(
        model_name=args.model_name,
        summary=summary,
        benchmark_name=args.benchmark_name,
        result_source=result_source,
    )
    output_df = build_output_table(summary, patch_rows)

    plan = build_dry_run_plan(
        entity=args.entity,
        project=args.project,
        model_name=args.model_name,
        benchmark_name=args.benchmark_name,
        run_name=run_name,
        artifact_name=artifact_name,
        official_eval_dir=official_eval_dir,
        patch_path=patch_path,
        summary=summary,
        row_count=len(output_df),
        expected_total=args.expected_total,
        tags=tags,
        config=config,
    )

    if dry_run_requested:
        emit_plan(plan, args.plan_json)
        return

    if args.validated_dry_run_plan_json is None:
        raise SystemExit(
            "--validated-dry-run-plan-json is required before writing relogged "
            "SWE-Bench Pro results to W&B. Run the same command with --dry-run "
            "--plan-json first, review the plan, then pass that plan path."
        )
    validate_dry_run_plan_for_write(
        plan_path=args.validated_dry_run_plan_json.resolve(),
        expected_plan=plan,
    )
    if args.external_action_approval_report_json is None:
        raise SystemExit(
            "--external-action-approval-report-json is required before writing "
            "relogged SWE-Bench Pro results to W&B. Render and verify the current "
            "external_action_approval_packet.json with all required approvals "
            "granted, then pass the verifier report path."
        )
    if args.external_action_approval_source_packet_json is None:
        raise SystemExit(
            "--external-action-approval-source-packet-json is required before "
            "writing relogged SWE-Bench Pro results to W&B. Pass the source "
            "approval packet that was reviewed by the approval report."
        )
    validate_external_action_approval_for_wandb_write(
        args.external_action_approval_report_json.resolve(),
        entity=args.entity,
        project=args.project,
        expected_source_packet_path=args.external_action_approval_source_packet_json.resolve(),
    )

    wandb.login()
    with wandb.init(
        entity=args.entity,
        project=args.project,
        name=run_name,
        job_type="evaluation-relog",
        tags=tags,
        notes=args.notes,
        config=config,
        settings=wandb.Settings(init_timeout=300),
    ) as run:
        run.log(
            {
                "agentic_swe_leaderboard_table": wandb.Table(dataframe=leaderboard),
                "agentic_swe_output_table": wandb.Table(dataframe=output_df),
                "agentic_swe_results": summary,
                "agentic_swe/pass_at_1": float(summary["pass_at_1"]),
                "agentic_swe/resolved_instances": int(summary["resolved_instances"]),
                "agentic_swe/total_instances": int(summary["total_instances"]),
                "agentic_swe/unresolved_instances": int(summary["unresolved_instances"]),
            }
        )
        run.log_artifact(
            make_artifact(
                artifact_name=artifact_name,
                official_eval_dir=official_eval_dir,
                patch_path=patch_path,
                summary=summary,
                model_name=args.model_name,
                benchmark_name=args.benchmark_name,
            ),
            aliases=["latest", "production"],
        )
        print(
            json.dumps(
                {
                    "run_url": run.url,
                    "run_id": run.id,
                    "entity": args.entity,
                    "project": args.project,
                    "run_name": run_name,
                    "artifact_name": artifact_name,
                    "pass_at_1": summary["pass_at_1"],
                    "resolved_instances": summary["resolved_instances"],
                    "total_instances": summary["total_instances"],
                },
                ensure_ascii=False,
                indent=2,
            )
        )


if __name__ == "__main__":
    main()
