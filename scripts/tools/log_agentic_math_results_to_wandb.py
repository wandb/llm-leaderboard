#!/usr/bin/env python3
"""
Log existing Agentic Math OpenClaw results to W&B without running inference.

This is for recovery/audit situations where `results.jsonl` and `summary.json`
already exist locally, but the corresponding W&B run did not finish or did not
receive the leaderboard tables. It intentionally uses the same W&B keys as
`scripts/evaluator/agentic_math.py` so downstream Taiwan aggregation can read
the result normally.
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
    "answered_instances",
    "correct_instances",
    "incorrect_instances",
    "accuracy",
    "correctness",
}
NEMOCLAW_SESSION_AUDIT_COLUMNS = ("nemoclaw_session_audit_ok", "nemoclaw_session_audit")


def read_json(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as f:
        obj = json.load(f)
    if not isinstance(obj, dict):
        raise ValueError(f"{path} must contain a JSON object")
    return obj


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as f:
        for line_no, line in enumerate(f, 1):
            if not line.strip():
                continue
            obj = json.loads(line)
            if not isinstance(obj, dict):
                raise ValueError(f"{path}:{line_no} must contain a JSON object")
            rows.append(obj)
    return rows


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def source_sha256s(results_dir: Path) -> dict[str, str]:
    return {
        "summary_json": sha256_file(results_dir / "summary.json"),
        "results_jsonl": sha256_file(results_dir / "results.jsonl"),
    }


def validate_summary(summary: dict[str, Any], rows: list[dict[str, Any]]) -> None:
    missing = sorted(REQUIRED_SUMMARY_KEYS - set(summary))
    if missing:
        raise ValueError(f"summary.json is missing required keys: {missing}")

    total = int(summary["total_instances"])
    correct = int(summary["correct_instances"])
    incorrect = int(summary["incorrect_instances"])
    answered = int(summary["answered_instances"])

    row_total = len(rows)
    row_correct = sum(1 for row in rows if row.get("correct") is True)
    row_incorrect = sum(1 for row in rows if row.get("correct") is not True)
    row_answered = sum(1 for row in rows if row.get("predicted_answer") not in (None, ""))

    mismatches = []
    if total != row_total:
        mismatches.append(f"total_instances={total} but results rows={row_total}")
    if correct != row_correct:
        mismatches.append(f"correct_instances={correct} but row correct={row_correct}")
    if incorrect != row_incorrect:
        mismatches.append(f"incorrect_instances={incorrect} but row incorrect={row_incorrect}")
    if answered != row_answered:
        mismatches.append(f"answered_instances={answered} but row answered={row_answered}")
    if mismatches:
        raise ValueError("; ".join(mismatches))

    expected_accuracy = correct / total if total else 0.0
    actual_accuracy = float(summary["accuracy"])
    if abs(actual_accuracy - expected_accuracy) > 1e-12:
        raise ValueError(
            f"accuracy={actual_accuracy} but correct/total={expected_accuracy}"
        )


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
                "answered_samples": int(summary["answered_instances"]),
                "correct_count": int(summary["correct_instances"]),
                "accuracy": float(summary["accuracy"]),
                "correctness": float(summary["correctness"]),
                "result_source": result_source,
            }
        ]
    )


def ensure_output_observability_columns(output_df: pd.DataFrame) -> pd.DataFrame:
    for column in NEMOCLAW_SESSION_AUDIT_COLUMNS:
        if column not in output_df.columns:
            output_df[column] = None
    return output_df


def make_artifact(
    *,
    artifact_name: str,
    results_dir: Path,
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
            "correct_instances": summary["correct_instances"],
            "accuracy": summary["accuracy"],
            "source_results_dir": str(results_dir),
            "source_sha256": source_sha256s(results_dir),
            "no_inference": True,
        },
    )
    for filename in ("summary.json", "results.jsonl"):
        artifact.add_file(str(results_dir / filename), name=filename)
    return artifact


def build_run_tags(extra_tags: list[str]) -> list[str]:
    return sorted(
        {
            "taiwan",
            "agentic_math",
            "olymmath-hard-zh-tw",
            "relog",
            "no-inference",
            *extra_tags,
        }
    )


def build_run_config(
    *,
    model_name: str,
    benchmark_name: str,
    results_dir: Path,
    summary: dict[str, Any],
) -> dict[str, Any]:
    return {
        "model": {"pretrained_model_name_or_path": model_name},
        "benchmark": benchmark_name,
        "agentic_math": {
            "run_openclaw": False,
            "results_dir": str(results_dir),
            "source_runner_version": summary.get("runner_version"),
            "source_model": summary.get("model"),
            "source_thinking": summary.get("thinking"),
        },
        "relog": {
            "no_inference": True,
            "source_results_dir": str(results_dir),
            "source_sha256": source_sha256s(results_dir),
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
    results_dir: Path,
    summary: dict[str, Any],
    row_count: int,
    tags: list[str],
    config: dict[str, Any],
) -> dict[str, Any]:
    source_hashes = source_sha256s(results_dir)
    return {
        "schema_version": 1,
        "ok": True,
        "will_write_wandb": False,
        "benchmark": "agentic_math",
        "entity": entity,
        "project": project,
        "run_name": run_name,
        "job_type": "evaluation-relog",
        "tags": tags,
        "config": config,
        "source": {
            "results_dir": str(results_dir),
            "summary_json": str(results_dir / "summary.json"),
            "results_jsonl": str(results_dir / "results.jsonl"),
            "source_sha256": source_hashes,
        },
        "would_log": {
            "tables": {
                "agentic_math_leaderboard_table": 1,
                "agentic_math_output_table": row_count,
            },
            "summary_metrics": {
                "agentic_math/accuracy": float(summary["accuracy"]),
                "agentic_math/correct_instances": int(summary["correct_instances"]),
                "agentic_math/total_instances": int(summary["total_instances"]),
                "agentic_math/answered_instances": int(summary["answered_instances"]),
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
            "--benchmark agentic_math "
            f"--expected-total {int(summary['total_instances'])} "
            f"--expected-run-config model.pretrained_model_name_or_path={model_name} "
            f"--expected-run-config relog.source_sha256.summary_json={source_hashes['summary_json']} "
            f"--expected-run-config relog.source_sha256.results_jsonl={source_hashes['results_jsonl']} "
            "--expected-run-job-type evaluation-relog "
            "--json outputs/taiwan_full_eval/wandb_completion/agentic_math-RUN_ID_AFTER_WANDB_LOG.json"
        ),
    }


def build_validation_failed_plan(
    *,
    entity: str,
    project: str,
    model_name: str,
    benchmark_name: str,
    results_dir: Path,
    error: str,
) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "ok": False,
        "status": "validation_failed",
        "will_write_wandb": False,
        "benchmark": "agentic_math",
        "entity": entity,
        "project": project,
        "model_name": model_name,
        "benchmark_name": benchmark_name,
        "source": {
            "results_dir": str(results_dir),
            "summary_json": str(results_dir / "summary.json"),
            "results_jsonl": str(results_dir / "results.jsonl"),
            "source_sha256": source_sha256s(results_dir),
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
    parser.add_argument(
        "--results-dir",
        type=Path,
        required=True,
        help="Directory containing summary.json and results.jsonl.",
    )
    parser.add_argument("--entity", default="llm-leaderboard")
    parser.add_argument("--project", default="tc-leaderboard")
    parser.add_argument("--model-name", required=True)
    parser.add_argument("--benchmark-name", default="OlymMATH-HARD zh-TW")
    parser.add_argument(
        "--run-name",
        default=None,
        help="W&B run name. Defaults to a deterministic recovery name.",
    )
    parser.add_argument(
        "--artifact-name",
        default=None,
        help="W&B artifact name for local result files.",
    )
    parser.add_argument(
        "--tag",
        action="append",
        default=[],
        help="Additional W&B tag. Can be repeated.",
    )
    parser.add_argument(
        "--notes",
        default=(
            "Recovery run: logs existing Agentic Math results to W&B without "
            "model inference."
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
    results_dir = args.results_dir.resolve()
    summary_path = results_dir / "summary.json"
    results_path = results_dir / "results.jsonl"
    if not summary_path.exists():
        raise FileNotFoundError(summary_path)
    if not results_path.exists():
        raise FileNotFoundError(results_path)

    summary = read_json(summary_path)
    rows = read_jsonl(results_path)
    dry_run_requested = args.dry_run or args.plan_json
    try:
        validate_summary(summary, rows)
    except ValueError as exc:
        if dry_run_requested:
            emit_plan(
                build_validation_failed_plan(
                    entity=args.entity,
                    project=args.project,
                    model_name=args.model_name,
                    benchmark_name=args.benchmark_name,
                    results_dir=results_dir,
                    error=str(exc),
                ),
                args.plan_json,
            )
            raise SystemExit(1)
        raise

    run_name = args.run_name or (
        "taiwan-agentic-math-relog-"
        + args.model_name.replace("/", "-").replace(":", "-")
    )
    artifact_name = args.artifact_name or (
        "agentic-math-results-"
        + args.model_name.replace("/", "-").replace(":", "-")
    )
    result_source = str(results_dir)
    tags = build_run_tags(args.tag)
    config = build_run_config(
        model_name=args.model_name,
        benchmark_name=args.benchmark_name,
        results_dir=results_dir,
        summary=summary,
    )

    output_df = ensure_output_observability_columns(pd.DataFrame(rows))
    leaderboard = build_leaderboard(
        model_name=args.model_name,
        summary=summary,
        benchmark_name=args.benchmark_name,
        result_source=result_source,
    )

    plan = build_dry_run_plan(
        entity=args.entity,
        project=args.project,
        model_name=args.model_name,
        benchmark_name=args.benchmark_name,
        run_name=run_name,
        artifact_name=artifact_name,
        results_dir=results_dir,
        summary=summary,
        row_count=len(output_df),
        tags=tags,
        config=config,
    )

    if dry_run_requested:
        emit_plan(plan, args.plan_json)
        return

    if args.validated_dry_run_plan_json is None:
        raise SystemExit(
            "--validated-dry-run-plan-json is required before writing relogged "
            "Agentic Math results to W&B. Run the same command with --dry-run "
            "--plan-json first, review the plan, then pass that plan path."
        )
    validate_dry_run_plan_for_write(
        plan_path=args.validated_dry_run_plan_json.resolve(),
        expected_plan=plan,
    )
    if args.external_action_approval_report_json is None:
        raise SystemExit(
            "--external-action-approval-report-json is required before writing "
            "relogged Agentic Math results to W&B. Render and verify the current "
            "external_action_approval_packet.json with all required approvals "
            "granted, then pass the verifier report path."
        )
    if args.external_action_approval_source_packet_json is None:
        raise SystemExit(
            "--external-action-approval-source-packet-json is required before "
            "writing relogged Agentic Math results to W&B. Pass the source "
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
                "agentic_math_leaderboard_table": wandb.Table(dataframe=leaderboard),
                "agentic_math_output_table": wandb.Table(dataframe=output_df),
                "agentic_math_results": summary,
                "agentic_math/accuracy": float(summary["accuracy"]),
                "agentic_math/correct_instances": int(summary["correct_instances"]),
                "agentic_math/total_instances": int(summary["total_instances"]),
                "agentic_math/answered_instances": int(summary["answered_instances"]),
            }
        )
        run.log_artifact(
            make_artifact(
                artifact_name=artifact_name,
                results_dir=results_dir,
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
                    "accuracy": summary["accuracy"],
                    "correct_instances": summary["correct_instances"],
                    "total_instances": summary["total_instances"],
                },
                ensure_ascii=False,
                indent=2,
            )
        )


if __name__ == "__main__":
    main()
