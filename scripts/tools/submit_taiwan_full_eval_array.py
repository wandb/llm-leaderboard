#!/usr/bin/env python3
"""Submit isolated Taiwan evaluation jobs through slotd or Slurm."""

from __future__ import annotations

import argparse
import json
import os
import re
import shlex
import subprocess
import sys
import threading
import time
from pathlib import Path
from typing import Any, Sequence


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_MODEL_MANIFEST = REPO_ROOT / "configs" / "taiwan_full_eval_models.yaml"
DEFAULT_BASE_CONFIG = REPO_ROOT / "configs" / "base_config_taiwan.yaml"
DEFAULT_ENV_FILE = REPO_ROOT / ".env"
DEFAULT_BUNDLE_ROOT = REPO_ROOT / "outputs" / "slurm" / "taiwan_eval_jobs"
SBATCH_SCRIPT = REPO_ROOT / "scripts" / "slurm" / "taiwan_full_eval_array.sbatch"
BATCH_RUNNER = REPO_ROOT / "scripts" / "tools" / "run_taiwan_full_eval_batch.py"
SAFE_ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]*$")
PROTECTED_RUNNER_ARGS = {
    "--allow-cash-cost-exempt-execution",
    "--base-config",
    "--env-file",
    "--generated-config-dir",
    "--manifest",
    "--model",
    "--output-root",
    "--phase",
    "--prepare-only",
    "--python",
    "--wandb-run-id-prefix",
}


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(
        f".{path.name}.{os.getpid()}.{threading.get_ident()}.tmp"
    )
    try:
        with temporary.open("w", encoding="utf-8") as handle:
            handle.write(
                json.dumps(payload, ensure_ascii=False, indent=2) + "\n"
            )
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def resolve_path(path: Path, repo_root: Path) -> Path:
    if path.is_absolute():
        return path.resolve()
    return (repo_root / path).resolve()


def absolute_path_preserving_symlinks(path: Path, repo_root: Path) -> Path:
    candidate = path if path.is_absolute() else repo_root / path
    return Path(os.path.abspath(candidate))


def require_file(path: Path, label: str) -> None:
    if not path.is_file():
        raise ValueError(f"{label} not found: {path}")


def validate_safe_id(value: str, label: str) -> str:
    if not SAFE_ID_RE.fullmatch(value):
        raise ValueError(
            f"{label} must contain only letters, digits, dot, underscore, or dash: {value!r}"
        )
    return value


def validate_runner_args(values: Sequence[str]) -> list[str]:
    runner_args = [str(value) for value in values]
    for value in runner_args:
        option = value.split("=", 1)[0]
        if option in PROTECTED_RUNNER_ARGS:
            raise ValueError(
                f"{option} is managed by the array launcher and cannot be passed "
                "through --runner-arg"
            )
    return runner_args


def execution_requirements(args: argparse.Namespace, repo_root: Path) -> dict[str, Path]:
    if not args.execute:
        return {}

    required_text = {
        "--run-purpose": args.run_purpose,
        "--wandb-run-id-prefix": args.wandb_run_id_prefix,
    }
    if not args.allow_cash_cost_exempt_execution:
        required_text["--expected-cost-band"] = args.expected_cost_band
    missing = [name for name, value in required_text.items() if not str(value or "").strip()]
    if missing:
        raise ValueError(
            "External execution requires explicit accountability fields: "
            + ", ".join(missing)
        )

    required_paths = {
        "pre_run_budget_estimate_json": args.pre_run_budget_estimate_json,
        "external_action_approval_report_json": args.external_action_approval_report_json,
        "external_action_approval_source_packet_json": (
            args.external_action_approval_source_packet_json
        ),
    }
    if args.allow_cash_cost_exempt_execution:
        return {}
    resolved: dict[str, Path] = {}
    for label, raw_path in required_paths.items():
        if raw_path is None:
            raise ValueError(
                "External execution requires --" + label.replace("_", "-")
            )
        path = resolve_path(raw_path, repo_root)
        require_file(path, label)
        resolved[label] = path
    return resolved


def runner_args_for_job(
    *,
    args: argparse.Namespace,
    repo_root: Path,
    model_slug: str,
    output_root: Path,
    generated_config_dir: Path,
    execution_paths: dict[str, Path],
    extra_runner_args: Sequence[str],
) -> list[str]:
    python = absolute_path_preserving_symlinks(args.python, repo_root)
    command = [
        str(python),
        str(BATCH_RUNNER),
        "--manifest",
        str(resolve_path(args.manifest, repo_root)),
        "--model",
        model_slug,
        "--phase",
        args.phase,
        "--output-root",
        str(output_root),
        "--generated-config-dir",
        str(generated_config_dir),
        "--base-config",
        str(resolve_path(args.base_config, repo_root)),
        "--env-file",
        str(resolve_path(args.env_file, repo_root)),
        "--python",
        str(python),
    ]
    if not args.execute:
        command.append("--prepare-only")
    else:
        command.extend(["--run-purpose", args.run_purpose])
        command.extend(["--wandb-run-id-prefix", args.wandb_run_id_prefix])
        if args.allow_cash_cost_exempt_execution:
            command.append("--allow-cash-cost-exempt-execution")
        else:
            command.extend(
                [
                    "--expected-cost-band",
                    args.expected_cost_band,
                    "--pre-run-budget-estimate-json",
                    str(execution_paths["pre_run_budget_estimate_json"]),
                    "--external-action-approval-report-json",
                    str(execution_paths["external_action_approval_report_json"]),
                    "--external-action-approval-source-packet-json",
                    str(execution_paths["external_action_approval_source_packet_json"]),
                ]
            )
    if args.yes:
        command.append("--yes")
    command.extend(extra_runner_args)
    return command


def build_job_manifest(args: argparse.Namespace) -> tuple[Path, dict[str, Any]]:
    repo_root = resolve_path(args.repo_root, REPO_ROOT)
    if not repo_root.is_dir():
        raise ValueError(f"repository root not found: {repo_root}")

    models: list[str] = []
    seen: set[str] = set()
    for raw_model in args.model:
        model = validate_safe_id(raw_model, "model slug")
        if model in seen:
            raise ValueError(f"duplicate model slug: {model}")
        seen.add(model)
        models.append(model)
    if not models:
        raise ValueError("at least one --model is required")

    batch_id = validate_safe_id(
        args.batch_id or time.strftime("taiwan-eval-%Y%m%dT%H%M%S"),
        "batch id",
    )
    bundle_root = resolve_path(args.bundle_root, repo_root) / batch_id
    if bundle_root.exists() and any(bundle_root.iterdir()):
        raise ValueError(
            f"job bundle already exists and is non-empty; use a fresh --batch-id: {bundle_root}"
        )
    logs_dir = bundle_root / "logs"
    logs_dir.mkdir(parents=True, exist_ok=True)

    manifest = resolve_path(args.manifest, repo_root)
    base_config = resolve_path(args.base_config, repo_root)
    env_file = resolve_path(args.env_file, repo_root)
    python = absolute_path_preserving_symlinks(args.python, repo_root)
    require_file(manifest, "model manifest")
    require_file(base_config, "base config")
    require_file(python, "Python interpreter")
    if args.execute:
        require_file(env_file, "environment file")
    require_file(SBATCH_SCRIPT, "sbatch script")
    require_file(BATCH_RUNNER, "batch runner")

    extra_runner_args = validate_runner_args(args.runner_arg)
    execution_paths = execution_requirements(args, repo_root)
    jobs = []
    for index, model_slug in enumerate(models):
        task_root = bundle_root / "tasks" / model_slug
        output_root = task_root / "runner"
        generated_config_dir = task_root / "generated_configs"
        command = runner_args_for_job(
            args=args,
            repo_root=repo_root,
            model_slug=model_slug,
            output_root=output_root,
            generated_config_dir=generated_config_dir,
            execution_paths=execution_paths,
            extra_runner_args=extra_runner_args,
        )
        jobs.append(
            {
                "index": index,
                "model_slug": model_slug,
                "output_root": str(output_root),
                "generated_config_dir": str(generated_config_dir),
                "command": command,
            }
        )

    payload = {
        "schema_version": 1,
        "kind": "taiwan_full_eval_slurm_array",
        "created_at": time.time(),
        "batch_id": batch_id,
        "scheduler_compatibility": ["slotd", "slurm"],
        "repo_root": str(repo_root),
        "launcher_python": str(python),
        "phase": args.phase,
        "prepare_only": not args.execute,
        "max_parallel": min(args.max_parallel, len(jobs)),
        "resources": {
            "partition": args.partition,
            "cpus_per_task": args.cpus_per_task,
            "memory": args.mem,
            "time_limit": args.time,
        },
        "jobs": jobs,
    }
    job_manifest_path = bundle_root / "job_manifest.json"
    write_json(job_manifest_path, payload)
    return job_manifest_path, payload


def build_sbatch_command(
    args: argparse.Namespace,
    job_manifest_path: Path,
    payload: dict[str, Any],
) -> list[str]:
    jobs = payload["jobs"]
    array = f"0-{len(jobs) - 1}%{payload['max_parallel']}"
    logs_dir = job_manifest_path.parent / "logs"
    export = ",".join(
        [
            "ALL",
            f"TAIWAN_LB_ROOT={payload['repo_root']}",
            f"TAIWAN_EVAL_LAUNCHER_PYTHON={payload['launcher_python']}",
            f"TAIWAN_EVAL_JOB_MANIFEST={job_manifest_path}",
        ]
    )
    return [
        args.submit_command,
        "--parsable",
        "--job-name",
        args.job_name,
        "--partition",
        args.partition,
        "--cpus-per-task",
        str(args.cpus_per_task),
        "--mem",
        args.mem,
        "--time",
        args.time,
        "--array",
        array,
        "--chdir",
        payload["repo_root"],
        "--output",
        str(logs_dir / "%A_%a.out"),
        "--error",
        str(logs_dir / "%A_%a.err"),
        "--export",
        export,
        str(SBATCH_SCRIPT),
    ]


def submit_jobs(args: argparse.Namespace) -> int:
    try:
        if args.max_parallel < 1:
            raise ValueError("--max-parallel must be at least 1")
        if args.cpus_per_task < 1:
            raise ValueError("--cpus-per-task must be at least 1")
        job_manifest_path, payload = build_job_manifest(args)
        command = build_sbatch_command(args, job_manifest_path, payload)
    except ValueError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2

    submission_path = job_manifest_path.parent / "submission.json"
    submission = {
        "schema_version": 1,
        "job_manifest": str(job_manifest_path),
        "submitted": bool(args.submit),
        "command": command,
        "command_shell": shlex.join(command),
        "created_at": time.time(),
    }
    if not args.submit:
        submission["status"] = "dry_run"
        write_json(submission_path, submission)
        print(shlex.join(command))
        print(f"job manifest: {job_manifest_path}")
        return 0

    result = subprocess.run(
        command,
        cwd=payload["repo_root"],
        text=True,
        capture_output=True,
        check=False,
    )
    submission.update(
        {
            "status": "submitted" if result.returncode == 0 else "submission_failed",
            "returncode": result.returncode,
            "stdout": result.stdout,
            "stderr": result.stderr,
        }
    )
    write_json(submission_path, submission)
    if result.stdout:
        print(result.stdout, end="")
    if result.stderr:
        print(result.stderr, end="", file=sys.stderr)
    print(f"job manifest: {job_manifest_path}")
    return result.returncode


def load_job_manifest(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"could not read job manifest {path}: {exc}") from exc
    if not isinstance(payload, dict) or payload.get("schema_version") != 1:
        raise ValueError(f"unsupported job manifest schema: {path}")
    jobs = payload.get("jobs")
    if not isinstance(jobs, list) or not jobs:
        raise ValueError(f"job manifest has no jobs: {path}")
    return payload


def task_command(
    job_manifest_path: Path,
    index: int,
) -> tuple[dict[str, Any], dict[str, Any], list[str]]:
    payload = load_job_manifest(job_manifest_path)
    jobs = payload["jobs"]
    if index < 0 or index >= len(jobs):
        raise ValueError(f"array index {index} is outside 0-{len(jobs) - 1}")
    job = jobs[index]
    command = job.get("command")
    if not isinstance(command, list) or not command or not all(
        isinstance(item, str) and item for item in command
    ):
        raise ValueError(f"job {index} has an invalid command")
    return payload, job, command


def run_task(args: argparse.Namespace) -> int:
    manifest_path = args.job_manifest.resolve()
    try:
        payload, job, command = task_command(manifest_path, args.index)
    except ValueError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2

    repo_root = Path(payload["repo_root"])
    output_root = Path(job["output_root"])
    output_root.mkdir(parents=True, exist_ok=True)
    task_record = {
        "schema_version": 1,
        "job_manifest": str(manifest_path),
        "array_index": args.index,
        "model_slug": job["model_slug"],
        "command": command,
        "slurm_job_id": os.environ.get("SLURM_JOB_ID", ""),
        "slurm_array_job_id": os.environ.get("SLURM_ARRAY_JOB_ID", ""),
        "slurm_array_task_id": os.environ.get("SLURM_ARRAY_TASK_ID", ""),
        "started_at": time.time(),
    }
    write_json(output_root / "slurm_task.json", task_record)
    print(
        f"Starting Taiwan evaluation array task {args.index}: {job['model_slug']}",
        flush=True,
    )
    print(shlex.join(command), flush=True)
    os.chdir(repo_root)
    environment = os.environ.copy()
    environment["PYTHONUNBUFFERED"] = "1"
    try:
        result = subprocess.run(
            command,
            cwd=repo_root,
            env=environment,
            check=False,
        )
    except BaseException as exc:
        task_record.update(
            {
                "status": "interrupted",
                "error_type": type(exc).__name__,
                "error": str(exc),
                "ended_at": time.time(),
            }
        )
        write_json(output_root / "slurm_task.json", task_record)
        raise
    task_record.update(
        {
            "status": "completed" if result.returncode == 0 else "failed",
            "returncode": result.returncode,
            "ended_at": time.time(),
        }
    )
    write_json(output_root / "slurm_task.json", task_record)
    return result.returncode


def add_submit_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--model", action="append", default=[], help="Model slug; repeatable.")
    parser.add_argument(
        "--phase",
        choices=["full", "nonagentic", "agentic", "agentic_aggregate"],
        default="full",
    )
    parser.add_argument("--batch-id")
    parser.add_argument("--bundle-root", type=Path, default=DEFAULT_BUNDLE_ROOT)
    parser.add_argument("--repo-root", type=Path, default=REPO_ROOT)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MODEL_MANIFEST)
    parser.add_argument("--base-config", type=Path, default=DEFAULT_BASE_CONFIG)
    parser.add_argument("--env-file", type=Path, default=DEFAULT_ENV_FILE)
    parser.add_argument("--python", type=Path, default=REPO_ROOT / ".venv" / "bin" / "python")
    parser.add_argument("--execute", action="store_true", help="Allow external model execution.")
    parser.add_argument("--run-purpose", default="")
    parser.add_argument("--expected-cost-band", default="")
    parser.add_argument("--pre-run-budget-estimate-json", type=Path)
    parser.add_argument("--external-action-approval-report-json", type=Path)
    parser.add_argument("--external-action-approval-source-packet-json", type=Path)
    parser.add_argument("--wandb-run-id-prefix", default="")
    parser.add_argument(
        "--allow-cash-cost-exempt-execution",
        action="store_true",
        help=(
            "Skip cash-budget artifacts only for generated configs that the "
            "batch runner verifies as cash-cost-exempt."
        ),
    )
    parser.add_argument("--yes", action="store_true")
    parser.add_argument(
        "--runner-arg",
        action="append",
        default=[],
        help="Additional batch-runner argument. Use --runner-arg=--flag for options.",
    )
    parser.add_argument("--max-parallel", type=int, default=2)
    parser.add_argument("--partition", default="cpu")
    parser.add_argument("--cpus-per-task", type=int, default=16)
    parser.add_argument("--mem", default="64G")
    parser.add_argument("--time", default="36:00:00")
    parser.add_argument("--job-name", default="taiwan-eval")
    parser.add_argument("--submit-command", default="sbatch")
    parser.add_argument(
        "--submit",
        action="store_true",
        help="Submit through sbatch. Without this flag, only write and print the plan.",
    )


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create and submit portable slotd/Slurm Taiwan evaluation arrays."
    )
    subparsers = parser.add_subparsers(dest="command", required=True)
    submit_parser = subparsers.add_parser("submit", help="Create an array job bundle.")
    add_submit_arguments(submit_parser)
    task_parser = subparsers.add_parser("run-task", help=argparse.SUPPRESS)
    task_parser.add_argument("--job-manifest", type=Path, required=True)
    task_parser.add_argument("--index", type=int, required=True)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    if args.command == "submit":
        return submit_jobs(args)
    if args.command == "run-task":
        return run_task(args)
    raise AssertionError(f"unexpected command: {args.command}")


if __name__ == "__main__":
    raise SystemExit(main())
