#!/usr/bin/env python3
"""
Evaluate SWE-bench Pro patches with the official Scale evaluator checkout.

Expected setup:

  git clone https://github.com/scaleapi/SWE-bench_Pro-os external/SWE-bench_Pro-os

Then:

  python3 scripts/tools/evaluate_swebench_pro_patches.py \
    --official-repo external/SWE-bench_Pro-os \
    --raw-sample-path data/taiwan/swebench_pro_public/subsets/smoke.csv \
    --patch-path outputs/swebench_pro_openclaw/patches.json \
    --output-dir outputs/swebench_pro_eval/smoke \
    --use-local-docker
"""

from __future__ import annotations

import argparse
import json
import os
import shlex
import subprocess
import sys
from pathlib import Path
from typing import Any

MIN_DOCKER_SDK_VERSION = (7, 1, 0)


def run_command(command: list[str], cwd: Path, check: bool = True) -> subprocess.CompletedProcess[str]:
    result = subprocess.run(command, cwd=str(cwd), text=True, capture_output=True, check=False)
    if check and result.returncode != 0:
        raise RuntimeError(
            "Command failed\n"
            f"cmd: {' '.join(shlex.quote(part) for part in command)}\n"
            f"cwd: {cwd}\n"
            f"returncode: {result.returncode}\n"
            f"stdout:\n{result.stdout}\n"
            f"stderr:\n{result.stderr}"
        )
    return result


def validate_official_repo(path: Path) -> None:
    required = [
        path / "swe_bench_pro_eval.py",
        path / "helper_code" / "image_uri.py",
        path / "run_scripts",
    ]
    missing = [str(item) for item in required if not item.exists()]
    if missing:
        raise FileNotFoundError(
            "SWE-bench Pro official checkout is incomplete. Missing:\n"
            + "\n".join(f"- {item}" for item in missing)
        )


def parse_version(value: str) -> tuple[int, ...]:
    parts: list[int] = []
    for part in value.split("."):
        digits = ""
        for char in part:
            if char.isdigit():
                digits += char
            else:
                break
        if not digits:
            break
        parts.append(int(digits))
    return tuple(parts)


def check_local_docker() -> None:
    try:
        import docker
    except Exception as exc:
        raise SystemExit(
            "Python Docker SDK is required for --use-local-docker. "
            "Install it with: python3 -m pip install --user -U 'docker>=7.1.0'"
        ) from exc

    raw_version = getattr(docker, "__version__", "0")
    version = parse_version(raw_version)
    if version < MIN_DOCKER_SDK_VERSION:
        minimum = ".".join(str(part) for part in MIN_DOCKER_SDK_VERSION)
        raise SystemExit(
            f"Python Docker SDK {raw_version} is too old for current requests/urllib3. "
            f"Install {minimum}+ with: python3 -m pip install --user -U 'docker>={minimum}'"
        )

    try:
        client = docker.from_env()
        client.ping()
    except Exception as exc:
        raise SystemExit(
            "Docker is not reachable from the current shell. Start Docker or fix DOCKER_HOST, "
            "then verify with: docker info"
        ) from exc
    finally:
        close = getattr(locals().get("client", None), "close", None)
        if callable(close):
            close()


def load_eval_results(output_dir: Path) -> dict[str, bool]:
    result_path = output_dir / "eval_results.json"
    if not result_path.exists():
        raise FileNotFoundError(f"Expected eval results not found: {result_path}")
    raw = json.loads(result_path.read_text(encoding="utf-8"))
    return {str(key): bool(value) for key, value in raw.items()}


def write_summary(output_dir: Path, eval_results: dict[str, bool], command: list[str]) -> dict[str, Any]:
    total = len(eval_results)
    resolved = sum(1 for value in eval_results.values() if value)
    summary = {
        "total_instances": total,
        "resolved_instances": resolved,
        "unresolved_instances": total - resolved,
        "pass_at_1": resolved / total if total else 0.0,
        "resolved_ids": sorted([key for key, value in eval_results.items() if value]),
        "unresolved_ids": sorted([key for key, value in eval_results.items() if not value]),
        "command": command,
    }
    (output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return summary


def sanitize_artifact_component(value: str) -> str:
    return (
        value.replace("/", "-")
        .replace(":", "-")
        .replace(" ", "-")
        .replace("_", "-")
        .lower()
    )


def make_result_artifact(args: argparse.Namespace, summary: dict[str, Any]):
    import wandb

    artifact = wandb.Artifact(
        "agentic-swe-swebench-pro-"
        + sanitize_artifact_component(args.model_name)
        + "-results",
        type="evaluation-results",
        metadata={
            "model_name": args.model_name,
            "benchmark": "SWE-Bench Pro",
            "total_instances": summary["total_instances"],
            "resolved_instances": summary["resolved_instances"],
            "pass_at_1": summary["pass_at_1"],
            "source_output_dir": str(args.output_dir),
            "patch_path": str(args.patch_path),
        },
    )
    if args.patch_path.exists():
        artifact.add_file(str(args.patch_path), name="patches.json")
    for filename in (
        "summary.json",
        "eval_results.json",
        "official_invocation.json",
        "official_stdout.log",
        "official_stderr.log",
    ):
        path = args.output_dir / filename
        if path.exists():
            artifact.add_file(str(path), name=f"official_eval/{filename}")
    return artifact


def maybe_log_wandb(args: argparse.Namespace, summary: dict[str, Any]) -> None:
    if not args.wandb:
        return
    if not args.entity or not args.project:
        raise ValueError("--entity and --project are required with --wandb")
    import pandas as pd
    import wandb

    wandb.login()
    with wandb.init(
        entity=args.entity,
        project=args.project,
        job_type="swebench-pro-eval",
        name=args.run_name or "swebench-pro-eval",
        config=vars(args),
    ) as run:
        leaderboard = pd.DataFrame(
            [
                {
                    "model_name": args.model_name,
                    "total_samples": summary["total_instances"],
                    "issues_resolved": summary["resolved_instances"],
                    "pass_at_1": summary["pass_at_1"],
                }
            ]
        )
        per_instance = pd.DataFrame(
            [
                {"instance_id": iid, "resolved": iid in set(summary["resolved_ids"])}
                for iid in sorted(summary["resolved_ids"] + summary["unresolved_ids"])
            ]
        )
        run.log(
            {
                "agentic_swe_leaderboard_table": wandb.Table(dataframe=leaderboard),
                "agentic_swe_output_table": wandb.Table(dataframe=per_instance),
                "agentic_swe_results": summary,
                "agentic_swe/pass_at_1": float(summary["pass_at_1"]),
                "agentic_swe/resolved_instances": int(summary["resolved_instances"]),
                "agentic_swe/total_instances": int(summary["total_instances"]),
                "agentic_swe/unresolved_instances": int(summary["unresolved_instances"]),
            }
        )
        run.log_artifact(make_result_artifact(args, summary), aliases=["latest", "production"])


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--official-repo", type=Path, default=Path("external/SWE-bench_Pro-os"))
    parser.add_argument("--raw-sample-path", type=Path, required=True)
    parser.add_argument("--patch-path", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--dockerhub-username", default="jefzda")
    parser.add_argument("--scripts-dir", type=Path)
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--use-local-docker", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--docker-platform")
    parser.add_argument("--redo", action="store_true")
    parser.add_argument("--block-network", action="store_true")
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--model-name", default="openclaw")
    parser.add_argument("--wandb", action="store_true")
    parser.add_argument("--entity")
    parser.add_argument("--project")
    parser.add_argument("--run-name")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    official_repo = args.official_repo.resolve()
    validate_official_repo(official_repo)
    if args.use_local_docker:
        check_local_docker()
    scripts_dir = (args.scripts_dir or official_repo / "run_scripts").resolve()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    command = [
        args.python,
        str(official_repo / "swe_bench_pro_eval.py"),
        "--raw_sample_path",
        str(args.raw_sample_path.resolve()),
        "--patch_path",
        str(args.patch_path.resolve()),
        "--output_dir",
        str(args.output_dir.resolve()),
        "--scripts_dir",
        str(scripts_dir),
        "--num_workers",
        str(args.num_workers),
        "--dockerhub_username",
        args.dockerhub_username,
    ]
    if args.use_local_docker:
        command.append("--use_local_docker")
    if args.docker_platform:
        command.extend(["--docker_platform", args.docker_platform])
    if args.redo:
        command.append("--redo")
    if args.block_network:
        command.append("--block_network")

    result = run_command(command, cwd=official_repo, check=False)
    (args.output_dir / "official_stdout.log").write_text(result.stdout, encoding="utf-8")
    (args.output_dir / "official_stderr.log").write_text(result.stderr, encoding="utf-8")
    (args.output_dir / "official_invocation.json").write_text(
        json.dumps(
            {
                "command": command,
                "returncode": result.returncode,
                "cwd": str(official_repo),
            },
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    if result.returncode != 0:
        raise SystemExit(result.returncode)

    summary = write_summary(args.output_dir, load_eval_results(args.output_dir), command)
    maybe_log_wandb(args, summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
