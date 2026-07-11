#!/usr/bin/env python3
"""Evaluate SWE-bench Lite patches with the official SWE-bench harness.

This script is intentionally a thin adapter. It converts the OpenClaw patch
records used in this repository into the official SWE-bench prediction format
and delegates scoring to ``swebench.harness.run_evaluation``.
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


DEFAULT_DATASET_NAME = "princeton-nlp/SWE-bench_Lite"
MIN_DOCKER_SDK_VERSION = (7, 1, 0)


def run_command(
    command: list[str],
    *,
    cwd: Path,
    env: dict[str, str],
    check: bool = True,
) -> subprocess.CompletedProcess[str]:
    result = subprocess.run(
        command,
        cwd=str(cwd),
        env=env,
        text=True,
        capture_output=True,
        check=False,
    )
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
        path / "swebench" / "harness" / "run_evaluation.py",
        path / "swebench" / "harness" / "reporting.py",
        path / "pyproject.toml",
    ]
    missing = [str(item) for item in required if not item.exists()]
    if missing:
        raise FileNotFoundError(
            "Official SWE-bench checkout is incomplete. Missing:\n"
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
            "Python Docker SDK is required. Install it with: "
            "python3 -m pip install --user -U 'docker>=7.1.0'"
        ) from exc

    raw_version = getattr(docker, "__version__", "0")
    version = parse_version(raw_version)
    if version < MIN_DOCKER_SDK_VERSION:
        minimum = ".".join(str(part) for part in MIN_DOCKER_SDK_VERSION)
        raise SystemExit(
            f"Python Docker SDK {raw_version} is too old. Install {minimum}+ with: "
            f"python3 -m pip install --user -U 'docker>={minimum}'"
        )

    try:
        client = docker.from_env()
        client.ping()
    except Exception as exc:
        raise SystemExit("Docker is not reachable from this shell. Check `docker info`.") from exc
    finally:
        close = getattr(locals().get("client", None), "close", None)
        if callable(close):
            close()


def load_patch_rows(path: Path) -> list[dict[str, Any]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, list):
        raise ValueError(f"{path} must contain a JSON list")
    rows: list[dict[str, Any]] = []
    for index, row in enumerate(payload, start=1):
        if not isinstance(row, dict):
            raise ValueError(f"{path}:{index} must contain a JSON object")
        if not row.get("instance_id"):
            raise ValueError(f"{path}:{index} is missing instance_id")
        rows.append(row)
    return rows


def load_instance_ids(path: Path | None, rows: list[dict[str, Any]]) -> list[str]:
    if path is None:
        return [str(row["instance_id"]) for row in rows]
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, list):
        raise ValueError(f"{path} must contain a JSON list")
    ids = [str(item) for item in payload]
    row_ids = {str(row["instance_id"]) for row in rows}
    missing = [instance_id for instance_id in ids if instance_id not in row_ids]
    if missing:
        raise ValueError(f"{path} contains IDs without patches: {missing[:10]}")
    return ids


def write_predictions(
    *,
    patch_rows: list[dict[str, Any]],
    instance_ids: list[str],
    model_name: str,
    output_path: Path,
) -> list[dict[str, Any]]:
    rows_by_id = {str(row["instance_id"]): row for row in patch_rows}
    predictions: list[dict[str, Any]] = []
    for instance_id in instance_ids:
        row = rows_by_id[instance_id]
        model_patch = row.get("model_patch", row.get("patch", ""))
        predictions.append(
            {
                "instance_id": instance_id,
                "model_patch": "" if model_patch is None else str(model_patch),
                "model_name_or_path": model_name,
            }
        )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        for prediction in predictions:
            f.write(json.dumps(prediction, ensure_ascii=False) + "\n")
    return predictions


def report_filename(model_name: str, run_id: str) -> str:
    return model_name.replace("/", "__") + f".{run_id}.json"


def write_eval_results(output_dir: Path, report: dict[str, Any]) -> dict[str, bool]:
    resolved = set(report.get("resolved_ids") or [])
    all_ids = (
        set(report.get("submitted_ids") or [])
        | set(report.get("resolved_ids") or [])
        | set(report.get("unresolved_ids") or [])
        | set(report.get("empty_patch_ids") or [])
        | set(report.get("error_ids") or [])
    )
    eval_results = {instance_id: instance_id in resolved for instance_id in sorted(all_ids)}
    (output_dir / "eval_results.json").write_text(
        json.dumps(eval_results, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return eval_results


def write_summary(
    output_dir: Path,
    *,
    report: dict[str, Any],
    eval_results: dict[str, bool],
    command: list[str],
) -> dict[str, Any]:
    total = len(eval_results)
    resolved = sum(1 for value in eval_results.values() if value)
    summary = {
        "total_instances": total,
        "submitted_instances": int(report.get("submitted_instances") or total),
        "completed_instances": int(report.get("completed_instances") or 0),
        "resolved_instances": resolved,
        "unresolved_instances": total - resolved,
        "empty_patch_instances": int(report.get("empty_patch_instances") or 0),
        "error_instances": int(report.get("error_instances") or 0),
        "pass_at_1": resolved / total if total else 0.0,
        "resolved_ids": sorted([key for key, value in eval_results.items() if value]),
        "unresolved_ids": sorted([key for key, value in eval_results.items() if not value]),
        "official_report": report,
        "command": command,
    }
    (output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return summary


def build_command(args: argparse.Namespace, predictions_path: Path, instance_ids: list[str]) -> list[str]:
    command = [
        args.python,
        "-m",
        "swebench.harness.run_evaluation",
        "--dataset_name",
        args.dataset_name,
        "--split",
        args.split,
        "--predictions_path",
        str(predictions_path.resolve()),
        "--max_workers",
        str(args.max_workers),
        "--run_id",
        args.run_id,
        "--timeout",
        str(args.timeout),
        "--cache_level",
        args.cache_level,
        "--clean",
        str(bool(args.clean)).lower(),
        "--force_rebuild",
        str(bool(args.force_rebuild)).lower(),
        "--namespace",
        args.namespace,
        "--report_dir",
        str(args.output_dir.resolve()),
    ]
    if args.rewrite_reports:
        command.extend(["--rewrite_reports", "true"])
    command.append("--instance_ids")
    command.extend(instance_ids)
    return command


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--official-repo", type=Path, default=Path("external/SWE-bench"))
    parser.add_argument("--dataset-name", default=DEFAULT_DATASET_NAME)
    parser.add_argument("--split", default="test")
    parser.add_argument("--instance-ids-json", type=Path)
    parser.add_argument("--patch-path", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--model-name", default="openclaw")
    parser.add_argument("--run-id", default="agentic-swe-lite")
    parser.add_argument("--max-workers", type=int, default=4)
    parser.add_argument("--timeout", type=int, default=1800)
    parser.add_argument("--cache-level", choices=["none", "base", "env", "instance"], default="env")
    parser.add_argument("--clean", action="store_true")
    parser.add_argument("--force-rebuild", action="store_true")
    parser.add_argument("--rewrite-reports", action="store_true")
    parser.add_argument("--namespace", default="swebench")
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--prepare-only", action="store_true")
    parser.add_argument("--no-docker-check", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    official_repo = args.official_repo.resolve()
    validate_official_repo(official_repo)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    patch_rows = load_patch_rows(args.patch_path)
    instance_ids = load_instance_ids(args.instance_ids_json, patch_rows)
    predictions_path = args.output_dir / "predictions.jsonl"
    write_predictions(
        patch_rows=patch_rows,
        instance_ids=instance_ids,
        model_name=args.model_name,
        output_path=predictions_path,
    )

    command = build_command(args, predictions_path, instance_ids)
    invocation = {
        "command": command,
        "official_repo": str(official_repo),
        "patch_path": str(args.patch_path),
        "predictions_path": str(predictions_path),
        "instance_ids": instance_ids,
        "prepare_only": bool(args.prepare_only),
    }
    (args.output_dir / "official_invocation.json").write_text(
        json.dumps(invocation, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    if args.prepare_only:
        print(json.dumps(invocation, ensure_ascii=False, indent=2))
        return

    if not args.no_docker_check:
        check_local_docker()

    env = dict(os.environ)
    env["PYTHONPATH"] = str(official_repo) + os.pathsep + env.get("PYTHONPATH", "")
    result = run_command(command, cwd=args.output_dir.resolve(), env=env, check=False)
    (args.output_dir / "official_stdout.log").write_text(result.stdout, encoding="utf-8")
    (args.output_dir / "official_stderr.log").write_text(result.stderr, encoding="utf-8")
    if result.returncode != 0:
        raise SystemExit(result.returncode)

    report_path = args.output_dir / report_filename(args.model_name, args.run_id)
    if not report_path.exists():
        raise FileNotFoundError(f"Expected official SWE-bench report not found: {report_path}")
    report = json.loads(report_path.read_text(encoding="utf-8"))
    eval_results = write_eval_results(args.output_dir, report)
    summary = write_summary(
        args.output_dir,
        report=report,
        eval_results=eval_results,
        command=command,
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
