#!/usr/bin/env python3
"""Evaluate SWE-bench Lite patches with the official SWE-bench harness.

This script is intentionally a thin adapter. It converts the OpenClaw patch
records used in this repository into the official SWE-bench prediction format
and delegates scoring to ``swebench.harness.run_evaluation``.
"""

from __future__ import annotations

import argparse
from collections import deque
from concurrent.futures import ThreadPoolExecutor, as_completed
import hashlib
import json
import os
import shlex
import shutil
import signal
import subprocess
import sys
from pathlib import Path
from typing import Any

TOOLS_DIR = Path(__file__).resolve().parent
if str(TOOLS_DIR) not in sys.path:
    sys.path.insert(0, str(TOOLS_DIR))

from agentic_swe_partial_credit import build_diagnostic_partial_credit, from_swebench_report


DEFAULT_DATASET_NAME = "princeton-nlp/SWE-bench_Lite"
MIN_DOCKER_SDK_VERSION = (7, 1, 0)


def resolve_executable(value: str) -> str:
    """Resolve an executable before child processes change working directory."""
    candidate = Path(value).expanduser()
    if candidate.is_absolute() or candidate.parent != Path("."):
        absolute = Path(os.path.abspath(candidate))
        if not absolute.is_file():
            raise FileNotFoundError(f"Executable not found: {absolute}")
        # Do not resolve symlinks: a venv's python symlink must retain its venv path.
        return str(absolute)
    resolved = shutil.which(value)
    if resolved is None:
        raise FileNotFoundError(f"Executable not found on PATH: {value}")
    return os.path.abspath(resolved)


def run_command(
    command: list[str],
    *,
    cwd: Path,
    env: dict[str, str],
    check: bool = True,
    timeout: float | None = None,
) -> subprocess.CompletedProcess[str]:
    proc = subprocess.Popen(
        command,
        cwd=str(cwd),
        env=env,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        start_new_session=True,
    )
    try:
        stdout, stderr = proc.communicate(timeout=timeout)
    except subprocess.TimeoutExpired:
        os.killpg(proc.pid, signal.SIGTERM)
        try:
            stdout, stderr = proc.communicate(timeout=30)
        except subprocess.TimeoutExpired:
            os.killpg(proc.pid, signal.SIGKILL)
            stdout, stderr = proc.communicate()
        raise TimeoutError(
            f"Command timed out after {timeout}s: "
            + " ".join(shlex.quote(part) for part in command)
        )
    result = subprocess.CompletedProcess(command, proc.returncode, stdout, stderr)
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


def check_official_harness_runtime(*, python: str, official_repo: Path) -> None:
    python_path = (
        os.path.abspath(python)
        if os.sep in python or (os.altsep and os.altsep in python)
        else python
    )
    env = dict(os.environ)
    env["PYTHONPATH"] = str(official_repo) + os.pathsep + env.get("PYTHONPATH", "")
    command = [
        python_path,
        "-c",
        (
            "import swebench; "
            "import swebench.harness.run_evaluation"
        ),
    ]
    result = run_command(
        command,
        cwd=official_repo,
        env=env,
        check=False,
        timeout=60,
    )
    if result.returncode != 0:
        raise RuntimeError(
            "Official SWE-bench Python runtime is incomplete. "
            "Install the dependencies declared by external/SWE-bench/pyproject.toml "
            "before paid inference.\n"
            f"python: {python_path}\n"
            f"official_repo: {official_repo}\n"
            f"stderr:\n{result.stderr}"
        )


def validate_complete_report(
    report: dict[str, Any],
    *,
    expected_instance_ids: list[str],
) -> None:
    expected = set(expected_instance_ids)
    submitted = set(report.get("submitted_ids") or [])
    completed = set(report.get("completed_ids") or [])
    error_ids = set(report.get("error_ids") or [])
    incomplete_ids = set(report.get("incomplete_ids") or [])
    problems: list[str] = []
    if submitted != expected:
        problems.append(
            f"submitted IDs differ (expected={len(expected)}, actual={len(submitted)})"
        )
    if completed != expected:
        problems.append(
            f"completed IDs differ (expected={len(expected)}, actual={len(completed)})"
        )
    if error_ids:
        problems.append(f"error_ids={len(error_ids)}")
    if incomplete_ids:
        problems.append(f"incomplete_ids={len(incomplete_ids)}")
    if int(report.get("error_instances") or 0) != 0:
        problems.append(f"error_instances={report.get('error_instances')}")
    if int(report.get("completed_instances") or 0) != len(expected):
        problems.append(
            "completed_instances="
            f"{report.get('completed_instances')} (expected {len(expected)})"
        )
    if problems:
        raise RuntimeError(
            "Official SWE-bench grading is incomplete and cannot be converted "
            "into model scores: "
            + "; ".join(problems)
        )


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


def round_robin_instance_ids_by_repo(instance_ids: list[str]) -> list[str]:
    """Spread same-repository Docker jobs across the official worker queue."""
    repo_order: list[str] = []
    buckets: dict[str, deque[str]] = {}
    for instance_id in instance_ids:
        repo = instance_id.split("__", 1)[0]
        if repo not in buckets:
            repo_order.append(repo)
            buckets[repo] = deque()
        buckets[repo].append(instance_id)

    scheduled: list[str] = []
    while len(scheduled) < len(instance_ids):
        for repo in repo_order:
            if buckets[repo]:
                scheduled.append(buckets[repo].popleft())
    return scheduled


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


def collect_partial_eval_results(
    output_dir: Path,
    *,
    instance_ids: list[str],
    run_id: str,
) -> dict[str, dict[str, Any]]:
    """Collect per-instance F2P/P2P evidence from the current official run."""
    wanted = set(instance_ids)
    detail_by_id: dict[str, tuple[int, dict[str, Any]]] = {}
    logs_root = output_dir / "logs" / "run_evaluation"
    report_roots = [logs_root / run_id]
    isolated_root = output_dir / "isolated"
    if isolated_root.exists():
        for result_path in isolated_root.glob("*/result.json"):
            try:
                result = json.loads(result_path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                continue
            task_run_id = result.get("run_id") if isinstance(result, dict) else None
            if task_run_id:
                report_roots.append(logs_root / str(task_run_id))
    report_paths = sorted(
        {
            report_path
            for report_root in report_roots
            if report_root.exists()
            for report_path in report_root.rglob("report.json")
        }
    )
    for report_path in report_paths:
        try:
            payload = json.loads(report_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        if not isinstance(payload, dict):
            continue
        mtime_ns = report_path.stat().st_mtime_ns
        for instance_id in wanted.intersection(payload):
            detail = payload.get(instance_id)
            if not isinstance(detail, dict):
                continue
            current = detail_by_id.get(instance_id)
            if current is None or mtime_ns >= current[0]:
                detail_by_id[instance_id] = (mtime_ns, detail)

    results: dict[str, dict[str, Any]] = {}
    for instance_id in instance_ids:
        detail_entry = detail_by_id.get(instance_id)
        if detail_entry is None:
            results[instance_id] = build_diagnostic_partial_credit(
                resolved=False,
                f2p_passed=0,
                f2p_total=0,
                p2p_passed=0,
                p2p_total=0,
                patch_applied=False,
                scoreable=False,
                evidence_source="swebench_report_missing",
            )
        else:
            results[instance_id] = from_swebench_report(instance_id, detail_entry[1])
    (output_dir / "partial_eval_results.json").write_text(
        json.dumps(results, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return results


def write_summary(
    output_dir: Path,
    *,
    report: dict[str, Any],
    eval_results: dict[str, bool],
    partial_eval_results: dict[str, dict[str, Any]] | None,
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
        "diagnostic_score_with_partial": (
            sum(
                1.0
                if is_resolved
                else float(
                    (partial_eval_results or {}).get(instance_id, {}).get(
                        "diagnostic_score_with_partial", 0.0
                    )
                )
                for instance_id, is_resolved in eval_results.items()
            )
            / total
            if total
            else 0.0
        ),
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


def build_command(
    args: argparse.Namespace,
    predictions_path: Path,
    instance_ids: list[str],
    *,
    run_id: str | None = None,
    report_dir: Path | None = None,
    max_workers: int | None = None,
) -> list[str]:
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
        str(args.max_workers if max_workers is None else max_workers),
        "--run_id",
        run_id or args.run_id,
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
        str((report_dir or args.output_dir).resolve()),
    ]
    if args.rewrite_reports:
        command.extend(["--rewrite_reports", "true"])
    command.append("--instance_ids")
    command.extend(instance_ids)
    return command


def isolated_run_id(base_run_id: str, instance_id: str) -> str:
    digest = hashlib.sha256(instance_id.encode("utf-8")).hexdigest()[:10]
    return f"{base_run_id}-{digest}"


def load_detailed_instance_report(
    *,
    cwd: Path,
    run_id: str,
    model_name: str,
    instance_id: str,
) -> dict[str, Any] | None:
    report_path = (
        cwd
        / "logs"
        / "run_evaluation"
        / run_id
        / model_name.replace("/", "__")
        / instance_id
        / "report.json"
    )
    try:
        payload = json.loads(report_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    detail = payload.get(instance_id) if isinstance(payload, dict) else None
    return detail if isinstance(detail, dict) else None


def single_instance_summary(
    instance_id: str,
    detail: dict[str, Any],
) -> dict[str, Any]:
    resolved = detail.get("resolved") is True
    empty_patch = detail.get("patch_exists") is False
    return {
        "schema_version": 2,
        "total_instances": 1,
        "submitted_instances": 1,
        "completed_instances": 1,
        "resolved_instances": int(resolved),
        "unresolved_instances": int(not resolved),
        "empty_patch_instances": int(empty_patch),
        "error_instances": 0,
        "submitted_ids": [instance_id],
        "completed_ids": [instance_id],
        "resolved_ids": [instance_id] if resolved else [],
        "unresolved_ids": [] if resolved else [instance_id],
        "empty_patch_ids": [instance_id] if empty_patch else [],
        "error_ids": [],
        "incomplete_ids": [],
    }


def evaluate_isolated_instances(
    args: argparse.Namespace,
    *,
    predictions_path: Path,
    instance_ids: list[str],
    cwd: Path,
    env: dict[str, str],
) -> tuple[dict[str, Any], list[list[str]]]:
    isolated_root = args.output_dir / "isolated"
    isolated_root.mkdir(parents=True, exist_ok=True)
    commands: list[list[str]] = []

    def evaluate_one(instance_id: str) -> dict[str, Any]:
        task_dir = isolated_root / instance_id.replace("/", "_")
        task_dir.mkdir(parents=True, exist_ok=True)
        result_path = task_dir / "result.json"
        if result_path.exists():
            try:
                cached = json.loads(result_path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                cached = None
            if (
                isinstance(cached, dict)
                and cached.get("instance_id") == instance_id
                and cached.get("ok") is True
            ):
                return cached
            if isinstance(cached, dict) and cached.get("instance_id") == instance_id:
                cached_run_id = str(cached.get("run_id") or "")
                detail = load_detailed_instance_report(
                    cwd=cwd,
                    run_id=cached_run_id,
                    model_name=args.model_name,
                    instance_id=instance_id,
                )
                if detail is not None:
                    cached_error = str(cached.get("error") or "")
                    cached["ok"] = True
                    cached["error"] = ""
                    cached["official_report"] = single_instance_summary(
                        instance_id, detail
                    )
                    cached["cleanup_warning"] = (
                        cached.get("cleanup_warning")
                        or cached_error
                        or "Recovered from the detailed official report."
                    )
                    result_path.write_text(
                        json.dumps(cached, ensure_ascii=False, indent=2) + "\n",
                        encoding="utf-8",
                    )
                    return cached

        timeout = float(args.timeout) + float(args.isolated_cleanup_grace_seconds)
        attempts: list[dict[str, Any]] = []
        report: dict[str, Any] = {}
        ok = False
        error = ""
        cleanup_warning = ""
        task_run_id = ""
        command: list[str] = []
        max_attempts = 1 + max(0, int(args.isolated_infrastructure_retries))
        for attempt_number in range(1, max_attempts + 1):
            base_run_id = isolated_run_id(args.run_id, instance_id)
            task_run_id = (
                base_run_id
                if attempt_number == 1
                else f"{base_run_id}-attempt{attempt_number}"
            )
            command = build_command(
                args,
                predictions_path,
                [instance_id],
                run_id=task_run_id,
                report_dir=task_dir,
                max_workers=1,
            )
            commands.append(command)
            try:
                result = run_command(
                    command,
                    cwd=cwd,
                    env=env,
                    check=False,
                    timeout=timeout,
                )
                stdout = result.stdout
                stderr = result.stderr
                returncode: int | None = result.returncode
                error = (
                    ""
                    if result.returncode == 0
                    else f"official harness return code {result.returncode}"
                )
            except TimeoutError as exc:
                stdout = ""
                stderr = ""
                returncode = None
                error = str(exc)

            for filename, content in (
                ("official_stdout.log", stdout),
                ("official_stderr.log", stderr),
                (f"official_stdout.attempt{attempt_number}.log", stdout),
                (f"official_stderr.attempt{attempt_number}.log", stderr),
            ):
                (task_dir / filename).write_text(content, encoding="utf-8")

            report_name = report_filename(args.model_name, task_run_id)
            report_path = task_dir / report_name
            fallback_report_path = cwd / report_name
            if not report_path.exists() and fallback_report_path.exists():
                report_path.write_text(
                    fallback_report_path.read_text(encoding="utf-8"),
                    encoding="utf-8",
                )
            if report_path.exists():
                try:
                    report = json.loads(report_path.read_text(encoding="utf-8"))
                except (OSError, json.JSONDecodeError):
                    report = {}

            detail = load_detailed_instance_report(
                cwd=cwd,
                run_id=task_run_id,
                model_name=args.model_name,
                instance_id=instance_id,
            )
            if not report and detail is not None:
                report = single_instance_summary(instance_id, detail)
                cleanup_warning = (
                    error
                    or "Official summary was missing after the detailed report was saved."
                )
            elif report and error:
                cleanup_warning = error
            ok = bool(report)
            attempts.append(
                {
                    "attempt": attempt_number,
                    "run_id": task_run_id,
                    "returncode": returncode,
                    "accepted_detailed_report": detail is not None and bool(report),
                    "error": error,
                }
            )
            if ok:
                error = ""
                break

        payload = {
            "instance_id": instance_id,
            "run_id": task_run_id,
            "ok": ok,
            "error": error,
            "cleanup_warning": cleanup_warning,
            "official_report": report,
            "command": command,
            "attempts": attempts,
        }
        result_path.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        return payload

    outcomes: dict[str, dict[str, Any]] = {}
    with ThreadPoolExecutor(max_workers=max(1, int(args.max_workers))) as executor:
        future_to_id = {
            executor.submit(evaluate_one, instance_id): instance_id
            for instance_id in instance_ids
        }
        for future in as_completed(future_to_id):
            instance_id = future_to_id[future]
            outcomes[instance_id] = future.result()
            completed = len(outcomes)
            ok_count = sum(1 for row in outcomes.values() if row.get("ok"))
            print(
                f"Official isolated SWE-bench progress: {completed}/{len(instance_ids)} "
                f"(completed={ok_count}, errors={completed - ok_count})",
                flush=True,
            )

    resolved_ids: list[str] = []
    completed_ids: list[str] = []
    error_ids: list[str] = []
    empty_patch_ids: list[str] = []
    predictions_by_id = {
        row["instance_id"]: row
        for row in (
            json.loads(line)
            for line in predictions_path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        )
    }
    for instance_id in instance_ids:
        outcome = outcomes[instance_id]
        report = outcome.get("official_report")
        if outcome.get("ok") and isinstance(report, dict):
            completed_ids.append(instance_id)
            if instance_id in set(report.get("resolved_ids") or []):
                resolved_ids.append(instance_id)
        else:
            error_ids.append(instance_id)
        if not str(predictions_by_id[instance_id].get("model_patch") or ""):
            empty_patch_ids.append(instance_id)

    unresolved_ids = [
        instance_id for instance_id in instance_ids if instance_id not in set(resolved_ids)
    ]
    report = {
        "schema_version": 2,
        "total_instances": len(instance_ids),
        "submitted_instances": len(instance_ids),
        "completed_instances": len(completed_ids),
        "resolved_instances": len(resolved_ids),
        "unresolved_instances": len(unresolved_ids),
        "empty_patch_instances": len(empty_patch_ids),
        "error_instances": len(error_ids),
        "submitted_ids": instance_ids,
        "completed_ids": completed_ids,
        "resolved_ids": resolved_ids,
        "unresolved_ids": unresolved_ids,
        "empty_patch_ids": empty_patch_ids,
        "error_ids": error_ids,
        "incomplete_ids": error_ids,
    }
    return report, commands


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--official-repo", type=Path, default=Path("external/SWE-bench"))
    parser.add_argument("--dataset-name", default=DEFAULT_DATASET_NAME)
    parser.add_argument("--split", default="test")
    parser.add_argument("--instance-ids-json", type=Path)
    parser.add_argument("--patch-path", type=Path)
    parser.add_argument("--output-dir", type=Path)
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
    parser.add_argument(
        "--isolate-instances",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "Run each instance in its own official-harness subprocess. This contains "
            "Docker cleanup stalls and makes successful instances resumable."
        ),
    )
    parser.add_argument(
        "--isolated-cleanup-grace-seconds",
        type=float,
        default=120.0,
        help="Extra wall time beyond --timeout before killing a stuck isolated harness.",
    )
    parser.add_argument(
        "--isolated-infrastructure-retries",
        type=int,
        default=1,
        help=(
            "Retries for an isolated official-harness failure that leaves no "
            "scoreable detailed report. Model test failures are not retried."
        ),
    )
    parser.add_argument(
        "--schedule-by-repo",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "Round-robin instance IDs by repository so concurrent official workers "
            "do not contend on the same large Docker environment."
        ),
    )
    parser.add_argument("--prepare-only", action="store_true")
    parser.add_argument(
        "--runtime-check-only",
        action="store_true",
        help="Validate official harness imports and Docker before paid inference.",
    )
    parser.add_argument("--no-docker-check", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.python = resolve_executable(args.python)
    official_repo = args.official_repo.resolve()
    validate_official_repo(official_repo)
    if args.runtime_check_only:
        check_official_harness_runtime(
            python=args.python,
            official_repo=official_repo,
        )
        if not args.no_docker_check:
            check_local_docker()
        print(
            json.dumps(
                {
                    "ok": True,
                    "python": args.python,
                    "official_repo": str(official_repo),
                    "docker_checked": not args.no_docker_check,
                },
                ensure_ascii=False,
                indent=2,
            )
        )
        return
    if args.patch_path is None or args.output_dir is None:
        raise SystemExit(
            "--patch-path and --output-dir are required unless "
            "--runtime-check-only is used"
        )
    args.output_dir.mkdir(parents=True, exist_ok=True)

    patch_rows = load_patch_rows(args.patch_path)
    instance_ids = load_instance_ids(args.instance_ids_json, patch_rows)
    if args.schedule_by_repo:
        instance_ids = round_robin_instance_ids_by_repo(instance_ids)
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
        check_official_harness_runtime(
            python=args.python,
            official_repo=official_repo,
        )
        check_local_docker()

    env = dict(os.environ)
    env["PYTHONPATH"] = str(official_repo) + os.pathsep + env.get("PYTHONPATH", "")
    if args.isolate_instances:
        report, commands = evaluate_isolated_instances(
            args,
            predictions_path=predictions_path,
            instance_ids=instance_ids,
            cwd=args.output_dir.resolve(),
            env=env,
        )
        command = ["isolated-instance-harness", f"count={len(commands)}"]
        report_path = args.output_dir / report_filename(args.model_name, args.run_id)
        report_path.write_text(
            json.dumps(report, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
    else:
        result = run_command(command, cwd=args.output_dir.resolve(), env=env, check=False)
        (args.output_dir / "official_stdout.log").write_text(
            result.stdout, encoding="utf-8"
        )
        (args.output_dir / "official_stderr.log").write_text(
            result.stderr, encoding="utf-8"
        )
        if result.returncode != 0:
            raise SystemExit(result.returncode)

        report_path = args.output_dir / report_filename(args.model_name, args.run_id)
        if not report_path.exists():
            raise FileNotFoundError(
                f"Expected official SWE-bench report not found: {report_path}"
            )
        report = json.loads(report_path.read_text(encoding="utf-8"))
    validate_complete_report(report, expected_instance_ids=instance_ids)
    eval_results = write_eval_results(args.output_dir, report)
    partial_eval_results = collect_partial_eval_results(
        args.output_dir,
        instance_ids=instance_ids,
        run_id=args.run_id,
    )
    summary = write_summary(
        args.output_dir,
        report=report,
        eval_results=eval_results,
        partial_eval_results=partial_eval_results,
        command=command,
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
