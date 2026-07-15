"""Run DeepSWE tasks with the Nejumi OpenClaw Pier adapter."""

from __future__ import annotations

import argparse
import json
import os
import re
import signal
import subprocess
import sys
import threading
import time
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[2]
AGENT_IMPORT_PATH = "deepswe_openclaw_pier_agent:NejumiDeepSWEOpenClawAgent"
DEFAULT_TASKS_ROOT = REPO_ROOT / "external" / "deep-swe" / "tasks"
DEFAULT_JOBS_DIR = REPO_ROOT / "outputs" / "deepswe_openclaw_pier_jobs"
DEEPSWE_SANDBOX_DEPS_SCRIPT = REPO_ROOT / "scripts" / "setup" / "install_deepswe_sandbox_deps.sh"
RUNNER_VERSION = "run-deepswe-openclaw-2026-07-11-v1"
DOCKER_IMAGE_RE = re.compile(r'^\s*docker_image\s*=\s*["\']([^"\']+)["\']\s*$')
LANGUAGE_RE = re.compile(r'^\s*language\s*=\s*["\']([^"\']+)["\']\s*$')
NEMOCLAW_STALE_SHIELDS_LOCK_RE = re.compile(
    r"shields transition lock '(?P<path>[^']+)': recorded owner PID "
    r"(?P<pid>\d+) is not running",
    re.IGNORECASE,
)
NEMOCLAW_STATE_DIR = Path.home() / ".nemoclaw" / "state"

LANGUAGE_REQUIRED_TOOLS = {
    "go": ["go"],
    "golang": ["go"],
    "javascript": ["node"],
    "typescript": ["node"],
    "python": ["python3"],
    "rust": ["cargo", "rustc"],
}


class DeepSWEPreflightError(RuntimeError):
    def __init__(self, message: str, payload: dict[str, Any] | None = None) -> None:
        super().__init__(message)
        self.payload = payload or {}


def load_dotenv(path: Path) -> None:
    if not path.exists():
        return
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip().strip("'\"")
        if key and key not in os.environ:
            os.environ[key] = value


def read_json(path: Path) -> dict[str, Any]:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (FileNotFoundError, json.JSONDecodeError):
        return {}


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def read_task_names(path: Path | None) -> list[str] | None:
    if path is None:
        return None
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, list) or not all(isinstance(item, str) for item in payload):
        raise ValueError(f"{path} must contain a JSON list of task-name strings")
    return payload


def configured_task_names(args: argparse.Namespace) -> list[str] | None:
    task_names = read_task_names(args.task_names_file)
    if args.include_task_name:
        include_names = list(args.include_task_name)
        if task_names is None:
            task_names = include_names
        else:
            include_set = set(include_names)
            filtered = [name for name in task_names if name in include_set]
            missing = [name for name in include_names if name not in set(task_names)]
            if missing:
                raise ValueError(
                    "--include-task-name contains names not present in --task-names-file: "
                    + ", ".join(missing)
                )
            task_names = filtered
    return task_names


def read_task_docker_image(tasks_root: Path, task_name: str) -> str | None:
    task_toml = tasks_root / task_name / "task.toml"
    try:
        for line in task_toml.read_text(encoding="utf-8").splitlines():
            match = DOCKER_IMAGE_RE.match(line)
            if match:
                return match.group(1)
    except FileNotFoundError:
        return None
    return None


def read_task_language(tasks_root: Path, task_name: str) -> str | None:
    task_toml = tasks_root / task_name / "task.toml"
    try:
        for line in task_toml.read_text(encoding="utf-8").splitlines():
            match = LANGUAGE_RE.match(line)
            if match:
                return match.group(1).strip().lower()
    except FileNotFoundError:
        return None
    return None


def selected_task_names_for_preflight(args: argparse.Namespace) -> list[str]:
    task_names = configured_task_names(args)
    if task_names is not None:
        return task_names
    if not args.tasks_root.exists():
        return []
    return sorted(
        path.name for path in args.tasks_root.iterdir() if (path / "task.toml").exists()
    )


def selected_task_languages(args: argparse.Namespace) -> dict[str, str | None]:
    return {
        task_name: read_task_language(args.tasks_root, task_name)
        for task_name in selected_task_names_for_preflight(args)
    }


def required_tools_for_languages(languages: dict[str, str | None]) -> dict[str, list[str]]:
    tools: dict[str, list[str]] = {}
    for language in sorted({value for value in languages.values() if value}):
        required = LANGUAGE_REQUIRED_TOOLS.get(language)
        if required:
            tools[language] = required
    return tools


def _pid_exists(pid: int) -> bool:
    if pid <= 0:
        return False
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def _remove_stale_nemoclaw_locks_from_text(text: str) -> list[str]:
    removed: list[str] = []
    state_dir = NEMOCLAW_STATE_DIR.resolve(strict=False)
    for match in NEMOCLAW_STALE_SHIELDS_LOCK_RE.finditer(text or ""):
        lock_path = Path(match.group("path")).expanduser()
        lock_resolved = lock_path.resolve(strict=False)
        try:
            if not lock_resolved.is_relative_to(state_dir):
                continue
        except ValueError:
            continue
        if not lock_resolved.name.startswith("shields-transition-lock-"):
            continue
        try:
            recorded_pid = int(match.group("pid"))
        except (TypeError, ValueError):
            continue
        if _pid_exists(recorded_pid):
            continue
        lock_payload = read_json(lock_resolved)
        try:
            lock_pid = int(lock_payload.get("pid"))
        except (TypeError, ValueError):
            continue
        if lock_pid != recorded_pid:
            continue
        if _pid_exists(recorded_pid):
            continue
        try:
            lock_resolved.unlink()
            removed.append(str(lock_resolved))
        except FileNotFoundError:
            pass
    return removed


def _check_tools_with_command(command: list[str]) -> dict[str, dict[str, str | None]]:
    removed_stale_locks: list[str] = []
    result = subprocess.run(
        command,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )
    if result.returncode != 0:
        removed_stale_locks = _remove_stale_nemoclaw_locks_from_text(
            (result.stderr or "") + "\n" + (result.stdout or "")
        )
        if removed_stale_locks:
            result = subprocess.run(
                command,
                text=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                check=False,
            )
    if result.returncode != 0:
        stale_lock_note = (
            "\nremoved stale NeMoClaw locks before retry:\n"
            + "\n".join(removed_stale_locks)
            if removed_stale_locks
            else ""
        )
        raise RuntimeError(
            "DeepSWE OpenClaw sandbox preflight failed before model execution: "
            f"tool inspection command failed with exit code {result.returncode}\n"
            f"stdout:\n{result.stdout[-2000:]}\n"
            f"stderr:\n{result.stderr[-2000:]}"
            f"{stale_lock_note}"
        )
    availability: dict[str, dict[str, str | None]] = {}
    for raw_line in result.stdout.splitlines():
        parts = raw_line.rstrip("\n").split("\t", 2)
        if len(parts) < 2:
            continue
        command_name, status = parts[0], parts[1]
        availability[command_name] = {
            "available": status == "OK",
            "path": parts[2] if len(parts) > 2 and status == "OK" else None,
        }
    return availability


def openclaw_execution_tool_availability(
    args: argparse.Namespace, required_tools: list[str]
) -> dict[str, dict[str, str | bool | None]]:
    required_tools = sorted(set(required_tools))
    if not required_tools:
        return {}
    if bool(getattr(args, "no_local", True)) and getattr(args, "nemoclaw_sandbox", None):
        shell_script = (
            'for cmd in "$@"; do '
            'if command -v "$cmd" >/dev/null 2>&1; then '
            'printf "%s\\tOK\\t%s\\n" "$cmd" "$(command -v "$cmd")"; '
            'else printf "%s\\tMISSING\\n" "$cmd"; fi; '
            "done"
        )
        command = [
            str(args.nemoclaw_bin),
            "sandbox",
            "exec",
            str(args.nemoclaw_sandbox),
            "--workdir",
            str(args.nemoclaw_workdir),
            "--no-tty",
            "--timeout",
            "30",
            "--",
            "sh",
            "-c",
            shell_script,
            "_",
            *required_tools,
        ]
        return _check_tools_with_command(command)

    availability: dict[str, dict[str, str | bool | None]] = {}
    for tool in required_tools:
        path = None
        for directory in os.environ.get("PATH", "").split(os.pathsep):
            candidate = Path(directory) / tool
            if candidate.exists() and os.access(candidate, os.X_OK):
                path = str(candidate)
                break
        availability[tool] = {"available": path is not None, "path": path}
    return availability


def preflight_openclaw_sandbox_tools(args: argparse.Namespace) -> dict[str, Any]:
    if not bool(getattr(args, "preflight_openclaw_sandbox_tools", True)):
        return {"enabled": False}

    languages = selected_task_languages(args)
    required_by_language = required_tools_for_languages(languages)
    required_tools = sorted({tool for tools in required_by_language.values() for tool in tools})
    availability = openclaw_execution_tool_availability(args, required_tools)
    missing = [tool for tool in required_tools if not availability.get(tool, {}).get("available")]
    payload = {
        "enabled": True,
        "task_languages": languages,
        "required_tools_by_language": required_by_language,
        "required_tools": required_tools,
        "availability": availability,
        "missing_tools": missing,
        "execution_mode": "nemoclaw_sandbox"
        if bool(getattr(args, "no_local", True)) and getattr(args, "nemoclaw_sandbox", None)
        else "local",
    }
    if missing:
        affected_languages = [
            language
            for language, tools in required_by_language.items()
            if any(tool in missing for tool in tools)
        ]
        raise DeepSWEPreflightError(
            "DeepSWE OpenClaw sandbox preflight failed before model execution: "
            "selected tasks require commands missing from the OpenClaw execution sandbox: "
            + ", ".join(missing)
            + f" (affected languages: {', '.join(affected_languages)}). "
            "This is an evaluation setup mismatch, not a model incorrect answer. "
            "Run OpenClaw inside the DeepSWE/Pier task image or remove those tasks from this run.",
            payload,
        )
    return payload


def selected_docker_images(args: argparse.Namespace) -> list[str]:
    task_names = configured_task_names(args)
    if task_names is None:
        return []
    images: list[str] = []
    seen: set[str] = set()
    missing: list[str] = []
    for task_name in task_names:
        image = read_task_docker_image(args.tasks_root, task_name)
        if not image:
            missing.append(task_name)
            continue
        if image not in seen:
            seen.add(image)
            images.append(image)
    if missing:
        raise RuntimeError(
            "DeepSWE Docker preflight could not find docker_image in task.toml for: "
            + ", ".join(missing)
        )
    return images


def docker_image_present(image: str) -> bool:
    result = subprocess.run(
        ["docker", "image", "inspect", image],
        text=True,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        check=False,
    )
    return result.returncode == 0


def pull_docker_image(image: str, *, retries: int, retry_seconds: float) -> None:
    attempts = max(1, int(retries))
    last_output = ""
    for attempt in range(1, attempts + 1):
        print(f"Pre-pulling DeepSWE Docker image ({attempt}/{attempts}): {image}", flush=True)
        result = subprocess.run(
            ["docker", "pull", image],
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            check=False,
        )
        last_output = result.stdout or ""
        if result.returncode == 0 and docker_image_present(image):
            return
        if attempt < attempts:
            time.sleep(max(0.0, float(retry_seconds)))
    raise RuntimeError(
        "DeepSWE Docker preflight failed before model execution: "
        f"could not pull required image {image}\n{last_output[-2000:]}"
    )


def preflight_docker_images(args: argparse.Namespace) -> dict[str, Any]:
    if not bool(getattr(args, "preflight_docker_images", True)):
        return {"enabled": False, "images": []}
    images = selected_docker_images(args)
    missing = [image for image in images if not docker_image_present(image)]
    if missing and bool(getattr(args, "prepull_missing_docker_images", True)):
        for image in missing:
            pull_docker_image(
                image,
                retries=max(1, int(getattr(args, "docker_pull_retries", 3) or 1)),
                retry_seconds=float(getattr(args, "docker_pull_retry_seconds", 30.0) or 0.0),
            )
        missing = [image for image in images if not docker_image_present(image)]
    if missing:
        raise RuntimeError(
            "DeepSWE Docker preflight failed before model execution: missing required images: "
            + ", ".join(missing)
            + ". Pull/cache these images first or enable --prepull-missing-docker-images."
        )
    return {
        "enabled": True,
        "images": images,
        "missing_after_preflight": missing,
        "prepull_missing": bool(getattr(args, "prepull_missing_docker_images", True)),
    }


def prepare_openclaw_sandbox_runtime(args: argparse.Namespace) -> dict[str, Any]:
    if not bool(getattr(args, "prepare_sandbox_runtime", True)):
        return {"enabled": False, "reason": "disabled"}
    if not bool(getattr(args, "no_local", True)) or not getattr(args, "nemoclaw_sandbox", None):
        return {"enabled": False, "reason": "not_nemoclaw_sandbox_mode"}
    task_names = configured_task_names(args)
    if task_names is None:
        return {"enabled": False, "reason": "task_names_file_required"}
    if not DEEPSWE_SANDBOX_DEPS_SCRIPT.exists():
        raise RuntimeError(
            "DeepSWE OpenClaw sandbox runtime preflight failed before model execution: "
            f"missing setup script {DEEPSWE_SANDBOX_DEPS_SCRIPT}"
        )
    setup_task_names_file = args.task_names_file
    if setup_task_names_file is None or task_names != read_task_names(setup_task_names_file):
        setup_task_names_file = args.output_dir / "sandbox_runtime_task_names.json"
        write_json(setup_task_names_file, task_names)
    command = [
        str(DEEPSWE_SANDBOX_DEPS_SCRIPT),
        "--sandbox",
        str(args.nemoclaw_sandbox),
        "--nemoclaw-bin",
        str(args.nemoclaw_bin),
        "--tasks-root",
        str(args.tasks_root),
        "--task-names-file",
        str(setup_task_names_file),
        "--no-pull",
    ]
    started_at = time.time()
    result = subprocess.run(
        command,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=False,
    )
    payload = {
        "enabled": True,
        "command": command,
        "returncode": result.returncode,
        "stdout": result.stdout[-6000:],
        "elapsed_seconds": time.time() - started_at,
    }
    if result.returncode != 0:
        raise RuntimeError(
            "DeepSWE OpenClaw sandbox runtime preflight failed before model execution. "
            "The selected task image dependencies could not be made visible inside "
            f"the NeMoClaw sandbox. See output:\n{result.stdout[-4000:]}"
        )
    return payload


def verifier_score(verifier_result: Any) -> float | None:
    if not isinstance(verifier_result, dict):
        return None
    for key in ("reward", "score", "mean", "pass", "passed"):
        value = verifier_result.get(key)
        if isinstance(value, bool):
            return 1.0 if value else 0.0
        if isinstance(value, (int, float)):
            return float(value)
    rewards = verifier_result.get("rewards")
    if isinstance(rewards, dict):
        return verifier_score(rewards)
    metrics = verifier_result.get("metrics")
    if isinstance(metrics, dict):
        return verifier_score(metrics)
    return None


def deepswe_task_id_from_name(task_name: Any) -> str | None:
    if not isinstance(task_name, str) or not task_name.strip():
        return None
    return task_name.rstrip("/").split("/")[-1] or None


def fallback_openclaw_metadata(output_dir: Path, task_name: Any) -> dict[str, Any]:
    task_id = deepswe_task_id_from_name(task_name)
    if not task_id:
        return {}
    metadata_path = output_dir / "openclaw" / task_id / "deepswe_openclaw_metadata.json"
    return read_json(metadata_path)


def _numeric_value(value: Any) -> float:
    if isinstance(value, bool):
        return 0.0
    if isinstance(value, (int, float)):
        return float(value)
    return 0.0


def usage_from_weave_agents_verifier(path_value: Any) -> dict[str, Any]:
    if not isinstance(path_value, str) or not path_value:
        return {}
    try:
        payload = read_json(Path(path_value))
    except OSError:
        return {}

    checks = payload.get("checks")
    if isinstance(checks, list):
        for check in checks:
            if (
                not isinstance(check, dict)
                or check.get("name") != "usage"
                or check.get("ok") is not True
            ):
                continue
            input_tokens = _numeric_value(check.get("agent_input_tokens")) + _numeric_value(
                check.get("trace_input_tokens")
            )
            output_tokens = _numeric_value(check.get("agent_output_tokens")) + _numeric_value(
                check.get("trace_output_tokens")
            )
            if input_tokens + output_tokens > 0:
                return {
                    "inputTokens": int(input_tokens),
                    "outputTokens": int(output_tokens),
                    "cacheReadInputTokens": 0,
                    "usageSource": "weave_agents_trace",
                    "usageApproximate": True,
                }

    health = payload.get("content_capture_health")
    if isinstance(health, dict):
        input_tokens = _numeric_value(health.get("trace_input_tokens"))
        output_tokens = _numeric_value(health.get("trace_output_tokens"))
        if input_tokens + output_tokens <= 0:
            input_tokens = _numeric_value(health.get("conversation_input_tokens"))
            output_tokens = _numeric_value(health.get("conversation_output_tokens"))
        if input_tokens + output_tokens > 0:
            return {
                "inputTokens": int(input_tokens),
                "outputTokens": int(output_tokens),
                "cacheReadInputTokens": 0,
                "usageSource": "weave_agents_trace",
                "usageApproximate": True,
            }

    return {}


def openclaw_usage_with_trace_fallback(openclaw: dict[str, Any]) -> dict[str, Any] | None:
    usage = openclaw.get("openclaw_usage")
    if isinstance(usage, dict) and usage:
        return usage
    return usage_from_weave_agents_verifier(openclaw.get("weave_agents_verifier_json")) or usage


def posthoc_trace_budget_violations(
    openclaw: dict[str, Any],
    usage: dict[str, Any] | None,
) -> list[dict[str, Any]]:
    if not isinstance(usage, dict) or not usage:
        return []
    runtime_budget = openclaw.get("runtime_budget")
    if not isinstance(runtime_budget, dict):
        return []
    limits = runtime_budget.get("limits")
    if not isinstance(limits, dict):
        return []

    checks = [
        ("inputTokens", "max_cumulative_input_tokens", "trace_cumulative_input_tokens_exceeded"),
        ("outputTokens", "max_cumulative_output_tokens", "trace_cumulative_output_tokens_exceeded"),
    ]
    violations: list[dict[str, Any]] = []
    for usage_key, limit_key, violation_type in checks:
        observed = _numeric_value(usage.get(usage_key))
        limit = _numeric_value(limits.get(limit_key))
        if observed > 0 and limit > 0 and observed > limit:
            violations.append(
                {
                    "type": violation_type,
                    "observed": int(observed),
                    "limit": int(limit),
                    "source": usage.get("usageSource") or "openclaw_usage",
                }
            )
    return violations


def collect_results(job_dir: Path, output_dir: Path, *, command: list[str], elapsed: float) -> None:
    job_result = read_json(job_dir / "result.json")
    rows: list[dict[str, Any]] = []
    for trial_dir in sorted(path for path in job_dir.iterdir() if path.is_dir()):
        result = read_json(trial_dir / "result.json")
        if not result:
            continue
        agent_result = result.get("agent_result") if isinstance(result.get("agent_result"), dict) else {}
        metadata = agent_result.get("metadata") if isinstance(agent_result.get("metadata"), dict) else {}
        openclaw = metadata.get("openclaw") if isinstance(metadata.get("openclaw"), dict) else {}
        if not openclaw:
            openclaw = fallback_openclaw_metadata(output_dir, result.get("task_name"))
            if openclaw:
                metadata = dict(metadata)
                metadata["openclaw"] = openclaw
                agent_result = dict(agent_result)
                agent_result["metadata"] = metadata
        usage = openclaw_usage_with_trace_fallback(openclaw)
        posthoc_violations = posthoc_trace_budget_violations(openclaw, usage)
        score = verifier_score(result.get("verifier_result"))
        if posthoc_violations:
            score = 0.0
        disqualified_reason = openclaw.get("openclaw_disqualified_reason")
        if posthoc_violations and not disqualified_reason:
            disqualified_reason = "runtime_budget_exceeded"
        rows.append(
            {
                "task_name": result.get("task_name"),
                "trial_name": result.get("trial_name"),
                "trial_uri": result.get("trial_uri"),
                "score": score,
                "resolved": bool(score and score > 0) and not posthoc_violations,
                "exception": result.get("exception_info"),
                "started_at": result.get("started_at"),
                "finished_at": result.get("finished_at"),
                "agent_result": agent_result,
                "openclaw_result_path": openclaw.get("openclaw_result_path"),
                "openclaw_disqualified_reason": disqualified_reason,
                "openclaw_tool_call_count": openclaw.get("openclaw_tool_call_count"),
                "openclaw_usage": usage,
                "runtime_budget_posthoc_trace_violations": posthoc_violations,
                "weave_agents_ok": openclaw.get("weave_agents_ok"),
                "weave_agents_verifier_json": openclaw.get("weave_agents_verifier_json"),
                "weave_agents_conversation_url": openclaw.get("weave_agents_conversation_url"),
                "weave_agents_conversation_link_html": openclaw.get(
                    "weave_agents_conversation_link_html"
                ),
                "nemoclaw_session_audit_ok": openclaw.get("nemoclaw_session_audit_ok"),
                "patch_apply": metadata.get("patch_apply"),
                "result_path": str(trial_dir / "result.json"),
            }
        )
    scored = [row for row in rows if row["score"] is not None]
    resolved = sum(1 for row in scored if row["resolved"])
    summary = {
        "runner_version": RUNNER_VERSION,
        "job_dir": str(job_dir),
        "pier_job_result": job_result,
        "command": command,
        "elapsed_seconds": elapsed,
        "total_trials": len(rows),
        "scored_trials": len(scored),
        "resolved_trials": resolved,
        "pass_at_1": (resolved / len(scored)) if scored else None,
        "exceptions": sum(1 for row in rows if row["exception"]),
        "non_scoreable_policy_blocks": sum(
            1 for row in rows if is_non_scoreable_policy_failure(read_json(Path(row["result_path"])))
        ),
        "non_scoreable_provider_timeouts": sum(
            1 for row in rows if is_non_scoreable_provider_timeout(read_json(Path(row["result_path"])))
        ),
        "non_scoreable_llm_response_idle_timeouts": sum(
            1
            for row in rows
            if is_non_scoreable_llm_response_idle_timeout(read_json(Path(row["result_path"])))
        ),
        "llm_response_idle_timeouts": sum(
            1
            for row in rows
            if is_llm_response_idle_timeout(read_json(Path(row["result_path"])))
        ),
        "non_scoreable_configuration_errors": sum(
            1 for row in rows if is_non_scoreable_configuration_error(read_json(Path(row["result_path"])))
        ),
        "weave_agents_ok": sum(1 for row in rows if row.get("weave_agents_ok") is True),
        "native_trace_missing": sum(1 for row in rows if row.get("weave_agents_ok") is not True),
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    write_jsonl(output_dir / "results.jsonl", rows)
    (output_dir / "results.json").write_text(
        json.dumps(rows, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def write_dry_run_results(
    output_dir: Path,
    *,
    command: list[str],
    config_path: Path,
    args: argparse.Namespace,
) -> None:
    config = read_json(config_path)
    dataset = (config.get("datasets") or [{}])[0]
    task_names = dataset.get("task_names")
    if task_names is None:
        task_names = []
    rows = [
        {
            "task_name": f"datacurve/{task_name}",
            "trial_name": None,
            "score": None,
            "resolved": False,
            "exception": None,
            "started_at": None,
            "finished_at": None,
            "agent_result": {},
            "openclaw_result_path": None,
            "openclaw_disqualified_reason": None,
            "openclaw_tool_call_count": None,
            "openclaw_usage": None,
            "weave_agents_ok": None,
            "weave_agents_conversation_url": None,
            "weave_agents_conversation_link_html": None,
            "nemoclaw_session_audit_ok": None,
            "patch_apply": None,
            "result_path": None,
            "dry_run": True,
        }
        for task_name in task_names
    ]
    summary = {
        "runner_version": RUNNER_VERSION,
        "job_dir": None,
        "pier_job_result": None,
        "command": command,
        "config_path": str(config_path),
        "elapsed_seconds": 0.0,
        "total_trials": len(rows) or int(args.n_tasks or 0),
        "scored_trials": 0,
        "resolved_trials": 0,
        "pass_at_1": None,
        "exceptions": 0,
        "weave_agents_ok": 0,
        "native_trace_missing": len(rows),
        "dry_run": True,
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    write_json(output_dir / "summary.json", summary)
    write_jsonl(output_dir / "results.jsonl", rows)
    write_json(output_dir / "results.json", rows)


def add_agent_kwarg(command: list[str], key: str, value: Any) -> None:
    if value is None:
        return
    command.extend(["--agent-kwarg", f"{key}={value}"])


def jsonable(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {key: jsonable(item) for key, item in value.items()}
    if isinstance(value, list):
        return [jsonable(item) for item in value]
    if isinstance(value, tuple):
        return [jsonable(item) for item in value]
    return value


def build_agent_kwargs(args: argparse.Namespace) -> dict[str, Any]:
    if (
        args.verify_weave_agents
        and args.use_task_agent
        and args.nemoclaw_sandbox
        and not args.no_local
    ):
        raise ValueError(
            "DeepSWE native Weave Agents verification requires --no-local with "
            "--use-task-agent and --nemoclaw-sandbox. Use --local only for explicit "
            "non-native-trace debugging."
        )
    output_root = args.output_dir / "openclaw"
    kwargs: dict[str, Any] = {
        "repo_root": REPO_ROOT,
        "output_root": output_root,
        "prefix": args.prefix,
        "openclaw_model": args.model,
        "openclaw_model_params_json": getattr(args, "openclaw_model_params_json", None),
        "openclaw_model_overrides_json": getattr(
            args,
            "openclaw_model_overrides_json",
            None,
        ),
        "thinking": args.thinking,
        "agent": args.agent,
        "openclaw_timeout": args.openclaw_timeout,
        "openclaw_max_attempts": args.openclaw_max_attempts,
        "openclaw_retry_base_seconds": args.openclaw_retry_base_seconds,
        "max_input_tokens": args.max_input_tokens,
        "max_cumulative_input_tokens": args.max_cumulative_input_tokens,
        "max_cumulative_output_tokens": args.max_cumulative_output_tokens,
        "max_tool_calls": args.max_tool_calls,
        "max_agent_turns": args.max_agent_turns,
        "max_tool_wall_seconds": args.max_tool_wall_seconds,
        "final_assistant_idle_salvage_seconds": args.final_assistant_idle_salvage_seconds,
        "llm_response_idle_timeout_seconds": args.llm_response_idle_timeout_seconds,
        "require_actual_token_usage": str(args.require_actual_token_usage).lower(),
        "nemoclaw_bin": args.nemoclaw_bin,
        "nemoclaw_sandbox": args.nemoclaw_sandbox,
        "nemoclaw_workdir": args.nemoclaw_workdir,
        "nemoclaw_openclaw_config_path": args.nemoclaw_openclaw_config_path,
        "nemoclaw_checkout_transfer_mode": args.nemoclaw_checkout_transfer_mode,
        "nemoclaw_checkout_transfer_timeout": args.nemoclaw_checkout_transfer_timeout,
        "openclaw_tool_profile": args.openclaw_tool_profile,
        "task_agent_prefix": args.task_agent_prefix,
        "session_prefix": args.session_prefix,
        "no_local": str(args.no_local).lower(),
        "use_task_agent": str(args.use_task_agent).lower(),
        "restart_gateway_before_run": str(args.restart_gateway_before_run).lower(),
        "verify_weave_agents": str(args.verify_weave_agents).lower(),
        "weave_agents_entity": args.weave_agents_entity,
        "weave_agents_project": args.weave_agents_project,
        "weave_agents_agent_name": args.weave_agents_agent_name,
        "weave_agents_limit": args.weave_agents_limit,
        "weave_agents_verification_timeout": args.weave_agents_verification_timeout,
        "weave_agents_poll_seconds": args.weave_agents_poll_seconds,
        "allow_failed_preflight": str(args.allow_failed_preflight).lower(),
    }
    if args.deny_tool:
        kwargs["deny_tool"] = ",".join(args.deny_tool)
    if args.deny_argument_pattern:
        kwargs["deny_argument_pattern"] = "\n".join(args.deny_argument_pattern)
    return jsonable(kwargs)


def build_job_config(args: argparse.Namespace, job_name: str) -> dict[str, Any]:
    task_names = configured_task_names(args)
    dataset: dict[str, Any] = {
        "path": str(args.tasks_root),
        "task_names": task_names,
        "exclude_task_names": None,
        "n_tasks": None if task_names else args.n_tasks,
        "sample_seed": None if task_names else args.sample_seed,
    }
    return {
        "job_name": job_name,
        "jobs_dir": str(args.jobs_dir),
        "n_attempts": 1,
        "timeout_multiplier": 1.0,
        "agent_timeout_multiplier": args.agent_timeout_multiplier,
        "verifier_timeout_multiplier": None,
        "agent_setup_timeout_multiplier": None,
        "environment_build_timeout_multiplier": None,
        "debug": False,
        "n_concurrent_trials": args.n_concurrent,
        "quiet": args.quiet,
        "retry": {"max_retries": 0},
        "environment": {
            "type": "docker",
            "force_build": False,
            "delete": args.delete,
        },
        "verifier": {
            "disable": args.disable_verification,
        },
        "agents": [
            {
                "import_path": AGENT_IMPORT_PATH,
                "model_name": args.model,
                "kwargs": build_agent_kwargs(args),
                "env": {},
            }
        ],
        "datasets": [dataset],
        "tasks": [],
        "artifacts": [],
    }


def build_command(config_path: Path) -> list[str]:
    command = ["pier", "run", "--config", str(config_path), "--yes"]
    return command


def is_environment_setup_failure(result: dict[str, Any]) -> bool:
    if not isinstance(result.get("exception_info"), dict):
        return False
    if result.get("agent_result") is not None:
        return False
    if result.get("agent_execution") is not None:
        return False
    if result.get("environment_setup") is not None:
        return True
    message = ""
    exception = result.get("exception_info")
    if isinstance(exception, dict):
        message = str(exception.get("exception_message") or "")
    lowered = message.lower()
    return any(
        token in lowered
        for token in (
            "docker compose command failed",
            "image pull",
            "toomanyrequests",
            "rate exceeded",
            "no container found for service",
        )
    )


def is_non_scoreable_policy_failure(result: dict[str, Any]) -> bool:
    # Tool-policy blocks are scoreable model interactions. They are retained in
    # result metadata, but they should not abort a paid run or mark the trial as
    # harness-invalid: the agent can observe the tool error and continue.
    return False


def is_non_scoreable_provider_timeout(result: dict[str, Any]) -> bool:
    exception = result.get("exception_info")
    message = ""
    if isinstance(exception, dict):
        message = "\n".join(
            str(exception.get(key) or "")
            for key in ("exception_type", "exception_message", "traceback")
        )
    if "non_scoreable_provider_timeout" in message:
        return True
    agent_result = result.get("agent_result")
    metadata = agent_result.get("metadata") if isinstance(agent_result, dict) else None
    openclaw = metadata.get("openclaw") if isinstance(metadata, dict) else None
    if not isinstance(openclaw, dict):
        return False
    if openclaw.get("deepswe_non_scoreable_reason") == "provider_timeout":
        return True
    runtime_budget = openclaw.get("runtime_budget")
    live = runtime_budget.get("live") if isinstance(runtime_budget, dict) else None
    if not isinstance(live, dict):
        return False
    try:
        provider_timeout_count = int(live.get("live_provider_timeout_count") or 0)
    except (TypeError, ValueError):
        provider_timeout_count = 0
    if provider_timeout_count > 0:
        return True
    if live.get("reason") == "live_provider_timeout":
        return True
    if live.get("interrupt_reason") == "live_provider_timeout":
        return True
    exceeded_limits = live.get("exceeded_limits")
    if isinstance(exceeded_limits, list) and "live_provider_timeout" in exceeded_limits:
        return True
    return False


def is_non_scoreable_llm_response_idle_timeout(result: dict[str, Any]) -> bool:
    return False


def is_llm_response_idle_timeout(result: dict[str, Any]) -> bool:
    exception = result.get("exception_info")
    message = ""
    if isinstance(exception, dict):
        message = "\n".join(
            str(exception.get(key) or "")
            for key in ("exception_type", "exception_message", "traceback")
        )
    if "llm_response_idle_timeout" in message:
        return True
    agent_result = result.get("agent_result")
    metadata = agent_result.get("metadata") if isinstance(agent_result, dict) else None
    openclaw = metadata.get("openclaw") if isinstance(metadata, dict) else None
    if not isinstance(openclaw, dict):
        return False
    if openclaw.get("deepswe_scoreable_failure_reason") == "llm_response_idle_timeout":
        return True
    runtime_budget = openclaw.get("runtime_budget")
    live = runtime_budget.get("live") if isinstance(runtime_budget, dict) else None
    if not isinstance(live, dict):
        return False
    if live.get("reason") == "llm_response_idle_timeout":
        return True
    if live.get("interrupt_reason") == "llm_response_idle_timeout":
        return True
    exceeded_limits = live.get("exceeded_limits")
    return isinstance(exceeded_limits, list) and "llm_response_idle_timeout" in exceeded_limits


def is_non_scoreable_configuration_error(result: dict[str, Any]) -> bool:
    exception = result.get("exception_info")
    message = ""
    if isinstance(exception, dict):
        message = "\n".join(
            str(exception.get(key) or "")
            for key in ("exception_type", "exception_message", "traceback")
        )
    if "NonScoreableOpenClawConfigurationError" in message:
        return True
    if "non_scoreable_configuration_error" in message:
        return True
    agent_result = result.get("agent_result")
    metadata = agent_result.get("metadata") if isinstance(agent_result, dict) else None
    openclaw = metadata.get("openclaw") if isinstance(metadata, dict) else None
    if not isinstance(openclaw, dict):
        return False
    if openclaw.get("deepswe_non_scoreable_reason") == "unsupported_model_thinking":
        return True
    stderr = str(openclaw.get("stderr") or "")
    return "Thinking level" in stderr and "is not supported for" in stderr


def terminate_process_tree(proc: subprocess.Popen[str]) -> None:
    if proc.poll() is not None:
        return
    try:
        os.killpg(proc.pid, signal.SIGTERM)
    except ProcessLookupError:
        return
    except Exception:
        proc.terminate()
    deadline = time.time() + 10
    while proc.poll() is None and time.time() < deadline:
        time.sleep(0.2)
    if proc.poll() is None:
        try:
            os.killpg(proc.pid, signal.SIGKILL)
        except ProcessLookupError:
            return
        except Exception:
            proc.kill()


def start_environment_failure_watchdog(
    *,
    job_dir: Path,
    proc: subprocess.Popen[str],
    stop_event: threading.Event,
    poll_seconds: float = 2.0,
    watch_environment_setup: bool = True,
    watch_non_scoreable_policy: bool = True,
) -> tuple[threading.Thread, dict[str, Any]]:
    state: dict[str, Any] = {"failure": None}
    seen: set[Path] = set()

    def run() -> None:
        while not stop_event.wait(max(0.2, poll_seconds)):
            if proc.poll() is not None:
                return
            if not job_dir.exists():
                continue
            for result_path in sorted(job_dir.glob("*/result.json")):
                if result_path in seen:
                    continue
                try:
                    result = json.loads(result_path.read_text(encoding="utf-8"))
                except (OSError, json.JSONDecodeError):
                    continue
                seen.add(result_path)
                failure_reason = None
                if watch_environment_setup and is_environment_setup_failure(result):
                    failure_reason = "environment_setup_failure"
                elif watch_non_scoreable_policy and is_non_scoreable_policy_failure(result):
                    failure_reason = "non_scoreable_policy_block"
                elif watch_non_scoreable_policy and is_non_scoreable_provider_timeout(result):
                    failure_reason = "non_scoreable_provider_timeout"
                elif watch_non_scoreable_policy and is_non_scoreable_configuration_error(result):
                    failure_reason = "non_scoreable_configuration_error"
                if failure_reason is None:
                    continue
                exception = result.get("exception_info") if isinstance(result.get("exception_info"), dict) else {}
                failure = {
                    "reason": failure_reason,
                    "trial_name": result.get("trial_name"),
                    "task_name": result.get("task_name"),
                    "result_path": str(result_path),
                    "exception_type": exception.get("exception_type"),
                    "exception_message": exception.get("exception_message"),
                }
                state["failure"] = failure
                failfast_path = job_dir / f"failfast_{failure_reason}.json"
                try:
                    failfast_path.write_text(
                        json.dumps(failure, ensure_ascii=False, indent=2) + "\n",
                        encoding="utf-8",
                    )
                except OSError:
                    pass
                print(
                    f"DeepSWE fail-fast: {failure_reason}; "
                    f"aborting remaining trials. task={failure['task_name']} "
                    f"trial={failure['trial_name']}",
                    flush=True,
                )
                terminate_process_tree(proc)
                return

    thread = threading.Thread(target=run, name="deepswe-env-failfast", daemon=True)
    thread.start()
    return thread, state


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tasks-root", type=Path, default=DEFAULT_TASKS_ROOT)
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/deepswe_openclaw"))
    parser.add_argument("--jobs-dir", type=Path, default=DEFAULT_JOBS_DIR)
    parser.add_argument("--job-name")
    parser.add_argument("--task-names-file", type=Path)
    parser.add_argument("--n-tasks", type=int, default=16)
    parser.add_argument("--sample-seed", type=int, default=0)
    parser.add_argument("--n-concurrent", type=int, default=1)
    parser.add_argument("--include-task-name", action="append")
    parser.add_argument("--model", required=True)
    parser.add_argument(
        "--openclaw-model-params-json",
        "--openclaw-extra-body-json",
        dest="openclaw_model_params_json",
        help=(
            "JSON object forwarded to the OpenClaw runner and merged into the "
            "selected model entry's params."
        ),
    )
    parser.add_argument(
        "--openclaw-model-overrides-json",
        dest="openclaw_model_overrides_json",
        help=(
            "JSON object forwarded to the OpenClaw runner and merged into the "
            "selected model entry itself, e.g. {\"maxTokens\":4096}."
        ),
    )
    parser.add_argument("--thinking", default="high")
    parser.add_argument("--agent", default="main")
    parser.add_argument("--prefix", default="deepswe-openclaw")
    parser.add_argument("--openclaw-timeout", type=int, default=3600)
    parser.add_argument("--openclaw-max-attempts", type=int, default=1)
    parser.add_argument("--openclaw-retry-base-seconds", type=float, default=15.0)
    parser.add_argument("--max-input-tokens", type=int, default=1_000_000)
    parser.add_argument("--max-cumulative-input-tokens", type=int, default=1_000_000)
    parser.add_argument("--max-cumulative-output-tokens", type=int, default=500_000)
    parser.add_argument("--max-tool-calls", type=int, default=40)
    parser.add_argument("--max-agent-turns", type=int, default=40)
    parser.add_argument("--max-tool-wall-seconds", type=int, default=300)
    parser.add_argument("--final-assistant-idle-salvage-seconds", type=float, default=60.0)
    parser.add_argument(
        "--llm-response-idle-timeout-seconds",
        type=float,
        default=900.0,
        help=(
            "If the live OpenClaw session is waiting after a tool result for this "
            "many seconds without a new assistant response, interrupt as a "
            "non-scoreable LLM response idle timeout. 0 disables this watchdog."
        ),
    )
    parser.add_argument("--require-actual-token-usage", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--nemoclaw-bin", default="nemoclaw")
    parser.add_argument("--nemoclaw-sandbox", default="nejumi-taiwan")
    parser.add_argument("--nemoclaw-workdir", default="/sandbox")
    parser.add_argument("--nemoclaw-openclaw-config-path", default="/sandbox/.openclaw/openclaw.json")
    parser.add_argument("--nemoclaw-checkout-transfer-mode", choices=["copy", "visible"], default="copy")
    parser.add_argument("--nemoclaw-checkout-transfer-timeout", type=int, default=600)
    parser.add_argument("--openclaw-tool-profile", default="coding")
    parser.add_argument("--task-agent-prefix", default="tw-deepswe")
    parser.add_argument("--session-prefix", default="deepswe")
    local_mode = parser.add_mutually_exclusive_group()
    local_mode.add_argument(
        "--no-local",
        dest="no_local",
        action="store_true",
        default=True,
        help="Run OpenClaw through the NeMoClaw Gateway rather than host-local OpenClaw.",
    )
    local_mode.add_argument(
        "--local",
        dest="no_local",
        action="store_false",
        help="Debug-only: run OpenClaw locally instead of through the NeMoClaw Gateway.",
    )
    parser.add_argument("--use-task-agent", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--restart-gateway-before-run", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--verify-weave-agents", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--weave-agents-entity", default=os.environ.get("WANDB_ENTITY", "llm-leaderboard"))
    parser.add_argument("--weave-agents-project", default=os.environ.get("WANDB_PROJECT", "tc-leaderboard"))
    parser.add_argument("--weave-agents-agent-name", default="nejumi-taiwan-openclaw")
    parser.add_argument("--weave-agents-limit", type=int, default=50)
    parser.add_argument("--weave-agents-verification-timeout", type=float, default=120.0)
    parser.add_argument("--weave-agents-poll-seconds", type=float, default=5.0)
    parser.add_argument(
        "--deny-tool",
        action="append",
        default=[
            "code_execution",
            "web_search",
            "web_fetch",
            "browser",
            "browser_*",
        ],
    )
    parser.add_argument(
        "--deny-argument-pattern",
        action="append",
        default=[
            r"https?://",
            r"\b(?:ftp|sftp|ssh)://",
            r"\bgit\+",
            r"\bgit\s+(?:clone|fetch|pull|ls-remote)\b",
            r"\b(curl|wget)\b",
        ],
    )
    parser.add_argument("--allow-failed-preflight", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--preflight-docker-images", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--prepare-sandbox-runtime", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--preflight-openclaw-sandbox-tools", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--prepull-missing-docker-images", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--docker-pull-retries", type=int, default=3)
    parser.add_argument("--docker-pull-retry-seconds", type=float, default=30.0)
    parser.add_argument("--fail-fast-environment-setup", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--fail-fast-non-scoreable-policy", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--environment-failfast-poll-seconds", type=float, default=2.0)
    parser.add_argument("--agent-timeout-multiplier", type=float)
    parser.add_argument("--disable-verification", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--dry-run", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--preflight-only", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--delete", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--quiet", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--env-file", type=Path, default=REPO_ROOT / ".env")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    load_dotenv(args.env_file)
    args.tasks_root = args.tasks_root.resolve()
    args.jobs_dir = args.jobs_dir.resolve()
    args.output_dir = args.output_dir.resolve()
    if args.task_names_file is not None:
        args.task_names_file = args.task_names_file.resolve()
    job_name = args.job_name or f"deepswe-openclaw-{int(time.time())}"
    config_path = args.output_dir / "pier_job_config.json"
    write_json(config_path, build_job_config(args, job_name))
    command = build_command(config_path)
    if args.dry_run:
        write_dry_run_results(
            args.output_dir,
            command=command,
            config_path=config_path,
            args=args,
        )
        print(f"DeepSWE dry run written to {args.output_dir}")
        return
    docker_preflight = preflight_docker_images(args)
    write_json(args.output_dir / "docker_preflight.json", docker_preflight)
    sandbox_runtime_preflight = prepare_openclaw_sandbox_runtime(args)
    write_json(args.output_dir / "sandbox_runtime_preflight.json", sandbox_runtime_preflight)
    try:
        sandbox_tool_preflight = preflight_openclaw_sandbox_tools(args)
    except DeepSWEPreflightError as exc:
        if exc.payload:
            write_json(args.output_dir / "openclaw_sandbox_tool_preflight.json", exc.payload)
        raise
    write_json(args.output_dir / "openclaw_sandbox_tool_preflight.json", sandbox_tool_preflight)
    if args.preflight_only:
        print(
            "DeepSWE preflight passed for "
            f"{len(docker_preflight.get('images', []))} Docker images and "
            f"{len(sandbox_tool_preflight.get('required_tools', []))} OpenClaw sandbox tools."
        )
        return
    env = os.environ.copy()
    env["PYTHONPATH"] = (
        str(REPO_ROOT / "scripts" / "tools")
        + os.pathsep
        + str(REPO_ROOT)
        + os.pathsep
        + env.get("PYTHONPATH", "")
    )

    print("Running:", " ".join(str(part) for part in command), flush=True)
    started_at = time.time()
    proc = subprocess.Popen(
        command,
        cwd=str(REPO_ROOT),
        env=env,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        bufsize=1,
        start_new_session=True,
    )
    stop_watchdog = threading.Event()
    watchdog_thread = None
    watchdog_state: dict[str, Any] = {"failure": None}
    if bool(getattr(args, "fail_fast_environment_setup", True)) or bool(
        getattr(args, "fail_fast_non_scoreable_policy", True)
    ):
        watchdog_thread, watchdog_state = start_environment_failure_watchdog(
            job_dir=args.jobs_dir / job_name,
            proc=proc,
            stop_event=stop_watchdog,
            poll_seconds=float(getattr(args, "environment_failfast_poll_seconds", 2.0) or 2.0),
            watch_environment_setup=bool(getattr(args, "fail_fast_environment_setup", True)),
            watch_non_scoreable_policy=bool(getattr(args, "fail_fast_non_scoreable_policy", True)),
        )
    stdout_parts: list[str] = []
    assert proc.stdout is not None
    try:
        for line in proc.stdout:
            print(line, end="", flush=True)
            stdout_parts.append(line)
    finally:
        stop_watchdog.set()
        if watchdog_thread is not None:
            watchdog_thread.join(timeout=5)
    returncode = proc.wait()
    elapsed = time.time() - started_at
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "pier_stdout.log").write_text("".join(stdout_parts), encoding="utf-8")
    collect_results(args.jobs_dir / job_name, args.output_dir, command=command, elapsed=elapsed)
    if watchdog_state.get("failure"):
        failure = watchdog_state["failure"]
        reason = failure.get("reason") or "unknown_failure"
        write_json(args.output_dir / f"failfast_{reason}.json", failure)
        if reason == "environment_setup_failure":
            raise RuntimeError(
                "DeepSWE aborted because an environment setup failure occurred before model execution: "
                f"{failure.get('task_name')}"
            )
        if reason == "non_scoreable_policy_block":
            raise RuntimeError(
                "DeepSWE aborted because the benchmark policy blocked the task before a scoreable "
                f"patch could be produced: {failure.get('task_name')}"
            )
        raise RuntimeError(f"DeepSWE aborted by fail-fast watchdog: {reason}")
    if returncode != 0:
        raise SystemExit(returncode)


if __name__ == "__main__":
    main()
