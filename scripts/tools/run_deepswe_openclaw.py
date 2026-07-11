"""Run DeepSWE tasks with the Nejumi OpenClaw Pier adapter."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[2]
AGENT_IMPORT_PATH = "deepswe_openclaw_pier_agent:NejumiDeepSWEOpenClawAgent"
DEFAULT_TASKS_ROOT = REPO_ROOT / "external" / "deep-swe" / "tasks"
DEFAULT_JOBS_DIR = REPO_ROOT / "outputs" / "deepswe_openclaw_pier_jobs"
RUNNER_VERSION = "run-deepswe-openclaw-2026-07-11-v1"


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


def verifier_score(verifier_result: Any) -> float | None:
    if not isinstance(verifier_result, dict):
        return None
    for key in ("reward", "score", "mean", "pass", "passed"):
        value = verifier_result.get(key)
        if isinstance(value, bool):
            return 1.0 if value else 0.0
        if isinstance(value, (int, float)):
            return float(value)
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
        score = verifier_score(result.get("verifier_result"))
        rows.append(
            {
                "task_name": result.get("task_name"),
                "trial_name": result.get("trial_name"),
                "trial_uri": result.get("trial_uri"),
                "score": score,
                "resolved": bool(score and score > 0),
                "exception": result.get("exception_info"),
                "started_at": result.get("started_at"),
                "finished_at": result.get("finished_at"),
                "agent_result": agent_result,
                "openclaw_result_path": openclaw.get("openclaw_result_path"),
                "openclaw_disqualified_reason": openclaw.get("openclaw_disqualified_reason"),
                "openclaw_tool_call_count": openclaw.get("openclaw_tool_call_count"),
                "openclaw_usage": openclaw.get("openclaw_usage"),
                "weave_agents_ok": openclaw.get("weave_agents_ok"),
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
    task_names = read_task_names(args.task_names_file)
    if args.include_task_name:
        task_names = list(task_names or []) + list(args.include_task_name)
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
            "process",
            "process_*",
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
            r"\b(curl|wget)\b",
            r"\b(?:python(?:3)?\s+-m\s+)?pip(?:3)?\s+install\b(?![^\n;&|]*(?:\s(?:-e|--editable)\s+['\"]?(?:\.|/sandbox/checkouts/|file:)|\s['\"]?(?:\.|/sandbox/checkouts/|file:)))",
            r"\b(requests|urllib|httpx)\.",
        ],
    )
    parser.add_argument("--allow-failed-preflight", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--agent-timeout-multiplier", type=float)
    parser.add_argument("--disable-verification", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--dry-run", action=argparse.BooleanOptionalAction, default=False)
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
    )
    stdout_parts: list[str] = []
    assert proc.stdout is not None
    for line in proc.stdout:
        print(line, end="", flush=True)
        stdout_parts.append(line)
    returncode = proc.wait()
    elapsed = time.time() - started_at
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "pier_stdout.log").write_text("".join(stdout_parts), encoding="utf-8")
    collect_results(args.jobs_dir / job_name, args.output_dir, command=command, elapsed=elapsed)
    if returncode != 0:
        raise SystemExit(returncode)


if __name__ == "__main__":
    main()
