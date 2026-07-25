import json
import subprocess
import sys
from pathlib import Path
from typing import Any

import pandas as pd
import wandb
import weave

from config_singleton import WandbConfigSingleton
from evaluator.evaluate_utils.subprocess_runner import run_streaming_command


REPO_ROOT = Path(__file__).resolve().parents[2]
OPENCLAW_RUNNER = REPO_ROOT / "scripts" / "tools" / "run_agentic_math_openclaw.py"
DEFAULT_NEMOCLAW_OPENCLAW_CONFIG_PATH = "/sandbox/.openclaw/openclaw.json"
DEFAULT_MAX_INPUT_TOKENS = 500_000
DEFAULT_MAX_TOOL_CALLS = 40
DEFAULT_MAX_AGENT_TURNS = 40
DEFAULT_MAX_TOOL_WALL_SECONDS = 120
AGENTIC_MATH_OUTPUT_TABLE_REQUIRED_COLUMNS = (
    "billable_openclaw_usage",
    "billable_openclaw_attempt_count",
    "billable_openclaw_attempts",
    "billable_openclaw_wall_seconds",
    "nemoclaw_session_audit_ok",
    "nemoclaw_session_audit",
    "nemoclaw_session_copy_source",
    "nemoclaw_session_copied_bytes",
    "conversation_order_ok",
    "conversation_order",
    "tool_policy_ok",
    "tool_policy_violations",
    "weave_agents_ok",
    "weave_agents_required",
    "weave_agents_agent_name",
    "weave_agents_conversation_id",
    "weave_agents_conversation_id_contains",
    "weave_agents_conversation_url",
    "weave_agents_trace_id",
    "weave_agents_url",
    "weave_agents_trace_url",
    "weave_agents_verifier_json",
    "weave_agents_error",
    "openclaw_result_path",
    "openclaw_invocation_path",
    "openclaw_invocation_sha256",
    "openclaw_command_sha256",
    "openclaw_config_source",
)


def _cfg_get(cfg_obj: Any, key: str, default: Any = None) -> Any:
    try:
        return cfg_obj.get(key, default)
    except Exception:
        return getattr(cfg_obj, key, default)


def _as_list(value: Any) -> list[Any]:
    if value is None:
        return []
    if isinstance(value, (str, bytes)):
        return [value]
    if isinstance(value, (list, tuple)):
        return list(value)
    if isinstance(value, dict):
        return [value]
    if hasattr(value, "__iter__"):
        return list(value)
    return [value]


def _json_cli_arg(value: Any) -> str:
    try:
        from omegaconf import OmegaConf

        if OmegaConf.is_config(value):
            value = OmegaConf.to_container(value, resolve=True)
    except Exception:
        pass
    return json.dumps(value, ensure_ascii=False, sort_keys=True)


def _run_command(
    command: list[str],
    *,
    timeout: float | None = None,
) -> subprocess.CompletedProcess[str]:
    return run_streaming_command(command, cwd=REPO_ROOT, timeout=timeout)


def _dataset_path(cfg, run) -> Path:
    local_dataset_dir = _cfg_get(cfg.agentic_math, "local_dataset_dir")
    if local_dataset_dir:
        dataset_dir = Path(local_dataset_dir)
    else:
        artifact = run.use_artifact(cfg.agentic_math.artifacts_path, type="dataset")
        artifact_root = Path(artifact.download())
        dataset_dir = artifact_root / cfg.agentic_math.dataset_dir

    subset = _cfg_get(cfg.agentic_math, "subset", "leaderboard")
    if subset in {"full", "leaderboard", "smoke"}:
        jsonl_path = dataset_dir / "subsets" / f"{subset}.jsonl"
    else:
        jsonl_path = dataset_dir / "subsets" / f"{subset}.jsonl"
    if not jsonl_path.exists():
        raise FileNotFoundError(jsonl_path)
    return jsonl_path


def _build_openclaw_command(cfg, jsonl_path: Path, output_dir: Path) -> list[str]:
    use_task_agent = _cfg_get(cfg.agentic_math, "use_task_agent", True)
    nemoclaw_sandbox = _cfg_get(cfg.agentic_math, "nemoclaw_sandbox")
    no_local = bool(_cfg_get(cfg.agentic_math, "no_local", False))
    if no_local and (
        nemoclaw_sandbox is None
        or not str(nemoclaw_sandbox).strip()
        or str(nemoclaw_sandbox).strip().lower() in {"none", "null"}
    ):
        raise ValueError(
            "agentic_math.nemoclaw_sandbox must name an isolated NeMoClaw "
            "sandbox when agentic_math.no_local is true"
        )
    command = [
        sys.executable,
        str(OPENCLAW_RUNNER),
        "--dataset-jsonl",
        str(jsonl_path),
        "--output-dir",
        str(output_dir / "openclaw"),
        "--prefix",
        str(_cfg_get(cfg.agentic_math, "prefix", "openclaw")),
        "--thinking",
        str(_cfg_get(cfg.agentic_math, "thinking", "high")),
        "--agent",
        str(_cfg_get(cfg.agentic_math, "agent", "main")),
        "--openclaw-timeout",
        str(_cfg_get(cfg.agentic_math, "openclaw_timeout", 1200)),
        "--openclaw-max-attempts",
        str(_cfg_get(cfg.agentic_math, "openclaw_max_attempts", 3)),
        "--openclaw-retry-base-seconds",
        str(_cfg_get(cfg.agentic_math, "openclaw_retry_base_seconds", 15)),
        "--provider-recovery-rounds",
        str(_cfg_get(cfg.agentic_math, "provider_recovery_rounds", 2)),
        "--provider-recovery-base-seconds",
        str(_cfg_get(cfg.agentic_math, "provider_recovery_base_seconds", 60)),
        "--num-workers",
        str(_cfg_get(cfg.agentic_math, "num_workers", 1)),
        "--task-start-min-interval-seconds",
        str(_cfg_get(cfg.agentic_math, "task_start_min_interval_seconds", 0.0)),
        "--max-input-tokens",
        str(_cfg_get(cfg.agentic_math, "max_input_tokens", DEFAULT_MAX_INPUT_TOKENS)),
        "--max-cumulative-input-tokens",
        str(
            _cfg_get(
                cfg.agentic_math,
                "max_cumulative_input_tokens",
                _cfg_get(cfg.agentic_math, "max_input_tokens", DEFAULT_MAX_INPUT_TOKENS),
            )
        ),
        "--max-cumulative-output-tokens",
        str(_cfg_get(cfg.agentic_math, "max_cumulative_output_tokens", 0)),
        "--max-tool-calls",
        str(_cfg_get(cfg.agentic_math, "max_tool_calls", DEFAULT_MAX_TOOL_CALLS)),
        "--max-agent-turns",
        str(_cfg_get(cfg.agentic_math, "max_agent_turns", DEFAULT_MAX_AGENT_TURNS)),
        "--max-tool-wall-seconds",
        str(_cfg_get(cfg.agentic_math, "max_tool_wall_seconds", DEFAULT_MAX_TOOL_WALL_SECONDS)),
    ]
    if bool(_cfg_get(cfg.agentic_math, "require_actual_token_usage", False)):
        command.append("--require-actual-token-usage")
    profile = _cfg_get(cfg.agentic_math, "profile")
    if profile:
        command.extend(["--profile", str(profile)])
    openclaw_config_template = _cfg_get(cfg.agentic_math, "openclaw_config_template")
    if openclaw_config_template:
        command.extend(["--openclaw-config-template", str(openclaw_config_template)])
    openclaw_tool_profile = _cfg_get(cfg.agentic_math, "openclaw_tool_profile")
    if openclaw_tool_profile:
        command.extend(["--openclaw-tool-profile", str(openclaw_tool_profile)])
    if not use_task_agent:
        command.append("--no-use-task-agent")
    task_agent_prefix = _cfg_get(cfg.agentic_math, "task_agent_prefix")
    if task_agent_prefix:
        command.extend(["--task-agent-prefix", str(task_agent_prefix)])
    session_prefix = _cfg_get(cfg.agentic_math, "session_prefix")
    if session_prefix:
        command.extend(["--session-prefix", str(session_prefix)])
    model = _cfg_get(cfg.agentic_math, "openclaw_model") or _cfg_get(
        cfg.model, "pretrained_model_name_or_path", ""
    )
    if model:
        command.extend(["--model", str(model)])
    openclaw_model_params = _cfg_get(cfg.agentic_math, "openclaw_model_params")
    if openclaw_model_params is not None:
        command.extend(["--openclaw-model-params-json", _json_cli_arg(openclaw_model_params)])
    openclaw_model_overrides = _cfg_get(cfg.agentic_math, "openclaw_model_overrides")
    if openclaw_model_overrides is not None:
        command.extend(
            ["--openclaw-model-overrides-json", _json_cli_arg(openclaw_model_overrides)]
        )
    if nemoclaw_sandbox:
        command.extend(["--nemoclaw-sandbox", str(nemoclaw_sandbox)])
        command.extend(["--nemoclaw-bin", str(_cfg_get(cfg.agentic_math, "nemoclaw_bin", "nemoclaw"))])
        command.extend(
            ["--nemoclaw-workdir", str(_cfg_get(cfg.agentic_math, "nemoclaw_workdir", "/sandbox"))]
        )
        nemoclaw_openclaw_config_path = _cfg_get(
            cfg.agentic_math,
            "nemoclaw_openclaw_config_path",
            DEFAULT_NEMOCLAW_OPENCLAW_CONFIG_PATH,
        )
        command.extend(["--nemoclaw-openclaw-config-path", str(nemoclaw_openclaw_config_path)])
    if _cfg_get(cfg.agentic_math, "allow_failed_preflight", False):
        command.append("--allow-failed-preflight")
    if no_local:
        command.append("--no-local")
    if _cfg_get(cfg.agentic_math, "weave_sidecar", False) or _cfg_get(
        cfg.agentic_math, "weave_sidecar_strict", False
    ):
        raise ValueError(
            "agentic_math.weave_sidecar is disabled. Use native weave-openclaw "
            "Agents traces only; manual sidecar traces are not valid evidence."
        )
    command.append("--no-weave-sidecar")
    if _cfg_get(cfg.agentic_math, "verify_weave_agents", False):
        command.append("--verify-weave-agents")
    for cfg_key, cli_key in (
        ("weave_agents_entity", "--weave-agents-entity"),
        ("weave_agents_project", "--weave-agents-project"),
        ("weave_agents_agent_name", "--weave-agents-agent-name"),
        ("weave_agents_env_file", "--weave-agents-env-file"),
        ("weave_agents_limit", "--weave-agents-limit"),
        ("weave_agents_verification_timeout", "--weave-agents-verification-timeout"),
        ("weave_agents_poll_seconds", "--weave-agents-poll-seconds"),
    ):
        value = _cfg_get(cfg.agentic_math, cfg_key)
        if value is not None:
            command.extend([cli_key, str(value)])
    for denied_tool in _as_list(_cfg_get(cfg.agentic_math, "deny_tool")):
        command.extend(["--deny-tool", str(denied_tool)])
    for pattern in _as_list(_cfg_get(cfg.agentic_math, "deny_argument_pattern")):
        command.extend(["--deny-argument-pattern", str(pattern)])
    if _cfg_get(cfg.agentic_math, "redo", False):
        command.append("--redo")
    if _cfg_get(cfg, "testmode", False):
        command.extend(["--limit", str(_cfg_get(cfg.agentic_math, "testmode_max_samples", 1))])
        command.append("--dry-run")
    elif _cfg_get(cfg.agentic_math, "dry_run", False):
        command.append("--dry-run")
    else:
        limit = _cfg_get(cfg.agentic_math, "limit")
        if limit is not None:
            command.extend(["--limit", str(limit)])
    return command


def _run_openclaw(cfg, jsonl_path: Path, output_dir: Path) -> Path:
    command = _build_openclaw_command(cfg, jsonl_path, output_dir)
    _run_command(
        command,
        timeout=float(
            _cfg_get(cfg.agentic_math, "benchmark_timeout_seconds", 21_600)
        ),
    )
    return output_dir / "openclaw"


def preflight(cfg, run, output_dir: Path) -> dict[str, Any]:
    """Resolve the real dataset and validate the Math runtime without model calls."""
    jsonl_path = _dataset_path(cfg, run)
    selected_rows = [
        json.loads(line)
        for line in jsonl_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    limit = _cfg_get(cfg.agentic_math, "limit")
    if limit is not None:
        selected_rows = selected_rows[: int(limit)]
    selected_task_ids = [str(row.get("task_id") or "") for row in selected_rows]

    if not _cfg_get(cfg.agentic_math, "run_openclaw", True):
        results_dir = _cfg_get(cfg.agentic_math, "results_dir")
        if not results_dir:
            raise ValueError(
                "agentic_math.results_dir is required when run_openclaw is false"
            )
        reused_dir = Path(str(results_dir)).resolve()
        summary, output_df = _load_results(reused_dir)
        _validate_output_table_columns(output_df)
        actual_task_ids = [str(value) for value in output_df["task_id"].tolist()]
        errors = []
        if len(actual_task_ids) != len(set(actual_task_ids)):
            errors.append("reused results contain duplicate task_id values")
        if set(actual_task_ids) != set(selected_task_ids):
            missing = sorted(set(selected_task_ids) - set(actual_task_ids))
            extra = sorted(set(actual_task_ids) - set(selected_task_ids))
            errors.append(
                f"reused task coverage mismatch (missing={missing}, extra={extra})"
            )
        if int(summary.get("total_instances") or -1) != len(selected_task_ids):
            errors.append(
                "reused summary total_instances does not match selected dataset "
                f"({summary.get('total_instances')} != {len(selected_task_ids)})"
            )
        report = {
            "schema_version": 1,
            "benchmark": "agentic_math",
            "ok": not errors,
            "mode": "reuse",
            "dataset_jsonl": str(jsonl_path.resolve()),
            "results_dir": str(reused_dir),
            "selected_task_count": len(selected_task_ids),
            "result_task_count": len(actual_task_ids),
            "errors": errors,
            "will_run_model": False,
            "will_run_gateway": False,
            "will_run_weave_verification": False,
        }
        report_path = output_dir / "preflight.json"
        report_path.parent.mkdir(parents=True, exist_ok=True)
        report_path.write_text(
            json.dumps(report, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        return {
            "benchmark": "agentic_math",
            "ok": not errors,
            "returncode": 0 if not errors else 2,
            "command": None,
            "report_path": str(report_path),
            "report": report,
            "error": None if not errors else "; ".join(errors),
            "will_run_model": False,
            "will_run_gateway": False,
            "will_run_grading": False,
            "will_initialize_wandb": False,
        }

    command = _build_openclaw_command(cfg, jsonl_path, output_dir)
    command.append("--preflight-only")
    try:
        result = subprocess.run(
            command,
            cwd=REPO_ROOT,
            text=True,
            capture_output=True,
            check=False,
            timeout=float(
                _cfg_get(cfg.agentic_math, "preflight_timeout_seconds", 180)
            ),
        )
    except subprocess.TimeoutExpired as exc:
        return {
            "benchmark": "agentic_math",
            "ok": False,
            "returncode": 124,
            "command": command,
            "report_path": None,
            "report": None,
            "error": f"Agentic Math preflight timed out after {exc.timeout}s",
            "will_run_model": False,
            "will_run_gateway": False,
            "will_run_grading": False,
            "will_initialize_wandb": False,
        }
    report_path = output_dir / "openclaw" / "preflight.json"
    report = None
    if report_path.exists():
        try:
            report = json.loads(report_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            report = None
    ok = result.returncode == 0 and isinstance(report, dict) and report.get("ok") is True
    error_text = (result.stderr or result.stdout).strip()
    return {
        "benchmark": "agentic_math",
        "ok": ok,
        "returncode": result.returncode,
        "command": command,
        "report_path": str(report_path),
        "report": report,
        "error": None if ok else error_text[-4000:],
        "will_run_model": False,
        "will_run_gateway": False,
        "will_run_grading": False,
        "will_initialize_wandb": False,
    }


def _load_results(output_dir: Path) -> tuple[dict[str, Any], pd.DataFrame]:
    summary_path = output_dir / "summary.json"
    results_path = output_dir / "results.jsonl"
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    rows = []
    with results_path.open(encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return summary, pd.DataFrame(rows)


def _nemoclaw_audit_metrics(summary: dict[str, Any]) -> dict[str, int]:
    return {
        "agentic_math/nemoclaw_session_audit_required_instances": int(
            summary.get("nemoclaw_session_audit_required_instances") or 0
        ),
        "agentic_math/nemoclaw_session_audit_passed_instances": int(
            summary.get("nemoclaw_session_audit_passed_instances") or 0
        ),
        "agentic_math/nemoclaw_session_audit_failed_instances": int(
            summary.get("nemoclaw_session_audit_failed_instances") or 0
        ),
    }


def _validate_output_table_columns(output_df: pd.DataFrame) -> None:
    missing = [
        column
        for column in AGENTIC_MATH_OUTPUT_TABLE_REQUIRED_COLUMNS
        if column not in output_df.columns
    ]
    if missing:
        raise ValueError(
            "agentic_math_output_table is missing required observability columns: "
            f"{missing}"
        )


def _json_cell_for_wandb_table(value: Any) -> Any:
    if isinstance(value, (dict, list, tuple)):
        return json.dumps(value, ensure_ascii=False, sort_keys=True, default=str)
    return value


def _prepare_output_table_for_wandb(output_df: pd.DataFrame) -> pd.DataFrame:
    table_df = output_df.copy()
    for column in table_df.columns:
        if table_df[column].map(lambda value: isinstance(value, (dict, list, tuple))).any():
            table_df[column] = table_df[column].map(_json_cell_for_wandb_table)
    return table_df


def _sanitize_artifact_component(value: str) -> str:
    return (
        value.replace("/", "-")
        .replace(":", "-")
        .replace(" ", "-")
        .replace("_", "-")
        .lower()
    )


def _make_result_artifact(
    *,
    artifact_name: str,
    result_dir: Path,
    summary: dict[str, Any],
    model_name: str,
    subset: str,
    max_input_tokens: int,
    max_tool_calls: int,
    max_agent_turns: int,
    max_tool_wall_seconds: int,
    nemoclaw_sandbox: str,
) -> wandb.Artifact:
    artifact = wandb.Artifact(
        artifact_name,
        type="evaluation-results",
        metadata={
            "model_name": model_name,
            "benchmark": "OlymMATH-HARD zh-TW",
            "total_instances": summary["total_instances"],
            "correct_instances": summary["correct_instances"],
            "accuracy": summary["accuracy"],
            "source_result_dir": str(result_dir),
            "subset": subset,
            "max_input_tokens": max_input_tokens,
            "max_tool_calls": max_tool_calls,
            "max_agent_turns": max_agent_turns,
            "max_tool_wall_seconds": max_tool_wall_seconds,
            "nemoclaw_sandbox": nemoclaw_sandbox,
        },
    )
    for filename in ("summary.json", "results.jsonl"):
        path = result_dir / filename
        if path.exists():
            artifact.add_file(str(path), name=filename)
    return artifact


def _log_summary(
    run,
    cfg,
    summary: dict[str, Any],
    output_df: pd.DataFrame,
    result_dir: Path,
) -> None:
    _validate_output_table_columns(output_df)
    output_table_df = _prepare_output_table_for_wandb(output_df)
    model_name = _cfg_get(cfg.model, "pretrained_model_name_or_path", "openclaw")
    subset = str(_cfg_get(cfg.agentic_math, "subset", "leaderboard"))
    max_input_tokens = int(
        _cfg_get(cfg.agentic_math, "max_input_tokens", DEFAULT_MAX_INPUT_TOKENS) or 0
    )
    max_tool_calls = int(
        _cfg_get(cfg.agentic_math, "max_tool_calls", DEFAULT_MAX_TOOL_CALLS) or 0
    )
    max_agent_turns = int(
        _cfg_get(cfg.agentic_math, "max_agent_turns", DEFAULT_MAX_AGENT_TURNS) or 0
    )
    max_tool_wall_seconds = int(
        _cfg_get(cfg.agentic_math, "max_tool_wall_seconds", DEFAULT_MAX_TOOL_WALL_SECONDS) or 0
    )
    nemoclaw_sandbox = str(_cfg_get(cfg.agentic_math, "nemoclaw_sandbox", "") or "")
    leaderboard = pd.DataFrame(
        [
            {
                "model_name": model_name,
                "total_samples": summary["total_instances"],
                "answered_samples": summary["answered_instances"],
                "correct_count": summary["correct_instances"],
                "accuracy": summary["accuracy"],
                "correctness": summary["correctness"],
            }
        ]
    )
    run.log(
        {
            "agentic_math_leaderboard_table": wandb.Table(dataframe=leaderboard),
            "agentic_math_output_table": wandb.Table(dataframe=output_table_df),
            "agentic_math_results": summary,
            "agentic_math/accuracy": float(summary["accuracy"]),
            "agentic_math/correct_instances": int(summary["correct_instances"]),
            "agentic_math/total_instances": int(summary["total_instances"]),
            "agentic_math/answered_instances": int(summary["answered_instances"]),
            "agentic_math/billable_attempt_count": int(
                summary.get("billable_openclaw_attempt_count") or 0
            ),
            "agentic_math/billable_retry_count": int(
                summary.get("billable_openclaw_retry_count") or 0
            ),
            "agentic_math/billable_wall_seconds": float(
                summary.get("billable_openclaw_wall_seconds") or 0.0
            ),
            "agentic_math/billable_input_tokens": float(
                (summary.get("billable_openclaw_usage") or {}).get("inputTokens") or 0.0
            ),
            "agentic_math/billable_output_tokens": float(
                (summary.get("billable_openclaw_usage") or {}).get("outputTokens") or 0.0
            ),
            "agentic_math/billable_cost_usd": float(
                (summary.get("billable_openclaw_usage") or {}).get("costUsd") or 0.0
            ),
            "agentic_math/max_input_tokens": max_input_tokens,
            "agentic_math/max_tool_calls": max_tool_calls,
            "agentic_math/max_agent_turns": max_agent_turns,
            "agentic_math/max_tool_wall_seconds": max_tool_wall_seconds,
            "agentic_math/runtime_budget_exceeded_instances": int(
                summary.get("runtime_budget_exceeded_instances") or 0
            ),
            "agentic_math/weave_agents_required_instances": int(
                summary.get("weave_agents_required_instances") or 0
            ),
            "agentic_math/weave_agents_passed_instances": int(
                summary.get("weave_agents_passed_instances") or 0
            ),
            "agentic_math/weave_agents_failed_instances": int(
                summary.get("weave_agents_failed_instances") or 0
            ),
            **_nemoclaw_audit_metrics(summary),
        }
    )
    run.log_artifact(
        _make_result_artifact(
            artifact_name=(
                "agentic-math-olymmath-hard-zh-tw-"
                + _sanitize_artifact_component(model_name)
                + "-results"
            ),
            result_dir=result_dir,
            summary=summary,
            model_name=model_name,
            subset=subset,
            max_input_tokens=max_input_tokens,
            max_tool_calls=max_tool_calls,
            max_agent_turns=max_agent_turns,
            max_tool_wall_seconds=max_tool_wall_seconds,
            nemoclaw_sandbox=nemoclaw_sandbox,
        ),
        aliases=["latest", "production"],
    )


@weave.op(call_display_name=lambda _: "[Agentic Math] " + WandbConfigSingleton.get_instance().config.wandb.run_name)
def evaluate():
    instance = WandbConfigSingleton.get_instance()
    run = instance.run
    cfg = instance.config

    jsonl_path = _dataset_path(cfg, run)
    output_dir = Path(_cfg_get(cfg.agentic_math, "output_dir", "outputs/agentic_math"))
    output_dir.mkdir(parents=True, exist_ok=True)

    if _cfg_get(cfg.agentic_math, "run_openclaw", True):
        runner_output_dir = _run_openclaw(cfg, jsonl_path, output_dir)
    else:
        results_dir = _cfg_get(cfg.agentic_math, "results_dir")
        if not results_dir:
            raise ValueError("agentic_math.results_dir is required when run_openclaw is false")
        runner_output_dir = Path(results_dir)

    summary, output_df = _load_results(runner_output_dir)
    if _cfg_get(cfg, "testmode", False) or _cfg_get(cfg.agentic_math, "dry_run", False):
        run.log(
            {
                "agentic_math_dry_run_table": wandb.Table(
                    dataframe=pd.DataFrame(
                        [
                            {
                                "dataset_jsonl": str(jsonl_path),
                                "results_dir": str(runner_output_dir),
                                "evaluated": False,
                            }
                        ]
                    )
                )
            }
        )
        return None

    _log_summary(run, cfg, summary, output_df, runner_output_dir)
    return summary
