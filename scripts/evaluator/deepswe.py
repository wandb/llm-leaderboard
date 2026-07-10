import json
import subprocess
import sys
from pathlib import Path
from typing import Any

import pandas as pd
import wandb
import weave

from config_singleton import WandbConfigSingleton


REPO_ROOT = Path(__file__).resolve().parents[2]
RUNNER = REPO_ROOT / "scripts" / "tools" / "run_deepswe_openclaw.py"
DEFAULT_TASKS_ROOT = REPO_ROOT / "external" / "deep-swe" / "tasks"
DEFAULT_DATASET_DIR = REPO_ROOT / "data" / "taiwan" / "deepswe"
DEFAULT_MAX_INPUT_TOKENS = 1_000_000
DEFAULT_MAX_TOOL_CALLS = 40
DEFAULT_MAX_AGENT_TURNS = 40
DEFAULT_MAX_TOOL_WALL_SECONDS = 300
DEEPSWE_OUTPUT_TABLE_REQUIRED_COLUMNS = (
    "task_name",
    "score",
    "resolved",
    "exception",
    "openclaw_result_path",
    "openclaw_disqualified_reason",
    "openclaw_tool_call_count",
    "openclaw_usage",
    "weave_agents_ok",
    "weave_agents_conversation_url",
    "weave_agents_conversation_link_html",
    "nemoclaw_session_audit_ok",
    "patch_apply",
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
    return [value]


def _run_command(command: list[str]) -> subprocess.CompletedProcess[str]:
    print("Running:", " ".join(command))
    proc = subprocess.Popen(
        command,
        cwd=str(REPO_ROOT),
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        bufsize=1,
        start_new_session=True,
    )
    stdout_parts: list[str] = []
    assert proc.stdout is not None
    for line in proc.stdout:
        print(line, end="", flush=True)
        stdout_parts.append(line)
    returncode = proc.wait()
    stdout = "".join(stdout_parts)
    if returncode != 0:
        raise RuntimeError(
            f"Command failed with return code {returncode}: {command}\n{stdout[-4000:]}"
        )
    return subprocess.CompletedProcess(command, returncode, stdout=stdout, stderr="")


def _dataset_dir(cfg, run) -> Path:
    local_dataset_dir = _cfg_get(cfg.deepswe, "local_dataset_dir")
    if local_dataset_dir:
        return Path(local_dataset_dir)
    artifact_path = _cfg_get(cfg.deepswe, "artifacts_path")
    if artifact_path:
        artifact = run.use_artifact(artifact_path, type="dataset")
        artifact_root = Path(artifact.download())
        return artifact_root / _cfg_get(cfg.deepswe, "dataset_dir", "deepswe")
    return DEFAULT_DATASET_DIR


def _task_names_path(cfg, run) -> Path:
    task_names_file = _cfg_get(cfg.deepswe, "task_names_file")
    if task_names_file:
        path = Path(task_names_file)
    else:
        subset = str(_cfg_get(cfg.deepswe, "subset", "pilot_16"))
        path = _dataset_dir(cfg, run) / "subsets" / f"{subset}_task_names.json"
    if not path.exists():
        raise FileNotFoundError(
            f"{path} does not exist. Run scripts/data_uploader/prepare_deepswe.py first."
        )
    return path


def _run_openclaw(cfg, task_names_path: Path, output_dir: Path) -> Path:
    command = [
        sys.executable,
        str(RUNNER),
        "--tasks-root",
        str(_cfg_get(cfg.deepswe, "tasks_root", DEFAULT_TASKS_ROOT)),
        "--task-names-file",
        str(task_names_path),
        "--output-dir",
        str(output_dir / "runner"),
        "--jobs-dir",
        str(output_dir / "pier_jobs"),
        "--job-name",
        str(_cfg_get(cfg.deepswe, "job_name", "deepswe-openclaw")),
        "--model",
        str(
            _cfg_get(cfg.deepswe, "openclaw_model")
            or _cfg_get(cfg.model, "pretrained_model_name_or_path", "")
        ),
        "--thinking",
        str(_cfg_get(cfg.deepswe, "thinking", "high")),
        "--agent",
        str(_cfg_get(cfg.deepswe, "agent", "main")),
        "--prefix",
        str(_cfg_get(cfg.deepswe, "prefix", "deepswe-openclaw")),
        "--n-concurrent",
        str(_cfg_get(cfg.deepswe, "n_concurrent", 1)),
        "--openclaw-timeout",
        str(_cfg_get(cfg.deepswe, "openclaw_timeout", 3600)),
        "--openclaw-max-attempts",
        str(_cfg_get(cfg.deepswe, "openclaw_max_attempts", 1)),
        "--openclaw-retry-base-seconds",
        str(_cfg_get(cfg.deepswe, "openclaw_retry_base_seconds", 15)),
        "--max-input-tokens",
        str(_cfg_get(cfg.deepswe, "max_input_tokens", DEFAULT_MAX_INPUT_TOKENS)),
        "--max-cumulative-input-tokens",
        str(
            _cfg_get(
                cfg.deepswe,
                "max_cumulative_input_tokens",
                _cfg_get(cfg.deepswe, "max_input_tokens", DEFAULT_MAX_INPUT_TOKENS),
            )
        ),
        "--max-cumulative-output-tokens",
        str(_cfg_get(cfg.deepswe, "max_cumulative_output_tokens", 500_000)),
        "--max-tool-calls",
        str(_cfg_get(cfg.deepswe, "max_tool_calls", DEFAULT_MAX_TOOL_CALLS)),
        "--max-agent-turns",
        str(_cfg_get(cfg.deepswe, "max_agent_turns", DEFAULT_MAX_AGENT_TURNS)),
        "--max-tool-wall-seconds",
        str(_cfg_get(cfg.deepswe, "max_tool_wall_seconds", DEFAULT_MAX_TOOL_WALL_SECONDS)),
        "--nemoclaw-bin",
        str(_cfg_get(cfg.deepswe, "nemoclaw_bin", "nemoclaw")),
        "--nemoclaw-sandbox",
        str(_cfg_get(cfg.deepswe, "nemoclaw_sandbox", "nejumi-taiwan")),
        "--nemoclaw-workdir",
        str(_cfg_get(cfg.deepswe, "nemoclaw_workdir", "/sandbox")),
        "--nemoclaw-openclaw-config-path",
        str(_cfg_get(cfg.deepswe, "nemoclaw_openclaw_config_path", "/sandbox/.openclaw/openclaw.json")),
        "--nemoclaw-checkout-transfer-mode",
        str(_cfg_get(cfg.deepswe, "nemoclaw_checkout_transfer_mode", "copy")),
        "--nemoclaw-checkout-transfer-timeout",
        str(_cfg_get(cfg.deepswe, "nemoclaw_checkout_transfer_timeout", 600)),
        "--openclaw-tool-profile",
        str(_cfg_get(cfg.deepswe, "openclaw_tool_profile", "coding")),
        "--task-agent-prefix",
        str(_cfg_get(cfg.deepswe, "task_agent_prefix", "tw-deepswe")),
        "--session-prefix",
        str(_cfg_get(cfg.deepswe, "session_prefix", "deepswe")),
        "--weave-agents-entity",
        str(_cfg_get(cfg.deepswe, "weave_agents_entity", "llm-leaderboard")),
        "--weave-agents-project",
        str(_cfg_get(cfg.deepswe, "weave_agents_project", "tc-leaderboard")),
        "--weave-agents-agent-name",
        str(_cfg_get(cfg.deepswe, "weave_agents_agent_name", "nejumi-taiwan-openclaw")),
        "--weave-agents-limit",
        str(_cfg_get(cfg.deepswe, "weave_agents_limit", 50)),
        "--weave-agents-verification-timeout",
        str(_cfg_get(cfg.deepswe, "weave_agents_verification_timeout", 120)),
        "--weave-agents-poll-seconds",
        str(_cfg_get(cfg.deepswe, "weave_agents_poll_seconds", 5)),
    ]
    for boolean_cfg, positive, negative in (
        ("require_actual_token_usage", "--require-actual-token-usage", "--no-require-actual-token-usage"),
        ("no_local", "--no-local", "--no-no-local"),
        ("use_task_agent", "--use-task-agent", "--no-use-task-agent"),
        ("restart_gateway_before_run", "--restart-gateway-before-run", "--no-restart-gateway-before-run"),
        ("verify_weave_agents", "--verify-weave-agents", "--no-verify-weave-agents"),
        ("delete", "--delete", "--no-delete"),
        ("disable_verification", "--disable-verification", "--no-disable-verification"),
    ):
        command.append(positive if _cfg_get(cfg.deepswe, boolean_cfg, True) else negative)
    if _cfg_get(cfg.deepswe, "dry_run", False) or _cfg_get(cfg, "testmode", False):
        command.append("--dry-run")
    if _cfg_get(cfg.deepswe, "allow_failed_preflight", False):
        command.append("--allow-failed-preflight")
    if _cfg_get(cfg.deepswe, "quiet", False):
        command.append("--quiet")
    agent_timeout_multiplier = _cfg_get(cfg.deepswe, "agent_timeout_multiplier")
    if agent_timeout_multiplier is not None:
        command.extend(["--agent-timeout-multiplier", str(agent_timeout_multiplier)])
    for denied_tool in _as_list(_cfg_get(cfg.deepswe, "deny_tool")):
        command.extend(["--deny-tool", str(denied_tool)])
    for pattern in _as_list(_cfg_get(cfg.deepswe, "deny_argument_pattern")):
        command.extend(["--deny-argument-pattern", str(pattern)])

    _run_command(command)
    return output_dir / "runner"


def _load_results(output_dir: Path) -> tuple[dict[str, Any], pd.DataFrame]:
    summary = json.loads((output_dir / "summary.json").read_text(encoding="utf-8"))
    rows = []
    with (output_dir / "results.jsonl").open(encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return summary, pd.DataFrame(rows)


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


def _validate_output_table_columns(output_df: pd.DataFrame) -> None:
    missing = [
        column for column in DEEPSWE_OUTPUT_TABLE_REQUIRED_COLUMNS if column not in output_df.columns
    ]
    if missing:
        raise ValueError(f"deepswe_output_table is missing required columns: {missing}")


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
) -> wandb.Artifact:
    artifact = wandb.Artifact(
        artifact_name,
        type="evaluation-results",
        metadata={
            "model_name": model_name,
            "benchmark": "DeepSWE",
            "total_trials": summary.get("total_trials"),
            "resolved_trials": summary.get("resolved_trials"),
            "pass_at_1": summary.get("pass_at_1"),
            "source_result_dir": str(result_dir),
            "subset": subset,
        },
    )
    for filename in ("summary.json", "results.jsonl", "results.json", "pier_job_config.json"):
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
    subset = str(_cfg_get(cfg.deepswe, "subset", "pilot_16"))
    leaderboard = pd.DataFrame(
        [
            {
                "model_name": model_name,
                "total_samples": int(summary.get("total_trials") or 0),
                "scored_samples": int(summary.get("scored_trials") or 0),
                "resolved_count": int(summary.get("resolved_trials") or 0),
                "pass_at_1": summary.get("pass_at_1"),
                "exceptions": int(summary.get("exceptions") or 0),
            }
        ]
    )
    run.log(
        {
            "deepswe_leaderboard_table": wandb.Table(dataframe=leaderboard),
            "deepswe_output_table": wandb.Table(dataframe=output_table_df),
            "deepswe_results": summary,
            "deepswe/pass_at_1": float(summary.get("pass_at_1") or 0.0),
            "deepswe/resolved_trials": int(summary.get("resolved_trials") or 0),
            "deepswe/total_trials": int(summary.get("total_trials") or 0),
            "deepswe/scored_trials": int(summary.get("scored_trials") or 0),
            "deepswe/exceptions": int(summary.get("exceptions") or 0),
            "deepswe/weave_agents_ok": int(summary.get("weave_agents_ok") or 0),
            "deepswe/native_trace_missing": int(summary.get("native_trace_missing") or 0),
        }
    )
    run.log_artifact(
        _make_result_artifact(
            artifact_name=(
                "deepswe-openclaw-"
                + _sanitize_artifact_component(model_name)
                + "-results"
            ),
            result_dir=result_dir,
            summary=summary,
            model_name=model_name,
            subset=subset,
        ),
        aliases=["latest", "production"],
    )


@weave.op(call_display_name=lambda _: "[DeepSWE] " + WandbConfigSingleton.get_instance().config.wandb.run_name)
def evaluate():
    instance = WandbConfigSingleton.get_instance()
    run = instance.run
    cfg = instance.config

    task_names_path = _task_names_path(cfg, run)
    output_dir = Path(_cfg_get(cfg.deepswe, "output_dir", "outputs/deepswe"))
    output_dir.mkdir(parents=True, exist_ok=True)

    if _cfg_get(cfg.deepswe, "run_openclaw", True):
        runner_output_dir = _run_openclaw(cfg, task_names_path, output_dir)
    else:
        results_dir = _cfg_get(cfg.deepswe, "results_dir")
        if not results_dir:
            raise ValueError("deepswe.results_dir is required when run_openclaw is false")
        runner_output_dir = Path(results_dir)

    summary, output_df = _load_results(runner_output_dir)
    if _cfg_get(cfg, "testmode", False) or _cfg_get(cfg.deepswe, "dry_run", False):
        run.log(
            {
                "deepswe_dry_run_table": wandb.Table(
                    dataframe=pd.DataFrame(
                        [
                            {
                                "task_names_file": str(task_names_path),
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
