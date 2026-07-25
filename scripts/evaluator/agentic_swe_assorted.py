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
RUNNER = REPO_ROOT / "scripts" / "tools" / "run_agentic_swe_assorted.py"

OUTPUT_TABLE_REQUIRED_COLUMNS = (
    "source_benchmark",
    "source_dataset",
    "source_subset",
    "source_instance_id",
    "agentic_swe_tier",
    "instance_id",
    "resolved",
    "score",
    "official_score",
    "official_partial_credit",
    "official_score_version",
    "binary_score",
    "diagnostic_partial_credit",
    "diagnostic_score_with_partial",
    "diagnostic_partial_credit_version",
    "weave_agents_conversation_url",
    "openclaw_tool_call_count",
    "openclaw_usage",
    "billable_openclaw_usage",
    "billable_openclaw_attempt_count",
    "billable_openclaw_attempts",
    "billable_openclaw_wall_seconds",
    "nemoclaw_session_audit_ok",
    "nemoclaw_session_audit_required",
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
    from evaluator.evaluate_utils.subprocess_runner import run_streaming_command

    return run_streaming_command(command, cwd=REPO_ROOT, timeout=timeout)


def _append_optional(command: list[str], cfg, cfg_key: str, cli_key: str) -> None:
    value = _cfg_get(cfg.agentic_swe_assorted, cfg_key)
    if value is not None:
        command.extend([cli_key, str(value)])


def _build_assorted_command(cfg, output_dir: Path) -> list[str]:
    assorted = cfg.agentic_swe_assorted
    model = _cfg_get(assorted, "openclaw_model") or _cfg_get(
        cfg.model, "pretrained_model_name_or_path", ""
    )
    nemoclaw_sandbox = _cfg_get(assorted, "nemoclaw_sandbox")
    if (
        nemoclaw_sandbox is None
        or not str(nemoclaw_sandbox).strip()
        or str(nemoclaw_sandbox).strip().lower() in {"none", "null"}
    ):
        raise ValueError(
            "agentic_swe_assorted.nemoclaw_sandbox must name an isolated NeMoClaw "
            "sandbox before Agentic SWE-Assorted can run. Refusing to pass the literal "
            "string 'None' to the paid runner."
        )
    command = [
        sys.executable,
        str(RUNNER),
        "--model",
        str(model),
        "--output-dir",
        str(output_dir / "runner"),
        "--prefix",
        str(_cfg_get(assorted, "prefix", "agentic-swe-assorted")),
        "--thinking",
        str(_cfg_get(assorted, "thinking", "high")),
        "--agent",
        str(_cfg_get(assorted, "agent", "main")),
        "--tier-weights",
        str(_cfg_get(assorted, "tier_weights", "low=1,middle=1,high=1")),
        "--low-middle-jsonl",
        str(_cfg_get(assorted, "low_middle_jsonl")),
        "--low-middle-instance-ids-json",
        str(_cfg_get(assorted, "low_middle_instance_ids_json")),
        "--deepswe-metadata-jsonl",
        str(_cfg_get(assorted, "deepswe_metadata_jsonl")),
        "--deepswe-task-names-file",
        str(_cfg_get(assorted, "deepswe_task_names_file")),
        "--deepswe-tasks-root",
        str(_cfg_get(assorted, "deepswe_tasks_root")),
        "--deepswe-public-trials-json",
        str(_cfg_get(assorted, "deepswe_public_trials_json")),
        "--official-swebench-repo",
        str(_cfg_get(assorted, "official_swebench_repo")),
        "--checkout-root",
        str(_cfg_get(assorted, "checkout_root")),
        "--swe-workers",
        str(_cfg_get(assorted, "swe_workers", 4)),
        "--high-workers",
        str(_cfg_get(assorted, "high_workers", 2)),
        "--eval-workers",
        str(_cfg_get(assorted, "eval_workers", 4)),
        "--swe-task-start-min-interval-seconds",
        str(_cfg_get(assorted, "swe_task_start_min_interval_seconds", 5.0)),
        "--swe-openclaw-timeout",
        str(_cfg_get(assorted, "swe_openclaw_timeout", 900)),
        "--high-openclaw-timeout",
        str(_cfg_get(assorted, "high_openclaw_timeout", 1800)),
        "--eval-timeout",
        str(_cfg_get(assorted, "eval_timeout", 1800)),
        "--openclaw-max-attempts",
        str(_cfg_get(assorted, "openclaw_max_attempts", 2)),
        "--openclaw-retry-base-seconds",
        str(_cfg_get(assorted, "openclaw_retry_base_seconds", 15)),
        "--provider-recovery-rounds",
        str(_cfg_get(assorted, "provider_recovery_rounds", 2)),
        "--provider-recovery-base-seconds",
        str(_cfg_get(assorted, "provider_recovery_base_seconds", 60)),
        "--max-input-tokens",
        str(_cfg_get(assorted, "max_input_tokens", 1_000_000)),
        "--max-cumulative-input-tokens",
        str(_cfg_get(assorted, "max_cumulative_input_tokens", 1_000_000)),
        "--high-max-cumulative-input-tokens",
        str(_cfg_get(assorted, "high_max_cumulative_input_tokens", 14_000_000)),
        "--max-cumulative-output-tokens",
        str(_cfg_get(assorted, "max_cumulative_output_tokens", 500_000)),
        "--max-tool-calls",
        str(_cfg_get(assorted, "max_tool_calls", 40)),
        "--high-max-tool-calls",
        str(_cfg_get(assorted, "high_max_tool_calls", 200)),
        "--max-agent-turns",
        str(_cfg_get(assorted, "max_agent_turns", 40)),
        "--high-max-agent-turns",
        str(_cfg_get(assorted, "high_max_agent_turns", 150)),
        "--max-tool-wall-seconds",
        str(_cfg_get(assorted, "max_tool_wall_seconds", 120)),
        "--llm-response-idle-timeout-seconds",
        str(_cfg_get(assorted, "llm_response_idle_timeout_seconds", 900)),
        "--nemoclaw-bin",
        str(_cfg_get(assorted, "nemoclaw_bin", "nemoclaw")),
        "--nemoclaw-sandbox",
        str(nemoclaw_sandbox),
        "--nemoclaw-workdir",
        str(_cfg_get(assorted, "nemoclaw_workdir", "/sandbox")),
        "--nemoclaw-openclaw-config-path",
        str(_cfg_get(assorted, "nemoclaw_openclaw_config_path", "/sandbox/.openclaw/openclaw.json")),
        "--nemoclaw-checkout-transfer-mode",
        str(_cfg_get(assorted, "nemoclaw_checkout_transfer_mode", "copy")),
        "--nemoclaw-checkout-transfer-timeout",
        str(_cfg_get(assorted, "nemoclaw_checkout_transfer_timeout", 600)),
        "--openclaw-tool-profile",
        str(_cfg_get(assorted, "openclaw_tool_profile", "coding")),
        "--task-agent-prefix",
        str(_cfg_get(assorted, "task_agent_prefix", "tw-swe-assorted")),
        "--session-prefix",
        str(_cfg_get(assorted, "session_prefix", "agentic-swe-assorted")),
        "--weave-agents-entity",
        str(_cfg_get(assorted, "weave_agents_entity", "llm-leaderboard")),
        "--weave-agents-project",
        str(_cfg_get(assorted, "weave_agents_project", "tc-leaderboard")),
        "--weave-agents-agent-name",
        str(_cfg_get(assorted, "weave_agents_agent_name", "nejumi-taiwan-openclaw")),
        "--weave-agents-limit",
        str(_cfg_get(assorted, "weave_agents_limit", 100)),
        "--weave-agents-verification-timeout",
        str(_cfg_get(assorted, "weave_agents_verification_timeout", 120)),
        "--weave-agents-poll-seconds",
        str(_cfg_get(assorted, "weave_agents_poll_seconds", 5)),
        "--swebench-namespace",
        str(_cfg_get(assorted, "swebench_namespace", "swebench")),
        "--swebench-cache-level",
        str(_cfg_get(assorted, "swebench_cache_level", "env")),
    ]
    for cfg_key, cli_key in (
        ("low_middle_limit", "--low-middle-limit"),
        ("low_limit", "--low-limit"),
        ("middle_limit", "--middle-limit"),
        ("high_limit", "--high-limit"),
        ("deepswe_public_model", "--deepswe-public-model"),
        ("deepswe_public_effort", "--deepswe-public-effort"),
        ("deepswe_budget_preflight", "--deepswe-budget-preflight"),
        ("deepswe_preflight_hard_stat", "--deepswe-preflight-hard-stat"),
        ("eval_run_id", "--eval-run-id"),
    ):
        _append_optional(command, cfg, cfg_key, cli_key)

    openclaw_model_params = _cfg_get(assorted, "openclaw_model_params")
    if openclaw_model_params is not None:
        command.extend(["--openclaw-model-params-json", _json_cli_arg(openclaw_model_params)])
    high_openclaw_model_params = _cfg_get(assorted, "high_openclaw_model_params")
    if high_openclaw_model_params is not None:
        command.extend(
            [
                "--high-openclaw-model-params-json",
                _json_cli_arg(high_openclaw_model_params),
            ]
        )
    openclaw_model_overrides = _cfg_get(assorted, "openclaw_model_overrides")
    if openclaw_model_overrides is not None:
        command.extend(
            ["--openclaw-model-overrides-json", _json_cli_arg(openclaw_model_overrides)]
        )
    high_openclaw_model_overrides = _cfg_get(assorted, "high_openclaw_model_overrides")
    if high_openclaw_model_overrides is not None:
        command.extend(
            [
                "--high-openclaw-model-overrides-json",
                _json_cli_arg(high_openclaw_model_overrides),
            ]
        )

    for boolean_cfg, positive, negative in (
        ("require_actual_token_usage", "--require-actual-token-usage", "--no-require-actual-token-usage"),
        ("use_task_agent", "--use-task-agent", "--no-use-task-agent"),
        ("verify_weave_agents", "--verify-weave-agents", "--no-verify-weave-agents"),
    ):
        command.append(positive if _cfg_get(assorted, boolean_cfg, True) else negative)
    if _cfg_get(assorted, "skip_low_middle", False):
        command.append("--skip-low-middle")
    if _cfg_get(assorted, "skip_high", False):
        command.append("--skip-high")
    for results_path in _as_list(_cfg_get(assorted, "reuse_high_results_jsonl")):
        command.extend(["--reuse-high-results-jsonl", str(results_path)])
    reuse_low_middle_patches_json = _cfg_get(
        assorted, "reuse_low_middle_patches_json"
    )
    if reuse_low_middle_patches_json:
        command.extend(
            [
                "--reuse-low-middle-patches-json",
                str(reuse_low_middle_patches_json),
            ]
        )
    if _cfg_get(assorted, "dry_run", False) or _cfg_get(cfg, "testmode", False):
        command.append("--dry-run")
    if _cfg_get(assorted, "no_docker_check", False):
        command.append("--no-docker-check")
    if _cfg_get(assorted, "allow_deepswe_budget_mismatch", False):
        command.append("--allow-deepswe-budget-mismatch")
    for denied_tool in _as_list(_cfg_get(assorted, "deny_tool")):
        command.extend(["--deny-tool", str(denied_tool)])
    for pattern in _as_list(_cfg_get(assorted, "deny_argument_pattern")):
        command.extend(["--deny-argument-pattern", str(pattern)])

    return command


def _run_assorted(cfg, output_dir: Path) -> Path:
    command = _build_assorted_command(cfg, output_dir)
    _run_command(
        command,
        timeout=float(
            _cfg_get(
                cfg.agentic_swe_assorted,
                "benchmark_timeout_seconds",
                50_400,
            )
        ),
    )
    return output_dir / "runner"


def preflight(cfg, output_dir: Path) -> dict[str, Any]:
    preflight_output_dir = output_dir / "runner"
    command = _build_assorted_command(cfg, output_dir)
    command.append("--preflight-only")
    try:
        result = subprocess.run(
            command,
            cwd=REPO_ROOT,
            text=True,
            capture_output=True,
            check=False,
            timeout=float(
                _cfg_get(
                    cfg.agentic_swe_assorted,
                    "preflight_timeout_seconds",
                    180,
                )
            ),
        )
    except subprocess.TimeoutExpired as exc:
        return {
            "benchmark": "agentic_swe_assorted",
            "ok": False,
            "returncode": 124,
            "command": command,
            "report_path": None,
            "report": None,
            "error": (
                "Agentic SWE-Assorted preflight timed out after "
                f"{exc.timeout}s"
            ),
            "will_run_model": False,
            "will_run_gateway": False,
            "will_run_grading": False,
            "will_initialize_wandb": False,
        }
    report_path = preflight_output_dir / "preflight.json"
    report = None
    if report_path.exists():
        try:
            report = json.loads(report_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            report = None
    ok = result.returncode == 0 and isinstance(report, dict) and report.get("ok") is True
    error_text = (result.stderr or result.stdout).strip()
    return {
        "benchmark": "agentic_swe_assorted",
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


def _load_results(result_dir: Path) -> tuple[dict[str, Any], pd.DataFrame, pd.DataFrame]:
    summary = json.loads((result_dir / "summary.json").read_text(encoding="utf-8"))
    rows = [
        json.loads(line)
        for line in (result_dir / "output_table.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    for row in rows:
        audit = row.get("nemoclaw_session_audit")
        if not isinstance(audit, dict):
            continue
        copy_status = audit.get("copy")
        if row.get("nemoclaw_session_audit_required") is None:
            row["nemoclaw_session_audit_required"] = audit.get("required")
        if row.get("nemoclaw_session_copy_source") in (None, ""):
            row["nemoclaw_session_copy_source"] = (
                copy_status.get("source") if isinstance(copy_status, dict) else None
            )
        if row.get("nemoclaw_session_copied_bytes") in (None, ""):
            row["nemoclaw_session_copied_bytes"] = audit.get(
                "copied_session_bytes"
            )
            if (
                row["nemoclaw_session_copied_bytes"] in (None, "")
                and isinstance(copy_status, dict)
            ):
                row["nemoclaw_session_copied_bytes"] = copy_status.get("bytes")
    leaderboard = json.loads((result_dir / "leaderboard_table.json").read_text(encoding="utf-8"))
    return summary, pd.DataFrame(rows), pd.DataFrame(leaderboard)


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
    missing = [column for column in OUTPUT_TABLE_REQUIRED_COLUMNS if column not in output_df.columns]
    if missing:
        raise ValueError(f"agentic_swe_output_table is missing required columns: {missing}")


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
) -> wandb.Artifact:
    artifact = wandb.Artifact(
        artifact_name,
        type="evaluation-results",
        metadata={
            "model_name": model_name,
            "benchmark": "Agentic SWE-Assorted",
            "total_instances": summary.get("total", {}).get("total_instances"),
            "resolved_instances": summary.get("total", {}).get("resolved_instances"),
            "pass_at_1": summary.get("scoring", {}).get("weighted_pass_at_1"),
            "binary_pass_at_1": summary.get("scoring", {}).get(
                "weighted_binary_pass_at_1"
            ),
            "source_result_dir": str(result_dir),
        },
    )
    for filename in ("summary.json", "output_table.jsonl", "leaderboard_table.json", "report.md"):
        path = result_dir / filename
        if path.exists():
            artifact.add_file(str(path), name=filename)
    return artifact


def _metric_float(value: Any) -> float:
    return float(value) if isinstance(value, (int, float)) else 0.0


def _log_summary(
    run,
    cfg,
    summary: dict[str, Any],
    output_df: pd.DataFrame,
    leaderboard: pd.DataFrame,
    result_dir: Path,
) -> None:
    _validate_output_table_columns(output_df)
    table_df = _prepare_output_table_for_wandb(output_df)
    model_name = _cfg_get(cfg.model, "pretrained_model_name_or_path", "openclaw")
    scoring = summary.get("scoring", {}) if isinstance(summary.get("scoring"), dict) else {}
    total = summary.get("total", {}) if isinstance(summary.get("total"), dict) else {}
    by_tier = summary.get("by_tier", {}) if isinstance(summary.get("by_tier"), dict) else {}
    audit_required = int(
        output_df["nemoclaw_session_audit_required"].map(lambda value: value is True).sum()
    )
    audit_passed = int(
        output_df["nemoclaw_session_audit_ok"].map(lambda value: value is True).sum()
    )
    metrics = {
        "agentic_swe_leaderboard_table": wandb.Table(dataframe=leaderboard),
        "agentic_swe_output_table": wandb.Table(dataframe=table_df),
        "agentic_swe_results": summary,
        "agentic_swe/micro_pass_at_1": _metric_float(total.get("micro_pass_at_1")),
        "agentic_swe/weighted_present_pass_at_1": _metric_float(
            scoring.get("weighted_present_pass_at_1")
        ),
        "agentic_swe/weighted_present_binary_pass_at_1": _metric_float(
            scoring.get("weighted_present_binary_pass_at_1")
        ),
        "agentic_swe/weighted_present_diagnostic_score_with_partial": _metric_float(
            scoring.get("weighted_present_diagnostic_score_with_partial")
        ),
        "agentic_swe/weighted_pass_at_1_complete": int(
            bool(scoring.get("weighted_pass_at_1_complete"))
        ),
        "agentic_swe/resolved_instances": int(total.get("resolved_instances") or 0),
        "agentic_swe/total_instances": int(total.get("total_instances") or 0),
        "agentic_swe/billable_attempt_count": int(total.get("billable_attempt_count") or 0),
        "agentic_swe/billable_retry_count": int(total.get("billable_retry_count") or 0),
        "agentic_swe/billable_wall_seconds": float(total.get("billable_wall_seconds") or 0.0),
        "agentic_swe/billable_input_tokens": float(
            (total.get("usage") or {}).get("input_tokens") or 0.0
        ),
        "agentic_swe/billable_output_tokens": float(
            (total.get("usage") or {}).get("output_tokens") or 0.0
        ),
        "agentic_swe/billable_cost_usd": float(
            (total.get("usage") or {}).get("cost_usd") or 0.0
        ),
        "agentic_swe/nemoclaw_session_audit_required_patches": audit_required,
        "agentic_swe/nemoclaw_session_audit_passed_patches": audit_passed,
        "agentic_swe/nemoclaw_session_audit_failed_patches": max(
            0, audit_required - audit_passed
        ),
        "agentic_swe/low/pass_at_1": _metric_float(by_tier.get("low", {}).get("pass_at_1")),
        "agentic_swe/middle/pass_at_1": _metric_float(
            by_tier.get("middle", {}).get("pass_at_1")
        ),
        "agentic_swe/high/pass_at_1": _metric_float(by_tier.get("high", {}).get("pass_at_1")),
        "agentic_swe/low/score": _metric_float(
            by_tier.get("low", {}).get("official_score")
        ),
        "agentic_swe/middle/score": _metric_float(
            by_tier.get("middle", {}).get("official_score")
        ),
        "agentic_swe/high/score": _metric_float(
            by_tier.get("high", {}).get("official_score")
        ),
        "agentic_swe/low/diagnostic_score_with_partial": _metric_float(
            by_tier.get("low", {}).get("diagnostic_score_with_partial")
        ),
        "agentic_swe/middle/diagnostic_score_with_partial": _metric_float(
            by_tier.get("middle", {}).get("diagnostic_score_with_partial")
        ),
        "agentic_swe/high/diagnostic_score_with_partial": _metric_float(
            by_tier.get("high", {}).get("diagnostic_score_with_partial")
        ),
    }
    if scoring.get("weighted_pass_at_1") is not None:
        metrics["agentic_swe/pass_at_1"] = float(scoring["weighted_pass_at_1"])
        metrics["agentic_swe/weighted_pass_at_1"] = float(scoring["weighted_pass_at_1"])
        metrics["agentic_swe/score"] = float(scoring["weighted_pass_at_1"])
    if scoring.get("weighted_binary_pass_at_1") is not None:
        metrics["agentic_swe/binary_pass_at_1"] = float(
            scoring["weighted_binary_pass_at_1"]
        )
    if scoring.get("weighted_diagnostic_score_with_partial") is not None:
        metrics["agentic_swe/diagnostic_score_with_partial"] = float(
            scoring["weighted_diagnostic_score_with_partial"]
        )
    run.log(metrics)
    run.log_artifact(
        _make_result_artifact(
            artifact_name=(
                "agentic-swe-assorted-"
                + _sanitize_artifact_component(model_name)
                + "-results"
            ),
            result_dir=result_dir,
            summary=summary,
            model_name=model_name,
        ),
        aliases=["latest", "production"],
    )


@weave.op(call_display_name=lambda _: "[Agentic SWE-Assorted] " + WandbConfigSingleton.get_instance().config.wandb.run_name)
def evaluate():
    instance = WandbConfigSingleton.get_instance()
    run = instance.run
    cfg = instance.config

    output_dir = Path(_cfg_get(cfg.agentic_swe_assorted, "output_dir", "outputs/agentic_swe_assorted"))
    output_dir.mkdir(parents=True, exist_ok=True)

    if _cfg_get(cfg.agentic_swe_assorted, "run_openclaw", True):
        result_dir = _run_assorted(cfg, output_dir)
    else:
        results_dir = _cfg_get(cfg.agentic_swe_assorted, "results_dir")
        if not results_dir:
            raise ValueError(
                "agentic_swe_assorted.results_dir is required when run_openclaw is false"
            )
        result_dir = Path(results_dir)

    summary, output_df, leaderboard = _load_results(result_dir)
    if _cfg_get(cfg, "testmode", False) or _cfg_get(cfg.agentic_swe_assorted, "dry_run", False):
        run.log(
            {
                "agentic_swe_dry_run_table": wandb.Table(
                    dataframe=pd.DataFrame(
                        [
                            {
                                "results_dir": str(result_dir),
                                "evaluated": False,
                            }
                        ]
                    )
                )
            }
        )
        return summary

    _log_summary(run, cfg, summary, output_df, leaderboard, result_dir)
    return summary
