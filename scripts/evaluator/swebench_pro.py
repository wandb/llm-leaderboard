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
OPENCLAW_RUNNER = REPO_ROOT / "scripts" / "tools" / "run_swebench_pro_openclaw.py"
EVAL_RUNNER = REPO_ROOT / "scripts" / "tools" / "evaluate_swebench_pro_patches.py"
DEFAULT_TAIWAN_SUBSET = "leaderboard_compact_80"
DEFAULT_NEMOCLAW_OPENCLAW_CONFIG_PATH = "/sandbox/.openclaw/openclaw.json"
DEFAULT_MAX_INPUT_TOKENS = 1_000_000
DEFAULT_MAX_TOOL_CALLS = 40
DEFAULT_MAX_AGENT_TURNS = 40
AGENTIC_SWE_OUTPUT_TABLE_REQUIRED_COLUMNS = (
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


def _run_command(command: list[str]) -> subprocess.CompletedProcess[str]:
    print("Running:", " ".join(command))
    proc = subprocess.Popen(
        command,
        cwd=str(REPO_ROOT),
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
    stdout = "".join(stdout_parts)
    if returncode != 0:
        raise RuntimeError(
            f"Command failed with return code {returncode}: {command}\n{stdout[-4000:]}"
        )
    return subprocess.CompletedProcess(command, returncode, stdout=stdout, stderr="")


def _dataset_paths(cfg, run) -> tuple[Path, Path]:
    local_dataset_dir = _cfg_get(cfg.swebench_pro, "local_dataset_dir")
    if local_dataset_dir:
        dataset_dir = Path(local_dataset_dir)
    else:
        artifact_path = cfg.swebench_pro.artifacts_path
        artifact = run.use_artifact(artifact_path, type="dataset")
        artifact_root = Path(artifact.download())
        dataset_dir = artifact_root / cfg.swebench_pro.dataset_dir

    subset = _cfg_get(cfg.swebench_pro, "subset", DEFAULT_TAIWAN_SUBSET)
    if subset == "full_public":
        jsonl_path = dataset_dir / "subsets" / "full_public.jsonl"
        csv_path = dataset_dir / "subsets" / "full_public.csv"
    elif subset == "smoke":
        jsonl_path = dataset_dir / "subsets" / "smoke.jsonl"
        csv_path = dataset_dir / "subsets" / "smoke.csv"
    else:
        jsonl_path = dataset_dir / "subsets" / f"{subset}.jsonl"
        csv_path = dataset_dir / "subsets" / f"{subset}.csv"

    if not jsonl_path.exists():
        raise FileNotFoundError(
            f"{jsonl_path} does not exist. For Taiwan SWE-Bench Pro, build/upload "
            "`leaderboard_compact_80` with scripts/data_uploader/prepare_swebench_pro.py "
            "--compact-leaderboard-size 80 before running the default configuration."
        )
    if not csv_path.exists():
        raise FileNotFoundError(
            f"{csv_path} does not exist. For Taiwan SWE-Bench Pro, build/upload "
            "`leaderboard_compact_80` with scripts/data_uploader/prepare_swebench_pro.py "
            "--compact-leaderboard-size 80 before running the default configuration."
        )
    return jsonl_path, csv_path


def _run_openclaw(cfg, jsonl_path: Path, output_dir: Path) -> Path:
    command = [
        sys.executable,
        str(OPENCLAW_RUNNER),
        "--dataset-jsonl",
        str(jsonl_path),
        "--output-dir",
        str(output_dir / "openclaw"),
        "--checkout-root",
        str(Path(_cfg_get(cfg.swebench_pro, "checkout_root", "outputs/swebench_pro_checkouts"))),
        "--prefix",
        str(_cfg_get(cfg.swebench_pro, "prefix", "openclaw")),
        "--thinking",
        str(_cfg_get(cfg.swebench_pro, "thinking", "medium")),
        "--agent",
        str(_cfg_get(cfg.swebench_pro, "agent", "nejumi-taiwan")),
        "--openclaw-timeout",
        str(_cfg_get(cfg.swebench_pro, "openclaw_timeout", 3600)),
        "--openclaw-max-attempts",
        str(_cfg_get(cfg.swebench_pro, "openclaw_max_attempts", 3)),
        "--openclaw-retry-base-seconds",
        str(_cfg_get(cfg.swebench_pro, "openclaw_retry_base_seconds", 15)),
        "--max-input-tokens",
        str(_cfg_get(cfg.swebench_pro, "max_input_tokens", DEFAULT_MAX_INPUT_TOKENS)),
        "--max-tool-calls",
        str(_cfg_get(cfg.swebench_pro, "max_tool_calls", DEFAULT_MAX_TOOL_CALLS)),
        "--max-agent-turns",
        str(_cfg_get(cfg.swebench_pro, "max_agent_turns", DEFAULT_MAX_AGENT_TURNS)),
    ]
    profile = _cfg_get(cfg.swebench_pro, "profile")
    if profile:
        command.extend(["--profile", str(profile)])
    openclaw_config_template = _cfg_get(cfg.swebench_pro, "openclaw_config_template")
    if openclaw_config_template:
        command.extend(["--openclaw-config-template", str(openclaw_config_template)])
    openclaw_tool_profile = _cfg_get(cfg.swebench_pro, "openclaw_tool_profile")
    if openclaw_tool_profile:
        command.extend(["--openclaw-tool-profile", str(openclaw_tool_profile)])
    nemoclaw_sandbox = _cfg_get(cfg.swebench_pro, "nemoclaw_sandbox")
    if nemoclaw_sandbox:
        command.extend(["--nemoclaw-sandbox", str(nemoclaw_sandbox)])
        command.extend(["--nemoclaw-bin", str(_cfg_get(cfg.swebench_pro, "nemoclaw_bin", "nemoclaw"))])
        nemoclaw_openclaw_config_path = _cfg_get(
            cfg.swebench_pro,
            "nemoclaw_openclaw_config_path",
            DEFAULT_NEMOCLAW_OPENCLAW_CONFIG_PATH,
        )
        command.extend(
            [
                "--nemoclaw-openclaw-config-path",
                str(nemoclaw_openclaw_config_path),
            ]
        )
        nemoclaw_workdir = _cfg_get(cfg.swebench_pro, "nemoclaw_workdir")
        if nemoclaw_workdir:
            command.extend(["--nemoclaw-workdir", str(nemoclaw_workdir)])
        nemoclaw_checkout_sandbox_root = _cfg_get(
            cfg.swebench_pro,
            "nemoclaw_checkout_sandbox_root",
        )
        if nemoclaw_checkout_sandbox_root:
            command.extend(
                [
                    "--nemoclaw-checkout-sandbox-root",
                    str(nemoclaw_checkout_sandbox_root),
                ]
            )
        nemoclaw_checkout_transfer_mode = _cfg_get(
            cfg.swebench_pro,
            "nemoclaw_checkout_transfer_mode",
        )
        if nemoclaw_checkout_transfer_mode:
            command.extend(
                [
                    "--nemoclaw-checkout-transfer-mode",
                    str(nemoclaw_checkout_transfer_mode),
                ]
            )
        nemoclaw_checkout_transfer_timeout = _cfg_get(
            cfg.swebench_pro,
            "nemoclaw_checkout_transfer_timeout",
        )
        if nemoclaw_checkout_transfer_timeout:
            command.extend(
                [
                    "--nemoclaw-checkout-transfer-timeout",
                    str(nemoclaw_checkout_transfer_timeout),
                ]
            )
    if not _cfg_get(cfg.swebench_pro, "use_task_agent", True):
        command.append("--no-use-task-agent")
    task_agent_prefix = _cfg_get(cfg.swebench_pro, "task_agent_prefix")
    if task_agent_prefix:
        command.extend(["--task-agent-prefix", str(task_agent_prefix)])
    session_prefix = _cfg_get(cfg.swebench_pro, "session_prefix")
    if session_prefix:
        command.extend(["--session-prefix", str(session_prefix)])
    model = _cfg_get(cfg.swebench_pro, "openclaw_model") or _cfg_get(
        cfg.model, "pretrained_model_name_or_path", ""
    )
    if model:
        command.extend(["--model", str(model)])
    if _cfg_get(cfg.swebench_pro, "allow_failed_preflight", False):
        command.append("--allow-failed-preflight")
    if _cfg_get(cfg.swebench_pro, "no_local", False):
        command.append("--no-local")
    if _cfg_get(cfg.swebench_pro, "weave_sidecar", False) or _cfg_get(
        cfg.swebench_pro, "weave_sidecar_strict", False
    ):
        raise ValueError(
            "swebench_pro.weave_sidecar is disabled. Use native weave-openclaw "
            "Agents traces only; manual sidecar traces are not valid evidence."
        )
    command.append("--no-weave-sidecar")
    if _cfg_get(cfg.swebench_pro, "verify_weave_agents", False):
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
        value = _cfg_get(cfg.swebench_pro, cfg_key)
        if value is not None:
            command.extend([cli_key, str(value)])
    for denied_tool in _as_list(_cfg_get(cfg.swebench_pro, "deny_tool")):
        command.extend(["--deny-tool", str(denied_tool)])
    for pattern in _as_list(_cfg_get(cfg.swebench_pro, "deny_argument_pattern")):
        command.extend(["--deny-argument-pattern", str(pattern)])
    if _cfg_get(cfg.swebench_pro, "skip_agent", False):
        command.append("--skip-agent")
    if _cfg_get(cfg.swebench_pro, "redo", False):
        command.append("--redo")
    if _cfg_get(cfg, "testmode", False):
        command.extend(["--limit", str(_cfg_get(cfg.swebench_pro, "testmode_max_samples", 1))])
        command.append("--dry-run")
    elif _cfg_get(cfg.swebench_pro, "dry_run", False):
        command.append("--dry-run")
    _run_command(command)
    return output_dir / "openclaw" / "patches.json"


def _run_official_eval(cfg, csv_path: Path, patch_path: Path, output_dir: Path) -> dict[str, Any]:
    command = [
        sys.executable,
        str(EVAL_RUNNER),
        "--official-repo",
        str(_cfg_get(cfg.swebench_pro, "official_repo", "external/SWE-bench_Pro-os")),
        "--raw-sample-path",
        str(csv_path),
        "--patch-path",
        str(patch_path),
        "--output-dir",
        str(output_dir / "official_eval"),
        "--dockerhub-username",
        str(_cfg_get(cfg.swebench_pro, "dockerhub_username", "jefzda")),
        "--num-workers",
        str(_cfg_get(cfg.swebench_pro, "num_workers", 8)),
        "--model-name",
        str(_cfg_get(cfg.model, "pretrained_model_name_or_path", "openclaw")),
    ]
    if _cfg_get(cfg.swebench_pro, "use_local_docker", True):
        command.append("--use-local-docker")
    else:
        command.append("--no-use-local-docker")
    docker_platform = _cfg_get(cfg.swebench_pro, "docker_platform")
    if docker_platform:
        command.extend(["--docker-platform", str(docker_platform)])
    if _cfg_get(cfg.swebench_pro, "redo", False):
        command.append("--redo")
    if _cfg_get(cfg.swebench_pro, "block_network", False):
        command.append("--block-network")

    _run_command(command)
    summary_path = output_dir / "official_eval" / "summary.json"
    return json.loads(summary_path.read_text(encoding="utf-8"))


def _read_patch_rows(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, list):
        raise ValueError(f"{path} must contain a JSON list")
    rows: list[dict[str, Any]] = []
    for index, row in enumerate(payload, start=1):
        if not isinstance(row, dict):
            raise ValueError(f"{path}:{index} must contain a JSON object")
        rows.append(row)
    return rows


def _nemoclaw_audit_counts(patch_rows: list[dict[str, Any]]) -> dict[str, int]:
    required = [
        row
        for row in patch_rows
        if isinstance(row.get("nemoclaw_session_audit"), dict)
        and row["nemoclaw_session_audit"].get("required") is True
    ]
    passed = [row for row in required if row.get("nemoclaw_session_audit_ok") is True]
    failed = [row for row in required if row.get("nemoclaw_session_audit_ok") is False]
    return {
        "required": len(required),
        "passed": len(passed),
        "failed": len(failed),
    }


def _validate_output_table_columns(output_df: pd.DataFrame) -> None:
    missing = [
        column
        for column in AGENTIC_SWE_OUTPUT_TABLE_REQUIRED_COLUMNS
        if column not in output_df.columns
    ]
    if missing:
        raise ValueError(
            "agentic_swe_output_table is missing required observability columns: "
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
    output_dir: Path,
    patch_path: Path,
    summary: dict[str, Any],
    model_name: str,
    subset: str,
    max_input_tokens: int,
    max_tool_calls: int,
    max_agent_turns: int,
    nemoclaw_sandbox: str,
) -> wandb.Artifact:
    artifact = wandb.Artifact(
        artifact_name,
        type="evaluation-results",
        metadata={
            "model_name": model_name,
            "benchmark": "SWE-Bench Pro",
            "total_instances": summary["total_instances"],
            "resolved_instances": summary["resolved_instances"],
            "pass_at_1": summary["pass_at_1"],
            "source_output_dir": str(output_dir),
            "patch_path": str(patch_path),
            "subset": subset,
            "max_input_tokens": max_input_tokens,
            "max_tool_calls": max_tool_calls,
            "max_agent_turns": max_agent_turns,
            "nemoclaw_sandbox": nemoclaw_sandbox,
        },
    )
    if patch_path.exists():
        artifact.add_file(str(patch_path), name="patches.json")
    official_eval_dir = output_dir / "official_eval"
    for filename in (
        "summary.json",
        "eval_results.json",
        "official_invocation.json",
        "official_stdout.log",
        "official_stderr.log",
    ):
        path = official_eval_dir / filename
        if path.exists():
            artifact.add_file(str(path), name=f"official_eval/{filename}")
    return artifact


def _log_summary(run, cfg, summary: dict[str, Any], output_dir: Path, patch_path: Path) -> None:
    model_name = _cfg_get(cfg.model, "pretrained_model_name_or_path", "openclaw")
    leaderboard = pd.DataFrame(
        [
            {
                "model_name": model_name,
                "total_samples": summary["total_instances"],
                "issues_resolved": summary["resolved_instances"],
                "pass_at_1": summary["pass_at_1"],
            }
        ]
    )
    resolved = set(summary.get("resolved_ids", []))
    patch_rows = _read_patch_rows(patch_path)
    patch_by_instance = {
        str(row.get("instance_id")): row
        for row in patch_rows
        if row.get("instance_id") is not None
    }
    per_instance_rows = []
    for iid in sorted(summary.get("resolved_ids", []) + summary.get("unresolved_ids", [])):
        patch = patch_by_instance.get(iid) or {}
        audit = patch.get("nemoclaw_session_audit")
        copy_status = audit.get("copy") if isinstance(audit, dict) else None
        per_instance_rows.append(
            {
                "instance_id": iid,
                "resolved": iid in resolved,
                "has_patch_record": bool(patch),
                "openclaw_returncode": patch.get("openclaw_returncode"),
                "tool_policy_ok": patch.get("tool_policy_ok"),
                "tool_policy_violations": patch.get("tool_policy_violations"),
                "conversation_order_ok": patch.get("conversation_order_ok"),
                "conversation_order": patch.get("conversation_order"),
                "weave_agents_ok": patch.get("weave_agents_ok"),
                "weave_agents_required": patch.get("weave_agents_required"),
                "weave_agents_agent_name": patch.get("weave_agents_agent_name"),
                "weave_agents_conversation_id": patch.get("weave_agents_conversation_id"),
                "weave_agents_conversation_id_contains": patch.get(
                    "weave_agents_conversation_id_contains"
                ),
                "weave_agents_conversation_url": patch.get("weave_agents_conversation_url"),
                "weave_agents_conversation_link_html": patch.get(
                    "weave_agents_conversation_link_html"
                ),
                "weave_agents_trace_id": patch.get("weave_agents_trace_id"),
                "weave_agents_url": patch.get("weave_agents_url"),
                "weave_agents_trace_url": patch.get("weave_agents_trace_url"),
                "weave_agents_verifier_json": patch.get("weave_agents_verifier_json"),
                "weave_agents_error": patch.get("weave_agents_error"),
                "nemoclaw_session_audit_ok": patch.get("nemoclaw_session_audit_ok"),
                "nemoclaw_session_audit_required": (
                    audit.get("required") if isinstance(audit, dict) else None
                ),
                "nemoclaw_session_copy_source": (
                    copy_status.get("source") if isinstance(copy_status, dict) else None
                ),
                "nemoclaw_session_copied_bytes": (
                    audit.get("copied_session_bytes") if isinstance(audit, dict) else None
                ),
                "openclaw_tool_call_count": patch.get("openclaw_tool_call_count"),
                "openclaw_result_path": patch.get("openclaw_result_path"),
                "openclaw_invocation_path": patch.get("openclaw_invocation_path"),
                "openclaw_invocation_sha256": patch.get("openclaw_invocation_sha256"),
                "openclaw_command_sha256": patch.get("openclaw_command_sha256"),
                "openclaw_config_source": patch.get("openclaw_config_source"),
            }
        )
    per_instance = pd.DataFrame(per_instance_rows)
    _validate_output_table_columns(per_instance)
    per_instance_table = _prepare_output_table_for_wandb(per_instance)
    audit_counts = _nemoclaw_audit_counts(patch_rows)
    weave_agents_required = [row for row in patch_rows if row.get("weave_agents_required") is True]
    weave_agents_passed = [row for row in weave_agents_required if row.get("weave_agents_ok") is True]
    weave_agents_failed = [row for row in weave_agents_required if row.get("weave_agents_ok") is not True]
    run.log(
        {
            "agentic_swe_leaderboard_table": wandb.Table(dataframe=leaderboard),
            "agentic_swe_output_table": wandb.Table(dataframe=per_instance_table),
            "agentic_swe_results": summary,
            "agentic_swe/subset": _cfg_get(cfg.swebench_pro, "subset", DEFAULT_TAIWAN_SUBSET),
            "agentic_swe/max_input_tokens": int(
                _cfg_get(cfg.swebench_pro, "max_input_tokens", DEFAULT_MAX_INPUT_TOKENS) or 0
            ),
            "agentic_swe/max_tool_calls": int(
                _cfg_get(cfg.swebench_pro, "max_tool_calls", DEFAULT_MAX_TOOL_CALLS) or 0
            ),
            "agentic_swe/max_agent_turns": int(
                _cfg_get(cfg.swebench_pro, "max_agent_turns", DEFAULT_MAX_AGENT_TURNS) or 0
            ),
            "agentic_swe/nemoclaw_sandbox": str(
                _cfg_get(cfg.swebench_pro, "nemoclaw_sandbox", "") or ""
            ),
            "agentic_swe/pass_at_1": float(summary["pass_at_1"]),
            "agentic_swe/resolved_instances": int(summary["resolved_instances"]),
            "agentic_swe/total_instances": int(summary["total_instances"]),
            "agentic_swe/unresolved_instances": int(summary["unresolved_instances"]),
            "agentic_swe/weave_agents_required_patches": len(weave_agents_required),
            "agentic_swe/weave_agents_passed_patches": len(weave_agents_passed),
            "agentic_swe/weave_agents_failed_patches": len(weave_agents_failed),
            "agentic_swe/nemoclaw_session_audit_required_patches": audit_counts["required"],
            "agentic_swe/nemoclaw_session_audit_passed_patches": audit_counts["passed"],
            "agentic_swe/nemoclaw_session_audit_failed_patches": audit_counts["failed"],
        }
    )
    run.log_artifact(
        _make_result_artifact(
            artifact_name=(
                "agentic-swe-swebench-pro-"
                + _sanitize_artifact_component(model_name)
                + "-results"
            ),
            output_dir=output_dir,
            patch_path=patch_path,
            summary=summary,
            model_name=model_name,
            subset=str(_cfg_get(cfg.swebench_pro, "subset", DEFAULT_TAIWAN_SUBSET)),
            max_input_tokens=int(
                _cfg_get(cfg.swebench_pro, "max_input_tokens", DEFAULT_MAX_INPUT_TOKENS) or 0
            ),
            max_tool_calls=int(
                _cfg_get(cfg.swebench_pro, "max_tool_calls", DEFAULT_MAX_TOOL_CALLS) or 0
            ),
            max_agent_turns=int(
                _cfg_get(cfg.swebench_pro, "max_agent_turns", DEFAULT_MAX_AGENT_TURNS) or 0
            ),
            nemoclaw_sandbox=str(_cfg_get(cfg.swebench_pro, "nemoclaw_sandbox", "") or ""),
        ),
        aliases=["latest", "production"],
    )


@weave.op(call_display_name=lambda _: "[SWE-bench Pro] " + WandbConfigSingleton.get_instance().config.wandb.run_name)
def evaluate():
    instance = WandbConfigSingleton.get_instance()
    run = instance.run
    cfg = instance.config

    jsonl_path, csv_path = _dataset_paths(cfg, run)
    output_dir = Path(_cfg_get(cfg.swebench_pro, "output_dir", "outputs/swebench_pro"))
    output_dir.mkdir(parents=True, exist_ok=True)

    patch_config = _cfg_get(cfg.swebench_pro, "patch_path")
    patch_path = Path(patch_config) if patch_config else None
    if _cfg_get(cfg.swebench_pro, "run_openclaw", True):
        patch_path = _run_openclaw(cfg, jsonl_path, output_dir)

    if patch_path is None or not patch_path.exists():
        raise FileNotFoundError(
            "No SWE-bench Pro patch file found. Set swebench_pro.patch_path or enable run_openclaw."
        )

    if _cfg_get(cfg, "testmode", False) or _cfg_get(cfg.swebench_pro, "dry_run", False):
        run.log(
            {
                "agentic_swe_dry_run_table": wandb.Table(
                    dataframe=pd.DataFrame(
                        [
                            {
                                "dataset_jsonl": str(jsonl_path),
                                "raw_sample_csv": str(csv_path),
                                "patch_path": str(patch_path),
                                "evaluated": False,
                            }
                        ]
                    )
                )
            }
        )
        return None

    if _cfg_get(cfg.swebench_pro, "evaluate", True):
        summary = _run_official_eval(cfg, csv_path, patch_path, output_dir)
        _log_summary(run, cfg, summary, output_dir, patch_path)
        return summary
    return None
