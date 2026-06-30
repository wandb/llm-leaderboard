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
DEFAULT_MAX_INPUT_TOKENS = 1_000_000
DEFAULT_MAX_TOOL_CALLS = 60


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
        nemoclaw_openclaw_config_path = _cfg_get(cfg.swebench_pro, "nemoclaw_openclaw_config_path")
        if nemoclaw_openclaw_config_path:
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
    if _cfg_get(cfg.swebench_pro, "weave_sidecar", False):
        command.append("--weave-sidecar")
        if _cfg_get(cfg.swebench_pro, "weave_sidecar_strict", False):
            command.append("--weave-sidecar-strict")
        else:
            command.append("--no-weave-sidecar-strict")
    else:
        command.append("--no-weave-sidecar")
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
    per_instance = pd.DataFrame(
        [
            {"instance_id": iid, "resolved": iid in resolved}
            for iid in sorted(summary.get("resolved_ids", []) + summary.get("unresolved_ids", []))
        ]
    )
    run.log(
        {
            "agentic_swe_leaderboard_table": wandb.Table(dataframe=leaderboard),
            "agentic_swe_output_table": wandb.Table(dataframe=per_instance),
            "agentic_swe_results": summary,
            "agentic_swe/subset": _cfg_get(cfg.swebench_pro, "subset", DEFAULT_TAIWAN_SUBSET),
            "agentic_swe/max_input_tokens": int(
                _cfg_get(cfg.swebench_pro, "max_input_tokens", DEFAULT_MAX_INPUT_TOKENS) or 0
            ),
            "agentic_swe/max_tool_calls": int(
                _cfg_get(cfg.swebench_pro, "max_tool_calls", DEFAULT_MAX_TOOL_CALLS) or 0
            ),
            "agentic_swe/nemoclaw_sandbox": str(
                _cfg_get(cfg.swebench_pro, "nemoclaw_sandbox", "") or ""
            ),
            "agentic_swe/pass_at_1": float(summary["pass_at_1"]),
            "agentic_swe/resolved_instances": int(summary["resolved_instances"]),
            "agentic_swe/total_instances": int(summary["total_instances"]),
            "agentic_swe/unresolved_instances": int(summary["unresolved_instances"]),
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
