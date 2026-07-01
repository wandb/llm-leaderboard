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
OPENCLAW_RUNNER = REPO_ROOT / "scripts" / "tools" / "run_agentic_math_openclaw.py"


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


def _run_openclaw(cfg, jsonl_path: Path, output_dir: Path) -> Path:
    use_task_agent = _cfg_get(cfg.agentic_math, "use_task_agent", True)
    nemoclaw_sandbox = _cfg_get(cfg.agentic_math, "nemoclaw_sandbox")
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
    ]
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
    if nemoclaw_sandbox:
        command.extend(["--nemoclaw-sandbox", str(nemoclaw_sandbox)])
        command.extend(["--nemoclaw-bin", str(_cfg_get(cfg.agentic_math, "nemoclaw_bin", "nemoclaw"))])
        command.extend(
            ["--nemoclaw-workdir", str(_cfg_get(cfg.agentic_math, "nemoclaw_workdir", "/sandbox"))]
        )
        nemoclaw_openclaw_config_path = _cfg_get(cfg.agentic_math, "nemoclaw_openclaw_config_path")
        if nemoclaw_openclaw_config_path:
            command.extend(["--nemoclaw-openclaw-config-path", str(nemoclaw_openclaw_config_path)])
    if _cfg_get(cfg.agentic_math, "allow_failed_preflight", False):
        command.append("--allow-failed-preflight")
    if _cfg_get(cfg.agentic_math, "no_local", False):
        command.append("--no-local")
    if _cfg_get(cfg.agentic_math, "weave_sidecar", False):
        command.append("--weave-sidecar")
        if _cfg_get(cfg.agentic_math, "weave_sidecar_strict", False):
            command.append("--weave-sidecar-strict")
        else:
            command.append("--no-weave-sidecar-strict")
    else:
        command.append("--no-weave-sidecar")
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
    _run_command(command)
    return output_dir / "openclaw"


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


def _log_summary(run, cfg, summary: dict[str, Any], output_df: pd.DataFrame) -> None:
    model_name = _cfg_get(cfg.model, "pretrained_model_name_or_path", "openclaw")
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
            "agentic_math_output_table": wandb.Table(dataframe=output_df),
            "agentic_math_results": summary,
            "agentic_math/accuracy": float(summary["accuracy"]),
            "agentic_math/correct_instances": int(summary["correct_instances"]),
            "agentic_math/total_instances": int(summary["total_instances"]),
            "agentic_math/answered_instances": int(summary["answered_instances"]),
            **_nemoclaw_audit_metrics(summary),
        }
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

    _log_summary(run, cfg, summary, output_df)
    return summary
