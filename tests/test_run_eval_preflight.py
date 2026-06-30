import json
import os
import subprocess
import sys
from textwrap import dedent
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]


def write_yaml(path: Path, text: str) -> None:
    path.write_text(dedent(text).strip() + "\n", encoding="utf-8")


def run_preflight(config: Path, base_config: Path, output_json: Path) -> subprocess.CompletedProcess:
    env = os.environ.copy()
    env["PYTHONPATH"] = str(REPO_ROOT / "scripts")
    env.pop("NEJUMI_MAIN_STARTED", None)
    return subprocess.run(
        [
            sys.executable,
            "scripts/run_eval.py",
            "--config",
            str(config),
            "--base-config",
            str(base_config),
            "--preflight",
            "--preflight-json",
            str(output_json),
        ],
        cwd=REPO_ROOT,
        env=env,
        text=True,
        capture_output=True,
        check=False,
    )


def test_run_eval_preflight_writes_no_execution_payload(tmp_path):
    base_config = tmp_path / "base.yaml"
    config = tmp_path / "config.yaml"
    output_json = tmp_path / "preflight.json"

    write_yaml(
        base_config,
        """
        wandb:
          entity: llm-leaderboard
          project: tc-leaderboard
          run_name: preflight-base
        api: openai_responses
        model:
          pretrained_model_name_or_path: base-model
        generator:
          max_tokens: 2048
        run:
          agentic_math: false
          swebench_pro: false
        agentic_math:
          max_tokens: 2048
        swebench_pro:
          max_tokens: 2048
        """,
    )
    write_yaml(
        config,
        """
        wandb:
          run_name: preflight-agentic
        model:
          pretrained_model_name_or_path: test-model
        run:
          agentic_math: true
          swebench_pro: true
        """,
    )

    result = run_preflight(config, base_config, output_json)

    assert result.returncode == 0, result.stderr + result.stdout
    payload = json.loads(output_json.read_text(encoding="utf-8"))
    assert payload["ok"] is True
    assert payload["status"] == "passed"
    assert payload["model"] == "test-model"
    assert payload["wandb"]["run_name"] == "preflight-agentic"
    assert payload["enabled_benchmarks"] == ["agentic_math", "swebench_pro"]
    assert payload["will_initialize_wandb"] is False
    assert payload["will_log_wandb_artifacts"] is False
    assert payload["will_initialize_weave"] is False
    assert payload["will_start_inference_engine"] is False
    assert payload["will_run_evaluators"] is False
    assert "Wandb API key loaded" not in result.stdout
    assert "Warning: WANDB_API_KEY" not in result.stdout
    assert "config_singleton not available" not in result.stderr


def test_run_eval_preflight_fails_on_critical_token_validation(tmp_path):
    base_config = tmp_path / "base.yaml"
    config = tmp_path / "config.yaml"
    output_json = tmp_path / "preflight.json"

    write_yaml(
        base_config,
        """
        wandb:
          entity: llm-leaderboard
          project: tc-leaderboard
          run_name: preflight-base
        api: openai_responses
        model:
          pretrained_model_name_or_path: base-model
        generator:
          max_tokens: 0
        run:
          agentic_math: false
        agentic_math:
          max_tokens: 0
        """,
    )
    write_yaml(
        config,
        """
        run:
          agentic_math: true
        """,
    )

    result = run_preflight(config, base_config, output_json)

    assert result.returncode == 2
    payload = json.loads(output_json.read_text(encoding="utf-8"))
    assert payload["ok"] is False
    assert payload["status"] == "failed"
    assert payload["token_validation"]["has_errors"] is True
    assert payload["will_initialize_wandb"] is False
