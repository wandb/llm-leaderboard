import json
import os
import subprocess
import sys
import ast
from textwrap import dedent
from pathlib import Path
import importlib.util


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


def test_run_eval_preflight_schedules_taiwan_full_evaluators(tmp_path):
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
          bfcl: false
          swebench: false
          swebench_pro: false
          mtbench: false
          jbbq: false
          toxicity: false
          jtruthfulqa: false
          hle: false
          hallulens: false
          hallulens_zh_tw: false
          arc_agi: false
          m_ifeval: false
          ifeval_zh_tw: false
          ts_bench: false
          twbias: false
          tceval_v2: false
          script_adherence: false
          jaster: false
          jmmlu_robustness: false
          tmmluplus_robustness: false
          aggregate: false
          aggregate_taiwan: false
        agentic_math:
          max_tokens: 2048
        swebench_pro:
          max_tokens: 2048
        """,
    )
    write_yaml(
        config,
        """
        run:
          agentic_math: true
          bfcl: true
          swebench_pro: true
          mtbench: true
          hle: true
          hallulens_zh_tw: true
          arc_agi: true
          ifeval_zh_tw: true
          ts_bench: true
          tceval_v2: true
          script_adherence: true
          jaster: true
          tmmluplus_robustness: true
          aggregate_taiwan: true
        """,
    )

    result = run_preflight(config, base_config, output_json)

    assert result.returncode == 0, result.stderr + result.stdout
    payload = json.loads(output_json.read_text(encoding="utf-8"))
    assert payload["scheduled_evaluators"] == [
        "bfcl",
        "agentic_math",
        "swebench_pro",
        "mtbench",
        "script_adherence",
        "hle",
        "hallulens_zh_tw",
        "arc_agi",
        "ifeval_zh_tw",
        "ts_bench",
        "tceval_v2",
        "jaster",
        "aggregate_taiwan",
    ]
    assert payload["dispatch_validation"]["auxiliary_run_flags"] == ["tmmluplus_robustness"]
    assert payload["dispatch_validation"]["unsupported_truthy_run_flags"] == []


def test_run_eval_preflight_fails_on_unknown_truthy_run_flag(tmp_path):
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
        agentic_math:
          max_tokens: 2048
        """,
    )
    write_yaml(
        config,
        """
        run:
          imaginary_taiwan_benchmark: true
        """,
    )

    result = run_preflight(config, base_config, output_json)

    assert result.returncode == 2
    payload = json.loads(output_json.read_text(encoding="utf-8"))
    assert payload["ok"] is False
    assert payload["status"] == "failed"
    assert payload["dispatch_validation"]["unsupported_truthy_run_flags"] == [
        "imaginary_taiwan_benchmark"
    ]
    assert payload["will_initialize_wandb"] is False


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


def _literal_assignment_from_source(path: Path, name: str):
    tree = ast.parse(path.read_text(encoding="utf-8"))
    for node in tree.body:
        if isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name) and target.id == name:
                    return ast.literal_eval(node.value)
        if isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name) and node.target.id == name:
            return ast.literal_eval(node.value)
    raise AssertionError(f"{name} not found in {path}")


def test_taiwan_run_flags_are_known_to_run_eval_dispatch_contract():
    tools_dir = REPO_ROOT / "scripts" / "tools"
    if str(tools_dir) not in sys.path:
        sys.path.insert(0, str(tools_dir))
    path = tools_dir / "prepare_taiwan_full_eval_configs.py"
    spec = importlib.util.spec_from_file_location(path.stem, path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)

    run_eval_path = REPO_ROOT / "scripts" / "run_eval.py"
    benchmark_map = _literal_assignment_from_source(run_eval_path, "BENCHMARK_MAP")
    auxiliary = _literal_assignment_from_source(run_eval_path, "AUXILIARY_RUN_FLAGS")
    known = set(benchmark_map) | set(auxiliary)

    assert set(module.RUN_FLAGS).issubset(known)
