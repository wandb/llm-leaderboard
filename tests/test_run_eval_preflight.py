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


def run_preflight(
    config: Path,
    base_config: Path,
    output_json: Path,
    extra_env: dict[str, str | None] | None = None,
) -> subprocess.CompletedProcess:
    env = os.environ.copy()
    env["PYTHONPATH"] = str(REPO_ROOT / "scripts")
    env["NEJUMI_DISABLE_DOTENV"] = "1"
    env.setdefault("OPENAI_API_KEY", "test-openai-key")
    env.pop("NEJUMI_MAIN_STARTED", None)
    if extra_env:
        for key, value in extra_env.items():
            if value is None:
                env.pop(key, None)
            else:
                env[key] = value
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
          deepswe: false
          agentic_swe_assorted: false
        agentic_math:
          max_tokens: 2048
        swebench_pro:
          max_tokens: 2048
        deepswe:
          max_tokens: 2048
        agentic_swe_assorted:
          max_tokens: 2048
          run_openclaw: false
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
          agentic_swe_assorted: true
        """,
    )

    result = run_preflight(config, base_config, output_json)

    assert result.returncode == 0, result.stderr + result.stdout
    payload = json.loads(output_json.read_text(encoding="utf-8"))
    assert payload["ok"] is True
    assert payload["status"] == "passed"
    assert payload["model"] == "test-model"
    assert payload["wandb"]["run_name"] == "preflight-agentic"
    assert payload["enabled_benchmarks"] == ["agentic_math", "agentic_swe_assorted"]
    assert payload["will_initialize_wandb"] is False
    assert payload["will_log_wandb_artifacts"] is False
    assert payload["will_initialize_weave"] is False
    assert payload["will_start_inference_engine"] is False
    assert payload["will_run_evaluators"] is False
    assert "Wandb API key loaded" not in result.stdout
    assert "Warning: WANDB_API_KEY" not in result.stdout
    assert "config_singleton not available" not in result.stderr


def test_run_eval_preflight_rejects_taiwan_run_without_scope_contract(tmp_path):
    base_config = tmp_path / "base.yaml"
    config = tmp_path / "config.yaml"
    output_json = tmp_path / "preflight.json"

    write_yaml(
        base_config,
        """
        wandb:
          entity: llm-leaderboard
          project: nejumi-leaderboard4
          run_name: default-run
        api: openai_responses
        model:
          pretrained_model_name_or_path: test-model
        generator:
          max_tokens: 2048
        run:
          agentic_math: false
        """,
    )
    write_yaml(
        config,
        """
        wandb:
          run_name: taiwan/full/test-model
        """,
    )

    result = run_preflight(config, base_config, output_json)

    assert result.returncode == 2
    payload = json.loads(output_json.read_text(encoding="utf-8"))
    scope = payload["runtime_validation"]["wandb_scope"]
    assert scope["ok"] is False
    assert scope["project"] == "nejumi-leaderboard4"
    assert "must declare wandb.expected_entity" in "\n".join(scope["errors"])
    assert "wandb_scope_contract: failed" in result.stdout


def test_run_eval_preflight_rejects_wandb_scope_mismatch(tmp_path):
    base_config = tmp_path / "base.yaml"
    config = tmp_path / "config.yaml"
    output_json = tmp_path / "preflight.json"

    write_yaml(
        base_config,
        """
        wandb:
          entity: llm-leaderboard
          project: nejumi-leaderboard4
          run_name: default-run
        api: openai_responses
        model:
          pretrained_model_name_or_path: test-model
        generator:
          max_tokens: 2048
        run:
          agentic_math: false
        """,
    )
    write_yaml(
        config,
        """
        wandb:
          run_name: taiwan/full/test-model
          expected_entity: llm-leaderboard
          expected_project: tc-leaderboard
        """,
    )

    result = run_preflight(config, base_config, output_json)

    assert result.returncode == 2
    payload = json.loads(output_json.read_text(encoding="utf-8"))
    scope = payload["runtime_validation"]["wandb_scope"]
    assert scope["ok"] is False
    assert scope["expected_project"] == "tc-leaderboard"
    assert "project scope mismatch" in "\n".join(scope["errors"])


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
          deepswe: false
          agentic_swe_assorted: false
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
        deepswe:
          max_tokens: 2048
        agentic_swe_assorted:
          max_tokens: 2048
          run_openclaw: false
        """,
    )
    write_yaml(
        config,
        """
        run:
          agentic_math: true
          bfcl: true
          agentic_swe_assorted: true
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
        "agentic_swe_assorted",
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


def _write_agentic_swe_static_preflight_configs(
    tmp_path: Path,
    *,
    high_input_cap: int,
) -> tuple[Path, Path, Path]:
    base_config = tmp_path / "base.yaml"
    config = tmp_path / "config.yaml"
    output_json = tmp_path / "preflight.json"
    low_middle = REPO_ROOT / (
        "data/taiwan/swebench_lite_assorted/subsets/low_middle_v3_40.jsonl"
    )
    low_middle_ids = REPO_ROOT / (
        "data/taiwan/swebench_lite_assorted/subsets/"
        "low_middle_v3_40_instance_ids.json"
    )
    high_metadata = REPO_ROOT / (
        "data/taiwan/deepswe/subsets/"
        "essential_anchored_high_10_model_fidelity_cost_balanced.jsonl"
    )
    high_names = REPO_ROOT / (
        "data/taiwan/deepswe/subsets/"
        "essential_anchored_high_10_model_fidelity_cost_balanced_task_names.json"
    )
    public_trials = REPO_ROOT / "outputs/deepswe_subset_analysis/deepswe_v1_1_trials.json"

    write_yaml(
        base_config,
        f"""
        wandb:
          entity: llm-leaderboard
          project: tc-leaderboard
          run_name: agentic-swe-static-preflight
        api: openai_responses
        model:
          pretrained_model_name_or_path: gpt-4.1-mini-2025-04-14
        generator:
          max_tokens: 2048
        run:
          agentic_swe_assorted: false
        agentic_swe_assorted:
          max_tokens: 2048
          run_openclaw: true
          output_dir: '{tmp_path / "agentic_swe"}'
          thinking: off
          nemoclaw_sandbox: nejumi-taiwan
          low_middle_jsonl: '{low_middle}'
          low_middle_instance_ids_json: '{low_middle_ids}'
          low_limit: 1
          middle_limit: 1
          deepswe_metadata_jsonl: '{high_metadata}'
          deepswe_task_names_file: '{high_names}'
          deepswe_public_trials_json: '{public_trials}'
          high_limit: 10
          deepswe_budget_preflight: error
          deepswe_preflight_hard_stat: p90
          high_max_agent_turns: 150
          high_max_cumulative_input_tokens: {high_input_cap}
          no_docker_check: true
        """,
    )
    write_yaml(
        config,
        """
        run:
          agentic_swe_assorted: true
        """,
    )
    return base_config, config, output_json


def test_run_eval_preflight_blocks_agentic_swe_budget_mismatch_before_execution(
    tmp_path,
):
    base_config, config, output_json = _write_agentic_swe_static_preflight_configs(
        tmp_path,
        high_input_cap=13_000_000,
    )

    result = run_preflight(config, base_config, output_json)

    assert result.returncode == 2
    payload = json.loads(output_json.read_text(encoding="utf-8"))
    runtime = payload["runtime_validation"]
    assert runtime["ok"] is False
    assert len(runtime["benchmark_preflights"]) == 1
    assert runtime["benchmark_preflights"][0]["ok"] is False
    assert any(
        "13829502" in error and "13000000" in error
        for error in runtime["errors"]
    )
    assert payload["will_initialize_wandb"] is False
    assert payload["will_run_evaluators"] is False


def test_run_eval_preflight_accepts_frozen_high10_with_14m_cap(tmp_path):
    base_config, config, output_json = _write_agentic_swe_static_preflight_configs(
        tmp_path,
        high_input_cap=14_000_000,
    )

    result = run_preflight(config, base_config, output_json)

    assert result.returncode == 0, result.stderr + result.stdout
    payload = json.loads(output_json.read_text(encoding="utf-8"))
    runtime = payload["runtime_validation"]
    assert runtime["ok"] is True
    [benchmark_preflight] = runtime["benchmark_preflights"]
    assert benchmark_preflight["ok"] is True
    assert benchmark_preflight["report"]["selected_counts"] == {
        "low_middle": 2,
        "low": 1,
        "middle": 1,
        "high": 10,
    }
    assert benchmark_preflight["report"]["will_run_model"] is False
    assert benchmark_preflight["report"]["will_run_gateway"] is False
    assert benchmark_preflight["report"]["will_run_grading"] is False


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
          mtbench: false
        mtbench:
          generator_config:
            max_tokens: 0
        """,
    )
    write_yaml(
        config,
        """
        run:
          mtbench: true
        """,
    )

    result = run_preflight(config, base_config, output_json)

    assert result.returncode == 2
    payload = json.loads(output_json.read_text(encoding="utf-8"))
    assert payload["ok"] is False
    assert payload["status"] == "failed"
    assert payload["token_validation"]["has_errors"] is True
    assert payload["will_initialize_wandb"] is False


def test_run_eval_preflight_fails_for_api_twbias_hf_perplexity_without_model_path(tmp_path):
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
          pretrained_model_name_or_path: gpt-4.1-mini-2025-04-14
        generator:
          max_tokens: 2048
        run:
          twbias: false
        twbias:
          backend: hf_perplexity
          model_path: null
          allow_unknown_license: true
        """,
    )
    write_yaml(
        config,
        """
        run:
          twbias: true
        """,
    )

    result = run_preflight(config, base_config, output_json)

    assert result.returncode == 2
    payload = json.loads(output_json.read_text(encoding="utf-8"))
    assert payload["ok"] is False
    assert payload["runtime_validation"]["ok"] is False
    assert "TWBias hf_perplexity requires direct HF/local model access" in (
        payload["runtime_validation"]["errors"][0]
    )
    assert payload["will_initialize_wandb"] is False


def test_run_eval_preflight_fails_before_execution_when_openrouter_key_missing(tmp_path):
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
        api: openai-compatible
        base_url: https://openrouter.ai/api/v1
        model:
          pretrained_model_name_or_path: z-ai/glm-5.2
        generator:
          max_tokens: 2048
        run:
          bfcl: false
        bfcl:
          max_tokens: 2048
        """,
    )
    write_yaml(
        config,
        """
        run:
          bfcl: true
        """,
    )

    result = run_preflight(
        config,
        base_config,
        output_json,
        extra_env={
            "NEJUMI_DISABLE_DOTENV": "1",
            "OPENROUTER_API_KEY": None,
            "OPENAI_COMPATIBLE_API_KEY": None,
            "NEJUMI_OPENROUTER_API_KEY_ENV": None,
        },
    )

    assert result.returncode == 2
    payload = json.loads(output_json.read_text(encoding="utf-8"))
    assert payload["ok"] is False
    assert payload["runtime_validation"]["ok"] is False
    assert "Missing credential for answer model OpenRouter API" in (
        payload["runtime_validation"]["errors"][0]
    )
    assert payload["runtime_validation"]["credential_checks"][0]["required_any_of"] == [
        "OPENROUTER_API_KEY",
        "OPENAI_COMPATIBLE_API_KEY",
    ]
    assert payload["will_initialize_wandb"] is False


def test_run_eval_preflight_passes_when_openrouter_key_present(tmp_path):
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
        api: openai-compatible
        base_url: https://openrouter.ai/api/v1
        model:
          pretrained_model_name_or_path: z-ai/glm-5.2
        generator:
          max_tokens: 2048
        run:
          bfcl: false
        bfcl:
          max_tokens: 2048
        """,
    )
    write_yaml(
        config,
        """
        run:
          bfcl: true
        """,
    )

    result = run_preflight(
        config,
        base_config,
        output_json,
        extra_env={
            "NEJUMI_DISABLE_DOTENV": "1",
            "OPENROUTER_API_KEY": "test-openrouter-key",
            "OPENAI_COMPATIBLE_API_KEY": None,
            "NEJUMI_OPENROUTER_API_KEY_ENV": None,
        },
    )

    assert result.returncode == 0, result.stderr + result.stdout
    payload = json.loads(output_json.read_text(encoding="utf-8"))
    assert payload["ok"] is True
    assert payload["runtime_validation"]["ok"] is True
    assert payload["runtime_validation"]["credential_checks"][0]["present_envs"] == [
        "OPENROUTER_API_KEY"
    ]


def test_run_eval_preflight_fails_before_execution_when_wandb_inference_key_missing(tmp_path):
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
        api: openai-compatible
        base_url: https://api.inference.wandb.ai/v1
        model:
          pretrained_model_name_or_path: meta-llama/Llama-3.3-70B-Instruct
        generator:
          max_tokens: 2048
        run:
          bfcl: false
        bfcl:
          max_tokens: 2048
        """,
    )
    write_yaml(
        config,
        """
        run:
          bfcl: true
        """,
    )

    result = run_preflight(
        config,
        base_config,
        output_json,
        extra_env={
            "NEJUMI_DISABLE_DOTENV": "1",
            "WANDB_API_KEY": None,
            "OPENAI_COMPATIBLE_API_KEY": None,
        },
    )

    assert result.returncode == 2
    payload = json.loads(output_json.read_text(encoding="utf-8"))
    assert payload["ok"] is False
    assert payload["runtime_validation"]["ok"] is False
    assert "Missing credential for answer model W&B Inference API" in (
        payload["runtime_validation"]["errors"][0]
    )
    assert payload["runtime_validation"]["credential_checks"][0]["required_any_of"] == [
        "WANDB_API_KEY",
        "OPENAI_COMPATIBLE_API_KEY",
    ]
    assert payload["will_initialize_wandb"] is False


def test_run_eval_preflight_passes_when_wandb_inference_key_present(tmp_path):
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
        api: openai-compatible
        base_url: https://api.inference.wandb.ai/v1
        model:
          pretrained_model_name_or_path: meta-llama/Llama-3.3-70B-Instruct
        generator:
          max_tokens: 2048
        run:
          bfcl: false
        bfcl:
          max_tokens: 2048
        """,
    )
    write_yaml(
        config,
        """
        run:
          bfcl: true
        """,
    )

    result = run_preflight(
        config,
        base_config,
        output_json,
        extra_env={
            "NEJUMI_DISABLE_DOTENV": "1",
            "WANDB_API_KEY": "test-wandb-key",
            "OPENAI_COMPATIBLE_API_KEY": None,
        },
    )

    assert result.returncode == 0, result.stderr + result.stdout
    payload = json.loads(output_json.read_text(encoding="utf-8"))
    assert payload["ok"] is True
    assert payload["runtime_validation"]["ok"] is True
    assert payload["runtime_validation"]["credential_checks"][0]["present_envs"] == [
        "WANDB_API_KEY"
    ]


def _write_bfcl_v4_web_preflight_configs(
    tmp_path: Path,
    *,
    backend: str,
) -> tuple[Path, Path, Path]:
    base_config = tmp_path / "base.yaml"
    config = tmp_path / "config.yaml"
    output_json = tmp_path / "preflight.json"
    write_yaml(
        base_config,
        f"""
        wandb:
          entity: llm-leaderboard
          project: tc-leaderboard
          run_name: bfcl-v4-web-preflight
        api: openai_responses
        model:
          pretrained_model_name_or_path: gpt-4.1-mini-2025-04-14
        generator:
          max_tokens: 2048
        run:
          bfcl: false
        bfcl:
          version: v4
          test_category: web_search_base
          web_search:
            backend: {backend}
        """,
    )
    write_yaml(config, "run:\n  bfcl: true")
    return base_config, config, output_json


def test_bfcl_v4_direct_search_preflight_does_not_require_serpapi(
    tmp_path,
):
    base_config, config, output_json = (
        _write_bfcl_v4_web_preflight_configs(
            tmp_path,
            backend="duckduckgo_html",
        )
    )
    fake_modules = tmp_path / "fake_modules"
    fake_modules.mkdir()
    (fake_modules / "html2text.py").write_text("", encoding="utf-8")

    result = run_preflight(
        config,
        base_config,
        output_json,
        extra_env={
            "PYTHONPATH": (
                f"{REPO_ROOT / 'scripts'}{os.pathsep}{fake_modules}"
            ),
            "SERPAPI_API_KEY": None,
        },
    )

    assert result.returncode == 0, result.stderr + result.stdout
    payload = json.loads(output_json.read_text(encoding="utf-8"))
    assert payload["runtime_validation"]["ok"] is True
    assert all(
        check["label"] != "BFCL v4 SerpAPI web search"
        for check in payload["runtime_validation"]["credential_checks"]
    )


def test_bfcl_v4_serpapi_preflight_requires_key(tmp_path):
    base_config, config, output_json = (
        _write_bfcl_v4_web_preflight_configs(
            tmp_path,
            backend="serpapi",
        )
    )
    fake_modules = tmp_path / "fake_modules"
    fake_modules.mkdir()
    for module in ("html2text", "serpapi"):
        (fake_modules / f"{module}.py").write_text("", encoding="utf-8")

    result = run_preflight(
        config,
        base_config,
        output_json,
        extra_env={
            "PYTHONPATH": (
                f"{REPO_ROOT / 'scripts'}{os.pathsep}{fake_modules}"
            ),
            "SERPAPI_API_KEY": None,
        },
    )

    assert result.returncode == 2
    payload = json.loads(output_json.read_text(encoding="utf-8"))
    assert payload["runtime_validation"]["ok"] is False
    assert any(
        "Missing credential for BFCL v4 SerpAPI web search" in error
        for error in payload["runtime_validation"]["errors"]
    )


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
