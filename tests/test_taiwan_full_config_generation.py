import argparse
import sys
from pathlib import Path

from omegaconf import OmegaConf
import pytest

ROOT = Path(__file__).resolve().parents[1]
TOOLS_DIR = ROOT / "scripts" / "tools"
OPENAI_CANARY_MANIFEST = ROOT / "configs" / "taiwan_openai_canary_models.yaml"
FULL_EVAL_MANIFEST = ROOT / "configs" / "taiwan_full_eval_models.yaml"
BASE_TAIWAN_CONFIG = ROOT / "configs" / "base_config_taiwan.yaml"
if str(TOOLS_DIR) not in sys.path:
    sys.path.insert(0, str(TOOLS_DIR))

from prepare_taiwan_full_eval_configs import (
    DEFAULT_TAIWAN_NEMOCLAW_SANDBOX,
    DEFAULT_NEMOCLAW_OPENCLAW_CONFIG_PATH,
    build_override,
    generate_configs,
    select_models,
)


def _minimal_manifest(tmp_path: Path) -> Path:
    path = tmp_path / "models.yaml"
    path.write_text(
        """
models:
  - slug: gpt-4_1-mini-openai-direct-canary
    source_config: config-gpt-4.1-mini-2025-04-14.yaml
    run_name: "taiwan/full/openai/gpt-4.1-mini: production-spec-one-model"
    japanese_runname: "openai/gpt-4-1-mini-2025-04-14"
    openclaw_model: "openai-direct/gpt-4.1-mini-2025-04-14"
    reasoning_effort: null
    agentic_thinking: "off"
    swe_thinking: "off"
    canary: true
""".lstrip(),
        encoding="utf-8",
    )
    return path


def _args(tmp_path: Path, phase: str, *, manifest: Path | None = None) -> argparse.Namespace:
    return argparse.Namespace(
        manifest=manifest or _minimal_manifest(tmp_path),
        output_dir=tmp_path / "generated",
        output_root=tmp_path / "outputs" / "taiwan_full_eval",
        model=["gpt-4_1-mini-openai-direct-canary"],
        phase=phase,
        canary=False,
        include_final_only=False,
        include_suspended=False,
        agentic_math_nemoclaw_sandbox=None,
        agentic_math_nemoclaw_bin="nemoclaw",
        agentic_math_nemoclaw_workdir="/sandbox",
        agentic_math_nemoclaw_openclaw_config_path=None,
        swebench_pro_nemoclaw_sandbox=None,
        swebench_pro_nemoclaw_bin="nemoclaw",
        swebench_pro_nemoclaw_checkout_sandbox_root=None,
        swebench_pro_nemoclaw_checkout_transfer_mode=None,
        swebench_pro_nemoclaw_workdir=None,
        swebench_pro_nemoclaw_openclaw_config_path=None,
        deepswe_nemoclaw_sandbox=None,
        deepswe_nemoclaw_bin="nemoclaw",
        deepswe_nemoclaw_workdir="/sandbox",
        deepswe_nemoclaw_openclaw_config_path=None,
        agentic_swe_assorted_nemoclaw_sandbox=None,
        agentic_swe_assorted_nemoclaw_bin="nemoclaw",
        agentic_swe_assorted_nemoclaw_workdir="/sandbox",
        agentic_swe_assorted_nemoclaw_checkout_transfer_mode=None,
        agentic_swe_assorted_nemoclaw_openclaw_config_path=None,
    )


def test_nonagentic_phase_skips_agentic_and_aggregate(tmp_path):
    [config_path] = generate_configs(_args(tmp_path, "nonagentic"))
    cfg = OmegaConf.load(config_path)

    assert cfg.run.agentic_math is False
    assert cfg.run.swebench_pro is False
    assert cfg.run.deepswe is False
    assert cfg.run.agentic_swe_assorted is False
    assert cfg.run.aggregate_taiwan is False
    assert cfg.run.bfcl is True
    assert cfg.run.mtbench is True
    assert cfg.run.hle is True
    resolved = OmegaConf.merge(OmegaConf.load(BASE_TAIWAN_CONFIG), cfg)
    assert resolved.mtbench.judge.model == "gpt-5.5-2026-04-23"
    assert resolved.hle.judge.model == "gpt-5.5-2026-04-23"
    assert resolved.hallulens_zh_tw.judge.model == "gpt-5.5-2026-04-23"


def test_agentic_aggregate_phase_reuses_completed_outputs(tmp_path):
    [config_path] = generate_configs(_args(tmp_path, "agentic_aggregate"))
    cfg = OmegaConf.load(config_path)
    output_root = tmp_path / "outputs" / "taiwan_full_eval"
    slug = "gpt-4_1-mini-openai-direct-canary"

    assert cfg.run.agentic_math is True
    assert cfg.run.swebench_pro is False
    assert cfg.run.deepswe is False
    assert cfg.run.agentic_swe_assorted is True
    assert cfg.run.aggregate_taiwan is True
    assert cfg.run.bfcl is False
    assert cfg.run.mtbench is False
    assert cfg.agentic_math.run_openclaw is False
    assert Path(cfg.agentic_math.results_dir) == output_root / "agentic_math" / slug / "openclaw"
    assert cfg.agentic_swe_assorted.run_openclaw is False
    assert Path(cfg.agentic_swe_assorted.results_dir) == (
        output_root / "agentic_swe_assorted" / slug / "runner"
    )


def test_agentic_phase_runs_only_agentic_generation(tmp_path):
    [config_path] = generate_configs(_args(tmp_path, "agentic"))
    cfg = OmegaConf.load(config_path)

    assert cfg.run.agentic_math is True
    assert cfg.run.swebench_pro is False
    assert cfg.run.deepswe is False
    assert cfg.run.agentic_swe_assorted is True
    assert cfg.run.aggregate_taiwan is False
    assert cfg.run.bfcl is False
    assert cfg.run.mtbench is False
    assert cfg.provider_rate_limit.enabled is True
    assert cfg.provider_rate_limit.key == "llm:gpt-4_1-mini-openai-direct-canary"
    assert cfg.provider_rate_limit.min_request_interval_sec == 1.0
    assert cfg.provider_rate_limit.request_jitter_sec == 0.25
    assert cfg.agentic_math.run_openclaw is True
    assert cfg.wandb.entity == "llm-leaderboard"
    assert cfg.wandb.project == "tc-leaderboard"
    assert cfg.wandb.expected_entity == "llm-leaderboard"
    assert cfg.wandb.expected_project == "tc-leaderboard"
    assert cfg.swebench_pro.run_openclaw is True
    assert cfg.deepswe.run_openclaw is True
    assert cfg.agentic_math.results_dir is None
    assert cfg.swebench_pro.patch_path is None
    assert cfg.deepswe.results_dir is None
    assert cfg.agentic_swe_assorted.results_dir is None
    assert cfg.agentic_math.limit == 50
    assert cfg.agentic_math.num_workers == 8
    assert cfg.agentic_math.task_start_min_interval_seconds == 5.0
    assert cfg.agentic_math.max_input_tokens == 500_000
    assert cfg.agentic_math.max_tool_calls == 40
    assert cfg.agentic_math.max_agent_turns == 40
    assert cfg.agentic_math.max_tool_wall_seconds == 120
    assert cfg.agentic_swe_assorted.low_limit == 20
    assert cfg.agentic_swe_assorted.middle_limit == 20
    assert cfg.agentic_swe_assorted.high_limit == 10
    assert cfg.agentic_swe_assorted.tier_weights == "low=1,middle=1,high=1"
    assert cfg.agentic_swe_assorted.reuse_high_results_jsonl == []
    assert cfg.agentic_swe_assorted.reuse_low_middle_patches_json is None
    assert cfg.agentic_swe_assorted.swe_workers == 4
    assert cfg.agentic_swe_assorted.high_workers == 2
    assert cfg.agentic_swe_assorted.max_input_tokens == 1_000_000
    assert cfg.agentic_swe_assorted.max_cumulative_input_tokens == 1_000_000
    assert cfg.agentic_swe_assorted.max_tool_calls == 40
    assert cfg.agentic_swe_assorted.max_agent_turns == 40
    assert cfg.agentic_swe_assorted.high_max_tool_calls == 200
    assert cfg.agentic_swe_assorted.high_max_agent_turns == 150
    assert cfg.agentic_swe_assorted.high_max_cumulative_input_tokens == 14_000_000
    assert cfg.agentic_swe_assorted.deepswe_preflight_hard_stat == "p90"
    assert cfg.agentic_swe_assorted.max_tool_wall_seconds == 120
    assert cfg.bfcl.num_threads == 4
    assert cfg.bfcl.provider_min_request_interval_sec == 2.0
    assert cfg.bfcl.provider_request_jitter_sec == 0.5
    assert cfg.bfcl.consecutive_failure_fail_fast == 5
    assert cfg.execution.cash_cost_exempt is False
    assert cfg.execution.cash_cost_exempt_reason == ""


def test_cash_cost_exemption_is_propagated_from_manifest(tmp_path):
    override = build_override(
        {
            "slug": "wandb-free-model",
            "run_name": "wandb-free-model",
            "openclaw_model": "wandb-inference/provider/model",
            "cash_cost_exempt": True,
            "cash_cost_exempt_reason": "employee account has no cash charge",
        },
        tmp_path / "outputs",
        phase="full",
    )

    assert override["execution"] == {
        "cash_cost_exempt": True,
        "cash_cost_exempt_reason": "employee account has no cash charge",
    }


def test_full_phase_supports_verified_math_and_bfcl_recovery_sources(tmp_path):
    math_results_dir = tmp_path / "previous" / "agentic_math" / "openclaw"
    bfcl_result_dir = tmp_path / "previous" / "bfcl" / "result"

    override = build_override(
        {
            "slug": "recovery-model",
            "run_name": "recovery-model",
            "openclaw_model": "openai-direct/test-model",
            "agentic_math_run_openclaw": False,
            "agentic_math_results_dir": str(math_results_dir),
            "bfcl_run_generation": False,
            "bfcl_result_dir": str(bfcl_result_dir),
        },
        tmp_path / "outputs",
        phase="full",
    )

    assert override["agentic_math"]["run_openclaw"] is False
    assert Path(override["agentic_math"]["results_dir"]) == math_results_dir
    assert override["bfcl"]["run_generation"] is False
    assert Path(override["bfcl"]["result_dir"]) == bfcl_result_dir


def test_bfcl_recovery_requires_an_explicit_result_directory(tmp_path):
    with pytest.raises(ValueError, match="bfcl_result_dir"):
        build_override(
            {
                "slug": "broken-recovery-model",
                "run_name": "broken-recovery-model",
                "openclaw_model": "openai-direct/test-model",
                "bfcl_run_generation": False,
            },
            tmp_path / "outputs",
            phase="full",
        )


def test_taiwan_base_uses_bfcl_v4_full_profile():
    cfg = OmegaConf.load(BASE_TAIWAN_CONFIG)

    assert cfg.bfcl.version == "v4"
    assert cfg.bfcl.profile == "full"
    assert set(str(cfg.bfcl.test_category).split()) == {
        "simple_python",
        "simple_java",
        "simple_javascript",
        "multiple",
        "irrelevance",
        "live_simple",
        "live_multiple",
        "live_irrelevance",
        "live_relevance",
        "multi_turn_base",
        "multi_turn_miss_func",
        "multi_turn_miss_param",
        "memory_kv",
        "memory_vector",
        "memory_rec_sum",
        "web_search_base",
        "web_search_no_snippet",
    }
    assert cfg.bfcl.web_search.backend == "ddgs"


def test_agentic_math_nemoclaw_cli_override_is_agentic_math_only(tmp_path):
    args = _args(tmp_path, "agentic")
    args.agentic_math_nemoclaw_sandbox = "nejumi-taiwan"
    args.agentic_math_nemoclaw_workdir = "/workspace"

    [config_path] = generate_configs(args)
    cfg = OmegaConf.load(config_path)

    assert cfg.agentic_math.nemoclaw_sandbox == "nejumi-taiwan"
    assert cfg.agentic_math.nemoclaw_bin == "nemoclaw"
    assert cfg.agentic_math.nemoclaw_workdir == "/workspace"
    assert cfg.agentic_math.nemoclaw_openclaw_config_path == DEFAULT_NEMOCLAW_OPENCLAW_CONFIG_PATH
    assert cfg.agentic_math.use_task_agent is True
    assert "nemoclaw_sandbox" not in cfg.swebench_pro
    assert "nemoclaw_sandbox" not in cfg.deepswe


def test_swebench_pro_nemoclaw_cli_override_is_swe_only(tmp_path):
    args = _args(tmp_path, "agentic")
    args.swebench_pro_nemoclaw_sandbox = "nejumi-taiwan"
    args.swebench_pro_nemoclaw_checkout_sandbox_root = "/sandbox/checkouts"
    args.swebench_pro_nemoclaw_checkout_transfer_mode = "copy"

    [config_path] = generate_configs(args)
    cfg = OmegaConf.load(config_path)

    assert cfg.swebench_pro.nemoclaw_sandbox == "nejumi-taiwan"
    assert cfg.swebench_pro.nemoclaw_bin == "nemoclaw"
    assert cfg.swebench_pro.nemoclaw_checkout_sandbox_root == "/sandbox/checkouts"
    assert cfg.swebench_pro.nemoclaw_checkout_transfer_mode == "copy"
    assert cfg.swebench_pro.nemoclaw_openclaw_config_path == DEFAULT_NEMOCLAW_OPENCLAW_CONFIG_PATH
    assert cfg.agentic_math.nemoclaw_sandbox == DEFAULT_TAIWAN_NEMOCLAW_SANDBOX
    assert "nemoclaw_sandbox" not in cfg.deepswe


def test_nemoclaw_cli_openclaw_config_path_override(tmp_path):
    args = _args(tmp_path, "agentic")
    args.agentic_math_nemoclaw_sandbox = "nejumi-taiwan"
    args.agentic_math_nemoclaw_openclaw_config_path = "/custom/math/openclaw.json"
    args.swebench_pro_nemoclaw_sandbox = "nejumi-taiwan"
    args.swebench_pro_nemoclaw_openclaw_config_path = "/custom/swe/openclaw.json"
    args.deepswe_nemoclaw_sandbox = "nejumi-taiwan"
    args.deepswe_nemoclaw_openclaw_config_path = "/custom/deepswe/openclaw.json"

    [config_path] = generate_configs(args)
    cfg = OmegaConf.load(config_path)

    assert cfg.agentic_math.nemoclaw_openclaw_config_path == "/custom/math/openclaw.json"
    assert cfg.swebench_pro.nemoclaw_openclaw_config_path == "/custom/swe/openclaw.json"
    assert cfg.deepswe.nemoclaw_openclaw_config_path == "/custom/deepswe/openclaw.json"


def test_agentic_math_nemoclaw_manifest_override_does_not_touch_swebench_pro(tmp_path):
    override = build_override(
        {
            "slug": "model-a",
            "run_name": "model-a",
            "openclaw_model": "provider/model-a",
            "agentic_math_nemoclaw_sandbox": "nejumi-taiwan",
            "agentic_math_nemoclaw_bin": "/usr/local/bin/nemoclaw",
            "agentic_math_nemoclaw_workdir": "/sandbox/work",
        },
        tmp_path / "outputs",
        phase="full",
    )

    assert override["agentic_math"]["nemoclaw_sandbox"] == "nejumi-taiwan"
    assert override["agentic_math"]["nemoclaw_bin"] == "/usr/local/bin/nemoclaw"
    assert override["agentic_math"]["nemoclaw_workdir"] == "/sandbox/work"
    assert (
        override["agentic_math"]["nemoclaw_openclaw_config_path"]
        == DEFAULT_NEMOCLAW_OPENCLAW_CONFIG_PATH
    )
    assert override["agentic_math"]["use_task_agent"] is True
    assert "nemoclaw_sandbox" not in override["swebench_pro"]
    assert "nemoclaw_sandbox" not in override["deepswe"]


def test_swebench_pro_nemoclaw_manifest_override_does_not_touch_agentic_math(tmp_path):
    override = build_override(
        {
            "slug": "model-a",
            "run_name": "model-a",
            "openclaw_model": "provider/model-a",
            "swebench_pro_nemoclaw_sandbox": "nejumi-taiwan",
            "swebench_pro_nemoclaw_bin": "/usr/local/bin/nemoclaw",
            "swebench_pro_nemoclaw_checkout_sandbox_root": "/sandbox/checkouts",
            "swebench_pro_nemoclaw_checkout_transfer_mode": "copy",
        },
        tmp_path / "outputs",
        phase="full",
    )

    assert override["swebench_pro"]["nemoclaw_sandbox"] == "nejumi-taiwan"
    assert override["swebench_pro"]["nemoclaw_bin"] == "/usr/local/bin/nemoclaw"
    assert override["swebench_pro"]["nemoclaw_checkout_sandbox_root"] == "/sandbox/checkouts"
    assert override["swebench_pro"]["nemoclaw_checkout_transfer_mode"] == "copy"
    assert (
        override["swebench_pro"]["nemoclaw_openclaw_config_path"]
        == DEFAULT_NEMOCLAW_OPENCLAW_CONFIG_PATH
    )
    assert (
        override["agentic_math"]["nemoclaw_sandbox"]
        == DEFAULT_TAIWAN_NEMOCLAW_SANDBOX
    )
    assert "nemoclaw_sandbox" not in override["deepswe"]


def test_deepswe_nemoclaw_manifest_override_does_not_touch_other_agentic_benchmarks(tmp_path):
    override = build_override(
        {
            "slug": "model-a",
            "run_name": "model-a",
            "openclaw_model": "provider/model-a",
            "deepswe_nemoclaw_sandbox": "nejumi-taiwan",
            "deepswe_nemoclaw_bin": "/usr/local/bin/nemoclaw",
            "deepswe_nemoclaw_workdir": "/sandbox/work",
        },
        tmp_path / "outputs",
        phase="agentic",
    )

    assert override["deepswe"]["nemoclaw_sandbox"] == "nejumi-taiwan"
    assert override["deepswe"]["nemoclaw_bin"] == "/usr/local/bin/nemoclaw"
    assert override["deepswe"]["nemoclaw_workdir"] == "/sandbox/work"
    assert (
        override["deepswe"]["nemoclaw_openclaw_config_path"]
        == DEFAULT_NEMOCLAW_OPENCLAW_CONFIG_PATH
    )
    assert (
        override["agentic_math"]["nemoclaw_sandbox"]
        == DEFAULT_TAIWAN_NEMOCLAW_SANDBOX
    )
    assert "nemoclaw_sandbox" not in override["swebench_pro"]


def test_manifest_judge_override_is_rejected(tmp_path):
    with pytest.raises(ValueError, match="Manifest-level judge overrides are forbidden"):
        build_override(
            {
                "slug": "model-a",
                "run_name": "model-a",
                "openclaw_model": "provider/model-a",
                "judge_model": "openrouter/openai/gpt-4.1-nano",
                "judge_parallel": 2,
                "judge_params": {},
            },
            tmp_path / "outputs",
            phase="full",
        )


def test_canary_generates_openai_direct_only(tmp_path):
    args = _args(tmp_path, "full")
    args.model = None
    args.canary = True

    [config_path] = generate_configs(args)
    cfg = OmegaConf.load(config_path)

    assert config_path.name == "config-taiwan-full-gpt-4_1-mini-openai-direct-canary.yaml"
    assert cfg.model.pretrained_model_name_or_path == "gpt-4.1-mini-2025-04-14"
    assert cfg.agentic_math.openclaw_model == "openai-direct/gpt-4.1-mini-2025-04-14"
    assert cfg.swebench_pro.openclaw_model == "openai-direct/gpt-4.1-mini-2025-04-14"
    assert cfg.deepswe.openclaw_model == "openai-direct/gpt-4.1-mini-2025-04-14"
    assert cfg.agentic_swe_assorted.openclaw_model == "openai-direct/gpt-4.1-mini-2025-04-14"
    assert cfg.agentic_math.limit == 50
    assert cfg.agentic_swe_assorted.low_limit == 20
    assert cfg.agentic_swe_assorted.middle_limit == 20
    assert cfg.agentic_swe_assorted.high_limit == 10
    assert cfg.run.deepswe is False


def test_openai_direct_canary_manifest_generates_openai_configs(tmp_path):
    args = _args(tmp_path, "agentic", manifest=OPENAI_CANARY_MANIFEST)
    args.model = None
    args.canary = True

    [config_path] = generate_configs(args)
    cfg = OmegaConf.load(config_path)

    assert config_path.name == "config-taiwan-full-gpt-4_1-mini-openai-direct-canary.yaml"
    assert cfg.model.pretrained_model_name_or_path == "gpt-4.1-mini-2025-04-14"
    assert cfg.agentic_math.openclaw_model == "openai-direct/gpt-4.1-mini-2025-04-14"
    assert cfg.swebench_pro.openclaw_model == "openai-direct/gpt-4.1-mini-2025-04-14"
    assert cfg.deepswe.openclaw_model == "openai-direct/gpt-4.1-mini-2025-04-14"
    assert cfg.agentic_swe_assorted.openclaw_model == "openai-direct/gpt-4.1-mini-2025-04-14"
    assert cfg.agentic_math.thinking == "off"
    assert cfg.swebench_pro.thinking == "off"
    assert cfg.deepswe.thinking == "off"
    assert cfg.agentic_swe_assorted.thinking == "off"
    assert "mtbench" not in cfg
    resolved = OmegaConf.merge(
        OmegaConf.load(ROOT / "configs" / "base_config_taiwan.yaml"),
        cfg,
    )
    assert resolved.mtbench.judge.model == "gpt-5.5-2026-04-23"
    assert resolved.mtbench.judge.params.reasoning.effort == "low"
    assert resolved.hle.judge.model == "gpt-5.5-2026-04-23"
    assert resolved.hle.judge.params.reasoning.effort == "medium"
    assert resolved.hallulens_zh_tw.judge.model == "gpt-5.5-2026-04-23"
    assert resolved.hallulens_zh_tw.judge.params.reasoning.effort == "low"
    assert resolved.agentic_math.limit == 50
    assert resolved.agentic_math.max_tool_calls == 40
    assert resolved.agentic_math.max_agent_turns == 40
    assert resolved.agentic_math.max_tool_wall_seconds == 120
    assert resolved.swebench_pro.max_tool_calls == 40
    assert resolved.swebench_pro.max_agent_turns == 40
    assert resolved.swebench_pro.max_tool_wall_seconds == 300
    assert resolved.deepswe.max_tool_calls == 40
    assert resolved.deepswe.max_agent_turns == 40
    assert resolved.deepswe.max_tool_wall_seconds == 300
    assert resolved.agentic_swe_assorted.max_cumulative_input_tokens == 1_000_000
    assert resolved.agentic_swe_assorted.max_tool_calls == 40
    assert resolved.agentic_swe_assorted.max_agent_turns == 40
    assert resolved.agentic_swe_assorted.high_max_tool_calls == 200
    assert resolved.agentic_swe_assorted.high_max_agent_turns == 150


def test_default_luna_manifest_generates_math50_assorted80(tmp_path):
    args = _args(tmp_path, "full", manifest=FULL_EVAL_MANIFEST)
    args.model = ["gpt-5_6-luna-openai-direct-high"]

    [config_path] = generate_configs(args)
    cfg = OmegaConf.load(config_path)

    assert config_path.name == "config-taiwan-full-gpt-5_6-luna-openai-direct-high.yaml"
    assert cfg.api == "openai_responses"
    assert cfg.model.pretrained_model_name_or_path == "gpt-5.6-luna"
    assert cfg.model.bfcl_model_id == "OpenAIResponsesHandler-FC"
    assert cfg.agentic_math.openclaw_model == "openai-direct/gpt-5.6-luna"
    assert cfg.agentic_math.limit == 50
    assert cfg.run.swebench_pro is False
    assert cfg.run.agentic_swe_assorted is True
    assert cfg.agentic_swe_assorted.low_limit == 20
    assert cfg.agentic_swe_assorted.middle_limit == 20
    assert cfg.agentic_swe_assorted.high_limit == 10
    assert cfg.agentic_swe_assorted.deepswe_metadata_jsonl.endswith(
        "essential_anchored_high_10_model_fidelity_cost_balanced.jsonl"
    )
    assert cfg.run.deepswe is False
    assert cfg.agentic_math.max_tool_calls == 40
    assert cfg.agentic_math.max_agent_turns == 40
    assert cfg.agentic_math.max_tool_wall_seconds == 120
    assert cfg.swebench_pro.max_tool_calls == 40
    assert cfg.swebench_pro.max_agent_turns == 40
    assert cfg.swebench_pro.max_tool_wall_seconds == 300
    assert cfg.agentic_swe_assorted.openclaw_max_attempts == 2
    assert cfg.agentic_math.openclaw_model_overrides.maxTokens == 65536
    assert cfg.agentic_swe_assorted.high_openclaw_model_overrides.maxTokens == 65536
    assert cfg.swebench_pro.openclaw_model_overrides.maxTokens == 65536
    assert cfg.deepswe.openclaw_model_overrides.maxTokens == 65536
    assert cfg.agentic_swe_assorted.openclaw_model_overrides.maxTokens == 65536


def test_anthropic_fable_manifest_generates_direct_routes(tmp_path):
    args = _args(tmp_path, "full", manifest=FULL_EVAL_MANIFEST)
    args.model = ["claude-fable-5-anthropic-direct-high"]
    args.include_final_only = True

    [config_path] = generate_configs(args)
    cfg = OmegaConf.load(config_path)

    assert config_path.name == "config-taiwan-full-claude-fable-5-anthropic-direct-high.yaml"
    assert cfg.api == "anthropic"
    assert cfg.model.pretrained_model_name_or_path == "claude-fable-5"
    assert cfg.model.bfcl_model_id == "Claude-FC"
    assert cfg.generator.effort == "high"
    assert "thinking" not in cfg.model
    assert cfg.agentic_math.openclaw_model == "anthropic/claude-fable-5"
    assert cfg.swebench_pro.openclaw_model == "anthropic/claude-fable-5"
    assert cfg.deepswe.openclaw_model == "anthropic/claude-fable-5"
    assert cfg.agentic_swe_assorted.openclaw_model == "anthropic/claude-fable-5"
    assert cfg.agentic_math.thinking == "high"
    assert cfg.swebench_pro.thinking == "high"
    assert cfg.deepswe.thinking == "high"
    assert cfg.agentic_swe_assorted.thinking == "high"
    assert cfg.agentic_math.openclaw_model_overrides.maxTokens == 128000
    assert cfg.agentic_swe_assorted.high_openclaw_model_overrides.maxTokens == 128000
    assert cfg.agentic_swe_assorted.openclaw_model_overrides.maxTokens == 128000


@pytest.mark.parametrize(
    ("slug", "api", "model", "openclaw_model", "sandbox", "thinking"),
    [
        (
            "gpt-5_6-sol-openai-direct-max",
            "openai_responses",
            "gpt-5.6-sol",
            "openai-direct/gpt-5.6-sol",
            "nejumi-tw-sol",
            "max",
        ),
        (
            "claude-sonnet-5-anthropic-direct-high",
            "anthropic",
            "claude-sonnet-5",
            "anthropic/claude-sonnet-5",
            "nejumi-tw-sonnet5",
            "high",
        ),
    ],
)
def test_parallel_direct_models_generate_isolated_sandbox_configs(
    tmp_path,
    slug,
    api,
    model,
    openclaw_model,
    sandbox,
    thinking,
):
    args = _args(tmp_path, "full", manifest=FULL_EVAL_MANIFEST)
    args.model = [slug]

    [config_path] = generate_configs(args)
    cfg = OmegaConf.load(config_path)

    assert cfg.api == api
    assert cfg.model.pretrained_model_name_or_path == model
    assert cfg.agentic_math.openclaw_model == openclaw_model
    assert cfg.agentic_math.nemoclaw_sandbox == sandbox
    assert cfg.agentic_swe_assorted.nemoclaw_sandbox == sandbox
    assert cfg.agentic_math.thinking == thinking
    assert cfg.agentic_swe_assorted.thinking == thinking
    if slug == "claude-sonnet-5-anthropic-direct-high":
        assert cfg.agentic_swe_assorted.high_max_agent_turns == 180
        assert cfg.agentic_swe_assorted.high_max_cumulative_input_tokens == 20_000_000


def test_active_full_manifest_retains_suspended_openrouter_routes_safely():
    manifest = OmegaConf.to_container(OmegaConf.load(FULL_EVAL_MANIFEST), resolve=True)
    models = manifest["models"]

    [canary] = [model for model in models if model.get("canary")]
    assert canary["slug"] == "gpt-5_6-luna-openai-direct-high"
    assert canary["openclaw_model"].startswith("openai-direct/")
    assert not canary.get("suspended", False)

    [wandb_glm] = [model for model in models if model["slug"] == "glm-5_2-wandb-inference"]
    assert wandb_glm["openclaw_model"] == "wandb-inference/zai-org/GLM-5.2"
    assert not wandb_glm.get("suspended", False)

    openrouter_models = [
        model for model in models if "openrouter" in model["openclaw_model"].lower()
    ]
    assert openrouter_models
    assert all(model.get("suspended") for model in openrouter_models)
    assert all(model.get("suspension_reason") for model in openrouter_models)

    selected_by_default = select_models(models, None)
    assert all(not model.get("suspended") for model in selected_by_default)
    assert all(
        "openrouter" not in model["openclaw_model"].lower()
        for model in selected_by_default
    )


def test_twbias_stays_excluded_from_generated_full_configs(tmp_path):
    args = _args(tmp_path, "full", manifest=OPENAI_CANARY_MANIFEST)
    args.model = None
    args.canary = True

    [config_path] = generate_configs(args)
    cfg = OmegaConf.load(config_path)

    assert cfg.run.twbias is False
    assert "taiwan_aggregate" not in cfg


def test_manifest_openclaw_model_params_can_be_section_specific(tmp_path):
    override = build_override(
        {
            "slug": "model-a",
            "run_name": "model-a",
            "openclaw_model": "openrouter-direct/example/model-a",
            "openclaw_model_params": {
                "provider": {"only": ["global/provider"], "allow_fallbacks": False}
            },
            "openclaw_model_overrides": {"maxTokens": 4096},
            "swebench_pro_openclaw_model_params": {
                "provider": {"only": ["swe/provider"], "allow_fallbacks": True}
            },
            "swebench_pro_openclaw_model_overrides": {"maxTokens": 8192},
            "deepswe_openclaw_model_params": {
                "provider": {"only": ["deepswe/provider"], "require_parameters": True}
            },
            "deepswe_openclaw_model_overrides": {"maxTokens": 2048},
            "agentic_swe_assorted_high_openclaw_model_params": {
                "maxTokens": 65536,
                "provider": {"only": ["high/provider"]},
            },
        },
        tmp_path / "outputs",
        phase="agentic",
    )

    assert override["agentic_math"]["openclaw_model_params"]["provider"]["only"] == [
        "global/provider"
    ]
    assert override["swebench_pro"]["openclaw_model_params"]["provider"]["only"] == [
        "swe/provider"
    ]
    assert override["swebench_pro"]["openclaw_model_params"]["provider"]["allow_fallbacks"] is True
    assert override["deepswe"]["openclaw_model_params"]["provider"]["only"] == [
        "deepswe/provider"
    ]
    assert override["deepswe"]["openclaw_model_params"]["provider"]["require_parameters"] is True
    assert override["agentic_math"]["openclaw_model_overrides"]["maxTokens"] == 4096
    assert override["swebench_pro"]["openclaw_model_overrides"]["maxTokens"] == 8192
    assert override["deepswe"]["openclaw_model_overrides"]["maxTokens"] == 2048
    assert override["agentic_swe_assorted"]["high_openclaw_model_params"] == {
        "maxTokens": 65536,
        "provider": {"only": ["high/provider"]},
    }


def test_default_selection_skips_final_only_models():
    models = [
        {"slug": "glm", "canary": True},
        {"slug": "opus", "final_only": True},
        {"slug": "deepseek"},
    ]

    assert [model["slug"] for model in select_models(models, None)] == [
        "glm",
        "deepseek",
    ]


def test_final_only_model_requires_explicit_include_flag():
    models = [{"slug": "opus", "final_only": True}]

    with pytest.raises(SystemExit, match="--include-final-only"):
        select_models(models, ["opus"])

    assert select_models(models, ["opus"], include_final_only=True) == models


def test_suspended_models_are_excluded_unless_explicitly_enabled():
    models = [
        {"slug": "direct"},
        {"slug": "openrouter", "suspended": True},
    ]

    assert select_models(models, None) == [models[0]]
    with pytest.raises(SystemExit, match="--include-suspended"):
        select_models(models, ["openrouter"])
    assert select_models(models, ["openrouter"], include_suspended=True) == [models[1]]

    with pytest.raises(SystemExit, match="requires at least one explicit --model"):
        select_models(models, None, include_suspended=True)


def test_suspended_canary_is_rejected_even_with_include_flag():
    models = [{"slug": "paused", "canary": True, "suspended": True}]

    with pytest.raises(SystemExit, match="Canary model paused is suspended"):
        select_models(models, None, canary=True, include_suspended=True)
