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
    assert cfg.agentic_swe_assorted.low_limit == 36
    assert cfg.agentic_swe_assorted.middle_limit == 36
    assert cfg.agentic_swe_assorted.high_limit == 8
    assert cfg.agentic_swe_assorted.tier_weights == "low=1,middle=1,high=1"
    assert cfg.agentic_swe_assorted.swe_workers == 4
    assert cfg.agentic_swe_assorted.high_workers == 2
    assert cfg.agentic_swe_assorted.max_input_tokens == 1_000_000
    assert cfg.agentic_swe_assorted.max_tool_calls == 40
    assert cfg.agentic_swe_assorted.max_agent_turns == 40
    assert cfg.agentic_swe_assorted.high_max_tool_calls == 120
    assert cfg.agentic_swe_assorted.high_max_agent_turns == 120
    assert cfg.agentic_swe_assorted.high_max_cumulative_input_tokens == 12_000_000
    assert cfg.agentic_swe_assorted.max_tool_wall_seconds == 120
    assert cfg.bfcl.num_threads == 4
    assert cfg.bfcl.provider_min_request_interval_sec == 2.0
    assert cfg.bfcl.provider_request_jitter_sec == 0.5
    assert cfg.bfcl.consecutive_failure_fail_fast == 5


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
    assert "nemoclaw_sandbox" not in cfg.agentic_math
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
    assert "nemoclaw_sandbox" not in override["agentic_math"]
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
    assert "nemoclaw_sandbox" not in override["agentic_math"]
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
    assert cfg.agentic_swe_assorted.low_limit == 36
    assert cfg.agentic_swe_assorted.middle_limit == 36
    assert cfg.agentic_swe_assorted.high_limit == 8
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
    assert resolved.agentic_swe_assorted.max_tool_calls == 40
    assert resolved.agentic_swe_assorted.max_agent_turns == 40
    assert resolved.agentic_swe_assorted.high_max_tool_calls == 120
    assert resolved.agentic_swe_assorted.high_max_agent_turns == 120


def test_default_glm_manifest_generates_math50_swe40(tmp_path):
    args = _args(tmp_path, "full", manifest=FULL_EVAL_MANIFEST)
    args.model = ["glm-5_2-openrouter-reasoning"]

    [config_path] = generate_configs(args)
    cfg = OmegaConf.load(config_path)

    assert config_path.name == "config-taiwan-full-glm-5_2-openrouter-reasoning.yaml"
    assert cfg.agentic_math.limit == 50
    assert cfg.run.swebench_pro is False
    assert cfg.run.agentic_swe_assorted is True
    assert cfg.agentic_swe_assorted.low_limit == 36
    assert cfg.agentic_swe_assorted.middle_limit == 36
    assert cfg.agentic_swe_assorted.high_limit == 8
    assert cfg.agentic_swe_assorted.deepswe_metadata_jsonl.endswith(
        "essential_anchored_high_8_glm52max_cap100_10m_lang_balanced.jsonl"
    )
    assert cfg.run.deepswe is False
    assert cfg.agentic_math.max_tool_calls == 40
    assert cfg.agentic_math.max_agent_turns == 40
    assert cfg.agentic_math.max_tool_wall_seconds == 120
    assert cfg.swebench_pro.max_tool_calls == 40
    assert cfg.swebench_pro.max_agent_turns == 40
    assert cfg.swebench_pro.max_tool_wall_seconds == 300
    assert list(cfg.agentic_math.openclaw_model_params.provider.only) == ["z-ai/fp8"]
    assert cfg.agentic_math.openclaw_model_params.provider.allow_fallbacks is False
    assert list(cfg.swebench_pro.openclaw_model_params.provider.only) == ["z-ai/fp8"]
    assert cfg.swebench_pro.openclaw_model_params.provider.require_parameters is True
    assert list(cfg.deepswe.openclaw_model_params.provider.only) == ["z-ai/fp8"]
    assert list(cfg.agentic_swe_assorted.openclaw_model_params.provider.only) == ["z-ai/fp8"]
    assert cfg.agentic_swe_assorted.openclaw_model_params.provider.require_parameters is True
    assert cfg.agentic_math.openclaw_model_overrides.maxTokens == 4096
    assert cfg.swebench_pro.openclaw_model_overrides.maxTokens == 4096
    assert cfg.deepswe.openclaw_model_overrides.maxTokens == 4096
    assert cfg.agentic_swe_assorted.openclaw_model_overrides.maxTokens == 4096


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
