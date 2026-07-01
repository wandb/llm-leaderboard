import argparse
import sys
from pathlib import Path

from omegaconf import OmegaConf
import pytest

ROOT = Path(__file__).resolve().parents[1]
TOOLS_DIR = ROOT / "scripts" / "tools"
OPENAI_CANARY_MANIFEST = ROOT / "configs" / "taiwan_openai_canary_models.yaml"
if str(TOOLS_DIR) not in sys.path:
    sys.path.insert(0, str(TOOLS_DIR))

from prepare_taiwan_full_eval_configs import (
    DEFAULT_MANIFEST,
    DEFAULT_NEMOCLAW_OPENCLAW_CONFIG_PATH,
    build_override,
    generate_configs,
    select_models,
)


def _args(tmp_path: Path, phase: str) -> argparse.Namespace:
    return argparse.Namespace(
        manifest=DEFAULT_MANIFEST,
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
    )


def test_nonagentic_phase_skips_agentic_and_aggregate(tmp_path):
    [config_path] = generate_configs(_args(tmp_path, "nonagentic"))
    cfg = OmegaConf.load(config_path)

    assert cfg.run.agentic_math is False
    assert cfg.run.swebench_pro is False
    assert cfg.run.aggregate_taiwan is False
    assert cfg.run.bfcl is True
    assert cfg.run.mtbench is True
    assert cfg.run.hle is True
    assert cfg.mtbench.judge.model == "gpt-4.1-mini-2025-04-14"
    assert cfg.hle.judge.model == "gpt-4.1-mini-2025-04-14"
    assert cfg.hallulens_zh_tw.judge.model == "gpt-4.1-mini-2025-04-14"


def test_agentic_aggregate_phase_reuses_completed_outputs(tmp_path):
    [config_path] = generate_configs(_args(tmp_path, "agentic_aggregate"))
    cfg = OmegaConf.load(config_path)
    output_root = tmp_path / "outputs" / "taiwan_full_eval"
    slug = "gpt-4_1-mini-openai-direct-canary"

    assert cfg.run.agentic_math is True
    assert cfg.run.swebench_pro is True
    assert cfg.run.aggregate_taiwan is True
    assert cfg.run.bfcl is False
    assert cfg.run.mtbench is False
    assert cfg.agentic_math.run_openclaw is False
    assert Path(cfg.agentic_math.results_dir) == output_root / "agentic_math" / slug / "openclaw"
    assert cfg.swebench_pro.run_openclaw is False
    assert Path(cfg.swebench_pro.patch_path) == (
        output_root / "swebench_pro" / slug / "openclaw" / "patches.json"
    )


def test_agentic_phase_runs_only_agentic_generation(tmp_path):
    [config_path] = generate_configs(_args(tmp_path, "agentic"))
    cfg = OmegaConf.load(config_path)

    assert cfg.run.agentic_math is True
    assert cfg.run.swebench_pro is True
    assert cfg.run.aggregate_taiwan is False
    assert cfg.run.bfcl is False
    assert cfg.run.mtbench is False
    assert cfg.agentic_math.run_openclaw is True
    assert cfg.swebench_pro.run_openclaw is True
    assert cfg.agentic_math.results_dir is None
    assert cfg.swebench_pro.patch_path is None
    assert cfg.swebench_pro.subset == "leaderboard_compact_80"
    assert cfg.swebench_pro.max_input_tokens == 1_000_000
    assert cfg.swebench_pro.max_tool_calls == 60


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


def test_nemoclaw_cli_openclaw_config_path_override(tmp_path):
    args = _args(tmp_path, "agentic")
    args.agentic_math_nemoclaw_sandbox = "nejumi-taiwan"
    args.agentic_math_nemoclaw_openclaw_config_path = "/custom/math/openclaw.json"
    args.swebench_pro_nemoclaw_sandbox = "nejumi-taiwan"
    args.swebench_pro_nemoclaw_openclaw_config_path = "/custom/swe/openclaw.json"

    [config_path] = generate_configs(args)
    cfg = OmegaConf.load(config_path)

    assert cfg.agentic_math.nemoclaw_openclaw_config_path == "/custom/math/openclaw.json"
    assert cfg.swebench_pro.nemoclaw_openclaw_config_path == "/custom/swe/openclaw.json"


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
    assert cfg.swebench_pro.subset == "leaderboard_compact_80"


def test_openai_direct_canary_manifest_generates_openai_configs(tmp_path):
    args = _args(tmp_path, "agentic")
    args.manifest = OPENAI_CANARY_MANIFEST
    args.model = None
    args.canary = True

    [config_path] = generate_configs(args)
    cfg = OmegaConf.load(config_path)

    assert config_path.name == "config-taiwan-full-gpt-4_1-mini-openai-direct-canary.yaml"
    assert cfg.model.pretrained_model_name_or_path == "gpt-4.1-mini-2025-04-14"
    assert cfg.agentic_math.openclaw_model == "openai-direct/gpt-4.1-mini-2025-04-14"
    assert cfg.swebench_pro.openclaw_model == "openai-direct/gpt-4.1-mini-2025-04-14"
    assert cfg.agentic_math.thinking == "off"
    assert cfg.swebench_pro.thinking == "off"
    assert cfg.mtbench.judge.model == "gpt-4.1-mini-2025-04-14"


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
