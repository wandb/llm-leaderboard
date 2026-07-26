import sys
from pathlib import Path

from omegaconf import OmegaConf

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from config_base_selection import (
    DEFAULT_BASE_CONFIG,
    TAIWAN_BASE_CONFIG,
    select_base_config_name,
)


def test_explicit_base_config_always_wins():
    cfg = OmegaConf.create(
        {
            "wandb": {
                "run_name": "taiwan/full/model",
                "expected_project": "tc-leaderboard",
            }
        }
    )

    assert select_base_config_name(cfg, "/tmp/custom-base.yaml") == (
        "/tmp/custom-base.yaml"
    )


def test_taiwan_scope_selects_taiwan_base_config():
    cfg = OmegaConf.create(
        {"wandb": {"expected_project": "tc-leaderboard", "run_name": "custom"}}
    )

    assert select_base_config_name(cfg, None) == TAIWAN_BASE_CONFIG


def test_taiwan_run_name_is_a_defensive_fallback():
    cfg = OmegaConf.create({"wandb": {"run_name": "taiwan/full/model"}})

    assert select_base_config_name(cfg, None) == TAIWAN_BASE_CONFIG


def test_non_taiwan_config_selects_default_base_config():
    cfg = OmegaConf.create(
        {"wandb": {"project": "nejumi-leaderboard4", "run_name": "model"}}
    )

    assert select_base_config_name(cfg, None) == DEFAULT_BASE_CONFIG
