from __future__ import annotations

from typing import Any

from omegaconf import OmegaConf


DEFAULT_BASE_CONFIG = "base_config.yaml"
TAIWAN_BASE_CONFIG = "base_config_taiwan.yaml"
TAIWAN_WANDB_PROJECT = "tc-leaderboard"


def select_base_config_name(
    custom_cfg: Any,
    explicit_base_config: str | None,
) -> str:
    """Choose the leaderboard base config without relying on operator memory."""
    if explicit_base_config:
        return explicit_base_config

    expected_project = str(
        OmegaConf.select(custom_cfg, "wandb.expected_project", default="") or ""
    ).strip()
    run_name = str(
        OmegaConf.select(custom_cfg, "wandb.run_name", default="") or ""
    ).strip()
    if expected_project == TAIWAN_WANDB_PROJECT or run_name.startswith("taiwan/"):
        return TAIWAN_BASE_CONFIG
    return DEFAULT_BASE_CONFIG
