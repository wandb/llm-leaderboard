import sys
from pathlib import Path

from omegaconf import OmegaConf

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "scripts"))

from evaluator.evaluate_utils.validation_helpers import (
    check_token_allocation,
    get_max_output_tokens,
)


def test_mtbench_rejects_short_global_fallback():
    cfg = OmegaConf.create(
        {
            "generator": {"max_tokens": 128},
            "mtbench": {},
        }
    )

    valid, message = check_token_allocation(cfg, "mtbench")

    assert valid is False
    assert "最低値: 1024" in message


def test_hle_rejects_short_global_fallback():
    cfg = OmegaConf.create(
        {
            "generator": {"max_tokens": 128},
            "hle": {},
        }
    )

    valid, message = check_token_allocation(cfg, "hle")

    assert valid is False
    assert "最低値: 4096" in message


def test_benchmark_specific_limit_overrides_global_default():
    cfg = OmegaConf.create(
        {
            "generator": {"max_tokens": 128},
            "mtbench": {"generator_config": {"max_tokens": 1024}},
            "hle": {"generator_config": {"max_tokens": 4096}},
        }
    )

    assert get_max_output_tokens(cfg, "mtbench") == 1024
    assert get_max_output_tokens(cfg, "hle") == 4096
    assert check_token_allocation(cfg, "mtbench")[0] is True
    assert check_token_allocation(cfg, "hle")[0] is True


def test_agentic_runner_does_not_report_irrelevant_global_token_default():
    cfg = OmegaConf.create(
        {
            "generator": {"max_tokens": 128},
            "agentic_math": {},
        }
    )

    valid, message = check_token_allocation(cfg, "agentic_math")

    assert valid is True
    assert "共通generator.max_tokensは未使用" in message
    assert "(128)" not in message
