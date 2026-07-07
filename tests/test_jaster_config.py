import sys
from pathlib import Path

from omegaconf import OmegaConf


REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = REPO_ROOT / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

from evaluator.jaster import _get_override_max_tokens


def test_get_override_max_tokens_returns_none_when_key_is_missing():
    cfg = OmegaConf.create({"jaster": {"dataset_dir": "tmmluplus_robust"}})

    assert _get_override_max_tokens(cfg, "jaster") is None


def test_get_override_max_tokens_reads_configured_value():
    cfg = OmegaConf.create({"jaster": {"override_max_tokens": 32768}})

    assert _get_override_max_tokens(cfg, "jaster") == 32768
