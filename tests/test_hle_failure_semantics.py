import asyncio
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest
from omegaconf import OmegaConf

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

from evaluator import hle


def _instance_with_artifact_error(error):
    class FailingRun:
        def use_artifact(self, *_args, **_kwargs):
            raise error

    config = OmegaConf.create(
        {
            "testmode": False,
            "model": {"pretrained_model_name_or_path": "test-model"},
            "hle": {
                "artifact_path": "entity/project/hle:test",
                "dataset_dir": "hle",
                "generator_config": {},
                "judge": {
                    "model": "judge-model",
                    "parallel": 1,
                    "params": {},
                },
            },
        }
    )
    return SimpleNamespace(run=FailingRun(), config=config, llm=object())


def test_hle_artifact_failure_is_not_reported_as_completion(monkeypatch):
    instance = _instance_with_artifact_error(RuntimeError("artifact unavailable"))
    monkeypatch.setattr(
        hle.WandbConfigSingleton,
        "get_instance",
        lambda: instance,
    )

    with pytest.raises(RuntimeError, match="artifact download failed"):
        asyncio.run(hle.evaluate_async())
