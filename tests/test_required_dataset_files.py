from pathlib import Path
import sys
from types import SimpleNamespace

import pytest
from omegaconf import OmegaConf

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

from evaluator import jaster, tceval_v2


class _Artifact:
    def __init__(self, root: Path):
        self.root = root

    def download(self):
        return str(self.root)


class _Run:
    def __init__(self, root: Path):
        self.root = root

    def use_artifact(self, *_args, **_kwargs):
        return _Artifact(self.root)


def test_tceval_requires_every_configured_task_subset_file(tmp_path, monkeypatch):
    (tmp_path / "dataset").mkdir()
    instance = SimpleNamespace(
        run=_Run(tmp_path),
        llm=object(),
        config=OmegaConf.create(
            {
                "testmode": False,
                "model": {"pretrained_model_name_or_path": "test-model"},
                "generator": {"max_tokens": 32},
                "tceval_v2": {
                    "artifacts_path": "entity/project/data:test",
                    "dataset_dir": "dataset",
                    "tasks": ["drcd"],
                },
            }
        ),
    )
    monkeypatch.setattr(
        tceval_v2.WandbConfigSingleton,
        "get_instance",
        lambda: instance,
    )

    with pytest.raises(FileNotFoundError, match="required task file"):
        tceval_v2.evaluate()


def test_jaster_requires_every_scheduled_task_subset_file(tmp_path, monkeypatch):
    (tmp_path / "dataset").mkdir()
    instance = SimpleNamespace(
        run=_Run(tmp_path),
        llm=object(),
        config=OmegaConf.create(
            {
                "testmode": False,
                "num_few_shots": 2,
                "model": {"pretrained_model_name_or_path": "test-model"},
                "generator": {"max_tokens": 32},
                "run": {
                    "tmmluplus_robustness": False,
                    "jmmlu_robustness": False,
                },
                "jaster": {
                    "artifacts_path": "entity/project/data:test",
                    "dataset_dir": "dataset",
                    "tasks": ["tmmluplus"],
                },
            }
        ),
    )
    monkeypatch.setattr(
        jaster.WandbConfigSingleton,
        "get_instance",
        lambda: instance,
    )

    with pytest.raises(FileNotFoundError, match="required task file"):
        jaster.evaluate_n_shot(few_shots=False)
