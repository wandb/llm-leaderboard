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


def test_bounded_map_limits_live_tasks_and_processes_every_item():
    active = 0
    peak_active = 0
    completed = []

    async def worker(item):
        nonlocal active, peak_active
        active += 1
        peak_active = max(peak_active, active)
        await asyncio.sleep(0.001)
        completed.append(item["id"])
        active -= 1

    items = [{"id": index} for index in range(25)]
    asyncio.run(
        hle._bounded_map(
            items,
            worker,
            concurrency=4,
            desc="test",
        )
    )

    assert sorted(completed) == list(range(25))
    assert peak_active == 4


def test_bounded_map_checkpoint_stress_keeps_fd_usage_stable(tmp_path):
    fd_dir = Path("/proc/self/fd")
    if not fd_dir.exists():
        pytest.skip("Linux /proc file-descriptor accounting is unavailable")

    before = len(list(fd_dir.iterdir()))

    async def worker(item):
        (tmp_path / f"{item['id']}.json").write_text("{}", encoding="utf-8")
        await asyncio.sleep(0)

    items = [{"id": index} for index in range(1500)]
    asyncio.run(
        hle._bounded_map(
            items,
            worker,
            concurrency=32,
            desc="checkpoint stress",
        )
    )

    after = len(list(fd_dir.iterdir()))
    assert after <= before + 4
