import asyncio
import sys
import time
from pathlib import Path
from types import SimpleNamespace as SNS

import pytest

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = ROOT / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

from evaluator.evaluate_utils import provider_rate_limiter
from evaluator.evaluate_utils.provider_rate_limiter import (
    ProviderRequestRateLimiter,
    get_provider_request_rate_limiter,
)


def test_provider_request_rate_limiter_paces_second_request(monkeypatch):
    monotonic_values = iter([100.0, 100.001])
    sleeps: list[float] = []

    monkeypatch.setattr(provider_rate_limiter.time, "monotonic", lambda: next(monotonic_values))
    monkeypatch.setattr(provider_rate_limiter.time, "sleep", lambda seconds: sleeps.append(seconds))

    limiter = ProviderRequestRateLimiter(min_interval_sec=0.01, jitter_sec=0.0)
    limiter.wait()
    limiter.wait()

    assert sleeps == pytest.approx([0.009])


def test_provider_request_rate_limiter_is_shared_by_key():
    key = "test-provider-rate-limiter-shared-key"
    first = get_provider_request_rate_limiter(key, min_interval_sec=1.0, jitter_sec=0.0)
    second = get_provider_request_rate_limiter(key, min_interval_sec=0.1, jitter_sec=2.0)

    assert second is first
    assert second.min_interval_sec == 1.0
    assert second.jitter_sec == 2.0


def test_llm_async_processor_uses_configured_provider_limiter(monkeypatch):
    import evaluator.evaluate_utils.llm_async_processor as processor_module
    from evaluator.evaluate_utils.llm_async_processor import LLMAsyncProcessor

    class FakeConfig(dict):
        inference_interval = 0

    cfg = FakeConfig(
        batch_size=2,
        provider_rate_limit={
            "enabled": True,
            "key": "test-llm-async-processor-provider-limiter",
            "min_request_interval_sec": 0.02,
            "request_jitter_sec": 0.0,
        },
    )
    monkeypatch.setattr(
        processor_module.WandbConfigSingleton,
        "get_instance",
        staticmethod(lambda: SNS(config=cfg)),
    )

    call_times: list[float] = []

    class FakeLLM:
        model = "fake-model"

        async def ainvoke(self, messages, **kwargs):
            call_times.append(time.monotonic())
            return "ok"

    processor = LLMAsyncProcessor(
        FakeLLM(),
        inputs=[
            ([{"role": "user", "content": "a"}], {}),
            ([{"role": "user", "content": "b"}], {}),
        ],
    )
    results = asyncio.run(processor.get_results_async())

    assert results == ["ok", "ok"]
    assert len(call_times) == 2
    assert call_times[1] - call_times[0] >= 0.015
