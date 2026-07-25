import asyncio
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from types import SimpleNamespace

import httpx
import openai


ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))


from evaluator.evaluate_utils.llm_async_processor import LLMAsyncProcessor
from llm_inference_adapter import LLMResponse


class FakeConfig(dict):
    inference_interval = 0

    def __getattr__(self, key):
        try:
            return self[key]
        except KeyError as exc:
            raise AttributeError(key) from exc


def test_async_timeout_is_retried_before_success(monkeypatch):
    cfg = FakeConfig(batch_size=1, provider_rate_limit={"enabled": False})
    monkeypatch.setattr(
        "evaluator.evaluate_utils.llm_async_processor."
        "WandbConfigSingleton.get_instance",
        lambda: SimpleNamespace(config=cfg),
    )
    monkeypatch.setattr(
        "evaluator.evaluate_utils.llm_async_processor.backoff.full_jitter",
        lambda _value: 0,
    )

    class FlakyLLM:
        model = "fake"

        def __init__(self):
            self.calls = 0

        async def ainvoke(self, messages, **_kwargs):
            self.calls += 1
            if self.calls == 1:
                raise openai.APITimeoutError(
                    request=httpx.Request("POST", "https://api.example.test/v1/responses")
                )
            return LLMResponse(content=messages[0]["content"])

    llm = FlakyLLM()
    processor = LLMAsyncProcessor(
        llm,
        batch_size=1,
        backoff_max_tries=2,
        backoff_max_time=5,
    )
    try:
        response = processor.process_single(
            [{"role": "user", "content": "recovered"}]
        )
        assert response.content == "recovered"
        assert llm.calls == 2
    finally:
        processor.close_sync_loop()


def test_sync_calls_share_one_persistent_event_loop(monkeypatch):
    cfg = FakeConfig(batch_size=4, provider_rate_limit={"enabled": False})
    monkeypatch.setattr(
        "evaluator.evaluate_utils.llm_async_processor."
        "WandbConfigSingleton.get_instance",
        lambda: SimpleNamespace(config=cfg),
    )

    class FakeLLM:
        model = "fake"

        def __init__(self):
            self.loop_ids = []

        async def ainvoke(self, messages, **_kwargs):
            self.loop_ids.append(id(asyncio.get_running_loop()))
            await asyncio.sleep(0.01)
            return LLMResponse(content=messages[0]["content"])

    llm = FakeLLM()
    processor = LLMAsyncProcessor(llm, batch_size=4)
    try:
        with ThreadPoolExecutor(max_workers=4) as executor:
            futures = [
                executor.submit(
                    processor.process_single,
                    [{"role": "user", "content": str(index)}],
                )
                for index in range(8)
            ]
            assert sorted(future.result().content for future in futures) == [
                str(index) for index in range(8)
            ]
        assert len(set(llm.loop_ids)) == 1
    finally:
        processor.close_sync_loop()


def test_responses_native_history_items_are_accepted(monkeypatch):
    cfg = FakeConfig(batch_size=1, provider_rate_limit={"enabled": False})
    monkeypatch.setattr(
        "evaluator.evaluate_utils.llm_async_processor."
        "WandbConfigSingleton.get_instance",
        lambda: SimpleNamespace(config=cfg),
    )

    class FakeLLM:
        model = "fake"

        async def ainvoke(self, messages, **_kwargs):
            return LLMResponse(content=str(len(messages)))

    processor = LLMAsyncProcessor(FakeLLM(), batch_size=1)
    try:
        response = processor.process_single(
            [
                {"role": "user", "content": "hello"},
                {"type": "reasoning", "encrypted_content": "opaque"},
                {
                    "type": "function_call_output",
                    "call_id": "call-1",
                    "output": "done",
                },
            ]
        )
        assert response.content == "3"
    finally:
        processor.close_sync_loop()


def test_responses_developer_role_is_accepted(monkeypatch):
    cfg = FakeConfig(batch_size=1, provider_rate_limit={"enabled": False})
    monkeypatch.setattr(
        "evaluator.evaluate_utils.llm_async_processor."
        "WandbConfigSingleton.get_instance",
        lambda: SimpleNamespace(config=cfg),
    )

    class FakeLLM:
        model = "fake"

        async def ainvoke(self, messages, **_kwargs):
            return LLMResponse(content=messages[0]["role"])

    processor = LLMAsyncProcessor(FakeLLM(), batch_size=1)
    try:
        response = processor.process_single(
            [
                {"role": "developer", "content": "Follow the tool policy."},
                {"role": "user", "content": "Use the memory tool."},
            ]
        )
        assert response.content == "developer"
    finally:
        processor.close_sync_loop()


def test_anthropic_structured_content_history_is_accepted(monkeypatch):
    cfg = FakeConfig(batch_size=1, provider_rate_limit={"enabled": False})
    monkeypatch.setattr(
        "evaluator.evaluate_utils.llm_async_processor."
        "WandbConfigSingleton.get_instance",
        lambda: SimpleNamespace(config=cfg),
    )

    class FakeLLM:
        model = "fake"

        async def ainvoke(self, messages, **_kwargs):
            return LLMResponse(content=str(len(messages)))

    processor = LLMAsyncProcessor(FakeLLM(), batch_size=1)
    try:
        response = processor.process_single(
            [
                {"role": "user", "content": "hello"},
                {
                    "role": "assistant",
                    "content": [
                        {
                            "type": "tool_use",
                            "id": "call-1",
                            "name": "weather",
                            "input": {"city": "Taipei"},
                        }
                    ],
                },
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "tool_result",
                            "tool_use_id": "call-1",
                            "content": "sunny",
                        }
                    ],
                },
            ]
        )
        assert response.content == "3"
    finally:
        processor.close_sync_loop()


def test_batch_callback_checkpoints_each_completed_result(monkeypatch):
    cfg = FakeConfig(batch_size=3, provider_rate_limit={"enabled": False})
    monkeypatch.setattr(
        "evaluator.evaluate_utils.llm_async_processor."
        "WandbConfigSingleton.get_instance",
        lambda: SimpleNamespace(config=cfg),
    )

    class OutOfOrderLLM:
        model = "fake"

        async def ainvoke(self, messages, **_kwargs):
            value = int(messages[0]["content"])
            await asyncio.sleep((3 - value) * 0.01)
            return LLMResponse(content=str(value))

    inputs = [
        ([{"role": "user", "content": str(index)}], {})
        for index in range(3)
    ]
    completed = []
    processor = LLMAsyncProcessor(OutOfOrderLLM(), inputs=inputs, batch_size=3)
    try:
        results = processor.get_results(
            on_result=lambda index, response: completed.append(
                (index, response.content)
            )
        )
        assert [response.content for response in results] == ["0", "1", "2"]
        assert completed == [(2, "2"), (1, "1"), (0, "0")]
    finally:
        processor.close_sync_loop()


def test_explicit_zero_interval_is_preserved(monkeypatch):
    cfg = FakeConfig(batch_size=1, provider_rate_limit={"enabled": False})
    cfg.inference_interval = 7
    monkeypatch.setattr(
        "evaluator.evaluate_utils.llm_async_processor."
        "WandbConfigSingleton.get_instance",
        lambda: SimpleNamespace(config=cfg),
    )

    processor = LLMAsyncProcessor(object(), inference_interval=0)
    assert processor.inference_interval == 0


def test_network_retry_defaults_are_configurable(monkeypatch):
    cfg = FakeConfig(
        batch_size=1,
        provider_rate_limit={"enabled": False},
        network={"retry": {"max_time_sec": 45, "max_tries": 6}},
    )
    monkeypatch.setattr(
        "evaluator.evaluate_utils.llm_async_processor."
        "WandbConfigSingleton.get_instance",
        lambda: SimpleNamespace(config=cfg),
    )

    processor = LLMAsyncProcessor(object())

    assert processor.backoff_max_time == 45
    assert processor.backoff_max_tries == 6
