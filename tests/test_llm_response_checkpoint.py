import sys
import inspect
from pathlib import Path
from types import SimpleNamespace


ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))


from evaluator.evaluate_utils.llm_response_checkpoint import (
    JSONItemCheckpointStore,
    LLMResponseCheckpointStore,
    run_checkpointed_batch,
)
from llm_inference_adapter import LLMResponse, ToolCall


def test_checkpoint_round_trip_and_request_binding(tmp_path):
    store = LLMResponseCheckpointStore(tmp_path, model_name="model-a")
    messages = [{"role": "user", "content": "question"}]
    kwargs = {"max_tokens": 32}
    response = LLMResponse(
        content="answer",
        reasoning_content="reason",
        tool_calls=[ToolCall(name="lookup", arguments={"x": 1}, id="call-1")],
        prompt_tokens=7,
        completion_tokens=3,
        finish_reason="stop",
    )

    store.save("task:test:0", response, messages=messages, kwargs=kwargs)
    loaded = store.load("task:test:0", messages=messages, kwargs=kwargs)

    assert loaded is not None
    assert loaded.content == "answer"
    assert loaded.reasoning_content == "reason"
    assert loaded.prompt_tokens == 7
    assert loaded.tool_calls[0].name == "lookup"
    assert (
        store.load(
            "task:test:0",
            messages=[{"role": "user", "content": "changed"}],
            kwargs=kwargs,
        )
        is None
    )


def test_corrupt_checkpoint_is_treated_as_missing(tmp_path):
    store = LLMResponseCheckpointStore(tmp_path, model_name="model-a")
    path = store._path("task:test:0")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("{broken", encoding="utf-8")

    assert (
        store.load(
            "task:test:0",
            messages=[{"role": "user", "content": "question"}],
            kwargs={},
        )
        is None
    )


def test_json_checkpoint_is_atomic_and_request_bound(tmp_path):
    store = JSONItemCheckpointStore(tmp_path, model_name="judge-a")
    request = {
        "messages": [{"role": "user", "content": "grade this"}],
        "params": {"reasoning": {"effort": "medium"}},
    }

    store.save("case-1", {"correct": "yes"}, request=request)

    assert store.load("case-1", request=request) == {"correct": "yes"}
    assert (
        store.load(
            "case-1",
            request={
                "messages": [{"role": "user", "content": "different"}],
                "params": request["params"],
            },
        )
        is None
    )


def test_checkpointed_batch_resumes_only_missing_or_changed_requests(
    tmp_path,
    monkeypatch,
):
    class Config(dict):
        inference_interval = 0

        def __getattr__(self, key):
            return self[key]

    cfg = Config(batch_size=3, provider_rate_limit={"enabled": False})
    monkeypatch.setattr(
        "evaluator.evaluate_utils.llm_async_processor."
        "WandbConfigSingleton.get_instance",
        lambda: SimpleNamespace(config=cfg),
    )

    class CountingLLM:
        model = "model-a"

        def __init__(self):
            self.calls = []

        async def ainvoke(self, messages, **_kwargs):
            content = messages[0]["content"]
            self.calls.append(content)
            return LLMResponse(content=f"answer:{content}")

    llm = CountingLLM()
    store = LLMResponseCheckpointStore(tmp_path, model_name="model-a")
    inputs = [
        ([{"role": "user", "content": str(index)}], {"max_tokens": 8})
        for index in range(3)
    ]

    first = run_checkpointed_batch(
        llm=llm,
        inputs=inputs,
        keys=["a", "b", "c"],
        checkpoint_store=store,
        label="test",
    )
    second = run_checkpointed_batch(
        llm=llm,
        inputs=inputs,
        keys=["a", "b", "c"],
        checkpoint_store=store,
        label="test",
    )
    changed_inputs = list(inputs)
    changed_inputs[1] = (
        [{"role": "user", "content": "changed"}],
        {"max_tokens": 8},
    )
    third = run_checkpointed_batch(
        llm=llm,
        inputs=changed_inputs,
        keys=["a", "b", "c"],
        checkpoint_store=store,
        label="test",
    )

    assert [result.content for result in first] == [
        "answer:0",
        "answer:1",
        "answer:2",
    ]
    assert [result.content for result in second] == [
        "answer:0",
        "answer:1",
        "answer:2",
    ]
    assert [result.content for result in third] == [
        "answer:0",
        "answer:changed",
        "answer:2",
    ]
    assert sorted(llm.calls) == ["0", "1", "2", "changed"]


def test_checkpointed_batch_supports_processor_loaded_before_callback_upgrade(
    tmp_path,
    monkeypatch,
):
    class LegacyProcessor:
        closed = False

        def __init__(self, llm, inputs, **_kwargs):
            self.llm = llm
            self.inputs = inputs
            self._sync_loop = None

        def get_results(self):
            assert "on_result" not in inspect.signature(
                self.get_results
            ).parameters
            return [
                LLMResponse(content=f"legacy:{messages[0]['content']}")
                for messages, _kwargs in self.inputs
            ]

        def close_sync_loop(self):
            type(self).closed = True

    monkeypatch.setattr(
        "evaluator.evaluate_utils.llm_async_processor.LLMAsyncProcessor",
        LegacyProcessor,
    )
    inputs = [
        ([{"role": "user", "content": "one"}], {"max_tokens": 8}),
        ([{"role": "user", "content": "two"}], {"max_tokens": 8}),
    ]
    store = LLMResponseCheckpointStore(tmp_path, model_name="legacy-model")

    results = run_checkpointed_batch(
        llm=SimpleNamespace(model="legacy-model"),
        inputs=inputs,
        keys=["one", "two"],
        checkpoint_store=store,
        label="legacy test",
    )

    assert [result.content for result in results] == [
        "legacy:one",
        "legacy:two",
    ]
    assert LegacyProcessor.closed is True
