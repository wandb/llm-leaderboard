from pathlib import Path
from types import SimpleNamespace
import asyncio
import sys
import time
from types import SimpleNamespace as SNS

import pytest


REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPTS_ROOT = REPO_ROOT / "scripts"
BFCL_ROOT = REPO_ROOT / "scripts" / "evaluator" / "evaluate_utils" / "bfcl_pkg"
sys.path.insert(0, str(SCRIPTS_ROOT))
sys.path.insert(0, str(BFCL_ROOT))


def test_bfcl_generation_resume_branch_uses_model_name_and_returns_handler():
    source = (
        REPO_ROOT
        / "scripts"
        / "evaluator"
        / "evaluate_utils"
        / "bfcl_pkg"
        / "bfcl"
        / "_llm_response_generation.py"
    ).read_text(encoding="utf-8")

    assert "handler = build_handler(args.model_name, args.temperature)" in source
    assert "previously generated for {args.model_name}" in source
    assert "previously generated for {args.model}" not in source
    assert "return handler" in source


def test_bfcl_generation_retries_case_timeout_before_scoring_failure(monkeypatch):
    from bfcl import _llm_response_generation as generation

    submitted = []
    written = []
    attempts = {}

    def fake_inference(handler, test_case, include_input_log, exclude_state_log):
        case_id = test_case["id"]
        submitted.append(case_id)
        attempts[case_id] = attempts.get(case_id, 0) + 1
        if case_id == "case_1" and attempts[case_id] == 1:
            time.sleep(0.2)
        return {"id": test_case["id"], "result": []}

    class FakeHandler:
        model_style = object()

        def write(self, result, result_dir, update_mode=False):
            written.append(result)

    monkeypatch.setattr(generation, "multi_threaded_inference", fake_inference)

    args = SimpleNamespace(
        num_threads=1,
        include_input_log=False,
        exclude_state_log=False,
        result_dir=Path("/tmp/bfcl-timeout-test"),
        case_timeout_sec=0.05,
        request_timeout_sec=None,
        case_timeout_retries=1,
        progress_poll_sec=0.01,
    )

    generation.generate_results(
        args,
        "fake-model",
        [{"id": "case_1"}, {"id": "case_2"}],
        handler=FakeHandler(),
    )

    assert submitted == ["case_1", "case_1", "case_2"]
    assert [entry["id"] for entry in written] == ["case_1", "case_2"]
    assert written[0] == {"id": "case_1", "result": []}
    assert written[1] == {"id": "case_2", "result": []}


def test_bfcl_generation_scores_case_timeout_after_retry_exhausted(monkeypatch):
    from bfcl import _llm_response_generation as generation

    submitted = []
    written = []

    def fake_inference(handler, test_case, include_input_log, exclude_state_log):
        submitted.append(test_case["id"])
        if test_case["id"] == "case_1":
            time.sleep(0.2)
        return {"id": test_case["id"], "result": []}

    class FakeHandler:
        model_style = object()

        def write(self, result, result_dir, update_mode=False):
            written.append(result)

    monkeypatch.setattr(generation, "multi_threaded_inference", fake_inference)

    args = SimpleNamespace(
        num_threads=1,
        include_input_log=False,
        exclude_state_log=False,
        result_dir=Path("/tmp/bfcl-timeout-test"),
        case_timeout_sec=0.05,
        request_timeout_sec=None,
        case_timeout_retries=1,
        progress_poll_sec=0.01,
    )

    generation.generate_results(
        args,
        "fake-model",
        [{"id": "case_1"}, {"id": "case_2"}],
        handler=FakeHandler(),
    )

    assert submitted == ["case_1", "case_1", "case_2"]
    assert [entry["id"] for entry in written] == ["case_1", "case_2"]
    assert written[0]["timeout"] is True
    assert written[0]["error"] == "bfcl_case_timeout"
    assert written[0]["timeout_attempts"] == 2
    assert written[0]["timeout_retries"] == 1
    assert "BFCL case timeout" in written[0]["result"]
    assert written[1] == {"id": "case_2", "result": []}


def test_bfcl_async_generation_respects_num_threads():
    from bfcl import _llm_response_generation as generation

    active = 0
    max_active = 0
    written = []

    class FakeAsyncHandler:
        async def inference_async(self, test_case, include_input_log, exclude_state_log):
            nonlocal active, max_active
            active += 1
            max_active = max(max_active, active)
            try:
                await asyncio.sleep(0.02)
                return [], {"latency": 0.02}
            finally:
                active -= 1

        def write(self, result, result_dir, update_mode=False):
            written.append(result)

    args = SimpleNamespace(
        model_name="fake-async-model",
        num_threads=2,
        include_input_log=False,
        exclude_state_log=False,
        result_dir=Path("/tmp/bfcl-async-concurrency-test"),
        case_timeout_sec=1,
        request_timeout_sec=None,
        case_timeout_retries=0,
    )
    test_cases = [{"id": f"case_{i}", "function": []} for i in range(5)]

    asyncio.run(generation.async_generate_results(args, FakeAsyncHandler(), test_cases))

    assert max_active <= 2
    assert sorted(entry["id"] for entry in written) == [f"case_{i}" for i in range(5)]


def test_bfcl_async_generation_scores_case_timeout_after_retry_exhausted():
    from bfcl import _llm_response_generation as generation

    attempts = {}
    written = []

    class FakeAsyncHandler:
        async def inference_async(self, test_case, include_input_log, exclude_state_log):
            case_id = test_case["id"]
            attempts[case_id] = attempts.get(case_id, 0) + 1
            if case_id == "case_1":
                await asyncio.sleep(0.2)
            return [], {"latency": 0.0}

        def write(self, result, result_dir, update_mode=False):
            written.append(result)

    args = SimpleNamespace(
        model_name="fake-async-model",
        num_threads=1,
        include_input_log=False,
        exclude_state_log=False,
        result_dir=Path("/tmp/bfcl-async-timeout-test"),
        case_timeout_sec=0.05,
        request_timeout_sec=None,
        case_timeout_retries=1,
    )

    asyncio.run(
        generation.async_generate_results(
            args,
            FakeAsyncHandler(),
            [{"id": "case_1", "function": []}, {"id": "case_2", "function": []}],
        )
    )

    assert attempts == {"case_1": 2, "case_2": 1}
    assert [entry["id"] for entry in written] == ["case_1", "case_2"]
    assert written[0]["timeout"] is True
    assert written[0]["error"] == "bfcl_case_timeout"
    assert written[0]["timeout_attempts"] == 2
    assert written[0]["timeout_retries"] == 1
    assert "BFCL case timeout" in written[0]["result"]
    assert written[1] == {"id": "case_2", "result": [], "latency": 0.0}


def test_bfcl_async_generation_fail_fast_on_consecutive_inference_errors():
    from bfcl import _llm_response_generation as generation

    written = []

    class FakeAsyncHandler:
        async def inference_async(self, test_case, include_input_log, exclude_state_log):
            raise RuntimeError("provider exhausted")

        def write(self, result, result_dir, update_mode=False):
            written.append(result)

    args = SimpleNamespace(
        model_name="fake-async-model",
        num_threads=1,
        include_input_log=False,
        exclude_state_log=False,
        result_dir=Path("/tmp/bfcl-async-consecutive-error-test"),
        case_timeout_sec=1,
        request_timeout_sec=None,
        case_timeout_retries=0,
        consecutive_failure_fail_fast=3,
    )

    with pytest.raises(generation.BFCLStalledError, match="3 consecutive inference failures"):
        asyncio.run(
            generation.async_generate_results(
                args,
                FakeAsyncHandler(),
                [{"id": f"case_{i}", "function": []} for i in range(5)],
            )
        )

    assert [entry["id"] for entry in written] == ["case_0", "case_1", "case_2"]
    assert all(str(entry["result"]).startswith("Error during inference:") for entry in written)


def test_bfcl_effective_case_timeout_ignores_request_timeout():
    from bfcl import _llm_response_generation as generation

    args = SimpleNamespace(case_timeout_sec=600, request_timeout_sec=300)
    assert generation._effective_case_timeout_sec(args) == 600

    args = SimpleNamespace(case_timeout_sec=None, request_timeout_sec=300)
    assert generation._effective_case_timeout_sec(args) is None

    args = SimpleNamespace(case_timeout_sec=120, request_timeout_sec=300)
    assert generation._effective_case_timeout_sec(args) == 120


def test_bfcl_collect_test_cases_retries_timeout_results(tmp_path):
    from bfcl import _llm_response_generation as generation

    model_name = "fake/model"
    result_dir = tmp_path / "result"
    model_dir = result_dir / model_name.replace("/", "_")
    model_dir.mkdir(parents=True)
    result_file = model_dir / "BFCL_v3_simple_result.json"
    result_file.write_text(
        "\n".join(
            [
                '{"id": "simple_0", "result": [], "error": "bfcl_case_timeout"}',
                '{"id": "simple_1", "result": []}',
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    args = SimpleNamespace(
        result_dir=result_dir,
        allow_overwrite=False,
        run_ids=False,
        artifacts_path="",
        retry_failed_cases=True,
    )

    test_cases = generation.collect_test_cases(
        args,
        model_name,
        ["simple"],
        ["BFCL_v3_simple.json"],
        [
            {"id": "simple_0", "function": []},
            {"id": "simple_1", "function": []},
            {"id": "simple_2", "function": []},
        ],
    )

    assert [case["id"] for case in test_cases] == ["simple_0", "simple_2"]


def test_openai_client_forwards_per_request_timeout_and_retries():
    from llm_inference_adapter import OpenAIClient

    captured = {}

    class FakeCompletions:
        async def create(self, **params):
            captured["params"] = params
            return SNS(
                choices=[
                    SNS(
                        message=SNS(content="ok", tool_calls=None),
                        finish_reason="stop",
                    )
                ],
                usage=SNS(prompt_tokens=1, completion_tokens=2),
            )

    class FakeAsyncClient:
        def __init__(self):
            self.chat = SNS(completions=FakeCompletions())
            self.max_retries = None

        def with_options(self, **kwargs):
            self.max_retries = kwargs.get("max_retries")
            return self

    client = OpenAIClient.__new__(OpenAIClient)
    client.async_client = FakeAsyncClient()
    client.model = "fake-model"
    client.kwargs = {}
    client.allowed_params = {"max_tokens"}
    client.param_mapping = {}

    asyncio.run(
        client.ainvoke(
            [{"role": "user", "content": "hello"}],
            timeout=12.5,
            request_max_retries=1,
            max_tokens=16,
        )
    )

    assert captured["params"]["timeout"] == 12.5
    assert captured["params"]["max_tokens"] == 16
    assert client.async_client.max_retries == 1


def test_llm_async_processor_uses_instance_backoff_policy(monkeypatch):
    import evaluator.evaluate_utils.llm_async_processor as processor_module
    from evaluator.evaluate_utils.llm_async_processor import LLMAsyncProcessor

    class FakeConfig(dict):
        inference_interval = 0

    monkeypatch.setattr(
        processor_module.WandbConfigSingleton,
        "get_instance",
        staticmethod(lambda: SNS(config=FakeConfig(batch_size=1))),
    )

    class TimeoutLLM:
        async def ainvoke(self, messages, **kwargs):
            raise TimeoutError("provider stalled")

    processor = LLMAsyncProcessor(
        TimeoutLLM(),
        backoff_max_time=0.2,
        backoff_max_tries=50,
    )
    started = time.monotonic()
    with pytest.raises(TimeoutError):
        asyncio.run(processor.process_single_async([{"role": "user", "content": "x"}]))

    assert time.monotonic() - started < 2.0


def test_bfcl_openai_compatible_handler_wires_request_controls():
    source = (
        REPO_ROOT
        / "scripts"
        / "evaluator"
        / "evaluate_utils"
        / "bfcl_pkg"
        / "bfcl"
        / "model_handler"
        / "openai_compatible_handler.py"
    ).read_text(encoding="utf-8")

    assert 'OmegaConf.select(cfg, "bfcl.request_timeout_sec"' in source
    assert 'OmegaConf.select(cfg, "bfcl.request_max_retries"' in source
    assert "**self._request_control_kwargs()" in source
