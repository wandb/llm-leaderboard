from pathlib import Path
from types import SimpleNamespace
import asyncio
import sys
import time

from omegaconf import OmegaConf


REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPTS_ROOT = REPO_ROOT / "scripts"
BFCL_ROOT = REPO_ROOT / "scripts" / "evaluator" / "evaluate_utils" / "bfcl_pkg"
sys.path.insert(0, str(SCRIPTS_ROOT))
sys.path.insert(0, str(BFCL_ROOT))


def _fake_cfg():
    return OmegaConf.create(
        {
            "batch_size": 1,
            "inference_interval": 0,
            "openai": {
                "http_timeout": {
                    "connect": 1,
                    "read": 10,
                    "write": 10,
                    "pool": 1,
                }
            },
        }
    )


async def _start_hanging_server():
    connection_tasks = []

    async def read_headers(reader):
        data = b""
        while b"\r\n\r\n" not in data:
            chunk = await reader.read(1024)
            if not chunk:
                break
            data += chunk
        return data

    async def handle(reader, writer):
        connection_tasks.append(asyncio.current_task())
        await read_headers(reader)
        await asyncio.Event().wait()

    server = await asyncio.start_server(handle, "127.0.0.1", 0)
    return server, connection_tasks


async def _start_429_server():
    async def read_headers(reader):
        data = b""
        while b"\r\n\r\n" not in data:
            chunk = await reader.read(1024)
            if not chunk:
                break
            data += chunk
        return data

    async def handle(reader, writer):
        await read_headers(reader)
        body = (
            b'{"error":{"message":"local rate limit storm",'
            b'"type":"rate_limit_error","code":"rate_limit_exceeded"}}'
        )
        writer.write(
            b"HTTP/1.1 429 Too Many Requests\r\n"
            b"Content-Type: application/json\r\n"
            + f"Content-Length: {len(body)}\r\n".encode()
            + b"\r\n"
            + body
        )
        await writer.drain()
        writer.close()
        await writer.wait_closed()

    server = await asyncio.start_server(handle, "127.0.0.1", 0)
    return server


def _patch_singletons(monkeypatch):
    import llm_inference_adapter as adapter_module
    import evaluator.evaluate_utils.llm_async_processor as processor_module

    instance = SimpleNamespace(config=_fake_cfg())
    monkeypatch.setattr(
        adapter_module.WandbConfigSingleton,
        "get_instance",
        staticmethod(lambda: instance),
    )
    monkeypatch.setattr(
        processor_module.WandbConfigSingleton,
        "get_instance",
        staticmethod(lambda: instance),
    )


async def _run_bfcl_case_against_base_url(base_url, *, backoff_max_time=5):
    from bfcl import _llm_response_generation as generation
    from evaluator.evaluate_utils.llm_async_processor import LLMAsyncProcessor
    from llm_inference_adapter import OpenAIClient

    client = OpenAIClient(
        api_key="test-key",
        base_url=base_url,
        model="local-probe-model",
    )
    processor = LLMAsyncProcessor(
        client,
        backoff_max_time=backoff_max_time,
        backoff_max_tries=50,
    )

    class HTTPProbeHandler:
        async def inference_async(self, test_case, include_input_log, exclude_state_log):
            await processor.process_single_async(
                [{"role": "user", "content": "hello"}],
                timeout=10,
                request_max_retries=0,
                max_tokens=16,
            )
            return [], {"latency": 0.0}

    return await generation.async_inference(
        HTTPProbeHandler(),
        {"id": "local_probe_case", "function": []},
        include_input_log=False,
        exclude_state_log=False,
        case_timeout_sec=0.5,
        case_timeout_retries=0,
    )


def test_bfcl_wait_for_cancels_real_openai_client_silent_http(monkeypatch):
    _patch_singletons(monkeypatch)

    async def scenario():
        server, connection_tasks = await _start_hanging_server()
        host, port = server.sockets[0].getsockname()
        started = time.monotonic()
        try:
            result = await _run_bfcl_case_against_base_url(
                f"http://{host}:{port}/v1",
                backoff_max_time=5,
            )
        finally:
            server.close()
            await server.wait_closed()
            for task in connection_tasks:
                task.cancel()
            await asyncio.gather(*connection_tasks, return_exceptions=True)
        return result, time.monotonic() - started

    result, elapsed = asyncio.run(scenario())

    assert elapsed < 6.0
    assert result["error"] == "bfcl_case_timeout"
    assert result["timeout"] is True


def test_bfcl_wait_for_cancels_real_openai_client_429_storm(monkeypatch):
    _patch_singletons(monkeypatch)

    async def scenario():
        server = await _start_429_server()
        host, port = server.sockets[0].getsockname()
        started = time.monotonic()
        try:
            result = await _run_bfcl_case_against_base_url(
                f"http://{host}:{port}/v1",
                backoff_max_time=5,
            )
        finally:
            server.close()
            await server.wait_closed()
        return result, time.monotonic() - started

    result, elapsed = asyncio.run(scenario())

    assert elapsed < 6.0
    assert result["error"] == "bfcl_case_timeout"
    assert result["timeout"] is True
