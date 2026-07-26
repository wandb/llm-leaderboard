import asyncio
import json
import sys
import types
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = ROOT / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))


if "mistralai" not in sys.modules:
    mistralai_stub = types.ModuleType("mistralai")
    mistralai_stub.Mistral = object
    sys.modules["mistralai"] = mistralai_stub

from llm_inference_adapter import (
    AnthropicClient,
    _LoopLocalAsyncClientMixin,
    _create_owned_async_openai_client,
    _normalize_anthropic_messages,
    _normalize_openai_responses_input,
    _normalize_openai_responses_tools,
    _parse_tool_call_arguments,
    _parse_structured_response_content,
)
from pydantic import BaseModel, Field


class StructuredFunctionCalls(BaseModel):
    function_calls: list[dict] = Field(default_factory=list)
    unavailable_reason: str = ""


def test_anthropic_client_maps_effort_to_output_config(monkeypatch):
    captured = {}

    class FakeMessages:
        def create(self, **kwargs):
            captured.update(kwargs)
            return types.SimpleNamespace(
                content=[types.SimpleNamespace(type="text", text="ok")]
            )

    class FakeAnthropic:
        def __init__(self, **_kwargs):
            self.messages = FakeMessages()

    monkeypatch.setattr("llm_inference_adapter.Anthropic", FakeAnthropic)
    client = AnthropicClient(
        api_key="test",
        model="claude-fable-5",
        max_tokens=128_000,
        effort="xhigh",
    )

    response = asyncio.run(client.ainvoke([{"role": "user", "content": "hello"}]))

    assert response.content == "ok"
    assert captured["model"] == "claude-fable-5"
    assert captured["max_tokens"] == 128_000
    assert captured["output_config"] == {"effort": "xhigh"}
    assert "effort" not in captured
    assert "thinking" not in captured


def test_openai_responses_normalizes_chat_tool_schema_and_history():
    tools = [
        {
            "type": "function",
            "function": {
                "name": "weather",
                "description": "Forecast",
                "parameters": {"type": "object", "properties": {}},
            },
        }
    ]
    assert _normalize_openai_responses_tools(tools) == [
        {
            "type": "function",
            "name": "weather",
            "description": "Forecast",
            "parameters": {"type": "object", "properties": {}},
        }
    ]

    history = _normalize_openai_responses_input(
        [
            {"role": "system", "content": "policy"},
            {
                "role": "assistant",
                "content": None,
                "tool_calls": [
                    {
                        "id": "call-1",
                        "type": "function",
                        "function": {
                            "name": "weather",
                            "arguments": '{"city":"Taipei"}',
                        },
                    }
                ],
            },
            {"role": "tool", "tool_call_id": "call-1", "content": "sunny"},
        ]
    )
    assert history == [
        {"role": "developer", "content": "policy"},
        {
            "type": "function_call",
            "call_id": "call-1",
            "name": "weather",
            "arguments": '{"city":"Taipei"}',
        },
        {
            "type": "function_call_output",
            "call_id": "call-1",
            "output": "sunny",
        },
    ]


def test_anthropic_normalizes_chat_tool_history():
    messages, system = _normalize_anthropic_messages(
        [
            {"role": "system", "content": "policy"},
            {
                "role": "assistant",
                "content": None,
                "tool_calls": [
                    {
                        "id": "call-1",
                        "type": "function",
                        "function": {
                            "name": "weather",
                            "arguments": '{"city":"Taipei"}',
                        },
                    }
                ],
            },
            {"role": "tool", "tool_call_id": "call-1", "content": "sunny"},
        ]
    )
    assert system == "policy"
    assert messages[0] == {
        "role": "assistant",
        "content": [
            {
                "type": "tool_use",
                "id": "call-1",
                "name": "weather",
                "input": {"city": "Taipei"},
            }
        ],
    }
    assert messages[1]["content"][0]["type"] == "tool_result"


def test_anthropic_client_sends_tools_and_parses_tool_use(monkeypatch):
    captured = {}

    class FakeMessages:
        def create(self, **kwargs):
            captured.update(kwargs)
            return types.SimpleNamespace(
                content=[
                    types.SimpleNamespace(
                        type="tool_use",
                        id="call-7",
                        name="weather",
                        input={"city": "Taipei"},
                    )
                ],
                usage=types.SimpleNamespace(input_tokens=11, output_tokens=7),
            )

    class FakeAnthropic:
        def __init__(self, **_kwargs):
            self.messages = FakeMessages()

    monkeypatch.setattr("llm_inference_adapter.Anthropic", FakeAnthropic)
    client = AnthropicClient(api_key="test", model="claude-test", max_tokens=64)
    response = asyncio.run(
        client.ainvoke(
            [{"role": "user", "content": "weather?"}],
            tools=[
                {
                    "name": "weather",
                    "description": "Forecast",
                    "input_schema": {"type": "object", "properties": {}},
                }
            ],
        )
    )

    assert captured["tools"][0]["name"] == "weather"
    assert response.tool_calls[0].name == "weather"
    assert response.tool_calls[0].arguments == {"city": "Taipei"}
    assert response.prompt_tokens == 11
    assert response.completion_tokens == 7


def test_anthropic_client_drops_top_p_when_temperature_is_configured(monkeypatch):
    captured = {}

    class FakeMessages:
        def create(self, **kwargs):
            captured.update(kwargs)
            return types.SimpleNamespace(
                content=[types.SimpleNamespace(type="text", text="ok")],
                usage=types.SimpleNamespace(input_tokens=3, output_tokens=1),
            )

    class FakeAnthropic:
        def __init__(self, **_kwargs):
            self.messages = FakeMessages()

    monkeypatch.setattr("llm_inference_adapter.Anthropic", FakeAnthropic)
    client = AnthropicClient(api_key="test", model="claude-test", max_tokens=64)

    asyncio.run(
        client.ainvoke(
            [{"role": "user", "content": "hello"}],
            temperature=0.01,
            top_p=1.0,
        )
    )

    assert captured["temperature"] == 0.01
    assert "top_p" not in captured


def test_parse_tool_call_arguments_accepts_valid_json():
    parsed = _parse_tool_call_arguments('{"city": "Tokyo", "days": 3}')
    assert parsed == {"city": "Tokyo", "days": 3}


def test_parse_tool_call_arguments_repairs_missing_closing_brace():
    parsed = _parse_tool_call_arguments('{"city": "Tokyo", "days": 3')
    assert parsed == {"city": "Tokyo", "days": 3}


def test_parse_tool_call_arguments_repairs_code_fence_and_trailing_comma():
    parsed = _parse_tool_call_arguments(
        '```json\n{"city": "Tokyo", "days": 3,}\n```'
    )
    assert parsed == {"city": "Tokyo", "days": 3}


def test_parse_tool_call_arguments_repairs_missing_comma_between_fields():
    parsed = _parse_tool_call_arguments(
        '{"city": "Tokyo" "days": 3}'
    )
    assert parsed == {"city": "Tokyo", "days": 3}


def test_parse_tool_call_arguments_accepts_python_literal_style_dict():
    parsed = _parse_tool_call_arguments(
        "{'city': 'Tokyo', 'weekend': True, 'note': None}"
    )
    assert parsed == {"city": "Tokyo", "weekend": True, "note": None}


def test_parse_tool_call_arguments_raises_on_irreparable_input():
    try:
        _parse_tool_call_arguments('{"city": Tokyo ???')
    except json.JSONDecodeError:
        return
    assert False, "Expected JSONDecodeError for irreparable tool call arguments"


def test_parse_structured_response_content_repairs_extra_outer_opening_brace():
    parsed, normalized = _parse_structured_response_content(
        StructuredFunctionCalls,
        '{{"function_calls": [], "unavailable_reason": "no function exists"}',
    )

    assert normalized == '{"function_calls": [], "unavailable_reason": "no function exists"}'
    assert parsed.function_calls == []
    assert parsed.unavailable_reason == "no function exists"


def test_parse_structured_response_content_repairs_extra_outer_closing_brace():
    parsed, normalized = _parse_structured_response_content(
        StructuredFunctionCalls,
        '{"function_calls": [], "unavailable_reason": ""}}',
    )

    assert normalized == '{"function_calls": [], "unavailable_reason": ""}'
    assert parsed.function_calls == []


def test_parse_structured_response_content_repairs_unclosed_code_fence():
    parsed, normalized = _parse_structured_response_content(
        StructuredFunctionCalls,
        '```{\n    "function_calls": [],\n    "unavailable_reason": ""\n}',
    )

    assert normalized == '{\n    "function_calls": [],\n    "unavailable_reason": ""\n}'
    assert parsed.function_calls == []


def test_parse_structured_response_content_repairs_unclosed_json_code_fence():
    parsed, normalized = _parse_structured_response_content(
        StructuredFunctionCalls,
        '```json\n{"function_calls": [], "unavailable_reason": ""}',
    )

    assert normalized == '{"function_calls": [], "unavailable_reason": ""}'
    assert parsed.function_calls == []


def test_loop_local_async_client_reopens_cleanly_across_event_loops():
    created = []

    class FakeAsyncClient:
        def __init__(self):
            self.is_closed = False
            created.append(self)

        async def close(self):
            self.is_closed = True

    class Holder(_LoopLocalAsyncClientMixin):
        def __init__(self):
            self._initialize_loop_local_async_client(FakeAsyncClient)

    holder = Holder()

    async def use_and_close():
        client = holder._get_async_client()
        await holder.aclose()
        return client

    first = asyncio.run(use_and_close())
    second = asyncio.run(use_and_close())

    assert first is not second
    assert first.is_closed
    assert second.is_closed


def test_openai_async_client_uses_explicitly_owned_http_transport(monkeypatch):
    created_http_clients = []

    class FakeHttpClient:
        def __init__(self, **kwargs):
            self.kwargs = kwargs
            self.is_closed = False
            created_http_clients.append(self)

        async def aclose(self):
            self.is_closed = True

    class FakeOpenAIClient:
        def __init__(self, **kwargs):
            self.kwargs = kwargs
            self._client = kwargs["http_client"]

        async def close(self):
            await self._client.aclose()

    monkeypatch.setattr(
        "llm_inference_adapter.httpx.AsyncClient",
        FakeHttpClient,
    )

    client = _create_owned_async_openai_client(
        FakeOpenAIClient,
        {"api_key": "test", "timeout": "configured-timeout"},
    )

    assert len(created_http_clients) == 1
    assert client.kwargs["http_client"] is created_http_clients[0]
    assert created_http_clients[0].kwargs == {
        "timeout": "configured-timeout",
        "limits": __import__("openai").DEFAULT_CONNECTION_LIMITS,
        "follow_redirects": True,
    }
    asyncio.run(client.close())
    assert created_http_clients[0].is_closed


def test_openai_async_client_preserves_caller_owned_http_transport(monkeypatch):
    caller_http_client = object()

    class FakeOpenAIClient:
        def __init__(self, **kwargs):
            self.kwargs = kwargs

    def fail_if_created(**kwargs):
        raise AssertionError("unexpected replacement HTTP client")

    monkeypatch.setattr(
        "llm_inference_adapter.httpx.AsyncClient",
        fail_if_created,
    )

    client = _create_owned_async_openai_client(
        FakeOpenAIClient,
        {"api_key": "test", "http_client": caller_http_client},
    )

    assert client.kwargs["http_client"] is caller_http_client


def test_openai_async_client_closes_transport_when_construction_fails(monkeypatch):
    created_http_clients = []

    class FakeHttpClient:
        def __init__(self, **kwargs):
            self.is_closed = False
            created_http_clients.append(self)

        async def aclose(self):
            self.is_closed = True

    class BrokenOpenAIClient:
        def __init__(self, **kwargs):
            raise ValueError("invalid client configuration")

    monkeypatch.setattr(
        "llm_inference_adapter.httpx.AsyncClient",
        FakeHttpClient,
    )

    with pytest.raises(ValueError, match="invalid client configuration"):
        _create_owned_async_openai_client(
            BrokenOpenAIClient,
            {"api_key": "test"},
        )

    assert len(created_http_clients) == 1
    assert created_http_clients[0].is_closed


def test_loop_local_async_client_rejects_unclosed_cross_loop_reuse():
    class FakeAsyncClient:
        is_closed = False

        async def close(self):
            self.is_closed = True

    class Holder(_LoopLocalAsyncClientMixin):
        def __init__(self):
            self._initialize_loop_local_async_client(FakeAsyncClient)

    holder = Holder()

    async def use_without_close():
        return holder._get_async_client()

    first = asyncio.run(use_without_close())
    with pytest.raises(RuntimeError, match="different event loop"):
        asyncio.run(use_without_close())

    assert not first.is_closed
