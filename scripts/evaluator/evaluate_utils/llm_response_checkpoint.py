from __future__ import annotations

import asyncio
import hashlib
import inspect
import json
import os
import re
from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Any, Iterable

from llm_inference_adapter import LLMResponse, ToolCall


def _json_default(value: Any) -> Any:
    if is_dataclass(value):
        return asdict(value)
    if hasattr(value, "model_dump"):
        return value.model_dump(mode="json")
    return repr(value)


def request_fingerprint(
    *,
    key: str,
    messages: Any,
    kwargs: Any,
    model_name: str,
) -> str:
    encoded = json.dumps(
        {
            "key": key,
            "messages": messages,
            "kwargs": kwargs,
            "model_name": model_name,
        },
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        default=_json_default,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def response_to_dict(response: LLMResponse) -> dict[str, Any]:
    return {
        "content": response.content,
        "reasoning_content": response.reasoning_content,
        "reasoning": response.reasoning,
        "reasoning_details": response.reasoning_details,
        "response_items": response.response_items,
        "tool_calls": [
            asdict(call) if is_dataclass(call) else call
            for call in (response.tool_calls or [])
        ],
        "prompt_tokens": response.prompt_tokens,
        "completion_tokens": response.completion_tokens,
        "finish_reason": response.finish_reason,
        "content_was_none": response.content_was_none,
    }


def response_from_dict(payload: dict[str, Any]) -> LLMResponse:
    raw_tool_calls = payload.get("tool_calls") or None
    tool_calls = None
    if raw_tool_calls is not None:
        tool_calls = [
            ToolCall(**call) if isinstance(call, dict) else call
            for call in raw_tool_calls
        ]
    return LLMResponse(
        content=str(payload.get("content") or ""),
        reasoning_content=str(payload.get("reasoning_content") or ""),
        reasoning=payload.get("reasoning"),
        reasoning_details=payload.get("reasoning_details"),
        response_items=payload.get("response_items"),
        tool_calls=tool_calls,
        prompt_tokens=payload.get("prompt_tokens"),
        completion_tokens=payload.get("completion_tokens"),
        finish_reason=payload.get("finish_reason"),
        content_was_none=bool(payload.get("content_was_none", False)),
    )


def _safe_component(value: str) -> str:
    component = re.sub(r"[^A-Za-z0-9_.-]+", "_", value).strip("._")
    return component or "item"


class LLMResponseCheckpointStore:
    """Atomic, request-bound checkpoints for large evaluator batches."""

    def __init__(self, root: Path, *, model_name: str) -> None:
        self.root = root
        self.model_name = str(model_name)

    def _path(self, key: str) -> Path:
        digest = hashlib.sha256(key.encode("utf-8")).hexdigest()[:16]
        return self.root / f"{_safe_component(key)[:96]}-{digest}.json"

    def load(
        self,
        key: str,
        *,
        messages: Any,
        kwargs: Any,
    ) -> LLMResponse | None:
        path = self._path(key)
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return None
        expected = request_fingerprint(
            key=key,
            messages=messages,
            kwargs=kwargs,
            model_name=self.model_name,
        )
        if (
            not isinstance(payload, dict)
            or payload.get("schema_version") != 1
            or payload.get("key") != key
            or payload.get("request_fingerprint") != expected
            or not isinstance(payload.get("response"), dict)
        ):
            return None
        return response_from_dict(payload["response"])

    def save(
        self,
        key: str,
        response: LLMResponse,
        *,
        messages: Any,
        kwargs: Any,
    ) -> None:
        path = self._path(key)
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "schema_version": 1,
            "key": key,
            "request_fingerprint": request_fingerprint(
                key=key,
                messages=messages,
                kwargs=kwargs,
                model_name=self.model_name,
            ),
            "response": response_to_dict(response),
        }
        temporary = path.with_suffix(".json.tmp")
        temporary.write_text(
            json.dumps(
                payload,
                ensure_ascii=False,
                indent=2,
                default=_json_default,
            )
            + "\n",
            encoding="utf-8",
        )
        os.replace(temporary, path)


class JSONItemCheckpointStore:
    """Atomic JSON checkpoints whose reuse is tied to the exact request."""

    def __init__(self, root: Path, *, model_name: str) -> None:
        self.root = root
        self.model_name = str(model_name)

    def _path(self, key: str) -> Path:
        digest = hashlib.sha256(key.encode("utf-8")).hexdigest()[:16]
        return self.root / f"{_safe_component(key)[:96]}-{digest}.json"

    def load(self, key: str, *, request: Any) -> Any | None:
        path = self._path(key)
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return None
        expected = request_fingerprint(
            key=key,
            messages=request,
            kwargs={},
            model_name=self.model_name,
        )
        if (
            not isinstance(payload, dict)
            or payload.get("schema_version") != 1
            or payload.get("key") != key
            or payload.get("request_fingerprint") != expected
            or "value" not in payload
        ):
            return None
        return payload["value"]

    def save(self, key: str, value: Any, *, request: Any) -> None:
        path = self._path(key)
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "schema_version": 1,
            "key": key,
            "request_fingerprint": request_fingerprint(
                key=key,
                messages=request,
                kwargs={},
                model_name=self.model_name,
            ),
            "value": value,
        }
        temporary = path.with_suffix(".json.tmp")
        temporary.write_text(
            json.dumps(
                payload,
                ensure_ascii=False,
                indent=2,
                default=_json_default,
            )
            + "\n",
            encoding="utf-8",
        )
        os.replace(temporary, path)


def default_checkpoint_root(run: Any, benchmark: str) -> Path:
    run_id = _safe_component(str(getattr(run, "id", "local")))
    return Path("outputs/evaluator_item_checkpoints") / run_id / _safe_component(
        benchmark
    )


def run_checkpointed_batch(
    *,
    llm: Any,
    inputs: list[tuple[Any, dict[str, Any]]],
    keys: Iterable[str],
    checkpoint_store: LLMResponseCheckpointStore,
    processor_kwargs: dict[str, Any] | None = None,
    label: str = "LLM batch",
) -> list[LLMResponse]:
    """Run missing requests and atomically persist each completion."""
    from evaluator.evaluate_utils.llm_async_processor import LLMAsyncProcessor

    request_keys = [str(key) for key in keys]
    if len(request_keys) != len(inputs):
        raise ValueError(
            f"{label}: request key count {len(request_keys)} does not match "
            f"input count {len(inputs)}"
        )

    responses: list[LLMResponse | None] = [None] * len(inputs)
    missing_inputs = []
    missing_global_indices = []
    for global_index, (key, request) in enumerate(zip(request_keys, inputs)):
        messages, kwargs = request
        cached = checkpoint_store.load(key, messages=messages, kwargs=kwargs)
        if cached is None:
            missing_global_indices.append(global_index)
            missing_inputs.append(request)
        else:
            responses[global_index] = cached

    print(
        f"{label} request checkpoints: "
        f"{len(inputs) - len(missing_inputs)} reused, "
        f"{len(missing_inputs)} pending",
        flush=True,
    )
    if missing_inputs:
        processor = LLMAsyncProcessor(
            llm=llm,
            inputs=missing_inputs,
            **(processor_kwargs or {}),
        )

        def save_completed(local_index: int, response: LLMResponse) -> None:
            global_index = missing_global_indices[local_index]
            key = request_keys[global_index]
            messages, kwargs = inputs[global_index]
            checkpoint_store.save(
                key,
                response,
                messages=messages,
                kwargs=kwargs,
            )
            responses[global_index] = response

        try:
            if "on_result" in inspect.signature(
                processor.get_results
            ).parameters:
                processor.get_results(on_result=save_completed)
            else:
                # A long-running run may import this helper after loading an
                # older processor class. Keep that paid run alive; fresh runs
                # use the incremental callback path above.
                legacy_results = processor.get_results()
                for local_index, response in enumerate(legacy_results):
                    save_completed(local_index, response)
        finally:
            # Runs started before loop-local clients were introduced can import
            # this helper later while still holding the old processor class.
            # Close that class's async client on its owning loop before closing
            # the loop itself, preventing delayed "event loop is closed" errors.
            if not hasattr(processor, "close_async_client"):
                async_client = getattr(processor.llm, "async_client", None)
                close = (
                    getattr(async_client, "aclose", None)
                    or getattr(async_client, "close", None)
                )
                loop = getattr(processor, "_sync_loop", None)
                if close is not None and loop is not None and loop.is_running():
                    async def close_legacy_client() -> None:
                        result = close()
                        if inspect.isawaitable(result):
                            await result

                    try:
                        asyncio.run_coroutine_threadsafe(
                            close_legacy_client(),
                            loop,
                        ).result(timeout=15)
                    except Exception as exc:
                        print(
                            f"{label}: legacy async client cleanup warning: "
                            f"{type(exc).__name__}: {exc}",
                            flush=True,
                        )
            processor.close_sync_loop()

    missing_count = sum(response is None for response in responses)
    if missing_count:
        raise RuntimeError(f"{label} inference incomplete: {missing_count} missing")
    return [response for response in responses if response is not None]
