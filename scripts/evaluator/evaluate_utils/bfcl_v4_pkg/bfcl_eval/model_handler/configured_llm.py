import copy
import json
import threading
import time
from typing import Any

from config_singleton import WandbConfigSingleton
from evaluator.evaluate_utils.llm_async_processor import LLMAsyncProcessor
from evaluator.evaluate_utils.provider_rate_limiter import (
    get_provider_request_rate_limiter,
)
from omegaconf import OmegaConf

from bfcl_eval.constants.enums import ModelStyle
from bfcl_eval.model_handler.api_inference.openai_completion import (
    OpenAICompletionsHandler,
)
from bfcl_eval.model_handler.base_handler import BaseHandler


def _plain_dict(value: Any) -> dict:
    if value is None:
        return {}
    if isinstance(value, dict):
        return copy.deepcopy(value)
    try:
        return OmegaConf.to_container(value, resolve=True) or {}
    except Exception:
        return {}


def _deep_merge(base: dict, override: dict) -> dict:
    merged = copy.deepcopy(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _deep_merge(merged[key], value)
        else:
            merged[key] = copy.deepcopy(value)
    return merged


def _positive_float(value: Any) -> float | None:
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return None
    return parsed if parsed > 0 else None


def _nonnegative_int(value: Any) -> int | None:
    try:
        return max(0, int(value))
    except (TypeError, ValueError):
        return None


def _model_style_for_api(api_type: Any) -> ModelStyle:
    normalized = str(api_type or "").strip().lower()
    if normalized in {"openai_responses", "xai_responses"}:
        return ModelStyle.OPENAI_RESPONSES
    if normalized == "anthropic":
        return ModelStyle.ANTHROPIC
    return ModelStyle.OPENAI_COMPLETIONS


class ConfiguredLLMHandler(OpenAICompletionsHandler):
    """BFCL v4 handler backed by the leaderboard's configured LLM adapter.

    The provider, endpoint, credentials, model name, extra_body, timeout, and
    retry behavior remain YAML-driven. This keeps API and OpenAI-compatible OSS
    evaluation on one path without importing every optional upstream SDK.
    """

    def __init__(
        self,
        model_name,
        temperature,
        registry_name,
        is_fc_model,
        **kwargs,
    ) -> None:
        BaseHandler.__init__(
            self,
            model_name=model_name,
            temperature=temperature,
            registry_name=registry_name,
            is_fc_model=is_fc_model,
            **kwargs,
        )

        instance = WandbConfigSingleton.get_instance()
        if instance is None or getattr(instance, "llm", None) is None:
            raise RuntimeError(
                "ConfiguredLLMHandler requires an initialized "
                "WandbConfigSingleton with an LLM client."
            )
        cfg = instance.config
        self.api_type = str(getattr(cfg, "api", "")).strip().lower()
        self.model_style = _model_style_for_api(self.api_type)
        self.provider_model_name = str(cfg.model.pretrained_model_name_or_path)
        self.generator_config = _deep_merge(
            _plain_dict(getattr(cfg, "generator", {})),
            _plain_dict(OmegaConf.select(cfg, "bfcl.generator_config", default={})),
        )
        self.max_tokens = self.generator_config.pop("max_tokens", None)

        self.request_timeout_sec = _positive_float(
            OmegaConf.select(cfg, "bfcl.request_timeout_sec", default=None)
        )
        self.request_max_retries = _nonnegative_int(
            OmegaConf.select(cfg, "bfcl.request_max_retries", default=1)
        )
        min_interval = _positive_float(
            OmegaConf.select(
                cfg, "bfcl.provider_min_request_interval_sec", default=0
            )
        ) or 0.0
        jitter = _positive_float(
            OmegaConf.select(cfg, "bfcl.provider_request_jitter_sec", default=0)
        ) or 0.0
        rate_limit_key = str(
            OmegaConf.select(
                cfg,
                "bfcl.provider_rate_limit_key",
                default=f"bfcl-v4:{self.provider_model_name}",
            )
        )
        self.provider_request_limiter = get_provider_request_rate_limiter(
            rate_limit_key,
            min_interval_sec=min_interval,
            jitter_sec=jitter,
        )

        self.llm_processor = LLMAsyncProcessor(
            instance.llm,
            backoff_max_time=_positive_float(
                OmegaConf.select(cfg, "bfcl.backoff_max_time_sec", default=90)
            ),
            backoff_max_tries=_nonnegative_int(
                OmegaConf.select(cfg, "bfcl.backoff_max_tries", default=4)
            ),
            provider_rate_limit_enabled=False,
        )
        self._case_state = threading.local()

    def begin_case(self, case_timeout_sec: float | None) -> None:
        self._case_state.deadline = (
            time.monotonic() + case_timeout_sec
            if case_timeout_sec is not None and case_timeout_sec > 0
            else None
        )

    def end_case(self) -> None:
        self._case_state.deadline = None

    def _request_kwargs(self) -> dict:
        kwargs = {
            **self.generator_config,
            "model": self.provider_model_name,
        }
        if self.max_tokens is not None:
            kwargs["max_tokens"] = self.max_tokens
        request_timeout = self.request_timeout_sec
        deadline = getattr(self._case_state, "deadline", None)
        if deadline is not None:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise TimeoutError("BFCL v4 case deadline exceeded")
            request_timeout = (
                min(request_timeout, remaining)
                if request_timeout is not None
                else remaining
            )
        if request_timeout is not None:
            kwargs["timeout"] = max(0.1, request_timeout)
        if self.request_max_retries is not None:
            kwargs["request_max_retries"] = self.request_max_retries
        return kwargs

    def _query_FC(self, inference_data: dict):
        messages = inference_data["message"]
        tools = inference_data["tools"]
        inference_data["inference_input_log"] = {
            "message": repr(messages),
            "tools": tools,
        }
        kwargs = self._request_kwargs()
        if tools:
            kwargs["tools"] = tools

        started_at = time.monotonic()
        self.provider_request_limiter.wait()
        response = self.llm_processor.process_single(messages, **kwargs)
        return response, time.monotonic() - started_at

    def _query_prompting(self, inference_data: dict):
        messages = inference_data["message"]
        inference_data["inference_input_log"] = {"message": repr(messages)}

        started_at = time.monotonic()
        self.provider_request_limiter.wait()
        response = self.llm_processor.process_single(
            messages, **self._request_kwargs()
        )
        return response, time.monotonic() - started_at

    def _parse_query_response_FC(self, api_response: Any) -> dict:
        tool_calls = list(api_response.tool_calls or [])
        if tool_calls:
            model_responses = [
                {tool_call.name: json.dumps(tool_call.arguments, ensure_ascii=False)}
                for tool_call in tool_calls
            ]
            assistant_message = {
                "role": "assistant",
                "content": api_response.content or None,
                "tool_calls": [
                    {
                        "id": tool_call.id,
                        "type": tool_call.type,
                        "function": {
                            "name": tool_call.name,
                            "arguments": json.dumps(
                                tool_call.arguments, ensure_ascii=False
                            ),
                        },
                    }
                    for tool_call in tool_calls
                ],
            }
            if api_response.reasoning_details is not None:
                assistant_message["reasoning_details"] = (
                    api_response.reasoning_details
                )
            if api_response.reasoning:
                assistant_message["reasoning"] = api_response.reasoning
        else:
            model_responses = api_response.content
            assistant_message = {
                "role": "assistant",
                "content": api_response.content,
            }

        history_message: Any = assistant_message
        if self.model_style == ModelStyle.OPENAI_RESPONSES:
            history_message = list(api_response.response_items or [])

        response_data = {
            "model_responses": model_responses,
            "model_responses_message_for_chat_history": history_message,
            "tool_call_ids": [tool_call.id for tool_call in tool_calls],
            "input_token": api_response.prompt_tokens or 0,
            "output_token": api_response.completion_tokens or 0,
        }
        if api_response.reasoning_content:
            response_data["reasoning_content"] = api_response.reasoning_content
        if api_response.reasoning_details is not None:
            response_data["reasoning_details"] = api_response.reasoning_details
        return response_data

    @staticmethod
    def _responses_messages(messages: list[dict]) -> list[dict]:
        converted = copy.deepcopy(messages)
        for message in converted:
            if message.get("role") == "system":
                message["role"] = "developer"
        return converted

    def add_first_turn_message_FC(
        self, inference_data: dict, first_turn_message: list[dict]
    ) -> dict:
        if self.model_style == ModelStyle.OPENAI_RESPONSES:
            first_turn_message = self._responses_messages(first_turn_message)
        inference_data["message"].extend(first_turn_message)
        return inference_data

    def _add_next_turn_user_message_FC(
        self, inference_data: dict, user_message: list[dict]
    ) -> dict:
        if self.model_style == ModelStyle.OPENAI_RESPONSES:
            user_message = self._responses_messages(user_message)
        inference_data["message"].extend(user_message)
        return inference_data

    def _add_assistant_message_FC(
        self, inference_data: dict, model_response_data: dict
    ) -> dict:
        history = model_response_data["model_responses_message_for_chat_history"]
        if self.model_style == ModelStyle.OPENAI_RESPONSES:
            inference_data["message"].extend(history)
        else:
            inference_data["message"].append(history)
        return inference_data

    def _add_execution_results_FC(
        self,
        inference_data: dict,
        execution_results: list[str],
        model_response_data: dict,
    ) -> dict:
        if self.model_style != ModelStyle.OPENAI_RESPONSES:
            return super()._add_execution_results_FC(
                inference_data, execution_results, model_response_data
            )
        for execution_result, tool_call_id in zip(
            execution_results, model_response_data["tool_call_ids"]
        ):
            inference_data["message"].append(
                {
                    "type": "function_call_output",
                    "call_id": tool_call_id,
                    "output": execution_result,
                }
            )
        return inference_data

    def _parse_query_response_prompting(self, api_response: Any) -> dict:
        history_message: Any = {
            "role": "assistant",
            "content": api_response.content,
        }
        if self.model_style == ModelStyle.OPENAI_RESPONSES:
            history_message = list(api_response.response_items or [])
        response_data = {
            "model_responses": api_response.content,
            "model_responses_message_for_chat_history": history_message,
            "input_token": api_response.prompt_tokens or 0,
            "output_token": api_response.completion_tokens or 0,
        }
        if api_response.reasoning_content:
            response_data["reasoning_content"] = api_response.reasoning_content
        return response_data

    def add_first_turn_message_prompting(
        self, inference_data: dict, first_turn_message: list[dict]
    ) -> dict:
        if self.model_style == ModelStyle.OPENAI_RESPONSES:
            first_turn_message = self._responses_messages(first_turn_message)
        inference_data["message"].extend(first_turn_message)
        return inference_data

    def _add_next_turn_user_message_prompting(
        self, inference_data: dict, user_message: list[dict]
    ) -> dict:
        if self.model_style == ModelStyle.OPENAI_RESPONSES:
            user_message = self._responses_messages(user_message)
        inference_data["message"].extend(user_message)
        return inference_data

    def _add_assistant_message_prompting(
        self, inference_data: dict, model_response_data: dict
    ) -> dict:
        history = model_response_data["model_responses_message_for_chat_history"]
        if self.model_style == ModelStyle.OPENAI_RESPONSES:
            inference_data["message"].extend(history)
        else:
            inference_data["message"].append(history)
        return inference_data
