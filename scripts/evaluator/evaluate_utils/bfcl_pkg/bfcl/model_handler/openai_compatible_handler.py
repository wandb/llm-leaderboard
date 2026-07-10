import os
import time
from concurrent.futures import ThreadPoolExecutor
from typing import Optional
import json
import copy
from ..constants.type_mappings import GORILLA_TO_OPENAPI
from .base_handler import BaseHandler
from .model_style import ModelStyle
from .utils import (
    convert_to_tool,
    convert_to_function_call,
    default_decode_ast_prompting,
    default_decode_execute_prompting,
    func_doc_language_specific_pre_processing,
    system_prompt_pre_processing_chat_model,
)
from evaluator.evaluate_utils.llm_async_processor import LLMAsyncProcessor
from evaluator.evaluate_utils.provider_rate_limiter import (
    get_provider_request_rate_limiter,
)
from overrides import EnforceOverrides, final, override
from config_singleton import WandbConfigSingleton
from omegaconf import OmegaConf


def _to_plain_dict(value):
    if value is None:
        return {}
    if isinstance(value, dict):
        return copy.deepcopy(value)
    try:
        return OmegaConf.to_container(value, resolve=True) or {}
    except Exception:
        return {}


def _deep_merge_dicts(base: dict, override: dict) -> dict:
    merged = copy.deepcopy(base)
    for key, value in override.items():
        if (
            isinstance(value, dict)
            and isinstance(merged.get(key), dict)
        ):
            merged[key] = _deep_merge_dicts(merged[key], value)
        else:
            merged[key] = copy.deepcopy(value)
    return merged


def _positive_float_or_none(value):
    if value is None:
        return None
    try:
        value = float(value)
    except (TypeError, ValueError):
        return None
    return value if value > 0 else None


def _nonnegative_int_or_none(value):
    if value is None:
        return None
    try:
        value = int(value)
    except (TypeError, ValueError):
        return None
    return max(0, value)


class OpenAICompatibleHandler(BaseHandler, EnforceOverrides):
    def __init__(self, model_name, temperature) -> None:
        # temperatureは後方互換のため残しているがgenerator_configから取るので使用しない
        super().__init__(model_name, temperature)

        self.model_style = ModelStyle.OpenAI_Completions

        # Will be overridden in batch_inference method
        # Used to indicate where the tokenizer and config should be loaded from
        instance = WandbConfigSingleton.get_instance()
        cfg = instance.config
        self.model_name_huggingface = cfg.model.pretrained_model_name_or_path
        self.generator_config = _deep_merge_dicts(
            _to_plain_dict(getattr(cfg, "generator", {})),
            _to_plain_dict(cfg.bfcl.generator_config),
        )
        self.max_tokens = self.generator_config.pop("max_tokens") # 使用済みTokenに応じて調整するためgenerator_configから取り除く
        self.request_timeout_sec = _positive_float_or_none(
            OmegaConf.select(cfg, "bfcl.request_timeout_sec", default=None)
        )
        self.request_max_retries = _nonnegative_int_or_none(
            OmegaConf.select(cfg, "bfcl.request_max_retries", default=1)
        )
        provider_min_interval_sec = _positive_float_or_none(
            OmegaConf.select(cfg, "bfcl.provider_min_request_interval_sec", default=0)
        ) or 0.0
        provider_jitter_sec = _positive_float_or_none(
            OmegaConf.select(cfg, "bfcl.provider_request_jitter_sec", default=0)
        ) or 0.0
        provider_key = str(
            OmegaConf.select(
                cfg,
                "bfcl.provider_rate_limit_key",
                default=f"bfcl:{self.model_name_huggingface}",
            )
        )
        self.provider_request_limiter = get_provider_request_rate_limiter(
            provider_key,
            min_interval_sec=provider_min_interval_sec,
            jitter_sec=provider_jitter_sec,
        )
        backoff_max_time = _positive_float_or_none(
            OmegaConf.select(cfg, "bfcl.backoff_max_time_sec", default=90)
        )
        backoff_max_tries = _nonnegative_int_or_none(
            OmegaConf.select(cfg, "bfcl.backoff_max_tries", default=4)
        )

        # Read from env vars with fallbacks
        llm = instance.llm
        self.llm_ap = LLMAsyncProcessor(
            llm,
            backoff_max_time=backoff_max_time,
            backoff_max_tries=backoff_max_tries,
            provider_rate_limit_enabled=False,
        )

    def _request_control_kwargs(self) -> dict:
        kwargs = {}
        if self.request_timeout_sec is not None:
            kwargs["timeout"] = self.request_timeout_sec
        if self.request_max_retries is not None:
            kwargs["request_max_retries"] = self.request_max_retries
        return kwargs

    @override
    def decode_ast(self, result, language="Python"):
        if "FC" in self.model_name or self.is_fc_model:
            decoded_output = []
            for invoked_function in result:
                name = list(invoked_function.keys())[0]
                params = invoked_function[name]
                decoded_output.append({name: params})
            return decoded_output
        else:
            return default_decode_ast_prompting(result, language)

    @override
    def decode_execute(self, result):
        if "FC" in self.model_name or self.is_fc_model:
            return convert_to_function_call(result)
        else:
            return default_decode_execute_prompting(result)

    #### FC methods ####

    @override
    def _query_FC(self, inference_data: dict):
        message: list[dict] = inference_data["message"]
        tools = inference_data["tools"]
        inference_data["inference_input_log"] = {"message": repr(message), "tools": tools}

        kwargs = {
            "model": self.model_name.replace("-FC", ""),
            "max_tokens": self.max_tokens,
            **self.generator_config,
            **self._request_control_kwargs(),
        }

        if len(tools) > 0:
            kwargs["tools"] = tools

        start_time = time.time()
        self.provider_request_limiter.wait()
        api_response = self.llm_ap.process_single(message, **kwargs)
        end_time = time.time()

        return api_response, end_time - start_time

    @override
    async def _query_FC_async(self, inference_data: dict):
        message: list[dict] = inference_data["message"]
        tools = inference_data["tools"]
        inference_data["inference_input_log"] = {"message": repr(message), "tools": tools}

        kwargs = {
            "model": self.model_name.replace("-FC", ""),
            "max_tokens": self.max_tokens,
            **self.generator_config,
            **self._request_control_kwargs(),
        }

        if len(tools) > 0:
            kwargs["tools"] = tools

        start_time = time.time()
        await self.provider_request_limiter.wait_async()
        api_response = await self.llm_ap.process_single_async(message, **kwargs)
        end_time = time.time()

        return api_response, end_time - start_time

    @override
    def _pre_query_processing_FC(self, inference_data: dict, test_entry: dict) -> dict:
        inference_data["message"] = []
        return inference_data

    @override
    def _compile_tools(self, inference_data: dict, test_entry: dict) -> dict:
        functions: list = test_entry["function"]
        test_category: str = test_entry["id"].rsplit("_", 1)[0]

        functions = func_doc_language_specific_pre_processing(functions, test_category)
        tools = convert_to_tool(functions, GORILLA_TO_OPENAPI, ModelStyle.OpenAI_Completions)

        inference_data["tools"] = tools

        return inference_data

    @override
    def _parse_query_response_FC(self, api_response: any) -> dict:
        try:
            model_responses = [
                {func_call.name: func_call.arguments}
                for func_call in api_response.tool_calls
            ]
            tool_call_ids = [
                func_call.id for func_call in api_response.tool_calls
            ]
        except Exception:
            model_responses = api_response.content
            tool_call_ids = []

        model_responses_message_for_chat_history = {"role": "assistant", "content": api_response.content}

        response_data = {
            "model_responses": model_responses,
            "model_responses_message_for_chat_history": model_responses_message_for_chat_history,
            "tool_call_ids": tool_call_ids,
            "input_token": api_response.prompt_tokens,
            "output_token": api_response.completion_tokens,
        }
        self._add_reasoning_content_if_available_FC(api_response, response_data)
        return response_data

    @override
    def add_first_turn_message_FC(
        self, inference_data: dict, first_turn_message: list[dict]
    ) -> dict:
        inference_data["message"].extend(first_turn_message)
        return inference_data

    @override
    def _add_next_turn_user_message_FC(
        self, inference_data: dict, user_message: list[dict]
    ) -> dict:
        inference_data["message"].extend(user_message)
        return inference_data

    @override
    def _add_assistant_message_FC(
        self, inference_data: dict, model_response_data: dict
    ) -> dict:
        inference_data["message"].append(
            model_response_data["model_responses_message_for_chat_history"]
        )
        return inference_data

    @override
    def _add_execution_results_FC(
        self,
        inference_data: dict,
        execution_results: list[str],
        model_response_data: dict,
    ) -> dict:
        # Add the execution results to the current round result, one at a time
        for execution_result, tool_call_id in zip(
            execution_results, model_response_data["tool_call_ids"]
        ):
            tool_message = {
                "role": "tool",
                "content": execution_result,
                "tool_call_id": tool_call_id,
            }
            inference_data["message"].append(tool_message)

        return inference_data

    def _add_reasoning_content_if_available_FC(
        self, api_response: any, response_data: dict
    ) -> None:
        """
        OpenAI models don't show reasoning content in the api response,
        but many other models that use the OpenAI interface do, such as DeepSeek and Grok.
        This method is included here to avoid code duplication.

        These models often don't take reasoning content in the chat history for next turn.
        Thus, this method saves reasoning content to response_data (for local result file) if present in the response,
        but does not include it in the chat history.
        """
        # Preserve tool_call information but strip the unsupported `reasoning_content` field before inserting into chat history.
        if api_response.tool_calls:
            assistant_message = {
                "role": "assistant",
                "tool_calls": [
                    {
                        "id": tool_call.id,
                        "type": tool_call.type,
                        "function": {
                            "name": tool_call.name,
                            "arguments": json.dumps(tool_call.arguments),
                        },
                    }
                    for tool_call in api_response.tool_calls
                ],
            }
            if api_response.content:
                assistant_message["content"] = api_response.content

            # OpenRouter reasoning preservation (important for reasoning models + tool use)
            # See: https://openrouter.ai/docs/guides/best-practices/reasoning-tokens#preserving-reasoning
            reasoning_details = getattr(api_response, "reasoning_details", None)
            if reasoning_details is not None:
                assistant_message["reasoning_details"] = reasoning_details
            # Prefer explicit `reasoning` field if present, otherwise fall back to reasoning_content
            reasoning_plain = getattr(api_response, "reasoning", None)
            if reasoning_plain:
                assistant_message["reasoning"] = reasoning_plain
            elif getattr(api_response, "reasoning_content", None):
                assistant_message["reasoning"] = api_response.reasoning_content

            response_data["model_responses_message_for_chat_history"] = assistant_message

        # If no tool_calls, we still need to strip reasoning_content.
        elif api_response.reasoning_content:
            response_data["model_responses_message_for_chat_history"] = {
                "role": "assistant",
                "content": api_response.content,
            }

        # Capture the reasoning trace so it can be logged to the local result file.
        if api_response.reasoning_content:
            response_data["reasoning_content"] = api_response.reasoning_content
        # Also capture structured reasoning blocks if present (for debugging / result files)
        if getattr(api_response, "reasoning_details", None) is not None:
            response_data["reasoning_details"] = api_response.reasoning_details

    #### Prompting methods ####

    # NOTE: _format_prompt is no longer used since we switched to Chat Completion API
    # The vLLM server now applies the chat template automatically
    # def _format_prompt(self, messages, function):
    #     """
    #     Manually apply the chat template to construct the formatted prompt.
    #     This way, we can have full control over the final formatted prompt and is generally recommended for advanced use cases.
    #     """
    #     raise NotImplementedError(
    #         "OSS Models should implement their own prompt formatting."
    #     )

    @override
    def _query_prompting(self, inference_data: dict):
        # We use the OpenAI Chat Completions API
        function: list[dict] = inference_data["function"]
        message: list[dict] = inference_data["message"]

        # Chat Completion API uses messages directly, no need for _format_prompt
        # The vLLM server will apply the chat template automatically
        inference_data["inference_input_log"] = {"messages": message}

        kwargs = {
            "model": self.model_name,
            "max_tokens": self.max_tokens,
            **self.generator_config,
            **self._request_control_kwargs(),
        }

        start_time = time.time()
        self.provider_request_limiter.wait()
        api_response = self.llm_ap.process_single(message, **kwargs)
        end_time = time.time()

        return api_response, end_time - start_time

    @override
    async def _query_prompting_async(self, inference_data: dict):
        # We use the OpenAI Chat Completions API
        message: list[dict] = inference_data["message"]

        kwargs = {
            "model": self.model_name,
            "max_tokens": self.max_tokens,
            **self.generator_config,
            **self._request_control_kwargs(),
        }

        start_time = time.time()
        await self.provider_request_limiter.wait_async()
        api_response = await self.llm_ap.process_single_async(message, **kwargs)
        end_time = time.time()

        return api_response, end_time - start_time

    @override
    def _pre_query_processing_prompting(self, test_entry: dict) -> dict:
        functions: list = test_entry["function"]
        test_category: str = test_entry["id"].rsplit("_", 1)[0]

        functions = func_doc_language_specific_pre_processing(functions, test_category)

        test_entry["question"][0] = system_prompt_pre_processing_chat_model(
            test_entry["question"][0], functions, test_category
        )

        return {"message": [], "function": functions}

    @override
    def _parse_query_response_prompting(self, api_response: any) -> dict:
        return {
            "model_responses": api_response.content,
            "input_token": api_response.prompt_tokens,
            "output_token": api_response.completion_tokens,
        }

    @override
    def add_first_turn_message_prompting(
        self, inference_data: dict, first_turn_message: list[dict]
    ) -> dict:
        inference_data["message"].extend(first_turn_message)
        return inference_data

    @override
    def _add_next_turn_user_message_prompting(
        self, inference_data: dict, user_message: list[dict]
    ) -> dict:
        inference_data["message"].extend(user_message)
        return inference_data

    @override
    def _add_assistant_message_prompting(
        self, inference_data: dict, model_response_data: dict
    ) -> dict:
        inference_data["message"].append(
            {"role": "assistant", "content": model_response_data["model_responses"]}
        )
        return inference_data

    @override
    def _add_execution_results_prompting(
        self, inference_data: dict, execution_results: list[str], model_response_data: dict
    ) -> dict:
        for execution_result, decoded_model_response in zip(
            execution_results, model_response_data["model_responses_decoded"]
        ):
            inference_data["message"].append(
                {
                    "role": "tool",
                    "name": decoded_model_response,
                    "content": execution_result,
                }
            )

        return inference_data
