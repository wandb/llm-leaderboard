import json
import os
import time
import asyncio
import inspect

from ...constants.type_mappings import GORILLA_TO_OPENAPI
from ..base_handler import BaseHandler
from ..model_style import ModelStyle
from ..utils import (
    convert_to_function_call,
    convert_to_tool,
    default_decode_ast_prompting,
    default_decode_execute_prompting,
    format_execution_results_prompting,
    func_doc_language_specific_pre_processing,
    retry_with_backoff,
    system_prompt_pre_processing_chat_model,
)
from openai import OpenAI, RateLimitError
from openai.types.responses import Response
from config_singleton import WandbConfigSingleton
from omegaconf import OmegaConf

class OpenAIResponsesHandler(BaseHandler):
    def __init__(self, model_name, temperature, llm=None, use_encrypted_reasoning=True) -> None:
        super().__init__(model_name, temperature)
        self.model_style = ModelStyle.OpenAI_Responses
        # Load config and llm if available
        try:
            instance = WandbConfigSingleton.get_instance()
            cfg = instance.config
            self.model_name_huggingface = cfg.model.pretrained_model_name_or_path
            self.model = cfg.model
            self.generator_config = OmegaConf.to_container(cfg.bfcl.generator_config)
            # Remove max_tokens to manage it separately if needed
            self.max_tokens = self.generator_config.pop("max_tokens")
            if llm is None:
                llm = getattr(instance, "llm", None)
        except Exception:
            # Singleton not initialized; proceed without config/llm
            self.model_name_huggingface = None
            self.generator_config = {}
            self.max_tokens = None
        # Prefer an externally provided Responses client (e.g., OpenAIResponsesClient)
        # If provided and it exposes an AsyncOpenAI via `async_client`, use that; otherwise, fall back to default OpenAI client
        if llm is not None and hasattr(llm, "async_client") and hasattr(llm.async_client, "responses"):
            self.client = llm.async_client  # AsyncOpenAI
            self._delegate_model_to_client = True
            self._client_is_async = True
        else:
            self.client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
            self._delegate_model_to_client = False
            self._client_is_async = False
        self.use_encrypted_reasoning = use_encrypted_reasoning

    @staticmethod
    def _substitute_prompt_role(prompts: list[dict]) -> list[dict]:
        # OpenAI allows `system` role in the prompt, but it is meant for "messages added by OpenAI"
        # For our use case, it is recommended to use `developer` role instead.
        # See https://model-spec.openai.com/2025-04-11.html#definitions
        for prompt in prompts:
            if prompt["role"] == "system":
                prompt["role"] = "developer"

        return prompts

    def decode_ast(self, result, language="Python"):
        if "FC" in self.model_name or self.is_fc_model:
            decoded_output = []
            for invoked_function in result:
                name = list(invoked_function.keys())[0]
                params = json.loads(invoked_function[name])
                decoded_output.append({name: params})
            return decoded_output
        else:
            return default_decode_ast_prompting(result, language)

    def decode_execute(self, result):
        if "FC" in self.model_name or self.is_fc_model:
            return convert_to_function_call(result)
        else:
            return default_decode_execute_prompting(result)

    @retry_with_backoff(error_type=RateLimitError)
    def generate_with_backoff(self, **kwargs):
        start_time = time.time()
        result = self.client.responses.create(**kwargs)
        # Support both sync and async OpenAI clients transparently
        if inspect.isawaitable(result):
            api_response = asyncio.run(result)
        else:
            api_response = result
        end_time = time.time()

        return api_response, end_time - start_time

    #### FC methods ####

    def _query_FC(self, inference_data: dict):
        message: list[dict] = inference_data["message"]
        tools = inference_data["tools"]

        inference_data["inference_input_log"] = {
            "message": repr(message),
            "tools": tools,
        }

        kwargs = {
            "input": message,
            "store": False,
        }

        # Use include only if specified in YAML generator config
        include_from_cfg = self.generator_config.get("include") if hasattr(self, "generator_config") else None
        if include_from_cfg:
            kwargs["include"] = include_from_cfg

        # Only pass model if we are not delegating model selection to the provided client
        # Responses API requires model explicitly
        kwargs["model"] = self.model.pretrained_model_name_or_path

        # OpenAI reasoning models don't support temperature parameter
        if not any(x in self.model_name_huggingface for x in ["o1", "o3", "o4"]):
            kwargs["temperature"] = self.temperature

        if len(tools) > 0:
            kwargs["tools"] = tools

        return self.generate_with_backoff(**kwargs)

    def _pre_query_processing_FC(self, inference_data: dict, test_entry: dict) -> dict:
        for round_idx in range(len(test_entry["question"])):
            test_entry["question"][round_idx] = OpenAIResponsesHandler._substitute_prompt_role(
                test_entry["question"][round_idx]
            )

        inference_data["message"] = []

        return inference_data

    def _compile_tools(self, inference_data: dict, test_entry: dict) -> dict:
        functions: list = test_entry["function"]
        test_category: str = test_entry["id"].rsplit("_", 1)[0]

        functions = func_doc_language_specific_pre_processing(functions, test_category)

        tools = convert_to_tool(functions, GORILLA_TO_OPENAPI, self.model_style)

        inference_data["tools"] = tools

        return inference_data

    def _parse_query_response_FC(self, api_response: Response) -> dict:
        model_responses = []
        tool_call_ids = []

        # Extract tool calls and build a store=false safe history payload (function_call items only)
        safe_history_items = []
        for item in api_response.output:
            if item.type == "function_call":
                model_responses.append({item.name: item.arguments})
                tool_call_ids.append(item.call_id)
                safe_history_items.append(
                    {
                        "type": "function_call",
                        "name": item.name,
                        "arguments": item.arguments,
                        "call_id": item.call_id,
                    }
                )

        if not model_responses:  # If there are no function calls
            model_responses = api_response.output_text

        # Summarize reasoning if present (for logging only)
        reasoning_content = ""
        for item in api_response.output:
            if item.type == "reasoning":
                for summary in item.summary:
                    reasoning_content += summary.text + "\n"

        return {
            "model_responses": model_responses,
            "model_responses_message_for_chat_history": safe_history_items,
            "tool_call_ids": tool_call_ids,
            "reasoning_content": reasoning_content,
            "input_token": api_response.usage.input_tokens,
            "output_token": api_response.usage.output_tokens,
        }

    def add_first_turn_message_FC(
        self, inference_data: dict, first_turn_message: list[dict]
    ) -> dict:
        inference_data["message"].extend(first_turn_message)
        return inference_data

    def _add_next_turn_user_message_FC(
        self, inference_data: dict, user_message: list[dict]
    ) -> dict:
        inference_data["message"].extend(user_message)
        return inference_data

    def _add_assistant_message_FC(
        self, inference_data: dict, model_response_data: dict
    ) -> dict:
        # With store=false, avoid pushing opaque response objects back.
        # Only push function_call items we explicitly constructed.
        inference_data["message"].extend(
            model_response_data["model_responses_message_for_chat_history"]
        )
        return inference_data

    def _add_execution_results_FC(
        self,
        inference_data: dict,
        execution_results: list[str],
        model_response_data: dict,
    ) -> dict:
        # With store=false, return tool outputs in the Responses input shape
        for execution_result, tool_call_id in zip(
            execution_results, model_response_data["tool_call_ids"]
        ):
            tool_message = {
                "type": "function_call_output",
                "call_id": tool_call_id,
                "output": execution_result,
            }
            inference_data["message"].append(tool_message)

        return inference_data

    #### Prompting methods ####

    def _query_prompting(self, inference_data: dict):
        inference_data["inference_input_log"] = {"message": repr(inference_data["message"])}

        kwargs = {
            "input": inference_data["message"],
            "store": False,
        }

        # Use include only if specified in YAML generator config
        include_from_cfg = self.generator_config.get("include") if hasattr(self, "generator_config") else None
        if include_from_cfg:
            kwargs["include"] = include_from_cfg

        # Responses API requires model explicitly
        kwargs["model"] = self.model.pretrained_model_name_or_path

        # OpenAI reasoning models don't support temperature parameter
        if "o3" not in self.model_name and "o4-mini" not in self.model_name:
            kwargs["temperature"] = self.temperature

        return self.generate_with_backoff(**kwargs)

    def _pre_query_processing_prompting(self, test_entry: dict) -> dict:
        functions: list = test_entry["function"]
        test_category: str = test_entry["id"].rsplit("_", 1)[0]

        functions = func_doc_language_specific_pre_processing(functions, test_category)

        test_entry["question"][0] = system_prompt_pre_processing_chat_model(
            test_entry["question"][0], functions, test_category
        )

        for round_idx in range(len(test_entry["question"])):
            test_entry["question"][round_idx] = OpenAIResponsesHandler._substitute_prompt_role(
                test_entry["question"][round_idx]
            )

        return {"message": []}

    def _parse_query_response_prompting(self, api_response: Response) -> dict:
        # OpenAI reasoning models don't show full reasoning content in the api response,
        # but only a summary of the reasoning content.
        reasoning_content = ""
        for item in api_response.output:
            if item.type == "reasoning":
                for summary in item.summary:
                    reasoning_content += summary.text + "\n"

        return {
            "model_responses": api_response.output_text,
            "model_responses_message_for_chat_history": api_response.output,
            "reasoning_content": reasoning_content,
            "input_token": api_response.usage.input_tokens,
            "output_token": api_response.usage.output_tokens,
        }

    def add_first_turn_message_prompting(
        self, inference_data: dict, first_turn_message: list[dict]
    ) -> dict:
        inference_data["message"].extend(first_turn_message)
        return inference_data

    def _add_next_turn_user_message_prompting(
        self, inference_data: dict, user_message: list[dict]
    ) -> dict:
        inference_data["message"].extend(user_message)
        return inference_data

    def _add_assistant_message_prompting(
        self, inference_data: dict, model_response_data: dict
    ) -> dict:
        inference_data["message"].extend(
            model_response_data["model_responses_message_for_chat_history"]
        )
        return inference_data

    def _add_execution_results_prompting(
        self,
        inference_data: dict,
        execution_results: list[str],
        model_response_data: dict,
    ) -> dict:
        formatted_results_message = format_execution_results_prompting(
            inference_data, execution_results, model_response_data
        )
        inference_data["message"].append(
            {"role": "user", "content": formatted_results_message}
        )

        return inference_data
