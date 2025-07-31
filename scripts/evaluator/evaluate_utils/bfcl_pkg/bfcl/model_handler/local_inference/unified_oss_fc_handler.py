import json
import time
import traceback
from typing import Any
import warnings

from .base_oss_handler import OSSHandler
from ..utils import (
    convert_to_tool,
    convert_to_function_call,
    func_doc_language_specific_pre_processing,
)
from ...constants.type_mappings import GORILLA_TO_OPENAPI
from ..model_style import ModelStyle
from overrides import override
from config_singleton import WandbConfigSingleton

class UnifiedOSSFCHandler(OSSHandler):
    """
    統合OSSハンドラー（FC版）

    Tokenizerがtools引数に対応しており、vLLMのtool_call_parserに対応しているモデル用のOpenAI互換共通ハンドラー
    https://docs.vllm.ai/en/stable/features/tool_calling.html
    """

    def __init__(self, model_name, temperature) -> None:
        super().__init__(model_name, temperature)
        self.is_fc_model = True
        self.model_name_huggingface = model_name.replace("-FC", "")

        instance = WandbConfigSingleton.get_instance()
        cfg = instance.config

        if cfg.api in ["vllm", "vllm-docker"]:
            assert cfg.vllm.tool_call_parser is not None, "cfg.vllm.tool_call_parser is not set"
            assert cfg.vllm.enable_auto_tool_choice, "cfg.vllm.enable_auto_tool_choice is false"
        elif cfg.api == "openai-compatible":
            warnings.warn(
                "UnifiedOSSFCHandler only supports vLLM server but cfg.api is set to openai-compatible. "
                "Make sure to cfg.base_url is set to vLLM server launched with tool_call_parser/enable_auto_tool_choice. "
                "See: https://docs.vllm.ai/en/stable/features/tool_calling.html"
            )

        print(f"[UnifiedOSSFCHandler] 初期化完了 - モデル: {model_name}")

    @override
    def decode_ast(self, result, language="Python"):
        decoded_output = []
        for invoked_function in result:
            name = list(invoked_function.keys())[0]
            params = json.loads(invoked_function[name])
            decoded_output.append({name: params})
        return decoded_output

    @override
    def decode_execute(self, result):
        return convert_to_function_call(result)

    @override
    def _multi_threaded_inference(
        self, test_case, include_input_log: bool, exclude_state_log: bool
    ):
        """
        This is a wrapper function to make sure that, if an error occurs during inference, the process does not stop.
        """
        assert type(test_case["function"]) is list

        try:
            if "multi_turn" in test_case["id"]:
                model_responses, metadata = self.inference_multi_turn_FC(
                    test_case, include_input_log, exclude_state_log
                )
            else:
                model_responses, metadata = self.inference_single_turn_FC(
                    test_case, include_input_log
                )
        except Exception as e:
            print("-" * 100)
            print(
                "❗️❗️ Error occurred during inference. Maximum reties reached for rate limit or other error. Continuing to next test case."
            )
            print(f"❗️❗️ Test case ID: {test_case['id']}, Error: {str(e)}")
            traceback.print_exc(limit=10)
            print("-" * 100)

            model_responses = f"Error during inference: {str(e)}"
            metadata = {
                "traceback": traceback.format_exc(),
            }

        result_to_write = {
            "id": test_case["id"],
            "result": model_responses,
        }
        result_to_write.update(metadata)

        return result_to_write

    #### FC methods: most of the methods are copied from OpenAICompletionsHandler ####

    @override
    def _query_FC(self, inference_data: dict):
        message: list[dict] = inference_data["message"]
        tools = inference_data["tools"]
        inference_data["inference_input_log"] = {"message": repr(message), "tools": tools}

        # For chat completions, we need to estimate token count from messages
        # Use apply_chat_template with `tools` arg to estimate input token count accurately
        input_token_count = len(self.tokenizer.apply_chat_template(message, tokenize=True, tools=tools))

        # Determine the number of tokens to request. Cap it at cfg.bfcl.max_tokens if the model has a larger limit.
        if self.max_context_length < input_token_count + 2:
            # If the prompt is already at the max length, just request 1000 token, we will get an error anyway
            leftover_tokens_count = 1000
        else:
            leftover_tokens_count = min(
                self.max_tokens, # cfg.bfcl.max_tokens
                self.max_context_length - input_token_count - 2,
            )

        kwargs = {
            "messages": message,
            "model": self.model_name.replace("-FC", ""),
            "temperature": self.temperature,
            "max_tokens": leftover_tokens_count,
            # "store": False, removed: because vLLM server doesn't support it
        }

        if len(tools) > 0:
            kwargs["tools"] = tools
            kwargs["tool_choice"] = "auto" # vLLM requires tool_choice=auto to parse tool output

        start_time = time.time()
        api_response = self.client.chat.completions.create(**kwargs)
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
        # self.model_style is ModelStyle.OSSModel, but explicitly set to OpenAI_Completions to use OpenAI format
        tools = convert_to_tool(functions, GORILLA_TO_OPENAPI, ModelStyle.OpenAI_Completions)

        inference_data["tools"] = tools

        return inference_data

    @override
    def _parse_query_response_FC(self, api_response: any) -> dict:
        try:
            model_responses = [
                {func_call.function.name: func_call.function.arguments}
                for func_call in api_response.choices[0].message.tool_calls
            ]
            tool_call_ids = [
                func_call.id for func_call in api_response.choices[0].message.tool_calls
            ]
        except:
            model_responses = api_response.choices[0].message.content
            tool_call_ids = []

        model_responses_message_for_chat_history = api_response.choices[0].message

        response_data =  {
            "model_responses": model_responses,
            "model_responses_message_for_chat_history": model_responses_message_for_chat_history,
            "tool_call_ids": tool_call_ids,
            "input_token": api_response.usage.prompt_tokens,
            "output_token": api_response.usage.completion_tokens,
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
        self, api_response: Any, response_data: dict
    ) -> None:
        """
        OpenAI models don't show reasoning content in the api response,
        but many other models that use the OpenAI interface do, such as DeepSeek and Grok.
        This method is included here to avoid code duplication.

        These models often don't take reasoning content in the chat history for next turn.
        Thus, this method saves reasoning content to response_data (for local result file) if present in the response,
        but does not include it in the chat history.
        """
        # Original assistant message object (contains `reasoning_content` on DeepSeek).
        message = api_response.choices[0].message

        # Preserve tool_call information but strip the unsupported `reasoning_content` field before inserting into chat history.
        if getattr(message, "tool_calls", None):
            assistant_message = {
                "role": "assistant",
                "content": message.content,
                "tool_calls": [
                    {
                        "id": tool_call.id,
                        "type": tool_call.type,
                        "function": {
                            "name": tool_call.function.name,
                            "arguments": tool_call.function.arguments,
                        },
                    }
                    for tool_call in message.tool_calls
                ],
            }
            response_data["model_responses_message_for_chat_history"] = assistant_message

        # If no tool_calls, we still need to strip reasoning_content.
        elif hasattr(message, "reasoning_content"):
            response_data["model_responses_message_for_chat_history"] = {
                "role": "assistant",
                "content": message.content,
            }

        # Capture the reasoning trace so it can be logged to the local result file.
        if hasattr(message, "reasoning_content"):
            response_data["reasoning_content"] = message.reasoning_content
