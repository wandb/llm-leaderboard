import os
import time
from concurrent.futures import ThreadPoolExecutor
from typing import Optional
import json
from ...constants.type_mappings import GORILLA_TO_OPENAPI
from ..base_handler import BaseHandler
from ..model_style import ModelStyle
from ..utils import (
    convert_to_tool,
    convert_to_function_call,
    default_decode_ast_prompting,
    default_decode_execute_prompting,
    func_doc_language_specific_pre_processing,
    system_prompt_pre_processing_chat_model,
)
from evaluator.evaluate_utils.llm_async_processor import LLMAsyncProcessor
from overrides import EnforceOverrides, final, override
from config_singleton import WandbConfigSingleton
from omegaconf import OmegaConf


class OSSHandler(BaseHandler, EnforceOverrides):
    def __init__(self, model_name, temperature) -> None:
        # temperatureは後方互換のため残しているがgenerator_configから取るので使用しない
        super().__init__(model_name, temperature)

        
        self.model_style = ModelStyle.OSSMODEL

        # Will be overridden in batch_inference method
        # Used to indicate where the tokenizer and config should be loaded from
        instance = WandbConfigSingleton.get_instance()
        cfg = instance.config
        self.model_name_huggingface = cfg.model.pretrained_model_name_or_path
        self.generator_config = OmegaConf.to_container(cfg.bfcl.generator_config)
        self.max_tokens = self.generator_config.pop("max_tokens") # 使用済みTokenに応じて調整するためgenerator_configから取り除く

        # Read from env vars with fallbacks
        llm = instance.llm
        self.llm_ap = LLMAsyncProcessor(llm)

    def setup_tokenizer(self, local_model_path: Optional[str] = None):
        from transformers import AutoConfig, AutoTokenizer

        # Determine the model source
        if local_model_path is not None:
            # Validate the local_model_path
            if not os.path.isdir(local_model_path):
                raise ValueError(
                    f"local_model_path '{local_model_path}' does not exist or is not a directory."
                )

            required_files = ["config.json", "tokenizer_config.json"]
            for file_name in required_files:
                if not os.path.exists(os.path.join(local_model_path, file_name)):
                    raise ValueError(
                        f"Required file '{file_name}' not found in local_model_path '{local_model_path}'."
                    )

            self.model_path_or_id = local_model_path
            load_kwargs = {
                "pretrained_model_name_or_path": self.model_path_or_id,
                "local_files_only": True,
                "trust_remote_code": True,
            }
        else:
            self.model_path_or_id = self.model_name_huggingface
            load_kwargs = {
                "pretrained_model_name_or_path": self.model_path_or_id,
                "trust_remote_code": True,
            }

        self.tokenizer = AutoTokenizer.from_pretrained(**load_kwargs)
        config = AutoConfig.from_pretrained(**load_kwargs)

        if hasattr(config, "max_position_embeddings"):
            self.max_context_length = config.max_position_embeddings
        elif self.tokenizer.model_max_length is not None:
            self.max_context_length = self.tokenizer.model_max_length
        else:
            if not hasattr(self, "max_context_length"):
                raise ValueError(
                    "Model does not have a max_position_embeddings attribute or tokenizer.model_max_length attribute. Please set the max_context_length attribute in the corresponding model handler."
                )
        print(f"Max context length: {self.max_context_length}")

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
            "model": self.model_name.replace("-FC", ""),
            "max_tokens": leftover_tokens_count,
            **self.generator_config,
            # "store": False, removed: because vLLM server doesn't support it
        }

        if len(tools) > 0:
            kwargs["tools"] = tools
            kwargs["tool_choice"] = "auto" # vLLM requires tool_choice=auto to parse tool output
        api_response = self.llm_ap.process_single(message, **kwargs)
        return api_response

    @override
    async def _query_FC_async(self, inference_data: dict):
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
            "model": self.model_name.replace("-FC", ""),
            "max_tokens": leftover_tokens_count,
            **self.generator_config,
            # "store": False, removed: because vLLM server doesn't support it
        }

        if len(tools) > 0:
            kwargs["tools"] = tools
            kwargs["tool_choice"] = "auto" # vLLM requires tool_choice=auto to parse tool output

        start_time = time.time()
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
        except:
            model_responses = api_response.content
            tool_call_ids = []

        model_responses_message_for_chat_history = api_response.content

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
                "content": api_response.content,
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

        # For chat completions, we need to estimate token count from messages
        # This is a rough estimation
        total_content = ""
        for msg in message:
            total_content += f"{msg['role']}: {msg['content']}\n"
        
        input_token_count = len(self.tokenizer.tokenize(total_content))

        # Determine the number of tokens to request. Cap it at 4096 if the model has a larger limit.
        if self.max_context_length < input_token_count + 2:
            # If the prompt is already at the max length, just request 1000 token, we will get an error anyway
            leftover_tokens_count = 1000
        else:
            leftover_tokens_count = min(
                self.max_tokens,
                self.max_context_length - input_token_count - 2,
            )

        kwargs = {
            "model": self.model_path_or_id,
            "max_tokens": leftover_tokens_count,
            "timeout": 72000,  # Avoid timeout errors
            **self.generator_config,
        }

        extra_body = {}
        if hasattr(self, "stop_token_ids"):
            extra_body["stop_token_ids"] = self.stop_token_ids
        if hasattr(self, "skip_special_tokens"):
            extra_body["skip_special_tokens"] = self.skip_special_tokens

        if len(extra_body) > 0:
            kwargs["extra_body"] = extra_body

        start_time = time.time()
        api_response = self.llm_ap.process_single(message, **kwargs)
        end_time = time.time()

        return api_response, end_time - start_time

    @override
    async def _query_prompting_async(self, inference_data: dict):
        # We use the OpenAI Chat Completions API
        function: list[dict] = inference_data["function"]
        message: list[dict] = inference_data["message"]

        # Chat Completion API uses messages directly, no need for _format_prompt
        # The vLLM server will apply the chat template automatically
        inference_data["inference_input_log"] = {"messages": message}

        # For chat completions, we need to estimate token count from messages
        # This is a rough estimation
        total_content = ""
        for msg in message:
            total_content += f"{msg['role']}: {msg['content']}\n"
        
        input_token_count = len(self.tokenizer.tokenize(total_content))

        # Determine the number of tokens to request. Cap it at 4096 if the model has a larger limit.
        if self.max_context_length < input_token_count + 2:
            # If the prompt is already at the max length, just request 1000 token, we will get an error anyway
            leftover_tokens_count = 1000
        else:
            leftover_tokens_count = min(
                self.max_tokens,
                self.max_context_length - input_token_count - 2,
            )

        kwargs = {
            "model": self.model_path_or_id,
            "max_tokens": leftover_tokens_count,
            "timeout": 72000,  # Avoid timeout errors
            **self.generator_config,
        }

        extra_body = {}
        if hasattr(self, "stop_token_ids"):
            extra_body["stop_token_ids"] = self.stop_token_ids
        if hasattr(self, "skip_special_tokens"):
            extra_body["skip_special_tokens"] = self.skip_special_tokens

        if len(extra_body) > 0:
            kwargs["extra_body"] = extra_body

        start_time = time.time()
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
