import os
import time
from typing import Optional
from ..model_style import ModelStyle
from overrides import EnforceOverrides, final, override
from ..openai_compatible_handler import OpenAICompatibleHandler
from ..model_style import ModelStyle


class OSSHandler(OpenAICompatibleHandler, EnforceOverrides):
    def __init__(self, model_name, temperature) -> None:
        super().__init__(model_name, temperature)

        self.model_name_huggingface = self.model_name.replace("-FC", "")
        self.model_style = ModelStyle.OSSMODEL

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

    def _estimate_leftover_tokens_count(self, inference_data: dict, fc=False) -> int:
        message: list[dict] = inference_data["message"]

        # For chat completions, we need to estimate token count from messages
        # Use apply_chat_template with `tools` arg to estimate input token count accurately
        if fc:
            tools = inference_data["tools"]
            input_token_count = len(self.tokenizer.apply_chat_template(message, tokenize=True, tools=tools))
        else:
            # non-FC models: tools are already included in the message
            input_token_count = len(self.tokenizer.apply_chat_template(message, tokenize=True))

        # Determine the number of tokens to request. Cap it at cfg.bfcl.max_tokens if the model has a larger limit.
        if self.max_context_length < input_token_count + 2:
            # If the prompt is already at the max length, just request 1000 token, we will get an error anyway
            leftover_tokens_count = 1000
        else:
            leftover_tokens_count = min(
                self.max_tokens, # cfg.bfcl.max_tokens
                self.max_context_length - input_token_count - 2,
            )

        return leftover_tokens_count

    #### FC methods ####

    @override
    def _query_FC(self, inference_data: dict):
        message: list[dict] = inference_data["message"]
        tools = inference_data["tools"]
        inference_data["inference_input_log"] = {"message": repr(message), "tools": tools}

        kwargs = {
            "model": self.model_name.replace("-FC", ""),
            "max_tokens": self._estimate_leftover_tokens_count(inference_data, fc=True),
            **self.generator_config,
            # "store": False, removed: because vLLM server doesn't support it
        }

        if len(tools) > 0:
            kwargs["tools"] = tools
            kwargs["tool_choice"] = "auto" # vLLM requires tool_choice=auto to parse tool output

        start_time = time.time()
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
            "max_tokens": self._estimate_leftover_tokens_count(inference_data, fc=True),
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
            "model": self.model_path_or_id,
            "max_tokens": self._estimate_leftover_tokens_count(inference_data, fc=False),
            "timeout": 72000,  # Avoid timeout errors
            **self.generator_config,
        }

        extra_body = {}
        if hasattr(self, "stop_token_ids"):
            extra_body["stop_token_ids"] = self.stop_token_ids
        if hasattr(self, "skip_special_tokens"):
            extra_body["skip_special_tokens"] = self.skip_special_tokens

        if len(extra_body) > 0:
            if "extra_body" in kwargs:
                kwargs["extra_body"] = {**kwargs["extra_body"], **extra_body}
            else:
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

        kwargs = {
            "model": self.model_path_or_id,
            "max_tokens": self._estimate_leftover_tokens_count(inference_data, fc=False),
            "timeout": 72000,  # Avoid timeout errors
            **self.generator_config,
        }

        extra_body = {}
        if hasattr(self, "stop_token_ids"):
            extra_body["stop_token_ids"] = self.stop_token_ids
        if hasattr(self, "skip_special_tokens"):
            extra_body["skip_special_tokens"] = self.skip_special_tokens

        if len(extra_body) > 0:
            if "extra_body" in kwargs:
                kwargs["extra_body"] = {**kwargs["extra_body"], **extra_body}
            else:
                kwargs["extra_body"] = extra_body

        start_time = time.time()
        api_response = await self.llm_ap.process_single_async(message, **kwargs)
        end_time = time.time()

        return api_response, end_time - start_time
