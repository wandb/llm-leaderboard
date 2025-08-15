import json
import re
import time
from typing import Dict, List, Any, Generic, TypeVar
from enum import Enum

from config_singleton import WandbConfigSingleton
from .base_oss_handler import OSSHandler
from overrides import override
from pydantic import BaseModel, Field
from ..utils import (
    convert_to_function_call,
    system_prompt_pre_processing_chat_model,
    func_doc_language_specific_pre_processing,
)
from ...constants.default_prompts import JSONSCHEMA_SYSTEM_PROMPT


# Function nameを指定された関数名に制限するためのGeneric型
FuncNamesT = TypeVar('FuncNamesT')

class FunctionCall(BaseModel, Generic[FuncNamesT]):
    function: FuncNamesT
    arguments: Dict[str, Any] = Field(default_factory=dict)

class FunctionCallList(BaseModel, Generic[FuncNamesT]):
    function_calls: List[FunctionCall[FuncNamesT]] = Field(default_factory=list)
    unavailable_reason: str = ""


class UnifiedOSSJsonSchemaHandler(OSSHandler):
    """
    統合OSSハンドラー（JsonSchema版）
    
    基本的にはbase_oss_handlerのデフォルト動作を使用し、
    JSONSchemaによるStructured output(vLLMの`guided_json`を使用)を対応する汎用ハンドラー
    https://docs.vllm.ai/en/latest/features/structured_outputs.html
    """
    
    def __init__(self, model_name, temperature) -> None:
        super().__init__(model_name, temperature)

        cfg = WandbConfigSingleton.get_instance().config
        handler_config = cfg.bfcl.handler_config.unified_oss_jsonschema
        self.execution_result_role = handler_config.execution_result_role
        self.execution_result_include_call_str = handler_config.execution_result_include_call_str
        self.execution_result_include_call_id = handler_config.execution_result_include_call_id
        self.execution_result_join_parallel_calls = handler_config.execution_result_join_parallel_calls
        print(f"[UnifiedOSSJsonSchemaHandler] 初期化完了 - モデル: {model_name}")

    @override
    def decode_ast(self, result, language="Python"):
        return result

    @override
    def decode_execute(self, result):
        if isinstance(result, str):
            return []
        return convert_to_function_call(result)

    @override
    async def _query_prompting_async(self, inference_data: dict):
        # We use the OpenAI Chat Completions API
        function: list[dict] = inference_data["function"]
        message: list[dict] = inference_data["message"]

        # Chat Completion API uses messages directly, no need for _format_prompt
        # The vLLM server will apply the chat template automatically
        inference_data["inference_input_log"] = {"messages": message}

        function_names = Enum('FuncNames', {func["name"]: func["name"] for func in function})

        kwargs = {
            "model": self.model_path_or_id,
            "max_tokens": self._estimate_leftover_tokens_count(inference_data, fc=False),
            "timeout": 72000,  # Avoid timeout errors
            **self.generator_config,
            "response_format": FunctionCallList[function_names]
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
            test_entry["question"][0], functions, test_category,
            system_prompt_template=JSONSCHEMA_SYSTEM_PROMPT,
        )

        # system roleが許可されていないので、最初のuser messageにsystem messageを結合
        if re.search("[Gg]emma-?2", self.model_name_huggingface):
            system_message = test_entry["question"][0].pop(0)
            assert test_entry["question"][0][0]["role"] == "user"
            test_entry["question"][0][0]["content"] = system_message["content"] + "\n\n" + test_entry["question"][0][0]["content"]

        return {"message": [], "function": functions}

    @override
    def _parse_query_response_prompting(self, api_response: any) -> dict:
        parsed_output = api_response.parsed_output
        if len(parsed_output.function_calls) > 0:
            model_responses = [
                {function_call.function.value: function_call.arguments}
                for function_call in parsed_output.function_calls
            ]
        else:
            model_responses = parsed_output.unavailable_reason

        # Token数削減のためモデルのレスポンスがindentありの場合indentなしのJSONに変換
        model_responses_message_for_chat_history = {
            "role": "assistant", "content": json.dumps(json.loads(api_response.content), ensure_ascii=False)}

        return {
            "model_responses": model_responses,
            "model_responses_for_chat_history": model_responses_message_for_chat_history,
            "input_token": api_response.prompt_tokens,
            "output_token": api_response.completion_tokens,
        }

    @override
    def _add_assistant_message_prompting(
        self, inference_data: dict, model_response_data: dict
    ) -> dict:
        inference_data["message"].append(
            model_response_data["model_responses_for_chat_history"]
        )
        return inference_data

    @override
    def _add_execution_results_prompting(
        self, inference_data: dict, execution_results: list[str], model_response_data: dict
    ) -> dict:
        """
        実行結果の追加
        
        Example:

        # デフォルト
        execution_result_include_call_str=True,
        execution_result_include_call_id=False,
        execution_result_join_parallel_calls=False,
        execution_result_role="tool"
        ```
        {"role": "tool", "content": "[1]: tool1(arg1=1, arg2=2)\n{\"result\": \"foo\"}"},
        {"role": "tool", "content": "[2]: tool2(arg1=1, arg2=2)\n{\"result\": \"bar\"}"},
        ```

        # Mistral系(tool_call_id必須)
        execution_result_include_call_str=False,
        execution_result_include_call_id=True,
        execution_result_join_parallel_calls=False,
        execution_result_role="tool"
        ```
        {"role": "tool", "content": "{\"result\": \"foo\"}", "tool_call_id": "fc_0000001"},
        {"role": "tool", "content": "{\"result\": \"bar\"}", "tool_call_id": "fc_0000002"},
        ```

        # Gemma系(tool role不可、user/assistant交互必須)
        execution_result_include_call_str=True,
        execution_result_include_call_id=False,
        execution_result_join_parallel_calls=True,
        execution_result_role="user"
        ```
        {"role": "user", "content": "[1]: tool1(arg1=1, arg2=2)\n{\"result\": \"foo\"}\n\n[2]: tool2(arg1=1, arg2=2)\n{\"result\": \"bar\"}"},
        ```
        """
        if len(execution_results) == 0:
            return inference_data

        if 'execution_count' not in inference_data:
            inference_data['execution_count'] = 0

        decoded_calls = self.decode_execute(model_response_data["model_responses"])

        contents = []
        for decoded_call, execution_result in zip(decoded_calls, execution_results):
            execution_result = '{}' if execution_result is None else execution_result
            inference_data['execution_count'] += 1
            # call_idを入れない場合はメッセージの先頭にシーケンス番号を入れる
            content = '' if self.execution_result_include_call_id else f'[{inference_data["execution_count"]}]: '
            # Parallel call時にTool呼び出しとの対応付けのためcall strを入れる
            if self.execution_result_include_call_str:
                content += decoded_call + "\n"
            # 実行結果
            content += execution_result
            contents.append(content)

        # roleがuser/assistant交互でないと通らないモデルの場合、連続したtoolメッセージをまとめる
        if self.execution_result_join_parallel_calls:
            contents = ["\n\n".join(contents)]
        
        for content in contents:
            msg = {
                "role": self.execution_result_role,
                "content": content,
            }
            # tool_call_idを展開できるモデルの場合はtool_call_idを追加
            if self.execution_result_include_call_id:
                if 'mistral' in self.model_name:
                    # Mistralは9桁英数のバリデーションがtemplate内にある
                    msg["tool_call_id"] = f'fc{str(inference_data["execution_count"]).ljust(7, "0")}'
                else:
                    msg["tool_call_id"] = f'fc_{str(inference_data["execution_count"])}'
            inference_data["message"].append(msg)
        
        return inference_data
