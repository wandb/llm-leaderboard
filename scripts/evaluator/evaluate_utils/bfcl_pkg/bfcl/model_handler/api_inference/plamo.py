import json
import os
import time
from typing import Any, Optional

from .openai_completion import OpenAICompletionsHandler
from ..model_style import ModelStyle
from ..utils import (
    combine_consecutive_user_prompts,
    func_doc_language_specific_pre_processing,
    retry_with_backoff,
    system_prompt_pre_processing_chat_model,
)
from openai import OpenAI, RateLimitError
from overrides import override


class PLaMoAPIHandler(OpenAICompletionsHandler):
    def __init__(self, model_name, temperature) -> None:
        super().__init__(model_name, temperature)
        self.model_style = ModelStyle.OpenAI_Completions
        
        # PLaMo API設定
        base_url = "https://api.platform.preferredai.jp/v1"
        api_key = os.getenv("OPENAI_COMPATIBLE_API_KEY")  # 統一された環境変数名を使用
        
        if not api_key:
            raise ValueError(
                "OPENAI_COMPATIBLE_API_KEY environment variable is required for PLaMo API. "
                "Please set your PLaMo API key in the environment."
            )
            
        self.client = OpenAI(
            base_url=base_url,
            api_key=api_key
        )

    @retry_with_backoff(error_type=[RateLimitError, json.JSONDecodeError])
    def generate_with_backoff(self, **kwargs):
        start_time = time.time()
        
        # PLaMoのサポートパラメータのみを使用
        plamo_supported_params = {
            "model", "messages", "max_tokens", "temperature", "top_p", 
            "tools", "tool_choice", "n", "stop", "stream"
        }
        
        # サポートされていないパラメータを除去
        filtered_kwargs = {k: v for k, v in kwargs.items() if k in plamo_supported_params}
        
        # PLaMo固有の調整
        if "max_tokens" in filtered_kwargs:
            # PLaMoの最大出力トークン数は4096
            filtered_kwargs["max_tokens"] = min(4096, filtered_kwargs.get("max_tokens", 4096))
        
        # PLaMoのデフォルト値を設定
        if "temperature" not in filtered_kwargs:
            filtered_kwargs["temperature"] = 0.1
        if "top_p" not in filtered_kwargs:
            filtered_kwargs["top_p"] = 0.9
        
        # 詳細なデバッグログ
        print(f"[PLaMo DEBUG] ===== API REQUEST DEBUG =====")
        print(f"[PLaMo DEBUG] Params: {list(filtered_kwargs.keys())}")
        print(f"[PLaMo DEBUG] Model: {filtered_kwargs.get('model')}")

        # メッセージを正規化（dict以外のオブジェクトも受容）
        if "messages" in filtered_kwargs and filtered_kwargs["messages"]:
            normalized_messages = []
            for m in filtered_kwargs["messages"]:
                if isinstance(m, dict):
                    normalized_messages.append(m)
                else:
                    # OpenAI SDKのChatCompletionMessageなどをdictへ変換
                    role = getattr(m, "role", None)
                    content = getattr(m, "content", None)
                    tool_calls = getattr(m, "tool_calls", None)
                    as_dict = {}
                    if role is not None:
                        as_dict["role"] = role
                    if content is not None:
                        as_dict["content"] = content if isinstance(content, str) else str(content)
                    if tool_calls:
                        tc_list = []
                        for tc in tool_calls:
                            tc_id = getattr(tc, "id", None)
                            func = getattr(tc, "function", None)
                            func_name = getattr(func, "name", None) if func is not None else None
                            func_args = getattr(func, "arguments", None) if func is not None else None
                            tc_item = {
                                "type": "function",
                                "function": {"name": func_name, "arguments": func_args},
                            }
                            if tc_id is not None:
                                tc_item["id"] = tc_id
                            tc_list.append(tc_item)
                        as_dict["tool_calls"] = tc_list
                    normalized_messages.append(as_dict)

            filtered_kwargs["messages"] = normalized_messages

        print(f"[PLaMo DEBUG] Messages count: {len(filtered_kwargs.get('messages', []))}")
        print(f"[PLaMo DEBUG] Max tokens: {filtered_kwargs.get('max_tokens')}")
        print(f"[PLaMo DEBUG] Temperature: {filtered_kwargs.get('temperature')}")
        print(f"[PLaMo DEBUG] Top-p: {filtered_kwargs.get('top_p')}")
        
        if "tools" in filtered_kwargs:
            tools = filtered_kwargs["tools"]
            print(f"[PLaMo DEBUG] Tools count: {len(tools)}")
            for i, tool in enumerate(tools):
                print(f"[PLaMo DEBUG] Tool {i}: {tool.get('function', {}).get('name', 'N/A')}")
                
        if "tool_choice" in filtered_kwargs:
            print(f"[PLaMo DEBUG] Tool choice: {filtered_kwargs['tool_choice']}")
        else:
            print(f"[PLaMo DEBUG] Tool choice: None (auto)")
            
        # メッセージ内容もログ出力
        messages = filtered_kwargs.get('messages', [])
        for i, msg in enumerate(messages):
            if isinstance(msg, dict):
                role = msg.get('role', 'unknown')
                raw_content = msg.get('content', '')
            else:
                role = getattr(msg, 'role', 'unknown')
                raw_content = getattr(msg, 'content', '')
            content_str = str(raw_content)
            content = content_str[:100] + '...' if len(content_str) > 100 else content_str
            print(f"[PLaMo DEBUG] Message {i} ({role}): {content}")
            
        print(f"[PLaMo DEBUG] ===== END DEBUG =====")
        
        try:
            api_response = self.client.chat.completions.create(**filtered_kwargs)
            end_time = time.time()
            print(f"[PLaMo DEBUG] API SUCCESS - Response time: {end_time - start_time:.2f}s")
            return api_response, end_time - start_time
        except Exception as e:
            print(f"[PLaMo DEBUG] API FAILED: {e}")
            print(f"[PLaMo DEBUG] Error type: {type(e)}")
            raise

    @override
    def _query_prompting(self, inference_data: dict):
        """
        Call the PLaMo API in prompting mode to get the response.
        Return the response object that can be used to feed into the decode method.
        """
        message: list[dict] = inference_data["message"]
        inference_data["inference_input_log"] = {"message": repr(message)}

        # PLaMoは現在 plamo-2.0-prime のみサポート
        api_name = "plamo-2.0-prime"

        return self.generate_with_backoff(
            model=api_name,
            messages=message,
            max_tokens=4096,
            temperature=self.temperature,
            top_p=0.9,
        )

    @override
    def _pre_query_processing_prompting(self, test_entry: dict) -> dict:
        functions: list = test_entry["function"]
        test_category: str = test_entry["id"].rsplit("_", 1)[0]

        functions = func_doc_language_specific_pre_processing(functions, test_category)
        # 先頭ターンにSystemプロンプトを付与（他ハンドラーと同様の流儀）
        test_entry["question"][0] = system_prompt_pre_processing_chat_model(
            test_entry["question"][0], functions, test_category
        )
        # 連続するuserを結合（必要なモデルで安定化）
        test_entry["question"][0] = combine_consecutive_user_prompts(test_entry["question"][0])

        # メッセージはここでは空を返し、後段で add_first_turn_message_prompting が埋める
        return {"message": []}

    @override  
    def _query_FC(self, inference_data: dict):
        """
        Call the PLaMo API in function calling mode to get the response.
        PLaMoのFunction Calling仕様に特別に対応
        """
        message: list[dict] = inference_data["message"]
        tools = inference_data["tools"]
        inference_data["inference_input_log"] = {
            "message": repr(message),
            "tools": tools,
        }

        api_name = "plamo-2.0-prime"
        
        # PLaMo特別仕様：複数関数の場合はFunction Callingを無効化
        # PLaMoは複数の関数を同時に処理できない可能性
        if len(tools) > 1:
            print(f"[PLaMo WARNING] PLaMo may not support multiple functions ({len(tools)} tools). Fallback to prompt mode.")
            # 複数関数の場合はプロンプトモードにフォールバック
            return self._fallback_to_prompt_mode(message, tools)
        
        # 単一関数の場合のみFunction Callingを使用
        tool_choice = None
        if len(tools) == 1:
            # PLaMoのサンプルコードに合わせた形式
            tool_choice = {
                "type": "function", 
                "function": {"name": tools[0]["function"]["name"]}
            }

        kwargs = {
            "model": api_name,
            "messages": message,
            "max_tokens": 4096,
            "temperature": self.temperature,
            "top_p": 0.9,
        }
        
        if len(tools) > 0:
            kwargs["tools"] = tools
            
        # PLaMoは単一関数の場合のみtool_choiceを強制指定
        if tool_choice is not None:
            kwargs["tool_choice"] = tool_choice
            
        print(f"[PLaMo FC] Using tool_choice: {tool_choice}")
        print(f"[PLaMo FC] Tools: {len(tools)} functions")
        
        return self.generate_with_backoff(**kwargs)

    def _fallback_to_prompt_mode(self, message: list[dict], tools: list[dict]):
        """PLaMoが複数関数をサポートしない場合のプロンプトモードフォールバック（MiniCPMスタイル）"""
        
        # MiniCPMスタイルの関数定義生成
        functions_def = []
        for tool in tools:
            func = tool.get("function", {})
            name = func.get("name", "unknown")
            description = func.get("description", "")
            params = func.get("parameters", {}).get("properties", {})
            
            # パラメータ情報を含む関数定義
            param_list = []
            for param_name, param_info in params.items():
                param_type = param_info.get("type", "str")
                param_desc = param_info.get("description", "")
                param_list.append(f"{param_name}: {param_type} - {param_desc}")
            
            required_params = func.get("parameters", {}).get("required", [])
            func_def = f"def {name}({', '.join(required_params)}):\n    \"\"\"{description}\"\"\""
            functions_def.append(func_def)
        
        # MiniCPMスタイルのシステムプロンプト
        system_content = f"""# Functions
Here is a list of functions that you can invoke:
```python
{chr(10).join(functions_def)}
```

# Function Call Rule and Output Format
- If the user's question can be answered without calling any function, please answer the user's question directly.
- If the user's question cannot be answered without calling any function, and the user has provided enough information to call functions to solve it, you should call the functions.
- Use default parameters unless the user has specified otherwise.
- You should answer in the following format:

<|tool_call_start|>
```python
func1(param1="value1", param2="value2")
func2(param1="value")
```
<|tool_call_end|>
{{answer the user's question directly or ask the user for more information}}"""
        
        # メッセージを再構築
        modified_message = message.copy()
        if modified_message and modified_message[0].get("role") == "system":
            modified_message[0]["content"] = system_content + "\n\n" + modified_message[0]["content"]
        else:
            modified_message.insert(0, {"role": "system", "content": system_content})
        
        # プロンプトモードで実行
        kwargs = {
            "model": "plamo-2.0-prime",
            "messages": modified_message,
            "max_tokens": 4096,
            "temperature": self.temperature,
            "top_p": 0.9,
        }
        
        print(f"[PLaMo FALLBACK] Using MiniCPM-style prompt mode for {len(tools)} functions")
        return self.generate_with_backoff(**kwargs)

    @override
    def _pre_query_processing_FC(self, inference_data: dict, test_entry: dict) -> dict:
        inference_data["message"] = []
        return inference_data
    
    @override
    def _compile_tools(self, inference_data: dict, test_entry: dict) -> dict:
        from ..utils import convert_to_tool
        from ...constants.type_mappings import GORILLA_TO_OPENAPI
        
        functions: list = test_entry["function"]
        test_category: str = test_entry["id"].rsplit("_", 1)[0]

        functions = func_doc_language_specific_pre_processing(functions, test_category)
        tools = convert_to_tool(functions, GORILLA_TO_OPENAPI, self.model_style)
        
        inference_data["tools"] = tools
        return inference_data
    
    @override
    def _parse_query_response_FC(self, api_response: any) -> dict:
        """PLaMoのAPI応答をパース（OpenAI標準パターン）"""
        try:
            # Function Calling応答を試行
            model_responses = [
                {func_call.function.name: func_call.function.arguments}
                for func_call in api_response.choices[0].message.tool_calls
            ]
            tool_call_ids = [
                func_call.id for func_call in api_response.choices[0].message.tool_calls
            ]
            print(f"[PLaMo] Parsed {len(model_responses)} function calls")
        except Exception as e:
            # 通常のテキスト応答の場合
            print(f"[PLaMo] No function calls found, treating as text: {e}")
            model_responses = api_response.choices[0].message.content
            tool_call_ids = []

        model_responses_message_for_chat_history = api_response.choices[0].message

        return {
            "model_responses": model_responses,
            "model_responses_message_for_chat_history": model_responses_message_for_chat_history,
            "tool_call_ids": tool_call_ids,
            "input_token": api_response.usage.prompt_tokens if hasattr(api_response, 'usage') else 0,
            "output_token": api_response.usage.completion_tokens if hasattr(api_response, 'usage') else 0,
        }
        
    @override
    def decode_ast(self, result, language="Python"):
        """PLaMoの応答をデコード（フォールバック対応）"""
        if "FC" in self.model_name or self.is_fc_model:
            # Function Calling モードの場合
            try:
                # resultが文字列の場合は、MiniCPMスタイルのパースを試行
                if isinstance(result, str):
                    print(f"[PLaMo] FC got string result, trying MiniCPM-style parsing: {result[:100]}...")
                    return self._parse_minicpm_style_output(result, language)
                
                # OpenAI標準のFunction Calling応答形式を想定
                decoded_output = []
                for invoked_function in result:
                    name = list(invoked_function.keys())[0]
                    params = json.loads(invoked_function[name])
                    decoded_output.append({name: params})
                return decoded_output
            except Exception as e:
                print(f"[PLaMo] FC decode error: {e}")
                print(f"[PLaMo] Result type: {type(result)}, Content: {str(result)[:200]}...")
                # 最終フォールバック
                from ..utils import default_decode_ast_prompting
                return default_decode_ast_prompting(result, language)
        else:
            # Prompt モードの場合
            from ..utils import default_decode_ast_prompting
            return default_decode_ast_prompting(result, language)

    def _parse_minicpm_style_output(self, result: str, language="Python"):
        """MiniCPMスタイルの出力をパース"""
        import re
        import ast
        
        # <|tool_call_start|>と<|tool_call_end|>の間を抽出
        pattern = r'<\|tool_call_start\|>(.*?)<\|tool_call_end\|>'
        matches = re.findall(pattern, result, re.DOTALL)
        
        if not matches:
            print(f"[PLaMo] No tool_call markers found, using default parsing")
            from ..utils import default_decode_ast_prompting
            return default_decode_ast_prompting(result, language)
        
        # Pythonコードブロックを抽出
        tool_call_content = matches[0].strip()
        if tool_call_content.startswith('```python'):
            tool_call_content = tool_call_content[9:]  # ```python を除去
        if tool_call_content.endswith('```'):
            tool_call_content = tool_call_content[:-3]  # ``` を除去
        
        try:
            # ASTパースでfunction callsを抽出
            parsed = ast.parse(tool_call_content.strip())
            decoded_output = []
            
            for node in ast.walk(parsed):
                if isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
                    func_name = node.func.id
                    func_args = {}
                    
                    # キーワード引数を抽出
                    for keyword in node.keywords:
                        if isinstance(keyword.value, ast.Constant):
                            func_args[keyword.arg] = keyword.value.value
                        elif isinstance(keyword.value, ast.Str):  # Python 3.7以前の互換性
                            func_args[keyword.arg] = keyword.value.s
                    
                    decoded_output.append({func_name: func_args})
            
            print(f"[PLaMo] Successfully parsed {len(decoded_output)} function calls")
            return decoded_output
            
        except Exception as e:
            print(f"[PLaMo] MiniCPM-style parsing failed: {e}")
            from ..utils import default_decode_ast_prompting
            return default_decode_ast_prompting(result, language)

    @override
    def decode_execute(self, result):
        """PLaMoの実行結果をデコード"""
        if "FC" in self.model_name or self.is_fc_model:
            from ..utils import convert_to_function_call
            return convert_to_function_call(result)
        else:
            from ..utils import default_decode_execute_prompting
            return default_decode_execute_prompting(result)

    # === 重要：Function Callingに必要なメソッド群 ===
    
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
        inference_data["message"].append(
            model_response_data["model_responses_message_for_chat_history"]
        )
        return inference_data

    def _add_execution_results_FC(
        self,
        inference_data: dict,
        execution_results: list[str],
        model_response_data: dict,
    ) -> dict:
        # PLaMo用の実行結果処理
        from ..utils import convert_execution_results_to_tool_messages
        
        # Add the execution results to the current round result, one at a time
        for i, execution_result in enumerate(execution_results):
            tool_message = convert_execution_results_to_tool_messages(
                execution_result, model_response_data["tool_call_ids"][i]
            )
            inference_data["message"].append(tool_message)

        return inference_data