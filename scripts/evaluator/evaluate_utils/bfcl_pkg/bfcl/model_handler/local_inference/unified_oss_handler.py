"""
統合OSSハンドラー - Nejumi Leaderboard用

このファイルは、Nejumi Leaderboardでの簡易OSS追加のために作成されました。
従来は各モデルごとに個別のハンドラーファイルが必要でしたが、
このハンドラーは出力パターンを自動検出し、適切にデコードします。

# 対応する機能と出力パターン

## A. モデル固有の前処理
- DeepSeek系: システムプロンプトのユーザープロンプト変換
- DeepSeek-Coder: 複雑なツールシステムプロンプト構築  
- Gemma: assistantロールのmodelロール置換
- FC系モデル: 独自システムプロンプト使用

## B. 推論内容の抽出
- DeepSeek-Reasoning, Qwen: </think>タグによる思考過程抽出
- 推論対応モデル: reasoning contentの分離と保持

## C. 複雑なツール呼び出し解析
- DeepSeek-Coder: <｜tool▁call▁begin｜>形式
- Phi-FC: <|tool_call|>形式 
- Qwen-FC: <tool_call>形式

## D. 出力パターン自動検出
1. **標準JSONパターン (Hammer系)**
   例: [{"name": "func_name", "arguments": {"arg1": "val1"}}]

2. **Markdown JSONパターン (DeepSeek系)**  
   例: ```json\n[{"name": "func", "arguments": {...}}]\n```

3. **XMLタグパターン (Hermes系)**
   例: <tool_call>\n{"name": "func", "arguments": {...}}\n</tool_call>

4. **特殊タグパターン (Llama 3.1系)**
   例: <|python_tag|>{"name": "func", "arguments": {...}}; {"name": "func2", ...}

5. **関数呼び出しタグパターン (Granite系)**
   例: <function_call> {"name": "func", "arguments": {...}}

6. **複雑な思考タグパターン (MiniCPM系)**
   例: <|thought_start|>...<|thought_end|>\n<|tool_call_start|>\n```python\nfunc(...)\n```\n<|tool_call_end|>

7. **改行区切りパターン (GLM系)**
   例: func_name\n{"arg1": "val1"}

## E. モデル固有の実行結果処理
- Llama系: ipythonロール使用
- DeepSeek系: userロール使用（toolロール非対応）
- 標準: toolロール使用

# 使用方法
model_config.pyで新しいモデルに対して model_handler=UnifiedOSSHandlerを指定するだけ
"""

import ast
import json
import re
from typing import Dict, List, Any, Union, Optional

from .base_oss_handler import OSSHandler
from ..utils import (
    combine_consecutive_user_prompts,
    convert_system_prompt_into_user_prompt,
    convert_to_function_call,
    default_decode_ast_prompting,
    default_decode_execute_prompting,
    func_doc_language_specific_pre_processing,
    system_prompt_pre_processing_chat_model,
)
from overrides import override


def fc2dict(sequence: str, 
           tool_call_start="<|tool_call_start|>",
           tool_call_end="<|tool_call_end|>",
           thought_start="<|thought_start|>", 
           thought_end="<|thought_end|>"):
    """MiniCPM形式の複雑なタグを処理"""
    try:
        # 思考過程を除去
        if thought_end in sequence and thought_start in sequence:
            thought_string, sequence = sequence.rsplit(thought_end, 1)
            thought_string = thought_string.split(thought_start, 1)[1]
        
        # ツール呼び出し部分を抽出
        if tool_call_start in sequence and tool_call_end in sequence:
            tool_call_string, content = sequence.rsplit(tool_call_end, 1)
            tool_call_string = tool_call_string.split(tool_call_start, 1)[1]
            
            # ```python...```の部分を除去
            tool_call_string = tool_call_string.strip()
            if tool_call_string.startswith("```python"):
                tool_call_string = tool_call_string[len("```python"):].strip()
            if tool_call_string.endswith("```"):
                tool_call_string = tool_call_string[:-3].strip()
            
            # AST解析で関数呼び出しを抽出
            try:
                parsed = ast.parse(tool_call_string, mode="eval")
                tool_calls = []
                
                if isinstance(parsed.body, ast.Call):
                    # 単一の関数呼び出し
                    func_name = parsed.body.func.id if hasattr(parsed.body.func, 'id') else str(parsed.body.func)
                    
                    # 引数を辞書に変換
                    args_dict = {}
                    for keyword in parsed.body.keywords:
                        args_dict[keyword.arg] = ast.literal_eval(keyword.value)
                    
                    tool_calls.append({"name": func_name, "arguments": args_dict})
                
                return {"tool_calls": tool_calls, "content": content.strip() if content else ""}
            except:
                return {"tool_calls": [], "content": sequence}
        
        return {"tool_calls": [], "content": sequence}
    except Exception:
        return {"tool_calls": [], "content": sequence}


class UnifiedOSSHandler(OSSHandler):
    """
    統合OSSハンドラー - 出力パターンを自動検出してデコード
    
    新しいOSSモデルを追加する際は、model_config.pyで
    model_handler=UnifiedOSSHandlerを指定するだけで使用可能
    
    主要な機能:
    - 自動的な出力パターン検出とデコード
    - モデル固有の前処理（システムプロンプト変換、ロール置換等）
    - 推論内容の抽出（reasoning content、思考過程）
    - 複雑なツール呼び出し形式の解析
    - モデル固有のメッセージ処理と実行結果追加
    """
    
    def __init__(self, model_name, temperature) -> None:
        super().__init__(model_name, temperature)
        
        # モデル名から特徴を推定（FC対応かどうかなど）
        self.is_fc_model = "-FC" in model_name or "fc" in model_name.lower()
        if self.is_fc_model:
            self.model_name_huggingface = model_name.replace("-FC", "")
        
        # モデル固有の設定を推定
        self._detect_model_characteristics(model_name)
        
    def _detect_model_characteristics(self, model_name: str):
        """モデル名から特徴を自動検出"""
        model_lower = model_name.lower()
        
        # 各モデルの特徴を検出
        self.is_deepseek_family = "deepseek" in model_lower
        self.is_deepseek_reasoning = "deepseek-r" in model_lower or "reasoning" in model_lower
        self.is_deepseek_coder = "deepseek" in model_lower and ("coder" in model_lower or "code" in model_lower)
        self.is_llama_family = "llama" in model_lower
        self.is_llama_31 = "llama-3.1" in model_lower or "llama_3_1" in model_lower
        self.is_gemma_family = "gemma" in model_lower
        self.is_phi_family = "phi" in model_lower
        self.is_phi_mini = "phi-4-mini" in model_lower
        self.is_qwen_family = "qwen" in model_lower
        self.is_granite_family = "granite" in model_lower
        self.is_hermes_family = "hermes" in model_lower
        self.is_hammer_family = "hammer" in model_lower
        self.is_minicpm_family = "minicpm" in model_lower
        self.is_glm_family = "glm" in model_lower
        
        # 推論機能の有無を検出
        self.supports_reasoning = (
            self.is_deepseek_reasoning or 
            self.is_qwen_family or
            "reasoning" in model_lower or
            "think" in model_lower
        )

    @override
    def decode_ast(self, result: str, language="Python") -> List[Dict[str, Any]]:
        """
        出力パターンを自動検出してデコード
        """
        if not result or not isinstance(result, str):
            return []
            
        result = result.strip()
        
        # パターン1: 標準JSON配列 (Hammer系)
        try:
            # 純粋なJSON配列をチェック
            if result.startswith('[') and result.endswith(']'):
                parsed = json.loads(result)
                if isinstance(parsed, list):
                    decoded_output = []
                    for item in parsed:
                        if isinstance(item, dict) and "name" in item and "arguments" in item:
                            decoded_output.append({item["name"]: item["arguments"]})
                        else:
                            decoded_output.append(item)
                    return decoded_output
        except json.JSONDecodeError:
            pass
        
        # パターン2: Markdownコードブロック内JSON (DeepSeek系)
        markdown_patterns = [
            r'```json\s*\n(.*?)\n```',
            r'```python\s*\n(.*?)\n```',
            r'```\s*\n(.*?)\n```'
        ]
        
        for pattern in markdown_patterns:
            match = re.search(pattern, result, re.DOTALL)
            if match:
                inner_content = match.group(1).strip()
                try:
                    parsed = json.loads(inner_content)
                    if isinstance(parsed, list):
                        decoded_output = []
                        for item in parsed:
                            if isinstance(item, dict) and "name" in item and "arguments" in item:
                                decoded_output.append({item["name"]: item["arguments"]})
                            else:
                                decoded_output.append(item)
                        return decoded_output
                    elif isinstance(parsed, dict) and "name" in parsed and "arguments" in parsed:
                        return [{parsed["name"]: parsed["arguments"]}]
                except json.JSONDecodeError:
                    pass
        
        # パターン3: XMLタグパターン (Hermes系)
        # <tool_call>...</tool_call>
        tool_call_matches = re.findall(r'<tool_call>\s*(.*?)\s*</tool_call>', result, re.DOTALL)
        if tool_call_matches:
            func_call = []
            for match in tool_call_matches:
                try:
                    match = match.replace("'", '"').strip()
                    tool_result = json.loads(match)
                    if isinstance(tool_result, dict) and "name" in tool_result and "arguments" in tool_result:
                        func_call.append({tool_result["name"]: tool_result["arguments"]})
                    else:
                        func_call.append(tool_result)
                except json.JSONDecodeError:
                    continue
            if func_call:
                return func_call
        
        # パターン4: 特殊タグパターン (Llama 3.1系)
        # <|python_tag|>...;...
        if "<|python_tag|>" in result:
            cleaned = result.replace("<|python_tag|>", "").strip()
            if ";" in cleaned:
                calls = cleaned.split(";")
                decoded_output = []
                for call in calls:
                    call = call.strip()
                    if call:
                        try:
                            parsed = json.loads(call)
                            if isinstance(parsed, dict) and "name" in parsed and "arguments" in parsed:
                                decoded_output.append({parsed["name"]: parsed["arguments"]})
                        except json.JSONDecodeError:
                            continue
                if decoded_output:
                    return decoded_output
        
        # パターン5: 関数呼び出しタグパターン (Granite系)
        # <function_call> {...}
        function_call_matches = re.findall(r'<function_call>\s*(.*?)(?=<function_call>|$)', result, re.DOTALL)
        if function_call_matches:
            decoded_outputs = []
            for match in function_call_matches:
                try:
                    match = match.strip()
                    parsed = json.loads(match)
                    if isinstance(parsed, dict):
                        func_name = parsed.get("name", "").strip()
                        args = parsed.get("arguments", {})
                        
                        if func_name == "no_function":
                            decoded_outputs.append("No function is called")
                            continue
                        
                        decoded_outputs.append({func_name: args})
                except json.JSONDecodeError:
                    decoded_outputs.append(match)
            
            if decoded_outputs:
                return decoded_outputs
        
        # パターン6: 複雑な思考タグパターン (MiniCPM系)
        if "<|tool_call_start|>" in result and "<|tool_call_end|>" in result:
            msg = fc2dict(result)
            if "tool_calls" in msg and msg["tool_calls"] and len(msg["tool_calls"]) > 0:
                return [{tool_call["name"]: tool_call["arguments"]} for tool_call in msg["tool_calls"]]
            else:
                return [msg["content"]] if msg.get("content") else []
        
        # パターン7: 改行区切りパターン (GLM系)
        lines = result.split("\n")
        if len(lines) >= 2:
            try:
                func_name = lines[0].strip()
                args_json = lines[1].strip()
                args = json.loads(args_json)
                return [{func_name: args}]
            except (json.JSONDecodeError, IndexError):
                pass
        
        # パターン8: 単純なJSONオブジェクト
        try:
            parsed = json.loads(result)
            if isinstance(parsed, dict) and "name" in parsed and "arguments" in parsed:
                return [{parsed["name"]: parsed["arguments"]}]
        except json.JSONDecodeError:
            pass
        
        # パターン9: デフォルトのAST解析 (最後の手段)
        try:
            return default_decode_ast_prompting(result, language)
        except Exception:
            # 全て失敗した場合は元のテキストを返す
            return [result] if result else []

    @override
    def decode_execute(self, result: str) -> Union[str, List]:
        """
        実行結果のデコード - decode_astと同じパターンを適用
        """
        if not result or not isinstance(result, str):
            return result
            
        result = result.strip()
        
        # Markdownコードブロックを除去
        markdown_patterns = [
            r'```json\s*\n(.*?)\n```',
            r'```python\s*\n(.*?)\n```',  
            r'```\s*\n(.*?)\n```'
        ]
        
        for pattern in markdown_patterns:
            match = re.search(pattern, result, re.DOTALL)
            if match:
                result = match.group(1).strip()
                break
        
        # MiniCPM形式の処理
        if "<|tool_call_start|>" in result and "<|tool_call_end|>" in result:
            msg = fc2dict(result)
            if "tool_calls" in msg and msg["tool_calls"] and len(msg["tool_calls"]) > 0:
                execution_list = []
                for tool_call in msg["tool_calls"]:
                    try:
                        function_name = tool_call["name"]
                        function_args = tool_call["arguments"]
                        args_str = ", ".join([f"{k}={repr(v)}" for k, v in function_args.items()])
                        execution_list.append(f"{function_name}({args_str})")
                    except:
                        continue
                return execution_list
            else:
                return msg.get("content", result)
        
        # GLM形式の処理
        lines = result.split("\n")
        if len(lines) >= 2:
            try:
                func_name = lines[0].strip()
                args_str = lines[1].strip()
                return convert_to_function_call([{func_name: args_str}])
            except:
                pass
        
        # デフォルトの処理
        try:
            return default_decode_execute_prompting(result)
        except Exception:
            return result

    @override
    def _pre_query_processing_prompting(self, test_entry: dict) -> dict:
        """
        クエリ前処理 - モデル固有の処理を自動適用
        """
        functions: list = test_entry["function"]
        test_category: str = test_entry["id"].rsplit("_", 1)[0]
        
        functions = func_doc_language_specific_pre_processing(functions, test_category)
        
        # FCモデルは通常独自のシステムプロンプトを使用
        if self.is_fc_model:
            return {"message": [], "function": functions}
        
        # DeepSeek-Coder: 複雑なシステムプロンプト構築
        if self.is_deepseek_coder:
            for round_idx in range(len(test_entry["question"])):
                test_entry["question"][round_idx] = convert_system_prompt_into_user_prompt(
                    test_entry["question"][round_idx]
                )
                test_entry["question"][round_idx] = combine_consecutive_user_prompts(
                    test_entry["question"][round_idx]
                )
            
            tool_system_prompt = "You are a helpful Assistant.\n\n## Tools\n\n### Function\n\nYou have the following functions available:\n\n"
            for function in functions:
                tool_system_prompt += f"- `{function['name']}`:\n"
                tool_system_prompt += f"```json\n{json.dumps(function, indent=4)}\n```\n"
            
            test_entry["question"][0].insert(
                0, {"role": "system", "content": tool_system_prompt}
            )
            return {"message": [], "function": functions}
        
        # DeepSeek-Reasoning: システムプロンプトをユーザープロンプトに変換
        if self.is_deepseek_reasoning or self.is_deepseek_family:
            test_entry["question"][0] = system_prompt_pre_processing_chat_model(
                test_entry["question"][0], functions, test_category
            )
            for round_idx in range(len(test_entry["question"])):
                test_entry["question"][round_idx] = convert_system_prompt_into_user_prompt(
                    test_entry["question"][round_idx]
                )
                test_entry["question"][round_idx] = combine_consecutive_user_prompts(
                    test_entry["question"][round_idx]
                )
            return {"message": [], "function": functions}
        
        # Gemma: システムプロンプト前処理＋ロール置換
        if self.is_gemma_family:
            test_entry["question"][0] = system_prompt_pre_processing_chat_model(
                test_entry["question"][0], functions, test_category
            )
            for round_idx in range(len(test_entry["question"])):
                test_entry["question"][round_idx] = combine_consecutive_user_prompts(
                    test_entry["question"][round_idx]
                )
                test_entry["question"][round_idx] = self._substitute_prompt_role(
                    test_entry["question"][round_idx]
                )
            return {"message": [], "function": functions}
        
        # 標準的な処理（システムプロンプトあり）
        test_entry["question"][0] = system_prompt_pre_processing_chat_model(
            test_entry["question"][0], functions, test_category
        )
        
        return {"message": [], "function": functions}
    
    @staticmethod
    def _substitute_prompt_role(prompts: List[dict]) -> List[dict]:
        """Gemma用: assistantロールをmodelロールに置換"""
        for prompt in prompts:
            if prompt["role"] == "assistant":
                prompt["role"] = "model"
        return prompts

    @override
    def _parse_query_response_prompting(self, api_response: Any) -> dict:
        """
        APIレスポンス解析 - 推論内容やツール呼び出しを抽出
        """
        model_response = api_response.choices[0].text
        
        # DeepSeek-Coder: 特殊なツール呼び出し抽出
        if self.is_deepseek_coder:
            extracted_tool_calls = self._extract_deepseek_coder_tool_calls(model_response)
            
            if len(extracted_tool_calls) > 0:
                model_responses_message_for_chat_history = {
                    "role": "assistant",
                    "content": None,
                    "tool_calls": extracted_tool_calls,
                }
                model_responses = [
                    {item["function"]["name"]: item["function"]["arguments"]}
                    for item in extracted_tool_calls
                ]
            else:
                model_responses_message_for_chat_history = {
                    "role": "assistant",
                    "content": model_response,
                }
                model_responses = model_response
            
            return {
                "model_responses": model_responses,
                "model_responses_message_for_chat_history": model_responses_message_for_chat_history,
                "input_token": api_response.usage.prompt_tokens,
                "output_token": api_response.usage.completion_tokens,
            }
        
        # Phi-FC: ツール呼び出し抽出
        if self.is_phi_family and self.is_fc_model:
            extracted_tool_calls = self._extract_phi_tool_calls(model_response)
            
            if self._is_tool_call_response_format(extracted_tool_calls) and len(extracted_tool_calls) > 0:
                model_responses = [
                    {item["name"]: item["arguments"]} for item in extracted_tool_calls
                ]
            else:
                model_responses = model_response
            
            return {
                "model_responses": model_responses,
                "model_responses_message_for_chat_history": model_response,
                "input_token": api_response.usage.prompt_tokens,
                "output_token": api_response.usage.completion_tokens,
            }
        
        # Qwen-FC: ツール呼び出し＋推論内容抽出
        if self.is_qwen_family and self.is_fc_model:
            extracted_tool_calls = self._extract_qwen_tool_calls(model_response)
            reasoning_content = ""
            cleaned_response = model_response
            
            if "</think>" in model_response:
                parts = model_response.split("</think>")
                reasoning_content = parts[0].rstrip("\n").split("<think>")[-1].lstrip("\n")
                cleaned_response = parts[-1].lstrip("\n")
            
            if len(extracted_tool_calls) > 0:
                model_responses_message_for_chat_history = {
                    "role": "assistant",
                    "content": "",
                    "tool_calls": extracted_tool_calls,
                }
            else:
                model_responses_message_for_chat_history = {
                    "role": "assistant",
                    "content": cleaned_response,
                }
            
            return {
                "model_responses": cleaned_response,
                "reasoning_content": reasoning_content,
                "model_responses_message_for_chat_history": model_responses_message_for_chat_history,
                "input_token": api_response.usage.prompt_tokens,
                "output_token": api_response.usage.completion_tokens,
            }
        
        # 推論対応モデル（DeepSeek-Reasoning, Qwen等）
        if self.supports_reasoning:
            reasoning_content = ""
            cleaned_response = model_response
            
            if "</think>" in model_response:
                parts = model_response.split("</think>")
                reasoning_content = parts[0].rstrip("\n").split("<think>")[-1].lstrip("\n")
                cleaned_response = parts[-1].lstrip("\n")
            
            return {
                "model_responses": cleaned_response,
                "reasoning_content": reasoning_content,
                "model_responses_message_for_chat_history": model_response,
                "input_token": api_response.usage.prompt_tokens,
                "output_token": api_response.usage.completion_tokens,
            }
        
        # 標準処理
        return {
            "model_responses": model_response,
            "input_token": api_response.usage.prompt_tokens,
            "output_token": api_response.usage.completion_tokens,
        }

    @override
    def _add_assistant_message_prompting(self, inference_data: dict, model_response_data: dict) -> dict:
        """
        アシスタントメッセージの追加 - モデル固有の処理
        """
        # DeepSeek-Coder: 特殊なメッセージ追加
        if self.is_deepseek_coder:
            inference_data["message"].append(
                model_response_data["model_responses_message_for_chat_history"],
            )
            return inference_data
        
        # Qwen-FC: 特殊なメッセージ追加
        if self.is_qwen_family and self.is_fc_model:
            inference_data["message"].append(
                model_response_data["model_responses_message_for_chat_history"],
            )
            return inference_data
        
        # 推論対応モデル: 元のレスポンス全体を保持
        if self.supports_reasoning and "model_responses_message_for_chat_history" in model_response_data:
            inference_data["message"].append({
                "role": "assistant",
                "content": model_response_data["model_responses_message_for_chat_history"],
            })
            return inference_data
        
        # 標準処理
        inference_data["message"].append({
            "role": "assistant",
            "content": model_response_data.get("model_responses", ""),
        })
        return inference_data

    @override  
    def _add_execution_results_prompting(self, inference_data: dict, execution_results: List[str], model_response_data: dict):
        """
        実行結果の追加 - モデル固有のロール使用
        """
        # Llama系: ipythonロールを使用
        if self.is_llama_family:
            for execution_result in execution_results:
                inference_data["message"].append({
                    "role": "ipython",
                    "content": execution_result,
                })
            return inference_data
        
        # DeepSeek系: userロールを使用（toolロール非対応）
        if self.is_deepseek_family:
            tool_message = {"role": "user", "content": []}
            for execution_result, decoded_model_response in zip(
                execution_results, model_response_data.get("model_responses_decoded", execution_results)
            ):
                tool_message["content"].append({
                    "role": "tool",
                    "name": str(decoded_model_response),
                    "content": execution_result,
                })
            inference_data["message"].append(tool_message)
            return inference_data
        
        # 標準処理: toolロールを使用
        for execution_result in execution_results:
            inference_data["message"].append({
                "role": "tool",
                "content": execution_result,
            })
        return inference_data

    def _extract_deepseek_coder_tool_calls(self, input_string: str) -> List[Dict]:
        """
        DeepSeek-Coder形式のツール呼び出し抽出
        入力例: "<｜tool▁calls▁begin｜><｜tool▁call▁begin｜>function<｜tool▁sep｜>function_name\n```json\n{...}\n```<｜tool▁call▁end｜>..."
        """
        pattern = re.compile(
            r"<｜tool▁call▁begin｜>(\w+)<｜tool▁sep｜>(.*?)(?:\n|\\n)```json(?:\n|\\n)(.*?)(?:\n|\\n)```<｜tool▁call▁end｜>"
        )
        
        matches = pattern.findall(input_string)
        result = []
        for match in matches:
            type_name = match[0]
            name = match[1]
            argument = match[2]
            try:
                argument = json.loads(argument)
            except:
                pass
            result.append({"type": type_name, "function": {"name": name, "arguments": argument}})
        return result

    def _extract_phi_tool_calls(self, input_string: str) -> List[Dict]:
        """
        Phi-FC形式のツール呼び出し抽出
        入力例: "<|tool_call|>[{\"name\": \"function_name\", \"arguments\": {...}}]<|/tool_call|>"
        """
        # 通常パターン
        pattern = r"<\|tool_call\|>(.*?)<\|/tool_call\|>"
        matches = re.findall(pattern, input_string, re.DOTALL)
        
        # 終了タグが欠けている場合
        if not matches:
            pattern = r"<\|tool_call\|>(.*?)(?:<\|/tool_call\|>)?$"
            matches = re.findall(pattern, input_string, re.DOTALL)
        
        result = []
        for match in matches:
            # 並列ツール呼び出しの場合、リスト形式でない場合がある
            if not match.startswith("[") and not match.endswith("]"):
                match = "[" + match + "]"
            
            try:
                match = json.loads(match)
            except json.JSONDecodeError:
                pass
            
            if isinstance(match, list):
                for item in match:
                    if isinstance(item, str):
                        item = eval(item)
                    result.append(item)
            else:
                result.append(match)
        
        return result

    def _extract_qwen_tool_calls(self, input_string: str) -> List[Dict]:
        """
        Qwen-FC形式のツール呼び出し抽出
        入力例: "<tool_call>\n{\"name\": \"function_name\", \"arguments\": {...}}\n</tool_call>"
        """
        pattern = r"<tool_call>\n(.*?)\n</tool_call>"
        matches = re.findall(pattern, input_string, re.DOTALL)
        
        result = []
        for match in matches:
            try:
                match = json.loads(match)
            except:
                pass
            result.append(match)
        return result

    @staticmethod
    def _is_tool_call_response_format(input_list: List) -> bool:
        """
        ツール呼び出しの形式チェック（Phi-FC用）
        """
        if not isinstance(input_list, list):
            return False
        
        for item in input_list:
            if not isinstance(item, dict):
                return False
            if "name" not in item:
                return False
            if "arguments" not in item:
                return False
            if len(item) != 2:
                return False
        
        return True