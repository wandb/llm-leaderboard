"""
統合OSSハンドラー - Nejumi Leaderboard用

このファイルは、Nejumi Leaderboardでの簡易OSS追加のために作成されました。
従来は各モデルごとに個別のハンドラーファイルが必要でしたが、
このハンドラーは出力パターンを自動検出し、適切にデコードします。

# 対応する出力パターン

## 1. 標準JSONパターン (Hammer系)
例: [{"name": "func_name", "arguments": {"arg1": "val1"}}]
使用モデル: Hammer, Falcon FC, MistralFC 等

## 2. Markdown JSONパターン (DeepSeek系)  
例: ```json\n[{"name": "func", "arguments": {...}}]\n```
例: ```python\n[{"name": "func", "arguments": {...}}]\n```
使用モデル: DeepSeek, Phi, DeepSeek-Coder 等

## 3. XMLタグパターン (Hermes系)
例: <tool_call>\n{"name": "func", "arguments": {...}}\n</tool_call>
使用モデル: Hermes, Qwen等

## 4. 特殊タグパターン (Llama 3.1系)
例: <|python_tag|>{"name": "func", "arguments": {...}}; {"name": "func2", ...}
使用モデル: Llama 3.1系

## 5. 関数呼び出しタグパターン (Granite系)
例: <function_call> {"name": "func", "arguments": {...}}
使用モデル: Granite系

## 6. 複雑な思考タグパターン (MiniCPM系)
例: <|thought_start|>...<|thought_end|>\n<|tool_call_start|>\n```python\nfunc(...)\n```\n<|tool_call_end|>
使用モデル: MiniCPM系

## 7. 改行区切りパターン (GLM系)
例: func_name\n{"arg1": "val1"}
使用モデル: GLM系

## 8. プレーンテキストパターン
関数呼び出しが見つからない場合やエラー時の処理

# 使用方法
model_config.pyで新しいモデルに対して model_handler=UnifiedOSSHandlerを指定するだけ
"""

import ast
import json
import re
from typing import Dict, List, Any, Union

from .base_oss_handler import OSSHandler
from ..utils import (
    convert_to_function_call,
    default_decode_ast_prompting,
    default_decode_execute_prompting,
    func_doc_language_specific_pre_processing,
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
    """
    
    def __init__(self, model_name, temperature) -> None:
        super().__init__(model_name, temperature)

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
        クエリ前処理 - 標準的な処理を適用
        """
        functions: list = test_entry["function"]
        test_category: str = test_entry["id"].rsplit("_", 1)[0]
        
        functions = func_doc_language_specific_pre_processing(functions, test_category)
        
        return {"message": [], "function": functions}

    @override  
    def _add_execution_results_prompting(self, inference_data: dict, execution_results: List[str], model_response_data: dict):
        """
        実行結果の追加 - 標準的なuserロールを使用
        """
        for execution_result in execution_results:
            inference_data["message"].append({
                "role": "user",
                "content": f"Execution result: {execution_result}",
            }) 