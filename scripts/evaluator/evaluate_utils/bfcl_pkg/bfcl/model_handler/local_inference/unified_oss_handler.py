"""
統合OSSハンドラー - シンプル版

基本方針:
- base_oss_handlerのデフォルト動作を最大限活用
- 明らかに必要な特殊ケースのみを最小限で処理  
- 複雑な推測や学習機能は排除
- 安全性重視（evalを使わない）

対応パターン:
1. 標準JSON形式（base_oss_handlerのデフォルト）
2. XMLツールタグ形式（<tool_call>...</tool_call>）
3. Markdownコードブロック内のJSON
4. 基本的な特殊タグ（<|python_tag|>など）

対応できないケース:
- 完全に独自のシステムプロンプト（Llama 3.1, Hermes等）
- 特殊なトークナイザー（DeepSeek-Coder等）
- 独自のロール（Gemmaの"model"ロール等）
- 複雑な推論形式（専用ハンドラーを使用推奨）
"""

import json
import re
from typing import Dict, List, Any

from .base_oss_handler import OSSHandler
from overrides import override


class UnifiedOSSHandler(OSSHandler):
    """
    統合OSSハンドラー（シンプル版）
    
    基本的にはbase_oss_handlerのデフォルト動作を使用し、
    一般的な出力形式のみ追加対応する汎用ハンドラー
    """
    
    def __init__(self, model_name, temperature) -> None:
        super().__init__(model_name, temperature)
        print(f"[UnifiedOSSHandler] 初期化完了 - モデル: {model_name}")

    @override
    def decode_ast(self, result, language="Python"):
        """
        一般的な出力形式のデコード
        複雑な推測は行わず、明らかなパターンのみ処理
        """
        if not result or not isinstance(result, str):
            return []
        
        result = result.strip()
        
        # パターン1: XMLツールタグ（Hermes系）
        if "<tool_call>" in result and "</tool_call>" in result:
            return self._decode_xml_tool_tags(result)
        
        # パターン2: 基本的な特殊タグ
        if "<|python_tag|>" in result:
            cleaned = result.replace("<|python_tag|>", "").strip()
            return super().decode_ast(cleaned, language)
        
        # パターン3: Markdownコードブロック
        markdown_match = re.search(r'```(?:json|python)\s*\n(.*?)\n```', result, re.DOTALL)
        if markdown_match:
            content = markdown_match.group(1).strip()
            return self._try_json_decode(content)
        
        # パターン4: コードブロック（前後の```のみ）
        if result.startswith("```") and result.endswith("```"):
            content = result[3:-3].strip()
            # json/pythonプレフィックス除去
            if content.startswith("json\n"):
                content = content[5:]
            elif content.startswith("python\n"):
                content = content[7:]
            return self._try_json_decode(content)
        
        # デフォルト: base_oss_handlerの動作
        return super().decode_ast(result, language)
    
    @override
    def decode_execute(self, result):
        """
        実行用デコード
        基本的にはdecode_astと同じロジック
        """
        if not result or not isinstance(result, str):
            return []
        
        # まずdecode_astで変換を試行
        decoded_ast = self.decode_ast(result)
        if decoded_ast:
            # ASTからexecution形式に変換
            execution_list = []
            for call in decoded_ast:
                for func_name, params in call.items():
                    param_str = ','.join([f'{k}={repr(v)}' for k, v in params.items()])
                    execution_list.append(f"{func_name}({param_str})")
            return execution_list
        
        # デフォルト処理
        return super().decode_execute(result)
    
    def _decode_xml_tool_tags(self, result: str) -> List[Dict]:
        """XMLツールタグのデコード（Hermes系）"""
        # <tool_call>...</tool_call>内容を抽出
        tool_call_pattern = re.findall(r'<tool_call>\s*(.*?)\s*</tool_call>', result, re.DOTALL)
        
        func_calls = []
        for match in tool_call_pattern:
            try:
                # JSONとして解析を試行
                tool_data = json.loads(match.strip())
                if isinstance(tool_data, dict) and "name" in tool_data:
                    func_name = tool_data["name"]
                    arguments = tool_data.get("arguments", {})
                    func_calls.append({func_name: arguments})
            except json.JSONDecodeError:
                print(f"[デコード] XMLツールタグのJSON解析失敗: {match[:100]}")
                continue
        
        return func_calls
    
    def _try_json_decode(self, content: str) -> List[Dict]:
        """JSONデコードを安全に試行"""
        try:
            parsed = json.loads(content)
            
            # 単一オブジェクトの場合
            if isinstance(parsed, dict):
                if "name" in parsed and "arguments" in parsed:
                    return [{parsed["name"]: parsed["arguments"]}]
                elif "name" in parsed and "parameters" in parsed:  # Llama系
                    return [{parsed["name"]: parsed["parameters"]}]
            
            # 配列の場合
            elif isinstance(parsed, list):
                func_calls = []
                for item in parsed:
                    if isinstance(item, dict):
                        if "name" in item and "arguments" in item:
                            func_calls.append({item["name"]: item["arguments"]})
                        elif "name" in item and "parameters" in item:  # Llama系
                            func_calls.append({item["name"]: item["parameters"]})
                return func_calls
                
        except json.JSONDecodeError:
            print(f"[デコード] JSON解析失敗: {content[:100]}")
        
        return []

    @override
    def _add_execution_results_prompting(
        self, inference_data: dict, execution_results: list[str], model_response_data: dict
    ) -> dict:
        """
        実行結果の追加
        標準的なtoolロールを使用（大部分のモデルで動作）
        """
        for execution_result in execution_results:
            inference_data["message"].append({
                "role": "tool",
                "content": execution_result,
            })
        
        return inference_data