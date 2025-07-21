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

## D. 出力パターン自動検出とデコード

### パターン1: 標準JSONパターン (Hammer系)
```
入力例: [{"name": "func_name", "arguments": {"arg1": "val1"}}]
説明: 純粋なJSON配列形式。最もシンプルで確実
検出: `[` で始まり `]` で終わる
処理: 直接JSON.loads() -> name/argumentsキーを抽出
```

### パターン2: Markdownコードブロック内JSON (DeepSeek系)
```
入力例: 
```json
[{"name": "func", "arguments": {"arg": "value"}}]
```
または
```python
[{"name": "func", "arguments": {"arg": "value"}}]
```
説明: マークダウンのコードブロック内にJSONが含まれる
検出: ```json または ```python で包まれたJSONを正規表現で抽出
処理: コードブロックを除去 -> JSON解析
```

### パターン3: XMLタグパターン (Hermes系)
```
入力例:
<tool_call>
{"name": "func", "arguments": {"arg": "value"}}
</tool_call>
説明: XMLライクなタグでツール呼び出しを包む
検出: <tool_call>...</tool_call> パターンを正規表現で抽出
処理: タグ内容を抽出 -> JSON解析 -> name/argumentsキーを抽出
```

### パターン4: 特殊タグパターン (Llama 3.1系)
```
入力例: 
<|python_tag|>{"name": "func", "arguments": {"arg": "val"}}; {"name": "func2", "arguments": {"arg2": "val2"}}
説明: 特殊なPythonタグ + セミコロン区切りで複数関数
検出: <|python_tag|> プレフィックスの有無
処理: タグ除去 -> セミコロン分割 -> 各部分をJSON解析
```

### パターン5: 関数呼び出しタグパターン (Granite系)
```
入力例:
<function_call> {"name": "func", "arguments": {"arg": "value"}}
説明: <function_call> タグで包む + 特殊な"no_function"処理
検出: <function_call> で始まるパターン
処理: タグ内容抽出 -> "no_function"チェック -> JSON解析
```

### パターン6: Phiツール呼び出しタグパターン (Phi-FC系)
```
入力例: <|tool_call|>[{"name": "func", "arguments": {"arg": "value"}}]<|/tool_call|>
説明: Phi独自のツール呼び出しタグ + 配列またはオブジェクト形式
検出: <|tool_call|>...(<|/tool_call|>|EOF) パターンを正規表現で抽出
処理: タグ内容を抽出 -> 配列形式に統一 -> JSON解析
```

### パターン7: 複雑な思考タグパターン (MiniCPM系)
```
入力例:
<|thought_start|>
ユーザーは計算を求めているので、calculate関数を使います
<|thought_end|>
<|tool_call_start|>
```python
calculate(x=5, y=10)
```
<|tool_call_end|>
計算結果をお見せします
説明: 思考過程とツール呼び出しを分離
検出: <|tool_call_start|>...<|tool_call_end|> パターン
処理: fc2dict()関数で思考過程分離 -> Python AST解析で関数抽出
```

### パターン8: 改行区切りパターン (GLM系)
```
入力例:
func_name
{"arg1": "val1"}
説明: 1行目に関数名、2行目に引数JSON
検出: 改行で分割して行数チェック
処理: 1行目を関数名として取得 -> 2行目をJSON解析
```

### パターン9: 単純なJSONオブジェクト
```
入力例: {"name": "func", "arguments": {"arg": "value"}}
説明: 配列ではない単一のJSONオブジェクト
検出: JSON形式だが配列でない
処理: 直接JSON解析 -> name/argumentsキーを抽出
```

### パターン10: デフォルトのAST解析 (最後の手段)
```
説明: 上記パターンがすべて失敗した場合のフォールバック
処理: default_decode_ast_prompting()を使用してAST解析を試行
失敗時: 空のリストを返す
```

## E. モデル固有の実行結果処理
- Llama系: ipythonロール使用
- DeepSeek系: userロール使用（toolロール非対応）
- 標準: toolロール使用

## F. 生の出力保存機能
- 全てのモデル出力を元の形で保存
- デバッグとパターン解析のため
- 失敗したデコードの原因分析に使用

# 使用方法
model_config.pyで新しいモデルに対して model_handler=UnifiedOSSHandlerを指定するだけ
"""

import ast
import json
import re
import requests
from typing import Dict, List, Any, Union, Optional
from pathlib import Path
import os

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
from ...constants.eval_config import RESULT_PATH
from config_singleton import WandbConfigSingleton



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
    - chat templateからの設定自動取得
    - 生の出力保存機能（デバッグ用）
    """
    
    def __init__(self, model_name, temperature) -> None:
        super().__init__(model_name, temperature)
        
        # Chat templateからの設定取得
        self._load_config_from_chat_template()
        
        # Chat templateから特徴を検出（model_nameは使用しない）
        self._detect_model_characteristics_from_chat_template()
        
        # 生の出力を保存するディレクトリを設定
        self._setup_raw_output_logging()
    
    def _load_config_from_chat_template(self):
        """Chat templateをconfig_singletonから取得"""
        
        instance = WandbConfigSingleton.get_instance()
        cfg = instance.config
        
        # プロジェクトルートからの絶対パスを構築
        current_file = Path(__file__).resolve()
        project_root = current_file.parents[7]  # 7階層上がプロジェクトルート
        local_chat_template_path = project_root / f"chat_templates/{cfg.model.get('chat_template')}.jinja"
        if local_chat_template_path.exists():
            with local_chat_template_path.open(encoding="utf-8") as f:
                chat_template = f.read()
                self.chat_template = chat_template
        else:
            print(f"[UnifiedOSSHandler] Chat templateファイルが見つかりません: {local_chat_template_path}")
            self.chat_template = None
   
    
    def _setup_raw_output_logging(self):
        """生の出力を保存するためのセットアップ"""
        try:
            # 結果ディレクトリを作成
            self.raw_output_dir = Path(RESULT_PATH) / f"{self.model_name.replace('/', '_')}"
            self.raw_output_dir.mkdir(parents=True, exist_ok=True)
            
            # 生の出力を保存するファイル
            self.raw_output_file = self.raw_output_dir / "raw_outputs_debug.txt"
            
            # ファイルを初期化
            with self.raw_output_file.open("w", encoding="utf-8") as f:
                f.write(f"# 生の出力ログ - {self.model_name}\n")
                f.write(f"# このファイルは、{self.model_name}の生の出力を記録します\n")
                f.write(f"# デコードパターンの分析やデバッグに使用してください\n\n")
                
            print(f"[UnifiedOSSHandler] 生の出力ログファイルを初期化: {self.raw_output_file}")
            
        except Exception as e:
            print(f"[UnifiedOSSHandler] 生の出力ログセットアップ中にエラー: {e}")
            self.raw_output_file = None
    
    def _log_raw_output(self, raw_output: str, test_entry_id: str = "unknown", step: str = "decode"):
        """生の出力をファイルに記録"""
        if not self.raw_output_file:
            return
            
        try:
            with self.raw_output_file.open("a", encoding="utf-8") as f:
                f.write(f"\n{'='*80}\n")
                f.write(f"Test Entry ID: {test_entry_id}\n")
                f.write(f"Step: {step}\n")
                f.write(f"Timestamp: {__import__('datetime').datetime.now().isoformat()}\n")
                f.write(f"{'='*80}\n")
                f.write(f"Raw Output:\n{raw_output}\n")
                f.write(f"{'='*80}\n")
        except Exception as e:
            print(f"[UnifiedOSSHandler] 生の出力ログ記録中にエラー: {e}")
    
    def _log_decode_result(self, pattern_name: str, result: any, step: str = "decode_ast"):
        """デコード結果をログに記録"""
        if not self.raw_output_file:
            return
            
        try:
            with self.raw_output_file.open("a", encoding="utf-8") as f:
                f.write(f"Applied Pattern: {pattern_name}\n")
                f.write(f"Decoded Result: {result}\n")
                f.write(f"Result Type: {type(result)}\n")
                if isinstance(result, list):
                    f.write(f"Result Length: {len(result)}\n")
                    for i, item in enumerate(result):
                        f.write(f"  Item {i}: {item} (type: {type(item)})\n")
                f.write(f"Step: {step}\n")
                f.write(f"-" * 40 + "\n")
        except Exception as e:
            print(f"[UnifiedOSSHandler] デコード結果ログ記録中にエラー: {e}")

    def _detect_model_characteristics_from_chat_template(self):
        """Chat templateから具体的なトークン・タグベースで特徴を自動検出（model_nameは使用しない）"""
        # デフォルト値の設定
        self.has_thinking_tags = False          # <think>...</think>タグの有無
        self.has_header_tags = False           # <|start_header_id|>などのヘッダータグの有無
        self.has_im_tags = False               # <|im_start|>/<|im_end|>タグの有無
        self.has_tool_call_tags = False        # <|tool_call_start|>などのツール呼び出しタグの有無
        self.has_function_call_tags = False    # <function_call>タグの有無
        self.has_python_tags = False           # <|python_tag|>タグの有無
        self.has_tool_xml_tags = False         # <tool_call>XMLタグの有無
        self.has_phi_tool_tags = False         # <|tool_call|>Phiタグの有無
        self.has_coder_tokens = False          # コーダー関連のトークンの有無
        self.has_ipython_role = False          # ipythonロールの有無
        self.has_model_role = False            # modelロール（Gemma用）の有無
        self.has_fc_tokens = False             # FC（Function Calling）関連トークンの有無
        self.supports_reasoning = False        # 推論サポートの有無
        self.has_markdown_code_blocks = False  # markdownコードブロックの有無
        
        # Chat templateから特徴を検出（model_nameは一切使用しない）
        if self.chat_template:
            chat_template_lower = self.chat_template.lower()
            
            # 思考タグの検出（推論機能）
            if ("<think>" in chat_template_lower and "</think>" in chat_template_lower) or \
               ("</think>" in chat_template_lower):
                self.has_thinking_tags = True
                self.supports_reasoning = True
            
            # ヘッダータグの検出（Llama系で使用）
            if any(keyword in chat_template_lower for keyword in [
                "<|start_header_id|>", "<|end_header_id|>", "<|eot_id|>"
            ]):
                self.has_header_tags = True
            
            # im_start/im_endタグの検出（Qwen系で使用）
            if "<|im_start|>" in chat_template_lower and "<|im_end|>" in chat_template_lower:
                self.has_im_tags = True
            
            # ツール呼び出しタグの検出（MiniCPM系で使用）
            if any(keyword in chat_template_lower for keyword in [
                "<|tool_call_start|>", "<|tool_call_end|>", "<|thought_start|>", "<|thought_end|>"
            ]):
                self.has_tool_call_tags = True
            
            # 関数呼び出しタグの検出（Granite系で使用）
            if "<function_call>" in chat_template_lower:
                self.has_function_call_tags = True
            
            # Pythonタグの検出（Llama 3.1系で使用）
            if "<|python_tag|>" in chat_template_lower:
                self.has_python_tags = True
            
            # XMLツールタグの検出（Hermes系・Qwen系で使用）
            if "<tool_call>" in chat_template_lower and "</tool_call>" in chat_template_lower:
                self.has_tool_xml_tags = True
            
            # Phiツールタグの検出（Phi系で使用）
            if "<|tool_call|>" in chat_template_lower:
                self.has_phi_tool_tags = True
            
            # コーダー関連トークンの検出
            if any(keyword in chat_template_lower for keyword in [
                "｜tool▁call▁begin｜", "｜tool▁sep｜", "｜tool▁call▁end｜",
                "coding", "programming", "code generation"
            ]):
                self.has_coder_tokens = True
            
            # ipythonロールの検出
            if "ipython" in chat_template_lower:
                self.has_ipython_role = True
            
            # modelロールの検出（Gemma用）
            if '"model"' in chat_template_lower or "'model'" in chat_template_lower:
                self.has_model_role = True
            
            # FC関連トークンの検出
            if any(keyword in chat_template_lower for keyword in [
                "function call", "tool_calls", "function_name", 
                "function calling", "fc", "-fc"
            ]):
                self.has_fc_tokens = True
            
            # マークダウンコードブロックの検出（DeepSeek系・Phi系で使用）
            if any(keyword in chat_template_lower for keyword in [
                "```json", "```python", "markdown", "code block"
            ]):
                self.has_markdown_code_blocks = True
        
        # Chat templateが取得できない場合の処理
        if not self.chat_template:
            print(f"[UnifiedOSSHandler] Chat templateが取得できません。デフォルト処理を使用します。")
            # トークンベースの検出ができないため、全てFalseのまま（汎用処理を使用）
        
        # 検出結果をログ出力
        detected_features = []
        if self.has_thinking_tags:
            detected_features.append("Thinking Tags (<think>)")
        if self.has_header_tags:
            detected_features.append("Header Tags (<|start_header_id|>)")
        if self.has_im_tags:
            detected_features.append("IM Tags (<|im_start|>)")
        if self.has_tool_call_tags:
            detected_features.append("Tool Call Tags (<|tool_call_start|>)")
        if self.has_function_call_tags:
            detected_features.append("Function Call Tags (<function_call>)")
        if self.has_python_tags:
            detected_features.append("Python Tags (<|python_tag|>)")
        if self.has_tool_xml_tags:
            detected_features.append("Tool XML Tags (<tool_call>)")
        if self.has_phi_tool_tags:
            detected_features.append("Phi Tool Tags (<|tool_call|>)")
        if self.has_coder_tokens:
            detected_features.append("Coder Tokens")
        if self.has_markdown_code_blocks:
            detected_features.append("Markdown Code Blocks")
        if self.has_ipython_role:
            detected_features.append("IPython Role")
        if self.has_model_role:
            detected_features.append("Model Role")
        if self.has_fc_tokens:
            detected_features.append("FC Tokens")
        if self.supports_reasoning:
            detected_features.append("Reasoning Support")
        
        if detected_features:
            print(f"[UnifiedOSSHandler] 検出されたトークン・タグ特徴: {', '.join(detected_features)}")
        else:
            print(f"[UnifiedOSSHandler] 特定のトークン・タグ特徴は検出されませんでした（汎用処理を使用）")

        # 後方互換性のための変数設定（既存のコードで使用されているため）
        # model_nameは一切使用せず、純粋にトークンベースで設定
        self.is_deepseek_family = self.has_thinking_tags or self.has_coder_tokens
        self.is_deepseek_reasoning = self.has_thinking_tags
        self.is_deepseek_coder = self.has_coder_tokens
        self.is_llama_family = self.has_header_tags or self.has_ipython_role
        self.is_llama_31 = self.has_python_tags
        self.is_qwen_family = self.has_im_tags
        self.is_gemma_family = self.has_model_role
        self.is_phi_family = False  # Phiは特定のトークンパターンがないため
        self.is_phi_mini = False
        self.is_granite_family = self.has_function_call_tags
        self.is_hermes_family = self.has_tool_xml_tags and not self.has_im_tags  # HermesはXMLタグだがim_tagsなし
        self.is_hammer_family = False  # Hammerは標準JSONなので特定のタグなし
        self.is_minicpm_family = self.has_tool_call_tags
        self.is_glm_family = False  # GLMは特定のトークンパターンがないため
        self.is_qwen_fc_family = self.has_tool_xml_tags and self.has_im_tags  # Qwen-FCはXMLタグ+im_tags
        self.is_phi_fc_family = self.has_phi_tool_tags  # Phi-FCは<|tool_call|>タグ
        
        # FCモデルの判定もトークンベースで実行
        self.is_fc_model = self.has_fc_tokens

    @override
    def decode_ast(self, result, language="Python"):
        """
        出力パターンを自動検出してデコード
        """
        # 生の出力をログに記録
        self._log_raw_output(result, step="decode_ast")
        
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
                        if isinstance(item, dict) and "name" in item:
                            # "arguments"キーまたは"parameters"キーを確認
                            params = item.get("arguments") or item.get("parameters", {})
                            # paramsが辞書でない場合は空の辞書にする
                            if not isinstance(params, dict):
                                params = {}
                            decoded_output.append({item["name"]: params})
                        elif isinstance(item, dict) and len(item) == 1:
                            # 既に{func_name: params}形式の場合
                            func_name = list(item.keys())[0]
                            params = item[func_name]
                            if not isinstance(params, dict):
                                params = {}
                            decoded_output.append({func_name: params})
                    self._log_decode_result("標準JSON配列", decoded_output)
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
                            if isinstance(item, dict) and "name" in item:
                                params = item.get("arguments") or item.get("parameters", {})
                                if not isinstance(params, dict):
                                    params = {}
                                decoded_output.append({item["name"]: params})
                            elif isinstance(item, dict) and len(item) == 1:
                                func_name = list(item.keys())[0]
                                params = item[func_name]
                                if not isinstance(params, dict):
                                    params = {}
                                decoded_output.append({func_name: params})
                        self._log_decode_result("Markdownコードブロック内JSON", decoded_output)
                        return decoded_output
                    elif isinstance(parsed, dict) and "name" in parsed:
                        params = parsed.get("arguments") or parsed.get("parameters", {})
                        if not isinstance(params, dict):
                            params = {}
                        return [{parsed["name"]: params}]
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
                    if isinstance(tool_result, dict) and "name" in tool_result:
                        params = tool_result.get("arguments") or tool_result.get("parameters", {})
                        if not isinstance(params, dict):
                            params = {}
                        func_call.append({tool_result["name"]: params})
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
                            if isinstance(parsed, dict) and "name" in parsed:
                                # "arguments"キーまたは"parameters"キーを確認
                                params = parsed.get("arguments") or parsed.get("parameters", {})
                                if not isinstance(params, dict):
                                    params = {}
                                decoded_output.append({parsed["name"]: params})
                        except json.JSONDecodeError:
                            continue
                if decoded_output:
                    return decoded_output
            else:
                # セミコロンがない場合でも単一の関数呼び出しをチェック
                try:
                    parsed = json.loads(cleaned)
                    if isinstance(parsed, dict) and "name" in parsed:
                        params = parsed.get("arguments") or parsed.get("parameters", {})
                        if not isinstance(params, dict):
                            params = {}
                        return [{parsed["name"]: params}]
                except json.JSONDecodeError:
                    pass
        
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
                        
                        if not isinstance(args, dict):
                            args = {}
                        decoded_outputs.append({func_name: args})
                except json.JSONDecodeError:
                    decoded_outputs.append(match)
            
            if decoded_outputs:
                return decoded_outputs
        
        # パターン6: Phiツール呼び出しタグパターン (Phi-FC系)
        # <|tool_call|>[{"name": "func", "arguments": {...}}]<|/tool_call|>
        phi_tool_matches = re.findall(r'<\|tool_call\|>(.*?)(?:<\|/tool_call\|>|$)', result, re.DOTALL)
        if phi_tool_matches:
            decoded_outputs = []
            for match in phi_tool_matches:
                match = match.strip()
                # Phiは配列でない場合があるため、配列に変換
                if not match.startswith("[") and not match.endswith("]"):
                    match = "[" + match + "]"
                
                try:
                    parsed = json.loads(match)
                    if isinstance(parsed, list):
                        for item in parsed:
                            if isinstance(item, str):
                                item = json.loads(item)
                            if isinstance(item, dict) and "name" in item:
                                params = item.get("arguments") or item.get("parameters", {})
                                if not isinstance(params, dict):
                                    params = {}
                                decoded_outputs.append({item["name"]: params})
                except json.JSONDecodeError:
                    continue
            
            if decoded_outputs:
                return decoded_outputs
        
        # パターン7: 複雑な思考タグパターン (MiniCPM系)
        if "<|tool_call_start|>" in result and "<|tool_call_end|>" in result:
            msg = fc2dict(result)
            if "tool_calls" in msg and msg["tool_calls"] and len(msg["tool_calls"]) > 0:
                decoded_output = []
                for tool_call in msg["tool_calls"]:
                    if isinstance(tool_call, dict) and "name" in tool_call:
                        params = tool_call.get("arguments", {})
                        if not isinstance(params, dict):
                            params = {}
                        decoded_output.append({tool_call["name"]: params})
                return decoded_output
            else:
                return []
        
        # パターン8: 改行区切りパターン (GLM系)
        lines = result.split("\n")
        if len(lines) >= 2:
            try:
                func_name = lines[0].strip()
                args_json = lines[1].strip()
                args = json.loads(args_json)
                if not isinstance(args, dict):
                    args = {}
                return [{func_name: args}]
            except (json.JSONDecodeError, IndexError):
                pass
        
        # パターン9: 単純なJSONオブジェクト
        try:
            parsed = json.loads(result)
            if isinstance(parsed, dict) and "name" in parsed:
                params = parsed.get("arguments") or parsed.get("parameters", {})
                if not isinstance(params, dict):
                    params = {}
                return [{parsed["name"]: params}]
        except json.JSONDecodeError:
            pass
        
        # パターン10: デフォルトのAST解析 (最後の手段)
        try:
            default_result = default_decode_ast_prompting(result, language)
            # デフォルト結果が正しい形式かチェックして修正
            if isinstance(default_result, list):
                corrected_result = []
                for item in default_result:
                    if isinstance(item, dict) and len(item) == 1:
                        func_name = list(item.keys())[0]
                        params = item[func_name]
                        if not isinstance(params, dict):
                            params = {}
                        corrected_result.append({func_name: params})
                    elif isinstance(item, dict) and "name" in item:
                        params = item.get("arguments") or item.get("parameters", {})
                        if not isinstance(params, dict):
                            params = {}
                        corrected_result.append({item["name"]: params})
                return corrected_result
            return []
        except Exception:
            # 全て失敗した場合は空のリストを返す
            return []

    @override
    def decode_execute(self, result):
        """
        実行結果のデコード - decode_astと同じパターンを適用
        """
        # 生の出力をログに記録
        self._log_raw_output(result, step="decode_execute")
        
        if not result or not isinstance(result, str):
            return []
            
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
                # MiniCPMでもツール呼び出しがない場合は空のリストを返す
                return []
        
        # パターン4（実行用）: 特殊タグパターン (Llama 3.1系)
        # <|python_tag|>...;...
        if "<|python_tag|>" in result:
            cleaned = result.replace("<|python_tag|>", "").strip()
            if ";" in cleaned:
                calls = cleaned.split(";")
                execution_list = []
                for call in calls:
                    call = call.strip()
                    if call:
                        try:
                            parsed = json.loads(call)
                            if isinstance(parsed, dict) and "name" in parsed:
                                # "arguments"キーまたは"parameters"キーを確認
                                params = parsed.get("arguments") or parsed.get("parameters", {})
                                func_name = parsed["name"]
                                args_str = ", ".join([f"{k}={repr(v)}" for k, v in params.items()])
                                execution_list.append(f"{func_name}({args_str})")
                        except json.JSONDecodeError:
                            continue
                if execution_list:
                    return execution_list
            else:
                # セミコロンがない場合でも単一の関数呼び出しをチェック
                try:
                    parsed = json.loads(cleaned)
                    if isinstance(parsed, dict) and "name" in parsed:
                        params = parsed.get("arguments") or parsed.get("parameters", {})
                        func_name = parsed["name"]
                        args_str = ", ".join([f"{k}={repr(v)}" for k, v in params.items()])
                        return [f"{func_name}({args_str})"]
                except json.JSONDecodeError:
                    pass
        
        # パターン5（実行用）: Phiツール呼び出しタグパターン (Phi-FC系)
        # <|tool_call|>[{"name": "func", "arguments": {...}}]<|/tool_call|>
        phi_tool_matches = re.findall(r'<\|tool_call\|>(.*?)(?:<\|/tool_call\|>|$)', result, re.DOTALL)
        if phi_tool_matches:
            execution_list = []
            for match in phi_tool_matches:
                match = match.strip()
                # Phiは配列でない場合があるため、配列に変換
                if not match.startswith("[") and not match.endswith("]"):
                    match = "[" + match + "]"
                
                try:
                    parsed = json.loads(match)
                    if isinstance(parsed, list):
                        for item in parsed:
                            if isinstance(item, str):
                                item = json.loads(item)
                            if isinstance(item, dict) and "name" in item:
                                func_name = item["name"]
                                params = item.get("arguments", {})
                                args_str = ", ".join([f"{k}={repr(v)}" for k, v in params.items()])
                                execution_list.append(f"{func_name}({args_str})")
                except json.JSONDecodeError:
                    continue
            
            if execution_list:
                return execution_list
        
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
            # デコードに失敗した場合は空のリストを返す（文字列ではなく）
            return []

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
        # vLLMの新旧APIレスポンス形式に対応
        try:
            # 新しい形式: api_response.choices[0].message.content
            model_response = api_response.choices[0].message.content
        except AttributeError:
            try:
                # 古い形式: api_response.choices[0].text
                model_response = api_response.choices[0].text
            except AttributeError:
                # フォールバック: 文字列として取得
                model_response = str(api_response.choices[0])
        
        # 生の出力をログに記録
        self._log_raw_output(model_response, step="parse_query_response")
        
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
        
        # Qwen-FC: ツール呼び出し＋推論内容抽出（FCモデルでも推論対応）
        if self.is_qwen_fc_family:
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
        
        # Phi-FC: ツール呼び出し抽出
        if self.is_phi_fc_family:
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
        if self.is_qwen_fc_family:
            inference_data["message"].append(
                model_response_data["model_responses_message_for_chat_history"],
            )
            return inference_data
        
        # Phi-FC: 特殊なメッセージ追加
        if self.is_phi_fc_family:
            inference_data["message"].append({
                "role": "assistant",
                "content": model_response_data["model_responses_message_for_chat_history"],
            })
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
    def _add_execution_results_prompting(
        self, inference_data: dict, execution_results: list[str], model_response_data: dict
    ) -> dict:
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
            r"<｜tool▁call▁begin｜>(\w+)(\n)?```json(?:\n|\\n)(.*?)(?:\n|\\n)```<｜tool▁call▁end｜>"
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
            match = match.strip()
            # 並列ツール呼び出しの場合、リスト形式でない場合がある
            if not match.startswith("[") and not match.endswith("]"):
                match = "[" + match + "]"
            
            try:
                parsed = json.loads(match)
                if isinstance(parsed, list):
                    for item in parsed:
                        if isinstance(item, str):
                            item = json.loads(item)
                        result.append(item)
                else:
                    result.append(parsed)
            except json.JSONDecodeError:
                continue
        
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