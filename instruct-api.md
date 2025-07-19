

# Task: BFCLにモデルを追加
##　追加したいモデル  ★(ここを変える)
- gpt-4.1-mini-2025-04-14
- o3-2025-04-16
- o4-mini-2025-04-16

## 変更するファイル
### model specific ★(ここを変える)
- /home/olachinkeigpu/Project/llm-leaderboard/scripts/evaluator/evaluate_utils/bfcl_pkg/bfcl/model_handler/api_inference/claude.py

### 共通
- bfcl/model_handler/base_handler.py
- bfcl/constants/model_config.py
- /home/olachinkeigpu/Project/llm-leaderboard/scripts/evaluator/evaluate_utils/bfcl_pkg/SUPPORTED_MODELS.md # Nejumi Leaderboardで追加したということをわかりやすく入れる

## 補足



## -------------------------------------------------------------------
## 進め方（Reference）
- gitで変更を管理しながら進め、いつでも戻れるようにして

#-----------モデル共有------------
# 追加のための細かいinstructionは下記
### 新しくモデルを追加する方法
- 公式の[Contributing Guide](./CONTRIBUTING.md)をご確認ください。以下、日本語でわかりやすく解説 & Nejumi Leaderboardに特化した対応について解説をします。

#### OSSモデルの場合
1. `bfcl/model_handler/local_inference/base_oss_handler.py`を確認しつつ、新しいモデルの新しいhandler classをllm-leaderboard/scripts/evaluator/evaluate_utils/bfcl_pkg/bfcl/model_handler/local_inferenceに作成してください。
  - handlerの作成については、こちらを参考にしてください。
2. その後`bfcl/constants/model_config.py`に、新しいモデルの情報を追加します。
3. modelごとのconfig内のbfcl_model_nameに`bfcl/constants/model_config.py`に追加したモデル名を記載してください

#### APIの場合
1. `bfcl/model_handler/base_handler.py`を確認しつつ、新しいモデルの新しいhandler classをllm-leaderboard/scripts/evaluator/evaluate_utils/bfcl_pkg/bfcl/model_handler/api_inferenceに作成して下さい。
2. その後`bfcl/constants/model_config.py`に、新しいモデルの情報を追加します。
3. modelごとのconfig内のbfcl_model_nameに`bfcl/constants/model_config.py`に追加したモデル名を記載してください

## 仕組み理解のための解説
### 質問1: bfcl/model_handler/base_handler.py は何をやっている？
**BaseHandlerクラス**は、**BFCL（Berkeley Function-calling Leaderboard）における言語モデルの評価を行うための基盤となる抽象クラス**です。

#### 🎯 主要な役割と機能

**1. モデル推論の統一インターフェース**
- 異なるAPIプロバイダー（OpenAI、Claude、Geminiなど）に対して共通のインターフェースを提供
- `inference()`メソッドが推論のエントリーポイントとして機能
- Function Calling（FC）モードとPromptingモードの両方をサポート

**2. シングルターンとマルチターンの対話処理**
- `inference_single_turn_FC/prompting()`: 単発の質問応答処理
- `inference_multi_turn_FC/prompting()`: 複数回の対話を行う処理
- マルチターンでは関数の実行結果を次のターンに引き継ぎ、連続的な対話が可能

**3. 関数呼び出し（Function Calling）の実行管理**
- テストエントリから関数定義を取得し、モデルが適切な関数を呼び出せるよう管理
- 関数の実行結果を取得し、次のクエリに反映
- `MAXIMUM_STEP_LIMIT`による無限ループ防止機能

**4. トークン数とレイテンシの計測**
- 入力・出力トークン数の正確な計測
- API呼び出しの応答時間測定
- 評価指標として重要なメタデータの収集

**5. 状態管理とログ記録**
- クラスインスタンスの状態変化を追跡
- 詳細な推論ログの記録（デバッグ用）
- 実行結果のJSON形式での永続化

**6. エラーハンドリング**
- モデル応答のデコード失敗時の適切な処理
- ステップ数上限による強制終了機能
- 実行時エラーの捕捉とログ記録

#### 🏗️ アーキテクチャ設計
BaseHandlerクラスは**テンプレートメソッドパターン**を採用しており、以下のメソッドが抽象メソッドとして定義され、各APIプロバイダーでの具体的な実装が必要です：

**Function Callingモード用:**
- `_query_FC()`: APIへの実際のクエリ実行
- `_pre_query_processing_FC()`: クエリ前の前処理
- `_compile_tools()`: 関数定義のコンパイル
- `_parse_query_response_FC()`: API応答の解析
- `add_first_turn_message_FC()`: 初回メッセージの追加
- `_add_assistant_message_FC()`: アシスタント応答の追加
- `_add_execution_results_FC()`: 実行結果の追加

**Promptingモード用:**
- `_query_prompting()`: プロンプトベースのクエリ実行
- `_pre_query_processing_prompting()`: プロンプト前処理
- `_parse_query_response_prompting()`: プロンプト応答の解析
- 対応するメッセージ追加メソッド群

#### 💡 FCモード vs Promptingモードの違い

| 項目 | FCモード | Promptingモード |
|------|----------|----------------|
| **出力形式** | 構造化されたJSON | 自然言語+関数呼び出し |
| **精度** | 高い（構造が保証） | 中程度（解析が必要） |
| **対応モデル** | OpenAI、Claude等の新しいモデル | より幅広いモデル |
| **実装の複雑さ** | シンプル | 複雑（テキスト解析が必要） |

**FCモードの例:**
```python
# モデル出力（構造化）
{"tool_calls": [{"function": {"name": "get_weather", "arguments": "{\"location\": \"東京\"}"}}]}
```

**Promptingモードの例:**
```python
# モデル出力（自然言語）
"[get_weather(location='東京')]"
# ↓ AST解析が必要
[{'get_weather': {'location': '東京'}}]
```

#### 🔧 AST解析（Abstract Syntax Tree解析）の仕組み

Promptingモードでは、モデルが出力した自然言語テキストからPythonの関数呼び出しを抽出するためにAST解析を使用します：

**1. テキスト前処理**
```python
# "[get_weather(location='東京')]" → "get_weather(location='東京')"
cleaned_input = input_str.strip("[]'")
```

**2. PythonのASTモジュールで構文解析**
```python
parsed = ast.parse(cleaned_input, mode="eval")
```

**3. 関数呼び出しと引数の抽出**
```python
# 最終出力: [{'get_weather': {'location': '東京'}}]
```

#### ⚡ 関数実行の仕組み

**重要**: APIモデル自体は関数を実行しません。実際の関数実行はBFCLシステム側で行われます。

**APIモデルの役割**: 
- 関数呼び出しの指示を生成するのみ
- 実際の処理は行わない

**BFCLシステムの役割**: 「実行エンジン」
- 実際のPythonクラスを動的にロード
- 関数を実際に実行（`eval()`使用）
- 実行結果をモデルに返却

```python
# 実際の関数実行プロセス
def execute_multi_turn_func_call():
    # 1. 実際のPythonクラスをロード
    class_instance = TradingBot()
    
    # 2. 関数実行
    result = eval("class_instance.place_order(symbol='AAPL', amount=100)")
    
    # 3. 結果をモデルに返却
    return result
```

### 質問2: bfcl/model_handler/api_inferenceで各モデルごとのファイルは何をやっている？

api_inferenceディレクトリには**20個以上のAPIプロバイダー専用ハンドラー**が含まれており、それぞれがBaseHandlerクラスを継承して特定のAPI仕様に対応した実装を提供しています。

#### 🔧 各ハンドラーの共通実装パターン

**各ハンドラーは以下を必ず実装:**
1. **APIクライアントの初期化**: 各サービス固有の認証とクライアント設定
2. **モデルスタイルの設定**: `ModelStyle`enum値の設定
3. **クエリメソッドの実装**: `_query_FC()`と`_query_prompting()`
4. **応答解析の実装**: API固有の応答形式からの標準形式への変換
5. **デコード機能**: `decode_ast()`と`decode_execute()`のオーバーライド
6. **エラーハンドリング**: API固有のエラー（レート制限等）への対応

#### 🏢 主要APIプロバイダーの特徴的な違い

**1. openai.py - OpenAIHandler**
```python
class OpenAIHandler(BaseHandler):
    def __init__(self, model_name, temperature):
        self.client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    
    def _query_FC(self, inference_data: dict):
        # シンプルで標準的
        return self.generate_with_backoff(
            messages=messages,
            model="gpt-4",
            tools=tools,
            temperature=0.7  # ただしo1モデルでは使用不可
        )
```
**特徴:**
- ✅ 最もシンプルな実装
- ✅ 標準的なFunction Calling形式
- ⚠️ o1/o3-miniモデルは温度パラメータ非対応

**2. claude.py - ClaudeHandler**
```python
class ClaudeHandler(BaseHandler):
    def _query_FC(self, inference_data: dict):
        # キャッシング機能付き
        if inference_data["caching_enabled"]:
            # 直近2つのユーザーメッセージをキャッシュ
            for message in reversed(messages):
                if message["role"] == "user":
                    message["content"][0]["cache_control"] = {"type": "ephemeral"}
        
        return self.generate_with_backoff(
            model="claude-3-sonnet",
            messages=messages_with_cache_control,
            tools=tools,
            max_tokens=8192  # モデルによって異なる
        )
```
**特徴:**
- 🚀 **キャッシング機能**: 直近2つのユーザーメッセージをキャッシュ
- 📏 **可変トークン制限**: Opusは4096、Sonnetは8192
- 🔄 **特殊なメッセージ処理**: cache_control フラグを動的に管理

**3. gemini.py - GeminiHandler**
```python
class GeminiHandler(BaseHandler):
    def _query_FC(self, inference_data: dict):
        # Google Cloud特有の複雑な変換
        func_declarations = []
        for function in inference_data["tools"]:
            func_declarations.append(
                FunctionDeclaration(
                    name=function["name"],
                    description=function["description"],
                    parameters=function["parameters"],
                )
            )
        
        tools = [Tool(function_declarations=func_declarations)]
        
        # システムプロンプトがある場合はクライアント再作成
        if "system_prompt" in inference_data:
            client = GenerativeModel(
                self.model_name,
                system_instruction=inference_data["system_prompt"]
            )
```
**特徴:**
- 🔧 **複雑な変換処理**: 関数をFunctionDeclaration→Toolオブジェクトに変換
- 🏗️ **動的クライアント生成**: システムプロンプトがある場合はモデル再インスタンス化
- 🌐 **Google Cloud統合**: Vertex AI経由でのアクセス

**4. その他の専用ハンドラー**
- **mistral.py**: Mistral AI API対応、独自のツール呼び出し形式
- **cohere.py**: Cohere API対応、独自のツール定義形式
- **yi.py**: Yi AI API対応
- **deepseek.py**: DeepSeek API対応
- **databricks.py**: Databricks API対応
- **nova.py**: Nova API対応
- **nexus.py**: Nexus API対応（セミコロン区切り形式）
- **gorilla.py**: Gorilla API対応
- **fireworks.py**: Fireworks AI API対応
- **nvidia.py**: NVIDIA API対応
- **writer.py**: Writer API対応
- **novita.py**: Novita API対応
- **qwq.py**: QwQ API対応
- **grok.py**: xAI Grok API対応

#### 📊 実装の複雑さ比較

| API | 実装複雑度 | 特殊機能 | 注意点 |
|-----|-------------|----------|--------|
| **OpenAI** | ⭐⭐ | o1モデル対応 | 最もシンプル |
| **Claude** | ⭐⭐⭐ | キャッシング | メッセージ形式が特殊 |
| **Gemini** | ⭐⭐⭐⭐ | 動的モデル生成 | Google Cloud設定必要 |
| **Cohere** | ⭐⭐⭐ | 独自ツール形式 | パラメータスキーマ変換 |
| **その他** | ⭐⭐ | 基本的な実装 | OpenAI互換が多い |

#### 🎨 Promptingモードでの特殊処理例

**Hermes（XMLタグベース）**
```python
def decode_ast(self, result):
    lines = result.split("\n")
    func_call = []
    for line in lines:
        if "<tool_call>" == line:
            flag = True
        elif "</tool_call>" == line:
            flag = False
        elif flag:
            tool_result = json.loads(line)
            func_call.append({tool_result["name"]: tool_result["arguments"]})
    return func_call
```

**MiningHandler（特殊パース）**
```python
def _parse_query_response_prompting(self, api_response):
    # <tool_calls>タグ内のJSONを抽出
    match = re.search(r'<tool_calls>\n(.*?)\n</tool_calls>', content, re.DOTALL)
    if match:
        tool_calls = match.group(1).strip()
        tool_calls = json.loads(tool_calls.replace("'",'"'))
    return {"model_responses": tool_calls, ...}
```