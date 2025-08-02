
# Nejumi Leaderboardで行ったBFCLの変更と補足

## Nejumi Leaderboardのために行った変更
このセクションでは、BFCLをNejumi Leaderboardに統合するために行った具体的な変更について詳細に説明します。

- 評価データセットの日本語化とサンプリング
    - qwen/qwen3-235b-a22bを用いてベース翻訳。人手で修正も実施
    - llm-leaderboard/scripts/translation/bfcl_translation.pyを利用
        - **ルール**: 関数名、コード関連内容は翻訳対象外
    - llm-leaderboard/scripts/translation/bfcl_multi_turn_count.pyを用いて、Turn数を計算
    - llm-leaderboard/scripts/translation/sort_bfcl_file.pyを用いて並び替え
    - llm-leaderboard/scripts/data_uploader/upload_dataset.pyを用いてW&Bにupload
    - データセットはWandBのartifactsに保存 [link](https://wandb.ai/llm-leaderboard/nejumi-leaderboard4/artifacts/dataset/bfcl)
    - Nejumi Leaderboardではサンプリングをして実装
        - 基本的に各カテゴリ30問を利用。30問に満たない問題は全問
        - live_parallel_multiple, live_multiple: 問題文に英語以外の質問が含む以下の問題を削除
            - live_parallel_multiple
                - live_parallel_multiple_1-1-0
                - live_parallel_multiple_2-2-0
                - live_parallel_multiple_3-2-1
            - parallel_multiple
                - live_multiple_2-1-0
                - live_multiple_4-2-1
                - live_multiple_6-3-1
                - live_multiple_7-3-2
                - live_multiple_10-4-2
                - live_multiple_14-4-6
                - live_multiple_16-4-8
                - live_multiple_19-4-11
                - live_multiple_20-4-12
                - live_multiple_21-4-13
                - live_multiple_22-4-14
        - 上記artifactsに保存するにあたり人手での翻訳確認の品質担保のため、以下の問題は50問に絞って保存
            - live_multiple, multiple, simple, parallel_multiple
        - possible answerに日本語のオプションを追加
            - live_multiple, live_parallel, multiple, simple, parallel_multiple
        - 指示文の言語指定をするべきと判断した問題に、英語で回答してという指示を追加
            - live_parallel, parallel_multiple
- `scripts/run_eval.py`にBFCL評価を統合
- BFCL依存関係に伴うuv.lockの更新とuvベースの依存関係管理への移行
- `scripts/evaluator/bfcl.py`の作成
  - WandBConfigSingletonとの統合
  - 設定の動的マージ（デフォルト + ユーザー設定）
  - テストモード対応（サンプル数制限）
  - WandB Artifactからのデータセット取得
  - 評価結果のWandBテーブル生成
- base_configへの設定パラメータの追加:
- bfclをpackageとしてdownloadしないように変更。bfcl_pkg内の絶対インポートを相対インポートに変換
- llm-leaderboard/scripts/evaluator/evaluate_utils/bfcl_pkg/bfcl/constants/eval_config.py内のpathを変更
- llm-leaderboard/scripts/evaluator/evaluate_utils/bfcl_pkg/bfcl/eval_checker/multi_turn_eval/func_source_code内のlong_context.pyを実行時にpathの問題で利用できないファイルがあったので、該当ファイルにlong_context.py内のプロンプトを追加
- W&Bへの結果表示
  - W&BのTableに詳細な結果を残すために、出力されるscore fileにより詳細な情報が追加されるように変更(成功・失敗両方のテストケースで詳細情報を包含)
- モデルごとのconfig fileにBFCLのmodel idを追加
- データの整合性: 問題ディレクトリとpossible_answerディレクトリの両方で同じ順番が保たれる(sortをfalseにするなど)
- クラス名ベースの比較への変更
    - 問題：type()比較が異なるモジュールオブジェクトで失敗(packageの方法を踏襲しなかったので問題になった)
    - 修正：__class__.__name__による比較に変更
    - 対象ファイル：multi_turn_checker.pyと各APIクラスファイル
- Leading Zerosエラーの修正
    - 問題：Python 3での8進数解釈によるTypeError
    - 修正：正規表現によるleading zerosの10進数変換
    - 対象ファイル：multi_turn_utils.py
- llm-leadrboardで起動されるvllmを利用するように変更
    - llm-leaderboard/scripts/evaluator/evaluate_utils/bfcl_pkg/bfcl/model_handler/local_inference/base_oss_handler.pyのvllm_hostとportを変更
- ローカルモデルのchat templateへの対応
    - オリジナルのBFCLでは、vllm起動時にchat templateを利用せず、推論実行時にモデルごとのclassでtemplateの対応を行なっていた。Nejumi leaderboardでは、vllm起動時にchat templateを利用するので、モデルごとのclass内でのchat templateを削除し、llm-leaderboard/scripts/evaluator/evaluate_utils/bfcl_pkg/bfcl/model_handler/local_inference/base_oss_handler.py内でOSSHandler内でChat Completion形式に対応できるようにした。これにより、モデルごとの設定項目が大幅に簡素化されました。
    - 不要になるメソッド
    - **`_format_prompt`**: Chat Completions APIが入力フォーマットを統一するため不要。チャットテンプレートの二重適用問題も解決される
    - 依然として必要なメソッド
    - **`decode_ast`/`decode_execute`**: 出力パースは模型固有のため必要
    - **`_pre_query_processing_prompting`**: 前処理は模型固有のため必要。詳細は以下で解説します。

## 新しくモデルを追加する方法
- 公式の[Contributing Guide](./CONTRIBUTING.md)をご確認ください。以下、日本語でわかりやすく解説 & Nejumi Leaderboardに特化した対応について解説をします。


## 構造解説（もう少しorganizeする必要あり）
### 実装詳細
- bfcl.pyが実装されると`_llm_response_generation.py`の中の`generation_main`が実装され、そこから`gerate_results`が呼び出される
- `build_handler`でhandlerがinstance化される
    - handlerはbfcl_model_idに紐付き、`bfcl/constants/model_config.py`でmappingされてinstance化される
    - base_handlerのclassが全てのベース
    - 例: Qwenの場合、Qwen/XX-FCという名前をbfcl_model_idで設定すると、`bfcl/model_handler/local_inference/qwen_fc.py`の`QwenFCHandler`が呼ばれる
    - `QwenFCHandler`は`OSSHandler`を継承し、`OSSHandler`は`BaseHandler`を継承

### OSSを実装する場合 (vllmでの実装, qwen, deepseekのようにベンダーのAPIを利用する場合はこちらではない)
- `OSSHandler`では、以下が実装
    - llm = instance.llm
    - Function callingを利用する場合、UnifiedOSSFCHandlerを利用
    - Function callingを利用しない場合、UnifiedOSSFCHandlerを利用(完璧とは言わないが、chat templateからできるだけ対応)

### APIを実装する場合 (ベンダーのAPI実装)
- APIを使っていく場合は、`BaseHandler`を踏襲したベンダーごとのclassが存在

### 疑問に思うポイント（もう少しorganizeする必要あり）
#### bfcl/model_handler/base_handler.py は何をやっている？
BaseHandlerクラスは、BFCL（Berkeley Function-calling Leaderboard）における言語モデルの評価を行うための基盤となる抽象クラスです。

- 主要な役割と機能
    1. モデル推論の統一インターフェース
    - 異なるAPIプロバイダー（OpenAI、Claude、Geminiなど）に対して共通のインターフェースを提供
    - `inference()`メソッドが推論のエントリーポイントとして機能
    - Function Calling（FC）モードとPromptingモードの両方をサポート

    2. シングルターンとマルチターンの対話処理
    - `inference_single_turn_FC/prompting()`: 単発の質問応答処理
    - `inference_multi_turn_FC/prompting()`: 複数回の対話を行う処理
    - マルチターンでは関数の実行結果を次のターンに引き継ぎ、連続的な対話が可能

    3. 関数呼び出し（Function Calling）の実行管理
    - テストエントリから関数定義を取得し、モデルが適切な関数を呼び出せるよう管理
    - 関数の実行結果を取得し、次のクエリに反映
    - `MAXIMUM_STEP_LIMIT`による無限ループ防止機能

    4. トークン数とレイテンシの計測
    - 入力・出力トークン数の正確な計測
    - API呼び出しの応答時間測定
    - 評価指標として重要なメタデータの収集

    5. 状態管理とログ記録
    - クラスインスタンスの状態変化を追跡
    - 詳細な推論ログの記録（デバッグ用）
    - 実行結果のJSON形式での永続化

    6. エラーハンドリング
    - モデル応答のデコード失敗時の適切な処理
    - ステップ数上限による強制終了機能
    - 実行時エラーの捕捉とログ記録

- アーキテクチャ設計
    - BaseHandlerクラスはテンプレートメソッドパターンを採用しており、以下のメソッドが抽象メソッドとして定義され、各APIプロバイダーでの具体的な実装が必要です：
    - Function Callingモード用:
        - `_query_FC()`: APIへの実際のクエリ実行
        - `_pre_query_processing_FC()`: クエリ前の前処理
        - `_compile_tools()`: 関数定義のコンパイル
        - `_parse_query_response_FC()`: API応答の解析
        - `add_first_turn_message_FC()`: 初回メッセージの追加
        - `_add_assistant_message_FC()`: アシスタント応答の追加
        - `_add_execution_results_FC()`: 実行結果の追加
    - Promptingモード用:
        - `_query_prompting()`: プロンプトベースのクエリ実行
        - `_pre_query_processing_prompting()`: プロンプト前処理
        - `_parse_query_response_prompting()`: プロンプト応答の解析
        - 対応するメッセージ追加メソッド群

    -💡 FCモード vs Promptingモードの違い

    | 項目 | FCモード | Promptingモード |
    |------|----------|----------------|
    | **出力形式** | 構造化されたJSON | 自然言語+関数呼び出し |
    | **精度** | 高い（構造が保証） | 中程度（解析が必要） |
    | **対応モデル** | OpenAI、Claude等の新しいモデル | より幅広いモデル |
    | **実装の複雑さ** | シンプル | 複雑（テキスト解析が必要） |

    FCモードの例:
    ```python
    # モデル出力（構造化）
    {"tool_calls": [{"function": {"name": "get_weather", "arguments": "{\"location\": \"東京\"}"}}]}
    ```

    Promptingモードの例:
    ```python
    # モデル出力（自然言語）
    "[get_weather(location='東京')]"
    # ↓ AST解析が必要
    [{'get_weather': {'location': '東京'}}]
    ```

- AST解析（Abstract Syntax Tree解析）の仕組み
    - Promptingモードでは、モデルが出力した自然言語テキストからPythonの関数呼び出しを抽出するためにAST解析を使用します：

    1. テキスト前処理
    ```python
    # "[get_weather(location='東京')]" → "get_weather(location='東京')"
    cleaned_input = input_str.strip("[]'")
    ```

    2. PythonのASTモジュールで構文解析
    ```python
    parsed = ast.parse(cleaned_input, mode="eval")
    ```

    3. 関数呼び出しと引数の抽出
    ```python
    # 最終出力: [{'get_weather': {'location': '東京'}}]
    ```

- 関数実行の仕組み
    - **重要**: APIモデル自体は関数を実行しません。実際の関数実行はBFCLシステム側で行われます。
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

#### 2: bfcl/model_handler/api_inferenceで各モデルごとのファイルは何をやっている？

- api_inferenceディレクトリには**20個以上のAPIプロバイダー専用ハンドラー**が含まれており、それぞれがBaseHandlerクラスを継承して特定のAPI仕様に対応した実装を提供しています。

    **各ハンドラーは以下を実装:**
    1. **APIクライアントの初期化**: 各サービス固有の認証とクライアント設定
    2. **モデルスタイルの設定**: `ModelStyle`enum値の設定
    3. **クエリメソッドの実装**: `_query_FC()`と`_query_prompting()`
    4. **応答解析の実装**: API固有の応答形式からの標準形式への変換
    5. **デコード機能**: `decode_ast()`と`decode_execute()`のオーバーライド
    6. **エラーハンドリング**: API固有のエラー（レート制限等）への対応


### 3: bfcl/model_handler/local_inference/base_oss_handler.pyがやっていることを教えて
**base_oss_handler.py**は、**OSS（オープンソース）モデル、つまりローカルで実行されるモデル用の基盤クラス**です。BaseHandlerを継承し、ローカルモデル特有の処理を実装しています。

- 🏗️ 主要な役割と機能
    1. Chat Completions API への対応（重要な変更点）

        従来のBFCL: 各モデルで個別にchat templateを処理
        ```python
        # 旧実装（削除済み）
        def _format_prompt(self, messages, function):
            # モデルごとに個別のchat template処理
            formatted_prompt = apply_chat_template(messages)
            return formatted_prompt
        ```

        現在のNejumi leaderboard: vLLMサーバー側でchat templateを統一処理
        ```python
        # 新実装
        def _query_prompting(self, inference_data: dict):
            # Chat Completions APIではvLLMサーバー側でchat templateが適用されるため、
            # _format_promptは使用せず、直接messagesを送信する
            api_response = self.client.chat.completions.create(
                model=self.model_path_or_id,
                temperature=self.temperature,
                messages=message,  # 直接メッセージを送信
                max_tokens=leftover_tokens_count,
            )
        ```

    2. vLLMサーバーとの通信管理
    ```python
    class OSSHandler(BaseHandler):
        def __init__(self, model_name, temperature, dtype="bfloat16"):
            # vLLMサーバーへの接続設定
            self.vllm_host = os.getenv("VLLM_ENDPOINT", "localhost")
            self.vllm_port = os.getenv("VLLM_PORT", VLLM_PORT)
            self.base_url = f"http://{self.vllm_host}:{self.vllm_port}/v1"
            self.client = OpenAI(base_url=self.base_url, api_key="EMPTY")
    ```

    3. バッチ推論の実装
    APIモデルと異なり、ローカルモデルは**サーバーを起動してからバッチで処理**することで効率化：

    ```python
    def batch_inference(self, test_entries, num_gpus, gpu_memory_utilization, ...):
        # 1. モデルとトークナイザーのロード
        self.tokenizer = AutoTokenizer.from_pretrained(**load_kwargs)
        config = AutoConfig.from_pretrained(**load_kwargs)
        
        # 2. コンテキスト長の設定
        if hasattr(config, "max_position_embeddings"):
            self.max_context_length = config.max_position_embeddings
        
        # 3. バッチ処理の実行
        # (個別のエントリーを一度にまとめて処理)
    ```

    4. デフォルトのデコード処理
    ```python
    @override
    def decode_ast(self, result, language="Python"):
        return default_decode_ast_prompting(result, language)

    @override
    def decode_execute(self, result):
        return default_decode_execute_prompting(result)
    ```

    5. トークン数の推定
    ```python
    # Chat Completions APIではメッセージからトークン数を推定
    messages_text = " ".join([msg.get("content", "") for msg in message])
    input_token_count = len(self.tokenizer.tokenize(messages_text))
    ```

- 処理フロー

    ```
    1. バッチ推論開始
    ↓
    2. モデル・トークナイザーのロード (vLLMサーバーがすでに起動されている場合はスキップ)
    ↓
    3. vLLMサーバーとの接続確立
    ↓
    4. テストエントリーの前処理
    ↓
    5. Chat Completions API経由でクエリ
    ↓
    6. 応答の解析・デコード
    ↓
    7. 結果の保存
    ```

### 4: bfcl/model_handler/local_inference内の追加のローカルモデルのクラスが何をしているかを教えて

local_inferenceディレクトリには**25個以上のローカルモデル専用ハンドラー**が含まれており、base_oss_handler.pyの**OSSHandler**を継承して、各モデル固有の処理を最小限の実装で提供しています。

- Nejumi Leaderboardのために削除されたメソッド
    - **`_format_prompt`**: Chat Completions APIがvLLMサーバー側で統一フォーマットを処理するため不要

- 依然として必要なメソッド
    - **`decode_ast`/`decode_execute`**: 出力パースはモデル固有のため必要
    - **`_pre_query_processing_prompting`**: 前処理はモデル固有のため必要
    - **`_add_execution_results_prompting`**: 実行結果の処理方法がモデルによって異なる

- モデル別の出力フォーマットと対応が必要な理由と具体例
    - モデル別特徴まとめ
        | モデル | 出力の特徴 | 主な処理 |
        |--------|------------|----------|
        | **Hammer** | 標準JSON | 最もシンプル |
        | **DeepSeek** | ```json\n...\n``` | プレフィックス除去 |
        | **Llama 3.1** | <python_tag>...;... | タグ除去+セミコロン分割 |
        | **MiniCPM** | 思考過程+ツールタグ | 複雑なタグ解析 |
        | **Phi** | ```json/python... | 複数プレフィックス対応 |
        | **GLM** | 改行区切り | 特殊な改行処理 |
        | **Granite** | <function_call>... | XMLライクタグ |

    - 出力フォーマットが異なる理由
        1. 学習データの違い
        - 各モデルが異なるデータセットで訓練されているため

        2. チャットテンプレートの違い
        - モデル固有のフォーマット規則があるため

        3. 設計思想の違い
        - 出力の詳細さや構造に対する考え方が異なるため

    1. シンプルなケース: hammer.py
        ```python
        class HammerHandler(OSSHandler):
            @override
            def decode_ast(self, result, language="Python"):
                # 単純なクリーンアップ + 直接JSONパース
                result = result.replace("```", "")
                try:
                    result = json.loads(result)
                except:
                    result = []
                
                decoded_output = []
                for invoked_function in result:
                    name = invoked_function["name"]
                    params = invoked_function["arguments"]
                    decoded_output.append({name: params})
                return decoded_output
        ```

        期待される標準フォーマット:
        ```json
        [{"name": "function_name", "arguments": {"param": "value"}}]
        ```

    2. 特殊フォーマット対応: deepseek.py
        ```python
        class DeepseekHandler(OSSHandler):
            @override
            def decode_ast(self, result, language="Python"):
                result = result.strip()
                # ```json プレフィックスを除去
                if result.startswith("```json"):
                    result = result[len("```json"):]
                if result.startswith("```python"):
                    result = result[len("```python"):]
                return super().decode_ast(result, language)
        ```

        DeepSeekの実際の出力例:
        ```
            ```json
            {"name": "calculate", "arguments": {"x": 5, "y": 10}}
            ```
        ```

    3. 複雑なフォーマット: llama_3_1.py
        ```python
        class Llama31Handler(OSSHandler):
            @override
            def decode_ast(self, result, language="Python"):
                # タグ除去、セミコロン区切り対応
                result = result.replace("<|python_tag|>", "").strip()
                calls = result.split(";")
                return [json.loads(call.strip()) for call in calls if call.strip()]
        ```

        Llama 3.1の実際の出力例:
        ```
        <|python_tag|>{"name": "calc", "arguments": {...}}; {"name": "func2", "arguments": {...}}
        ```

    4. 超複雑なフォーマット: minicpm_fc.py
        ```python
        def fc2dict(sequence: str, 
                tool_call_start="<|tool_call_start|>",
                tool_call_end="<|tool_call_end|>",
                thought_start="<|thought_start|>",
                thought_end="<|thought_end|>"):
            # 思考過程とツールコールタグを含む複雑なフォーマット
            if thought_end in sequence and thought_start in sequence:
                thought_string, sequence = sequence.rsplit(thought_end, 1)
                thought_string = thought_string.split(thought_start, 1)[1]
            
            if tool_call_start in sequence and tool_call_end in sequence:
                tool_call_string, content = sequence.rsplit(tool_call_end, 1)
                tool_call_string = tool_call_string.split(tool_call_start, 1)[1]
                # AST解析で関数呼び出しを抽出
                parsed = ast.parse(tool_call_string)
                # ...
        ```

        MiniCPMの実際の出力例:
        ```
        <|thought_start|>
        ユーザーは計算を求めているので、calculate関数を使います
        <|thought_end|>
        <|tool_call_start|>
            ```python
            calculate(x=5, y=10)
            ```
        <|tool_call_end|>
        計算結果をお見せします
        ```

    5. 実行結果の処理方法の違い
        - 標準的な処理（DeepSeek）
            ```python
            def _add_execution_results_prompting(self, inference_data, execution_results, model_response_data):
                # DeepSeekはtoolロールを受け付けないため、userロールを使用
                tool_message = {"role": "user", "content": []}
                for execution_result, decoded_model_response in zip(execution_results, model_response_data["model_responses_decoded"]):
                    tool_message["content"].append({
                        "role": "tool",
                        "name": decoded_model_response,
                        "content": execution_result,
                    })
                inference_data["message"].append(tool_message)
            ```

        - 特殊なロール使用（Llama）
            ```python
            def _add_execution_results_prompting(self, inference_data, execution_results, model_response_data):
                for execution_result in execution_results:
                    # Llamaは特殊な`ipython`ロールを使用
                    inference_data["message"].append({
                        "role": "ipython",
                        "content": execution_result,
                    })
            ```
