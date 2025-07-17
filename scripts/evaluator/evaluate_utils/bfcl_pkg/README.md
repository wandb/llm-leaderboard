# Berkeley Function Calling Leaderboard (BFCL)

> **Note**: This directory is imported from [Berkeley Function Call Leaderboard](https://github.com/ShishirPatil/gorilla/tree/main/berkeley-function-call-leaderboard) for Nejumi Leaderboard project.
> 
> Last updated by Keisuke Kamata on 2025/05/30
>
> To update this directory with the latest version:
> ```bash
> cd /path/to/project/root
> rm -rf scripts/evaluator/evaluate_utils/bfcl
> mkdir -p berkeley-function-call-leaderboard && cd berkeley-function-call-leaderboard
> git init && git remote add origin https://github.com/ShishirPatil/gorilla.git
> git config core.sparseCheckout true
> echo "berkeley-function-call-leaderboard/*" > .git/info/sparse-checkout
> git pull origin main
> cp -r berkeley-function-call-leaderboard/* ../scripts/evaluator/evaluate_utils/bfcl/
> cd .. && rm -rf berkeley-function-call-leaderboard
> ```

## Table of Contents

- [Berkeley Function Calling Leaderboard (BFCL)](#berkeley-function-calling-leaderboard-bfcl)
  - [Table of Contents](#table-of-contents)
  - [Introduction](#introduction)
  - [Installation \& Setup](#installation--setup)
    - [Basic Installation](#basic-installation)
    - [Extra Dependencies for Self-Hosted Models](#extra-dependencies-for-self-hosted-models)
    - [Setting up Environment Variables](#setting-up-environment-variables)
  - [Running Evaluations](#running-evaluations)
    - [Generating LLM Responses](#generating-llm-responses)
      - [Selecting Models and Test Categories](#selecting-models-and-test-categories)
      - [Output and Logging](#output-and-logging)
      - [For API-based Models](#for-api-based-models)
      - [For Locally-hosted OSS Models](#for-locally-hosted-oss-models)
        - [For Pre-existing OpenAI-compatible Endpoints](#for-pre-existing-openai-compatible-endpoints)
      - [(Alternate) Script Execution for Generation](#alternate-script-execution-for-generation)
    - [Evaluating Generated Responses](#evaluating-generated-responses)
      - [Output Structure](#output-structure)
      - [(Optional) WandB Evaluation Logging](#optional-wandb-evaluation-logging)
      - [(Alternate) Script Execution for Evaluation](#alternate-script-execution-for-evaluation)
  - [Contributing \& How to Add New Models](#contributing--how-to-add-new-models)
  - [Additional Resources](#additional-resources)

---

## Introduction

We introduce the Berkeley Function Calling Leaderboard (BFCL), the **first comprehensive and executable function call evaluation** dedicated to assessing Large Language Models' (LLMs) ability to invoke functions. Unlike previous evaluations, BFCL accounts for various forms of function calls, diverse scenarios, and executability.

💡 Read more in our blog posts:

- [BFCL v1 (original) Blog Post](https://gorilla.cs.berkeley.edu/blogs/8_berkeley_function_calling_leaderboard.html)
- [BFCL v2 (live dataset) Blog Post](https://gorilla.cs.berkeley.edu/blogs/12_bfcl_v2_live.html)
- [BFCL v3 (multi-turn) Blog Post](https://gorilla.cs.berkeley.edu/blogs/13_bfcl_v3_multi_turn.html)

🦍 See the live leaderboard at [Berkeley Function Calling Leaderboard](https://gorilla.cs.berkeley.edu/leaderboard.html#leaderboard)

![Architecture Diagram](./architecture_diagram.png)

---

## Installation & Setup

### Basic Installation

```bash
# Create a new Conda environment with Python 3.10
conda create -n BFCL python=3.10
conda activate BFCL

# Clone the Gorilla repository
git clone https://github.com/ShishirPatil/gorilla.git

# Change directory to the `berkeley-function-call-leaderboard`
cd gorilla/berkeley-function-call-leaderboard

# Install the package in editable mode
pip install -e .
```

### Extra Dependencies for Self-Hosted Models

For locally hosted models, choose one of the following backends, ensuring you have the right GPU and OS setup:

`sglang` is *much faster* than `vllm` but only supports newer GPUs with SM 80+ (Ampere etc).
If you are using an older GPU (T4/V100), you should use `vllm` instead as it supports a much wider range of GPUs.

**Using `vllm`:**
```bash
pip install -e .[oss_eval_vllm]
```

**Using `sglang`:**
```bash
pip install -e .[oss_eval_sglang]
```

*Optional:* If using `sglang`, we recommend installing `flashinfer` for speedups. Find instructions [here](https://docs.flashinfer.ai/installation.html).

### Setting up Environment Variables

We store environment variables in a `.env` file. We have provided a example `.env.example` file in the `gorilla/berkeley-function-call-leaderboard` directory. You should make a copy of this file, and fill in the necessary values.

```bash
cp .env.example .env
# Fill in necessary values in `.env`
```

If you are running any proprietary models, make sure the model API keys are included in your `.env` file. Models like GPT, Claude, Mistral, Gemini, Nova, will require them.

---

## Running Evaluations

### Generating LLM Responses

#### Selecting Models and Test Categories

- `MODEL_NAME`: For available models, please refer to [SUPPORTED_MODELS.md](./SUPPORTED_MODELS.md). If not specified, the default model `gorilla-openfunctions-v2` is used.
- `TEST_CATEGORY`: For available test categories, please refer to [TEST_CATEGORIES.md](./TEST_CATEGORIES.md). If not specified, all categories are included by default.

You can provide multiple models or test categories by separating them with commas. For example:

```bash
bfcl generate --model claude-3-5-sonnet-20241022-FC,gpt-4o-2024-11-20-FC --test-category simple,parallel,multiple,multi_turn
```

#### Output and Logging

- All generated model responses are stored in `./result/` folder, organized by model and test category: `result/MODEL_NAME/BFCL_v3_TEST_CATEGORY_result.json`
- To use a custom directory for the result file, specify using `--result-dir`; path should be relative to the `berkeley-function-call-leaderboard` root folder,

An inference log is included with the model responses to help analyze/debug the model's performance, and to better understand the model behavior. For more verbose logging, use the `--include-input-log` flag. Refer to [LOG_GUIDE.md](./LOG_GUIDE.md) for details on how to interpret the inference logs.

#### For API-based Models

```bash
bfcl generate --model MODEL_NAME --test-category TEST_CATEGORY --num-threads 1
```

- Use `--num-threads` to control the level of parallel inference. The default (`1`) means no parallelization.
- The maximum allowable threads depends on your API's rate limits.

#### For Locally-hosted OSS Models

```bash
bfcl generate \
  --model MODEL_NAME \
  --test-category TEST_CATEGORY \
  --backend {vllm|sglang} \
  --num-gpus 1 \
  --gpu-memory-utilization 0.9 \
  --local-model-path /path/to/local/model   # ← optional
```

- Choose your backend using `--backend vllm` or `--backend sglang`. The default backend is `vllm`.
- Control GPU usage by adjusting `--num-gpus` (default `1`, relevant for multi-GPU tensor parallelism) and `--gpu-memory-utilization` (default `0.9`), which can help avoid out-of-memory errors.
- `--local-model-path` (optional): Point this flag at a directory that already contains the model's files (`config.json`, tokenizer, weights, etc.). Use it only when you've pre‑downloaded the model and the weights live somewhere other than the default `$HF_HOME` cache.

##### For Pre-existing OpenAI-compatible Endpoints

If you have a server already running (e.g., vLLM in a SLURM cluster), you can bypass the vLLM/sglang setup phase and directly generate responses by using the `--skip-server-setup` flag:

```bash
bfcl generate --model MODEL_NAME --test-category TEST_CATEGORY --skip-server-setup
```

In addition, you should specify the endpoint and port used by the server. By default, the endpoint is `localhost` and the port is `1053`. These can be overridden by the `VLLM_ENDPOINT` and `VLLM_PORT` environment variables in the `.env` file:

```bash
VLLM_ENDPOINT=localhost
VLLM_PORT=1053
```

#### (Alternate) Script Execution for Generation

For those who prefer using script execution instead of the CLI, you can run the following command:

```bash
# Make sure you are inside the `berkeley-function-call-leaderboard` directory
python openfunctions_evaluation.py --model MODEL_NAME --test-category TEST_CATEGORY
```

When specifying multiple models or test categories, separate them with **spaces**, not commas. All other flags mentioned earlier are compatible with the script execution method as well.

### Evaluating Generated Responses

**Important:** You must have generated the model responses before running the evaluation.

Once you have the results, run:

```bash
bfcl evaluate --model MODEL_NAME --test-category TEST_CATEGORY
```

The `MODEL_NAME` and `TEST_CATEGORY` options are the same as those used in the [Generating LLM Responses](#generating-llm-responses) section. For details, refer to [SUPPORTED_MODELS.md](./SUPPORTED_MODELS.md) and [TEST_CATEGORIES.md](./TEST_CATEGORIES.md).

If in the previous step you stored the model responses in a custom directory, you should specify it using the `--result-dir` flag; path should be relative to the `berkeley-function-call-leaderboard` root folder.

> Note: For unevaluated test categories, they will be marked as `N/A` in the evaluation result csv files.
> For summary columns (e.g., `Overall Acc`, `Non_Live Overall Acc`, `Live Overall Acc`, and `Multi Turn Overall Acc`), the score reported will treat all unevaluated categories as 0 during calculation.

#### Output Structure

Evaluation scores are stored in `./score/`, mirroring the structure of `./result/`: `score/MODEL_NAME/BFCL_v3_TEST_CATEGORY_score.json`

- To use a custom directory for the score file, specify using `--score-dir`; path should be relative to the `berkeley-function-call-leaderboard` root folder.

Additionally, four CSV files are generated in `./score/`:

- `data_overall.csv` – Overall scores for each model. This is used for updating the leaderboard.
- `data_live.csv` – Detailed breakdown of scores for each Live (single-turn) test category.
- `data_non_live.csv` – Detailed breakdown of scores for each Non-Live (single-turn) test category.
- `data_multi_turn.csv` – Detailed breakdown of scores for each Multi-Turn test category.

#### (Optional) WandB Evaluation Logging

If you'd like to log evaluation results to WandB artifacts:

```bash
pip install -e.[wandb]
```

Mkae sure you also set `WANDB_BFCL_PROJECT=ENTITY:PROJECT` in `.env`.

#### (Alternate) Script Execution for Evaluation

For those who prefer using script execution instead of the CLI, you can run the following command:

```bash
# Make sure you are inside the `berkeley-function-call-leaderboard/bfcl/eval_checker` directory
cd bfcl/eval_checker
python eval_runner.py --model MODEL_NAME --test-category TEST_CATEGORY
```

When specifying multiple models or test categories, separate them with **spaces**, not commas. All other flags mentioned earlier are compatible with the script execution method as well.

## Contributing & How to Add New Models

We welcome contributions! To add a new model:

1. Review `bfcl/model_handler/base_handler.py` and/or `bfcl/model_handler/local_inference/base_oss_handler.py` (if your model is hosted locally).
2. Implement a new handler class for your model.
3. Update `bfcl/constants/model_config.py`.
4. Submit a Pull Request.

For detailed steps, please see the [Contributing Guide](./CONTRIBUTING.md).


## Additional Resources

- [Gorilla Discord](https://discord.gg/grXXvj9Whz) (`#leaderboard` channel)
- [Project Website](https://gorilla.cs.berkeley.edu/)

All the leaderboard statistics, and data used to train the models are released under Apache 2.0.
Gorilla is an open source effort from UC Berkeley and we welcome contributors.
Please email us your comments, criticisms, and questions. More information about the project can be found at [https://gorilla.cs.berkeley.edu/](https://gorilla.cs.berkeley.edu/)


## 補足 by Nejumi Leaderboard
### Nejumi Leaderboardのために行った変更
このセクションでは、BFCLをNejumi Leaderboardに統合するために行った具体的な変更について詳細に説明します。

#### 1 評価データセットの日本語化
- qwen/qwen3-235b-a22bを用いて翻訳
- llm-leaderboard/scripts/translation/bfcl_translation.pyを利用
- データセットはWandBのartifactsに保存 [link](https://wandb.ai/llm-leaderboard/nejumi-leaderboard4/artifacts/dataset/bfcl)
- **ルール**: 関数名、コード関連内容は翻訳対象外

#### 2 統合
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

#### 3 llm-leadrboardで起動されるvllmを利用するように変更
- llm-leaderboard/scripts/evaluator/evaluate_utils/bfcl_pkg/bfcl/model_handler/local_inference/base_oss_handler.pyのvllm_hostとportを変更

#### 4 ローカルモデルのchat templateへの対応
- オリジナルのBFCLでは、vllm起動時にchat templateを利用せず、推論実行時にモデルごとのclassでtemplateの対応を行なっていた。Nejumi leaderboardでは、vllm起動時にchat templateを利用するので、モデルごとのclass内でのchat templateを削除し、llm-leaderboard/scripts/evaluator/evaluate_utils/bfcl_pkg/bfcl/model_handler/local_inference/base_oss_handler.py内でOSSHandler内でChat Completion形式に対応できるようにした。これにより、モデルごとの設定項目が大幅に簡素化されました。
- 不要になるメソッド
  - **`_format_prompt`**: Chat Completions APIが入力フォーマットを統一するため不要。チャットテンプレートの二重適用問題も解決される
- 依然として必要なメソッド
  - **`decode_ast`/`decode_execute`**: 出力パースは模型固有のため必要
  - **`_pre_query_processing_prompting`**: 前処理は模型固有のため必要。詳細は以下で解説します。

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

**APIモデルの役割**: 「俳優」
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


### 質問3: bfcl/model_handler/local_inference/base_oss_handler.pyがやっていることを教えて

**base_oss_handler.py**は、**OSS（オープンソース）モデル、つまりローカルで実行されるモデル用の基盤クラス**です。BaseHandlerを継承し、ローカルモデル特有の処理を実装しています。

#### 🏗️ 主要な役割と機能

##### **1. Chat Completions API への対応（重要な変更点）**
**従来のBFCL**: 各モデルで個別にchat templateを処理
```python
# 旧実装（削除済み）
def _format_prompt(self, messages, function):
    # モデルごとに個別のchat template処理
    formatted_prompt = apply_chat_template(messages)
    return formatted_prompt
```

**現在のNejumi leaderboard**: vLLMサーバー側でchat templateを統一処理
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

##### **2. vLLMサーバーとの通信管理**
```python
class OSSHandler(BaseHandler):
    def __init__(self, model_name, temperature, dtype="bfloat16"):
        # vLLMサーバーへの接続設定
        self.vllm_host = os.getenv("VLLM_ENDPOINT", "localhost")
        self.vllm_port = os.getenv("VLLM_PORT", VLLM_PORT)
        self.base_url = f"http://{self.vllm_host}:{self.vllm_port}/v1"
        self.client = OpenAI(base_url=self.base_url, api_key="EMPTY")
```

##### **3. バッチ推論の実装**
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

##### **4. デフォルトのデコード処理**
```python
@override
def decode_ast(self, result, language="Python"):
    return default_decode_ast_prompting(result, language)

@override
def decode_execute(self, result):
    return default_decode_execute_prompting(result)
```

##### **5. トークン数の推定**
```python
# Chat Completions APIではメッセージからトークン数を推定
messages_text = " ".join([msg.get("content", "") for msg in message])
input_token_count = len(self.tokenizer.tokenize(messages_text))
```

#### ⚡ 処理フロー

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

### 質問4: bfcl/model_handler/local_inference内の追加のローカルモデルのクラスが何をしているかを教えて


local_inferenceディレクトリには**25個以上のローカルモデル専用ハンドラー**が含まれており、base_oss_handler.pyの**OSSHandler**を継承して、各モデル固有の処理を最小限の実装で提供しています。

#### **Nejumi Leaderboardのために削除されたメソッド**
- **`_format_prompt`**: Chat Completions APIがvLLMサーバー側で統一フォーマットを処理するため不要

#### **依然として必要なメソッド**
- **`decode_ast`/`decode_execute`**: 出力パースはモデル固有のため必要
- **`_pre_query_processing_prompting`**: 前処理はモデル固有のため必要
- **`_add_execution_results_prompting`**: 実行結果の処理方法がモデルによって異なる

#### 🎨 モデル別の出力フォーマットと対応が必要な理由と具体例

#### **1. シンプルなケース: hammer.py**
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

**期待される標準フォーマット:**
```json
[{"name": "function_name", "arguments": {"param": "value"}}]
```

#### **2. 特殊フォーマット対応: deepseek.py**
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

**DeepSeekの実際の出力例:**
```
```json
{"name": "calculate", "arguments": {"x": 5, "y": 10}}
```
```

#### **3. 複雑なフォーマット: llama_3_1.py**
```python
class Llama31Handler(OSSHandler):
    @override
    def decode_ast(self, result, language="Python"):
        # タグ除去、セミコロン区切り対応
        result = result.replace("<|python_tag|>", "").strip()
        calls = result.split(";")
        return [json.loads(call.strip()) for call in calls if call.strip()]
```

**Llama 3.1の実際の出力例:**
```
<|python_tag|>{"name": "calc", "arguments": {...}}; {"name": "func2", "arguments": {...}}
```

#### **4. 超複雑なフォーマット: minicpm_fc.py**
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

**MiniCPMの実際の出力例:**
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

### 🔄 実行結果の処理方法の違い

#### **標準的な処理（DeepSeek）**
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

#### **特殊なロール使用（Llama）**
```python
def _add_execution_results_prompting(self, inference_data, execution_results, model_response_data):
    for execution_result in execution_results:
        # Llamaは特殊な`ipython`ロールを使用
        inference_data["message"].append({
            "role": "ipython",
            "content": execution_result,
        })
```

### 📊 モデル別特徴まとめ

| モデル | 出力の特徴 | 主な処理 |
|--------|------------|----------|
| **Hammer** | 標準JSON | 最もシンプル |
| **DeepSeek** | ```json\n...\n``` | プレフィックス除去 |
| **Llama 3.1** | <python_tag>...;... | タグ除去+セミコロン分割 |
| **MiniCPM** | 思考過程+ツールタグ | 複雑なタグ解析 |
| **Phi** | ```json/python... | 複数プレフィックス対応 |
| **GLM** | 改行区切り | 特殊な改行処理 |
| **Granite** | <function_call>... | XMLライクタグ |

#### 💡 出力フォーマットが異なる理由

**1. 学習データの違い**
- 各モデルが異なるデータセットで訓練されているため

**2. チャットテンプレートの違い**
- モデル固有のフォーマット規則があるため

**3. 設計思想の違い**
- 出力の詳細さや構造に対する考え方が異なるため


