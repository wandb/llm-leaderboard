# アプローチ2: YAMLベース設定管理

このアプローチでは、YAMLファイルで基本設定を管理し、APIキーは環境変数で管理するシンプルな構成を採用しています。

## 特徴

- **YAML設定**: モデル設定や評価パラメータはYAMLファイルで管理
- **環境変数**: APIキーは`.env`ファイルで管理（セキュリティ向上）
- **シンプルな実行**: 1つのスクリプトで評価を実行
- **柔軟性**: 既存のシステムを壊さずに拡張

## セットアップ

### 1. 環境変数ファイルの作成

```bash
# テンプレートファイルをコピー
cp env.example .env

# .envファイルを編集してAPIキーを設定
nano .env
```

`.env`ファイルの例:
```bash
# Weights & Biases API Key
WANDB_API_KEY=your_actual_wandb_api_key_here

# vLLM Model Name
VLLM_MODEL_NAME=meta-llama/Meta-Llama-3-8B-Instruct

# Evaluation Config Path
EVAL_CONFIG_PATH=config-Meta-Llama-3-8B-Instruct.yaml
```

### 2. 設定ファイルの確認

`configs/config-Meta-Llama-3-8B-Instruct.yaml`が正しく設定されていることを確認:

```yaml
wandb:
  run_name: meta-llama/Meta-Llama-3-8B-Instruct
  api_key: ${WANDB_API_KEY}  # 環境変数から読み込み
api: vllm
# ... その他の設定
```

## 使用方法

### 基本的な実行

```bash
# デフォルト設定で実行
./evaluate_simple.sh

# 特定の設定ファイルを指定
./evaluate_simple.sh config-Meta-Llama-3-8B-Instruct.yaml

# カスタム環境変数ファイルを指定
./evaluate_simple.sh config-Meta-Llama-3-8B-Instruct.yaml my_env.env
```

### 手動実行（詳細制御）

```bash
# 環境変数を設定して実行
export $(grep -v '^#' .env | xargs)
docker exec llm-leaderboard bash -c "cd /workspace && source .venv/bin/activate && python3 scripts/run_eval.py --config config-Meta-Llama-3-8B-Instruct.yaml"
```

## トラブルシューティング

### APIキーエラー

```
wandb.errors.errors.UsageError: api_key not configured (no-tty)
```

**解決方法:**
1. `.env`ファイルに`WANDB_API_KEY`が正しく設定されているか確認
2. 環境変数が読み込まれているか確認: `echo $WANDB_API_KEY`

### コンテナが見つからない

```
Error response from daemon: container is not running
```

**解決方法:**
```bash
# コンテナを起動
docker-compose up -d

# コンテナの状態を確認
docker ps
```

### 設定ファイルが見つからない

```
Config file does not exist
```

**解決方法:**
1. `configs/`ディレクトリに設定ファイルが存在するか確認
2. ファイル名のスペルを確認

## 設定ファイルの構造

### YAML設定ファイル

```yaml
wandb:
  run_name: meta-llama/Meta-Llama-3-8B-Instruct
  api_key: ${WANDB_API_KEY}  # 環境変数から読み込み

api: vllm  # 使用するAPIタイプ
batch_size: 256

model:
  pretrained_model_name_or_path: meta-llama/Meta-Llama-3-8B-Instruct
  # ... その他のモデル設定

# APIキー設定（環境変数から読み込み）
api_keys:
  wandb: ${WANDB_API_KEY}
  openai: ${OPENAI_API_KEY}
  anthropic: ${ANTHROPIC_API_KEY}
```

### 環境変数ファイル

```bash
# 必須
WANDB_API_KEY=your_wandb_api_key

# オプション（使用するAPIに応じて）
OPENAI_API_KEY=your_openai_api_key
ANTHROPIC_API_KEY=your_anthropic_api_key
GOOGLE_API_KEY=your_google_api_key
AZURE_API_KEY=your_azure_api_key

# vLLM設定
VLLM_MODEL_NAME=meta-llama/Meta-Llama-3-8B-Instruct
EVAL_CONFIG_PATH=config-Meta-Llama-3-8B-Instruct.yaml
```

## 利点

1. **セキュリティ**: APIキーがコードにハードコーディングされない
2. **保守性**: 設定がYAMLファイルで一元管理される
3. **柔軟性**: 環境変数で動的に設定を変更可能
4. **シンプル**: 1つのスクリプトで評価を実行
5. **後方互換性**: 既存のシステムを壊さない

## 注意事項

- `.env`ファイルはGitにコミットしないでください
- APIキーは適切に管理し、漏洩しないよう注意してください
- 環境変数ファイルの権限を適切に設定してください（例: `chmod 600 .env`） 