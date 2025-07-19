# API Types 設定ガイド

## 概要

LLM Leaderboardでは、様々なLLMを評価するために複数のAPI typeをサポートしています。

## 推奨される設定

### 1. **vllm** （推奨・デフォルト）
ローカルモデルやHuggingFaceモデルを使用する場合の標準設定です。

- vLLMが別のDockerコンテナで起動されます
- 最もシンプルな指定方法
- GPU管理が明確で、リソース分離が可能

```yaml
# config例
api: "vllm"
# base_urlはデフォルトで "http://vllm:8000/v1" が使用されます
```

### 2. **vllm-docker**
`vllm`と同じ動作をします（エイリアス）。

```yaml
# config例
api: "vllm-docker"
base_url: "http://vllm:8000/v1"  # 明示的に指定も可能
```

起動方法（vllm、vllm-docker共通）：
```bash
# profileを指定してvllmコンテナも起動
docker-compose --profile vllm-docker up
```

### 3. **openai-compatible**
外部のOpenAI互換APIを使用する場合の設定です。

- vLLM、FastChat、llama.cpp server等のOpenAI互換サーバーに対応
- 既存の推論サーバーを活用可能

```yaml
# config例
api: "openai-compatible"
base_url: "https://your-api-endpoint.com/v1"
```

### 4. **商用API**
各社の商用APIを使用する場合：

- `openai`: OpenAI API
- `anthropic`: Anthropic Claude API
- `google`: Google Gemini API
- `mistral`: Mistral API
- `cohere`: Cohere API
- `upstage`: Upstage Solar API
- `xai`: xAI Grok API
- `azure-openai`: Azure OpenAI Service
- `amazon_bedrock`: Amazon Bedrock

## 非推奨の設定

以下のAPI typeは後方互換性のために維持されていますが、新規利用は推奨しません：

### **vllm-local** （非推奨）
- 評価コンテナ内でvLLMプロセスを起動
- リソース管理が複雑
- → **vllm**（推奨）への移行をお願いします

### **vllm-external** （非推奨）
- 名前が誤解を招きやすい（vLLM以外のサーバーでも使用）
- → **openai-compatible**への移行をお願いします

## セットアップと実行

### 1. 初期設定
```bash
./quick_setup.sh
```

新しいメニューから適切なモードを選択：
1. ローカルモデル（Docker vLLM）
2. HuggingFaceモデル（Docker vLLM）
3. 外部OpenAI互換API
4. 商用API

### 2. 実行
統一された起動スクリプトを使用：
```bash
./run.sh
```

スクリプトが自動的に適切な起動方法を選択します。

## 移行ガイド

### 既存の設定からの移行

既存の`api: "vllm"`設定は、そのまま動作します（Docker vLLMモードで起動）。

### vllm-local → vllm
```yaml
# 変更前
api: "vllm-local"

# 変更後
api: "vllm"
# base_urlは自動的に設定されます
```

### vllm-external → openai-compatible
```yaml
# 変更前
api: "vllm-external"
base_url: "http://your-server:8000/v1"

# 変更後
api: "openai-compatible"
base_url: "http://your-server:8000/v1"
```

## トラブルシューティング

### vllmモードでvLLMコンテナが起動しない
```bash
# vLLMのログを確認
docker-compose logs vllm

# profileが正しく指定されているか確認
docker-compose --profile vllm-docker ps
```

### 非推奨警告を無視したい場合
環境変数で警告を抑制できます：
```bash
export PYTHONWARNINGS="ignore::DeprecationWarning"
```

## 補足

- `vllm`と`vllm-docker`は同じ動作をします
- 新規設定では最もシンプルな`vllm`の使用を推奨します
- 既存の`vllm`設定は変更不要です（すでに推奨される動作をします） 