#!/bin/bash

# シンプルな評価実行スクリプト
# 使用方法: ./evaluate_simple.sh [config_name] [api_key_file]

set -e

# デフォルト値
DEFAULT_CONFIG="config-Meta-Llama-3-8B-Instruct.yaml"
DEFAULT_ENV_FILE=".env"

# 引数の処理
CONFIG_NAME=${1:-$DEFAULT_CONFIG}
ENV_FILE=${2:-$DEFAULT_ENV_FILE}

echo "=== LLM Leaderboard 評価実行スクリプト ==="
echo "設定ファイル: $CONFIG_NAME"
echo "環境変数ファイル: $ENV_FILE"
echo ""

# 環境変数ファイルの確認
if [ ! -f "$ENV_FILE" ]; then
    echo "警告: 環境変数ファイル '$ENV_FILE' が見つかりません"
    echo "env.exampleをコピーして設定してください:"
    echo "  cp env.example $ENV_FILE"
    echo "  そして、$ENV_FILEを編集してAPIキーを設定してください"
    echo ""
    echo "続行しますか？ (y/N)"
    read -r response
    if [[ ! "$response" =~ ^[Yy]$ ]]; then
        echo "実行を中止しました"
        exit 1
    fi
else
    echo "環境変数ファイルを読み込み中..."
    export $(grep -v '^#' "$ENV_FILE" | xargs)
fi

# 設定ファイルの確認
CONFIG_PATH="configs/$CONFIG_NAME"
if [ ! -f "$CONFIG_PATH" ]; then
    echo "エラー: 設定ファイル '$CONFIG_PATH' が見つかりません"
    echo "利用可能な設定ファイル:"
    ls -1 configs/*.yaml | sed 's|configs/||'
    exit 1
fi

# Dockerコンテナの確認
if ! docker ps | grep -q llm-leaderboard; then
    echo "エラー: llm-leaderboardコンテナが実行されていません"
    echo "まずコンテナを起動してください:"
    echo "  docker-compose up -d"
    exit 1
fi

echo "評価を開始します..."
echo ""

# 評価の実行
docker exec llm-leaderboard bash -c "cd /workspace && source .venv/bin/activate && python3 scripts/run_eval.py --config $CONFIG_NAME"

echo ""
echo "評価が完了しました！" 