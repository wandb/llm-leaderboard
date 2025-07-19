#!/bin/bash
# run.sh - 統一された起動スクリプト

set -e

# .envファイルの読み込み
if [ -f .env ]; then
    source .env
else
    echo "エラー: .envファイルが見つかりません"
    echo "先に ./quick_setup.sh を実行してください"
    exit 1
fi

# 共通の関数
create_network() {
    echo "1. ネットワークの作成..."
    docker network create llm-stack-network 2>/dev/null || true
}

start_sandbox() {
    echo "2. サンドボックス環境の起動..."
    docker-compose up -d ssrf-proxy dify-sandbox
    
    echo "   サンドボックスの起動を確認中..."
    sleep 5
    docker-compose ps ssrf-proxy dify-sandbox
}

# API_TYPEに基づいて適切な起動方法を選択
case "$API_TYPE" in
    "vllm"|"vllm-docker")
        echo "=== vLLM Dockerモードで起動 ==="
        create_network
        start_sandbox
        
        echo "3. vLLMサービスの起動..."
        docker-compose --profile vllm-docker up -d vllm
        
        echo "4. vLLMの起動を待機中..."
        echo "   vLLMログを確認しています..."
        
        # vLLMの起動を待つ
        max_retries=30
        retry_count=0
        while [ $retry_count -lt $max_retries ]; do
            if docker-compose logs vllm 2>&1 | grep -q "Application startup complete"; then
                echo "   ✅ vLLMが正常に起動しました"
                break
            fi
            echo -n "."
            sleep 2
            retry_count=$((retry_count + 1))
        done
        
        if [ $retry_count -eq $max_retries ]; then
            echo ""
            echo "⚠️  警告: vLLMの起動確認がタイムアウトしました"
            echo "手動でログを確認してください: docker-compose logs vllm"
        fi
        
        echo ""
        echo "5. 評価の実行..."
        docker-compose --profile vllm-docker up llm-leaderboard
        ;;
        
    "openai-compatible"|"vllm-external")
        echo "=== OpenAI互換APIモードで起動 ==="
        echo "外部APIエンドポイント: ${API_BASE_URL:-設定なし}"
        echo ""
        
        # 外部APIの接続確認（オプション）
        if [ -n "$API_BASE_URL" ]; then
            echo "外部APIへの接続を確認中..."
            if curl -s --connect-timeout 5 "$API_BASE_URL/health" > /dev/null 2>&1 || \
               curl -s --connect-timeout 5 "$API_BASE_URL/v1/models" > /dev/null 2>&1; then
                echo "✅ 外部APIに接続できました"
            else
                echo "⚠️  警告: 外部APIへの接続を確認できませんでした"
                echo "APIが起動していることを確認してください"
                read -p "続行しますか？ (y/N): " confirm
                if [[ ! "$confirm" =~ ^[Yy]$ ]]; then
                    exit 1
                fi
            fi
        fi
        
        create_network
        start_sandbox
        
        echo "3. 評価の実行..."
        docker-compose up llm-leaderboard
        ;;
        
    "vllm-local")
        echo "=== vLLMローカルモード（非推奨）で起動 ==="
        echo "⚠️  警告: vllm-local は非推奨です"
        echo "vllm または vllm-docker への移行を推奨します"
        echo ""
        echo "続行しますか？ (y/N)"
        read -r confirm
        if [[ ! "$confirm" =~ ^[Yy]$ ]]; then
            echo "中止しました"
            exit 0
        fi
        
        create_network
        start_sandbox
        
        echo "3. 評価の実行（vLLMは内部で起動します）..."
        docker-compose up llm-leaderboard
        ;;
        
    *)
        echo "=== APIモードで起動 ==="
        echo "APIプロバイダー: ${API_PROVIDER:-不明}"
        
        create_network
        start_sandbox
        
        echo "3. 評価の実行..."
        docker-compose up llm-leaderboard
        ;;
esac

echo ""
echo "=== 実行完了 ==="
echo "結果はWandBで確認できます"
echo "" 