#!/bin/bash
# generate_docker_override.sh - GPU構成に応じたdocker-compose override生成

set -e

if [ ! -f .env ]; then
    echo "エラー: .envファイルが見つかりません。"
    echo "quick_setup.shを実行して設定を作成してください。"
    exit 1
fi

# .envから設定を読み込み
source .env

echo "=== Docker Compose Override生成 ==="
echo ""

# GPU設定の解析
parse_gpu_ids() {
    local prefix="$1"
    local gpu_ids=()
    local i=0
    
    while true; do
        local var_name="${prefix}_GPU_${i}"
        local gpu_id="${!var_name}"
        if [ -z "$gpu_id" ]; then
            break
        fi
        gpu_ids+=("\"$gpu_id\"")
        ((i++))
    done
    
    printf '%s\n' "${gpu_ids[@]}"
}

# vLLM GPU設定
VLLM_DEVICE_IDS=($(parse_gpu_ids "VLLM"))
LEADERBOARD_DEVICE_IDS=($(parse_gpu_ids "LEADERBOARD"))

echo "vLLM GPUs: ${VLLM_DEVICE_IDS[*]}"
echo "Leaderboard GPUs: ${LEADERBOARD_DEVICE_IDS[*]}"
echo ""

# docker-compose.override.ymlの生成
cat > docker-compose.override.yml << EOL
services:
  vllm:
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              device_ids:
EOL

# vLLM GPU IDsを追加
for gpu_id in "${VLLM_DEVICE_IDS[@]}"; do
    echo "                - $gpu_id" >> docker-compose.override.yml
done

cat >> docker-compose.override.yml << EOL
              capabilities: [gpu]

  llm-leaderboard:
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              device_ids:
EOL

# Leaderboard GPU IDsを追加（存在する場合）
if [ ${#LEADERBOARD_DEVICE_IDS[@]} -gt 0 ]; then
    for gpu_id in "${LEADERBOARD_DEVICE_IDS[@]}"; do
        echo "                - $gpu_id" >> docker-compose.override.yml
    done
else
    echo "                - \"0\"  # デフォルトGPU" >> docker-compose.override.yml
fi

cat >> docker-compose.override.yml << EOL
              capabilities: [gpu]
EOL

echo "docker-compose.override.yml を生成しました。"
echo ""
echo "使用方法:"
echo "1. docker-compose up -d"
echo "   (自動的にoverride設定が適用されます)"
echo ""
echo "2. Leaderboard込みで起動:"
echo "   docker-compose --profile leaderboard up -d"
echo ""
echo "生成されたファイル: docker-compose.override.yml"