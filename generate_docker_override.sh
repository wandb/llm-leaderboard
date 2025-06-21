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

# docker-compose.override.yamlの生成
cat > docker-compose.override.yaml << EOL
services:
  vllm:
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
EOL

# vLLM GPU IDsを追加
if [ ${#VLLM_DEVICE_IDS[@]} -eq 0 ]; then
    echo "              device_ids: []" >> docker-compose.override.yaml
else
    echo "              device_ids:" >> docker-compose.override.yaml
    for gpu_id in "${VLLM_DEVICE_IDS[@]}"; do
        echo "                - $gpu_id" >> docker-compose.override.yaml
    done
fi

cat >> docker-compose.override.yaml << EOL
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
        echo "                - $gpu_id" >> docker-compose.override.yaml
    done
else
    echo "                - \"0\"  # デフォルトGPU" >> docker-compose.override.yaml
fi

cat >> docker-compose.override.yaml << EOL
              capabilities: [gpu]
EOL