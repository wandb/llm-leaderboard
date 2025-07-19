#!/bin/bash
set -e

# 使い方の表示
show_usage() {
    echo "使い方: $0 <モデル名またはconfigファイル名>"
    echo ""
    echo "例:"
    echo "  $0 Meta-Llama-3-8B-Instruct"
    echo "  $0 config-Meta-Llama-3-8B-Instruct.yaml"
    echo "  $0 Swallow-7b"  # 部分一致も可能
    echo ""
    echo "オプション:"
    echo "  -h, --help     このヘルプを表示"
    echo "  -l, --list     利用可能なモデルを一覧表示"
    echo "  -o, --offline  Wandbをオフラインモードで実行"
    echo "  -d, --debug    デバッグモードで実行"
}

# 利用可能なモデルを表示
list_models() {
    echo "利用可能なモデル:"
    echo ""
    for config in configs/config-*.yaml; do
        if [ -f "$config" ]; then
            # YAMLからモデル名を抽出（改善版）
            model_name=$(grep "pretrained_model_name_or_path:" "$config" | head -1 | sed 's/.*pretrained_model_name_or_path: *//' | sed 's/ *#.*//' | tr -d '"' | tr -d "'")
            config_name=$(basename "$config" .yaml | sed 's/config-//')
            printf "  %-40s → %s\n" "$config_name" "$model_name"
        fi
    done
}

# 設定ファイルを検索
find_config() {
    local query=$1
    
    # 完全一致を試す
    if [ -f "configs/config-${query}.yaml" ]; then
        echo "configs/config-${query}.yaml"
        return 0
    fi
    
    # .yaml付きで渡された場合
    if [ -f "configs/${query}" ]; then
        echo "configs/${query}"
        return 0
    fi
    
    # 部分一致を試す
    local matches=(configs/config-*${query}*.yaml)
    if [ -f "${matches[0]}" ]; then
        if [ ${#matches[@]} -eq 1 ]; then
            echo "${matches[0]}"
            return 0
        else
            echo "エラー: 複数の設定ファイルが見つかりました:" >&2
            for match in "${matches[@]}"; do
                echo "  - $match" >&2
            done
            return 1
        fi
    fi
    
    echo "エラー: 設定ファイルが見つかりません: $query" >&2
    return 1
}

# メイン処理
main() {
    local OFFLINE_MODE=false
    local DEBUG_MODE=false
    
    # オプション解析
    while [[ $# -gt 0 ]]; do
        case $1 in
            -h|--help)
                show_usage
                exit 0
                ;;
            -l|--list)
                list_models
                exit 0
                ;;
            -o|--offline)
                OFFLINE_MODE=true
                shift
                ;;
            -d|--debug)
                DEBUG_MODE=true
                shift
                ;;
            *)
                MODEL_QUERY=$1
                shift
                ;;
        esac
    done
    
    # モデルが指定されていない場合
    if [ -z "$MODEL_QUERY" ]; then
        show_usage
        exit 1
    fi
    
    # 設定ファイルを検索
    CONFIG_FILE=$(find_config "$MODEL_QUERY")
    if [ $? -ne 0 ]; then
        exit 1
    fi
    
    echo "📋 設定ファイル: $CONFIG_FILE"
    
    # YAMLからモデル名を抽出
    MODEL_NAME=$(grep "pretrained_model_name_or_path:" "$CONFIG_FILE" | head -1 | sed 's/.*pretrained_model_name_or_path: *//' | sed 's/ *#.*//' | tr -d '"' | tr -d "'" | xargs)
    echo "🤖 モデル: $MODEL_NAME"
    
    # 環境変数の設定
    export EVAL_CONFIG_PATH=$(basename "$CONFIG_FILE")
    export VLLM_MODEL_NAME="$MODEL_NAME"
    
    if [ "$OFFLINE_MODE" = true ]; then
        export WANDB_MODE=offline
        echo "📊 Wandb: オフラインモード"
    else
        export WANDB_MODE=online
    fi
    
    # デバッグ情報
    if [ "$DEBUG_MODE" = true ]; then
        echo ""
        echo "デバッグ情報:"
        echo "  EVAL_CONFIG_PATH=$EVAL_CONFIG_PATH"
        echo "  VLLM_MODEL_NAME=$VLLM_MODEL_NAME"
        echo "  WANDB_MODE=$WANDB_MODE"
        echo ""
    fi
    
    # Dockerコンテナの確認と起動
    if ! docker ps | grep -q llm-leaderboard; then
        echo "🚀 評価コンテナを起動中..."
        docker run -d --name llm-leaderboard --gpus all \
            --network llm-stack-network \
            -e EVAL_CONFIG_PATH="$EVAL_CONFIG_PATH" \
            -e VLLM_MODEL_NAME="$VLLM_MODEL_NAME" \
            -e WANDB_MODE="$WANDB_MODE" \
            -e NVIDIA_VISIBLE_DEVICES=all \
            -e DOCKER_HOST=unix:///var/run/docker.sock \
            -e CODE_EXECUTION_API_KEY=dify-sandbox \
            -e CODE_EXECUTION_ENDPOINT=http://dify-sandbox:8194 \
            -v ~/.cache/huggingface:/root/.cache/huggingface \
            -v ./configs:/workspace/configs:ro \
            -v ./scripts:/workspace/scripts:rw \
            -v /var/run/docker.sock:/var/run/docker.sock \
            --ipc=host \
            llm-stack-llm-leaderboard:latest \
            tail -f /dev/null
        
        # コンテナの起動を待つ
        sleep 3
    fi
    
    # 評価の実行
    echo ""
    echo "🔬 評価を開始します..."
    echo "=========================================="
    
    # リアルタイムでログを表示しながら実行
    docker exec -it llm-leaderboard bash -c "
        cd /workspace && 
        source .venv/bin/activate && 
        export WANDB_MODE=$WANDB_MODE &&
        python3 scripts/run_eval.py --config $EVAL_CONFIG_PATH
    "
    
    echo "=========================================="
    echo "✅ 評価が完了しました"
}

# スクリプトの実行
main "$@" 