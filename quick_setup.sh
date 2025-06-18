#!/bin/bash
# quick_setup.sh - LLM環境の簡単セットアップスクリプト

set -e

echo "=== LLM Leaderboard Environment Setup ==="
echo ""

# GPU数の検出
detect_gpus() {
    if command -v nvidia-smi &> /dev/null; then
        local gpu_count=$(nvidia-smi --list-gpus | wc -l)
        echo $gpu_count
    else
        echo "0"
    fi
}

GPU_COUNT=$(detect_gpus)

echo "検出されたGPU数: $GPU_COUNT"
echo ""

if [ "$GPU_COUNT" -eq 0 ]; then
    echo "エラー: GPUが検出されませんでした。"
    echo "NVIDIA ドライバーとDocker NVIDIAランタイムがインストールされていることを確認してください。"
    exit 1
fi

# 設定の選択
echo "GPU構成を選択してください:"
echo "1) 2GPU環境 (vLLM: GPU 0, Leaderboard: GPU 1)"
echo "2) 4GPU環境 (vLLM: GPU 0-1, Leaderboard: GPU 2-3)"
echo "3) 8GPU環境 (vLLM: GPU 0-3, Leaderboard: GPU 4-7)"
echo "4) カスタム設定"
echo "5) 最小構成"

read -p "選択 (1-5): " choice

case $choice in
    1)
        if [ "$GPU_COUNT" -lt 2 ]; then
            echo "エラー: 2GPU構成には最低2つのGPUが必要です。"
            exit 1
        fi
        VLLM_GPUS="0"
        LEADERBOARD_GPUS="1"
        TENSOR_PARALLEL_SIZE=1
        ;;
    2)
        if [ "$GPU_COUNT" -lt 4 ]; then
            echo "エラー: 4GPU構成には最低4つのGPUが必要です。"
            exit 1
        fi
        VLLM_GPUS="0,1"
        LEADERBOARD_GPUS="2,3"
        TENSOR_PARALLEL_SIZE=2
        ;;
    3)
        if [ "$GPU_COUNT" -lt 8 ]; then
            echo "エラー: 8GPU構成には最低8つのGPUが必要です。"
            exit 1
        fi
        VLLM_GPUS="0,1,2,3"
        LEADERBOARD_GPUS="4,5,6,7"
        TENSOR_PARALLEL_SIZE=4
        ;;
    4)
        read -p "vLLM用GPU (カンマ区切り, 例: 0,1): " VLLM_GPUS
        read -p "Leaderboard用GPU (カンマ区切り, 例: 2,3): " LEADERBOARD_GPUS
        IFS=',' read -ra VLLM_ARRAY <<< "$VLLM_GPUS"
        TENSOR_PARALLEL_SIZE=${#VLLM_ARRAY[@]}
        ;;
    5)
        VLLM_GPUS="0"
        LEADERBOARD_GPUS=""
        TENSOR_PARALLEL_SIZE=1
        ;;
    *)
        echo "無効な選択です。"
        exit 1
        ;;
esac

# モデル名の必須入力
echo ""
echo "=== モデル設定 ==="
echo "HuggingFaceのリポジトリ名または、ローカルの絶対パスを入力してください"
echo "例: tokyotech-llm/Swallow-7b-instruct-v0.1, /models/Swallow-7b-instruct-v0.1"
echo ""

# モデル名の必須入力
while true; do
    read -p "モデル名: " MODEL_NAME
    if [ -n "$MODEL_NAME" ]; then
        break
    else
        echo "エラー: モデル名は必須です。再度入力してください。"
    fi
done

# 絶対パスかどうかを判定し、Dockerマウント設定を準備
if [[ "$MODEL_NAME" = /* ]]; then
    # 絶対パスの場合
    if [ ! -d "$MODEL_NAME" ]; then
        echo "警告: 指定されたディレクトリが存在しません: $MODEL_NAME"
        read -p "続行しますか？ (y/N): " path_confirm
        if [[ ! "$path_confirm" =~ ^[Yy]$ ]]; then
            echo "セットアップを中止しました。"
            exit 0
        fi
    fi
    LOCAL_MODEL_PATH="$MODEL_NAME"
    echo "ローカルモデルが検出されました: $MODEL_NAME"
else
    # HuggingFaceリポジトリの場合
    LOCAL_MODEL_PATH=""
    echo "HuggingFaceリポジトリが指定されました: $MODEL_NAME"
fi

# 最大トークン長の入力
read -p "最大トークン長 [デフォルト: 4096]: " MAX_MODEL_LEN
MAX_MODEL_LEN=${MAX_MODEL_LEN:-4096}

# config fileの必須入力
show_config_files() {
    echo "現在利用可能なconfigファイル:"
    ls configs/*.yml configs/*.yaml 2>/dev/null | sed 's|configs/||' || echo "configファイルが見つかりません"
    echo
}

echo ""
echo "=== config設定 ==="
echo "'ls' と入力するとconfigファイル一覧を表示できます"

while true; do
    read -p "configファイル名を入力してください: " EVAL_CONFIG_PATH
    
    # lsコマンドで一覧表示
    if [ "$EVAL_CONFIG_PATH" = "ls" ] || [ "$EVAL_CONFIG_PATH" = "list" ]; then
        show_config_files
        continue
    fi

    if [ -n "$EVAL_CONFIG_PATH" ]; then
        # パスからファイル名だけを取得
        EVAL_CONFIG_PATH=$(basename "$EVAL_CONFIG_PATH")
        
        # .yml または .yaml で終わるかチェック
        if [[ "$EVAL_CONFIG_PATH" == *.yml ]] || [[ "$EVAL_CONFIG_PATH" == *.yaml ]]; then
            # ファイルが存在するかチェック
            if [ -f "configs/$EVAL_CONFIG_PATH" ]; then
                break
            else
                echo "エラー: ファイル 'configs/$EVAL_CONFIG_PATH' が見つかりません。"
                echo "1) 再度入力する"
                echo "2) 続行する（後でファイルを作成）"
                read -p "選択してください (1/2): " choice
                case $choice in
                    1)
                        continue
                        ;;
                    2)
                        echo "注意: ファイル 'configs/$EVAL_CONFIG_PATH' は後で作成してください。"
                        break
                        ;;
                    *)
                        echo "無効な選択です。再度入力してください。"
                        continue
                        ;;
                esac
            fi
        else
            echo "エラー: configファイルは.ymlまたは.yaml形式である必要があります。再度入力してください。"
        fi
    else
        echo "エラー: configファイル名は必須です。再度入力してください。"
    fi
done

# 必須APIキーの入力（非表示）
echo ""
echo "=== 必須APIキーの設定 ==="
echo "※ 入力内容は画面に表示されません"
echo ""

# WANDB APIキーの必須入力
while true; do
    echo -n "WANDB APIキーを入力してください: "
    read -s WANDB_API_KEY
    echo ""
    if [ -n "$WANDB_API_KEY" ]; then
        break
    else
        echo "エラー: WANDB APIキーは必須です。再度入力してください。"
    fi
done

# OpenAI APIキーの必須入力
while true; do
    echo -n "OpenAI APIキーを入力してください: "
    read -s OPENAI_API_KEY
    echo ""
    if [ -n "$OPENAI_API_KEY" ]; then
        break
    else
        echo "エラー: OpenAI APIキーは必須です。再度入力してください。"
    fi
done

# 任意APIキーの入力
echo ""
echo "=== 任意APIキーの設定 ==="
echo "※ 不要な場合はEnterキーで空白のまま進んでください"
echo "※ 入力内容は画面に表示されません"
echo ""

echo -n "HuggingFace Hub Token: "
read -s HUGGINGFACE_HUB_TOKEN
echo ""

echo -n "Anthropic API Key: "
read -s ANTHROPIC_API_KEY
echo ""

echo -n "Google API Key: "
read -s GOOGLE_API_KEY
echo ""

echo -n "Cohere API Key: "
read -s COHERE_API_KEY
echo ""

echo -n "Mistral API Key: "
read -s MISTRAL_API_KEY
echo ""

echo -n "Upstage API Key: "
read -s UPSTAGE_API_KEY
echo ""

# AWS設定
echo ""
echo "=== AWS設定（任意） ==="
echo ""

read -p "AWS Access Key ID: " AWS_ACCESS_KEY_ID
read -p "AWS Secret Access Key: " AWS_SECRET_ACCESS_KEY
read -p "AWS Default Region [デフォルト: us-east-1]: " AWS_DEFAULT_REGION
AWS_DEFAULT_REGION=${AWS_DEFAULT_REGION:-us-east-1}

# Azure OpenAI設定
echo ""
echo "=== Azure OpenAI設定（任意） ==="
echo ""

read -p "Azure OpenAI Endpoint: " AZURE_OPENAI_ENDPOINT
if [ -n "$AZURE_OPENAI_ENDPOINT" ]; then
    echo -n "Azure OpenAI API Key: "
    read -s AZURE_OPENAI_API_KEY
    echo ""
    read -p "OpenAI API Type [デフォルト: azure]: " OPENAI_API_TYPE
    OPENAI_API_TYPE=${OPENAI_API_TYPE:-azure}
else
    AZURE_OPENAI_API_KEY=""
    OPENAI_API_TYPE=""
fi

# 入力確認
echo ""
echo "=== 入力確認 ==="
echo "選択されたモデル: $MODEL_NAME"
echo "最大トークン長: $MAX_MODEL_LEN"
echo "WANDB APIキー: [設定済み]"
echo "OpenAI APIキー: [設定済み]"

# 任意項目の設定状況を表示
optional_keys=""
[ -n "$HUGGINGFACE_HUB_TOKEN" ] && optional_keys+="HuggingFace, "
[ -n "$ANTHROPIC_API_KEY" ] && optional_keys+="Anthropic, "
[ -n "$GOOGLE_API_KEY" ] && optional_keys+="Google, "
[ -n "$COHERE_API_KEY" ] && optional_keys+="Cohere, "
[ -n "$MISTRAL_API_KEY" ] && optional_keys+="Mistral, "
[ -n "$UPSTAGE_API_KEY" ] && optional_keys+="Upstage, "
[ -n "$AWS_ACCESS_KEY_ID" ] && optional_keys+="AWS, "
[ -n "$AZURE_OPENAI_ENDPOINT" ] && optional_keys+="Azure OpenAI, "

if [ -n "$optional_keys" ]; then
    optional_keys=${optional_keys%, }  # 末尾のカンマを削除
    echo "任意APIキー: $optional_keys"
else
    echo "任意APIキー: なし"
fi

echo ""
read -p "この設定で続行しますか？ (y/N): " confirm

if [[ ! "$confirm" =~ ^[Yy]$ ]]; then
    echo "セットアップを中止しました。"
    exit 0
fi

# 個別GPU環境変数を生成
generate_gpu_vars() {
    local gpu_list="$1"
    local prefix="$2"
    local vars=""
    
    if [ -z "$gpu_list" ]; then
        echo "${prefix}_GPU_COUNT=0"
        return
    fi
    
    IFS=',' read -ra GPU_ARRAY <<< "$gpu_list"
    for i in "${!GPU_ARRAY[@]}"; do
        vars+="${prefix}_GPU_${i}=${GPU_ARRAY[i]}"$'\n'
    done
    vars+="${prefix}_GPU_COUNT=${#GPU_ARRAY[@]}"
    echo "$vars"
}

# GPU設定を配列形式に変換
convert_to_array() {
    local gpu_list="$1"
    if [ -z "$gpu_list" ]; then
        echo "[]"
        return
    fi
    IFS=',' read -ra GPU_ARRAY <<< "$gpu_list"
    local result="["
    for i in "${!GPU_ARRAY[@]}"; do
        result+="\"${GPU_ARRAY[i]}\""
        if [[ $i -lt $((${#GPU_ARRAY[@]} - 1)) ]]; then
            result+=","
        fi
    done
    result+="]"
    echo "$result"
}

VLLM_GPU_VARS=$(generate_gpu_vars "$VLLM_GPUS" "VLLM")
LEADERBOARD_GPU_VARS=$(generate_gpu_vars "$LEADERBOARD_GPUS" "LEADERBOARD")
VLLM_GPU_IDS=$(convert_to_array "$VLLM_GPUS")
LEADERBOARD_GPU_IDS=$(convert_to_array "$LEADERBOARD_GPUS")

# env_filesディレクトリの作成
echo ""
echo "環境設定ファイルを作成中..."

mkdir -p ./env_files

# モデル名からファイル名を生成
MODEL_FILE_NAME=$(basename "$MODEL_NAME" | sed 's/[^a-zA-Z0-9_-]/_/g')
ENV_FILENAME="${MODEL_FILE_NAME}.env"
ENV_FILEPATH="./env_files/${ENV_FILENAME}"

echo "設定ファイル: $ENV_FILEPATH"

# 設定ファイルの内容を生成
cat > "$ENV_FILEPATH" << EOL
# LLM Stack Configuration
# Generated by quick_setup.sh

# Model Settings
MODEL_NAME=$MODEL_NAME
SERVED_MODEL_NAME=$MODEL_NAME
EVAL_CONFIG_PATH=$EVAL_CONFIG_PATH
DTYPE=half
MAX_MODEL_LEN=$MAX_MODEL_LEN
VLLM_PORT=8000

# GPU Configuration - Array Format (for compatibility)
VLLM_GPU_IDS=$VLLM_GPU_IDS
TENSOR_PARALLEL_SIZE=$TENSOR_PARALLEL_SIZE
LEADERBOARD_GPU_IDS=$LEADERBOARD_GPU_IDS
NVIDIA_VISIBLE_DEVICES=all
LOCAL_MODEL_PATH=$LOCAL_MODEL_PATH

# GPU Configuration - Individual GPU IDs
$VLLM_GPU_VARS
$LEADERBOARD_GPU_VARS

# Required API Keys
WANDB_API_KEY=$WANDB_API_KEY
OPENAI_API_KEY=$OPENAI_API_KEY

# Other API Keys
HUGGINGFACE_HUB_TOKEN=$HUGGINGFACE_HUB_TOKEN
ANTHROPIC_API_KEY=$ANTHROPIC_API_KEY
GOOGLE_API_KEY=$GOOGLE_API_KEY
COHERE_API_KEY=$COHERE_API_KEY
MISTRAL_API_KEY=$MISTRAL_API_KEY
UPSTAGE_API_KEY=$UPSTAGE_API_KEY

# AWS Configuration
AWS_ACCESS_KEY_ID=$AWS_ACCESS_KEY_ID
AWS_SECRET_ACCESS_KEY=$AWS_SECRET_ACCESS_KEY
AWS_DEFAULT_REGION=$AWS_DEFAULT_REGION

# Azure OpenAI
AZURE_OPENAI_ENDPOINT=$AZURE_OPENAI_ENDPOINT
AZURE_OPENAI_API_KEY=$AZURE_OPENAI_API_KEY
OPENAI_API_TYPE=$OPENAI_API_TYPE

# Other Settings
LANG=ja_JP.UTF-8
PYTHONPATH=/workspace
JUMANPP_COMMAND=/usr/local/bin/jumanpp
COMPOSE_PROJECT_NAME=llm-stack
EOL

# .envファイルにもコピー
echo "メインの.envファイルを作成中..."
cp "$ENV_FILEPATH" .env

echo "設定完了！"
echo ""
echo "=== 設定サマリー ==="
echo "モデル: $MODEL_NAME"
echo "最大トークン長: $MAX_MODEL_LEN"
echo "vLLM GPU: $VLLM_GPU_IDS (Tensor Parallel: $TENSOR_PARALLEL_SIZE)"
if [ -n "$LEADERBOARD_GPUS" ]; then
    echo "Leaderboard GPU: $LEADERBOARD_GPU_IDS"
else
    echo "Leaderboard: CPU実行"
fi
echo "設定ファイル: $ENV_FILEPATH"
echo "メイン設定: .env"
echo "WANDB APIキー: [設定済み]"
echo "OpenAI APIキー: [設定済み]"

if [ -n "$optional_keys" ]; then
    echo "任意APIキー: $optional_keys [設定済み]"
fi

echo "==================="
echo ""
echo "設定ファイルの場所:"
echo "- メイン設定: .env"
echo "- バックアップ: $ENV_FILEPATH"
echo ""

# generate_docker_override.shの実行
echo "=== Docker Override設定の生成 ==="
echo ""

if [ -f "./generate_docker_override.sh" ]; then
    echo "generate_docker_override.shを実行しています..."
    chmod +x ./generate_docker_override.sh
    
    if ./generate_docker_override.sh; then
        echo ""
        echo "✅ Docker Override設定が正常に生成されました！"
        echo ""
        echo "=== セットアップ完了 ==="
        echo "環境の準備が完了しました。以下のコマンドでコンテナを起動できます："
        echo ""
        echo "  docker-compose up -d"
        echo ""
        echo "または、特定のサービスのみ起動する場合："
        echo "  docker-compose up -d vllm"
        echo "  docker-compose up -d leaderboard"
        echo ""
    else
        echo ""
        echo "⚠️  generate_docker_override.shの実行でエラーが発生しました。"
        echo "手動で以下を実行してください："
        echo "  ./generate_docker_override.sh"
        echo ""
    fi
else
    echo "⚠️  generate_docker_override.shが見つかりません。"
    echo "次のステップ:"
    echo "1. 必要に応じて .env ファイルを直接編集してください"
    echo "2. generate_docker_override.shを実行してください"
    echo ""
fi