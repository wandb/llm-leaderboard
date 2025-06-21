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

# モデル入力方法の選択
echo ""
echo "=== モデル設定 ==="
echo "モデルの種類を選択してください:"
echo "1) ローカルモデル"
echo "2) HuggingFaceモデル"
echo "3) API"
echo ""

while true; do
    read -p "選択 (1-3): " model_choice
    case $model_choice in
        1|2|3)
            break
            ;;
        *)
            echo "無効な選択です。1-3の中から選択してください。"
            ;;
    esac
done

MODEL_NAME=""
LOCAL_MODEL_PATH=""
USE_API=false
API_PROVIDER=""
API_MODEL_NAME=""

case $model_choice in
    1)
        # ローカルモデル
        echo ""
        echo "=== ローカルモデル設定 ==="
        echo "モデルの絶対パスを入力してください"
        echo "例: /models/Swallow-7b-instruct-v0.1"
        echo ""
        
        while true; do
            read -p "モデルパス: " MODEL_NAME
            if [ -n "$MODEL_NAME" ]; then
                if [[ "$MODEL_NAME" = /* ]]; then
                    if [ ! -d "$MODEL_NAME" ]; then
                        echo "警告: 指定されたディレクトリが存在しません: $MODEL_NAME"
                        read -p "続行しますか？ (y/N): " path_confirm
                        if [[ ! "$path_confirm" =~ ^[Yy]$ ]]; then
                            echo "再度入力してください。"
                            continue
                        fi
                    fi
                    LOCAL_MODEL_PATH="$MODEL_NAME"
                    break
                else
                    echo "エラー: 絶対パスを入力してください（/で始まる）"
                fi
            else
                echo "エラー: モデルパスは必須です。再度入力してください。"
            fi
        done
        ;;
    2)
        # HuggingFaceモデル
        echo ""
        echo "=== HuggingFaceモデル設定 ==="
        echo "HuggingFaceのリポジトリ名を入力してください"
        echo "例: tokyotech-llm/Swallow-7b-instruct-v0.1"
        echo ""
        
        while true; do
            read -p "リポジトリ名: " MODEL_NAME
            if [ -n "$MODEL_NAME" ]; then
                break
            else
                echo "エラー: リポジトリ名は必須です。再度入力してください。"
            fi
        done
        ;;
    3)
        # API
        echo ""
        echo "=== API設定 ==="
        echo "APIプロバイダーを選択してください:"
        echo "1) OpenAI"
        echo "2) Anthropic"
        echo "3) Google"
        echo "4) Cohere"
        echo "5) Mistral"
        echo "6) Upstage"
        echo "7) Azure OpenAI"
        echo "8) Amazon Bedrock"
        echo "9) xAI"
        echo ""
        
        while true; do
            read -p "選択 (1-9): " api_choice
            case $api_choice in
                1) API_PROVIDER="openai"; break;;
                2) API_PROVIDER="anthropic"; break;;
                3) API_PROVIDER="google"; break;;
                4) API_PROVIDER="cohere"; break;;
                5) API_PROVIDER="mistral"; break;;
                6) API_PROVIDER="upstage"; break;;
                7) API_PROVIDER="azure_openai"; break;;
                8) API_PROVIDER="amazon_bedrock"; break;;
                9) API_PROVIDER="xai"; break;;
                *)
                    echo "無効な選択です。1-9の中から選択してください。"
                    ;;
            esac
        done
        
        USE_API=true
        MODEL_NAME="api_model"  # APIの場合はダミーの値
        ;;
esac

# API使用時以外は最大トークン長を設定
if [ "$USE_API" = false ]; then
    # 最大トークン長の入力
    read -p "最大トークン長 [デフォルト: 4096]: " MAX_MODEL_LEN
    MAX_MODEL_LEN=${MAX_MODEL_LEN:-4096}
else
    MAX_MODEL_LEN=4096  # APIの場合はデフォルト値
fi

# API設定
if [ "$USE_API" = true ]; then
    echo ""
    echo "=== ${API_PROVIDER^^} API設定 ==="
    echo "※ 入力内容は画面に表示されません"
    echo ""
    LEADERBOARD_GPUS=$(seq -s, 0 $((GPU_COUNT-1)))
    
    case $API_PROVIDER in
        "openai")
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
            ;;
        "anthropic")
            while true; do
                echo -n "Anthropic APIキーを入力してください: "
                read -s ANTHROPIC_API_KEY
                echo ""
                if [ -n "$ANTHROPIC_API_KEY" ]; then
                    break
                else
                    echo "エラー: Anthropic APIキーは必須です。再度入力してください。"
                fi
            done
            ;;
        "google")
            while true; do
                echo -n "Google APIキーを入力してください: "
                read -s GOOGLE_API_KEY
                echo ""
                if [ -n "$GOOGLE_API_KEY" ]; then
                    break
                else
                    echo "エラー: Google APIキーは必須です。再度入力してください。"
                fi
            done
            ;;
        "cohere")
            while true; do
                echo -n "Cohere APIキーを入力してください: "
                read -s COHERE_API_KEY
                echo ""
                if [ -n "$COHERE_API_KEY" ]; then
                    break
                else
                    echo "エラー: Cohere APIキーは必須です。再度入力してください。"
                fi
            done
            ;;
        "mistral")
            while true; do
                echo -n "Mistral APIキーを入力してください: "
                read -s MISTRAL_API_KEY
                echo ""
                if [ -n "$MISTRAL_API_KEY" ]; then
                    break
                else
                    echo "エラー: Mistral APIキーは必須です。再度入力してください。"
                fi
            done
            ;;
        "upstage")
            while true; do
                echo -n "Upstage APIキーを入力してください: "
                read -s UPSTAGE_API_KEY
                echo ""
                if [ -n "$UPSTAGE_API_KEY" ]; then
                    break
                else
                    echo "エラー: Upstage APIキーは必須です。再度入力してください。"
                fi
            done
            ;;
        "azure_openai")
            while true; do
                echo -n "Azure OpenAI APIキーを入力してください: "
                read -s AZURE_OPENAI_API_KEY
                echo ""
                if [ -n "$AZURE_OPENAI_API_KEY" ]; then
                    break
                else
                    echo "エラー: Azure OpenAI APIキーは必須です。再度入力してください。"
                fi
            done
            
            while true; do
                read -p "Azure OpenAI Endpoint: " AZURE_OPENAI_ENDPOINT
                if [ -n "$AZURE_OPENAI_ENDPOINT" ]; then
                    break
                else
                    echo "エラー: Azure OpenAI Endpointは必須です。再度入力してください。"
                fi
            done
            
            read -p "デプロイメント名: " AZURE_DEPLOYMENT_NAME
            read -p "APIバージョン [デフォルト: 2024-02-15-preview]: " AZURE_API_VERSION
            AZURE_API_VERSION=${AZURE_API_VERSION:-2024-02-15-preview}
            
            OPENAI_API_TYPE="azure"
            ;;
        "amazon_bedrock")
            read -p "AWS Access Key ID: " AWS_ACCESS_KEY_ID
            read -p "AWS Secret Access Key: " AWS_SECRET_ACCESS_KEY
            read -p "AWS Region [デフォルト: us-west-2]: " AWS_DEFAULT_REGION
            AWS_DEFAULT_REGION=${AWS_DEFAULT_REGION:-us-west-2}
            ;;
        "xai")
            while true; do
                echo -n "xAI APIキーを入力してください: "
                read -s XAI_API_KEY
                echo ""
                if [ -n "$XAI_API_KEY" ]; then
                    break
                else
                    echo "エラー: xAI APIキーは必須です。再度入力してください。"
                fi
            done
            ;;
    esac
else 
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
fi

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

if [ -z "$OPENAI_API_KEY" ]; then
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
fi

# 共通の任意APIキー設定（APIを使わない場合、またはメインAPI以外のキーが必要な場合）
if [ "$USE_API" = false ] || [ "$API_PROVIDER" != "anthropic" ]; then
    echo ""
    echo "=== 任意APIキーの設定 ==="
    echo "※ 不要な場合はEnterキーで空白のまま進んでください"
    echo "※ 入力内容は画面に表示されません"
    echo ""

    if [ -z "$HUGGINGFACE_HUB_TOKEN" ]; then
        echo -n "HuggingFace Hub Token: "
        read -s HUGGINGFACE_HUB_TOKEN
        echo ""
    fi

    if [ -z "$ANTHROPIC_API_KEY" ]; then
        echo -n "Anthropic API Key: "
        read -s ANTHROPIC_API_KEY
        echo ""
    fi

    if [ -z "$GOOGLE_API_KEY" ]; then
        echo -n "Google API Key: "
        read -s GOOGLE_API_KEY
        echo ""
    fi

    if [ -z "$COHERE_API_KEY" ]; then
        echo -n "Cohere API Key: "
        read -s COHERE_API_KEY
        echo ""
    fi

    if [ -z "$MISTRAL_API_KEY" ]; then
        echo -n "Mistral API Key: "
        read -s MISTRAL_API_KEY
        echo ""
    fi

    if [ -z "$UPSTAGE_API_KEY" ]; then
        echo -n "Upstage API Key: "
        read -s UPSTAGE_API_KEY
        echo ""
    fi

    if [ -z "$XAI_API_KEY" ]; then
        echo -n "xAI API Key: "
        read -s XAI_API_KEY
        echo ""
    fi
fi

# AWS設定（Amazon Bedrock以外の場合）
if [ "$API_PROVIDER" != "amazon_bedrock" ]; then
    echo ""
    echo "=== AWS設定（任意） ==="
    echo ""

    if [ -z "$AWS_ACCESS_KEY_ID" ]; then
        read -p "AWS Access Key ID: " AWS_ACCESS_KEY_ID
        read -p "AWS Secret Access Key: " AWS_SECRET_ACCESS_KEY
        read -p "AWS Default Region [デフォルト: us-west-2]: " AWS_DEFAULT_REGION
        AWS_DEFAULT_REGION=${AWS_DEFAULT_REGION:-us-west-2}
    fi
fi

# Azure OpenAI設定（Azure OpenAI以外の場合）
if [ "$API_PROVIDER" != "azure_openai" ]; then
    echo ""
    echo "=== Azure OpenAI設定（任意） ==="
    echo ""

    if [ -z "$AZURE_OPENAI_ENDPOINT" ]; then
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
    fi
fi

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

# 入力確認
echo ""
echo "=== 入力確認 ==="
if [ "$USE_API" = true ]; then
    echo "モード: API (${API_PROVIDER^^})"
else
    echo "モード: ローカル/HuggingFace"
    echo "選択されたモデル: $MODEL_NAME"
    echo "最大トークン長: $MAX_MODEL_LEN"
fi

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
[ -n "$XAI_API_KEY" ] && optional_keys+="xAI, "
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

# ファイル名の生成
if [ "$USE_API" = true ]; then
    ENV_FILENAME="${API_PROVIDER}_${API_MODEL_NAME//[^a-zA-Z0-9_-]/_}.env"
else
    MODEL_FILE_NAME=$(basename "$MODEL_NAME" | sed 's/[^a-zA-Z0-9_-]/_/g')
    ENV_FILENAME="${MODEL_FILE_NAME}.env"
fi
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

# API Configuration
USE_API=$USE_API
API_PROVIDER=$API_PROVIDER

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
WANDB_API_KEY=${WANDB_API_KEY:-}
OPENAI_API_KEY=${OPENAI_API_KEY:-}

# Other API Keys
HUGGINGFACE_HUB_TOKEN=${HUGGINGFACE_HUB_TOKEN:-}
ANTHROPIC_API_KEY=${ANTHROPIC_API_KEY:-}
GOOGLE_API_KEY=${GOOGLE_API_KEY:-}
COHERE_API_KEY=${COHERE_API_KEY:-}
MISTRAL_API_KEY=${MISTRAL_API_KEY:-}
UPSTAGE_API_KEY=${UPSTAGE_API_KEY:-}
XAI_API_KEY=${XAI_API_KEY:-}

# AWS Configuration
AWS_ACCESS_KEY_ID=${AWS_ACCESS_KEY_ID:-}
AWS_SECRET_ACCESS_KEY=${AWS_SECRET_ACCESS_KEY:-}
AWS_DEFAULT_REGION=${AWS_DEFAULT_REGION:-us-west-2}

# Azure OpenAI
AZURE_OPENAI_ENDPOINT=${AZURE_OPENAI_ENDPOINT:-}
AZURE_OPENAI_API_KEY=${AZURE_OPENAI_API_KEY:-}
AZURE_DEPLOYMENT_NAME=${AZURE_DEPLOYMENT_NAME:-}
AZURE_API_VERSION=${AZURE_API_VERSION:-}
OPENAI_API_TYPE=${OPENAI_API_TYPE:-}

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
if [ "$USE_API" = true ]; then
    echo "モード: API (${API_PROVIDER^^})"
else
    echo "モード: ローカル/HuggingFace"
    echo "モデル: $MODEL_NAME"
    echo "最大トークン長: $MAX_MODEL_LEN"
    echo "vLLM GPU: $VLLM_GPU_IDS (Tensor Parallel: $TENSOR_PARALLEL_SIZE)"
    if [ -n "$LEADERBOARD_GPUS" ]; then
        echo "Leaderboard GPU: $LEADERBOARD_GPU_IDS"
    else
        echo "Leaderboard: CPU実行"
    fi
fi

echo "設定ファイル: $ENV_FILEPATH"
echo "メイン設定: .env"

if [ "$USE_API" = false ]; then
    echo "WANDB APIキー: [設定済み]"
    echo "OpenAI APIキー: [設定済み]"
fi

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
        if [ "$USE_API" = true ]; then
            echo "  docker-compose up -d llm-leaderboard"
            echo ""
            echo "APIモードのため、vLLMサービスは不要です。"
        else
            echo "1. vllmサービスを起動："
            echo "  docker-compose up -d vllm"
            exho "2. vllmサーバーの起動確認"
            echo "  docker compose logs -f vllm"
            echo "3. 評価の実行"
            echo "  docker-compose up -d llm-leaderboard"
        fi
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