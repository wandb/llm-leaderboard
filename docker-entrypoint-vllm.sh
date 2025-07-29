#!/bin/bash

# Evaluation configuration path
if [ -z "$EVAL_CONFIG_PATH" ]; then
    echo "EVAL_CONFIG_PATH is not set. Please set it to the path of the evaluation configuration file."
    exit 1
fi
echo "EVAL_CONFIG_PATH: ${EVAL_CONFIG_PATH}"

# Model path and name
served_model_name=$(yq -r '.model.pretrained_model_name_or_path' configs/$EVAL_CONFIG_PATH)
echo "served_model_name: ${served_model_name}"

if [ -z $LOCAL_MODEL_PATH ]; then
    model_name=$served_model_name
    echo "model_name: ${model_name}"
else
    model_name=$LOCAL_MODEL_PATH
    echo "model_name: ${model_name}"
fi

# Build vLLM command arguments
vllm_args=(
    "--model" "${model_name}"
    "--served-model-name" "${served_model_name}"
)

# Number of GPUs
num_gpus=$(yq -r '.num_gpus' configs/$EVAL_CONFIG_PATH)
# Add tensor-parallel-size only if num_gpus is set and not null
if [ ! -z "$num_gpus" ] && [ "$num_gpus" != "null" ]; then
    vllm_args+=("--tensor-parallel-size" "${num_gpus}")
fi
echo "num_gpus: ${num_gpus}"

# 追加パラメータ
max_model_len=$(yq -r '.vllm.max_model_len' configs/$EVAL_CONFIG_PATH)
if [ ! -z "$max_model_len" ] && [ "$max_model_len" != "null" ]; then
    vllm_args+=("--max-model-len" "${max_model_len}")
fi
echo "max_model_len: ${max_model_len}"

dtype=$(yq -r '.vllm.dtype' configs/$EVAL_CONFIG_PATH)
if [ ! -z "$dtype" ] && [ "$dtype" != "null" ]; then
    vllm_args+=("--dtype" "${dtype}")
fi
echo "dtype: ${dtype}"

device_map=$(yq -r '.vllm.device_map' configs/$EVAL_CONFIG_PATH)
if [ ! -z "$device_map" ] && [ "$device_map" != "null" ]; then
    vllm_args+=("--device" "${device_map}")
fi
echo "device_map: ${device_map}"

gpu_memory_utilization=$(yq -r '.vllm.gpu_memory_utilization' configs/$EVAL_CONFIG_PATH)
if [ ! -z "$gpu_memory_utilization" ] && [ "$gpu_memory_utilization" != "null" ]; then
    vllm_args+=("--gpu-memory-utilization" "${gpu_memory_utilization}")
fi
echo "gpu_memory_utilization: ${gpu_memory_utilization}"

reasoning_parser=$(yq -r '.vllm.reasoning_parser' configs/$EVAL_CONFIG_PATH)
if [ ! -z "$reasoning_parser" ] && [ "$reasoning_parser" != "null" ]; then
    vllm_args+=("--reasoning-parser" "${reasoning_parser}")
fi
echo "reasoning_parser: ${reasoning_parser}"

trust_remote_code=$(yq -r '.vllm.trust_remote_code' configs/$EVAL_CONFIG_PATH)
if [ "$trust_remote_code" == "true" ]; then
    vllm_args+=("--trust-remote-code")
fi
echo "trust_remote_code: ${trust_remote_code}"

disable_triton_mma=$(yq -r '.vllm.disable_triton_mma' configs/$EVAL_CONFIG_PATH)
echo "disable_triton_mma: ${disable_triton_mma}"

# Triton MMA を無効にする場合
if [ "$disable_triton_mma" == "true" ]; then
    export VLLM_DISABLE_TRITON_MMA=1
fi

# Extra arguments
extra_args=($(yq -r '.vllm.extra_args[]' configs/$EVAL_CONFIG_PATH))
vllm_args+=("${extra_args[@]}")
echo "extra_args: ${extra_args[@]}"

# Add any additional arguments passed to the script
vllm_args+=("$@")

python3 -m vllm.entrypoints.openai.api_server "${vllm_args[@]}"
