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

# Number of GPUs
num_gpus=$(yq -r '.num_gpus' configs/$EVAL_CONFIG_PATH)
echo "num_gpus: ${num_gpus}"

# Extra arguments
extra_args=($(yq -r '.vllm.extra_args[]' configs/$EVAL_CONFIG_PATH))
echo "extra_args: ${extra_args[@]}"

# Build vLLM command arguments
vllm_args=(
    "--model" "${model_name}"
    "--served-model-name" "${served_model_name}"
    "--dtype" "float16"
)

# Add tensor-parallel-size only if num_gpus is set and not null
if [ ! -z "$num_gpus" ] && [ "$num_gpus" != "null" ]; then
    vllm_args+=("--tensor-parallel-size" "${num_gpus}")
fi

# Add extra arguments
vllm_args+=("${extra_args[@]}")

# Add any additional arguments passed to the script
vllm_args+=("$@")

python3 -m vllm.entrypoints.openai.api_server "${vllm_args[@]}"
