#!/bin/bash

SUDO_ENV=(
    sudo env
    "PATH=$PATH"
    "CUDA_HOME=$CUDA_HOME"
    "CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"
)

PYTHON_EXEC="/export/home/lanliwei.1/miniconda3/envs/vllm/bin/python"

"${SUDO_ENV[@]}" "$PYTHON_EXEC" \
    --model_name /export/home/lanliwei.1/models/Qwen/Qwen3-30B-A3B-Instruct-2507/ \
    --draft_model_name /export/home/lanliwei.1/models/Qwen/Qwen3-0.6B/ \
    --num_gpus_train 1 \
    --num_gpus_inference 1 \
    --num_gpus_transformer 1 \
    --num_speculative_tokens 4 \
    --max_tokens 128 \