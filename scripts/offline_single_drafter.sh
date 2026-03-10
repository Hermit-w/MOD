#!/bin/bash

SUDO_ENV=(
    sudo env
    "PATH=$PATH"
    "CUDA_HOME=$CUDA_HOME"
    "CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"
)

PYTHON_EXEC="/export/home/lanliwei.1/miniconda3/envs/vllm/bin/python"

"${SUDO_ENV[@]}" "$PYTHON_EXEC" online_entries.py \
    --model_name /export/home/lanliwei.1/models/Qwen/Qwen2-57B-A14B-Instruct/ \
    --draft_model_name /export/home/lanliwei.1/models/Qwen/Qwen2-0.5B-Instruct/ \
    --num_gpus_train 1 \
    --num_gpus_inference 2 \
    --num_gpus_transformer 1 \
    --num_speculative_tokens 2 \
    --max_tokens 128 \
    --batch_size 8 \
    --buffer_size_threshold 8 \
    --output_dir /export/home/lanliwei.1/code/MOD/saving_results/offline_single_drafter \
    --datasets "gsm8k" "spider" "finance" "code" \
    --datasets_split "train" \
    --num_samples_per_dataset 1000 \
    --learning_rate 1e-4 \
    --warmup_steps 0 \
    --weight_decay 0.0 \
    --lr_scheduler_type "constant" \
    --eval_strategy "no" \
    --save_strategy "no" \
    --logging_steps 1 \
    # --enable_online_update \
    # --dry_run \


sudo chown -R lanliwei.1 /export/home/lanliwei.1/code/MOD/saving_results/offline_single_drafter