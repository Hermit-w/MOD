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
    --num_speculative_tokens 4 \
    --max_tokens 128 \
    --batch_size 8 \
    --buffer_size_threshold 8 \
    --output_dir /export/home/lanliwei.1/code/MOD/saving_results/online_single_drafter \
    --enable_online_update \
    --datasets "finance" \
    --datasets_split "train" \
    --num_samples_per_dataset 1000 \
    --learning_rate 1e-5 \
    --warmup_steps 20 \
    --weight_decay 0.1 \
    --lr_scheduler_type "inverse_sqrt" \
    --eval_strategy "no" \
    --save_strategy "no" \
    --logging_steps 1 \
    --max_grad_norm 1.0 \
    --no_loss_on_wrong_tokens \
    # --dry_run \


sudo chown -R lanliwei.1 /export/home/lanliwei.1/code/MOD/saving_results/online_single_drafter