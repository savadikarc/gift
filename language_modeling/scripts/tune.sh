#!/bin/bash

# https://stackoverflow.com/questions/242538/unix-shell-script-find-out-which-directory-the-script-file-resides
# Absolute path to this script
SCRIPT=$(readlink -f "$0")
SCRIPT_DIR=$(dirname "$SCRIPT")
EXECUTION_DIR="$SCRIPT_DIR/.."

# Required for simple_peft imports
SIMPLE_PEFT_PATH="$SCRIPT_DIR/.."
export PYTHONPATH="$SIMPLE_PEFT_PATH":$PYTHONPATH

GPU=$1
TASK=$2
LR=$3
WARMUP_RATIO=$4
TUNER=$5

SEED=42

MODEL="yahma/llama-7b-hf"

BATCH_SIZE=16
ACCUM_STEPS=1
EPOCHS=3

PROJECT="wegeft"
WANDB_USER="savadikarc"
WANDB_DIR="$EXECUTION_DIR/artifacts"

CUDA_VISIBLE_DEVICES=$GPU python $SCRIPT_DIR/train_$TUNER.py --task $TASK \
    --data_dir_or_branch $EXECUTION_DIR/data/language_data \
    --output_dir $EXECUTION_DIR/artifacts/$PROJECT/$TASK/$MODEL/$TUNER/validation \
    --cache_dir $EXECUTION_DIR/data/cache/$TASK \
    --model $MODEL \
    --seed $SEED \
    --num_train_epochs $EPOCHS \
    --learning_rate $LR \
    --lr_scheduler_type linear \
    --evaluation_strategy no \
    --gradient_accumulation_steps $ACCUM_STEPS \
    --per_device_train_batch_size $BATCH_SIZE \
    --per_device_eval_batch_size 4 \
    --train_split train \
    --test_split validation \
    --validation_percent 0.1 \
    --warmup_ratio $WARMUP_RATIO \
    --weight_decay 0.0 \
    --dtype bfloat16 \
    --gradient_checkpointing True \
    --report_to none \
    --wandb_proj $PROJECT \
    --wandb_name $WANDB_USER \
    --wandb_dir $WANDB_DIR \
    --logging_steps 1 \
    --save_strategy no \
    ${@:6}
