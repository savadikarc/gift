#!/bin/bash
GPU=$1
LR=$2
SEED=$3
TUNER=$4

PROJECT=wegeft

RUN_NAME="$TUNER-meta_math-$SEED-$LR"
checkpoint_dir="./results/${PROJECT}_meta_math/$RUN_NAME"
output_dir="./results/${PROJECT}_meta_math/$RUN_NAME/eval"

echo "RUN_NAME: $RUN_NAME"
echo "checkpoint_dir: $checkpoint_dir"

# Training with MetaMathQA
CUDA_VISIBLE_DEVICES=$GPU python run_exp.py \
    +peft=$TUNER ++seed=$SEED +dataset_name=meta_math \
    +init=default ++model.learning_rate=$LR \
    ++wandb.name=$RUN_NAME \
    ++wandb.project=$PROJECT

# Evaluating on GSM8k
CUDA_VISIBLE_DEVICES=$GPU python eval_gsm8k.py \
    --checkpoint_dir $checkpoint_dir --output_dir $output_dir
