#!/bin/bash

GPU=$1
LR=$2
SEED=$3
TUNER=$4

PROJECT=wegeft

RUN_NAME="$TUNER-codefeedback-$SEED-$LR"
checkpoint_dir="./results/${PROJECT}_codefeedback/$RUN_NAME"
output_dir="./results/${PROJECT}_codefeedback/$RUN_NAME/eval"

echo "RUN_NAME: $RUN_NAME"
echo "checkpoint_dir: $checkpoint_dir"

# Train using codefeedback
CUDA_VISIBLE_DEVICES=$GPU python run_exp.py \
    +peft=$TUNER ++seed=$SEED +dataset_name=codefeedback \
    +init=default ++model.learning_rate=$LR \
    ++wandb.name=$RUN_NAME \
    ++wandb.project=$PROJECT

# Generate responses to humaneval
CUDA_VISIBLE_DEVICES=$GPU python eval_humaneval.py \
    --checkpoint_dir $checkpoint_dir --output_dir $output_dir

# Evaluate
evaluate_functional_correctness $output_dir/humaneval_samples.jsonl
