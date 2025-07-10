#!/bin/bash
GPU=$1
LR=$2
SEED=$3
TUNER=$4

PROJECT=wegeft

RUN_NAME="$TUNER-wizard_lm-$SEED-$LR"
checkpoint_dir="./results/${PROJECT}_wizard_lm/$RUN_NAME"
output_dir="./results/${PROJECT}_wizard_lm/$RUN_NAME/eval"

echo "RUN_NAME: $RUN_NAME"
echo "checkpoint_dir: $checkpoint_dir"

# Train using WizardLM
CUDA_VISIBLE_DEVICES=$GPU python run_exp.py \
    +peft=$TUNER ++seed=$SEED +dataset_name=wizard_lm \
    +init=default ++model.learning_rate=$LR \
    ++wandb.name=$RUN_NAME \
    ++wandb.project=wegeft

# Generate responses on MTBench
# CUDA_VISIBLE_DEVICES=$GPU python eval_mtbench.py \
#     --model-path $checkpoint_dir --model-id $RUN_NAME \
#     --answer-file $output_dir/answers.json

# export OPENAI_API_KEY=<KEY>

# CUDA_VISIBLE_DEVICES=$GPU python gen_judgment.py \
#     --answer-dir $ANSWER_DIR \
#     --output-dir $OUTPUT_DIR \
#     --parallel 5

# python get_score.py --file $OUTPUT_DIR/gpt-4_single.jsonl
