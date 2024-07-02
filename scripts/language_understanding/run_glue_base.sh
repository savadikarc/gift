#!/bin/bash

# https://stackoverflow.com/questions/242538/unix-shell-script-find-out-which-directory-the-script-file-resides
# Absolute path to this script
SCRIPT=$(readlink -f "$0")
SCRIPT_DIR=$(dirname "$SCRIPT")
EXECUTION_DIR="$SCRIPT_DIR/../.."

TASK_NAME=$1
GPU=$2
EPOCHS=$3
MODEL_NAME=FacebookAI/roberta-base

LRS=(1e-4 2.5e-4 5e-4 7.5e-4 1e-3)
WDS=(0.0 0.0001 0.001 0.01)

for LR in "${LRS[@]}"; do
  for WD in "${WDS[@]}"; do 
    CUDA_VISIBLE_DEVICES=$GPU python $SCRIPT_DIR/run_glue_no_trainer.py \
      --model_name_or_path $MODEL_NAME \
      --task_name $TASK_NAME \
      --max_length 512 \
      --per_device_train_batch_size 64 \
      --learning_rate $LR \
      --classifier_learning_rate $LR \
      --weight_decay $WD \
      --classifier_weight_decay $WD \
      --num_train_epochs $EPOCHS \
      --warmup_ratio 0.06 \
      --hyperparameter_search \
      --output_dir $EXECUTION_DIR/artifacts-nlp/glue/$TASK_NAME/ \
      ${@:4}
  done
done

# Final test run with 5 seeds
SEEDS=(42 43 44 45 46)
for SEED in "${SEEDS[@]}"; do
  CUDA_VISIBLE_DEVICES=$GPU python $SCRIPT_DIR/run_glue_no_trainer.py \
    --model_name_or_path $MODEL_NAME \
    --task_name $TASK_NAME \
    --max_length 512 \
    --per_device_train_batch_size 64 \
    --num_train_epochs $EPOCHS \
    --warmup_ratio 0.06 \
    --override_test_hyperparams \
    --output_dir $EXECUTION_DIR/artifacts-nlp/glue/$TASK_NAME/ \
    --seed $SEED \
    --do_predict \
    ${@:4}
done
