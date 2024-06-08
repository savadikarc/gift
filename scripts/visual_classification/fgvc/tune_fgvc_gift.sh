#!/bin/bash

# https://stackoverflow.com/questions/242538/unix-shell-script-find-out-which-directory-the-script-file-resides
# Absolute path to this script
SCRIPT=$(readlink -f "$0")
SCRIPT_DIR=$(dirname "$SCRIPT")
EXECUTION_DIR="$SCRIPT_DIR/../../.."

UTILS_DIR="$EXECUTION_DIR/gift_experiment_utils"

export PYTHONPATH=$PYTHONPATH:$UTILS_DIR

METHOD=gift
DATASET=$1
GPUS=$2
BACKBONE=$3

NUM_GPUS=1
DATA_SIZE=224
CONFIG_FILE="$EXECUTION_DIR/configs/training/$DATASET/$METHOD.yaml"
DATA_CONFIG_FILE="$EXECUTION_DIR/configs/data/$DATASET.yaml"
ASSETS_DIR="$EXECUTION_DIR/assets"

LRS=(1e-4 2.5e-4 5e-4 1e-3 2.5e-3 5e-3)
WDS=(0.01 0.001 0.0001 0.0)

for LR in "${LRS[@]}"; do
    for WD in "${WDS[@]}"; do
        CUDA_VISIBLE_DEVICES=$GPUS python $EXECUTION_DIR/scripts/visual_classification/train.py  \
            --img-size $DATA_SIZE \
            --pretrained \
            --config $CONFIG_FILE \
            --model $BACKBONE \
            --execution_dir $EXECUTION_DIR \
            --data-config $DATA_CONFIG_FILE \
            --method $METHOD \
            --lr $LR --weight-decay $WD \
            --hyperparameter_search --search_epochs 25 \
            ${@:4} \
            "DATA.NAME" $DATASET
    done
done
