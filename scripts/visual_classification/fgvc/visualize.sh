#!/bin/bash

# https://stackoverflow.com/questions/242538/unix-shell-script-find-out-which-directory-the-script-file-resides
# Absolute path to this script
SCRIPT=$(readlink -f "$0")
SCRIPT_DIR=$(dirname "$SCRIPT")
EXECUTION_DIR="$SCRIPT_DIR/../../.."

UTILS_DIR="$EXECUTION_DIR/gift_experiment_utils"

export PYTHONPATH=$PYTHONPATH:$UTILS_DIR

METHOD=$1
DATASET=$2
EXP_NAME=$3
GPUS=$4
BACKBONE=$5

NUM_GPUS=1
DATA_SIZE=224
CONFIG_FILE="$EXECUTION_DIR/configs/training/$DATASET/$METHOD.yaml"
DATA_CONFIG_FILE="$EXECUTION_DIR/configs/data/$DATASET.yaml"
ASSETS_DIR="$EXECUTION_DIR/assets"

CUDA_VISIBLE_DEVICES=$GPUS python $EXECUTION_DIR/scripts/visual_classification/visualize.py  \
    --config $CONFIG_FILE \
    --img-size $DATA_SIZE \
    --pretrained \
    --batch-size 1 \
    --model $BACKBONE \
    --experiment $EXP_NAME \
    --execution_dir $EXECUTION_DIR \
    --data-config $DATA_CONFIG_FILE \
    --method $METHOD \
    ${@:6} \
    "DATA.NAME" $DATASET
