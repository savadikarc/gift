# https://stackoverflow.com/questions/242538/unix-shell-script-find-out-which-directory-the-script-file-resides
# Absolute path to this script
SCRIPT=$(readlink -f "$0")
SCRIPT_DIR=$(dirname "$SCRIPT")

EXECUTION_DIR="$SCRIPT_DIR/../.."

GPU=$1
LR=$2
MODEL=$3

CUDA_VISIBLE_DEVICES=$GPU python $SCRIPT_DIR/train.py -task gsm8k \
    -data_dir $EXECUTION_DIR/data/language_data \
    --output_dir $EXECUTION_DIR/artifacts/gsm8k/$MODEL \
    -model $MODEL \
    -seed 42 -e 12 -lr $LR \
    -gradient_accumulation_steps 4 \
    -batch_size 8 \
    -eval_batch_size 4 \
    --dropout 0.0 \
    --test_split validation \
    --use_normalized_template \
    --greedy_decoding \
    --warmup_ratio 0.1 \
    --weight_decay 0.0 \
    ${@:4}
