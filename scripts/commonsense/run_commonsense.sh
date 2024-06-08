# https://stackoverflow.com/questions/242538/unix-shell-script-find-out-which-directory-the-script-file-resides
# Absolute path to this script
SCRIPT=$(readlink -f "$0")
SCRIPT_DIR=$(dirname "$SCRIPT")

EXECUTION_DIR="$SCRIPT_DIR/../.."

GPU=$1
LR=$2
SEED=$3
MODEL=$4

CUDA_VISIBLE_DEVICES=$GPU python $SCRIPT_DIR/train.py -task commonsense \
    -data_dir $EXECUTION_DIR/data/language_data \
    --output_dir $EXECUTION_DIR/artifacts/commonsense/$MODEL \
    -model $MODEL \
    -seed $SEED \
    -e 6 -lr $LR \
    -gradient_accumulation_steps 4 \
    -batch_size 8 \
    -eval_batch_size 4 \
    --dropout 0.00 \
    --test_split test \
    --use_normalized_template \
    --warmup_ratio 0.1 \
    --greedy_decoding \
    --weight_decay 0.0 \
    ${@:5}
