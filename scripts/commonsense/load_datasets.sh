# Taken from https://github.com/stanfordnlp/pyreft/blob/main/examples/loreft/load_datasets.sh
# clone the LLM-Adapters repository (we use the fork used by loreft)
git clone --depth=1 https://github.com/aryamanarora/LLM-Adapters.git
# clone repository for holding ultrafeedback dataset
git clone --depth=1 https://github.com/frankaging/ultrafeedback-dataset.git

DATA_DIR="../../data/language_data"

# move datasets
mv LLM-Adapters/dataset/* $DATA_DIR/
mkdir $DATA_DIR/commonsense_170k
mv LLM-Adapters/ft-training_set/commonsense_170k.json $DATA_DIR/commonsense_170k/train.json
mkdir $DATA_DIR/math_10k
mv LLM-Adapters/ft-training_set/math_10k.json $DATA_DIR/math_10k/train.json
mkdir $DATA_DIR/ultrafeedback
mv ultrafeedback-dataset/train.json $DATA_DIR/ultrafeedback/train.json

# clean
rm -rf LLM-Adapters
rm -rf ultrafeedback-dataset
