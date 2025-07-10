# Language Modeling Experiments

## Environment
```sh
conda create -n wegeft-language python=3.11
conda activate wegeft-language
conda install pytorch==2.3.1 torchvision==0.18.1 torchaudio==2.3.1 pytorch-cuda=12.1 -c pytorch -c nvidia
pip install -r requirements.txt

# Install the custom peft package
cd ../peft
pip install -e .
cd ../language_modeling

# For MetaMath, Codefeedback, WizardLM exps
pip install hydra-core==1.3.2
cd external
git clone https://github.com/openai/human-eval
pip install -e human-eval/
cd ..
```

## Math10k and Commonsense Reasoning Benchmarks
### Datasets

Run the following command to load all the datasets:

```bash
cd scripts
./load_datasets.sh
```

### Hyperparameter tuning

```sh
cd scripts
./tune.sh <GPU_ID> <TASK> <LR> <WARMUP_RATIO> <TUNER> [<OPTIONAL_ARGS>]
```
`<GPU_ID>` is the numeric ID of the GPU to be used (e.g., for cuda:0, use `<GPU_ID>` as `0`). We tune the hyperparameters using `yahma/llama-7b-hf`. For tuning the parameters for Math10k benchmark, use `gsm8k` as the <TASK>, and for tuning the parameters for Commonsense Reasoning, use `boolq`.

### Generic Training script
The training script supports multiple tuners and tasks
```bash
cd scripts
./train.sh <GPU_ID> <TASK> <MODEL> <LR> <WARMUP_RATIO> <TUNER> <SEED> [<OPTIONAL_ARGS>]
```
`<TASK>` can be `commonsense`, `math10k`.

`<MODEL>` is any valid model hosted on Hugging Face hub. We experiment with `yahma/llama-7b-hf`, `meta-llama/Llama-2-7b-hf`, and `meta-llama/Meta-Llama-3-8B`.

`<TUNER>` can be `wegeft`, `lora`, `vera`. Additional tuners can be added by writing a custom training script. See [train_wegeft.py](scripts/train_wegeft.py).

### Commonsense reasoning

The following script runs the commonsense reasoning task based on the hyperparameters found in the previous step:

e.g
```sh
# WeGeFT
./train.sh 0 commonsense meta-llama/Llama-2-7b-hf 2e-4 0.1 wegeft 42 --wegeft_rank 64 --wegeft_alpha 128 --wegeft_dropout 0.1

# LoRA
./train.sh 0 commonsense meta-llama/Llama-2-7b-hf 2e-4 0.1 lora 42 --lora_rank 16 --lora_alpha 32 --lora_dropout 0.05

# DoRA
./train.sh 0 commonsense meta-llama/Llama-2-7b-hf 2e-4 0.1 lora 42 --lora_rank 16 --lora_alpha 32 --lora_dropout 0.05 --use_dora

# VeRA
./train.sh 0 commonsense meta-llama/Llama-2-7b-hf 2e-4 0.1 vera 42 --vera_rank 1024 --vera_dropout 0.05
```
We report the average across 3 runs with seeds `42`, `43` and `44`. We use the same hyperparameters as those found for LLaMa-1 for Llama 2 and Llama3.

### Math10k
```sh
# WeGeFT
./train.sh 0 math10k meta-llama/Llama-2-7b-hf 4e-4 0.1 wegeft 42 --wegeft_rank 64 --wegeft_alpha 128 --wegeft_dropout 0.1
```

We report the average across 3 runs with seeds `42`, `43` and `44`. We use the same hyperparameters as those found for LLaMa-1 for Llama 2 and Llama3.

To log using wandb, use arguments `--wandb_proj $PROJECT --wandb_name $WANDB_USER --wandb_dir $WANDB_DIR`

## MetaMathQA, Codefeedback and WizardLM experiments

We build on top of [LoRA-GA](https://github.com/Outsider565/LoRA-GA) for experiments on MetaMathQA, Codefeedback and WizardLM. `math_code_instruct` contains the scripts to run the experiments. Use the following commands to run the experiments:

```sh
cd math_code_instruct
./run_metamath.sh <GPU_ID> <LR> <SEED> wegeft
./run_codefeedback.sh <GPU_ID> <LR> <SEED> wegeft
./run_wizardlm.sh <GPU_ID> <LR> <SEED> wegeft
```
