# Language Modelling Experiments

The code is modified from [LoReFT](https://github.com/stanfordnlp/pyreft/tree/main/examples/loreft). We use the same datasets as LoReFT.

# Datasets

Run the following command to load all the datasets:

```bash
sh load_datasets.sh
```

# Commonsense Reasoning

## Hyperparameter tuning

As described in our Appendix A in the paper, we follow LoReFT and use the last 300 samples from the GSM8k training data to tune the hyperparameters. An example script to tune the hyperparameters:

```sh
./run_gsm8k.sh <GPU_ID> 2.5e-4 meta-llama/Meta-Llama-3-8B --gift_rank 64
```
`<GPU_ID>` is the numeric ID of the GPU to be used (e.g., for cuda:0, use `<GPU_ID>` as `0`). Replace `2.5e-4` with the desired learning rate. The `meta-llama/Meta-Llama-3-8B` is the model you want to use (replace this with `yahma/llama-7b-hf` or `meta-llama/Llama-2-7b-hf` to run with LLaMa-1 Llama 2 respectively). The `--gift_rank 64` is the rank of the GIFT. The range of hyperparameters is given in Table 7 in the Appendix.

The following script runs the commonsense reasoning task based on the hyperparameters found in the previous step:
```bash
./run_commonsense.sh <GPU_ID> 2.5e-4 <SEED> meta-llama/Meta-Llama-3-8B --gift_rank 64
```
We report the average across 3 runs with seeds `42`, `43` and `44`. We use the same hyperparameters as those found for LLaMa-1 for Llama 2 and Llama3.

# Math Reasoning

We use the same hyperparameters as those found on the GSM8k dataset. The following script runs the math reasoning task:
```bash
./run_math.sh <GPU_ID> 2.5e-4 <SEED> yahma/llama-7b-hf --gift_rank 64
```
We report the average across 3 runs with seeds `42`, `43` and `44`.

# Instruction Tuning
Following ReFT, we use Alpaca-52k for hyperparameter tuning for learning rate and rank using LLaMA-1, and use [Alpaca-Eval v1.0](https://github.com/tatsu-lab/alpaca_eval/) for evaluation with GPT 4 Turbo as the annotator. We perform the final runs by training Llama-2 on the Ultrafeedback dataset and evaluate with GPT 4 as the annotator.

An example script to tune the hyperparameters using Alpaca52k:
```sh
./run_alpaca52k.sh <GPU_ID> <LR> <SEED> yahma/llama-7b-hf --gift_rank <RANK>
```
We set the seed to 42 during hyperparameter tuning.

To run instruction tuning with ultrafeedback, use
```sh
./run_ultrafeedback.sh <GPU_ID> <LR> <SEED> meta-llama/Llama-2-7b-hf --gift_rank <RANK>
```
We set the seed to 42 and 43 for the final runs.

## Evaluation using [Alpaca-Eval v1.0](https://github.com/tatsu-lab/alpaca_eval/)
First, install alpaca-eval using
```sh
pip install alpaca-eval
```
Next, download the [generations of text-davinci-003](https://github.com/tatsu-lab/alpaca_eval/blob/main/results/text_davinci_003/model_outputs.json) and save it in the directory `$OPENAI_OUTPUT_DIR/text_davinci_003`.

To evaluate the model using Alpaca-Eval v1.0 with GPT 4 Turbo as the annotator, use
```sh
export OPENAI_API_KEY=<YOUR_OPENAI_KEY>
export IS_ALPACA_EVAL_2=False
alpaca_eval --model_outputs $OUTPUT_DIR/alpaca_eval_test_outputs.json --annotators_config alpaca_eval_gpt4_turbo_fn --reference_outputs $OPENAI_OUTPUT_DIR/text_davinci_003/model_outputs.json
```

To evaluate the model using Alpaca-Eval v1.0 with GPT 4 as the annotator, use
```sh
export OPENAI_API_KEY=<YOUR_OPENAI_KEY>
export IS_ALPACA_EVAL_2=False
alpaca_eval --model_outputs $OUTPUT_DIR/alpaca_eval_test_outputs.json --annotators_config alpaca_eval_gpt4 --reference_outputs $OPENAI_OUTPUT_DIR/text_davinci_003/model_outputs.json
```