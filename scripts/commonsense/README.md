# Commonsense Reasoning

The code is modified from [LoReFT](https://github.com/stanfordnlp/pyreft/tree/main/examples/loreft). We use the same datasets as LoReFT.

## Datasets

Run the following command to load all the datasets:

```bash
sh load_datasets.sh
```

## Hyperparameter tuning

As described in our Appendix A in the paper, we follow LoReFT and use the last 300 samples from the GSM8k training data to tune the hyperparameters. An example script to tune the hyperparameters:

```sh
./run_gsm8k.sh <GPU_ID> 2.5e-4 meta-llama/Meta-Llama-3-8B --gift_rank 64
```
`<GPU_ID>` is the numeric ID of the GPU to be used (e.g., for cuda:0, use `<GPU_ID>` as `0`). Replace `2.5e-4` with the desired learning rate. The `meta-llama/Meta-Llama-3-8B` is the model you want to use (replace this with `yahma/llama-7b-hf` or `meta-llama/Llama-2-7b-hf` to run with LLaMa-1 Llama 2 respectively). The `--gift_rank 64` is the rank of the GIFT. The range of hyperparameters is given in Table 7 in the Appendix.

## Commonsense reasoning

The following script runs the commonsense reasoning task based on the hyperparameters found in the previous step:
```bash
./run_commonsense.sh <GPU_ID> 2.5e-4 <SEED> meta-llama/Meta-Llama-3-8B --gift_rank 64
```
We report the average across 3 runs with seeds `42`, `43` and `44`. We use the same hyperparameters as those found for LLaMa-1 for Llama 2 and Llama3.
