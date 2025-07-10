from .templates import *

task_config = {
    "commonsense": {
        "train_dataset": "commonsense_170k",
        "eval_datasets": [
            "boolq", "piqa", "social_i_qa", "hellaswag", "winogrande", "ARC-Easy", "ARC-Challenge", "openbookqa"
        ],
        "prompt_template": "%s\n", # Needed is task is composed of multiple benchmarks
        "trigger_tokens": "the correct answer is",
    },
    "math10k": {
        "train_dataset": "math10k",
        "eval_datasets": ["gsm8k", "SVAMP", "AQuA", "mawps"],
        "prompt_template": alpaca_prompt_no_input_template,
        "trigger_tokens": "### Response:",
    },
    "metamath": {
        "train_dataset": "meta-math/MetaMathQA",
        "eval_datasets": ["openai/gsm8k"],
        "prompt_template": alpaca_prompt_no_input_template,
        "trigger_tokens": "The answer is",
    },
    "gsm8k": {
        "train_dataset": "openai/gsm8k",
        "eval_datasets": ["openai/gsm8k"],
        "prompt_template": gsm8k_template,
        "trigger_tokens": "First think step by step and then answer the final number.\n",
    },
}

commonsense_datasets = ["boolq", "piqa", "social_i_qa", "hellaswag", "winogrande", "ARC-Easy", "ARC-Challenge", "openbookqa"]
for dataset in commonsense_datasets:
    task_config[dataset] = {
        "train_dataset": dataset,
        "eval_datasets": [dataset],
        "prompt_template": "%s\n", # Needed if task is composed of multiple benchmarks
        "trigger_tokens": "the correct answer is",
    }
