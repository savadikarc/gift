from .templates import *

GREEDY_DECODING = True

benchmark_config = {
    "commonsense_170k": {
        "query_field": "instruction",
        "response_field": "output",
        "prompt_template": "%s\n", # Needed is task is composed of multiple benchmarks
        "trigger_tokens": "the correct answer is",
        "generation_args": {
            GREEDY_DECODING: {
                "max_new_tokens": 32,
                "temperature": None,
                "top_p": None,
                "top_k": None,
                "num_beams": None,
                "do_sample": False,
            },
            not GREEDY_DECODING: {
                "max_new_tokens": 32,
                "temperature": 0.1,
                "top_p": 0.75,
                "top_k": 40,
                "num_beams": 1,
                "do_sample": True,
            },
        },
    },
    "math10k": {
        "query_field": "instruction",
        "response_field": "output",
        "prompt_template": alpaca_prompt_no_input_template,
        "trigger_tokens": "### Response:",
        "generation_args": {
            GREEDY_DECODING: {
                "max_new_tokens": 512,
                "temperature": None,
                "top_p": None,
                "top_k": None,
                "num_beams": None,
                "do_sample": False,
            },
            not GREEDY_DECODING: {
                "max_new_tokens": 512,
                "temperature": 0.3,
                "top_p": 0.75,
                "top_k": 40,
                "num_beams": 4,
                "do_sample": True,
            }
        },
    },
    "meta-math/MetaMathQA": {
        "query_field": "query",
        "response_field": "response",
        "prompt_template": alpaca_prompt_no_input_template,
        "trigger_tokens": "The answer is:",
        "generation_args": {
            GREEDY_DECODING: {
                "max_new_tokens": 256,
                "temperature": None,
                "top_p": None,
                "top_k": None,
                "num_beams": None,
                "do_sample": False,
            },
            not GREEDY_DECODING: {
                "max_new_tokens": 256,
                "temperature": 0.3,
                "top_p": 0.75,
                "top_k": 40,
                "num_beams": 4,
                "do_sample": True,
            }
        },
    },
    "openai/gsm8k": {
        "query_field": "question",
        "response_field": "answer",
        "prompt_template": gsm8k_template,
        "trigger_tokens": "First think step by step and then answer the final number.\n",
        "label_trigger_tokens": "####",
        "generation_args": {
            GREEDY_DECODING: {
                "max_new_tokens": 512,
                "temperature": None,
                "top_p": None,
                "top_k": None,
                "num_beams": None,
                "do_sample": False,
            },
            not GREEDY_DECODING: {
                "max_new_tokens": 512,
                "temperature": 0.3,
                "top_p": 0.75,
                "top_k": 40,
                "num_beams": 4,
                "do_sample": True,
            }
        },
    },
}

commonsense_170k_eval_sets = ["boolq", "piqa", "social_i_qa", "hellaswag", "winogrande", "ARC-Easy", "ARC-Challenge", "openbookqa"]
for eval_set in commonsense_170k_eval_sets:
    benchmark_config[eval_set] = benchmark_config["commonsense_170k"]

for dataset in ["gsm8k", "SVAMP", "AQuA", "mawps"]:
    benchmark_config[dataset] = {
        "query_field": "instruction",
        "response_field": "answer",
        "prompt_template": alpaca_prompt_no_input_template,
        "trigger_tokens": "### Response:",
        "generation_args": {
            GREEDY_DECODING: {
                "max_new_tokens": 512,
                "temperature": None,
                "top_p": None,
                "top_k": None,
                "num_beams": None,
                "do_sample": False,
            },
            not GREEDY_DECODING: {
                "max_new_tokens": 512,
                "temperature": 0.3,
                "top_p": 0.75,
                "top_k": 40,
                "num_beams": 4,
                "do_sample": True,
            }
        },
    }
