import inspect
import os
import warnings
from typing import Dict, Optional
from dataclasses import dataclass

import torch
import argparse
from transformers import (
    Trainer,
    AutoConfig,
    AutoTokenizer, 
    AutoModelForCausalLM, 
    AutoModelForSequenceClassification,
    DataCollatorForSeq2Seq,
    DataCollatorWithPadding,
    set_seed,
    TrainingArguments,
    HfArgumentParser,
)
from transformers.trainer_utils import EvalPrediction, is_torch_xla_available
if is_torch_xla_available():
    import torch_xla.core.xla_model as xm

import wandb
import evaluate
import datetime
import json
import numpy as np

from peft import get_peft_model, PeftModelForCausalLM

from simple_peft.task_config import task_config
from simple_peft.benchmark_config import benchmark_config
from simple_peft.datasets import LMDataset, DATASETS
from simple_peft.evaluation import EVALUATORS

device = "cuda" if torch.cuda.is_available() else "cpu"
classification_tasks = {"glue"}
dtype_mapping = {
    "float32": torch.float32,
    "float16": torch.float16,
    "bfloat16": torch.bfloat16,
    "float8": "float8",
}


def prepare_model_for_peft(model, use_gradient_checkpointing=True, gradient_checkpointing_kwargs=None):

    if gradient_checkpointing_kwargs is None:
        gradient_checkpointing_kwargs = {}

    for name, param in model.named_parameters():
        # freeze base model's layers
        param.requires_grad = False

    if use_gradient_checkpointing:
        # When having `use_reentrant=False` + gradient_checkpointing, there is no need for this hack
        if "use_reentrant" not in gradient_checkpointing_kwargs or gradient_checkpointing_kwargs["use_reentrant"]:
            # For backward compatibility
            if hasattr(model, "enable_input_require_grads"):
                model.enable_input_require_grads()
            else:

                def make_inputs_require_grad(module, input, output):
                    output.requires_grad_(True)

                model.get_input_embeddings().register_forward_hook(make_inputs_require_grad)

        # To support older transformers versions, check if the model supports gradient_checkpointing_kwargs
        _supports_gc_kwargs = "gradient_checkpointing_kwargs" in list(
            inspect.signature(model.gradient_checkpointing_enable).parameters
        )

        if not _supports_gc_kwargs and len(gradient_checkpointing_kwargs) > 0:
            warnings.warn(
                "gradient_checkpointing_kwargs is not supported in this version of transformers. The passed kwargs will be ignored."
                " if you want to use that feature, please upgrade to the latest version of transformers.",
                FutureWarning,
            )

        gc_enable_kwargs = (
            {} if not _supports_gc_kwargs else {"gradient_checkpointing_kwargs": gradient_checkpointing_kwargs}
        )
        # enable gradient checkpointing for memory efficiency
        model.gradient_checkpointing_enable(**gc_enable_kwargs)

    return model


def create_eval_datasets(args, tokenizer, dataset_names):
    eval_datasets = {}
    for dataset_name in dataset_names:
        Dataset: LMDataset = DATASETS[dataset_name]
        query_field = benchmark_config[dataset_name]["query_field"]
        response_field = benchmark_config[dataset_name]["response_field"]

        # First, check if the task has a required trigger tokens and prompt template
        trigger_tokens = task_config[args.task].get("trigger_tokens", None)
        prompt_template = task_config[args.task].get("prompt_template", None)

        # If not, use the default ones
        if trigger_tokens is None:
            trigger_tokens = benchmark_config[dataset_name]["trigger_tokens"]
        if prompt_template is None:
            prompt_template = benchmark_config[dataset_name]["prompt_template"]

        dataset_kwargs = {}
        raw_eval = Dataset(
            tokenizer, 
            query_field,
            response_field,
            trigger_tokens,
            prompt_template,
            data_path=args.data_dir_or_branch, 
            seed=args.seed, 
            data_split=args.test_split, 
            max_percent_example=args.max_percent_eval_example,
            validation_percent=args.validation_percent,
            **dataset_kwargs
        )
        eval_datasets[dataset_name] = [raw_eval, raw_eval.raw_dataset]

    return eval_datasets


@dataclass
class ExtendedTrainingArguments(TrainingArguments):
    task: Optional[str] = None
    data_dir_or_branch: str = "./datasets"
    eval_dataset: Optional[str] = None
    model: str = 'yahma/llama-7b-hf'
    wandb_proj: str = 'thorough-peft-internal'
    wandb_name: str = 'thorough-peft-internal'
    wandb_dir: str = 'wandb'
    run_name: str = None
    save_dir: str = None
    tuner: str = None

    # Data params
    test_split: str = "validation"
    validation_percent: float = 0.05
    train_on_inputs: bool = False
    max_length: int = 512
    dtype: str = "bfloat16" if device == "cuda" else "float32"  # Assuming `device` is defined elsewhere

    # Decoding params
    temperature: Optional[float] = None
    top_p: Optional[float] = None
    top_k: Optional[float] = None
    greedy_decoding: bool = True

    # Training params
    autocast: bool = False
    max_percent_eval_example: Optional[float] = None
    allow_cls_grad: bool = False
    attn_implementation: str = "sdpa"
    save_total_limit: int = 1


def finetune(
    args: ExtendedTrainingArguments
):
    """
    Generic Finetuning.
    """
    assert args.task in {
        "commonsense", "math", "gsm8k", "MATH",
        # "alpaca", "instruct", "ultrafeedback", "glue", "gsm8k",
        # "ultrafeedback_pair", "boolq"
    }

    dtype = dtype_mapping[args.dtype]

    # everything is guarded by a single seed
    set_seed(args.seed)

    model_name = args.model

    args.output_dir = f"{args.output_dir}/{args.run_name}"
    args.save_dir = f"{args.save_dir}/{args.run_name}"

    if args.warmup_steps > 0:
        assert args.warmup_ratio == 0., "Cannot specify both warmup_steps and warmup_ratio."
    if args.warmup_ratio > 0.:
        assert args.warmup_steps == 0, "Cannot specify both warmup_steps and warmup_ratio."

    # load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        model_max_length=args.max_length,
        padding_side="right",
        use_fast=False,
    )
    if tokenizer.unk_token == None and tokenizer.pad_token == None:
        # raw llama3
        print("adding a special padding token...")
        tokenizer.add_special_tokens({'pad_token': '[PAD]'})
        need_resize = True
    else:
        tokenizer.pad_token = tokenizer.unk_token
        need_resize = False

    # load dataset splits
    assert args.task in task_config, f"Unrecognized task: {args.task}"
    eval_datasets = task_config[args.task]["eval_datasets"] if args.eval_dataset is None else [args.eval_dataset]

    eval_datasets = create_eval_datasets(args, tokenizer, eval_datasets)

    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype=dtype if dtype != "float8" else None,  # save memory
        load_in_8bit=True if dtype == "float8" else False,
        attn_implementation=args.attn_implementation,
        device_map=device,
    )

    if need_resize:
        model.resize_token_embeddings(len(tokenizer))

    model = prepare_model_for_peft(model, use_gradient_checkpointing=args.gradient_checkpointing, gradient_checkpointing_kwargs={"use_reentrant": False})

    # if peft_config is None:
    #     # Assume full finetuning
    #     wrapped_model = model
    # else:
    wrapped_model: PeftModelForCausalLM = PeftModelForCausalLM.from_pretrained(model, args.output_dir, device_map=device)
    # Merge
    wrapped_model = wrapped_model.merge_and_unload(safe_merge=True)

    n_params = sum([p.numel() for p in wrapped_model.parameters() if p.requires_grad])
    num_non_trainable = sum([p.numel() for p in wrapped_model.parameters() if not p.requires_grad])
    percent_trainable = n_params / num_non_trainable * 100

    print(wrapped_model)

    # start wandb logging
    if "wandb" in args.report_to:
        run = wandb.init(
            project=f"{args.wandb_proj}", 
            entity=args.wandb_name,
            name=args.run_name,
            dir=args.wandb_dir,
        )
        args_dict = args.to_dict()
        run.summary.update(args_dict)
        wandb.log(
            {"train/n_params": n_params, "train/percent_trainable": percent_trainable})

    # training args
    if dtype==torch.float16 and args.autocast:
        print("Using FP16 with autocast.")
    if dtype==torch.bfloat16 and args.autocast:
        print("Using BF16 with autocast.")
    # Modify some arguments
    args.fp16 = dtype==torch.float16 and args.autocast
    args.bf16 = dtype==torch.bfloat16 and args.autocast

    # ensure everything is in eval mode
    wrapped_model.eval()

    # do eval
    eval_results = {}
    for dataset_name, (eval_dataset, raw_dataset) in eval_datasets.items():

        generation_args = benchmark_config[dataset_name]["generation_args"][args.greedy_decoding]
        print("Generation args: ", generation_args)
        trigger_tokens = eval_dataset.trigger_tokens
        label_trigger_tokens = benchmark_config[dataset_name].get("label_trigger_tokens", None)
        query_field = benchmark_config[dataset_name]["query_field"]
        response_field = benchmark_config[dataset_name]["response_field"]
        evaluator = EVALUATORS[dataset_name](
            args.task, 
            dataset_name, 
            wrapped_model, 
            tokenizer, 
            eval_dataset, 
            raw_dataset,
            trigger_tokens, 
            label_trigger_tokens,
            query_field,
            response_field,
            args.run_name, 
            args.per_device_eval_batch_size, 
            None,
            args.test_split, 
        )
        generations, stats = evaluator.evaluate(**generation_args)

        # log
        eval_results.update(stats)
        if "wandb" in args.report_to:
            wandb.log(stats)
        generations = stats if generations is None else generations
        result_json_file_name = f"{args.save_dir}/{dataset_name}_{args.test_split}_outputs.json"
        # Make parent dir
        os.makedirs(os.path.dirname(result_json_file_name), exist_ok=True)
        with open(result_json_file_name, 'w') as json_file:
            json.dump(generations, json_file, indent=4)

    # log final eval stats
    result_json_file_name = f"{args.save_dir}/eval_results.json"
    eval_results["n_params"] = n_params
    eval_results["percent_trainable"] = percent_trainable
    with open(result_json_file_name, 'w') as json_file:
        json.dump(eval_results, json_file, indent=4)

def main():

    parser = HfArgumentParser(ExtendedTrainingArguments)
    args = parser.parse_args_into_dataclasses()[0]

    finetune(args)


if __name__ == "__main__":
    main()