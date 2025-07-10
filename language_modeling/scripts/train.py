import inspect
import os
import warnings
from typing import Dict, Optional
from dataclasses import dataclass

import torch
from transformers import (
    Trainer,
    AutoTokenizer, 
    AutoModelForCausalLM, 
    DataCollatorForSeq2Seq,
    set_seed,
    TrainingArguments,
    HfArgumentParser,
)
from transformers.trainer_utils import is_torch_xla_available
if is_torch_xla_available():
    import torch_xla.core.xla_model as xm

import wandb
import datetime
import json

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


class LLMTrainer(Trainer):

    def _maybe_log_save_evaluate(self, tr_loss, grad_norm, model, trial, epoch, ignore_keys_for_eval):
        if self.control.should_log and self.state.global_step > self._globalstep_last_logged:
            if is_torch_xla_available():
                xm.mark_step()

            logs: Dict[str, float] = {}

            # all_gather + mean() to get average loss over all processes
            tr_loss_scalar = self._nested_gather(tr_loss).mean().item()

            # reset tr_loss to zero
            tr_loss -= tr_loss

            logs["loss"] = round(tr_loss_scalar / (self.state.global_step - self._globalstep_last_logged), 4)
            if grad_norm is not None:
                logs["grad_norm"] = grad_norm.detach().item() if isinstance(grad_norm, torch.Tensor) else grad_norm
            logs["learning_rate"] = self._get_learning_rate()

            # Track GPU memory for the current device
            logs["gpu_memory"] = torch.cuda.max_memory_allocated(device=self.args.device) / (1024.0 * 1024.0 * 1024.0)

            self._total_loss_scalar += tr_loss_scalar
            self._globalstep_last_logged = self.state.global_step
            self.store_flos()

            self.log(logs)

        metrics = None
        if self.control.should_evaluate:
            metrics = self.evaluate(ignore_keys=ignore_keys_for_eval)
            self._report_to_hp_search(trial, self.state.global_step, metrics)

            # Run delayed LR scheduler now that metrics are populated
            if isinstance(self.lr_scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                metric_to_check = self.args.metric_for_best_model
                if not metric_to_check.startswith("eval_"):
                    metric_to_check = f"eval_{metric_to_check}"
                self.lr_scheduler.step(metrics[metric_to_check])

        if self.control.should_save:
            self._save_checkpoint(model, trial, metrics=metrics)
            self.control = self.callback_handler.on_save(self.args, self.state, self.control)


def create_train_datasets(args, tokenizer, dataset_name):
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

    train_dataset = Dataset(
        tokenizer, 
        query_field,
        response_field,
        trigger_tokens,
        prompt_template,
        data_path=args.data_dir_or_branch, 
        seed=args.seed, 
        data_split=args.train_split, 
        validation_percent=args.validation_percent,
        cache_dir=args.cache_dir,
    )

    return train_dataset


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

        raw_eval = Dataset(
            tokenizer, 
            query_field,
            response_field,
            trigger_tokens,
            prompt_template,
            data_path=args.data_dir_or_branch, 
            seed=args.seed, 
            data_split=args.test_split, 
            validation_percent=args.validation_percent,
            cache_dir=args.cache_dir,
        )
        eval_datasets[dataset_name] = [raw_eval, raw_eval.raw_dataset]

    return eval_datasets


@dataclass
class ExtendedTrainingArguments(TrainingArguments):
    task: Optional[str] = None
    data_dir_or_branch: str = "./datasets"
    cache_dir: str = None
    train_dataset: Optional[str] = None
    eval_dataset: Optional[str] = None
    model: str = 'yahma/llama-7b-hf'
    wandb_proj: str = 'wegeft'
    wandb_name: str = 'wegeft'
    wandb_dir: str = 'wandb'

    # Data params
    train_split: str = "train"
    test_split: str = "validation"
    validation_percent: float = 0.1
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
    save_model: bool = False
    max_percent_train_example: Optional[float] = None
    max_percent_eval_example: Optional[float] = None
    allow_cls_grad: bool = False
    attn_implementation: str = "sdpa"
    save_total_limit: int = 1


def finetune(
    args: ExtendedTrainingArguments,
    peft_config=None
):
    """
    Generic Finetuning.
    """
    print("Greedy decoding: ", args.greedy_decoding)
    assert args.task in {
        "commonsense", "math10k", "gsm8k", 
        "boolq",
    }

    dtype = dtype_mapping[args.dtype]

    # everything is guarded by a single seed
    set_seed(args.seed)

    model_name = args.model
    model_str = args.model.split("/")[-1]
    train_dataset_str = args.train_dataset
    now = datetime.datetime.now().strftime("%Y%m%d%H%M%S%f")
    if args.train_dataset is not None:
        run_name = f"{model_str}.{args.task}.{train_dataset_str}.{args.train_split}.{args.test_split}.{args.test_split}.{now}.{args.learning_rate}.{args.seed}"
    else:
        run_name = f"{model_str}.{args.task}.{now}.{args.train_split}.{args.test_split}.{args.learning_rate}.{args.seed}"

    if args.warmup_steps > 0:
        assert args.warmup_ratio == 0., "Cannot specify both warmup_steps and warmup_ratio."
    if args.warmup_ratio > 0.:
        assert args.warmup_steps == 0, "Cannot specify both warmup_steps and warmup_ratio."

    # load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        model_max_length=args.max_length,
        padding_side="left",
        use_fast=False,
    )
    if tokenizer.unk_token is None and tokenizer.pad_token is None:
        # raw llama3
        print("adding a special padding token...")
        tokenizer.add_special_tokens({'pad_token': '[PAD]'})
        need_resize = True
    else:
        tokenizer.pad_token = tokenizer.unk_token
        need_resize = False

    # load dataset splits
    assert args.task in task_config, f"Unrecognized task: {args.task}"
    train_dataset = task_config[args.task]["train_dataset"] if args.train_dataset is None else train_dataset
    eval_datasets = task_config[args.task]["eval_datasets"] if args.eval_dataset is None else [args.eval_dataset]
        
    train_dataset = create_train_datasets(args, tokenizer, train_dataset)
    trigger_tokens = train_dataset.trigger_tokens

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

    data_collator = DataCollatorForSeq2Seq(
        tokenizer=tokenizer,
        model=model,
        label_pad_token_id=-100, # -100 is the default ignore value for torch CE loss
        padding="longest",
    )

    model = prepare_model_for_peft(model, use_gradient_checkpointing=args.gradient_checkpointing, gradient_checkpointing_kwargs={"use_reentrant": False})

    if peft_config is None:
        # Assume full finetuning
        wrapped_model = model
    else:
        wrapped_model: PeftModelForCausalLM = get_peft_model(model, peft_config, adapter_name=f"{args.task}_adapter")

    n_params = sum([p.numel() for p in wrapped_model.parameters() if p.requires_grad])
    num_non_trainable = sum([p.numel() for p in wrapped_model.parameters() if not p.requires_grad])
    percent_trainable = n_params / num_non_trainable * 100

    # num_trainable, percent_trainable = wrapped_model.num_trainable_parameters()
    print(f"Num. trainable parameters: {n_params/1e6:.4f}M. Percent of trainable parameters: {percent_trainable:.4f}%")

    # start wandb logging
    if "wandb" in args.report_to:
        run = wandb.init(
            project=f"{args.wandb_proj}", 
            entity=args.wandb_name,
            name=run_name,
            dir=args.wandb_dir,
        )
        args_dict = args.to_dict()
        run.summary.update(args_dict)
        wandb.log(
            {"train/n_params": n_params, "train/percent_trainable": percent_trainable})

    # # training args
    if dtype==torch.float16 and args.autocast:
        print("Using FP16 with autocast.")
    if dtype==torch.bfloat16 and args.autocast:
        print("Using BF16 with autocast.")
    # Modify some arguments
    args.output_dir = f"{args.output_dir}/{run_name}"
    args.run_name = run_name
    args.fp16 = dtype==torch.float16 and args.autocast
    args.bf16 = dtype==torch.bfloat16 and args.autocast

    # make trainer
    trainer_class = LLMTrainer # Check if we need a different class for cls tasks
    trainer = trainer_class(
        model=wrapped_model,
        tokenizer=tokenizer,
        args=args,
        train_dataset=train_dataset,
        eval_dataset=None,
        data_collator=data_collator,
        compute_metrics=None,
    )
    trainer.train()

    # dump config
    args_dict = args.to_dict()
    args_dict["n_params"] = n_params
    args_dict["percent_trainable"] = percent_trainable
    json_file_name = f"{args.output_dir}/args.json"
    with open(json_file_name, 'w') as json_file:
        json.dump(args_dict, json_file, indent=4)

    # save model
    if args.save_model:
        wrapped_model.save_pretrained(f"{args.output_dir}")

    # ensure everything is in eval mode
    wrapped_model.eval()

    # do eval
    eval_results = {}
    for dataset_name, (eval_dataset, raw_dataset) in eval_datasets.items():

        generation_args = benchmark_config[dataset_name]["generation_args"][args.greedy_decoding]
        print("Generation args: ", generation_args)
        label_trigger_tokens = benchmark_config[dataset_name].get("label_trigger_tokens", None)
        query_field = benchmark_config[dataset_name]["query_field"]
        response_field = benchmark_config[dataset_name]["response_field"]
        evaluator = EVALUATORS[dataset_name](
            dataset_name,
            wrapped_model, 
            tokenizer, 
            eval_dataset, 
            raw_dataset,
            trigger_tokens, 
            label_trigger_tokens,
            query_field,
            response_field,
            batch_size=args.per_device_eval_batch_size, 
            data_collator=data_collator if args.task in classification_tasks else None,
            split=args.test_split, 
        )
        generations, stats = evaluator.evaluate(**generation_args)

        # log
        eval_results.update(stats)
        if "wandb" in args.report_to:
            wandb.log(stats)
        generations = stats if generations is None else generations
        result_json_file_name = f"{args.output_dir}/{dataset_name}_{args.test_split}_outputs.json"
        # Make parent dir
        os.makedirs(os.path.dirname(result_json_file_name), exist_ok=True)
        with open(result_json_file_name, 'w') as json_file:
            json.dump(generations, json_file, indent=4)

    # log final eval stats
    result_json_file_name = f"{args.output_dir}/eval_results.json"
    eval_results["n_params"] = n_params
    eval_results["percent_trainable"] = percent_trainable
    with open(result_json_file_name, 'w') as json_file:
        json.dump(eval_results, json_file, indent=4)

    print(f"Training results can be found in {args.output_dir}/checkpoint")

def main():

    parser = HfArgumentParser(ExtendedTrainingArguments)
    args = parser.parse_args_into_dataclasses()[0]

    finetune(args)


if __name__ == "__main__":
    main()