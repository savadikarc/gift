import os
import torch
import argparse
from tqdm import tqdm, trange
from transformers import (
    Trainer,
    AutoConfig,
    AutoTokenizer, 
    AutoModelForCausalLM, 
    AutoModelForSequenceClassification,
    DataCollatorForSeq2Seq,
    DataCollatorWithPadding,
    get_linear_schedule_with_warmup,
    set_seed,
    TrainingArguments
)
from transformers.trainer_utils import EvalPrediction
import wandb
import evaluate
import datetime
import json
import math
import numpy as np
from torch.nn import CrossEntropyLoss
from torch.utils.data import DataLoader
from datasets import Dataset

from task_config import task_config
from dataset import LoReftGLUEDataset, LoReftSupervisedDataset
from compute_metrics import compute_metrics

from pyreft import (
    TaskType,
    get_reft_model,
    ReftConfig,
    ReftTrainerForCausalLM, 
    ReftTrainerForSequenceClassification,
    NoreftIntervention,   # remove ortho.
    LoreftIntervention,
    ConsreftIntervention, # constant bias only
    LobireftIntervention, # low-rank bitfit reft
    DireftIntervention,   # direct edit reft
    NodireftIntervention, # remove ortho + direct edit reft <- this is like LoRA on time-step
    ReftDataCollator
)

from gift.gift import GIFTConfig, GIFTWrapperForCausalLM, BLOCK_PARAMS

device = "cuda" if torch.cuda.is_available() else "cpu"
classification_tasks = {"glue"}
residual_stream_component_mapping = {
    "robertaformaskedlm": "roberta.encoder.layer[%s].output"
}
dtype_mapping = {
    "float32": torch.float32,
    "float16": torch.float16,
    "bfloat16": torch.bfloat16,
    "float8": "float8",
}
intervention_mapping = {
    "NoreftIntervention": NoreftIntervention,
    "LoreftIntervention": LoreftIntervention,
    "ConsreftIntervention": ConsreftIntervention,
    "LobireftIntervention": LobireftIntervention,
    "DireftIntervention": DireftIntervention,
    "NodireftIntervention": NodireftIntervention,
}

def build_gift(backbone, args):

    block_params = {
        k.replace("gift_block_", ""): v for k, v in vars(args).items() if k.startswith("gift_block")
    }
    # Keep only the params that are needed for the current block type
    block_params = {k: v for k, v in block_params.items() if k in BLOCK_PARAMS[block_params["block_type"]].keys()}

    share_projections = args.gift_share_projections and len(args.gift_target_modules) > 1

    # Hack
    enable_gift = None
    if args.gift_enable_gift is not None and "qkv" in args.gift_target_modules:
        enable_gift = {"qkv": [k in args.gift_enable_gift for k in ["q", "k", "v"]]}
        share_projections = args.gift_share_projections and (share_projections or sum(enable_gift["qkv"])>1)
    
    config = GIFTConfig(
        rank=args.gift_rank,
        dtype=args.gift_dtype,
        gift_parameters=block_params,
        in_projection_bias=args.gift_in_projection_bias,
        out_projection_bias=args.gift_out_projection_bias,
        target_modules=args.gift_target_modules,
        enable_gift=enable_gift,
        share_projections=args.gift_share_projections,
    )
    model = GIFTWrapperForCausalLM(
        config,
        backbone, 
    )
    return model


class CustomTrainer(Trainer):

    def create_optimizer(self):
        """
        Setup the optimizer.

        We provide a reasonable default that works well. If you want to use something else, you can pass a tuple in the
        Trainer's init through `optimizers`, or subclass and override this method in a subclass.
        """
        opt_model = self.model

        if self.optimizer is None:
            decay_parameters = self.get_decay_parameter_names(opt_model)
            print([n for n, p in opt_model.named_parameters() if (n in decay_parameters and p.requires_grad)])
            optimizer_grouped_parameters = [
                {
                    "params": [
                        p for n, p in opt_model.named_parameters() if (n in decay_parameters and p.requires_grad)
                    ],
                    "weight_decay": self.args.weight_decay,
                },
                {
                    "params": [
                        p for n, p in opt_model.named_parameters() if (n not in decay_parameters and p.requires_grad)
                    ],
                    "weight_decay": 0.0,
                },
            ]

            optimizer_cls, optimizer_kwargs = CustomTrainer.get_optimizer_cls_and_kwargs(self.args, opt_model)

            # Overwrite `params` in case it's created by `get_optimizer_cls_and_kwargs`
            # e.g. for GaLore optimizer.
            if "params" in optimizer_kwargs:
                optimizer_grouped_parameters = optimizer_kwargs.pop("params")

            # For layer-wise dummy optimizers we overwrite optimizer_grouped_parameters with `optimizer_dict`
            # to avoid arguments conflicts.
            if "optimizer_dict" in optimizer_kwargs:
                optimizer_grouped_parameters = optimizer_kwargs.pop("optimizer_dict")

            self.optimizer = optimizer_cls(optimizer_grouped_parameters, **optimizer_kwargs)

        return self.optimizer


def finetune(
    act_fn: str,
    add_bias: bool,
    model: str,
    layers: str,
    rank: int,
    position: str,
    epochs: int,
    seed: int,
    intervention_type: str,
    max_n_train_example: int,
    max_n_eval_example: int,
    is_wandb: bool,
    wandb_name: str,
    gradient_accumulation_steps: int,
    batch_size: int,
    output_dir: str,
    task: str,
    lr: float,
    schedule: str,
    data_dir: str,
    train_dataset: str,
    eval_dataset: str,
    save_model: bool,
    eval_batch_size: int,
    warmup_ratio: float,
    weight_decay: float,
    dropout: float,
    test_split: str,
    train_on_inputs: bool,
    max_length: int,
    use_normalized_template: bool,
    allow_cls_grad: bool,
    metric_for_best_model: str,
    dtype: str,
    logging_steps: int,
    wandb_dir: str,
    wandb_proj: str,
    share_weights: bool,
    greedy_decoding: bool,
    temperature: float,
    top_p: float,
    top_k: float,
    args,
    **kwargs,
):
    """
    Generic Representation Finetuning.
    """

    assert task in {
        "commonsense", "math", "alpaca", "instruct", "ultrafeedback", "glue", "gsm8k",
        "ultrafeedback_pair"
    }

    dtype = dtype_mapping[dtype]
    
    # store/log run details
    print(
        f"task: {task}, model: {model}, lr: {lr}, weight_decay: {weight_decay}, rank: {args.gift_rank}, "
        f"type: {args.gift_block_block_type}, "
        f"epoch: {epochs}, train_on_inputs: {train_on_inputs}, "
        f"max_length: {max_length}, allow_cls_grad: {allow_cls_grad}"
    )

    # everything is guarded by a single seed
    set_seed(seed)

    model_name = model
    model_str = model.split("/")[-1]
    train_dataset_str = train_dataset
    now = datetime.datetime.now().strftime("%Y%m%d%H%M%S%f")
    if train_dataset is not None:
        run_name = f"{model_str}.{task}.{train_dataset_str}.{test_split}.{now}.{args.gift_rank}.{lr}.{weight_decay}"
    else:
        run_name = f"{model_str}.{task}.{now}.{args.gift_rank}.{lr}.{weight_decay}"

    # which layers to intervene on
    if layers != "all":
        layers = [int(l) for l in layers.split(";")]
    else:
        temp_config = AutoConfig.from_pretrained(model)
        layers = [l for l in range(temp_config.num_hidden_layers)]

    # position str takes the following formats:
    # f1 -> first token; f2 -> first two tokens.
    # f1+l1 -> first and last tokens; f2+l2 -> first and last two tokens.
    # fn or ln shares the same intervention.
    if "+" in position and not share_weights:
        layers += layers

    # load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        model_max_length=max_length,
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
    assert task in task_config, f"Unrecognized task: {task}"
    train_datasets = task_config[task]["train_datasets"] if train_dataset is None else [train_dataset]
    if task == "glue":
        eval_datasets = [train_dataset]
    else:
        eval_datasets = task_config[task]["eval_datasets"] if eval_dataset is None else [eval_dataset]
        
    ReftDataset = LoReftGLUEDataset if task == "glue" else LoReftSupervisedDataset 
    train_dataset = ReftDataset(
        task, train_datasets[0] if task == "glue" or task == "ultrafeedback_pair" \
            else (os.path.join(data_dir, train_datasets[0]) if data_dir is not None else train_datasets[0]), 
        tokenizer, data_split="train", seed=seed, max_n_example=max_n_train_example,
        **{"num_interventions": len(layers), "position": position, 
           "share_weights": share_weights, "test_split": test_split}
    )
    trigger_tokens = train_dataset.trigger_tokens
    num_labels = train_dataset.num_labels

    all_eval_datasets = {}
    for eval_dataset in eval_datasets:
        test_splits = test_split.split(";")
        all_eval_datasets[eval_dataset] = {}
        for split in test_splits:
            raw_eval = ReftDataset(
                task, eval_dataset if task == "glue" else os.path.join(data_dir, eval_dataset), 
                tokenizer, data_split=split, seed=seed, max_n_example=max_n_eval_example,
                **{"num_interventions": len(layers), "position": position, 
                   "share_weights": share_weights}
            )
            all_eval_datasets[eval_dataset][split] = [raw_eval, raw_eval.raw_dataset]
    eval_datasets = all_eval_datasets

    if task == "glue":
        # we repartition the eval_datatsets into [1] 50% validation + [2] 50% test
        # we select the best model on [1] during training
        # we test the selected model on [2] to ensure fairness
        to_split_eval_datasets = eval_datasets[train_dataset_str][test_split][0]
        if len(to_split_eval_datasets) > 5000:
            in_train_n_eval_sample = 1000
        else:
            in_train_n_eval_sample = len(to_split_eval_datasets) // 2

        new_splits = torch.utils.data.random_split(
            to_split_eval_datasets, [len(to_split_eval_datasets)-in_train_n_eval_sample, in_train_n_eval_sample]
        )
        
        in_test_eval_datasets, in_train_eval_datasets = new_splits[0], new_splits[1]
        eval_datasets[train_dataset_str][test_split][0] = in_test_eval_datasets
        print("GLUE validation split (in training): ", len(in_train_eval_datasets))
        print("GLUE validation split (testing): ", len(eval_datasets[train_dataset_str][test_split][0]))

        is_regression = train_dataset_str == "stsb"
        metric = evaluate.load("glue", train_dataset_str)
        # You can define your custom compute_metrics function. It takes an `EvalPrediction` object (a namedtuple with a
        # predictions and label_ids field) and has to return a dictionary string to float.
        def in_training_compute_metrics(p: EvalPrediction):
            preds = p.predictions[0] if isinstance(p.predictions, tuple) else p.predictions
            preds = np.squeeze(preds) if is_regression else np.argmax(preds, axis=1)
            result = metric.compute(predictions=preds, references=p.label_ids)
            if len(result) > 1:
                result["combined_score"] = np.mean(list(result.values())).item()
            return result

    # load model based on task type.
    if task in classification_tasks:
        config = AutoConfig.from_pretrained(
            model, num_labels=num_labels,
            finetuning_task=train_dataset_str,
            load_in_8bit=True if dtype == "float8" else False,
            device_map=device
        )
        # full precision loading since usually for small models
        model = AutoModelForSequenceClassification.from_pretrained(
            model,
            config=config, # just providing the label
            torch_dtype=dtype if dtype != "float8" else None,
            load_in_8bit=True if dtype == "float8" else False,
            device_map=device
        )
    else:
        model = AutoModelForCausalLM.from_pretrained(
            model,
            torch_dtype=dtype if dtype != "float8" else None,  # save memory
            load_in_8bit=True if dtype == "float8" else False,
            device_map=device
        )
        config = model.config
    if need_resize:
        model.resize_token_embeddings(len(tokenizer))

    intervention_type = intervention_mapping[intervention_type]
        
    # select collator based on the type
    if task in classification_tasks:
        data_collator_fn = DataCollatorWithPadding(
            tokenizer=tokenizer,
            padding="longest"
        )
    else:
        data_collator_fn = DataCollatorForSeq2Seq(
            tokenizer=tokenizer,
            model=model,
            label_pad_token_id=-100,
            padding="longest"
        )
    data_collator = ReftDataCollator(data_collator=data_collator_fn)
    
    # reft_config = ReftConfig(representations=representations)
    # wrapped_model = get_reft_model(model, reft_config, set_device=not isinstance(dtype, str))

    # Chinmay:
    model.requires_grad_(False)
    wrapped_model = build_gift(model, args)
    print(wrapped_model)

    num_trainable, percent_trainable = wrapped_model.num_trainable_parameters()
    print(f"Num. trainable parameters: {num_trainable/1e6:.4f}M. Percent of trainable parameters: {percent_trainable:.4f}%")

    # for GLUE tasks, we enable gradients on the classifier head.
    # the parameter will be counted as well.
    if task == "glue" and allow_cls_grad:
        for param in wrapped_model.model.classifier.parameters():
            # wrapped_model with HF trainer will automatically pick up these params to optimize
            param.requires_grad = True

    # train enables dropout but no grads.
    # this line might not be necessary since HF trainer enables this by default.
    wrapped_model.backbone.train()
    n_params = num_trainable # wrapped_model.count_parameters(include_model=False)

    # start wandb logging
    if is_wandb:
        run = wandb.init(
            project=f"{wandb_proj}", 
            entity=wandb_name,
            name=run_name,
            dir=wandb_dir,
        )
        run.summary.update(vars(args))
        wandb.log(
            {"train/n_params": n_params, "train/percent_trainable": percent_trainable})

    # # training args
    training_args = TrainingArguments(
        output_dir=f"{output_dir}/{run_name}",
        run_name=run_name,
        num_train_epochs=epochs,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=eval_batch_size,
        gradient_accumulation_steps=gradient_accumulation_steps,
        evaluation_strategy="epoch" if task == "glue" else "no",
        save_strategy="epoch" if task == "glue" else "no",
        metric_for_best_model=metric_for_best_model if task == "glue" else None,
        load_best_model_at_end=True if task == "glue" else False,
        logging_strategy="steps",
        save_total_limit=1, # for GLUE, it will save 2 at max.
        logging_steps=logging_steps,
        lr_scheduler_type=schedule,
        learning_rate=lr,
        warmup_ratio=warmup_ratio,
        optim="adamw_torch",
        weight_decay=weight_decay,
        report_to="wandb" if is_wandb else "none",
        use_cpu=False if device == "cuda" else True,
        seed=seed,
        # until HF supports ReFT, this remains False! :)
        remove_unused_columns=False
    )

    # make trainer
    trainer_class = ReftTrainerForSequenceClassification \
        if task in classification_tasks else Trainer
    trainer = trainer_class(
        model=wrapped_model,
        tokenizer=tokenizer,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=in_train_eval_datasets if task == "glue" else None,
        data_collator=data_collator,
        compute_metrics=in_training_compute_metrics if task == "glue" else None,
    )
    trainer.train()

    # dump config
    args_dict = vars(args)
    args_dict["n_params"] = n_params
    json_file_name = f"{output_dir}/{run_name}/args.json"
    with open(json_file_name, 'w') as json_file:
        json.dump(args_dict, json_file, indent=4)

    # save model
    if save_model:
        wrapped_model.save(f"{output_dir}/{run_name}")

    # ensure everything is in eval mode
    # wrapped_model.model.eval()
    wrapped_model.backbone.eval()
    # for k,v in wrapped_model.interventions.items():
    #     _ = v[0].eval()

    print({"n_params": n_params})
    # do eval
    eval_results = {}
    for dataset_name in eval_datasets:
        # split evalset into chunks
        for split, (eval_dataset, data_items) in eval_datasets[dataset_name].items():
            
            generations, stats = compute_metrics(
                task, dataset_name, wrapped_model, tokenizer, eval_dataset, data_items,
                trigger_tokens, run_name, eval_batch_size, 
                data_collator if task in classification_tasks else None,
                split, greedy_decoding, temperature, top_p, top_k
            )

            # log
            eval_results.update(stats)
            if is_wandb:
                wandb.log(stats)
            generations = stats if generations is None else generations
            result_json_file_name = f"{output_dir}/{run_name}/{dataset_name}_{split}_outputs.json"
            with open(result_json_file_name, 'w') as json_file:
                json.dump(generations, json_file, indent=4)

    # log final eval stats
    result_json_file_name = f"{output_dir}/{run_name}/eval_results.json"
    eval_results["n_params"] = n_params
    with open(result_json_file_name, 'w') as json_file:
        json.dump(eval_results, json_file, indent=4)

    print(f"Training results can be found in {output_dir}/{run_name}")

def main():
    parser = argparse.ArgumentParser(description="A simple script that takes different arguments.")
    
    parser.add_argument('-task', '--task', type=str, default=None)
    parser.add_argument('-data_dir', '--data_dir', type=str, default="./datasets")
    parser.add_argument('-train_dataset', '--train_dataset', type=str, default=None)
    parser.add_argument('-eval_dataset', '--eval_dataset', type=str, default=None)
    parser.add_argument('-model', '--model', type=str, help='yahma/llama-7b-hf', default='yahma/llama-7b-hf')
    parser.add_argument('-seed', '--seed', type=int, help='42', default=42)
    parser.add_argument('-l', '--layers', type=str, help='2;10;18;26', default='2;10;18;26')
    parser.add_argument('-r', '--rank', type=int, help=8, default=8)
    parser.add_argument('-p', '--position', type=str, help='f1+l1', default='f1+l1')
    parser.add_argument('-e', '--epochs', type=int, help='1', default=1)
    parser.add_argument('-is_wandb', '--is_wandb', action='store_true')
    parser.add_argument('-wandb_name', '--wandb_name', type=str, default="reft")
    parser.add_argument('-save_model', '--save_model', action='store_true')
    parser.add_argument('-max_n_train_example', '--max_n_train_example', type=int, default=None)
    parser.add_argument('-max_n_eval_example', '--max_n_eval_example', type=int, default=None)
    parser.add_argument(
        '-type', '--intervention_type', type=str, 
        help='LoreftIntervention', default="LoreftIntervention")
    parser.add_argument('-gradient_accumulation_steps', '--gradient_accumulation_steps', type=int, default=4)
    parser.add_argument('-batch_size', '--batch_size', type=int, default=4)
    parser.add_argument('-eval_batch_size', '--eval_batch_size', type=int, default=4)
    parser.add_argument('-output_dir', '--output_dir', type=str, default="./official_results")
    parser.add_argument('-lr', '--lr', type=float, default=5e-3)
    parser.add_argument('-schedule', '--schedule', type=str, default='linear')
    parser.add_argument('-wu', '--warmup_ratio', type=float, default=0.00)
    parser.add_argument('-wd', '--weight_decay', type=float, default=0.00)
    parser.add_argument('-dropout', '--dropout', type=float, default=0.00)
    parser.add_argument('-act_fn', '--act_fn', type=str, default=None)
    parser.add_argument('-add_bias', '--add_bias', action='store_true')
    parser.add_argument('-test_split', '--test_split', type=str, default="validation")
    parser.add_argument('-train_on_inputs', '--train_on_inputs', action='store_true')
    parser.add_argument('-max_length', '--max_length', type=int, help=512, default=512)
    parser.add_argument('-nt', '--use_normalized_template', action='store_true')
    parser.add_argument('-allow_cls_grad', '--allow_cls_grad', action='store_true')
    parser.add_argument('-metric_for_best_model', '--metric_for_best_model', type=str, default="accuracy")
    parser.add_argument('-dtype', '--dtype', type=str, default="bfloat16" if device == "cuda" else "float32")
    parser.add_argument('-logging_steps', '--logging_steps', type=int, help=1, default=1)
    parser.add_argument('-wandb_dir', '--wandb_dir', type=str, default='wandb')
    parser.add_argument('-wandb_proj', '--wandb_proj', type=str, default='MyReFT')
    parser.add_argument('-sw', '--share_weights', action='store_true')
    parser.add_argument('-gd', '--greedy_decoding', action='store_true')

    # decoding params
    parser.add_argument('-t', '--temperature', type=float, default=None)
    parser.add_argument('-top_p', '--top_p', type=float, default=None)
    parser.add_argument('-top_k', '--top_k', type=float, default=None)

    # Chinmay
    # Hypernet
    group = parser.add_argument_group("Hypernet parameters")
    group.add_argument(
        "--gift_rank",
        type=int,
        default=16,
        help="Rank r in GIFT.",
    )
    group.add_argument(
        "--gift_dtype",
        type=str,
        default="bfloat16",
        help="dtype for GIFT.",
    )
    group.add_argument(
        "--gift_in_projection_bias",
        action="store_true",
        default=False,
        help="Add bias to the the first linear projection in gift (phi).",
    )
    group.add_argument(
        "--gift_out_projection_bias",
        action="store_true",
        default=False,
        help="Add bias to the the second linear projection in gift (psi).",
    )
    group.add_argument(
        "--gift_target_modules",
        default=["q_proj", "v_proj"],
        type=str,
        nargs="+",
        help="Module to apply finetuning on.",
    )
    group.add_argument(
        "--gift_enable_gift",
        default=None,
        type=str,
        nargs="+",
        help="If target module is a fused layer (qkv in ViT), which modules to apply GIFT to? E.g., for applying GIFT to Q and V, use --gift_enable_gift q v.",
    )
    group.add_argument(
        "--gift_share_projections",
        action="store_true",
        default=False,
        help="Share the linear projection between modules.",
    )
    group = parser.add_argument_group("GIFT Schema Block parameters")
    group.add_argument(
        "--gift_block_block_type",
        type=str,
        default="simple_block",
        choices=["simple_block", "transformer", "pamcat_transformer", "mlp_mixer", "mlp"],
        help="Block type in hypernet.",
    )
    # Transformer Block params
    group.add_argument(
        "--gift_block_num_blocks",
        type=int,
        default=1,
        help="Number of blocks in the chosen GIFT schema.",
    )
    group.add_argument(
        "--gift_block_num_heads",
        type=int,
        default=1,
        help="Number of attention heads in transformer, and pamcat_transformer.",
    )
    group.add_argument(
        "--gift_block_mlp_ratio",
        type=float,
        default=2.,
        help="MLP ratio in transformer, pamcat_transformer, mlp and mlp_mixer",
    )
    group.add_argument(
        "--gift_block_drop_path",
        type=float,
        default=0.,
        help="Drop Path in blocks.",
    )
    group.add_argument(
        "--gift_block_norm_layer",
        type=str,
        default="l2",
        choices=["l2", "none"],
        help="Normalization in the blocks.",
    )
    # PamCat
    group.add_argument(
        "--gift_block_num_clusters",
        type=int,
        default=64,
        help="Number of clusters in pamcat_transformer.",
    )
    group.add_argument(
        "--gift_block_cluster_activation",
        type=str,
        default="sigmoid",
        choices=["sigmoid", "softmax"],
        help="Clustering activation in pamcat_transformer.",
    )
    # MLP Mixer
    group.add_argument(
        "--gift_block_num_mixed_tokens",
        type=int,
        default=64,
        help="Number of mixed tokens in the the token mixing layer of mlp_mixer.",
    )
    group.add_argument(
        "--gift_block_channel_mixing_ratio",
        type=float,
        default=2.,
        help="MLP ratio as in transformers.",
    )
    # Simple down and up
    group.add_argument(
        "--gift_block_act_layer",
        type=str,
        default="identity",
        choices=["identity", "gelu", "sigmoid", ],
        help="Non-Lineraity between down and up layers.",
    )

    args = parser.parse_args()

    finetune(**vars(args), args=args)


if __name__ == "__main__":
    main()