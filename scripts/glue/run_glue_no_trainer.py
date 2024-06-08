# coding=utf-8
# Copyright 2021 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
""" Finetuning a 🤗 Transformers model for sequence classification on GLUE."""
from typing import List, Dict
import csv
from collections import OrderedDict
import operator
import glob
from datetime import datetime

import argparse
import json
import logging
import math
import os
import random
from pathlib import Path

import datasets
import evaluate
import torch
from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import set_seed
from datasets import load_dataset
from huggingface_hub import HfApi
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

import transformers
from transformers import (
    AutoConfig,
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
    PretrainedConfig,
    SchedulerType,
    default_data_collator,
    get_scheduler,
)
from transformers.utils import check_min_version, send_example_telemetry
from transformers.utils.versions import require_version

try:
    from timm.utils.log import setup_default_logging
except ImportError:
    setup_default_logging = None

from gift.gift import GIFTConfig, GIFTWrapperForSeqClassification, BLOCK_PARAMS


# Will error if the minimal version of Transformers is not installed. Remove at your own risks.
check_min_version("4.40.0")

logger = get_logger(__name__)

require_version("datasets>=1.8.0", "To fix: pip install -r examples/pytorch/text-classification/requirements.txt")

task_to_keys = {
    "cola": ("sentence", None),
    "mnli": ("premise", "hypothesis"),
    "mrpc": ("sentence1", "sentence2"),
    "qnli": ("question", "sentence"),
    "qqp": ("question1", "question2"),
    "rte": ("sentence1", "sentence2"),
    "sst2": ("sentence", None),
    "stsb": ("sentence1", "sentence2"),
    "wnli": ("sentence1", "sentence2"),
}


def parse_args():
    parser = argparse.ArgumentParser(description="Finetune a transformers model on a text classification task")
    parser.add_argument(
        "--task_name",
        type=str,
        default=None,
        help="The name of the glue task to train on.",
        choices=list(task_to_keys.keys()),
    )
    parser.add_argument(
        "--train_file", type=str, default=None, help="A csv or a json file containing the training data."
    )
    parser.add_argument(
        "--data_cache_dir", type=str, default=None, help="Data cache dir."
    )
    parser.add_argument(
        "--validation_file", type=str, default=None, help="A csv or a json file containing the validation data."
    )
    parser.add_argument(
        "--max_length",
        type=int,
        default=128,
        help=(
            "The maximum total input sequence length after tokenization. Sequences longer than this will be truncated,"
            " sequences shorter will be padded if `--pad_to_max_length` is passed."
        ),
    )
    parser.add_argument(
        "--pad_to_max_length",
        action="store_true",
        help="If passed, pad all samples to `max_length`. Otherwise, dynamic padding is used.",
    )
    parser.add_argument(
        "--model_name_or_path",
        type=str,
        help="Path to pretrained model or model identifier from huggingface.co/models.",
        required=True,
    )
    parser.add_argument(
        "--model_cache_dir",
        type=str,
        help="Model cache dir.",
        default=None,
    )
    parser.add_argument(
        "--use_slow_tokenizer",
        action="store_true",
        help="If passed, will use a slow tokenizer (not backed by the 🤗 Tokenizers library).",
    )
    parser.add_argument(
        "--per_device_train_batch_size",
        type=int,
        default=8,
        help="Batch size (per device) for the training dataloader.",
    )
    parser.add_argument(
        "--per_device_eval_batch_size",
        type=int,
        default=8,
        help="Batch size (per device) for the evaluation dataloader.",
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=5e-5,
        help="Initial learning rate (after the potential warmup period) to use.",
    )
    parser.add_argument(
        "--classifier_learning_rate",
        type=float,
        default=5e-5,
        help="Initial learning rate (after the potential warmup period) to use.",
    )
    parser.add_argument("--weight_decay", type=float, default=0.0, help="Weight decay to use.")
    parser.add_argument("--classifier_weight_decay", type=float, default=0.0, help="Weight decay to use.")
    parser.add_argument("--num_train_epochs", type=int, default=3, help="Total number of training epochs to perform.")
    parser.add_argument(
        "--max_train_steps",
        type=int,
        default=None,
        help="Total number of training steps to perform. If provided, overrides num_train_epochs.",
    )
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=1,
        help="Number of updates steps to accumulate before performing a backward/update pass.",
    )
    parser.add_argument(
        "--lr_scheduler_type",
        type=SchedulerType,
        default="linear",
        help="The scheduler type to use.",
        choices=["linear", "cosine", "cosine_with_restarts", "polynomial", "constant", "constant_with_warmup"],
    )
    parser.add_argument(
        "--warmup_ratio", type=float, default=0, help="Number of steps for the warmup in the lr scheduler."
    )
    parser.add_argument("--output_dir", type=str, default=None, help="Where to store the final model.")
    parser.add_argument("--seed", type=int, default=42, help="A seed for reproducible training.")
    parser.add_argument("--push_to_hub", action="store_true", help="Whether or not to push the model to the Hub.")
    parser.add_argument(
        "--hub_model_id", type=str, help="The name of the repository to keep in sync with the local `output_dir`."
    )
    parser.add_argument("--hub_token", type=str, help="The token to use to push to the Model Hub.")
    parser.add_argument(
        "--trust_remote_code",
        type=bool,
        default=False,
        help=(
            "Whether or not to allow for custom models defined on the Hub in their own modeling files. This option "
            "should only be set to `True` for repositories you trust and in which you have read the code, as it will "
            "execute code present on the Hub on your local machine."
        ),
    )
    parser.add_argument(
        "--checkpointing_steps",
        type=str,
        default=None,
        help="Whether the various states should be saved at the end of every n steps, or 'epoch' for each epoch.",
    )
    parser.add_argument(
        "--resume_from_checkpoint",
        type=str,
        default=None,
        help="If the training should continue from a checkpoint folder.",
    )
    parser.add_argument(
        "--with_tracking",
        action="store_true",
        help="Whether to enable experiment trackers for logging.",
    )
    parser.add_argument(
        "--do_predict",
        action="store_true",
        help="Do predictions on the test set for evaluation.",
    )
    parser.add_argument(
        "--report_to",
        type=str,
        default="all",
        help=(
            'The integration to report the results and logs to. Supported platforms are `"tensorboard"`,'
            ' `"wandb"`, `"comet_ml"` and `"clearml"`. Use `"all"` (default) to report to all integrations. '
            "Only applicable when `--with_tracking` is passed."
        ),
    )
    parser.add_argument(
        "--ignore_mismatched_sizes",
        action="store_true",
        help="Whether or not to enable to load a pretrained model whose head dimensions are different.",
    )
    parser.add_argument(
        "--hyperparameter_search",
        action="store_true",
        help="Is the script being run for hyperparameter search?",
    )
    parser.add_argument(
        "--override_test_hyperparams",
        action="store_true",
        help="Is the script being run for hyperparameter search?",
    )
    # GIFT
    group = parser.add_argument_group("GIFT parameters")
    group.add_argument(
        "--gift_rank",
        type=int,
        default=32,
        help="Rank r in GIFT.",
    )
    group.add_argument(
        "--gift_dtype",
        type=str,
        default="float32",
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
        default=["query", "value"],
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

    # Sanity checks
    if args.task_name is None and args.train_file is None and args.validation_file is None:
        raise ValueError("Need either a task name or a training/validation file.")
    else:
        if args.train_file is not None:
            extension = args.train_file.split(".")[-1]
            assert extension in ["csv", "json"], "`train_file` should be a csv or a json file."
        if args.validation_file is not None:
            extension = args.validation_file.split(".")[-1]
            assert extension in ["csv", "json"], "`validation_file` should be a csv or a json file."

    if args.push_to_hub:
        assert args.output_dir is not None, "Need an `output_dir` to create a repo when `--push_to_hub` is passed."

    return args


class AverageMeter:
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

def update_summary(
        epoch,
        train_metrics,
        eval_metrics,
        filename,
        lr=None,
        write_header=False,
        log_wandb=False,
):
    rowd = OrderedDict(epoch=epoch)
    rowd.update([('train_' + k, v) for k, v in train_metrics.items()])
    if eval_metrics:
        rowd.update([('eval_' + k, v) for k, v in eval_metrics.items()])
    if lr is not None:
        rowd['lr'] = lr
    if log_wandb:
        wandb.log(rowd)
    with open(filename, mode='a') as cf:
        dw = csv.DictWriter(cf, fieldnames=rowd.keys())
        if write_header:  # first iteration (epoch == 1 can't be used)
            dw.writeheader()
        dw.writerow(rowd)


def get_state_dict(model, unwrap_fn):
    return unwrap_fn(model).state_dict()


class CheckpointSaver:
    def __init__(
            self,
            model,
            args=None,
            model_ema=None,
            amp_scaler=None,
            checkpoint_prefix='checkpoint',
            recovery_prefix='recovery',
            checkpoint_dir='',
            recovery_dir='',
            decreasing=False,
            max_history=10,
            unwrap_fn=None):

        # objects to save state_dicts of
        self.model = model
        self.args = args
        self.model_ema = model_ema
        self.amp_scaler = amp_scaler

        # state
        self.checkpoint_files = []  # (filename, metric) tuples in order of decreasing betterness
        self.best_epoch = None
        self.best_metric = None
        self.curr_recovery_file = ''
        self.last_recovery_file = ''

        # config
        self.checkpoint_dir = checkpoint_dir
        self.recovery_dir = recovery_dir
        self.save_prefix = checkpoint_prefix
        self.recovery_prefix = recovery_prefix
        self.extension = '.pth.tar'
        self.decreasing = decreasing  # a lower metric is better if True
        self.cmp = operator.lt if decreasing else operator.gt  # True if lhs better than rhs
        self.max_history = max_history
        self.unwrap_fn = unwrap_fn
        assert self.max_history >= 1

    def save_checkpoint(self, epoch, metric=None):
        assert epoch >= 0
        tmp_save_path = os.path.join(self.checkpoint_dir, 'tmp' + self.extension)
        last_save_path = os.path.join(self.checkpoint_dir, 'last' + self.extension)
        self._save(tmp_save_path, epoch, metric)
        if os.path.exists(last_save_path):
            os.unlink(last_save_path)  # required for Windows support.
        os.rename(tmp_save_path, last_save_path)
        worst_file = self.checkpoint_files[-1] if self.checkpoint_files else None
        if (len(self.checkpoint_files) < self.max_history
                or metric is None or self.cmp(metric, worst_file[1])):
            if len(self.checkpoint_files) >= self.max_history:
                self._cleanup_checkpoints(1)
            filename = '-'.join([self.save_prefix, str(epoch)]) + self.extension
            save_path = os.path.join(self.checkpoint_dir, filename)
            os.link(last_save_path, save_path)
            self.checkpoint_files.append((save_path, metric))
            self.checkpoint_files = sorted(
                self.checkpoint_files, key=lambda x: x[1],
                reverse=not self.decreasing)  # sort in descending order if a lower metric is not better

            checkpoints_str = "Current checkpoints:\n"
            for c in self.checkpoint_files:
                checkpoints_str += ' {}\n'.format(c)
            logger.info(checkpoints_str)

            if metric is not None and (self.best_metric is None or self.cmp(metric, self.best_metric)):
                self.best_epoch = epoch
                self.best_metric = metric
                best_save_path = os.path.join(self.checkpoint_dir, 'model_best' + self.extension)
                if os.path.exists(best_save_path):
                    os.unlink(best_save_path)
                os.link(last_save_path, best_save_path)

        return (None, None) if self.best_metric is None else (self.best_metric, self.best_epoch)

    def _save(self, save_path, epoch, metric=None):
        state_dict = get_state_dict(self.model, self.unwrap_fn)
        torch.save(state_dict, save_path)

    def _cleanup_checkpoints(self, trim=0):
        trim = min(len(self.checkpoint_files), trim)
        delete_index = self.max_history - trim
        if delete_index < 0 or len(self.checkpoint_files) <= delete_index:
            return
        to_delete = self.checkpoint_files[delete_index:]
        for d in to_delete:
            try:
                logger.debug("Cleaning checkpoint: {}".format(d))
                os.remove(d[0])
            except Exception as e:
                logger.error("Exception '{}' while deleting checkpoint".format(e))
        self.checkpoint_files = self.checkpoint_files[:delete_index]

    def save_recovery(self, epoch, batch_idx=0):
        assert epoch >= 0
        filename = '-'.join([self.recovery_prefix, str(epoch), str(batch_idx)]) + self.extension
        save_path = os.path.join(self.recovery_dir, filename)
        self._save(save_path, epoch)
        if os.path.exists(self.last_recovery_file):
            try:
                logger.debug("Cleaning recovery: {}".format(self.last_recovery_file))
                os.remove(self.last_recovery_file)
            except Exception as e:
                logger.error("Exception '{}' while removing {}".format(e, self.last_recovery_file))
        self.last_recovery_file = self.curr_recovery_file
        self.curr_recovery_file = save_path

    def find_recovery(self):
        recovery_path = os.path.join(self.recovery_dir, self.recovery_prefix)
        files = glob.glob(recovery_path + '*' + self.extension)
        files = sorted(files)
        return files[0] if len(files) else ''


def get_best_hyperparameters(val_output_dir):
    
    # Get all directories from the val dir
    val_dirs = list(Path(val_output_dir).glob("*"))
    val_dirs = [d for d in val_dirs if d.is_dir()]

    metrics = []
    for d in val_dirs:
        run_name = d.stem
        metadata = {"run_name": run_name}
        with open(d / "best_metric.json") as f:
            metric = json.load(f)
            metadata.update(metric)
        with open(d / "run_args.json") as f:
            arguments = json.load(f)
            # Get the learning_rate, classifier_learning_rate, weight_decay, classifier_weight_decay from arguments dict
            hyperparameters = {
                "learning_rate": arguments["learning_rate"], 
                "classifier_learning_rate": arguments["classifier_learning_rate"], 
                "weight_decay": arguments["weight_decay"], 
                "classifier_weight_decay": arguments["classifier_weight_decay"]
            }
            metadata.update(hyperparameters)

        metrics.append(metadata)

    # Sort the metrics based on the validation accuracy
    metrics = sorted(metrics, key=lambda x: x["best_metric"], reverse=True)
    return metrics[0]


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
    model = GIFTWrapperForSeqClassification(
        config,
        backbone, 
    )
    return model


def main():
    args = parse_args()
    # Sending telemetry. Tracking the example usage helps us better allocate resources to maintain them. The
    # information sent is the one passed as arguments along with your Python/PyTorch versions.
    send_example_telemetry("run_glue_no_trainer", args)

    args.output_dir = args.output_dir + f"/{args.model_name_or_path.split('/')[-1]}/embed_{args.gift_rank}"
    val_output_dir = os.path.join(args.output_dir, "val")
    test_output_dir = os.path.join(args.output_dir, "test")

    if not args.hyperparameter_search and args.override_test_hyperparams:
        best_hyperparameters = get_best_hyperparameters(val_output_dir)
        args.learning_rate = best_hyperparameters["learning_rate"]
        args.classifier_learning_rate = best_hyperparameters["classifier_learning_rate"]
        args.weight_decay = best_hyperparameters["weight_decay"]
        args.classifier_weight_decay = best_hyperparameters["classifier_weight_decay"]

    # split_str = "val" if args.hyperparameter_search else "test"
    args.output_dir = val_output_dir if args.hyperparameter_search else test_output_dir
    settings_str = f"lr_{args.learning_rate}_clslr_{args.classifier_learning_rate}_wd_{args.weight_decay}"
    aux_str = f"_{datetime.now().strftime('%Y%m%d-%H%M%S')}"
    args.output_dir = os.path.join(args.output_dir, settings_str+aux_str)

    # Initialize the accelerator. We will let the accelerator handle device placement for us in this example.
    # If we're using tracking, we also need to initialize it here and it will by default pick up all supported trackers
    # in the environment
    accelerator = (
        Accelerator(log_with=args.report_to, project_dir=args.output_dir) if args.with_tracking else Accelerator()
    )
    # Make one log on every process with the configuration for debugging.
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )
    logger.info(accelerator.state, main_process_only=False)
    if accelerator.is_local_main_process:
        datasets.utils.logging.set_verbosity_warning()
        transformers.utils.logging.set_verbosity_info()
    else:
        datasets.utils.logging.set_verbosity_error()
        transformers.utils.logging.set_verbosity_error()

    # If passed along, set the training seed now.
    if args.seed is not None:
        set_seed(args.seed)

    # Handle the repository creation
    if accelerator.is_main_process:
        if args.push_to_hub:
            # Retrieve of infer repo_name
            repo_name = args.hub_model_id
            if repo_name is None:
                repo_name = Path(args.output_dir).absolute().name
            # Create repo and retrieve repo_id
            api = HfApi()
            repo_id = api.create_repo(repo_name, exist_ok=True, token=args.hub_token).repo_id

            with open(os.path.join(args.output_dir, ".gitignore"), "w+") as gitignore:
                if "step_*" not in gitignore:
                    gitignore.write("step_*\n")
                if "epoch_*" not in gitignore:
                    gitignore.write("epoch_*\n")
        elif args.output_dir is not None:
            os.makedirs(args.output_dir, exist_ok=True)
            # Save the args in a json file
            with open(os.path.join(args.output_dir, "run_args.json"), "w") as f:
                json.dump(vars(args), f)
    accelerator.wait_for_everyone()

    if accelerator.is_main_process and setup_default_logging is not None:
        log_path = os.path.join(args.output_dir, "train.log")
        setup_default_logging(log_path=log_path)

    # Get the datasets: you can either provide your own CSV/JSON training and evaluation files (see below)
    # or specify a GLUE benchmark task (the dataset will be downloaded automatically from the datasets Hub).

    # For CSV/JSON files, this script will use as labels the column called 'label' and as pair of sentences the
    # sentences in columns called 'sentence1' and 'sentence2' if such column exists or the first two columns not named
    # label if at least two columns are provided.

    # If the CSVs/JSONs contain only one non-label column, the script does single sentence classification on this
    # single column. You can easily tweak this behavior (see below)

    # In distributed training, the load_dataset function guarantee that only one local process can concurrently
    # download the dataset.
    if args.task_name is not None:
        # Downloading and loading a dataset from the hub.
        raw_datasets = load_dataset("nyu-mll/glue", args.task_name, cache_dir=args.data_cache_dir)
    else:
        # Loading the dataset from local csv or json file.
        data_files = {}
        if args.train_file is not None:
            data_files["train"] = args.train_file
        if args.validation_file is not None:
            data_files["validation"] = args.validation_file
        extension = (args.train_file if args.train_file is not None else args.validation_file).split(".")[-1]
        raw_datasets = load_dataset(extension, data_files=data_files)
    # See more about loading any type of standard or custom dataset at
    # https://huggingface.co/docs/datasets/loading_datasets.

    # Labels
    if args.task_name is not None:
        is_regression = args.task_name == "stsb"
        if not is_regression:
            label_list = raw_datasets["train"].features["label"].names
            num_labels = len(label_list)
        else:
            num_labels = 1
    else:
        # Trying to have good defaults here, don't hesitate to tweak to your needs.
        is_regression = raw_datasets["train"].features["label"].dtype in ["float32", "float64"]
        if is_regression:
            num_labels = 1
        else:
            # A useful fast method:
            # https://huggingface.co/docs/datasets/package_reference/main_classes.html#datasets.Dataset.unique
            label_list = raw_datasets["train"].unique("label")
            label_list.sort()  # Let's sort it for determinism
            num_labels = len(label_list)

    # Load pretrained model and tokenizer
    #
    # In distributed training, the .from_pretrained methods guarantee that only one local process can concurrently
    # download model & vocab.
    config = AutoConfig.from_pretrained(
        args.model_name_or_path,
        num_labels=num_labels,
        finetuning_task=args.task_name,
        trust_remote_code=args.trust_remote_code,
        cache_dir=args.model_cache_dir,
    )
    tokenizer = AutoTokenizer.from_pretrained(
        args.model_name_or_path, 
        use_fast=not args.use_slow_tokenizer, 
        trust_remote_code=args.trust_remote_code,
        cache_dir=args.model_cache_dir,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    config.pad_token_id = tokenizer.pad_token_id
    model = AutoModelForSequenceClassification.from_pretrained(
        args.model_name_or_path,
        from_tf=bool(".ckpt" in args.model_name_or_path),
        config=config,
        ignore_mismatched_sizes=args.ignore_mismatched_sizes,
        trust_remote_code=args.trust_remote_code,
        cache_dir=args.model_cache_dir,
    )
    model.requires_grad_(False)
    model.classifier.requires_grad_(True)

    # Initialize GIFT
    wrapped_model = build_gift(model, args)

    logger.info(f"{wrapped_model}")
    num_trainable, percent_trainable = wrapped_model.num_trainable_parameters()
    logger.info(f"Num. trainable parameters: {num_trainable/1e6:.4f}M. Percent of trainable parameters: {percent_trainable:.4f}%")

    # Preprocessing the datasets
    if args.task_name is not None:
        sentence1_key, sentence2_key = task_to_keys[args.task_name]
    else:
        # Again, we try to have some nice defaults but don't hesitate to tweak to your use case.
        non_label_column_names = [name for name in raw_datasets["train"].column_names if name != "label"]
        if "sentence1" in non_label_column_names and "sentence2" in non_label_column_names:
            sentence1_key, sentence2_key = "sentence1", "sentence2"
        else:
            if len(non_label_column_names) >= 2:
                sentence1_key, sentence2_key = non_label_column_names[:2]
            else:
                sentence1_key, sentence2_key = non_label_column_names[0], None

    # Some models have set the order of the labels to use, so let's make sure we do use it.
    label_to_id = None
    if (
        model.config.label2id != PretrainedConfig(num_labels=num_labels).label2id
        and args.task_name is not None
        and not is_regression
    ):
        # Some have all caps in their config, some don't.
        label_name_to_id = {k.lower(): v for k, v in model.config.label2id.items()}
        if sorted(label_name_to_id.keys()) == sorted(label_list):
            logger.info(
                f"The configuration of the model provided the following label correspondence: {label_name_to_id}. "
                "Using it!"
            )
            label_to_id = {i: label_name_to_id[label_list[i]] for i in range(num_labels)}
        else:
            logger.warning(
                "Your model seems to have been trained with labels, but they don't match the dataset: ",
                f"model labels: {sorted(label_name_to_id.keys())}, dataset labels: {sorted(label_list)}."
                "\nIgnoring the model labels as a result.",
            )
    elif args.task_name is None and not is_regression:
        label_to_id = {v: i for i, v in enumerate(label_list)}

    if label_to_id is not None:
        model.config.label2id = label_to_id
        model.config.id2label = {id: label for label, id in config.label2id.items()}
    elif args.task_name is not None and not is_regression:
        model.config.label2id = {l: i for i, l in enumerate(label_list)}
        model.config.id2label = {id: label for label, id in config.label2id.items()}

    padding = "max_length" if args.pad_to_max_length else False

    def preprocess_function(examples):
        # Tokenize the texts
        texts = (
            (examples[sentence1_key],) if sentence2_key is None else (examples[sentence1_key], examples[sentence2_key])
        )
        result = tokenizer(*texts, padding=padding, max_length=args.max_length, truncation=True)

        if "label" in examples:
            if label_to_id is not None:
                # Map labels to IDs (not necessary for GLUE tasks)
                result["labels"] = [label_to_id[l] for l in examples["label"]]
            else:
                # In all cases, rename the column to labels because the model will expect that.
                result["labels"] = examples["label"]
        return result

    with accelerator.main_process_first():
        processed_datasets = raw_datasets.map(
            preprocess_function,
            batched=True,
            remove_columns=raw_datasets["train"].column_names,
            desc="Running tokenizer on dataset",
        )

    train_dataset = processed_datasets["train"]
    eval_dataset = processed_datasets["validation_matched" if args.task_name == "mnli" else "validation"]
    if args.do_predict:
        predict_dataset = processed_datasets["test_matched" if args.task_name == "mnli" else "test"]

    # Log a few random samples from the training set:
    for index in random.sample(range(len(train_dataset)), 3):
        logger.info(f"Sample {index} of the training set: {train_dataset[index]}.")
    # Log a few random samples from the validation set:
    for index in random.sample(range(len(eval_dataset)), 3):
        logger.info(f"Sample {index} of the validation set: {eval_dataset[index]}.")
    if args.do_predict:
        # Log a few random samples from the test set:
        for index in random.sample(range(len(predict_dataset)), 3):
            logger.info(f"Sample {index} of the test set: {predict_dataset[index]}.")

    # DataLoaders creation:
    if args.pad_to_max_length:
        # If padding was already done ot max length, we use the default data collator that will just convert everything
        # to tensors.
        data_collator = default_data_collator
    else:
        # Otherwise, `DataCollatorWithPadding` will apply dynamic padding for us (by padding to the maximum length of
        # the samples passed). When using mixed precision, we add `pad_to_multiple_of=8` to pad all tensors to multiple
        # of 8s, which will enable the use of Tensor Cores on NVIDIA hardware with compute capability >= 7.5 (Volta).
        data_collator = DataCollatorWithPadding(tokenizer, pad_to_multiple_of=(8 if accelerator.use_fp16 else None))

    train_dataloader = DataLoader(
        train_dataset, shuffle=True, collate_fn=data_collator, batch_size=args.per_device_train_batch_size
    )
    eval_dataloader = DataLoader(eval_dataset, collate_fn=data_collator, batch_size=args.per_device_eval_batch_size)
    if args.do_predict:
        predict_dataloader = DataLoader(predict_dataset, collate_fn=data_collator, batch_size=args.per_device_eval_batch_size)

    # Optimizer
    # Split weights in two groups, one with weight decay and the other not.
    no_decay = ["bias", "LayerNorm.weight"]
    logger.info(f"WD: {args.weight_decay}, Cls WD: {args.classifier_weight_decay}")
    optimizer_grouped_parameters = [
        {
            "params": [p for n, p in wrapped_model.gift_named_parameters() if not any(nd in n for nd in no_decay)],
            "lr": args.learning_rate,
            "weight_decay": args.weight_decay,
        },
        {
            "params": [p for n, p in wrapped_model.gift_named_parameters() if any(nd in n for nd in no_decay)],
            "lr": args.learning_rate,
            "weight_decay": 0.0,
        },
        {
            "params": [p for n, p in wrapped_model.classifier_named_parameters() if not any(nd in n for nd in no_decay)],
            "lr": args.classifier_learning_rate,
            "weight_decay": args.classifier_weight_decay,
        },
        {
            "params": [p for n, p in wrapped_model.classifier_named_parameters() if any(nd in n for nd in no_decay)],
            "lr": args.classifier_learning_rate,
            "weight_decay": 0.0,
        },
        
    ]
    optimizer = torch.optim.AdamW(optimizer_grouped_parameters, lr=args.learning_rate)

    # Scheduler and math around the number of training steps.
    overrode_max_train_steps = False
    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args.gradient_accumulation_steps)
    if args.max_train_steps is None:
        args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
        overrode_max_train_steps = True

    num_warmup_steps = int(args.warmup_ratio * args.max_train_steps)

    lr_scheduler = get_scheduler(
        name=args.lr_scheduler_type,
        optimizer=optimizer,
        num_warmup_steps=num_warmup_steps,
        num_training_steps=args.max_train_steps,
    )

    # Prepare everything with our `accelerator`.
    if not args.do_predict:
        wrapped_model, optimizer, train_dataloader, eval_dataloader, lr_scheduler = accelerator.prepare(
            wrapped_model, optimizer, train_dataloader, eval_dataloader, lr_scheduler
        )
    else:
        wrapped_model, optimizer, train_dataloader, eval_dataloader, predict_dataloader, lr_scheduler = accelerator.prepare(
            wrapped_model, optimizer, train_dataloader, eval_dataloader, predict_dataloader, lr_scheduler
        )

    # We need to recalculate our total training steps as the size of the training dataloader may have changed
    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args.gradient_accumulation_steps)
    if overrode_max_train_steps:
        args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
    # Afterwards we recalculate our number of training epochs
    args.num_train_epochs = math.ceil(args.max_train_steps / num_update_steps_per_epoch)

    # Figure out how many steps we should save the Accelerator states
    checkpointing_steps = args.checkpointing_steps
    if checkpointing_steps is not None and checkpointing_steps.isdigit():
        checkpointing_steps = int(checkpointing_steps)

    # We need to initialize the trackers we use, and also store our configuration.
    # The trackers initializes automatically on the main process.
    if args.with_tracking:
        experiment_config = vars(args)
        # TensorBoard cannot log Enums, need the raw value
        experiment_config["lr_scheduler_type"] = experiment_config["lr_scheduler_type"].value
        accelerator.init_trackers("glue_no_trainer", experiment_config)

    # Get the metric function
    if args.task_name is not None:
        metric = evaluate.load("glue", args.task_name)
        saving_metric = {"stsb": "pearson", "cola": "matthews_correlation"}.get(args.task_name, "accuracy")
    else:
        metric = evaluate.load("accuracy")

    # Train!
    total_batch_size = args.per_device_train_batch_size * accelerator.num_processes * args.gradient_accumulation_steps

    logger.info("***** Running training *****")
    logger.info(f"  Num examples = {len(train_dataset)}")
    logger.info(f"  Num Epochs = {args.num_train_epochs}")
    logger.info(f"  Instantaneous batch size per device = {args.per_device_train_batch_size}")
    logger.info(f"  Total train batch size (w. parallel, distributed & accumulation) = {total_batch_size}")
    logger.info(f"  Gradient Accumulation steps = {args.gradient_accumulation_steps}")
    logger.info(f"  Total optimization steps = {args.max_train_steps}")
    # Only show the progress bar once on each machine.
    progress_bar = tqdm(range(args.max_train_steps), disable=not accelerator.is_local_main_process)
    completed_steps = 0
    starting_epoch = 0
    # Potentially load in the weights and states from a previous save
    if args.resume_from_checkpoint:
        if args.resume_from_checkpoint is not None or args.resume_from_checkpoint != "":
            checkpoint_path = args.resume_from_checkpoint
            path = os.path.basename(args.resume_from_checkpoint)
        else:
            # Get the most recent checkpoint
            dirs = [f.name for f in os.scandir(os.getcwd()) if f.is_dir()]
            dirs.sort(key=os.path.getctime)
            path = dirs[-1]  # Sorts folders by date modified, most recent checkpoint is the last
            checkpoint_path = path
            path = os.path.basename(checkpoint_path)

        accelerator.print(f"Resumed from checkpoint: {checkpoint_path}")
        accelerator.load_state(checkpoint_path)
        # Extract `epoch_{i}` or `step_{i}`
        training_difference = os.path.splitext(path)[0]

        if "epoch" in training_difference:
            starting_epoch = int(training_difference.replace("epoch_", "")) + 1
            resume_step = None
            completed_steps = starting_epoch * num_update_steps_per_epoch
        else:
            # need to multiply `gradient_accumulation_steps` to reflect real steps
            resume_step = int(training_difference.replace("step_", "")) * args.gradient_accumulation_steps
            starting_epoch = resume_step // len(train_dataloader)
            completed_steps = resume_step // args.gradient_accumulation_steps
            resume_step -= starting_epoch * len(train_dataloader)

    # update the progress_bar if load from checkpoint
    progress_bar.update(completed_steps)
    loss_meter = AverageMeter()
    
    saver = None
    if accelerator.is_main_process:
        saver = CheckpointSaver(wrapped_model, max_history=1, checkpoint_dir=args.output_dir, recovery_dir=args.output_dir, unwrap_fn=accelerator.unwrap_model)
    hyperparameter_search_epochs = args.num_train_epochs // 2
    for epoch in range(starting_epoch, args.num_train_epochs):
        model.train()
        if args.with_tracking:
            total_loss = 0
        if args.resume_from_checkpoint and epoch == starting_epoch and resume_step is not None:
            # We skip the first `n` batches in the dataloader when resuming from a checkpoint
            active_dataloader = accelerator.skip_first_batches(train_dataloader, resume_step)
        else:
            active_dataloader = train_dataloader
        for step, batch in enumerate(active_dataloader):
            outputs = model(**batch)
            loss = outputs.loss
            # We keep track of the loss at each epoch
            if args.with_tracking:
                total_loss += loss.detach().float()
            loss = loss / args.gradient_accumulation_steps
            loss_meter.update(loss.item(), args.per_device_train_batch_size)
            accelerator.backward(loss)
            if step % args.gradient_accumulation_steps == 0 or step == len(train_dataloader) - 1:
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()
                progress_bar.update(1)
                progress_bar.set_description(f"loss={loss_meter.avg:.4f}")
                completed_steps += 1

            if isinstance(checkpointing_steps, int):
                if completed_steps % checkpointing_steps == 0:
                    output_dir = f"step_{completed_steps}"
                    if args.output_dir is not None:
                        output_dir = os.path.join(args.output_dir, output_dir)
                    accelerator.save_state(output_dir)

            if completed_steps >= args.max_train_steps:
                break

        eval_metric = validate(accelerator, eval_dataloader, model, metric, is_regression)

        logger.info(f"epoch {epoch}: {eval_metric}")
        update_summary(epoch, {"loss": loss_meter.avg}, eval_metric, Path(args.output_dir, "summary.csv"), lr=lr_scheduler.get_last_lr()[0], write_header=epoch == 0)

        if accelerator.is_main_process and saver is not None:
            save_metric = eval_metric[saving_metric]
            best_metric, best_epoch = saver.save_checkpoint(
                epoch, metric=save_metric
            )

        if args.with_tracking:
            accelerator.log(
                {
                    "accuracy" if args.task_name is not None else "glue": eval_metric,
                    "train_loss": total_loss.item() / len(train_dataloader),
                    "epoch": epoch,
                    "step": completed_steps,
                },
                step=completed_steps,
            )

        if args.push_to_hub and epoch < args.num_train_epochs - 1:
            accelerator.wait_for_everyone()
            unwrapped_model = accelerator.unwrap_model(model)
            unwrapped_model.save_pretrained(
                args.output_dir, is_main_process=accelerator.is_main_process, save_function=accelerator.save
            )
            if accelerator.is_main_process:
                tokenizer.save_pretrained(args.output_dir)
                api.upload_folder(
                    commit_message=f"Training in progress epoch {epoch}",
                    folder_path=args.output_dir,
                    repo_id=repo_id,
                    repo_type="model",
                    token=args.hub_token,
                )

        if args.checkpointing_steps == "epoch":
            output_dir = f"epoch_{epoch}"
            if args.output_dir is not None:
                output_dir = os.path.join(args.output_dir, output_dir)
            accelerator.save_state(output_dir)

        if args.hyperparameter_search and epoch + 1 == hyperparameter_search_epochs:
            break

    if args.do_predict and not args.hyperparameter_search:

        # Load the state dict of the best model
        best_model_path = os.path.join(args.output_dir, "model_best.pth.tar")
        state_dict = torch.load(best_model_path, map_location="cpu")
        key_info = wrapped_model.load_state_dict(state_dict)
        logger.info(f"Loaded the best model from {best_model_path}.")

        # Verify correctness on validation set
        eval_metric = validate(accelerator, eval_dataloader, model, metric, is_regression)
        logger.info(f"Best Validation score: {eval_metric}")

        model.eval()
        samples_seen = 0
        all_predictions = []
        for step, batch in enumerate(predict_dataloader):
            del batch["labels"]
            with torch.no_grad():
                outputs = model(**batch)
            predictions = outputs.logits.argmax(dim=-1) if not is_regression else outputs.logits.squeeze()
            predictions, = accelerator.gather((predictions,))
            # If we are in a multiprocess environment, the last batch has duplicates
            if accelerator.num_processes > 1:
                if step == len(eval_dataloader) - 1:
                    predictions = predictions[: len(eval_dataloader.dataset) - samples_seen]
                    references = references[: len(eval_dataloader.dataset) - samples_seen]
                else:
                    samples_seen += references.shape[0]
            final_predictions = predictions.cpu().numpy().tolist()
            all_predictions = all_predictions + final_predictions

        output_predict_file = os.path.join(args.output_dir, f"predict_results_{args.task_name}.tsv")
        if accelerator.is_main_process:
            with open(output_predict_file, "w") as writer:
                logger.info(f"***** Predict results {args.task_name} *****")
                writer.write("index\tprediction\n")
                for index, item in enumerate(all_predictions):
                    if is_regression:
                        writer.write(f"{index}\t{item:3.3f}\n")
                    else:
                        if args.task_name in ["qnli", "rte"]:
                            item = label_list[item]
                        writer.write(f"{index}\t{item}\n")

    # Save the best metric and best epoch in a json file
    best_metric_file = os.path.join(args.output_dir, "best_metric.json")
    if accelerator.is_main_process:
        with open(best_metric_file, "w") as f:
            json.dump({"best_metric": best_metric, "best_epoch": best_epoch}, f)

    if args.hyperparameter_search and accelerator.is_main_process:
        # Remove all files starting from .pth.tar
        for f in os.listdir(args.output_dir):
            if f.endswith(".pth.tar"):
                os.remove(os.path.join(args.output_dir, f))
    
    if args.with_tracking:
        accelerator.end_training()

    if args.task_name == "mnli":
        # Final evaluation on mismatched validation set
        eval_dataset = processed_datasets["validation_mismatched"]
        eval_dataloader = DataLoader(
            eval_dataset, collate_fn=data_collator, batch_size=args.per_device_eval_batch_size
        )
        eval_dataloader = accelerator.prepare(eval_dataloader)

        model.eval()
        for step, batch in enumerate(eval_dataloader):
            outputs = model(**batch)
            predictions = outputs.logits.argmax(dim=-1)
            metric.add_batch(
                predictions=accelerator.gather(predictions),
                references=accelerator.gather(batch["labels"]),
            )

        eval_metric = metric.compute()
        logger.info(f"mnli-mm: {eval_metric}")

    if args.output_dir is not None:
        all_results = {f"eval_{k}": v for k, v in eval_metric.items()}
        with open(os.path.join(args.output_dir, "all_results.json"), "w") as f:
            json.dump(all_results, f)


def validate(accelerator, loader, model, metric, is_regression):

    model.eval()
    samples_seen = 0
    for step, batch in enumerate(loader):
        with torch.no_grad():
            outputs = model(**batch)
        predictions = outputs.logits.argmax(dim=-1) if not is_regression else outputs.logits.squeeze()
        predictions, references = accelerator.gather((predictions, batch["labels"]))
        # If we are in a multiprocess environment, the last batch has duplicates
        if accelerator.num_processes > 1:
            if step == len(loader) - 1:
                predictions = predictions[: len(loader.dataset) - samples_seen]
                references = references[: len(loader.dataset) - samples_seen]
            else:
                samples_seen += references.shape[0]
        metric.add_batch(
            predictions=predictions,
            references=references,
        )

    eval_metric = metric.compute()

    return eval_metric


if __name__ == "__main__":
    main()
