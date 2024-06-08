import argparse
import logging
import os
import time
from collections import OrderedDict
from contextlib import suppress
from datetime import datetime
from functools import partial
from pathlib import Path

import torch
import torch.nn as nn
import torchvision.utils
import yaml
from torch.nn.parallel import DistributedDataParallel as NativeDDP

from timm import utils
from timm.layers import convert_splitbn_model, convert_sync_batchnorm, set_fast_norm
from timm.models import (
    create_model,
    safe_model_name,
    resume_checkpoint,
    model_parameters,
)
from timm.data import resolve_data_config
from timm.optim import create_optimizer_v2, optimizer_kwargs
from timm.scheduler import create_scheduler_v2, scheduler_kwargs
from timm.utils import ApexScaler, NativeScaler

from visual_classification_utils.configs.config import get_cfg, BENCHMARK_NUM_CLASSES
from visual_classification_utils.data import loader as data_loader

from gift.model_builder import build_model

try:
    from apex import amp
    from apex.parallel import DistributedDataParallel as ApexDDP
    from apex.parallel import convert_syncbn_model

    has_apex = True
except ImportError:
    has_apex = False

has_native_amp = False
try:
    if getattr(torch.cuda.amp, "autocast") is not None:
        has_native_amp = True
except AttributeError:
    pass

try:
    import wandb

    has_wandb = True
except ImportError:
    has_wandb = False

try:
    from functorch.compile import memory_efficient_fusion

    has_functorch = True
except ImportError as e:
    has_functorch = False

has_compile = hasattr(torch, "compile")


_logger = logging.getLogger("train")

# The first arg parser parses out only the --config argument, this argument is used to
# load a yaml file containing key-values that override the defaults for the main parser below
config_parser = parser = argparse.ArgumentParser(
    description="Training Config", add_help=False
)
parser.add_argument(
    "-c",
    "--config",
    default="",
    type=str,
    metavar="FILE",
    help="YAML config file specifying default arguments",
)


parser = argparse.ArgumentParser(description="PyTorch ImageNet Training")

# Dataset parameters
group = parser.add_argument_group("Dataset parameters")

parser.add_argument("--data-dir", metavar="DIR", help="path to dataset (root dir)")
parser.add_argument(
    "--dataset",
    metavar="NAME",
    default="",
    help='dataset type + name ("<type>/<name>") (default: ImageFolder or ImageTar if empty)',
)
group.add_argument(
    "--train-split",
    metavar="NAME",
    default="train",
    help="dataset train split (default: train)",
)
group.add_argument(
    "--val-split",
    metavar="NAME",
    default="validation",
    help="dataset validation split (default: validation)",
)
group.add_argument(
    "--dataset-download",
    action="store_true",
    default=False,
    help="Allow download of dataset for torch/ and tfds/ datasets that support it.",
)
group.add_argument(
    "--class-map",
    default="",
    type=str,
    metavar="FILENAME",
    help='path to class to idx mapping file (default: "")',
)

# Model parameters
group = parser.add_argument_group("Model parameters")
group.add_argument(
    "--model",
    default="resnet50",
    type=str,
    metavar="MODEL",
    help='Name of model to train (default: "resnet50")',
)
group.add_argument(
    "--pretrained",
    action="store_true",
    default=False,
    help="Start with pretrained version of specified network (if avail)",
)
group.add_argument(
    "--initial-checkpoint",
    default="",
    type=str,
    metavar="PATH",
    help="Initialize model from this checkpoint (default: none)",
)
group.add_argument(
    "--resume",
    default="",
    type=str,
    metavar="PATH",
    help="Resume full model and optimizer state from checkpoint (default: none)",
)
group.add_argument(
    "--no-resume-opt",
    action="store_true",
    default=False,
    help="prevent resume of optimizer state when resuming model",
)
group.add_argument(
    "--num-classes",
    type=int,
    default=None,
    metavar="N",
    help="number of label classes (Model default if None)",
)
group.add_argument(
    "--gp",
    default=None,
    type=str,
    metavar="POOL",
    help="Global pool type, one of (fast, avg, max, avgmax, avgmaxc). Model default if None.",
)
group.add_argument(
    "--img-size",
    type=int,
    default=None,
    metavar="N",
    help="Image size (default: None => model default)",
)
group.add_argument(
    "--in-chans",
    type=int,
    default=None,
    metavar="N",
    help="Image input channels (default: None => 3)",
)
group.add_argument(
    "--input-size",
    default=None,
    nargs=3,
    type=int,
    metavar="N N N",
    help="Input all image dimensions (d h w, e.g. --input-size 3 224 224), uses model default if empty",
)
group.add_argument(
    "--crop-pct",
    default=None,
    type=float,
    metavar="N",
    help="Input image center crop percent (for validation only)",
)
group.add_argument(
    "--mean",
    type=float,
    nargs="+",
    default=None,
    metavar="MEAN",
    help="Override mean pixel value of dataset",
)
group.add_argument(
    "--std",
    type=float,
    nargs="+",
    default=None,
    metavar="STD",
    help="Override std deviation of dataset",
)
group.add_argument(
    "--interpolation",
    default="",
    type=str,
    metavar="NAME",
    help="Image resize interpolation type (overrides model)",
)
group.add_argument(
    "-b",
    "--batch-size",
    type=int,
    default=128,
    metavar="N",
    help="Input batch size for training (default: 128)",
)
group.add_argument(
    "-vb",
    "--validation-batch-size",
    type=int,
    default=None,
    metavar="N",
    help="Validation batch size override (default: None)",
)
group.add_argument(
    "--channels-last",
    action="store_true",
    default=False,
    help="Use channels_last memory layout",
)
group.add_argument(
    "--fuser",
    default="",
    type=str,
    help="Select jit fuser. One of ('', 'te', 'old', 'nvfuser')",
)
group.add_argument(
    "--grad-accum-steps",
    type=int,
    default=1,
    metavar="N",
    help="The number of steps to accumulate gradients (default: 1)",
)
group.add_argument(
    "--grad-checkpointing",
    action="store_true",
    default=False,
    help="Enable gradient checkpointing through model blocks/stages",
)
group.add_argument(
    "--fast-norm",
    default=False,
    action="store_true",
    help="enable experimental fast-norm",
)
group.add_argument("--model-kwargs", nargs="*", default={}, action=utils.ParseKwargs)
group.add_argument(
    "--head-init-scale", default=None, type=float, help="Head initialization scale"
)
group.add_argument(
    "--head-init-bias", default=None, type=float, help="Head initialization bias value"
)

# scripting / codegen
scripting_group = group.add_mutually_exclusive_group()
scripting_group.add_argument(
    "--torchscript",
    dest="torchscript",
    action="store_true",
    help="torch.jit.script the full model",
)
scripting_group.add_argument(
    "--torchcompile",
    nargs="?",
    type=str,
    default=None,
    const="inductor",
    help="Enable compilation w/ specified backend (default: inductor).",
)

# Optimizer parameters
group = parser.add_argument_group("Optimizer parameters")
group.add_argument(
    "--opt",
    default="sgd",
    type=str,
    metavar="OPTIMIZER",
    help='Optimizer (default: "sgd")',
)
group.add_argument(
    "--opt-eps",
    default=None,
    type=float,
    metavar="EPSILON",
    help="Optimizer Epsilon (default: None, use opt default)",
)
group.add_argument(
    "--opt-betas",
    default=None,
    type=float,
    nargs="+",
    metavar="BETA",
    help="Optimizer Betas (default: None, use opt default)",
)
group.add_argument(
    "--momentum",
    type=float,
    default=0.9,
    metavar="M",
    help="Optimizer momentum (default: 0.9)",
)
group.add_argument(
    "--weight-decay", type=float, default=2e-5, help="weight decay (default: 2e-5)"
)
group.add_argument(
    "--clip-grad",
    type=float,
    default=None,
    metavar="NORM",
    help="Clip gradient norm (default: None, no clipping)",
)
group.add_argument(
    "--clip-mode",
    type=str,
    default="norm",
    help='Gradient clipping mode. One of ("norm", "value", "agc")',
)
group.add_argument(
    "--layer-decay",
    type=float,
    default=None,
    help="layer-wise learning rate decay (default: None)",
)
group.add_argument("--opt-kwargs", nargs="*", default={}, action=utils.ParseKwargs)

# Learning rate schedule parameters
group = parser.add_argument_group("Learning rate schedule parameters")
group.add_argument(
    "--sched",
    type=str,
    default="cosine",
    metavar="SCHEDULER",
    help='LR scheduler (default: "step"',
)
group.add_argument(
    "--sched-on-updates",
    action="store_true",
    default=False,
    help="Apply LR scheduler step on update instead of epoch end.",
)
group.add_argument(
    "--lr",
    type=float,
    default=None,
    metavar="LR",
    help="learning rate, overrides lr-base if set (default: None)",
)
group.add_argument(
    "--lr-base",
    type=float,
    default=0.1,
    metavar="LR",
    help="base learning rate: lr = lr_base * global_batch_size / base_size",
)
group.add_argument(
    "--lr-base-size",
    type=int,
    default=256,
    metavar="DIV",
    help="base learning rate batch size (divisor, default: 256).",
)
group.add_argument(
    "--lr-base-scale",
    type=str,
    default="",
    metavar="SCALE",
    help='base learning rate vs batch_size scaling ("linear", "sqrt", based on opt if empty)',
)
group.add_argument(
    "--lr-noise",
    type=float,
    nargs="+",
    default=None,
    metavar="pct, pct",
    help="learning rate noise on/off epoch percentages",
)
group.add_argument(
    "--lr-noise-pct",
    type=float,
    default=0.67,
    metavar="PERCENT",
    help="learning rate noise limit percent (default: 0.67)",
)
group.add_argument(
    "--lr-noise-std",
    type=float,
    default=1.0,
    metavar="STDDEV",
    help="learning rate noise std-dev (default: 1.0)",
)
group.add_argument(
    "--lr-cycle-mul",
    type=float,
    default=1.0,
    metavar="MULT",
    help="learning rate cycle len multiplier (default: 1.0)",
)
group.add_argument(
    "--lr-cycle-decay",
    type=float,
    default=0.5,
    metavar="MULT",
    help="amount to decay each learning rate cycle (default: 0.5)",
)
group.add_argument(
    "--lr-cycle-limit",
    type=int,
    default=1,
    metavar="N",
    help="learning rate cycle limit, cycles enabled if > 1",
)
group.add_argument(
    "--lr-k-decay",
    type=float,
    default=1.0,
    help="learning rate k-decay for cosine/poly (default: 1.0)",
)
group.add_argument(
    "--warmup-lr",
    type=float,
    default=1e-5,
    metavar="LR",
    help="warmup learning rate (default: 1e-5)",
)
group.add_argument(
    "--min-lr",
    type=float,
    default=0,
    metavar="LR",
    help="lower lr bound for cyclic schedulers that hit 0 (default: 0)",
)
group.add_argument(
    "--epochs",
    type=int,
    default=300,
    metavar="N",
    help="number of epochs to train (default: 300)",
)
group.add_argument(
    "--epoch-repeats",
    type=float,
    default=0.0,
    metavar="N",
    help="epoch repeat multiplier (number of times to repeat dataset epoch per train epoch).",
)
group.add_argument(
    "--start-epoch",
    default=None,
    type=int,
    metavar="N",
    help="manual epoch number (useful on restarts)",
)
group.add_argument(
    "--decay-milestones",
    default=[90, 180, 270],
    type=int,
    nargs="+",
    metavar="MILESTONES",
    help="list of decay epoch indices for multistep lr. must be increasing",
)
group.add_argument(
    "--decay-epochs",
    type=float,
    default=90,
    metavar="N",
    help="epoch interval to decay LR",
)
group.add_argument(
    "--warmup-epochs",
    type=int,
    default=5,
    metavar="N",
    help="epochs to warmup LR, if scheduler supports",
)
group.add_argument(
    "--warmup-prefix",
    action="store_true",
    default=False,
    help="Exclude warmup period from decay schedule.",
),
group.add_argument(
    "--cooldown-epochs",
    type=int,
    default=0,
    metavar="N",
    help="epochs to cooldown LR at min_lr, after cyclic schedule ends",
)
group.add_argument(
    "--patience-epochs",
    type=int,
    default=10,
    metavar="N",
    help="patience epochs for Plateau LR scheduler (default: 10)",
)
group.add_argument(
    "--decay-rate",
    "--dr",
    type=float,
    default=0.1,
    metavar="RATE",
    help="LR decay rate (default: 0.1)",
)

# Augmentation & regularization parameters
group = parser.add_argument_group("Augmentation and regularization parameters")
group.add_argument(
    "--no-aug",
    action="store_true",
    default=False,
    help="Disable all training augmentation, override other train aug args",
)
group.add_argument(
    "--scale",
    type=float,
    nargs="+",
    default=[0.08, 1.0],
    metavar="PCT",
    help="Random resize scale (default: 0.08 1.0)",
)
group.add_argument(
    "--ratio",
    type=float,
    nargs="+",
    default=[3.0 / 4.0, 4.0 / 3.0],
    metavar="RATIO",
    help="Random resize aspect ratio (default: 0.75 1.33)",
)
group.add_argument(
    "--hflip", type=float, default=0.5, help="Horizontal flip training aug probability"
)
group.add_argument(
    "--vflip", type=float, default=0.0, help="Vertical flip training aug probability"
)
group.add_argument(
    "--color-jitter",
    type=float,
    default=0.4,
    metavar="PCT",
    help="Color jitter factor (default: 0.4)",
)
group.add_argument(
    "--aa",
    type=str,
    default=None,
    metavar="NAME",
    help='Use AutoAugment policy. "v0" or "original". (default: None)',
),
group.add_argument(
    "--aug-repeats",
    type=float,
    default=0,
    help="Number of augmentation repetitions (distributed training only) (default: 0)",
)
group.add_argument(
    "--aug-splits",
    type=int,
    default=0,
    help="Number of augmentation splits (default: 0, valid: 0 or >=2)",
)
group.add_argument(
    "--jsd-loss",
    action="store_true",
    default=False,
    help="Enable Jensen-Shannon Divergence + CE loss. Use with `--aug-splits`.",
)
group.add_argument(
    "--bce-loss",
    action="store_true",
    default=False,
    help="Enable BCE loss w/ Mixup/CutMix use.",
)
group.add_argument(
    "--bce-target-thresh",
    type=float,
    default=None,
    help="Threshold for binarizing softened BCE targets (default: None, disabled)",
)
group.add_argument(
    "--reprob",
    type=float,
    default=0.0,
    metavar="PCT",
    help="Random erase prob (default: 0.)",
)
group.add_argument(
    "--remode", type=str, default="pixel", help='Random erase mode (default: "pixel")'
)
group.add_argument(
    "--recount", type=int, default=1, help="Random erase count (default: 1)"
)
group.add_argument(
    "--resplit",
    action="store_true",
    default=False,
    help="Do not random erase first (clean) augmentation split",
)
group.add_argument(
    "--mixup",
    type=float,
    default=0.0,
    help="mixup alpha, mixup enabled if > 0. (default: 0.)",
)
group.add_argument(
    "--cutmix",
    type=float,
    default=0.0,
    help="cutmix alpha, cutmix enabled if > 0. (default: 0.)",
)
group.add_argument(
    "--cutmix-minmax",
    type=float,
    nargs="+",
    default=None,
    help="cutmix min/max ratio, overrides alpha and enables cutmix if set (default: None)",
)
group.add_argument(
    "--mixup-prob",
    type=float,
    default=1.0,
    help="Probability of performing mixup or cutmix when either/both is enabled",
)
group.add_argument(
    "--mixup-switch-prob",
    type=float,
    default=0.5,
    help="Probability of switching to cutmix when both mixup and cutmix enabled",
)
group.add_argument(
    "--mixup-mode",
    type=str,
    default="batch",
    help='How to apply mixup/cutmix params. Per "batch", "pair", or "elem"',
)
group.add_argument(
    "--mixup-off-epoch",
    default=0,
    type=int,
    metavar="N",
    help="Turn off mixup after this epoch, disabled if 0 (default: 0)",
)
group.add_argument(
    "--smoothing", type=float, default=0.1, help="Label smoothing (default: 0.1)"
)
group.add_argument(
    "--train-interpolation",
    type=str,
    default="random",
    help='Training interpolation (random, bilinear, bicubic default: "random")',
)
group.add_argument(
    "--drop", type=float, default=0.0, metavar="PCT", help="Dropout rate (default: 0.)"
)
group.add_argument(
    "--drop-connect",
    type=float,
    default=None,
    metavar="PCT",
    help="Drop connect rate, DEPRECATED, use drop-path (default: None)",
)
group.add_argument(
    "--drop-path",
    type=float,
    default=None,
    metavar="PCT",
    help="Drop path rate (default: None)",
)
group.add_argument(
    "--drop-block",
    type=float,
    default=None,
    metavar="PCT",
    help="Drop block rate (default: None)",
)

# Batch norm parameters (only works with gen_efficientnet based models currently)
group = parser.add_argument_group(
    "Batch norm parameters", "Only works with gen_efficientnet based models currently."
)
group.add_argument(
    "--bn-momentum",
    type=float,
    default=None,
    help="BatchNorm momentum override (if not None)",
)
group.add_argument(
    "--bn-eps",
    type=float,
    default=None,
    help="BatchNorm epsilon override (if not None)",
)
group.add_argument(
    "--sync-bn",
    action="store_true",
    help="Enable NVIDIA Apex or Torch synchronized BatchNorm.",
)
group.add_argument(
    "--dist-bn",
    type=str,
    default="reduce",
    help='Distribute BatchNorm stats between nodes after each epoch ("broadcast", "reduce", or "")',
)
group.add_argument(
    "--split-bn",
    action="store_true",
    help="Enable separate BN layers per augmentation split.",
)

# Model Exponential Moving Average
group = parser.add_argument_group("Model exponential moving average parameters")
group.add_argument(
    "--model-ema",
    action="store_true",
    default=False,
    help="Enable tracking moving average of model weights",
)
group.add_argument(
    "--model-ema-force-cpu",
    action="store_true",
    default=False,
    help="Force ema to be tracked on CPU, rank=0 node only. Disables EMA validation.",
)
group.add_argument(
    "--model-ema-decay",
    type=float,
    default=0.9998,
    help="decay factor for model weights moving average (default: 0.9998)",
)

# Misc
group = parser.add_argument_group("Miscellaneous parameters")
group.add_argument(
    "--seed", type=int, default=42, metavar="S", help="random seed (default: 42)"
)
group.add_argument(
    "--worker-seeding", type=str, default="all", help="worker seed mode (default: all)"
)
group.add_argument(
    "--log-interval",
    type=int,
    default=5,
    metavar="N",
    help="how many batches to wait before logging training status",
)
group.add_argument(
    "--recovery-interval",
    type=int,
    default=0,
    metavar="N",
    help="how many batches to wait before writing recovery checkpoint",
)
group.add_argument(
    "--checkpoint-hist",
    type=int,
    default=1,
    metavar="N",
    help="number of checkpoints to keep (default: 10)",
)
group.add_argument(
    "-j",
    "--workers",
    type=int,
    default=4,
    metavar="N",
    help="how many training processes to use (default: 4)",
)
group.add_argument(
    "--save-images",
    action="store_true",
    default=False,
    help="save images of input bathes every log interval for debugging",
)
group.add_argument(
    "--amp",
    action="store_true",
    default=False,
    help="use NVIDIA Apex AMP or Native AMP for mixed precision training",
)
group.add_argument(
    "--amp-dtype",
    default="float16",
    type=str,
    help="lower precision AMP dtype (default: float16)",
)
group.add_argument(
    "--amp-impl",
    default="native",
    type=str,
    help='AMP impl to use, "native" or "apex" (default: native)',
)
group.add_argument(
    "--no-ddp-bb",
    action="store_true",
    default=False,
    help="Force broadcast buffers for native DDP to off.",
)
group.add_argument(
    "--synchronize-step",
    action="store_true",
    default=False,
    help="torch.cuda.synchronize() end of each step",
)
group.add_argument(
    "--pin-mem",
    action="store_true",
    default=False,
    help="Pin CPU memory in DataLoader for more efficient (sometimes) transfer to GPU.",
)
group.add_argument(
    "--no-prefetcher",
    action="store_true",
    default=False,
    help="disable fast prefetcher",
)
group.add_argument(
    "--output",
    default="",
    type=str,
    metavar="PATH",
    help="path to output folder (default: none, current dir)",
)
group.add_argument(
    "--experiment",
    default="",
    type=str,
    metavar="NAME",
    help="name of train experiment, name of sub-folder for output",
)
group.add_argument(
    "--eval-metric",
    default="top1",
    type=str,
    metavar="EVAL_METRIC",
    help='Best metric (default: "top1"',
)
group.add_argument(
    "--tta",
    type=int,
    default=0,
    metavar="N",
    help="Test/inference time augmentation (oversampling) factor. 0=None (default: 0)",
)
group.add_argument("--local_rank", default=0, type=int)
group.add_argument(
    "--use-multi-epochs-loader",
    action="store_true",
    default=False,
    help="use the multi-epochs-loader to save time at the beginning of every epoch",
)
group.add_argument(
    "--log-wandb",
    action="store_true",
    default=False,
    help="log training and validation metrics to wandb",
)
group.add_argument(
    "--wandb-project", type=str, default="", help="Name of the WandB project."
)
group.add_argument(
    "--execution_dir", 
    type=str, 
    default=None, 
    help="Execution directory of the script.")
group.add_argument(
    "--artifact_dir", 
    type=str, 
    default="artifacts",
    help="Directory to store the artifacts."
)

# Custom
group = parser.add_argument_group("Custom parameters")
group.add_argument(
    "--data-config", 
    type=str, 
    default="", 
    help="Config for dataloaders."
)
group.add_argument(
    "--method", 
    type=str, 
    default="", 
    choices=["gift", "lora", "vpt", "bitfit"],
    help="Finetuning method."
)
group.add_argument(
    "--evaluate",
    action="store_true",
    default=False,
    help="Evaluate on test set",
)
group.add_argument(
    "--auto_scale_warmup_min_lr",
    action="store_true",
    default=False,
    help="Auto scale the LR",
)

# GIFT
group = parser.add_argument_group("GIFT parameters")
group.add_argument(
    "--gift_rank",
    type=int,
    default=16,
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
    default=["attn:proj"],
    type=str,
    nargs="+",
    help="Module to apply finetuning on (also used for determining LoRA modules).",
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

# GIFT Block parameters
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

# Other methods
# LoRA
group.add_argument(
    "--lora_rank",
    type=int,
    default=8,
    help="LoRA rank.",
)
# VPT
group.add_argument(
    "--num_prompt_tokens",
    type=int,
    default=5,
    help="Number of prompt tokens in VPT.",
)
group.add_argument(
    "--deep_prompts",
    action="store_true",
    default=False,
    help="Use Deep Prompts in VPT.",
)

group.add_argument(
    "--hyperparameter_search",
    action="store_true",
    default=False,
    help="Flag to indicate that hyperperameter search is ongoing.",
)
group.add_argument(
    "--search_epochs",
    type=int,
    default=50,
    help="Number of training epochs during hyperparameter search.",
)
group.add_argument(
    "--gift_checkpoint",
    type=str,
    default="",
    help="Previous GIFT checkpoint.",
)
group.add_argument(
    "--save-all-checkpoints",
    action="store_true",
    default=False,
    help="Whether to store the checkpoint at all the epochs.",
)
group.add_argument(
    "--reset_classifier",
    action="store_true",
    default=False,
    help="Reset the classifier.",
)


parser.add_argument(
    "dataopts",
    help="Modify config options using the command-line",
    default=None,
    nargs=argparse.REMAINDER,
)

def param_groups_weight_decay_filter(
        model: nn.Module,
        weight_decay=1e-5,
        no_weight_decay_list=()
):
    no_weight_decay_list = set(no_weight_decay_list)
    decay = []
    no_decay = []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue

        if param.ndim <= 1 or name.endswith(".bias") \
            or name in no_weight_decay_list \
            or "cluster" in name:
            _logger.info(f"Skipping weight decay for {name}")
            no_decay.append(param)
        else:
            decay.append(param)

    return [
        {'params': no_decay, 'weight_decay': 0.},
        {'params': decay, 'weight_decay': weight_decay}]


def get_loaders(args, cfg, logger, mean, std):
    logger.info("Loading training data (final training data for vtab)...")
    if cfg.DATA.NAME.startswith("vtab-") and args.evaluate:
        train_loader = data_loader.construct_trainval_loader(cfg, root=args.execution_dir, mean=mean, std=std)
    else:
        train_loader = data_loader.construct_train_loader(cfg, root=args.execution_dir, mean=mean, std=std)

    if not args.evaluate:
        logger.info("Loading validation data...")
        # not really needed for vtab
        val_loader = data_loader.construct_val_loader(cfg, root=args.execution_dir, mean=mean, std=std)
        eval_loader = val_loader
    
    logger.info("Loading test data...")
    if cfg.DATA.NO_TEST:
        logger.info("...no test data is constructed")
        test_loader = None
    elif args.evaluate:
        test_loader = data_loader.construct_test_loader(cfg, root=args.execution_dir, mean=mean, std=std)
        eval_loader = test_loader

    return train_loader, eval_loader


def _parse_args():
    # Do we have a config file to parse?
    args_config, remaining = config_parser.parse_known_args()
    if args_config.config:
        with open(args_config.config, "r") as f:
            cfg = yaml.safe_load(f)
            parser.set_defaults(**cfg)

    # The main arg parser parses the rest of the args, the usual
    # defaults will have been overridden if config file specified.
    args = parser.parse_args(remaining)

    # Cache the args as a text string to save them in the output dir later
    args_text = yaml.safe_dump(args.__dict__, default_flow_style=False)
    return args, args_text


def main():
    # utils.setup_default_logging()
    args, args_text = _parse_args()

    # Get the data config
    cfg = get_cfg()
    cfg.merge_from_file(args.data_config)
    print(args.dataopts)
    cfg.merge_from_list(args.dataopts)
    args.dataset = cfg.DATA.NAME

    cfg.DATA.NUMBER_CLASSES = BENCHMARK_NUM_CLASSES[cfg.DATA.NAME]

    args.num_classes = cfg.DATA.NUMBER_CLASSES

    if args.method == "lora":
        method_suffix = f"-rank-{args.lora_rank}"
    else:
        method_suffix = ""

    if args.experiment:
        exp_name = args.experiment
    else:
        eval_type = "test" if args.evaluate else "val"
        gift_modules = "-".join(args.gift_target_modules)
        penalty = str(args.weight_decay)
        exp_name = "-".join(
            [
                args.method+method_suffix,
                gift_modules,
                eval_type,
                str(args.lr),
                penalty,
                str(args.gift_rank),
                datetime.now().strftime("%Y%m%d-%H%M%S"),
            ]
        )
    wandb_exp_name = "-".join(exp_name.split("-")[:-1])
    args.output = args.output or str(Path(args.execution_dir, args.artifact_dir, cfg.DATA.NAME, args.method+method_suffix, exp_name))
    output_dir = utils.get_outdir(
        args.output if args.output else "./output/train", exp_name
    )

    if torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.benchmark = True

    args.prefetcher = not args.no_prefetcher
    args.grad_accum_steps = max(1, args.grad_accum_steps)
    device = utils.init_distributed_device(args)

    log_path = os.path.join(output_dir, f"train_log_{args.rank}.log")
    utils.setup_default_logging(log_path=log_path)
    if args.distributed:
        _logger.info(
            "Training in distributed mode with multiple processes, 1 device per process."
            f"Process {args.rank}, total {args.world_size}, device {args.device}."
        )
    else:
        _logger.info(f"Training with a single process on 1 device ({args.device}).")
    assert args.rank >= 0

    # resolve AMP arguments based on PyTorch / Apex availability
    use_amp = None
    amp_dtype = torch.float16
    if args.amp:
        if args.amp_impl == "apex":
            assert has_apex, "AMP impl specified as APEX but APEX is not installed."
            use_amp = "apex"
            assert args.amp_dtype == "float16"
        else:
            assert (
                has_native_amp
            ), "Please update PyTorch to a version with native AMP (or use APEX)."
            use_amp = "native"
            assert args.amp_dtype in ("float16", "bfloat16")
        if args.amp_dtype == "bfloat16":
            amp_dtype = torch.bfloat16

    utils.random_seed(args.seed, args.rank)

    if args.fuser:
        utils.set_jit_fuser(args.fuser)
    if args.fast_norm:
        set_fast_norm()

    in_chans = 3
    if args.in_chans is not None:
        in_chans = args.in_chans
    elif args.input_size is not None:
        in_chans = args.input_size[0]

    backbone = create_model(
        args.model,
        pretrained=args.pretrained,
        in_chans=in_chans,
        num_classes=args.num_classes,
        drop_rate=args.drop,
        drop_path_rate=args.drop_path,
        drop_block_rate=args.drop_block,
        global_pool=args.gp,
        bn_momentum=args.bn_momentum,
        bn_eps=args.bn_eps,
        scriptable=args.torchscript,
        **args.model_kwargs,
    )
    # print(backbone)
    if args.head_init_scale is not None:
        with torch.no_grad():
            backbone.get_classifier().weight.mul_(args.head_init_scale)
            backbone.get_classifier().bias.mul_(args.head_init_scale)
    if args.head_init_bias is not None:
        nn.init.constant_(backbone.get_classifier().bias, args.head_init_bias)

    if args.initial_checkpoint:
        state_dict: OrderedDict = torch.load(args.initial_checkpoint, map_location="cpu")["model"]
        # Remove the head
        state_dict.pop("head.weight", None)
        state_dict.pop("head.bias", None)
        keys = backbone.load_state_dict(state_dict, strict=False)
        _logger.info(f"Loaded backbone {args.initial_checkpoint}")
        print(keys)

    model = build_model(args, backbone)

    if args.gift_checkpoint:
        # Load the hypernet checkpoint
        gift_state_dict: OrderedDict = torch.load(args.gift_checkpoint, map_location="cpu")["state_dict"]
        # Remove the head
        gift_state_dict.pop("weight", None)
        gift_state_dict.pop("bias", None)
        keys = model.load_state_dict(gift_state_dict, strict=False)
        _logger.info(f"Loaded hypernet {args.gift_checkpoint}")
        _logger.info(f"{keys}")

    # Save the state dictionary of the initialized model
    if utils.is_primary(args):
        state_dict = model.state_dict()
        torch.save(state_dict, os.path.join(output_dir, "initialized_model.pth.tar"))

    # Log the number of trainable parameters
    if utils.is_primary(args):
        num_trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        num_params_head = backbone.embed_dim*args.num_classes + args.num_classes
        _logger.info(f"Number of trainable parameters: {num_trainable_params}")
        _logger.info(f"Number of trainable parameters w/o head: {num_trainable_params - num_params_head}")

    if utils.is_primary(args):
        _logger.info(model)

    if args.num_classes is None:
        assert hasattr(
            model, "num_classes"
        ), "Model must have `num_classes` attr if not set on cmd line/config."
        args.num_classes = (
            model.num_classes
        )  # FIXME handle model default vs config num_classes more elegantly

    if args.grad_checkpointing:
        model.set_grad_checkpointing(enable=True)

    if utils.is_primary(args):
        _logger.info(
            f"Model {safe_model_name(args.model)} created, param count:{sum([m.numel() for m in model.parameters()])}"
        )

    # setup augmentation batch splits for contrastive loss or split bn
    num_aug_splits = 0
    if args.aug_splits > 0:
        assert args.aug_splits > 1, "A split of 1 makes no sense"
        num_aug_splits = args.aug_splits

    # enable split bn (separate bn stats per batch-portion)
    if args.split_bn:
        assert num_aug_splits > 1 or args.resplit
        model = convert_splitbn_model(model, max(num_aug_splits, 2))

    # move model to GPU, enable channels last layout if set
    model.to(device=device)
    if args.channels_last:
        model.to(memory_format=torch.channels_last)

    # setup synchronized BatchNorm for distributed training
    if args.distributed and args.sync_bn:
        args.dist_bn = ""  # disable dist_bn when sync BN active
        assert not args.split_bn
        if has_apex and use_amp == "apex":
            # Apex SyncBN used with Apex AMP
            # WARNING this won't currently work with models using BatchNormAct2d
            model = convert_syncbn_model(model)
        else:
            model = convert_sync_batchnorm(model)
        if utils.is_primary(args):
            _logger.info(
                "Converted model to use Synchronized BatchNorm. WARNING: You may have issues if using "
                "zero initialized BN layers (enabled by default for ResNets) while sync-bn enabled."
            )

    if not args.lr:
        global_batch_size = args.batch_size * args.world_size * args.grad_accum_steps
        batch_ratio = global_batch_size / args.lr_base_size
        if not args.lr_base_scale:
            on = args.opt.lower()
            args.lr_base_scale = (
                "sqrt" if any([o in on for o in ("ada", "lamb")]) else "linear"
            )
        if args.lr_base_scale == "sqrt":
            batch_ratio = batch_ratio**0.5
        args.lr = args.lr_base * batch_ratio
        if utils.is_primary(args):
            _logger.info(
                f"Learning rate ({args.lr}) calculated from base learning rate ({args.lr_base}) "
                f"and effective global batch size ({global_batch_size}) with {args.lr_base_scale} scaling."
            )

        # << ------------- ivmcl, from PaCA
        if args.auto_scale_warmup_min_lr:
            warmup_lr = args.warmup_lr
            args.warmup_lr = args.warmup_lr * batch_ratio
            if utils.is_primary(args):
                _logger.info(
                    f"Warmup Learning rate ({args.warmup_lr}) calculated from base learning rate ({warmup_lr}) "
                    f"and effective global batch size ({global_batch_size}) with {args.lr_base_scale} scaling."
                )
            min_lr = args.min_lr
            args.min_lr = args.min_lr * batch_ratio
            if utils.is_primary(args):
                _logger.info(
                    f"Minimum Learning rate ({args.min_lr}) calculated from base learning rate ({min_lr}) "
                    f"and effective global batch size ({global_batch_size}) with {args.lr_base_scale} scaling."
                )
        # ivmcl ------------- >>

    opt_args = optimizer_kwargs(cfg=args)
    weight_decay = opt_args["weight_decay"]
    filter_fn = partial(param_groups_weight_decay_filter, weight_decay=weight_decay)
    opt_args["weight_decay"] = 0.
    optimizer = create_optimizer_v2(
        model,
        **opt_args,
        **args.opt_kwargs,
        param_group_fn=filter_fn,
    )

    # setup automatic mixed-precision (AMP) loss scaling and op casting
    amp_autocast = suppress  # do nothing
    loss_scaler = None
    if use_amp == "apex":
        assert device.type == "cuda"
        model, optimizer = amp.initialize(model, optimizer, opt_level="O1")
        loss_scaler = ApexScaler()
        if utils.is_primary(args):
            _logger.info("Using NVIDIA APEX AMP. Training in mixed precision.")
    elif use_amp == "native":
        try:
            amp_autocast = partial(
                torch.autocast, device_type=device.type, dtype=amp_dtype
            )
        except (AttributeError, TypeError):
            # fallback to CUDA only AMP for PyTorch < 1.10
            assert device.type == "cuda"
            amp_autocast = torch.cuda.amp.autocast
        if device.type == "cuda" and amp_dtype == torch.float16:
            # loss scaler only used for float16 (half) dtype, bfloat16 does not need it
            loss_scaler = NativeScaler()
        if utils.is_primary(args):
            _logger.info("Using native Torch AMP. Training in mixed precision.")
    else:
        if utils.is_primary(args):
            _logger.info("AMP not enabled. Training in float32.")

    # optionally resume from a checkpoint
    resume_epoch = None
    if args.resume:
        resume_epoch = resume_checkpoint(
            model,
            args.resume,
            optimizer=None if args.no_resume_opt else optimizer,
            loss_scaler=None if args.no_resume_opt else loss_scaler,
            log_info=utils.is_primary(args),
        )

    # setup distributed training
    if args.distributed:
        if has_apex and use_amp == "apex":
            # Apex DDP preferred unless native amp is activated
            if utils.is_primary(args):
                _logger.info("Using NVIDIA APEX DistributedDataParallel.")
            model = ApexDDP(model, delay_allreduce=True)
        else:
            if utils.is_primary(args):
                _logger.info("Using native Torch DistributedDataParallel.")
            model = NativeDDP(
                model, device_ids=[device], broadcast_buffers=not args.no_ddp_bb
            )
        # NOTE: EMA model does not need to be wrapped by DDP

    data_config = resolve_data_config(vars(args), model=backbone)
    # Override the data config in case of vit_base_patch16_224/vit_large_patch16_224 since we use
    # a different checkpoint than timm
    if any([args.model.startswith(x) for x in ["vit_base_patch16_224", "vit_large_patch16_224"]]) and args.initial_checkpoint:
        data_config["mean"] = data_loader.IMAGENET_MEAN
        data_config["std"] = data_loader.IMAGENET_STD

    if utils.is_primary(args):
        _logger.info(f"Using mean={data_config['mean']} and std={data_config['std']}")

    # create data loaders w/ augmentation pipeiine
    loader_train, loader_eval = get_loaders(args, cfg, _logger, data_config["mean"], data_config["std"])
    
    train_loss_fn = nn.CrossEntropyLoss()
    # cls_weights = loader_train.dataset.get_class_weights(self.cfg.DATA.CLASS_WEIGHTS_TYPE)
    # train_loss_fn = SoftmaxLoss(cls_weights)
    train_loss_fn = train_loss_fn.to(device=device)
    validate_loss_fn = nn.CrossEntropyLoss().to(device=device)

    # setup checkpoint saver and eval metric tracking
    eval_metric = args.eval_metric
    best_metric = None
    best_epoch = None
    saver = None
    # output_dir = None
    if utils.is_primary(args):
        decreasing = True if eval_metric == "loss" else False
        saver = utils.CheckpointSaver(
            model=model,
            optimizer=optimizer,
            args=args,
            model_ema=None,
            amp_scaler=loss_scaler,
            checkpoint_dir=output_dir,
            recovery_dir=output_dir,
            decreasing=decreasing,
            max_history=args.checkpoint_hist,
        )
        with open(os.path.join(output_dir, "args.yaml"), "w") as f:
            f.write(args_text)

    if utils.is_primary(args) and args.log_wandb:
        if has_wandb:
            wandb.init(project=args.wandb_project, group=cfg.DATA.NAME+args.method+method_suffix, name=wandb_exp_name, config=args)
        else:
            _logger.warning(
                "You've requested to log metrics to wandb but package not found. "
                "Metrics not being logged to wandb, try `pip install wandb`"
            )

    # setup learning rate schedule and starting epoch
    updates_per_epoch = (
        len(loader_train) + args.grad_accum_steps - 1
    ) // args.grad_accum_steps
    lr_scheduler, num_epochs = create_scheduler_v2(
        optimizer,
        **scheduler_kwargs(args),
        updates_per_epoch=updates_per_epoch,
    )
    start_epoch = 0
    if args.start_epoch is not None:
        # a specified start_epoch will always override the resume epoch
        start_epoch = args.start_epoch
    elif resume_epoch is not None:
        start_epoch = resume_epoch
    if lr_scheduler is not None and start_epoch > 0:
        if args.sched_on_updates:
            lr_scheduler.step_update(start_epoch * updates_per_epoch)
        else:
            lr_scheduler.step(start_epoch)

    if utils.is_primary(args):
        _logger.info(
            f'Scheduled epochs: {num_epochs}. LR stepped per {"epoch" if lr_scheduler.t_in_epochs else "update"}.'
        )

    try:
        for epoch in range(start_epoch, num_epochs):
            if args.distributed and hasattr(loader_train.sampler, "set_epoch"):
                loader_train.sampler.set_epoch(epoch)


            train_metrics, is_nan = train_one_epoch(
                epoch,
                model,
                loader_train,
                optimizer,
                train_loss_fn,
                args,
                cfg,
                lr_scheduler=lr_scheduler,
                saver=saver,
                output_dir=output_dir,
                amp_autocast=amp_autocast,
                loss_scaler=loss_scaler,
            )

            if is_nan:
                _logger.error("Loss is NaN, exiting...")
                break

            if args.distributed and args.dist_bn in ("broadcast", "reduce"):
                if utils.is_primary(args):
                    _logger.info("Distributing BatchNorm running means and vars")
                utils.distribute_bn(model, args.world_size, args.dist_bn == "reduce")

            if epoch == 0 or (epoch + 1) == num_epochs or args.hyperparameter_search:
                eval_metrics = validate(
                    model,
                    loader_eval,
                    validate_loss_fn,
                    args,
                    cfg,
                    amp_autocast=amp_autocast,
                )

            if output_dir is not None:
                lrs = [param_group["lr"] for param_group in optimizer.param_groups]
                if utils.is_primary(args):
                    utils.update_summary(
                        epoch,
                        train_metrics,
                        eval_metrics,
                        filename=os.path.join(output_dir, "summary.csv"),
                        lr=sum(lrs) / len(lrs),
                        write_header=best_metric is None,
                        log_wandb=args.log_wandb and has_wandb,
                    )

            if saver is not None:
                # save proper checkpoint with eval metric
                save_metric = eval_metrics[eval_metric]
                best_metric, best_epoch = saver.save_checkpoint(
                    epoch, metric=save_metric
                )
            
            if args.save_all_checkpoints:
                torch.save(model.state_dict(), os.path.join(output_dir, f"model_{epoch}.pth.tar"))

            if lr_scheduler is not None:
                # step LR for next epoch
                lr_scheduler.step(epoch + 1, eval_metrics[eval_metric])

            if args.hyperparameter_search and epoch == (args.search_epochs - 1):
                break

        if not args.evaluate:
            checkpoint_path = Path(output_dir)
            for checkpoint in checkpoint_path.glob("*.pth.tar"):
                checkpoint.unlink()

    except KeyboardInterrupt:
        pass

    if best_metric is not None:
        _logger.info("*** Best metric: {0} (epoch {1})".format(best_metric, best_epoch))

def get_input(cfg, data):
        data_name = cfg.DATA.NAME
        if not isinstance(data["image"], torch.Tensor):
            for k, v in data.items():
                if k != "data_name":
                    data[k] = torch.from_numpy(v)

        if data_name == "CELEBA":
            inputs = data["image"].float()
            reference_attribute = data["reference_attribute"]
            labels = data["label"]
            return inputs, reference_attribute, labels
        inputs = data["image"].float()
        labels = data["label"]
        return inputs, labels


def train_one_epoch(
    epoch,
    model,
    loader,
    optimizer,
    loss_fn,
    args,
    cfg,
    device=torch.device("cuda"),
    lr_scheduler=None,
    saver=None,
    output_dir=None,
    amp_autocast=suppress,
    loss_scaler=None,
):

    second_order = hasattr(optimizer, "is_second_order") and optimizer.is_second_order
    has_no_sync = hasattr(model, "no_sync")
    update_time_m = utils.AverageMeter()
    data_time_m = utils.AverageMeter()
    losses_m = utils.AverageMeter()

    model.train()

    accum_steps = args.grad_accum_steps
    last_accum_steps = len(loader) % accum_steps
    updates_per_epoch = (len(loader) + accum_steps - 1) // accum_steps
    num_updates = epoch * updates_per_epoch
    last_batch_idx = len(loader) - 1
    last_batch_idx_to_accum = len(loader) - last_accum_steps

    data_start_time = update_start_time = time.time()
    optimizer.zero_grad()
    update_sample_count = 0
    for batch_idx, data in enumerate(loader):
        last_batch = batch_idx == last_batch_idx
        need_update = last_batch or (batch_idx + 1) % accum_steps == 0
        update_idx = batch_idx // accum_steps
        if batch_idx >= last_batch_idx_to_accum:
            accum_steps = last_accum_steps

        input, target = get_input(cfg, data)

        if not args.prefetcher:
            input, target = input.to(device), target.to(device)
        if args.channels_last:
            input = input.contiguous(memory_format=torch.channels_last)

        # multiply by accum steps to get equivalent for full update
        data_time_m.update(accum_steps * (time.time() - data_start_time))

        def _forward():
            with amp_autocast():
                output = model(input)
                if isinstance(output, tuple):
                    output, = output
                loss = loss_fn(output, target)
            if accum_steps > 1:
                loss /= accum_steps
            return loss

        def _backward(_loss):
            if loss_scaler is not None:
                loss_scaler(
                    _loss,
                    optimizer,
                    clip_grad=args.clip_grad,
                    clip_mode=args.clip_mode,
                    parameters=model_parameters(
                        model, exclude_head="agc" in args.clip_mode
                    ),
                    create_graph=second_order,
                    need_update=need_update,
                )
            else:
                _loss.backward(create_graph=second_order)
                if need_update:
                    if args.clip_grad is not None:
                        utils.dispatch_clip_grad(
                            model_parameters(
                                model, exclude_head="agc" in args.clip_mode
                            ),
                            value=args.clip_grad,
                            mode=args.clip_mode,
                        )
                    optimizer.step()

        if has_no_sync and not need_update:
            with model.no_sync():
                loss = _forward()
                _backward(loss)
        else:
            loss = _forward()
            if torch.isnan(loss).any():
                return None, True
            _backward(loss)

        if not args.distributed:
            losses_m.update(loss.item() * accum_steps, input.size(0))
        update_sample_count += input.size(0)

        if not need_update:
            data_start_time = time.time()
            continue

        num_updates += 1
        optimizer.zero_grad()

        if args.synchronize_step and device.type == "cuda":
            torch.cuda.synchronize()
        time_now = time.time()
        update_time_m.update(time.time() - update_start_time)
        update_start_time = time_now

        if update_idx % args.log_interval == 0:
            lrl = [param_group["lr"] for param_group in optimizer.param_groups]
            lr = sum(lrl) / len(lrl)

            if args.distributed:
                reduced_loss = utils.reduce_tensor(loss.data, args.world_size)
                losses_m.update(reduced_loss.item() * accum_steps, input.size(0))

                update_sample_count *= args.world_size

            if utils.is_primary(args):
                memory_used = torch.cuda.max_memory_allocated() / (
                    1024.0 * 1024.0 * 1024.0
                )
                _logger.info(
                    f"Train: {epoch} [{update_idx:>4d}/{updates_per_epoch} "
                    f"({100. * update_idx / (updates_per_epoch - 1):>3.0f}%)]  "
                    f"Loss: {losses_m.val:#.3g} ({losses_m.avg:#.3g})  "
                    f"Time: {update_time_m.val:.3f}s, {update_sample_count / update_time_m.val:>7.2f}/s  "
                    f"({update_time_m.avg:.3f}s, {update_sample_count / update_time_m.avg:>7.2f}/s)  "
                    f"LR: {lr:.3e}  "
                    f"Data: {data_time_m.val:.3f} ({data_time_m.avg:.3f})  "
                    f"Memory: {memory_used:.3f}GB"
                )

                if args.save_images and output_dir:
                    torchvision.utils.save_image(
                        input,
                        os.path.join(output_dir, "train-batch-%d.jpg" % batch_idx),
                        padding=0,
                        normalize=True,
                    )

        if (
            saver is not None
            and args.recovery_interval
            and ((update_idx + 1) % args.recovery_interval == 0)
        ):
            saver.save_recovery(epoch, batch_idx=update_idx)

        if lr_scheduler is not None:
            lr_scheduler.step_update(num_updates=num_updates, metric=losses_m.avg)

        update_sample_count = 0
        data_start_time = time.time()
        # end for

    if hasattr(optimizer, "sync_lookahead"):
        optimizer.sync_lookahead()

    return OrderedDict([("loss", losses_m.avg)]), False


def validate(
    model,
    loader,
    loss_fn,
    args,
    cfg,
    device=torch.device("cuda"),
    amp_autocast=suppress,
    log_suffix="",
):
    batch_time_m = utils.AverageMeter()
    losses_m = utils.AverageMeter()
    top1_m = utils.AverageMeter()
    top5_m = utils.AverageMeter()

    model.eval()

    end = time.time()
    last_idx = len(loader) - 1
    with torch.no_grad():
        for batch_idx, data in enumerate(loader):
            last_batch = batch_idx == last_idx

            input, target = get_input(cfg, data)

            if not args.prefetcher:
                input = input.to(device)
                target = target.to(device)
            if args.channels_last:
                input = input.contiguous(memory_format=torch.channels_last)

            with amp_autocast():
                output = model(input)
                if isinstance(output, (tuple, list)):
                    output = output[0]

                # augmentation reduction
                reduce_factor = args.tta
                if reduce_factor > 1:
                    output = output.unfold(0, reduce_factor, reduce_factor).mean(dim=2)
                    target = target[0 : target.size(0) : reduce_factor]

                loss = loss_fn(output, target)
            acc1, acc5 = utils.accuracy(output, target, topk=(1, 5))

            if args.distributed:
                reduced_loss = utils.reduce_tensor(loss.data, args.world_size)
                acc1 = utils.reduce_tensor(acc1, args.world_size)
                acc5 = utils.reduce_tensor(acc5, args.world_size)
            else:
                reduced_loss = loss.data

            if device.type == "cuda":
                torch.cuda.synchronize()

            losses_m.update(reduced_loss.item(), input.size(0))
            top1_m.update(acc1.item(), output.size(0))
            top5_m.update(acc5.item(), output.size(0))

            batch_time_m.update(time.time() - end)
            end = time.time()
            if utils.is_primary(args) and (
                last_batch or batch_idx % args.log_interval == 0
            ):
                log_name = "Test" + log_suffix
                _logger.info(
                    f"{log_name}: [{batch_idx:>4d}/{last_idx}]  "
                    f"Time: {batch_time_m.val:.3f} ({batch_time_m.avg:.3f})  "
                    f"Loss: {losses_m.val:>7.3f} ({losses_m.avg:>6.3f})  "
                    f"Acc@1: {top1_m.val:>7.3f} ({top1_m.avg:>7.3f})  "
                    f"Acc@5: {top5_m.val:>7.3f} ({top5_m.avg:>7.3f})"
                )

    metrics = OrderedDict(
        [("loss", losses_m.avg), ("top1", top1_m.avg), ("top5", top5_m.avg)]
    )

    return metrics


if __name__ == "__main__":
    main()
