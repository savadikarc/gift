import argparse
import logging
import os
import time
from collections import OrderedDict, Counter
from contextlib import suppress
from datetime import datetime
from functools import partial
from pathlib import Path

import torch
import torch.nn as nn
import yaml

from timm import utils
from timm.data import resolve_data_config
from timm.layers import set_fast_norm
from timm.models import (
    create_model,
    safe_model_name,
)
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

# Augmentation & regularization parameters
group = parser.add_argument_group("Augmentation and regularization parameters")
group.add_argument(
    "--drop", type=float, default=0.0, metavar="PCT", help="Dropout rate (default: 0.)"
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
    "-j",
    "--workers",
    type=int,
    default=4,
    metavar="N",
    help="how many training processes to use (default: 4)",
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
group.add_argument("--local_rank", default=0, type=int)
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
    "--data-config", type=str, default="", help="Config for dataloaders."
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

# Hypernet Block parameters
group = parser.add_argument_group("Hypernet Block parameters")
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

group.add_argument(
    "--reset_classifier",
    action="store_true",
    default=False,
    help="Reset the classifier. Not used in the script, but required for compatibility with the model_builder.",
)

parser.add_argument(
    "dataopts",
    help="Modify config options using the command-line",
    default=None,
    nargs=argparse.REMAINDER,
)


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

    cfg.DATA.NUMBER_CLASSES = BENCHMARK_NUM_CLASSES[cfg.DATA.NAME]
    cfg.DATA.BATCH_SIZE = args.batch_size

    args.num_classes = cfg.DATA.NUMBER_CLASSES

    if args.method == "lora":
        method_suffix = f"-rank-{args.lora_rank}"
    else:
        method_suffix = ""

    if args.experiment:
        exp_name = args.experiment
    else:
        eval_type = "test" if args.evaluate else "val"
        gift_modules = "-".join(args.gift_modules)
        exp_name = "-".join(
            [
                args.method+method_suffix,
                gift_modules,
                eval_type,
                str(args.lr_base),
                str(args.lr_base),
                str(args.weight_decay),
                str(args.gift_downsample_ratio),
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
    device = utils.init_distributed_device(args)

    log_path = os.path.join(output_dir, f"validation.log")
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
        scriptable=False,
        **args.model_kwargs,
    )
    if args.head_init_scale is not None:
        with torch.no_grad():
            backbone.get_classifier().weight.mul_(args.head_init_scale)
            backbone.get_classifier().bias.mul_(args.head_init_scale)
    if args.head_init_bias is not None:
        nn.init.constant_(backbone.get_classifier().bias, args.head_init_bias)

    if args.initial_checkpoint:
        state_dict = torch.load(args.initial_checkpoint, map_location="cpu")["model"]
        # Remove the head
        state_dict.pop("head.weight", None)
        state_dict.pop("head.bias", None)
        keys = backbone.load_state_dict(state_dict, strict=False)
        _logger.info(f"Loaded {args.initial_checkpoint}")
        print(keys)
    
    backbone.requires_grad_(False)

    model = build_model(args, backbone)
    _logger.info(f"{model}")

    # TODO: Load checkpoint
    checkpoint_path = Path(output_dir, "last.pth.tar")
    state_dict = torch.load(checkpoint_path, map_location="cpu")["state_dict"]
    key_info = model.load_state_dict(state_dict, strict=False)
    _logger.info(f"Loaded {checkpoint_path}")
    _logger.info(f"Key Info: {key_info}")

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

    if utils.is_primary(args):
        _logger.info(
            f"Model {safe_model_name(args.model)} created, param count:{sum([m.numel() for m in model.parameters()])}"
        )

    # move model to GPU, enable channels last layout if set
    model.to(device=device)
    if args.channels_last:
        model.to(memory_format=torch.channels_last)

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
    
    validate_loss_fn = nn.CrossEntropyLoss().to(device=device)

    try:
        visualization_class_counter = Counter()
        eval_metrics = validate(
            model,
            visualization_class_counter,
            loader_eval,
            validate_loss_fn,
            args,
            cfg,
            amp_autocast=amp_autocast,
        )

        print(eval_metrics)

    except KeyboardInterrupt:
        pass

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


def validate(
    model,
    vis_counter,
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
                # output = model(input, use_hypernet=False)
                output = model(input)
                if isinstance(output, (tuple, list)):
                    output = output[0]

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
