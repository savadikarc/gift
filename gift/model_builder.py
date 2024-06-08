import logging
import loralib as lora
from .gift import GIFTWrapperForImageClassification, GIFTConfig, BLOCK_PARAMS
from .bitfit_wrapper import BitFiTWrapper
from .prompt_wrapper import PromptWrapper


_logger = logging.getLogger(__name__)

def build_lora(backbone, args):
    backbone.requires_grad_(False)
    for block in backbone.blocks:
        if "qkv" in args.gift_target_modules:
            _logger.info("Using LoRA for QKV")
            old_weight = block.attn.qkv.weight.data
            old_bias = block.attn.qkv.bias.data
            dim = block.attn.qkv.weight.shape[1]
            block.attn.qkv = lora.Linear(dim, dim*3, r=4)
            block.attn.qkv.weight.data = old_weight
            block.attn.qkv.bias.data = old_bias
        if "proj" in args.gift_target_modules or "attn:proj" in args.gift_target_modules:
            _logger.info(f"Using LoRA for projection matrix, rank={args.lora_rank}")
            old_weight = block.attn.proj.weight.data
            old_bias = block.attn.proj.bias.data
            dim = block.attn.proj.weight.shape[1]
            block.attn.proj = lora.Linear(dim, dim, r=args.lora_rank)
            block.attn.proj.weight.data = old_weight
            block.attn.proj.bias.data = old_bias
        if "q" in args.gift_target_modules and "v" in args.gift_target_modules:
            _logger.info(f"Using LoRA for Q and V, rank={args.lora_rank}")
            dim = block.attn.qkv.weight.shape[1]
            old_weight = block.attn.qkv.weight.data
            old_bias = block.attn.qkv.bias.data
            block.attn.qkv = lora.MergedLinear(dim, dim*3, r=args.lora_rank, enable_lora=[True, False, True])
            block.attn.qkv.weight.data = old_weight
            block.attn.qkv.bias.data = old_bias
        elif "v" in args.gift_target_modules:
            _logger.info(f"Using LoRA for V, rank={args.lora_rank}")
            dim = block.attn.qkv.weight.shape[1]
            old_weight = block.attn.qkv.weight.data
            old_bias = block.attn.qkv.bias.data
            block.attn.qkv = lora.MergedLinear(dim, dim*3, r=args.lora_rank, enable_lora=[False, False, True])
            block.attn.qkv.weight.data = old_weight
            block.attn.qkv.bias.data = old_bias
        if "fc1" in args.gift_target_modules:
            _logger.info(f"Using LoRA for FC1, rank={args.lora_rank}")
            old_weight = block.mlp.fc1.weight.data
            old_bias = block.mlp.fc1.bias.data
            din = old_weight.shape[1]
            dout = old_weight.shape[0]
            block.mlp.fc1 = lora.Linear(din, dout, r=args.lora_rank)
            block.mlp.fc1.weight.data = old_weight
            block.mlp.fc1.bias.data = old_bias
    lora.mark_only_lora_as_trainable(backbone)

    backbone.head.requires_grad_(True)

    return backbone

def build_gift(backbone, args):

    block_params = {
        k.replace("gift_block_", ""): v for k, v in vars(args).items() if k.startswith("gift_block")
    }
    # Kep only the params that are needed for the current block type
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
    model = GIFTWrapperForImageClassification(
        config,
        backbone, 
    )
    if args.reset_classifier:
        backbone.reset_classifier(args.num_classes)
    return model


def build_bitfit(backbone, args):
    backbone.requires_grad_(False)
    model = BitFiTWrapper(backbone)
    backbone.head.requires_grad_(True)
    return model


def build_prompt(backbone, args):
    backbone.requires_grad_(False)
    model = PromptWrapper(backbone, backbone.patch_embed.patch_size, args.num_prompt_tokens, backbone.embed_dim, args.deep_prompts)
    backbone.head.requires_grad_(True)
    return model


def build_model(args, backbone):
    
    if args.method == "lora":
        model = build_lora(backbone, args)
    elif args.method == "gift":
        model = build_gift(backbone, args)
    elif args.method == "bitfit":
        model = build_bitfit(backbone, args)
    elif args.method == "vpt":
        model = build_prompt(backbone, args)
    else:
        raise NotImplementedError(f"Unknown method {args.method}")

    return model
