#!/usr/bin/env python3

"""Config system (based on Detectron's)."""

import sys
from .config_node import CfgNode


# Global config object
_C = CfgNode()
# Example usage:
#   from configs.config import cfg

_C.DBG = False
_C.OUTPUT_DIR = "./output"
_C.RUN_N_TIMES = 5
# Perform benchmarking to select the fastest CUDNN algorithms to use
# Note that this may increase the memory usage and will likely not result
# in overall speedups when variable size inputs are used (e.g. COCO training)
_C.CUDNN_BENCHMARK = False

# Number of GPUs to use (applies to both training and testing)
_C.NUM_GPUS = 1
_C.NUM_SHARDS = 1

# Note that non-determinism may still be present due to non-deterministic
# operator implementations in GPU operator libraries
_C.SEED = None
_C.RETURN_VISUALIZATION = False

# ----------------------------------------------------------------------
# Model options
# ----------------------------------------------------------------------
_C.MODEL = CfgNode()
_C.MODEL.TRANSFER_TYPE = "linear"  # one of linear, end2end, prompt, adapter, side, partial-1, tinytl-bias
_C.MODEL.WEIGHT_PATH = ""  # if resume from some checkpoint file
_C.MODEL.TEST_WEIGHT_PATH = ""   # load trained weights for test
_C.MODEL.SAVE_CKPT = True
_C.MODEL.SAVE_MODEL_CKPT = False

_C.MODEL.MODEL_ROOT = ""  # root folder for pretrained model weights

_C.MODEL.TYPE = "vit"
_C.MODEL.SIZE = 'base'
_C.MODEL.MLP_NUM = 0

_C.MODEL.LINEAR = CfgNode()
_C.MODEL.LINEAR.MLP_SIZES = []
_C.MODEL.LINEAR.DROPOUT = 0.1

# ----------------------------------------------------------------------
# Prompt options
# ----------------------------------------------------------------------
_C.MODEL.PROMPT = CfgNode()
_C.MODEL.PROMPT.NUM_TOKENS = 100
_C.MODEL.PROMPT.LOCATION = "prepend"
# prompt initalizatioin: 
    # (1) default "random"
    # (2) "final-cls" use aggregated final [cls] embeddings from training dataset
    # (3) "cls-nolastl": use first 12 cls embeddings (exclude the final output) for deep prompt
    # (4) "cls-nofirstl": use last 12 cls embeddings (exclude the input to first layer)
_C.MODEL.PROMPT.INITIATION = "random"  # "final-cls", "cls-first12"
_C.MODEL.PROMPT.CLSEMB_FOLDER = ""
_C.MODEL.PROMPT.CLSEMB_PATH = ""
_C.MODEL.PROMPT.PROJECT = -1  # "projection mlp hidden dim"
_C.MODEL.PROMPT.DEEP = True # "whether do deep prompt or not, only for prepend location"


_C.MODEL.PROMPT.NUM_DEEP_LAYERS = None  # if set to be an int, then do partial-deep prompt tuning
_C.MODEL.PROMPT.REVERSE_DEEP = False  # if to only update last n layers, not the input layer
_C.MODEL.PROMPT.DEEP_SHARED = False  # if true, all deep layers will be use the same prompt emb
_C.MODEL.PROMPT.FORWARD_DEEP_NOEXPAND = False  # if true, will not expand input sequence for layers without prompt
# how to get the output emb for cls head:
    # original: follow the orignial backbone choice,
    # img_pool: image patch pool only
    # prompt_pool: prompt embd pool only
    # imgprompt_pool: pool everything but the cls token
_C.MODEL.PROMPT.VIT_POOL_TYPE = "original"
_C.MODEL.PROMPT.DROPOUT = 0.0
_C.MODEL.PROMPT.SAVE_FOR_EACH_EPOCH = False
# ----------------------------------------------------------------------
# adapter options
# ----------------------------------------------------------------------
_C.MODEL.ADAPTER = CfgNode()
_C.MODEL.ADAPTER.REDUCATION_FACTOR = 8
_C.MODEL.ADAPTER.STYLE = "Pfeiffer"

# ----------------------------------------------------------------------
# Solver options
# ----------------------------------------------------------------------
_C.SOLVER = CfgNode()
_C.SOLVER.LOSS = "softmax"
_C.SOLVER.LOSS_ALPHA = 0.01

_C.SOLVER.OPTIMIZER = "sgd"  # or "adamw"
_C.SOLVER.MOMENTUM = 0.9
_C.SOLVER.WEIGHT_DECAY = 0.0001
_C.SOLVER.WEIGHT_DECAY_BIAS = 0

_C.SOLVER.PATIENCE = 300


_C.SOLVER.SCHEDULER = "cosine"

_C.SOLVER.BASE_LR = 0.01
_C.SOLVER.BIAS_MULTIPLIER = 1.              # for prompt + bias

_C.SOLVER.WARMUP_EPOCH = 5
_C.SOLVER.TOTAL_EPOCH = 30
_C.SOLVER.LOG_EVERY_N = 1000


_C.SOLVER.DBG_TRAINABLE = False # if True, will print the name of trainable params
_C.SOLVER.PRIOR_DECAY = True # if False, no weight decay on global prior in top-down tuning

# ----------------------------------------------------------------------
# Dataset options
# ----------------------------------------------------------------------
_C.DATA = CfgNode()

_C.DATA.NAME = ""
_C.DATA.DATAPATH = ""
_C.DATA.FEATURE = ""  # e.g. inat2021_supervised

_C.DATA.PERCENTAGE = 1.0
_C.DATA.NUMBER_CLASSES = -1
_C.DATA.MULTILABEL = False
_C.DATA.CLASS_WEIGHTS_TYPE = "none"

_C.DATA.CROPSIZE = 224  # or 384

_C.DATA.NO_TEST = False
_C.DATA.BATCH_SIZE = 32
# Number of data loader workers per training process
_C.DATA.NUM_WORKERS = 4
# Load data to pinned host memory
_C.DATA.PIN_MEMORY = True

_C.DATA.CORRUPTION = 'gaussian_noise'
# _C.DATA.CORRUPTION = None
_C.DATA.SEVERITY = 0

_C.DIST_BACKEND = "nccl"
_C.DIST_INIT_PATH = "env://"
_C.DIST_INIT_FILE = ""

# ----------------------------------------------------------------------
# Number of classes per dataset for convenience
# ----------------------------------------------------------------------
BENCHMARK_NUM_CLASSES = {
    "vtab-cifar(num_classes=100)": 100,
    "vtab-caltech101": 102,
    "vtab-dtd": 47,
    "vtab-oxford_flowers102": 102,
    "vtab-oxford_iiit_pet": 37,
    "vtab-svhn": 10,
    "vtab-sun397": 397,
    "vtab-patch_camelyon": 2,
    "vtab-eurosat": 10,
    "vtab-resisc45": 45,
    "vtab-diabetic_retinopathy(config=\"btgraham-300\")": 5,
    "vtab-clevr(task=\"count_all\")": 8,
    "vtab-clevr(task=\"closest_object_distance\")": 6,
    "vtab-dmlab": 6,
    "vtab-kitti(task=\"closest_vehicle_distance\")": 4,
    "vtab-dsprites(predicted_attribute=\"label_x_position\",num_classes=16)": 16,
    "vtab-dsprites(predicted_attribute=\"label_orientation\",num_classes=16)": 16,
    "vtab-smallnorb(predicted_attribute=\"label_azimuth\")": 18,
    "vtab-smallnorb(predicted_attribute=\"label_elevation\")": 9,
    "CUB": 200,
    "nabirds": 555,
    "StanfordCars": 196,
    "StanfordDogs": 120,
    "OxfordFlowers": 102,
}

def get_cfg():
    """
    Get a copy of the default config.
    """
    return _C.clone()
