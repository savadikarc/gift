from .datasets_base import LMDataset
from .commonsense_170k import *
from .math import *

DATASETS = {
    "meta-math/MetaMathQA": MetaMathQA,
    "openai/gsm8k": GSM8K,
    "gsm8k": GSM8kJSON,
    "math10k": Math10k,
    "SVAMP": SVAMP,
    "AQuA": AQuA,
    "mawps": MAWPS,
    "commonsense_170k": Commonsense170K,
    "boolq": BoolQ,
    "piqa": PiQA,
    "social_i_qa": SocialIQA,
    "hellaswag": Hellaswag,
    "winogrande": Winogrande,
    "ARC-Easy": ARCEasy,
    "ARC-Challenge": ARCChallenge,
    "openbookqa": OpenBookQA,
}
