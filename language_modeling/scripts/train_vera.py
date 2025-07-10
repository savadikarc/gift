from dataclasses import dataclass
from typing import List
from transformers import HfArgumentParser
from peft import VeraConfig

from train import finetune, ExtendedTrainingArguments

@dataclass
class LoraTrainingArguments(ExtendedTrainingArguments):
    tuner: str = "vera"
    vera_rank: int = 1024
    target_modules: str = "q_proj,k_proj,v_proj,up_proj,down_proj"
    vera_dropout: float = 0.0


def main():

    parser = HfArgumentParser(LoraTrainingArguments)
    args = parser.parse_args_into_dataclasses()[0]

    # Make the peft config
    peft_config = VeraConfig(
        r=args.vera_rank,
        target_modules=args.target_modules.split(","),
        vera_dropout=args.vera_dropout,
        task_type="CAUSAL_LM",
    )

    finetune(args, peft_config=peft_config)


if __name__ == "__main__":
    main()