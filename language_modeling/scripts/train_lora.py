from dataclasses import dataclass, field
from typing import List
from transformers import HfArgumentParser
from peft import LoraConfig

from train import finetune, ExtendedTrainingArguments

@dataclass
class LoraTrainingArguments(ExtendedTrainingArguments):
    tuner: str = "lora"
    lora_rank: int = 16
    target_modules: str = "q_proj,k_proj,v_proj,up_proj,down_proj"
    lora_dropout: float = 0.0
    lora_alpha: int = 32
    use_dora: bool = False


def main():

    parser = HfArgumentParser(LoraTrainingArguments)
    args = parser.parse_args_into_dataclasses()[0]

    # Initialization
    init_lora_weights = True

    # Make the peft config
    peft_config = LoraConfig(
        r=args.lora_rank,
        target_modules=args.target_modules.split(","),
        lora_dropout=args.lora_dropout,
        lora_alpha=args.lora_alpha,
        use_dora=args.use_dora,
        init_lora_weights=init_lora_weights,
        task_type="CAUSAL_LM",
    )

    finetune(args, peft_config=peft_config)


if __name__ == "__main__":
    main()