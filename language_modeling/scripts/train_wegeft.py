from dataclasses import dataclass, field
from typing import List
from transformers import HfArgumentParser
from peft import WeGeFTConfig

from train import finetune, ExtendedTrainingArguments

@dataclass
class WeGeFTTrainingArguments(ExtendedTrainingArguments):
    tuner: str = "wegeft"
    target_modules: str = "q_proj,k_proj,v_proj,up_proj,down_proj"
    tied_modules: str = '[["model.layers.\d+.self_attn.q_proj"],["model.layers.\d+.self_attn.k_proj"],["model.layers.\d+.self_attn.v_proj"],["model.layers.\d+.mlp.up_proj"],["model.layers.\d+.mlp.down_proj"]]'
    transform_dim = '{"q_proj": "input", "k_proj": "input", "v_proj": "input", "up_proj": "input", "down_proj": "input"}'
    wegeft_rank: int = 64
    wegeft_alpha: float = 128
    wegeft_dropout: float = 0.0


def main():

    parser = HfArgumentParser(WeGeFTTrainingArguments)
    args: WeGeFTTrainingArguments = parser.parse_args_into_dataclasses()[0]

    # Make the peft config
    target_modules = args.target_modules.split(",")
    tied_modules = eval(args.tied_modules)
    transform_dim = eval(args.transform_dim)
    peft_config = WeGeFTConfig(
        target_modules=target_modules,
        tied_modules=tied_modules,
        transform_dim=transform_dim,
        r=args.wegeft_rank,
        wegeft_alpha=args.wegeft_alpha,
        wegeft_dropout=args.wegeft_dropout,
        task_type="CAUSAL_LM",
    )

    finetune(args, peft_config=peft_config)


if __name__ == "__main__":
    main()