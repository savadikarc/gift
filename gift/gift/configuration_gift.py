from typing import Dict
from typing import List, Optional, Union

from transformers import PretrainedConfig


TRANSFORMER_PARAMS = dict(
    block_type="transformer",
    num_blocks=1,
    num_heads=1,
    mlp_ratio=2.,
    drop_path=0.0,
    norm_layer="l2",
)
PAMCAT_PARAMS = dict(
    **TRANSFORMER_PARAMS, 
    num_clusters=64, 
    cluster_activation="sigmoid"
)
PAMCAT_PARAMS["block_type"] = "pamcat_transformer"
PAMCAT_ATTN_PARAMS = dict(
    block_type="pamcat_attn",
    num_blocks=1,
    num_heads=1,
    norm_layer="l2",
    num_clusters=64, 
    cluster_activation="sigmoid"
)
MLP_MIXER_PARAMS = dict(
    block_type="mlp_mixer", 
    num_blocks=1,
    num_mixed_tokens=64,
    channel_mixing_ratio=2.,
    drop_path=0.0,
    norm_layer="l2",
)
MLP_PARAMS = dict(
    block_type="mlp", 
    num_blocks=1,
    mlp_ratio=2.,
    drop_path=0.0,
    norm_layer="l2",
)
SIMPLE_BLOCK_PARAMS = dict(
    block_type="simple_block",
    act_layer="identity",
)

BLOCK_PARAMS = {
    "transformer": TRANSFORMER_PARAMS,
    "pamcat_transformer": PAMCAT_PARAMS,
    "pamcat_attn": PAMCAT_ATTN_PARAMS,
    "mlp_mixer": MLP_MIXER_PARAMS,
    "mlp": MLP_PARAMS,
    "simple_block": SIMPLE_BLOCK_PARAMS,
}


class GIFTConfig(PretrainedConfig):
    model_type = 'gift'
    _auto_class = 'AutoConfig'

    def __init__(
            self,
            rank: int = 16,
            dtype: str = "float32",
            gift_parameters: Optional[Dict] = SIMPLE_BLOCK_PARAMS,
            in_projection_bias: bool = False,
            out_projection_bias: bool = False,
            target_modules: Optional[Union[List[str], str]] = None,
            enable_gift: Optional[Dict[str, bool]] = None,
            share_projections: bool = False,
            layers_to_transform: Optional[Union[List[int], int]] = None,
            layers_pattern: Optional[Union[List[str], str]] = None,
            **kwargs
    ):
        self.rank = rank
        self.dtype = dtype
        self.gift_parameters = gift_parameters
        self.in_projection_bias = in_projection_bias
        self.out_projection_bias = out_projection_bias
        self.target_modules = target_modules
        self.enable_gift = enable_gift
        self.share_projections = share_projections
        self.layers_to_transform = layers_to_transform
        self.layers_pattern = layers_pattern
        super().__init__(**kwargs)

        self.validate()

    def validate(self):
        self.target_modules = (
            sorted(list(set(self.target_modules))) if isinstance(self.target_modules, list) else self.target_modules
        )
        # if target_modules is a regex expression, then layers_to_transform should be None
        if isinstance(self.target_modules, str) and self.layers_to_transform is not None:
            raise ValueError("`layers_to_transform` cannot be used when `target_modules` is a str.")

        # if target_modules is a regex expression, then layers_pattern should be None
        if isinstance(self.target_modules, str) and self.layers_pattern is not None:
            raise ValueError("`layers_pattern` cannot be used when `target_modules` is a str.")
