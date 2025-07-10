from collections import OrderedDict
import logging
import re
from typing import Any, Mapping, Union, List, Optional
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

# from timm.models.vision_transformer import VisionTransformer
from transformers import PreTrainedModel
from .hypernet import Hypernet, BLOCK_FNS

from .configuration_wegeft import WeGeFTConfig

_logger = logging.getLogger(__name__)


class WeGeFTLinear(nn.Linear):
    # WeGeFT implemented in a dense layer
    def __init__(
        self, 
        in_features: int, 
        out_features: int, 
        bias: bool = True,
        device=None,
        dtype=None,
        name=None,
    ):
        nn.Linear.__init__(self, in_features, out_features, bias=bias, device=device, dtype=dtype)

        # Freezing the pre-trained weight matrix
        self.weight.requires_grad = False
        if bias:
            self.bias.requires_grad = False
        self.merged = False
        self.name = name
        self.pre_delta_identity = nn.Identity()

    @torch.no_grad()
    def merge(self, weight=None, bias=None):
        raise NotImplementedError("merge method not implemented yet")
    
    def forward(self, x: torch.Tensor, delta_weight=None, scale=None):
        # print(f"Layer {self.name} delta_weight: {delta_weight.sum().item()}")
        result = F.linear(x, self.weight, bias=self.bias)
        result = self.pre_delta_identity(result)
        if delta_weight is not None:
            residual = F.linear(x, delta_weight, bias=None)
            result += residual
        if scale is not None:
            result *= scale
        return result
    

class WeGeFTMergedLinear(WeGeFTLinear):
    # WeGeFT implemented in a dense layer
    def __init__(
        self, 
        in_features: int, 
        out_features: int, 
        bias: bool = True,
        device=None,
        dtype=None,
        enable=[True],
        name=None,
    ):
        super().__init__(in_features, out_features, bias=bias, device=device, dtype=dtype, name=name)

        self.enable = enable
        self._out_features = out_features // len(enable)

    @torch.no_grad()
    def merge(self, weight=None, bias=None):
        raise NotImplementedError("merge method not implemented yet")
    
    def get_weight(self, idx):
        start = idx * self._out_features
        end = (idx + 1) * self._out_features
        return self.weight.data[start:end, :]


# Modified from huggingface peft
def check_target_module_exists(config, target_key, key: str):
    """A helper method to check if the passed module's key name matches any of the target modules in the adapter_config.

    Args:
        config (`LoraConfig` | `LycorisConfig`): A config to match target modules from
        key (`str`): A key to search any matches in config

    Returns:
        `bool` | `re.Match[str]` | `None`: True of match object if key matches any target modules from config, False or
        None if no match found
    """
    # target_module_found = re.fullmatch(target_key, key)
    # if:
    target_module_found = key.endswith(f".{target_key}")

    layer_indexes = getattr(config, "layers_to_transform", None)
    layers_pattern = getattr(config, "layers_pattern", None)

    is_using_layer_indexes = layer_indexes is not None and (
        len(layer_indexes) != 0 if isinstance(layer_indexes, list) else True
    )
    if is_using_layer_indexes and target_module_found:
        layer_index = None
        # TODO: It's still unclear how empty layers_pattern (None, [], or "") should behave
        # For now, empty layers_pattern means any layer pattern is ok
        if layers_pattern is None or len(layers_pattern) == 0:
            layer_index = re.match(r".*\.[^.]*\.(\d+)\.", key)
        else:
            layers_pattern = [layers_pattern] if isinstance(layers_pattern, str) else layers_pattern
            for pattern in layers_pattern:
                layer_index = re.match(rf".*\.{pattern}\.(\d+)\.", key)
                if layer_index is not None:
                    break

        if layer_index is None:
            target_module_found = False
        else:
            layer_index = int(layer_index.group(1))
            if isinstance(layer_indexes, int):
                target_module_found = layer_index == layer_indexes
            else:
                target_module_found = layer_index in layer_indexes

    return target_module_found


def _get_submodules(model, key):
    parent = model.get_submodule(".".join(key.split(".")[:-1]))
    target_name = key.split(".")[-1]
    target = model.get_submodule(key)
    return parent, target, target_name


dtype_mapping = {
    "float32": torch.float32,
    "float16": torch.float16,
    "bfloat16": torch.bfloat16,
    "float8": "float8",
}


class WeGeFTWrapper(nn.Module):

    def __init__(
            self, 
            config: WeGeFTConfig,
            backbone: Union[PreTrainedModel, nn.Module],
        ) -> None:
        super().__init__()

        self.config = config
        self.backbone = backbone
        self.backbone.requires_grad_(False)

        self.wegeft_dtype = getattr(config, "dtype", None)
        if self.wegeft_dtype is not None:
            self.wegeft_dtype = dtype_mapping[self.wegeft_dtype]

        self.num_layers = self.init_wegeft(config, backbone)

    def _init_wegeft_for_single_module(self, config: WeGeFTConfig, model, target_key):

        wegeft_block_fn = BLOCK_FNS[config.wegeft_parameters["block_type"]]
        wegeft_block_parameters = {k:v for k, v in config.wegeft_parameters.items() if k != "block_type"}

        key_list = [key for key, _ in model.named_modules()]
        
        _target_key = target_key.replace(":", ".")
        target_modules = [_get_submodules(model, key) for key in key_list if check_target_module_exists(config, _target_key, key)]

        module_in_features = set()
        module_out_features = set()
        num_layers = 0
        hook_handles = []
        for layer, (parent, target, target_name) in enumerate(target_modules):
            assert isinstance(target, nn.Linear), f"Only linear layers are supported for now, got {type(target)}"
            
            in_features = target.in_features
            out_features = target.out_features
            module_in_features.add(in_features)
            module_out_features.add(out_features)
            assert len(module_in_features) == 1, f"{target_key} Only isotropic layers are supported for now, got {module_in_features}"
            assert len(module_out_features) == 1, f"{target_key} Only isotropic layers are supported for now, got {module_out_features}"
            
            num_layers += 1

            # Replace the target module with WeGeFTLinear
            new_module = self.create_and_replace_module(parent, target_name, target)

            # Attach hooks
            _logger.info(f"Attaching hook to {target_key}, {target_name}, layer {layer}")
            handle = new_module.register_forward_pre_hook(self.residual_forward_hook(target_key, layer), with_kwargs=True)
            hook_handles.append(handle)

        self.hook_handles[target_key] = hook_handles

        in_features = module_in_features.pop()
        out_features = module_out_features.pop()
        if not config.share_projections:
            self.in_projections[target_key] = self.get_projection(in_features, config.rank, config.in_projection_bias)
            self.out_projections[target_key] = self.get_projection(config.rank, in_features, config.out_projection_bias, zero_init=True)
            if config.wegeft_parameters["block_type"] == "mlp_mixer":
                wegeft_block_parameters["num_tokens"] = out_features
            self.hypernets[target_key] = Hypernet(wegeft_block_fn, rank=config.rank, **wegeft_block_parameters)

        return num_layers, in_features, out_features
    
    def _init_wegeft_for_merged_modules(self, config: WeGeFTConfig, model, target_key, enable_wegeft):

        wegeft_block_fn = BLOCK_FNS[config.wegeft_parameters["block_type"]]
        wegeft_block_parameters = {k:v for k, v in config.wegeft_parameters.items() if k != "block_type"}

        key_list = [key for key, _ in model.named_modules()]
        
        _target_key = target_key.replace(":", ".")
        target_modules = [_get_submodules(model, key) for key in key_list if check_target_module_exists(config, _target_key, key)]

        try:
            base, target_key_components = target_key.rsplit(":", maxsplit=1)
            target_key_components = list(target_key_components)
            target_sub_keys = [":".join(base, target_sub_key) for target_sub_key in target_key_components]
        except ValueError:
            base = ""
            target_key_components = target_key
            target_key_components = target_sub_keys = list(target_key)

        num_layers = 0
        module_in_features = {sub_key: set() for sub_key, enable in zip(target_sub_keys, enable_wegeft) if enable}
        module_out_features = {sub_key: set() for sub_key, enable in zip(target_sub_keys, enable_wegeft) if enable}
        hypernets = {sub_key: nn.ModuleList() for sub_key, enable in zip(target_sub_keys, enable_wegeft) if enable}
        hook_handles = []
        for layer, (parent, target, target_name) in enumerate(target_modules):
            assert isinstance(target, nn.Linear), f"Only linear layers are supported for now, got {type(target)}"
            
            in_features = target.in_features
            out_features = target.out_features // len(target_sub_keys)
            for sub_key, enabled in zip(target_sub_keys, enable_wegeft):
                if not enabled:
                    continue
                module_in_features[sub_key].add(in_features)
                module_out_features[sub_key].add(out_features)
                assert len(module_in_features[sub_key]) == 1, f"{target_key} -> {sub_key} Only isotropic layers are supported for now, got {module_in_features[sub_key]}"
                assert len(module_out_features[sub_key]) == 1, f"{target_key} -> {sub_key} Only isotropic layers are supported for now, got {module_out_features[sub_key]}"
            
            num_layers += 1

            # Replace the target module with WeGeFTLinear
            new_module = self.create_and_replace_module(parent, target_name, target, enable_wegeft=enable_wegeft)

            # Attach hooks
            _logger.info(f"Attaching hook to {target_key}, {target_name}, layer {layer}")
            handle = new_module.register_forward_pre_hook(self.residual_forward_hook_merged(target_key, target_sub_keys, enable_wegeft, layer), with_kwargs=True)
            hook_handles.append(handle)

            # Initialize the hypernet for the layer
            if config.wegeft_parameters["block_type"] == "mlp_mixer":
                wegeft_block_parameters["num_tokens"] = out_features
            for sub_key, enabled in zip(target_sub_keys, enable_wegeft):
                if not enabled:
                    continue
                hypernets[sub_key].append(Hypernet(wegeft_block_fn, rank=config.rank, **wegeft_block_parameters))

        self.hook_handles[target_key] = hook_handles
        for sub_key, enable in zip(target_sub_keys, enable_wegeft):
            if not enable:
                continue
            self.hypernets[sub_key] = hypernets[sub_key]

            in_features = module_in_features[sub_key].pop()
            out_features = module_out_features[sub_key].pop()
            
            if not config.share_projections:
                self.in_projections[sub_key] = self.get_projection(in_features, config.rank, config.in_projection_bias)
                self.out_projections[sub_key] = self.get_projection(config.rank, out_features, config.out_projection_bias, zero_init=True)

        return num_layers, in_features, out_features

    def init_wegeft(self, config, model):
        config = self.config

        self.in_projections = nn.ModuleDict()
        self.hypernets = nn.ModuleDict()
        self.out_projections = nn.ModuleDict()
        self.hook_handles = dict()

        for target_key in config.target_modules:
            enable_wegeft = None
            if config.enable_wegeft is not None:
                enable_wegeft = config.enable_wegeft.get(target_key, None)

            if enable_wegeft is None:
                num_layers, in_features, out_features = self._init_wegeft_for_single_module(config, model, target_key)
            else:
                num_layers, in_features, out_features = self._init_wegeft_for_merged_modules(config, model, target_key, enable_wegeft)

            if config.share_projections:
                self.in_projections["shared_projection"] = self.get_projection(in_features, config.rank, config.in_projection_bias)
                self.out_projections["shared_projection"] = self.get_projection(config.rank, in_features, config.out_projection_bias, zero_init=True)

                wegeft_block_fn = BLOCK_FNS[config.wegeft_parameters["block_type"]]
                wegeft_block_parameters = {k:v for k, v in config.wegeft_parameters.items() if k != "block_type"}

                if config.wegeft_parameters["block_type"] == "mlp_mixer":
                    wegeft_block_parameters["num_tokens"] = out_features
                self.hypernets["shared_projection"] = Hypernet(wegeft_block_fn, rank=config.rank, **wegeft_block_parameters)

        # Cast the model to the hypernet dtype
        if self.wegeft_dtype is not None:
            self.in_projections.to(dtype=self.wegeft_dtype)
            self.out_projections.to(dtype=self.wegeft_dtype)
            self.hypernets.to(dtype=self.wegeft_dtype)

        return num_layers
    
    def get_projection(self, in_features, out_features, bias, zero_init=False):
        projection = nn.Linear(in_features, out_features, bias=bias)
        if zero_init:
            nn.init.zeros_(projection.weight)
            if bias:
                nn.init.zeros_(projection.bias)
        return projection
    
    def create_and_replace_module(self, parent, target_name, target, enable_wegeft=None):
        new_module = self.create_new_module(target, target_name, enable_wegeft=enable_wegeft)
        self._replace_module(parent, target_name, new_module)
        return new_module
    
    def _replace_module(self, parent, target_name, new_module):
        setattr(parent, target_name, new_module)

    def create_new_module(self, target, target_name, enable_wegeft=None):
        in_features = target.in_features
        out_features = target.out_features
        if enable_wegeft is None:
            new_module = WeGeFTLinear(in_features, out_features, bias=target.bias is not None, name=target_name)
        else:
            new_module = WeGeFTMergedLinear(in_features, out_features, bias=target.bias is not None, enable=enable_wegeft, name=target_name)
        if new_module.weight.dtype != target.weight.dtype:
            new_module = new_module.to(dtype=target.weight.dtype)
        new_module.weight.data.copy_(target.weight)
        if target.bias is not None:
            new_module.bias.data.copy_(target.bias)
        return new_module
    
    @torch.cuda.amp.autocast(enabled=False)
    def _wegeft_forward(self, target_key, layer, weight):
        cast_weights = self.wegeft_dtype is not None and weight.dtype != self.wegeft_dtype
        if cast_weights:
            org_type = weight.dtype
            weight = weight.to(dtype=self.wegeft_dtype)
        projection_key = target_key if not self.config.share_projections else "shared_projection"
        compressed_weight = self.in_projections[projection_key](weight)
        compressed_weight = self.hypernets[projection_key](compressed_weight)
        delta_weight = self.out_projections[projection_key](compressed_weight)

        if cast_weights:
            delta_weight = delta_weight.to(dtype=org_type)

        return delta_weight
    
    def residual_forward_hook(self, target_key, layer):
        def hook(module, input, kwargs):
            assert isinstance(module, WeGeFTLinear), f"Layer must be WeGeFTLinear, got {type(module)}."
            delta_weight = self._wegeft_forward(target_key, layer, module.weight.data)
            kwargs["delta_weight"] = delta_weight
            return input, kwargs
        return hook
    
    def residual_forward_hook_merged(self, target_key, target_sub_keys, enable_wegeft, layer):
        def hook(module, input, kwargs):
            assert isinstance(module, WeGeFTMergedLinear), f"Layer must be WeGeFTLinear, got {type(module)}."
            delta_weights = []
            for idx, (sub_key, enable) in enumerate(zip(target_sub_keys, enable_wegeft)):
                _weight = module.get_weight(idx)
                if enable:
                    delta_weight = self._wegeft_forward(sub_key, layer, _weight)
                else:
                    delta_weight = torch.zeros_like(_weight)
                delta_weights.append(delta_weight)
            kwargs["delta_weight"] = torch.cat(delta_weights, dim=0)
            return input, kwargs
        return hook
    
    def state_dict(self, destination=None, prefix='', keep_vars=False):
        # Only need to save the hypernets and classifier
        state_dict = OrderedDict()
        for key, value in super().state_dict(destination=destination, prefix=prefix, keep_vars=keep_vars).items():
            if "backbone" in key:
                continue
            state_dict[key] = value
        return state_dict

    def load_state_dict(self, state_dict: Mapping[str, Any], strict: bool = False, assign: bool = False):
        return super().load_state_dict(state_dict, strict=strict, assign=assign)

    def forward(self, *args, **kwargs):
        x = self.backbone(*args, **kwargs)
        return x

    def wegeft_parameters(self):
        return [p for n, p in self.named_parameters() if "backbone" not in n]
    
    def wegeft_named_parameters(self):
        return [(n, p) for n, p in self.named_parameters() if "backbone" not in n]

    def num_trainable_parameters(self):
        num_trainable_parameters = sum(p.numel() for n, p in self.named_parameters() if p.requires_grad)
        total_parameters = sum(p.numel() for n, p in self.named_parameters() if ("backbone" in n and not p.requires_grad))
        percent_trainable = num_trainable_parameters / total_parameters * 100
        return num_trainable_parameters, percent_trainable
    

class WeGeFTWrapperForImageClassification(WeGeFTWrapper):

    def __init__(
            self, 
            config: WeGeFTConfig,
            backbone: Union[PreTrainedModel, nn.Module],
        ) -> None:
        super().__init__(config, backbone)
        # Enable the classifier grad
        self.backbone.head.requires_grad_(True)

    def classifier_parameters(self):
        return [p for n, p in self.named_parameters() if "backbone" in n and "head" in n]
    
    def classifier_named_parameters(self):
        return [(n, p) for n, p in self.named_parameters() if "backbone" in n and "head" in n]
    
    def state_dict(self, destination=None, prefix='', keep_vars=False):
        # Only need to save the hypernets and classifier
        state_dict = OrderedDict()
        for key, value in nn.Module.state_dict(self, destination=destination, prefix=prefix, keep_vars=keep_vars).items():
            if "backbone" in key and "head" not in key:
                continue
            state_dict[key] = value
        return state_dict


class WeGeFTWrapperForSeqClassification(WeGeFTWrapper):

    def __init__(
            self, 
            config: WeGeFTConfig,
            backbone: Union[PreTrainedModel, nn.Module],
        ) -> None:
        super().__init__(config, backbone)
        # Enable the classifier grad
        self.backbone.classifier.requires_grad_(True)

    def classifier_parameters(self):
        return [p for n, p in self.named_parameters() if "classifier" in n]
    
    def classifier_named_parameters(self):
        return [(n, p) for n, p in self.named_parameters() if "classifier" in n]
    
    def state_dict(self, destination=None, prefix='', keep_vars=False):
        # Only need to save the hypernets and classifier
        state_dict = OrderedDict()
        for key, value in nn.Module.state_dict(self, destination=destination, prefix=prefix, keep_vars=keep_vars).items():
            if "backbone" in key and "classifier" not in key:
                continue
            state_dict[key] = value
        return state_dict
    
    def num_trainable_parameters(self):
        num_trainable_parameters = sum(p.numel() for n, p in self.named_parameters() if (p.requires_grad and "classifier" not in n))
        total_parameters = sum(p.numel() for n, p in self.named_parameters() if ("backbone" in n and not p.requires_grad))
        percent_trainable = num_trainable_parameters / total_parameters * 100
        return num_trainable_parameters, percent_trainable
    

class WeGeFTWrapperForCausalLM(WeGeFTWrapper):

    def forward(
            self, 
            input_ids: torch.LongTensor = None,
            attention_mask: Optional[torch.Tensor] = None,
            position_ids: Optional[torch.LongTensor] = None,
            past_key_values: Optional[List[torch.FloatTensor]] = None,
            inputs_embeds: Optional[torch.FloatTensor] = None,
            labels: Optional[torch.LongTensor] = None,
            use_cache: Optional[bool] = None,
            output_attentions: Optional[bool] = None,
            output_hidden_states: Optional[bool] = None,
            return_dict: Optional[bool] = None,
            cache_position: Optional[torch.LongTensor] = None,
            **kwargs, # Useless arguments go here
        ):
        # Replicate the sginature of the forward method of the CausalLM
        return self.backbone(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            labels=labels,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            cache_position=cache_position,
        )
    
    def generate(
            self,
            inputs: Optional[torch.Tensor] = None,
            generation_config = None,
            logits_processor = None,
            stopping_criteria = None,
            prefix_allowed_tokens_fn = None,
            synced_gpus: Optional[bool] = None,
            assistant_model: Optional["PreTrainedModel"] = None,
            streamer = None,
            negative_prompt_ids: Optional[torch.Tensor] = None,
            negative_prompt_attention_mask: Optional[torch.Tensor] = None,
            **kwargs,
        ):

        return self.backbone.generate(
            inputs=inputs,
            generation_config=generation_config,
            logits_processor=logits_processor,
            stopping_criteria=stopping_criteria,
            prefix_allowed_tokens_fn=prefix_allowed_tokens_fn,
            synced_gpus=synced_gpus,
            assistant_model=assistant_model,
            streamer=streamer,
            negative_prompt_ids=negative_prompt_ids,
            negative_prompt_attention_mask=negative_prompt_attention_mask,
            **kwargs
        )

    def save(self, save_directory):
        
        # Save the config
        config_path = Path(save_directory, "config")
        self.config.save_pretrained(config_path)

        checkpoint_path = Path(save_directory, "checkpoint.pth")
        torch.save(self.state_dict(), checkpoint_path)

    def gradient_checkpointing_enable(self, gradient_checkpointing_kwargs=None):
        self.backbone.gradient_checkpointing_enable(gradient_checkpointing_kwargs=gradient_checkpointing_kwargs)
