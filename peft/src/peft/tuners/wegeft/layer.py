# Copyright 2023-present the HuggingFace Inc. team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
from __future__ import annotations

import math
import warnings
from typing import Any, Optional, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers.pytorch_utils import Conv1D

from peft.tuners.tuners_utils import BaseTunerLayer, check_adapters_to_merge

from .config import WeGeFTConfig


class WeGeFTLayer(BaseTunerLayer):
    # All names of layers that may contain (trainable) adapter weights
    adapter_layer_names = ("wegeft_phi", "wegeft_psi")
    # All names of other parameters that may contain adapter-related parameters
    other_param_names = ("r", "scaling")

    def __init__(self, base_layer: nn.Module, transform_dim: Optional[str]="input", **kwargs) -> None:
        self.base_layer = base_layer
        self.transform_dim = transform_dim
        self.r = {}
        self.scaling = {}
        self.wegeft_phi = nn.ModuleDict({})
        self.wegeft_psi = nn.ModuleDict({})
        self.wegeft_dropout = nn.ModuleDict({})
        # Mark the weight as unmerged
        self._disable_adapters = False
        self.merged_adapters = []
        self._caches: dict[str, Any] = {}
        self.kwargs = kwargs

        base_layer = self.get_base_layer()
        if isinstance(base_layer, nn.Linear):
            in_features, out_features = base_layer.in_features, base_layer.out_features
        elif isinstance(base_layer, nn.Conv2d):
            # in_features, out_features = base_layer.in_channels, base_layer.out_channels
            raise ValueError("Conv2D is not supported in WeGeFT yet")
        elif isinstance(base_layer, nn.Embedding):
            # in_features, out_features = base_layer.num_embeddings, base_layer.embedding_dim
            raise ValueError("Embedding is not supported in WeGeFT yet")
        elif isinstance(base_layer, Conv1D):
            # in_features, out_features = (
            #     base_layer.weight.ds_shape if hasattr(base_layer.weight, "ds_shape") else base_layer.weight.shape
            # )
            raise ValueError("Conv1D is not supported in WeGeFT yet")
        else:
            raise ValueError(f"Unsupported layer type {type(base_layer)}")

        self.in_features = in_features
        self.out_features = out_features

    def update_layer(
        self, adapter_name, r, wegeft_alpha, init_wegeft_weights, wegeft_dropout=None,
        own_parameters=False,
    ):
        # This code works for linear layers, override for other layer types
        if r <= 0:
            raise ValueError(f"`r` should be a positive integer value but the value passed is {r}")

        self.r[adapter_name] = r
        self.scaling[adapter_name] = wegeft_alpha / r

        dim = self.in_features if self.transform_dim == "input" else self.out_features
        # Only create the weights if they are not shared, i.e. are unique to this layer
        # If the layer is shared, the weights will be provided by the hook
        if own_parameters:
            # Actual trainable parameters
            self.wegeft_phi[adapter_name] = nn.Linear(dim, r, bias=False)
            self.wegeft_psi[adapter_name] = nn.Linear(dim, r, bias=False)

            if init_wegeft_weights:
                self.reset_wegeft_parameters(adapter_name, init_wegeft_weights)

        if wegeft_dropout is not None and wegeft_dropout > 0.:
            self.wegeft_dropout[adapter_name] = nn.Dropout(wegeft_dropout)
        else:
            # self.phi_mask[adapter_name] = nn.Identity()
            self.wegeft_dropout[adapter_name] = nn.Identity()

        # check weight and qweight (for GPTQ)
        for weight_name in ("weight", "qweight"):
            weight = getattr(self.get_base_layer(), weight_name, None)
            if weight is not None:
                # the layer is already completely initialized, this is an update
                if weight.dtype.is_floating_point or weight.dtype.is_complex:
                    self.to(weight.device, dtype=weight.dtype)
                else:
                    self.to(weight.device)
                break

        self.set_adapter(self.active_adapters)

    def reset_wegeft_parameters(self, adapter_name, init_wegeft_weights):
        if init_wegeft_weights is False:
            return

        if adapter_name in self.wegeft_phi.keys():
            if init_wegeft_weights is True:
                # initialize A the same way as the default for nn.Linear and psi to zero
                # https://github.com/microsoft/LoRA/blob/a0a92e0f26c067cf94747bdbf1ce73793fa44d19/loralib/layers.py#L124
                nn.init.kaiming_uniform_(self.wegeft_psi[adapter_name].weight, a=math.sqrt(5))
                nn.init.zeros_(self.wegeft_phi[adapter_name].weight)
            else:
                raise ValueError(f"Unknown initialization {init_wegeft_weights=}")

    def _cache_store(self, key: str, value: Any) -> None:
        self._caches[key] = value

    def _cache_pop(self, key: str) -> Any:
        value = self._caches.pop(key)
        return value

    def copy_base_weight(self):
        self.base_weight_copy = self.get_base_layer().weight.data.clone()

    def _check_forward_args(self, x, *args, **kwargs):
        """Check if the arguments are compatible with the configs and state of the model"""
        adapter_names = kwargs.get("adapter_names", None)
        if adapter_names is None:
            return

        if len(x) != len(adapter_names):
            msg = (
                "Length of `adapter_names` should be the same as the number of inputs, but got "
                f"{len(adapter_names)} and {len(x)} respectively."
            )
            raise ValueError(msg)

        if self.merged:
            # It is unclear what would be the right thing to do if users pass adapter_names and there are merged
            # adapters. Therefore, it is better to raise an error in this case.
            msg = "Cannot pass `adapter_names` when there are merged adapters, please call `unmerge_adapter` first."
            raise ValueError(msg)

    def _mixed_batch_forward(
        self, x: torch.Tensor, *args: Any, adapter_names: list[str], **kwargs: Any
    ) -> torch.Tensor:
        # This is a special method that handles the case when users pass the argument `adapter_names`. This is an
        # extra argument that allows mixing different adapters in the same batch at inference time.
        raise NotImplementedError("Mixed batch forward is not implemented for WeGeFT yet")


class Linear(nn.Module, WeGeFTLayer):
    # WeGeFT implemented in a dense layer
    def __init__(
        self,
        base_layer,
        adapter_name: str,
        r: int = 0,
        wegeft_alpha: float = 1.0,
        wegeft_dropout: float = 0.0,
        fan_in_fan_out: bool = False,  # Set this to True if the layer to replace stores weight like (fan_in, fan_out)
        is_target_conv_1d_layer: bool = False,
        init_wegeft_weights: Union[bool, str] = True,
        own_parameters: bool = False,
        transform_dim: Optional[int] = "input",
        **kwargs,
    ) -> None:
        super().__init__()
        WeGeFTLayer.__init__(self, base_layer, transform_dim=transform_dim, **kwargs)
        self.fan_in_fan_out = fan_in_fan_out

        self._active_adapter = adapter_name
        self.update_layer(
            adapter_name,
            r,
            wegeft_alpha,
            init_wegeft_weights=init_wegeft_weights,
            wegeft_dropout=wegeft_dropout,
            own_parameters=own_parameters,
        )
        self.is_target_conv_1d_layer = is_target_conv_1d_layer

    def merge(
            self, 
            safe_merge: bool = False, 
            adapter_names: Optional[list[str]] = None, 
            wegeft_psi: Optional[list[str]] = None, 
            wegeft_phi: Optional[list[str]] = None) -> None:
        """
        Merge the active adapter weights into the base weights

        Args:
            safe_merge (`bool`, *optional*):
                If True, the merge operation will be performed in a copy of the original weights and check for NaNs
                before merging the weights. This is useful if you want to check if the merge operation will produce
                NaNs. Defaults to `False`.
            adapter_names (`list[str]`, *optional*):
                The list of adapter names that should be merged. If None, all active adapters will be merged. Defaults
                to `None`.
        """
        adapter_names = check_adapters_to_merge(self, adapter_names)
        if not adapter_names:
            # no adapter to merge
            return

        residual_weights = 0.
        for active_adapter in adapter_names:
            
            try:
                psi = wegeft_psi.get(active_adapter, None)
                if psi is None:
                    psi = self.wegeft_psi[active_adapter]
                phi = wegeft_phi.get(active_adapter, None)
                if phi is None:
                    phi = self.wegeft_phi[active_adapter]
            except KeyError as e:
                continue
    
            delta_weight = self.get_delta_weight(active_adapter, psi, phi)
            residual_weights = residual_weights + delta_weight

            self.merged_adapters.append(active_adapter)

        base_layer = self.get_base_layer()

        orig_weights = base_layer.weight.data.clone()
        orig_weights += residual_weights
        
        if safe_merge and not torch.isfinite(orig_weights).all():
            raise ValueError(
                f"NaNs detected in the merged weights."
            )

        base_layer.weight.data = orig_weights

    def unmerge(self) -> None:
        """
        This method unmerges all merged adapter layers from the base weights.
        """
        """
        This method unmerges all merged adapter layers from the base weights.
        """
        if not self.merged:
            warnings.warn("Already unmerged. Nothing to do.")
            return
        
        if self.base_weight_copy is None:
            raise ValueError("No base weight copy found. Cannot unmerge.")
        self.get_base_layer().weight.data = self.base_weight_copy
        self.base_weight_copy = None
        while len(self.merged_adapters) > 0:
            active_adapter = self.merged_adapters.pop()

    def get_delta_weight(self, adapter, wegeft_psi, wegeft_phi) -> torch.Tensor:
        """
        Compute the delta weight for the given adapter.

        Args:
            adapter (str):
                The name of the adapter for which the delta weight should be computed.
        """

        psi, phi = wegeft_psi.weight, wegeft_phi.weight

        device = psi.device
        dtype = psi.dtype
        weight = self.get_base_layer().weight.to(dtype)

        # In case users wants to merge the adapter weights that are in
        # float16 while being on CPU, we need to cast the weights to float32, perform the merge and then cast back to
        # float16 because the `@` and matmul operation in general is not supported in torch + cpu + fp16.
        cast_to_fp32 = device.type == "cpu" and dtype == torch.float16
        if cast_to_fp32:
            psi = psi.float()
            phi = phi.float()
            weight = weight.float()

        weight = self.wegeft_dropout[adapter](weight)

        if self.transform_dim == "input":
            delta_weight = (weight @ phi.t()) @ psi
        elif self.transform_dim == "output":
            delta_weight = (phi @ psi) @ weight

        if cast_to_fp32:
            delta_weight = delta_weight.to(dtype=dtype)

        scaling = self.scaling[adapter]

        return delta_weight * scaling

    def _forward_input_perspective(self, x: torch.Tensor, wegeft_phi, wegeft_psi, *args, **kwargs):

        result = self.base_layer(x, *args, **kwargs)
        torch_result_dtype = result.dtype

        for active_adapter in self.active_adapters:
            phi = wegeft_phi[active_adapter]
            psi = wegeft_psi[active_adapter]

            weights = self.get_base_layer().weight.to(phi.weight.dtype)
            weights = self.wegeft_dropout[active_adapter](weights)
            w_phi = phi(weights)
            
            x = x.to(psi.weight.dtype)
            scaling = self.scaling[active_adapter]
            result = result + F.linear(psi(x), w_phi) * scaling
        
        result = result.to(torch_result_dtype)

        return result
    
    def _forward_output_perspective(self, x: torch.Tensor, wegeft_phi, wegeft_psi, *args, **kwargs):

        result = self.base_layer(x, *args, **kwargs)
        torch_result_dtype = result.dtype

        residual_result = 0.
        for active_adapter in self.active_adapters:
            phi = wegeft_phi[active_adapter]
            psi = wegeft_psi[active_adapter]

            _result = result.to(psi.weight.dtype)
            residual_result = residual_result + phi(psi(_result)) * self.scaling[active_adapter]
        
        # Add in higher precision to avoid numerical instability
        result = result + residual_result
        # Cast back to the original dtype
        result = result.to(torch_result_dtype) # TODO: check if this is necessary especially if using bitsandbytes
        
        return result

    def forward(self, x: torch.Tensor, *args: Any, **kwargs: Any) -> torch.Tensor:
        self._check_forward_args(x, *args, **kwargs)
        adapter_names = kwargs.pop("adapter_names", None)

        # Check if the weights are provided in the kwargs (when sharing weights across layers)
        # wegeft_phi, wegeft_psi will be empty if the weights are not shared
        wegeft_phi = kwargs.pop("wegeft_phi", self.wegeft_phi)
        wegeft_psi = kwargs.pop("wegeft_psi", self.wegeft_psi)

        # If no weights are provided, use the current weights of the layer
        if len(wegeft_phi) == 0:
            wegeft_phi = self.wegeft_phi
        if len(wegeft_psi) == 0:
            wegeft_psi = self.wegeft_psi

        if self.disable_adapters:
            if self.merged:
                self.unmerge()
            result = self.base_layer(x, *args, **kwargs)
        elif adapter_names is not None:
            result = self._mixed_batch_forward(x, *args, adapter_names=adapter_names, **kwargs)
        elif self.merged:
            result = self.base_layer(x, *args, **kwargs)
        else:
            forward_func = {
                "input": self._forward_input_perspective,
                "output": self._forward_output_perspective,
            }[self.transform_dim]
            result = forward_func(x, wegeft_phi, wegeft_psi, *args, **kwargs)

        return result

    def __repr__(self) -> str:
        rep = super().__repr__()
        return "wegeft." + rep


def dispatch_default(
    target: torch.nn.Module,
    adapter_name: str,
    wegeft_config: WeGeFTConfig,
    **kwargs,
) -> Optional[torch.nn.Module]:
    new_module = None

    if isinstance(target, BaseTunerLayer):
        target_base_layer = target.get_base_layer()
    else:
        target_base_layer = target

    if isinstance(target_base_layer, torch.nn.Embedding):
        raise ValueError("Embedding is not supported in WeGeFT yet")
    elif isinstance(target_base_layer, torch.nn.Conv2d):
        raise ValueError("Conv2D is not supported in WeGeFT yet")
        # kwargs.update(lora_config.loftq_config)
        # new_module = Conv2d(target, adapter_name, **kwargs)
    elif isinstance(target_base_layer, torch.nn.Linear):
        if kwargs["fan_in_fan_out"]:
            warnings.warn(
                "fan_in_fan_out is set to True but the target module is `torch.nn.Linear`. "
                "Setting fan_in_fan_out to False."
            )
            kwargs["fan_in_fan_out"] = wegeft_config.fan_in_fan_out = False

        new_module = Linear(target, adapter_name, **kwargs)
    elif isinstance(target_base_layer, Conv1D):
        raise ValueError("Conv1D is not supported in WeGeFT yet")

    return new_module
