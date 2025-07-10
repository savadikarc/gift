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
import operator
import re
import warnings
from contextlib import contextmanager
from dataclasses import asdict, replace
from enum import Enum
from functools import partial, reduce
from itertools import chain
from typing import Literal, Optional

import torch
from torch import nn
from tqdm import tqdm

from peft.import_utils import is_bnb_4bit_available, is_bnb_available
from peft.tuners.tuners_utils import (
    BaseTuner,
    BaseTunerLayer,
    check_target_module_exists,
    onload_layer,
    replicate_layers,
)
from peft.utils import (
    TRANSFORMERS_MODELS_TO_LORA_TARGET_MODULES_MAPPING,
    ModulesToSaveWrapper,
    _freeze_adapter,
    _get_submodules,
    get_peft_model_state_dict,
    get_quantization_config,
)
from peft.utils.merge_utils import dare_linear, dare_ties, magnitude_prune, task_arithmetic, ties

from .config import WeGeFTConfig
from .layer import WeGeFTLayer, dispatch_default

def _adapter_names_pre_forward_hook(target, args, kwargs, adapter_names):
    # pre-forward hook to inject the adapter_names argument when using mixed adapter batches inference
    kwargs["adapter_names"] = adapter_names
    return args, kwargs


class WeGeFTModel(BaseTuner):
    """
    Creates Low Rank Adapter (LoRA) model from a pretrained transformers model.

    The method is described in detail in https://arxiv.org/abs/2106.09685.

    Args:
        model ([`torch.nn.Module`]): The model to be adapted.
        config ([`LoraConfig`]): The configuration of the Lora model.
        adapter_name (`str`): The name of the adapter, defaults to `"default"`.

    Returns:
        `torch.nn.Module`: The Lora model.

    Example:

        ```py
        >>> from transformers import AutoModelForSeq2SeqLM
        >>> from peft import LoraModel, LoraConfig

        >>> config = LoraConfig(
        ...     task_type="SEQ_2_SEQ_LM",
        ...     r=8,
        ...     lora_alpha=32,
        ...     target_modules=["q", "v"],
        ...     lora_dropout=0.01,
        ... )

        >>> model = AutoModelForSeq2SeqLM.from_pretrained("t5-base")
        >>> lora_model = LoraModel(model, config, "default")
        ```

        ```py
        >>> import torch
        >>> import transformers
        >>> from peft import LoraConfig, PeftModel, get_peft_model, prepare_model_for_kbit_training

        >>> rank = ...
        >>> target_modules = ["q_proj", "k_proj", "v_proj", "out_proj", "fc_in", "fc_out", "wte"]
        >>> config = LoraConfig(
        ...     r=4, lora_alpha=16, target_modules=target_modules, lora_dropout=0.1, bias="none", task_type="CAUSAL_LM"
        ... )
        >>> quantization_config = transformers.BitsAndBytesConfig(load_in_8bit=True)

        >>> tokenizer = transformers.AutoTokenizer.from_pretrained(
        ...     "kakaobrain/kogpt",
        ...     revision="KoGPT6B-ryan1.5b-float16",  # or float32 version: revision=KoGPT6B-ryan1.5b
        ...     bos_token="[BOS]",
        ...     eos_token="[EOS]",
        ...     unk_token="[UNK]",
        ...     pad_token="[PAD]",
        ...     mask_token="[MASK]",
        ... )
        >>> model = transformers.GPTJForCausalLM.from_pretrained(
        ...     "kakaobrain/kogpt",
        ...     revision="KoGPT6B-ryan1.5b-float16",  # or float32 version: revision=KoGPT6B-ryan1.5b
        ...     pad_token_id=tokenizer.eos_token_id,
        ...     use_cache=False,
        ...     device_map={"": rank},
        ...     torch_dtype=torch.float16,
        ...     quantization_config=quantization_config,
        ... )
        >>> model = prepare_model_for_kbit_training(model)
        >>> lora_model = get_peft_model(model, config)
        ```

    **Attributes**:
        - **model** ([`~transformers.PreTrainedModel`]) -- The model to be adapted.
        - **peft_config** ([`LoraConfig`]): The configuration of the Lora model.
    """

    prefix: str = "wegeft_"

    def __init__(self, model, config, adapter_name) -> None:
        self.parameter_origins = dict()
        self.parameter_sharing_hook_handles = dict()
        super().__init__(model, config, adapter_name)

        # print(self.parameter_origins)

    def _check_new_adapter_config(self, config: WeGeFTConfig) -> None:
        """
        A helper method to check the config when a new adapter is being added.

        Raise a ValueError if there is something wrong with the config or if it conflicts with existing adapters.

        """
        # TODO: there should be a check if any of the existing adapters actually has bias != "none", or else the check
        # does not fully correspond to the error message.
        if (len(self.peft_config) > 1) and (config.bias != "none"):
            raise ValueError(
                f"{self.__class__.__name__} supports only 1 adapter with bias. When using multiple adapters, "
                "set bias to 'none' for all adapters."
            )
        
    def inject_adapter(self, model: nn.Module, adapter_name: str):
        r"""
        Creates adapter layers and replaces the target modules with the adapter layers. This method is called under the
        hood by `peft.mapping.get_peft_model` if a non-prompt tuning adapter class is passed.

        The corresponding PEFT config is directly retrieved from the `peft_config` attribute of the BaseTuner class.

        Args:
            model (`nn.Module`):
                The model to be tuned.
            adapter_name (`str`):
                The adapter name.
        """
        # Prepare for the tied_modules and tied_layers
        peft_config = self.peft_config[adapter_name]

        peft_config = self._prepare_adapter_config_for_shared_modules(peft_config)

        return super().inject_adapter(model, adapter_name)

    @staticmethod
    def _check_target_module_exists(wegeft_config, key):
        return check_target_module_exists(wegeft_config, key)

    def _prepare_model(self, peft_config: WeGeFTConfig, model: nn.Module):
        r"""
        A private method to modify the model structure before adapter is applied.

        Args:
            peft_config (`PeftConfig`):
                The prepared adapter config.
            model (`nn.Module`):
                The model that is going to be adapted.
        """
        pass # keep for later, if needed

    def _parameter_origin_key(self, wegeft_config, current_key):
        
        composite_target_key = current_key
        for tying_patterns in wegeft_config.tied_modules:
            if any(re.match(tying_pattern, current_key) for tying_pattern in tying_patterns):
                composite_target_key = "-".join(tying_patterns)
                break

        return composite_target_key

    def _update_sharing_map(self, wegeft_config, adapter_name, target_name, current_key):
        # Only own the parameters if this layer is the first layer
        # among the shared layers

        parameter_origin_key = self._parameter_origin_key(
            wegeft_config, current_key
        )

        adapter_origins = self.parameter_origins.get(adapter_name, dict())
        parameter_origin = adapter_origins.get(parameter_origin_key, None)

        # If origin is same as the current key, then the parameters are not shared
        # and the adapter owns the parameters
        own_parameters = (parameter_origin_key == current_key) or (parameter_origin is None)
        
        if own_parameters:
            # Update the parameter_origins
            adapter_origins[parameter_origin_key] = current_key
            self.parameter_origins[adapter_name] = adapter_origins

        return own_parameters

    def _create_and_replace(
        self,
        wegeft_config: WeGeFTConfig,
        adapter_name,
        target,
        target_name,
        parent,
        current_key,
    ):
        if current_key is None:
            raise ValueError("Current Key shouldn't be `None`")

        # Regexp matching - Find key which matches current target_name in patterns provided
        pattern_keys = list(wegeft_config.rank_pattern.keys())
        target_name_key = next(filter(lambda key: re.match(rf".*\.{key}$", current_key), pattern_keys), current_key)
        r = wegeft_config.rank_pattern.get(target_name_key, wegeft_config.r)
        alpha = wegeft_config.alpha_pattern.get(target_name_key, wegeft_config.wegeft_alpha)
        transform_dim = wegeft_config.transform_dim.get(target_name, None)

        # Only own the parameters if this layer is the first layer
        # among the shared layers
        own_parameters = self._update_sharing_map(wegeft_config, adapter_name, target_name, current_key)

        kwargs = {
            "r": r,
            "wegeft_alpha": alpha,
            "fan_in_fan_out": wegeft_config.fan_in_fan_out,
            "init_wegeft_weights": wegeft_config.init_wegeft_weights,
            "own_parameters": own_parameters,
            "transform_dim": transform_dim,
            "wegeft_dropout": wegeft_config.wegeft_dropout,
        }

        if isinstance(target, WeGeFTLayer):
            target.update_layer(
                adapter_name,
                r,
                wegeft_alpha=wegeft_config.wegeft_alpha,
                init_wegeft_weights=wegeft_config.init_wegeft_weights,
                own_parameters=own_parameters,
                wegeft_dropout=wegeft_config.wegeft_dropout,
            )
        else:
            new_module = self._create_new_module(wegeft_config, adapter_name, target, **kwargs)
            if adapter_name not in self.active_adapters:
                # adding an additional adapter: it is not automatically trainable
                new_module.requires_grad_(False)
            self._replace_module(parent, target_name, new_module, target)
    
    def _create_sharing_hook(self, wegeft_config, adapter_names, target_name, current_key):
        
        adapter_names = adapter_names if isinstance(adapter_names, list) else [adapter_names]
        
        wegeft_phi = dict()
        wegeft_psi = dict()
        
        for adapter_name in adapter_names:
            # print(f"Creating sharing hook for {adapter_name} and {target_name}, key: {current_key}")
            parameter_origin_key = self._parameter_origin_key(
                wegeft_config, current_key
            )
            parameter_origin = self.parameter_origins[adapter_name][parameter_origin_key]
            if parameter_origin == current_key: # TODO: This will change if we the design is made such that the parameters are owned by this module
                continue

            _, origin_module, origin_module_name = _get_submodules(self.model, parameter_origin)

            assert isinstance(origin_module, WeGeFTLayer), f"Origin module should be a WeGeFTLayer, got {type(origin_module)}"
            wegeft_phi[adapter_name] = origin_module.wegeft_phi[adapter_name]
            wegeft_psi[adapter_name] = origin_module.wegeft_psi[adapter_name]

        def hook(module, args, kwargs):
            # Get the phi and psi from the parameter origin
            kwargs["wegeft_phi"] = wegeft_phi
            kwargs["wegeft_psi"] = wegeft_psi
            return args, kwargs
        
        return hook

    def _enable_sharing_hooks(self, adapter_name: str=None):
        
        adapter_name = self.active_adapters if adapter_name is None else adapter_name
        adapter_name = adapter_name if isinstance(adapter_name, list) else [adapter_name]
        key_list = [key for key, _ in self.model.named_modules()]

        for key in key_list:
            adapters_for_key = []
            for adapter in adapter_name:
                peft_config = self.peft_config[adapter]

                _check_for_modules_to_save = getattr(peft_config, "modules_to_save", None) is not None
                # Check for modules_to_save in case
                if _check_for_modules_to_save and any(
                    key.endswith(f"{module_to_save}") for module_to_save in peft_config.modules_to_save
                ):
                    continue

                if not self._check_target_module_exists(peft_config, key):
                    continue

                adapters_for_key.append(adapter)

            parent, target, target_name = _get_submodules(self.model, key)
            if isinstance(target, WeGeFTLayer):
                hook = self._create_sharing_hook(peft_config, adapter, target_name, key)
                handle = target.register_forward_pre_hook(hook, with_kwargs=True)
                self.parameter_sharing_hook_handles[key] = handle
    
    def _disable_sharing_hooks(self):
        # Remove self.parameter_sharing_hook_handles and empty the dict
        for key, handle in self.parameter_sharing_hook_handles.items():
            handle.remove()
        self.parameter_sharing_hook_handles = dict()

    def _replace_module(self, parent, child_name, new_module, child):
        setattr(parent, child_name, new_module)
        # It's not necessary to set requires_grad here, as that is handled by
        # _mark_only_adapters_as_trainable

        # child layer wraps the original module, unpack it
        if hasattr(child, "base_layer"):
            child = child.base_layer

        if not hasattr(new_module, "base_layer"):
            new_module.weight = child.weight
            if hasattr(child, "bias"):
                new_module.bias = child.bias

        if getattr(child, "state", None) is not None:
            if hasattr(new_module, "base_layer"):
                new_module.base_layer.state = child.state
            else:
                new_module.state = child.state
            new_module.to(child.weight.device)

        # dispatch to correct device
        for name, module in new_module.named_modules():
            if (self.prefix in name) or ("ranknum" in name):
                weight = (
                    child.qweight
                    if hasattr(child, "qweight")
                    else child.W_q
                    if hasattr(child, "W_q")
                    else child.weight
                )
                module.to(weight.device)

    def _mark_only_adapters_as_trainable(self, model: nn.Module) -> None:
        for n, p in model.named_parameters():
            if self.prefix not in n:
                p.requires_grad = False

        for active_adapter in self.active_adapters:
            bias = self.peft_config[active_adapter].bias
            if bias == "none":
                continue

            if bias == "all":
                for n, p in model.named_parameters():
                    if "bias" in n:
                        p.requires_grad = True
            elif bias == "wegeft_only":
                for m in model.modules():
                    if isinstance(m, WeGeFTLayer) and hasattr(m, "bias") and m.bias is not None:
                        m.bias.requires_grad = True
            else:
                raise NotImplementedError(f"Requested bias: {bias}, is not implemented.")

    @staticmethod
    def _create_new_module(wegeft_config, adapter_name, target, **kwargs):
        # Use the default dispatcher for now, but we can add more dispatchers in the future

        new_module = dispatch_default(target, adapter_name, wegeft_config=wegeft_config, **kwargs)

        if new_module is None:
            # no module could be matched
            raise ValueError(
                f"Target module {target} is not supported. Currently, only the following modules are supported: "
                "`torch.nn.Linear`."
            )

        return new_module

    def __getattr__(self, name: str):
        """Forward missing attributes to the wrapped module."""
        try:
            return super().__getattr__(name)  # defer to nn.Module's logic
        except AttributeError:
            return getattr(self.model, name)

    def get_peft_config_as_dict(self, inference: bool = False):
        config_dict = {}
        for key, value in self.peft_config.items():
            config = {k: v.value if isinstance(v, Enum) else v for k, v in asdict(value).items()}
            if inference:
                config["inference_mode"] = True
        config_dict[key] = config
        return config

    def _set_adapter_layers(self, enabled: bool = True) -> None:
        for module in self.model.modules():
            if isinstance(module, (BaseTunerLayer, ModulesToSaveWrapper)):
                module.enable_adapters(enabled)

    def enable_adapter_layers(self) -> None:
        """Enable all adapters.

        Call this if you have previously disabled all adapters and want to re-enable them.
        """
        self._set_adapter_layers(enabled=True)

    def disable_adapter_layers(self) -> None:
        """Disable all adapters.

        When disabling all adapters, the model output corresponds to the output of the base model.
        """
        for active_adapter in self.active_adapters:
            val = self.peft_config[active_adapter].bias
            if val != "none":
                msg = (
                    f"Careful, disabling adapter layers with bias configured to be '{val}' does not produce the same "
                    "output as the the base model would without adaption."
                )
                warnings.warn(msg)
        self._set_adapter_layers(enabled=False)

    def set_adapter(self, adapter_name: str | list[str]) -> None:
        """Set the active adapter(s).

        Additionally, this function will set the specified adapters to trainable (i.e., requires_grad=True). If this is
        not desired, use the following code.

        ```py
        >>> for name, param in model_peft.named_parameters():
        ...     if ...:  # some check on name (ex. if 'wegeft' in name)
        ...         param.requires_grad = False
        ```

        Args:
            adapter_name (`str` or `list[str]`): Name of the adapter(s) to be activated.
        """
        for module in self.model.modules():
            if isinstance(module, WeGeFTLayer):
                if module.merged:
                    warnings.warn("Adapter cannot be set when the model is merged. Unmerging the model first.")
                    module.unmerge()
                module.set_adapter(adapter_name)
        self.active_adapter = adapter_name

        # First, remove the existing hooks to avoid multiple
        # hooks on the same module
        self._disable_sharing_hooks()
        # Then, enable the sharing hooks for the active adapters
        self._enable_sharing_hooks(adapter_name)

    @contextmanager
    def _enable_peft_forward_hooks(self, *args, **kwargs):
        # If adapter_names is passed as an argument, we inject it into the forward arguments.
        adapter_names = kwargs.pop("adapter_names", None)
        if adapter_names is None:
            # nothing to do
            yield
            return

        if self.training:
            raise ValueError("Cannot pass `adapter_names` when the model is in training mode.")
        
        # TODO: handle the sharing hooks

        hook_handles = []
        for module in self.modules():
            if isinstance(module, WeGeFTLayer):
                pre_forward = partial(_adapter_names_pre_forward_hook, adapter_names=adapter_names)
                handle = module.register_forward_pre_hook(pre_forward, with_kwargs=True)
                hook_handles.append(handle)

        yield

        for handle in hook_handles:
            handle.remove()

    def _check_merge_allowed(self):
        """Verify that the configuration supports merging.

        Currently gptq quantization and replicated layers do not support merging.
        """
        if getattr(self.model, "quantization_method", None) == "gptq":
            raise ValueError("Cannot merge LORA layers when the model is gptq quantized")
        if self.peft_config.get("layer_replication"):
            raise ValueError("Cannot merge LORA layers when base model layers are replicated")

    @staticmethod
    def _prepare_adapter_config(peft_config, model_config):
        if peft_config.target_modules is None:
            if model_config["model_type"] not in TRANSFORMERS_MODELS_TO_LORA_TARGET_MODULES_MAPPING:
                raise ValueError("Please specify `target_modules` in `peft_config`")
            peft_config.target_modules = set(
                TRANSFORMERS_MODELS_TO_LORA_TARGET_MODULES_MAPPING[model_config["model_type"]]
            )

        if isinstance(peft_config.transform_dim, str):
            # split here coz hf does this. wtf.
            peft_config.transform_dim = {target_module.split(".")[-1]: peft_config.transform_dim for target_module in peft_config.target_modules}
        else:
            assert len(peft_config.transform_dim) == len(peft_config.target_modules)

        return peft_config
    
    @staticmethod
    def _prepare_adapter_config_for_shared_modules(peft_config):
        # Check tied modules
        if peft_config.tied_modules is None:
            # Bottleneck for now, since we need _prepare_adapter_config to be called first. This
            # cannot be done unless we hack into the _inject_adapter method of BaseTuner instead of overriding it.
            assert peft_config.target_modules is not None, "Please specify `target_modules` in `peft_config`"
            peft_config.tied_modules = [[module] for module in peft_config.target_modules]

        return peft_config

    def _unload_and_optionally_merge(
        self,
        merge=True,
        progressbar: bool = False,
        safe_merge: bool = False,
        adapter_names: Optional[list[str]] = None,
    ):
        if merge:
            self._check_merge_allowed()
        
        adapter_names = adapter_names or self.active_adapters

        key_list = [key for key, _ in self.model.named_modules() if self.prefix not in key]
        desc = "Unloading " + ("and merging " if merge else "") + "model"

        for key in tqdm(key_list, disable=not progressbar, desc=desc):
            try:
                parent, target, target_name = _get_submodules(self.model, key)
            except AttributeError:
                continue
            with onload_layer(target):
                if hasattr(target, "base_layer"):
                    if merge:
                        wegeft_phi, wegeft_psi = dict(), dict()
                        for adapter_name in adapter_names:
                            parameter_origin_key = self._parameter_origin_key(
                                self.peft_config[adapter_name], key
                            )
                            parameter_origin = self.parameter_origins[adapter_name][parameter_origin_key]
                            if parameter_origin == key:
                                continue

                            _, origin_module, origin_module_name = _get_submodules(self.model, parameter_origin)
                            assert isinstance(origin_module, WeGeFTLayer), f"Origin module should be a WeGeFTLayer, got {type(origin_module)}"
                            wegeft_phi[adapter_name] = origin_module.wegeft_phi[adapter_name]
                            wegeft_psi[adapter_name] = origin_module.wegeft_psi[adapter_name]

                        target.merge(safe_merge=safe_merge, adapter_names=adapter_names, wegeft_psi=wegeft_psi, wegeft_phi=wegeft_phi)

        # Need to separate out the actual merging because of parameter sharing
        for key in tqdm(key_list, disable=not progressbar, desc=desc):
            try:
                parent, target, target_name = _get_submodules(self.model, key)
            except AttributeError:
                continue
            with onload_layer(target):
                if hasattr(target, "base_layer"):
                    self._replace_module(parent, target_name, target.get_base_layer(), target)
                elif isinstance(target, ModulesToSaveWrapper):
                    # save any additional trainable modules part of `modules_to_save`
                    new_module = target.modules_to_save[target.active_adapter]
                    if hasattr(new_module, "base_layer"):
                        # check if the module is itself a tuner layer
                        if merge:
                            new_module.merge(safe_merge=safe_merge, adapter_names=adapter_names)
                        new_module = new_module.get_base_layer()
                    setattr(parent, target_name, new_module)

        return self.model

    def delete_adapter(self, adapter_name: str) -> None:
        """
        Deletes an existing adapter.

        Args:
            adapter_name (str): Name of the adapter to be deleted.
        """
        if adapter_name not in list(self.peft_config.keys()):
            raise ValueError(f"Adapter {adapter_name} does not exist")
        del self.peft_config[adapter_name]

        key_list = [key for key, _ in self.model.named_modules() if self.prefix not in key]
        new_adapter = None
        for key in key_list:
            _, target, _ = _get_submodules(self.model, key)
            if isinstance(target, WeGeFTLayer):
                target.delete_adapter(adapter_name)
                if new_adapter is None:
                    new_adapter = target.active_adapters[:]

        self.active_adapter = new_adapter or []

    def merge_and_unload(
        self, progressbar: bool = False, safe_merge: bool = False, adapter_names: Optional[list[str]] = None
    ) -> torch.nn.Module:
        r"""
        This method merges the LoRa layers into the base model. This is needed if someone wants to use the base model
        as a standalone model.

        Args:
            progressbar (`bool`):
                whether to show a progressbar indicating the unload and merge process
            safe_merge (`bool`):
                whether to activate the safe merging check to check if there is any potential Nan in the adapter
                weights
            adapter_names (`List[str]`, *optional*):
                The list of adapter names that should be merged. If None, all active adapters will be merged. Defaults
                to `None`.
        Example:

        ```py
        >>> from transformers import AutoModelForCausalLM
        >>> from peft import PeftModel

        >>> base_model = AutoModelForCausalLM.from_pretrained("tiiuae/falcon-40b")
        >>> peft_model_id = "smangrul/falcon-40B-int4-peft-lora-sfttrainer-sample"
        >>> model = PeftModel.from_pretrained(base_model, peft_model_id)
        >>> merged_model = model.merge_and_unload()
        ```
        """
        return self._unload_and_optionally_merge(
            progressbar=progressbar, safe_merge=safe_merge, adapter_names=adapter_names
        )

    def unload(self) -> torch.nn.Module:
        """
        Gets back the base model by removing all the lora modules without merging. This gives back the original base
        model.
        """
        return self._unload_and_optionally_merge(merge=False)
