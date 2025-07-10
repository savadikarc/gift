from collections import OrderedDict
from typing import Any, Mapping
import torch.nn as nn


class BitFiTWrapper(nn.Module):

    def __init__(self, backbone):
        super().__init__()
        self.backbone = backbone

        # Set the bias values to trainable
        for module in self.backbone.modules():
            if isinstance(module, nn.Linear):
                module.bias.requires_grad = True

    def forward(self, x):
        return self.backbone(x)
    
    def state_dict(self, destination=None, prefix='', keep_vars=False):
        # Only need to save the hypernets and classifier
        to_return = OrderedDict()
        state_dict = self.backbone.state_dict(destination=destination, prefix=prefix, keep_vars=keep_vars)
        for key, value in state_dict.items():
            if "head" in key:
                to_return[key] = value
            if "bias" in key:
                to_return[key] = value
        return to_return
    
    def load_state_dict(self, state_dict: Mapping[str, Any], strict: bool = False, assign: bool = False):
        return super().load_state_dict(state_dict, strict, assign)
