import torch
from torch import Tensor
import torch.nn as nn
import torch.nn.functional as F


class L2Norm(nn.Module):

    def __init__(self, dim):
        super().__init__()

    def forward(self, x):
        return F.normalize(x, dim=-1)

  
NORM_LAYERS = {
    "none": nn.Identity,
    "l2": L2Norm,
    "ln": nn.LayerNorm,
}


class HSigmoidv2(nn.Module):
    """(add ref)"""

    def __init__(self, inplace=True):
        super().__init__()
        self.inplace = inplace

    def forward(self, x, norm_factor=None):
        out = F.relu6(x + 3.0, inplace=self.inplace) / 6.0
        if norm_factor is not None:
            out = out / norm_factor
        return out

 
class Sigmoid(nn.Sigmoid):

    def forward(self, x, norm_factor=None):
        out = super().forward(x)
        if norm_factor is not None:
            out = out / norm_factor
        return out


class Softmax(nn.Softmax):

    def forward(self, x, norm_factor=None):
        # norm_factor is only for compatibility
        out = super().forward(x)
        return out


ACTIVATION_LAYERS = {
    "sigmoid": Sigmoid,
    "hsigmoid": HSigmoidv2,
    "softmax": Softmax,
    "gelu": nn.GELU,
    "standard_sigmoid": nn.Sigmoid,
}
