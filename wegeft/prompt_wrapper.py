from operator import mul
from functools import reduce
from collections import OrderedDict
import math
import torch
import torch.nn as nn

from timm.models.vision_transformer import VisionTransformer

class PromptWrapper(nn.Module):

    def __init__(self, backbone: VisionTransformer, patch_size=16, num_prompt_tokens=50, prompt_dim=768, deep_prompts=True):

        super().__init__()
        self.backbone = backbone
        self.num_prompt_tokens = num_prompt_tokens
        self.prompt_dim = prompt_dim
        self.deep_prompts = deep_prompts

        val = math.sqrt(6. / float(3 * reduce(mul, patch_size, 1) + prompt_dim))  # noqa

        self.prompt_embeddings = nn.Parameter(torch.zeros(
            1, num_prompt_tokens, prompt_dim))
        # xavier_uniform initialization
        nn.init.uniform_(self.prompt_embeddings.data, -val, val)

        if deep_prompts:  # noqa

            total_d_layer = len(backbone.blocks) - 1
            self.deep_prompt_embeddings = nn.Parameter(torch.zeros(
                total_d_layer, num_prompt_tokens, prompt_dim))
            # xavier_uniform initialization
            nn.init.uniform_(self.deep_prompt_embeddings.data, -val, val)

        # Attach pre-hooks
        self.backbone.blocks[0].register_forward_pre_hook(self.prompt_hook(0))
        if self.deep_prompts:
            for l, block in enumerate(self.backbone.blocks[1:]):
                block.register_forward_pre_hook(self.prompt_hook(l+1))

    
    def prompt_hook(self, layer):

        def hook(module, args):
            is_input_tuple = isinstance(args, tuple)
            input = args[0] if is_input_tuple else args

            B, N, C = input.shape
            cls_token = input[:, [0], :]
            # Attach prompt
            if layer == 0:
                prompt = self.prompt_embeddings.expand(B, -1, -1)
                input = torch.cat([cls_token, prompt, input[:, 1:, :]], dim=1)
            elif self.deep_prompts:
                prompt = self.deep_prompt_embeddings[layer - 1].expand(B, -1, -1)
                input = torch.cat([cls_token, prompt, input[:, self.num_prompt_tokens+1:, :]], dim=1)
            
            return input if not is_input_tuple else (input, *args[1:])
        
        return hook
        
    def forward(self, x):
        return self.backbone(x)

    def state_dict(self, destination=None, prefix='', keep_vars=False):
        to_return = OrderedDict()
        state_dict = super().state_dict(destination, prefix, keep_vars)
        for k, v in state_dict.items():
            if "head" in k or "prompt" in k:
                to_return[k] = v

        return to_return
