from typing import Callable, Tuple
import math
import logging

import torch
import torch.nn as nn
import torch.nn.functional as F

from timm.layers import use_fused_attn, DropPath
from torch.jit import Final

from .layers import ACTIVATION_LAYERS, NORM_LAYERS, L2Norm, Sigmoid, Softmax


_logger = logging.getLogger(__name__)


class Mlp(nn.Module):
    """ MLP as used in Vision Transformer, MLP-Mixer and related networks
    """
    def __init__(
            self,
            in_features,
            hidden_features=None,
            out_features=None,
            act_layer=nn.GELU,
            bias=True,
    ):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features

        self.fc1 = nn.Linear(in_features, hidden_features, bias=bias)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features, bias=bias)
    
    @torch.cuda.amp.autocast(enabled=False)
    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.fc2(x)
        return x
    
    def reset_parameters(self):
        self.fc1.reset_parameters()
        self.fc2.reset_parameters()


class Cluster(nn.Module):

    def __init__(self, dim, num_clusters, activation="sigmoid", norm_layer="l2", bias=True, **kwargs):
        super().__init__()
        self.cluster = nn.Linear(dim, num_clusters, bias=bias)

        self.norm = norm_layer(dim)
        if activation is Softmax:
            kwargs["dim"] = -1
        self.activation = ACTIVATION_LAYERS.get(activation, None)(**kwargs)
    @torch.cuda.amp.autocast(enabled=False)
    def forward(self, x):
        c = self.cluster(x).transpose(-2, -1) # B, M, N
        B, M, N = c.shape
        c = self.activation(c, N)
        z = torch.einsum("bmn,bnc->bmc", c, x) # B, M, C
        z = self.norm(z)
    
        return z


class PamCatAttention(nn.Module):
    fused_attn: Final[bool]

    def __init__(
            self,
            dim,
            num_clusters,
            num_heads=4,
            qkv_bias=True,
            norm_layer=L2Norm,
            cluster_activation=Sigmoid,
    ):
        super().__init__()
        assert dim % num_heads == 0, 'dim should be divisible by num_heads'
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5
        self.fused_attn = use_fused_attn()

        self.dim = dim

        self.q = nn.Linear(dim, dim, bias=qkv_bias)
        self.k = nn.Linear(dim, dim, bias=qkv_bias)
        self.v = nn.Linear(dim, dim, bias=qkv_bias)
        self.cluster = Cluster(dim, num_clusters, activation=cluster_activation, norm_layer=norm_layer, bias=qkv_bias)
        self.proj = nn.Linear(dim, dim)

    @torch.cuda.amp.autocast(enabled=False)
    def forward(self, x, visualize=False):
        B, N, C = x.shape

        # Cluster
        z = self.cluster(x) # B, M, C
        B_, M, C_ = z.shape

        q = self.q(x).reshape(B, N, self.num_heads, self.head_dim).permute(0, 2, 1, 3) # B, H, N, d
        k = self.k(z).reshape(B, M, self.num_heads, self.head_dim).permute(0, 2, 1, 3) # B, H, M, d
        v = self.v(z).reshape(B, M, self.num_heads, self.head_dim).permute(0, 2, 1, 3) # B, H, M, d

        vis_attn = None
        if self.fused_attn and not visualize:
            x = F.scaled_dot_product_attention(
                q, k, v,
                dropout_p=0.,
            )
        else:
            q = q * self.scale
            attn = q @ k.transpose(-2, -1) # B, H, N, M
            attn = attn.softmax(dim=-1)
            if visualize:
                vis_attn = attn
            x = attn @ v

        x = x.transpose(1, 2).reshape(B, N, self.dim)
        x = self.proj(x)
        return x, vis_attn


class Attention(nn.Module):
    fused_attn: Final[bool]

    def __init__(
            self,
            dim,
            num_heads=8,
            qkv_bias=True,
    ):
        super().__init__()
        assert dim % num_heads == 0, 'dim should be divisible by num_heads'
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5
        self.fused_attn = use_fused_attn()

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.proj = nn.Linear(dim, dim)

    @torch.cuda.amp.autocast(enabled=False)
    def forward(self, x, visualize=False):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)

        vis_attn = None
        if self.fused_attn and not visualize:
            x = F.scaled_dot_product_attention(
                q, k, v,
                dropout_p=0.,
            )
        else:
            q = q * self.scale
            attn = q @ k.transpose(-2, -1)
            attn = attn.softmax(dim=-1)
            if visualize:
                vis_attn = attn
            attn = self.attn_drop(attn)
            x = attn @ v

        x = x.transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        return x, vis_attn


class TransformerBlock(nn.Module):

    def __init__(
            self,
            dim,
            num_heads,
            mlp_ratio,
            qkv_bias=True,
            act_layer=nn.GELU,
            drop_path=0.,
            norm_layer="l2",
            mlp_layer=Mlp,
            attn_block=Attention,
            **attn_kwargs
    ):
        super().__init__()

        self.norm1 = NORM_LAYERS.get(norm_layer, nn.Identity())(dim)
        self.attn = attn_block(
            dim,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            **attn_kwargs
        )
        self.drop_path1 = DropPath(drop_path) if drop_path > 0. else nn.Identity()

        self.norm2 = NORM_LAYERS.get(norm_layer, nn.Identity())(dim)
        self.mlp = mlp_layer(
            in_features=dim,
            hidden_features=int(dim * mlp_ratio),
            act_layer=act_layer,
        )
        self.drop_path2 = DropPath(drop_path) if drop_path > 0. else nn.Identity()

    @torch.cuda.amp.autocast(enabled=False)
    def forward(self, w):
        _w, vis_attn = self.attn(self.norm1(w))
        w = w + self.drop_path1(_w)
        w = w + self.drop_path2(self.mlp(self.norm2(w)))
        return w


class VanillaTransformerBlock(TransformerBlock):

    def __init__(
            self,
            dim,
            num_heads,
            mlp_ratio,
            qkv_bias=True,
            act_layer=nn.GELU,
            drop_path=0.,
            norm_layer="l2",
            mlp_layer=Mlp,
    ):
        super().__init__(
            dim,
            num_heads,
            mlp_ratio,
            qkv_bias=qkv_bias,
            act_layer=act_layer,
            drop_path=drop_path,
            norm_layer=norm_layer,
            mlp_layer=mlp_layer,
            attn_block=Attention,
        )
    

class PamCatTransformerBlock(TransformerBlock):

    def __init__(
            self, 
            dim, 
            num_heads, 
            mlp_ratio, 
            qkv_bias=True, 
            act_layer=nn.GELU, 
            drop_path=0, 
            norm_layer="l2", 
            mlp_layer=Mlp,
            num_clusters=64,
            cluster_activation="sigmoid",
        ):
        super().__init__(
            dim,
            num_heads,
            mlp_ratio,
            qkv_bias=qkv_bias,
            act_layer=act_layer,
            drop_path=drop_path,
            norm_layer=norm_layer,
            mlp_layer=mlp_layer,
            attn_block=PamCatAttention,
            num_clusters=num_clusters,
            cluster_activation=cluster_activation,
        )


class PamCatAttnBlock(nn.Module):

    def __init__(
            self, 
            dim, 
            num_heads, 
            qkv_bias=True,
            norm_layer="l2", 
            num_clusters=64,
            cluster_activation="sigmoid",
            drop_path=0.,
        ):
        super().__init__()
        self.norm = NORM_LAYERS.get(norm_layer, nn.Identity())(dim)
        self.attn = PamCatAttention(
            dim,
            num_clusters=num_clusters,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            cluster_activation=cluster_activation,
        )

    @torch.cuda.amp.autocast(enabled=False)
    def forward(self, w):
        _w, vis_attn = self.attn(self.norm(w))
        w = w + _w
        return w


class MLPBlock(nn.Module):

    def __init__(
            self,
            dim,
            mlp_ratio=2.,
            act_layer=nn.GELU,
            drop_path=0.,
            norm_layer="l2",
            mlp_layer=Mlp,
    ):
        super().__init__()
        self.norm1 = NORM_LAYERS.get(norm_layer, nn.Identity())(dim)
        self.mlp = mlp_layer(
            in_features=dim,
            hidden_features=int(dim * mlp_ratio),
            act_layer=act_layer,
        )
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

    @torch.cuda.amp.autocast(enabled=False)
    def forward(self, w):
        w = w + self.drop_path(self.mlp(self.norm1(w)))
        return w
    

class MLPMixerBlock(nn.Module):

    def __init__(
            self,
            dim,
            num_tokens,
            num_mixed_tokens,
            channel_mixing_ratio=2.,
            act_layer=nn.GELU,
            drop_path=0.,
            norm_layer="l2",
            mlp_layer=Mlp,
    ):
        super().__init__()
        self.norm1 = NORM_LAYERS.get(norm_layer, nn.Identity())(dim)
        self.mlp1 = mlp_layer(
            in_features=num_tokens,
            hidden_features=num_mixed_tokens,
            act_layer=act_layer,
        )
        self.drop_path1 = DropPath(drop_path) if drop_path > 0. else nn.Identity()

        self.norm2 = NORM_LAYERS.get(norm_layer, nn.Identity())(dim)
        self.mlp2 = mlp_layer(
            in_features=dim,
            hidden_features=int(dim * channel_mixing_ratio),
            act_layer=act_layer,
        )
        self.drop_path2 = DropPath(drop_path) if drop_path > 0. else nn.Identity()

    @torch.cuda.amp.autocast(enabled=False)
    def forward(self, w):
        w = w + self.drop_path1(self.mlp1(self.norm1(w).transpose(1, 2)).transpose(1, 2))
        w = w + self.drop_path2(self.mlp2(self.norm2(w)))
        return w


class SimpleBlock(nn.Module):

    def __init__(
            self,
            dim,
            drop_path=0., # Remove later, kept now for compatibility
            act_layer="identity",
    ):
        super().__init__()
        self.act = ACTIVATION_LAYERS.get(act_layer, nn.Identity)()

    @torch.cuda.amp.autocast(enabled=False)
    def forward(self, w):
        return self.act(w)


class Hypernet(nn.Module):

    def __init__(
            self, 
            block_fn: Callable,
            num_blocks=1,
            rank=16,
            drop_path=0.,
            **block_kwargs
        ) -> None:
        super().__init__()

        self.rank = rank

        dpr = [x.item() for x in torch.linspace(0, drop_path, num_blocks)]  # stochastic depth decay rule
        self.blocks = nn.ModuleList([
            block_fn(
                rank,
                drop_path=dpr[i],
                **block_kwargs,
            )
            for i in range(num_blocks)
        ])

    @torch.cuda.amp.autocast(enabled=False)
    def forward(self, w: torch.Tensor):
        
        # TODO: check if we need to cast to desired dtype

        if w.ndim == 2:
            w = w.unsqueeze(0)
        else:
            assert w.ndim == 3, f"Invalid input shape {w.shape}"

        _, out_features, rank = w.shape
        assert rank == self.rank, f"Input shape {w.shape[1:]} does not match expected shape {(out_features, self.rank)}"

        # Transformer
        for block in self.blocks:
            w = block(w)
        
        return w[0]


BLOCK_FNS = dict(
    transformer=VanillaTransformerBlock,
    pamcat_transformer=PamCatTransformerBlock,
    pamcat_attn=PamCatAttnBlock,
    mlp=MLPBlock,
    mlp_mixer=MLPMixerBlock,
    simple_block=SimpleBlock,
)

