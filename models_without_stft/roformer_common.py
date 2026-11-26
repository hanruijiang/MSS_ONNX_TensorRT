"""Shared Roformer network layers."""
import torch
from torch import nn, einsum
from torch.nn import Module, ModuleList
import torch.nn.functional as F
from typing import Tuple
from einops import rearrange
from models_without_stft.bs_polarformer_no_stft import pope_attention


class Attend(nn.Module):
    def __init__(self, dropout=0.):
        super().__init__()
        self.attn_dropout = nn.Dropout(dropout)
        self.use_matmul = False

    def forward(self, q, k, v):
        scale = q.shape[-1] ** -0.5
        if self.use_matmul:
            sim = torch.matmul(q, k.transpose(-1, -2)) * scale
            return torch.matmul(self.attn_dropout(sim.softmax(dim=-1)), v)
        sim = einsum('b h i d, b h j d -> b h i j', q, k) * scale
        attn = self.attn_dropout(sim.softmax(dim=-1))
        return einsum('b h i j, b h j d -> b h i d', attn, v)


class RMSNorm(Module):
    def __init__(self, dim):
        super().__init__()
        self.scale = dim ** 0.5
        self.gamma = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        return F.normalize(x, dim=-1) * self.scale * self.gamma


class FeedForward(Module):
    def __init__(
            self,
            dim,
            mult=4,
            dropout=0.
    ):
        super().__init__()
        dim_inner = int(dim * mult)
        self.net = nn.Sequential(
            RMSNorm(dim),
            nn.Linear(dim, dim_inner),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim_inner, dim),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        return self.net(x)


class Attention(Module):
    def __init__(
            self,
            dim,
            heads=8,
            dim_head=64,
            dropout=0.,
            rotary_embed=None,
            pope_embed=None
    ):
        super().__init__()
        self.heads = heads
        self.scale = dim_head ** -0.5
        dim_inner = heads * dim_head

        self.rotary_embed = rotary_embed
        self.pope_embed = pope_embed
        self.attend = Attend(dropout=dropout)

        self.norm = RMSNorm(dim)
        self.to_qkv = nn.Linear(dim, dim_inner * 3, bias=False)

        self.to_gates = nn.Linear(dim, heads)

        self.to_out = nn.Sequential(
            nn.Linear(dim_inner, dim, bias=False),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        x = self.norm(x)

        if hasattr(self, 'to_qkv_gate'):
            qkv, gates = self.to_qkv_gate(x).split([self.to_qkv_gate.out_features - self.heads, self.heads], dim=-1)
        else:
            qkv, gates = self.to_qkv(x), self.to_gates(x)
        q, k, v = rearrange(qkv, 'b n (qkv h d) -> qkv b h n d', qkv=3, h=self.heads)

        if self.pope_embed is not None:
            out = pope_attention(q, k, v, self.pope_embed, self.scale)
        else:
            q = self.rotary_embed.rotate_queries_or_keys(q)
            k = self.rotary_embed.rotate_queries_or_keys(k)
            out = self.attend(q, k, v)

        out = out * rearrange(gates, 'b n h -> b h n 1').sigmoid()

        out = rearrange(out, 'b h n d -> b n (h d)')
        return self.to_out(out)


class Transformer(Module):
    def __init__(
            self,
            *,
            dim,
            depth,
            dim_head=64,
            heads=8,
            attn_dropout=0.,
            ff_dropout=0.,
            ff_mult=4,
            norm_output=True,
            rotary_embed=None,
            pope_embed=None,
    ):
        super().__init__()
        self.layers = ModuleList([])

        for _ in range(depth):
            attn = Attention(
                dim=dim,
                dim_head=dim_head,
                heads=heads,
                dropout=attn_dropout,
                rotary_embed=rotary_embed,
                pope_embed=pope_embed
            )

            self.layers.append(ModuleList([
                attn,
                FeedForward(dim=dim, mult=ff_mult, dropout=ff_dropout)
            ]))

        self.norm = RMSNorm(dim) if norm_output else nn.Identity()

    def forward(self, x):

        for attn, ff in self.layers:
            x = attn(x) + x
            x = ff(x) + x

        return self.norm(x)


class BandSplit(Module):
    def __init__(
            self,
            dim,
            dim_inputs: Tuple[int, ...]
    ):
        super().__init__()
        self.dim_inputs = dim_inputs
        self.to_features = ModuleList([])

        for dim_in in dim_inputs:
            net = nn.Sequential(
                RMSNorm(dim_in),
                nn.Linear(dim_in, dim)
            )

            self.to_features.append(net)

    def forward(self, x):
        x = x.split(self.dim_inputs, dim=-1)

        outs = []
        for split_input, to_feature in zip(x, self.to_features):
            split_output = to_feature(split_input)
            outs.append(split_output)

        return torch.stack(outs, dim=-2)


def MLP(dim_in, dim_out, dim_hidden, num_hidden_layers):
    net = []
    dims = (dim_in, *((dim_hidden,) * num_hidden_layers), dim_out)

    for ind, (layer_dim_in, layer_dim_out) in enumerate(zip(dims[:-1], dims[1:])):
        is_last = ind == (len(dims) - 2)

        net.append(nn.Linear(layer_dim_in, layer_dim_out))

        if is_last:
            continue

        net.append(nn.Tanh())

    return nn.Sequential(*net)


class MaskEstimator(Module):
    def __init__(
            self,
            dim,
            dim_inputs: Tuple[int, ...],
            num_hidden_layers,
            mlp_expansion_factor=4
    ):
        super().__init__()
        self.to_freqs = ModuleList([])
        dim_hidden = dim * mlp_expansion_factor

        for dim_in in dim_inputs:

            mlp = nn.Sequential(
                MLP(dim, dim_in * 2, dim_hidden, num_hidden_layers),
                nn.GLU(dim=-1)
            )

            self.to_freqs.append(mlp)

    def forward(self, x):
        x = x.unbind(dim=-2)

        outs = []

        for band_features, mlp in zip(x, self.to_freqs):
            freq_out = mlp(band_features)
            outs.append(freq_out)

        return torch.cat(outs, dim=-1)
