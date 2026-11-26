"""PoPE positional encoding and attention.

Adapted from lucidrains/PoPE-pytorch and bgkb/bs_polarformer (MIT):
https://github.com/lucidrains/PoPE-pytorch
https://huggingface.co/bgkb/bs_polarformer/blob/main/convert_to_onnx.py
Original model: ZFTurbo/MSST.
"""
import math

import torch
from torch import nn
import torch.nn.functional as F


class PoPE(nn.Module):
    def __init__(self, dim, heads, theta=10000):
        super().__init__()
        inv_freqs = theta ** -(torch.arange(dim).float() / dim)
        self.register_buffer('inv_freqs', inv_freqs)
        self.bias = nn.Parameter(torch.zeros(heads, dim))


def pope_tables(embed, length):
    pos = torch.arange(length, device=embed.inv_freqs.device, dtype=embed.inv_freqs.dtype)
    freqs = pos[:, None] * embed.inv_freqs[None, :]
    bias = embed.bias.detach().clamp(-2.0 * math.pi, 0.0)
    q = freqs[None, None, :, :]
    k = (freqs[None, :, :] + bias[:, None, :])[None, :, :, :]
    return q.cos(), q.sin(), k.cos(), k.sin()


def prepare_pope(embed, length):
    for name, value in zip(('q_cos', 'q_sin', 'k_cos', 'k_sin'), pope_tables(embed, length)):
        embed.register_buffer(name, value, persistent=False)


def pope_attention(q, k, v, pope_embed, scale):
    if hasattr(pope_embed, 'q_cos'):
        q_cos, q_sin, k_cos, k_sin = (pope_embed.q_cos, pope_embed.q_sin,
                                      pope_embed.k_cos, pope_embed.k_sin)
    else:
        q_cos, q_sin, k_cos, k_sin = pope_tables(pope_embed, q.shape[-2])
    q = F.softplus(q)
    k = F.softplus(k)
    q = torch.stack([q * q_cos, q * q_sin], dim=-1).flatten(-2)
    k = torch.stack([k * k_cos, k * k_sin], dim=-1).flatten(-2)
    return F.scaled_dot_product_attention(q, k, v, scale=scale)
