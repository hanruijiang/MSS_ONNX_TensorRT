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


def pope_attention(q, k, v, pope_embed, scale):
    pos = torch.arange(q.shape[-2], device=q.device, dtype=q.dtype)
    freqs = torch.einsum('i,j->ij', pos, pope_embed.inv_freqs.to(q.dtype))
    bias = pope_embed.bias.detach().to(q.dtype).clamp(-2.0 * math.pi, 0.0)
    freqs_q = freqs.unsqueeze(0).unsqueeze(0)
    freqs_k = (freqs.unsqueeze(0) + bias.unsqueeze(1)).unsqueeze(0)

    q = F.softplus(q)
    k = F.softplus(k)
    q = torch.stack([q * freqs_q.cos(), q * freqs_q.sin()], dim=-1).flatten(-2)
    k = torch.stack([k * freqs_k.cos(), k * freqs_k.sin()], dim=-1).flatten(-2)
    return F.scaled_dot_product_attention(q, k, v, scale=scale)
