"""Explicit attention for the retained Rotary spectral cores."""
from torch import nn, einsum


class Attend(nn.Module):
    def __init__(self, dropout=0., scale=None):
        super().__init__()
        self.scale = scale
        self.attn_dropout = nn.Dropout(dropout)

    def forward(self, q, k, v):
        scale = self.scale if self.scale is not None else q.shape[-1] ** -0.5
        sim = einsum('b h i d, b h j d -> b h i j', q, k) * scale
        attn = self.attn_dropout(sim.softmax(dim=-1))
        return einsum('b h i j, b h j d -> b h i d', attn, v)
