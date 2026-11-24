"""ONNX-compatible PolarFormer attention, adapted from https://huggingface.co/bgkb/bs_polarformer/blob/main/convert_to_onnx.py (MIT). Original model: ZFTurbo/MSST."""
import torch
import torch.nn as nn
from einops import rearrange
import torch.nn.functional as F

# Helper: replace PoPE attention with standard attention for ONNX export
# ---------------------------------------------------------------------------

def replace_pope_with_standard_attn(model: nn.Module):
    """Replace PoPE-based flash attention with equivalent standard attention.

    PoPE attention applies: q, k = softplus(q), softplus(k) then rotates by
    learned polar embeddings before doing scaled_dot_product_attention.
    For ONNX, we inline this into a custom forward.
    """
    class PoPEAttentionONNX(nn.Module):
        """ONNX-exportable replacement for PoPE-based Attention."""

        def __init__(self, attn_module):
            super().__init__()
            self.heads = attn_module.heads
            self.scale = attn_module.scale
            self.norm = attn_module.norm
            self.to_qkv = attn_module.to_qkv
            self.to_gates = attn_module.to_gates
            self.to_out = attn_module.to_out
            pope = attn_module.pope_embed
            if pope is None:
                raise ValueError('Expected PoPE attention')
            self.register_buffer('_pope_inv_freqs', pope.inv_freqs.detach().clone())
            self.register_buffer('_pope_bias', pope.bias.detach().clone())

        def forward(self, x):
            x_normed = self.norm(x)
            qkv = self.to_qkv(x_normed)
            q, k, v = rearrange(qkv, 'b n (qkv h d) -> qkv b h n d', qkv=3, h=self.heads)

            seq_len = q.shape[-2]
            dim = q.shape[-1]

            # Compute PoPE frequencies
            pos = torch.arange(seq_len, device=q.device, dtype=q.dtype)
            freqs = torch.einsum('i,j->ij', pos, self._pope_inv_freqs.to(q.dtype))
            bias = self._pope_bias.to(q.dtype).clamp(-2.0 * 3.14159265358979, 0.0)
            freqs_with_bias = freqs.unsqueeze(0) + bias.unsqueeze(1)  # (heads, seq, dim)

            rotate_dim = freqs.shape[-1]
            is_partial = rotate_dim < dim

            q_rot, k_rot = q, k
            if is_partial:
                q_rot = q[..., :rotate_dim]
                k_rot = k[..., :rotate_dim]
                q_rest = q[..., rotate_dim:]
                k_rest = k[..., rotate_dim:]

            # Apply softplus magnitude transform
            q_rot = F.softplus(q_rot)
            k_rot = F.softplus(k_rot)

            # Rotate q by freqs
            freqs_q = freqs.unsqueeze(0).unsqueeze(0)  # (1, 1, seq, dim)
            q_cos = q_rot * freqs_q.cos()
            q_sin = q_rot * freqs_q.sin()
            q_rot = torch.stack([q_cos, q_sin], dim=-1).flatten(-2)

            # Rotate k by freqs + bias
            freqs_k = freqs_with_bias.unsqueeze(0)  # (1, heads, seq, dim)
            k_cos = k_rot * freqs_k.cos()
            k_sin = k_rot * freqs_k.sin()
            k_rot = torch.stack([k_cos, k_sin], dim=-1).flatten(-2)

            if is_partial:
                q = torch.cat([q_rot, q_rest], dim=-1)
                k = torch.cat([k_rot, k_rest], dim=-1)
            else:
                q = q_rot
                k = k_rot

            out = F.scaled_dot_product_attention(q, k, v, scale=self.scale)

            gates = self.to_gates(x_normed)
            # gates: (B, N, H) -> (B, H, N, 1)
            out = out * rearrange(gates, 'b n h -> b h n 1').sigmoid()

            out = rearrange(out, 'b h n d -> b n (h d)')
            return self.to_out(out)

    # Walk through all transformer layers and replace Attention modules
    replaced = 0
    for layer_block in model.layers:
        for transformer in layer_block:
            for layer_pair in transformer.layers:
                attn = layer_pair[0]
                if getattr(attn, 'pope_embed', None) is not None:
                    new_attn = PoPEAttentionONNX(attn)
                    layer_pair[0] = new_attn
                    replaced += 1

    if not replaced:
        raise ValueError("No PoPE attention modules found")
    print(f"Replaced {replaced} attention modules for spectral inference/export")
    return model
