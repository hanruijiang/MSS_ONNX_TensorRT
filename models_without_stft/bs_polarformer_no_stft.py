"""ONNX-compatible PolarFormer core, adapted from https://huggingface.co/bgkb/bs_polarformer/blob/main/convert_to_onnx.py (MIT). Original model: ZFTurbo/MSST."""
import torch
import torch.nn as nn

class BSRoformerCore(nn.Module):
    """Wraps the inner neural-network portion of BSRoformer for ONNX export.

    Input:  stft_features  – (B, T, F*C) where F = n_freq_bins * audio_channels, C = 2 (real/imag)
    Output: mask           – (B, num_stems, F, T, 2) complex mask as real tensor
    """

    def __init__(self, model: nn.Module):
        super().__init__()
        self.band_split = model.band_split
        self.layers = model.layers
        self.final_norm = model.final_norm
        self.mask_estimators = model.mask_estimators
        self.skip_connection = model.skip_connection

    def forward(self, x):
        # x: (B, T, F_C)  where F_C = total_freq_bins * audio_channels * 2
        x = self.band_split(x)  # (B, T, num_bands, D)

        store = [None] * len(self.layers)
        for i, transformer_block in enumerate(self.layers):
            if len(transformer_block) == 3:
                linear_transformer, time_transformer, freq_transformer = transformer_block
                b, t, f, d = x.shape
                x = x.reshape(b, t * f, d)
                x = linear_transformer(x)
                x = x.reshape(b, t, f, d)
            else:
                time_transformer, freq_transformer = transformer_block

            if self.skip_connection:
                for j in range(i):
                    x = x + store[j]

            # Time transformer: process each frequency band across time
            x = x.permute(0, 2, 1, 3)  # (B, F, T, D)
            b, f, t, d = x.shape
            x = x.reshape(b * f, t, d)
            x = time_transformer(x)
            x = x.reshape(b, f, t, d)

            # Freq transformer: process each time step across frequency bands
            x = x.permute(0, 2, 1, 3)  # (B, T, F, D)
            b, t, f, d = x.shape
            x = x.reshape(b * t, f, d)
            x = freq_transformer(x)
            x = x.reshape(b, t, f, d)

            if self.skip_connection:
                store[i] = x

        x = self.final_norm(x)

        # Mask estimation
        masks = torch.stack([fn(x) for fn in self.mask_estimators], dim=1)
        # masks: (B, num_stems, T, F_C)
        masks = masks.unflatten(-1, (-1, 2))
        # masks: (B, num_stems, T, F, 2)  -> rearrange to (B, num_stems, F, T, 2)
        masks = masks.permute(0, 1, 3, 2, 4)
        return masks


# ---------------------------------------------------------------------------
# Helper: replace PoPE attention with standard attention for ONNX export
# ---------------------------------------------------------------------------

def replace_pope_with_standard_attn(model: nn.Module):
    """Replace PoPE-based flash attention with equivalent standard attention.

    PoPE attention applies: q, k = softplus(q), softplus(k) then rotates by
    learned polar embeddings before doing scaled_dot_product_attention.
    For ONNX, we inline this into a custom forward.
    """
    from einops import rearrange
    import torch.nn.functional as F

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
            self.pope_embed = attn_module.pope_embed
            self.has_pope = attn_module.pope_embed is not None
            self.has_rotary = attn_module.rotary_embed is not None
            self.rotary_embed = attn_module.rotary_embed

            # Pre-extract PoPE parameters
            if self.has_pope:
                self.pope_inv_freqs = attn_module.pope_embed.inv_freqs.clone()
                self.pope_bias = attn_module.pope_embed.bias.clone()
                # Register as buffers
                self.register_buffer('_pope_inv_freqs', self.pope_inv_freqs)
                self.register_buffer('_pope_bias', nn.Parameter(self.pope_bias))

        def forward(self, x):
            x_normed = self.norm(x)
            qkv = self.to_qkv(x_normed)
            q, k, v = rearrange(qkv, 'b n (qkv h d) -> qkv b h n d', qkv=3, h=self.heads)

            if self.has_pope:
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

            elif self.has_rotary:
                q = self.rotary_embed.rotate_queries_or_keys(q)
                k = self.rotary_embed.rotate_queries_or_keys(k)
                out = F.scaled_dot_product_attention(q, k, v)
            else:
                out = F.scaled_dot_product_attention(q, k, v)

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
                if hasattr(attn, 'pope_embed') or hasattr(attn, 'rotary_embed'):
                    new_attn = PoPEAttentionONNX(attn)
                    layer_pair[0] = new_attn
                    replaced += 1

    print(f"Replaced {replaced} attention modules for ONNX export")
    return model


