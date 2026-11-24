"""BS Roformer and BS PolarFormer spectral cores."""
import torch
from torch import nn
from torch.nn import Module, ModuleList
from typing import Tuple
from models_without_stft.roformer_common import BandSplit, MaskEstimator, RMSNorm, Transformer


class BSRoformer(Module):

    def __init__(
            self,
            dim,
            *,
            depth,
            stereo=False,
            num_stems=1,
            time_transformer_depth=2,
            freq_transformer_depth=2,
            freqs_per_bands: Tuple[int, ...],
            dim_head=64,
            heads=8,
            attn_dropout=0.,
            ff_dropout=0.,
            dim_freqs_in=1025,
            stft_n_fft=2048,
            mask_estimator_depth=2,
            mlp_expansion_factor=4,
            use_pope: bool = False,
    ):
        super().__init__()

        self.audio_channels = 2 if stereo else 1

        self.layers = ModuleList([])

        transformer_kwargs = dict(
            dim=dim,
            heads=heads,
            dim_head=dim_head,
            attn_dropout=attn_dropout,
            ff_dropout=ff_dropout,
            norm_output=False,
        )

        if use_pope:
            from models_without_stft.bs_polarformer_no_stft import PoPE
            time_pope_embed = PoPE(dim=dim_head, heads=heads)
            freq_pope_embed = PoPE(dim=dim_head, heads=heads)
            time_rotary_embed = None
            freq_rotary_embed = None
        else:
            from rotary_embedding_torch import RotaryEmbedding

            time_rotary_embed = RotaryEmbedding(dim=dim_head)
            freq_rotary_embed = RotaryEmbedding(dim=dim_head)
            time_pope_embed = freq_pope_embed = None

        for _ in range(depth):
            tran_modules = []
            tran_modules.append(
                Transformer(
                    depth=time_transformer_depth,
                    rotary_embed=time_rotary_embed,
                    pope_embed=time_pope_embed,
                    **transformer_kwargs
                )
            )
            tran_modules.append(
                Transformer(
                    depth=freq_transformer_depth,
                    rotary_embed=freq_rotary_embed,
                    pope_embed=freq_pope_embed,
                    **transformer_kwargs
                )
            )
            self.layers.append(nn.ModuleList(tran_modules))

        self.final_norm = RMSNorm(dim)

        freqs = stft_n_fft // 2 + 1
        if freqs != dim_freqs_in:
            raise ValueError(f'Expected {dim_freqs_in} STFT frequency bins, got {freqs}')

        freqs_per_bands = tuple(freqs_per_bands)
        assert sum(
            freqs_per_bands) == freqs, f'the number of freqs in the bands must equal {freqs} based on the STFT settings, but got {sum(freqs_per_bands)}'

        freqs_per_bands_with_complex = tuple(2 * f * self.audio_channels for f in freqs_per_bands)

        self.band_split = BandSplit(
            dim=dim,
            dim_inputs=freqs_per_bands_with_complex
        )

        self.mask_estimators = nn.ModuleList([])

        for _ in range(num_stems):
            mask_estimator = MaskEstimator(
                dim=dim,
                dim_inputs=freqs_per_bands_with_complex,
                num_hidden_layers=mask_estimator_depth - 1,
                mlp_expansion_factor=mlp_expansion_factor,
            )

            self.mask_estimators.append(mask_estimator)


    def forward(self, x):
        """Return complex spectral masks."""
        x = self.band_split(x)
        for time_transformer, freq_transformer in self.layers:
            b, t, f, d = x.shape
            x = x.permute(0, 2, 1, 3).reshape(b * f, t, d)
            x = time_transformer(x).reshape(b, f, t, d)
            x = x.permute(0, 2, 1, 3).reshape(b * t, f, d)
            x = freq_transformer(x).reshape(b, t, f, d)
        x = self.final_norm(x)
        masks = torch.stack([fn(x) for fn in self.mask_estimators], dim=1)
        return masks.unflatten(-1, (-1, 2)).permute(0, 1, 3, 2, 4)
