"""Segm Models spectral core."""
import torch
import torch.nn as nn
import segmentation_models_pytorch as smp
from models_without_stft.spectral_common import cac2cws, cws2cac


class Segm_Models_Net(nn.Module):
    def __init__(self, config, num_target_instruments):
        super().__init__()

        act = nn.GELU()

        self.num_target_instruments = num_target_instruments
        self.num_subbands = config.model.num_subbands

        dim_c = self.num_subbands * config.audio.num_channels * 2
        c = config.model.num_channels

        self.first_conv = nn.Conv2d(dim_c, c, 1, 1, 0, bias=False)

        self.unet_model = smp.Unet(
            encoder_name=config.model.encoder_name,
            encoder_weights=None,
            in_channels=c,
            classes=c,
        )

        self.final_conv = nn.Sequential(
            nn.Conv2d(c + dim_c, c, 1, 1, 0, bias=False),
            act,
            nn.Conv2d(c, self.num_target_instruments * dim_c, 1, 1, 0, bias=False)
        )

    def forward(self, x):

        mix = x = cac2cws(x, self.num_subbands)

        first_conv_out = x = self.first_conv(x)

        x = x.transpose(-1, -2)

        x = self.unet_model(x)

        x = x.transpose(-1, -2)

        x = x * first_conv_out  # reduce artifacts

        x = self.final_conv(torch.cat([mix, x], 1))

        x = cws2cac(x, self.num_subbands)

        if self.num_target_instruments > 1:
            b, c, f, t = x.shape
            x = x.reshape(b, self.num_target_instruments, -1, f, t)
        return x
