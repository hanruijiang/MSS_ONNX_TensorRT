"""MDX23C spectral core."""
import torch
import torch.nn as nn
from models_without_stft.spectral_common import cac2cws, cws2cac


class Upscale(nn.Module):
    def __init__(self, in_c, out_c, scale, norm, act):
        super().__init__()
        self.conv = nn.Sequential(
            norm(in_c),
            act,
            nn.ConvTranspose2d(in_channels=in_c, out_channels=out_c, kernel_size=scale, stride=scale, bias=False)
        )

    def forward(self, x):
        return self.conv(x)


class Downscale(nn.Module):
    def __init__(self, in_c, out_c, scale, norm, act):
        super().__init__()
        self.conv = nn.Sequential(
            norm(in_c),
            act,
            nn.Conv2d(in_channels=in_c, out_channels=out_c, kernel_size=scale, stride=scale, bias=False)
        )

    def forward(self, x):
        return self.conv(x)


class TFC_TDF(nn.Module):
    def __init__(self, in_c, c, l, f, bn, norm, act):
        super().__init__()

        self.blocks = nn.ModuleList()
        for _ in range(l):
            block = nn.Module()

            block.tfc1 = nn.Sequential(
                norm(in_c),
                act,
                nn.Conv2d(in_c, c, 3, 1, 1, bias=False),
            )
            block.tdf = nn.Sequential(
                norm(c),
                act,
                nn.Linear(f, f // bn, bias=False),
                norm(c),
                act,
                nn.Linear(f // bn, f, bias=False),
            )
            block.tfc2 = nn.Sequential(
                norm(c),
                act,
                nn.Conv2d(c, c, 3, 1, 1, bias=False),
            )
            block.shortcut = nn.Conv2d(in_c, c, 1, 1, 0, bias=False)

            self.blocks.append(block)
            in_c = c

    def forward(self, x):
        for block in self.blocks:
            s = block.shortcut(x)
            x = block.tfc1(x)
            x = x + block.tdf(x)
            x = block.tfc2(x)
            x = x + s
        return x


class TFC_TDF_net(nn.Module):
    def __init__(self, config, num_target_instruments):
        super().__init__()

        norm = lambda c: nn.InstanceNorm2d(c, affine=True)
        act = nn.GELU()

        self.num_target_instruments = num_target_instruments
        self.num_subbands = config.model.num_subbands

        dim_c = self.num_subbands * config.audio.num_channels * 2
        n = config.model.num_scales
        scale = config.model.scale
        l = config.model.num_blocks_per_scale
        c = config.model.num_channels
        g = config.model.growth
        bn = config.model.bottleneck_factor
        f = config.audio.dim_f // self.num_subbands

        self.first_conv = nn.Conv2d(dim_c, c, 1, 1, 0, bias=False)

        self.encoder_blocks = nn.ModuleList()
        for i in range(n):
            block = nn.Module()
            block.tfc_tdf = TFC_TDF(c, c, l, f, bn, norm, act)
            block.downscale = Downscale(c, c + g, scale, norm, act)
            f = f // scale[1]
            c += g
            self.encoder_blocks.append(block)

        self.bottleneck_block = TFC_TDF(c, c, l, f, bn, norm, act)

        self.decoder_blocks = nn.ModuleList()
        for i in range(n):
            block = nn.Module()
            block.upscale = Upscale(c, c - g, scale, norm, act)
            f = f * scale[1]
            c -= g
            block.tfc_tdf = TFC_TDF(2 * c, c, l, f, bn, norm, act)
            self.decoder_blocks.append(block)

        self.final_conv = nn.Sequential(
            nn.Conv2d(c + dim_c, c, 1, 1, 0, bias=False),
            act,
            nn.Conv2d(c, self.num_target_instruments * dim_c, 1, 1, 0, bias=False)
        )

    def forward(self, x):

        mix = x = cac2cws(x, self.num_subbands)

        first_conv_out = x = self.first_conv(x)

        x = x.transpose(-1, -2)

        encoder_outputs = []
        for block in self.encoder_blocks:
            x = block.tfc_tdf(x)
            encoder_outputs.append(x)
            x = block.downscale(x)

        x = self.bottleneck_block(x)

        for block in self.decoder_blocks:
            x = block.upscale(x)
            x = torch.cat([x, encoder_outputs.pop()], 1)
            x = block.tfc_tdf(x)

        x = x.transpose(-1, -2)

        x = x * first_conv_out  # reduce artifacts

        x = self.final_conv(torch.cat([mix, x], 1))

        x = cws2cac(x, self.num_subbands)

        if self.num_target_instruments > 1:
            b, c, f, t = x.shape
            x = x.reshape(b, self.num_target_instruments, -1, f, t)

        return x
