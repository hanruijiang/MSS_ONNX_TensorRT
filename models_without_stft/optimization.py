"""Spectral-core inference optimizations (adapted from MSST, MIT)."""
from collections import defaultdict

import torch
from torch import nn
from torch.nn import functional as F
from torch.nn.utils.fusion import fuse_conv_bn_eval


# Group equal-width bands to limit temporary activations.
def _groups(keys, group_size):
    groups = defaultdict(list)
    for index, key in enumerate(keys):
        groups[key].append(index)
    for indices in groups.values():
        for start in range(0, len(indices), group_size):
            yield tuple(indices[start:start + group_size])


class PackedLinear(nn.Module):
    """Independent per-band affine transforms, evaluated by batched GEMM."""

    def __init__(self, linears):
        super().__init__()
        self.register_buffer('weight', torch.stack([
            layer.weight.detach().transpose(0, 1) for layer in linears
        ]).contiguous())
        self.register_buffer('bias', torch.stack([
            layer.bias.detach() for layer in linears
        ]).unsqueeze(1) if linears[0].bias is not None else None)

    def forward(self, x):
        # [bands, ..., input_features] -> [bands, tokens, input_features]
        flat = x.reshape(x.shape[0], -1, x.shape[-1])
        if self.bias is None:
            out = torch.matmul(flat, self.weight)
        else:
            out = torch.matmul(flat, self.weight) + self.bias
        return out.reshape(*x.shape[:-1], self.weight.shape[-1])


class _BandGroup(nn.Module):
    def __init__(self, indices, nets):
        super().__init__()
        self.indices = indices
        self.scale = nets[0][0].scale
        self.register_buffer('gamma', torch.stack([
            net[0].gamma.detach() for net in nets
        ]))
        self.linear = PackedLinear([net[1] for net in nets])

    def forward(self, splits):
        x = torch.stack([splits[i] for i in self.indices])
        gamma = self.gamma.reshape(self.gamma.shape[0], *([1] * (x.ndim - 2)), self.gamma.shape[1])
        x = F.normalize(x, dim=-1) * self.scale * gamma
        return self.linear(x)


class PackedBandSplit(nn.Module):
    def __init__(self, original, group_size=4):
        super().__init__()
        self.dim_inputs = original.dim_inputs
        self.groups = nn.ModuleList([
            _BandGroup(indices, [original.to_features[i] for i in indices])
            for indices in _groups(self.dim_inputs, group_size)
        ])

    def forward(self, x):
        splits = x.split(self.dim_inputs, dim=-1)
        outs = [None] * len(self.dim_inputs)
        for group in self.groups:
            for index, value in zip(group.indices, group(splits).unbind(0)):
                outs[index] = value
        return torch.stack(outs, dim=-2)


class _MaskGroup(nn.Module):
    def __init__(self, indices, nets):
        super().__init__()
        self.indices = indices
        layers = []
        for position, layer in enumerate(nets[0][0]):
            if type(layer) is nn.Linear:
                layers.append(PackedLinear([net[0][position] for net in nets]))
            else:
                layers.append(layer)
        layers.append(nn.GLU(dim=-1))
        self.net = nn.Sequential(*layers)

    def forward(self, bands):
        return self.net(torch.stack([bands[i] for i in self.indices]))


def _mask_signature(net):
    return tuple((type(layer), layer.in_features, layer.out_features, layer.bias is not None)
                 if type(layer) is nn.Linear else (type(layer),)
                 for layer in net[0])


class PackedMaskEstimator(nn.Module):
    def __init__(self, original, group_size=4):
        super().__init__()
        self.num_bands = len(original.to_freqs)
        self.groups = nn.ModuleList([
            _MaskGroup(indices, [original.to_freqs[i] for i in indices])
            for indices in _groups([_mask_signature(net) for net in original.to_freqs], group_size)
        ])

    def forward(self, x):
        bands = x.unbind(dim=-2)
        outs = [None] * self.num_bands
        for group in self.groups:
            for index, value in zip(group.indices, group(bands).unbind(0)):
                outs[index] = value
        return torch.cat(outs, dim=-1)


def _fuse_qkv_gate(attn):
    qkv, gate = attn.to_qkv, attn.to_gates
    fused = nn.Linear(qkv.in_features, qkv.out_features + gate.out_features,
                      bias=True, device=qkv.weight.device, dtype=qkv.weight.dtype)
    fused.weight.copy_(torch.cat((qkv.weight, gate.weight), dim=0))
    fused.bias.zero_()
    fused.bias[qkv.out_features:].copy_(gate.bias)
    attn.to_qkv_gate = fused.requires_grad_(False).eval()
    del attn.to_qkv
    del attn.to_gates


def _fuse_sequential_bn(model):
    """Fold adjacent Conv2d/BatchNorm2d pairs."""
    total = 0
    for parent in model.modules():
        if not isinstance(parent, nn.Sequential) or type(parent).forward is not nn.Sequential.forward:
            continue
        keys = list(parent._modules)
        for a, b in zip(keys, keys[1:]):
            conv, bn = parent._modules[a], parent._modules[b]
            if type(conv) is nn.Conv2d and type(bn) is nn.BatchNorm2d and bn.track_running_stats:
                parent._modules[a] = fuse_conv_bn_eval(conv, bn)
                parent._modules[b] = nn.Identity()
                total += 1
    return total


@torch.no_grad()
def optimize_model(model, model_type, *, mode='safe', group_size=4):
    """Apply only inference-safe, ONNX-portable transforms after strict loading."""
    if mode not in ('off', 'safe'):
        raise ValueError(f'Unsupported optimization mode: {mode}')
    if group_size < 1:
        raise ValueError('group_size must be positive')
    if mode == 'off' or getattr(model, '_inference_optimized', False):
        return model, []
    model.eval()
    report = []
    if model_type in ('bs_roformer', 'mel_band_roformer', 'bs_polarformer'):
        from models_without_stft.roformer_common import Attention, BandSplit, MaskEstimator
        count = 0
        for module in list(model.modules()):
            if type(module) is Attention:
                _fuse_qkv_gate(module)
                module.attend.use_matmul = module.pope_embed is None
                count += 1
        if count:
            report.append(f'QKV+Gate attention: {count}')
        if type(model.band_split) is BandSplit:
            model.band_split = PackedBandSplit(model.band_split, group_size).eval()
            report.append(f'BandSplit packed groups: {len(model.band_split.groups)}')
        for i, estimator in enumerate(model.mask_estimators):
            if type(estimator) is MaskEstimator:
                model.mask_estimators[i] = PackedMaskEstimator(estimator, group_size).eval()
        report.append(f'MaskEstimator heads: {len(model.mask_estimators)}')
    if model_type == 'segm_models':
        report.append(f'Conv-BN pairs fused: {_fuse_sequential_bn(model)}')
    count = 0
    for parent in list(model.modules()):
        for name, child in list(parent.named_children()):
            if type(child) is nn.Dropout:
                setattr(parent, name, nn.Identity())
                count += 1
    if count:
        report.append(f'Dropout identities: {count}')
    model._inference_optimized = True
    return model, report


@torch.no_grad()
def prepare_positions(model, frames, *, static_time, precompute=True):
    """Prepare shared frequency-axis and optional time-axis position tables."""
    from models_without_stft.roformer_common import prepare_rotary
    from models_without_stft.bs_polarformer_no_stft import prepare_pope

    bands = len(model.band_split.dim_inputs)
    lengths = (frames if static_time and precompute else None, bands if precompute else None)
    for transformer, length in zip(model.layers[0], lengths):
        attention = transformer.layers[0][0]
        if attention.pope_embed is not None:
            if length is not None:
                prepare_pope(attention.pope_embed, length)
        else:
            prepare_rotary(attention.rotary_embed, length)

