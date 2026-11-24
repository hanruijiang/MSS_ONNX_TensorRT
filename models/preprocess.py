"""Spectral interfaces for the five retained pretrained presets."""
import torch
from einops import rearrange
from librosa import filters


class STFT:
    """MDX23C / Segm: channel-major real/imaginary planes."""
    def __init__(self, config):
        self.n_fft = config.n_fft
        self.hop = config.hop_length
        self.dim_f = config.dim_f

    def stft(self, audio):
        b, c, self.length = audio.shape
        self.window = torch.hann_window(self.n_fft, device=audio.device)
        z = torch.stft(audio.reshape(b*c, -1), self.n_fft, self.hop,
                       window=self.window, return_complex=True)
        return rearrange(torch.view_as_real(z), '(b c) f t r -> b (c r) f t',
                         b=b, c=c)[..., :self.dim_f, :].contiguous()

    def istft(self, output):
        if output.ndim == 4:
            output = output.unsqueeze(1)
        b, n, _, f, t = output.shape
        output = torch.nn.functional.pad(output.float(), (0, 0, 0, self.n_fft//2+1-f))
        z = torch.view_as_complex(rearrange(output, 'b n (c r) f t -> (b n c) f t r', r=2).contiguous())
        wave = torch.istft(z, self.n_fft, self.hop, window=self.window, length=self.length)
        return wave.reshape(b, n, 2, self.length)


class RoformerProcessor:
    """Frequency-major stereo features; mask application stays outside ONNX."""
    def __init__(self, config, model_type):
        m = config.model
        self.kwargs = dict(n_fft=m.stft_n_fft, hop_length=m.stft_hop_length,
                           win_length=m.stft_win_length, normalized=m.get('stft_normalized', False))
        self.channels = 2 if m.stereo else 1
        self.zero_dc = model_type == 'bs_polarformer'
        self.mel = model_type == 'mel_band_roformer'
        if self.mel:
            bank = torch.from_numpy(filters.mel(sr=m.sample_rate, n_fft=m.stft_n_fft, n_mels=m.num_bands))
            bank[0, 0] = bank[-1, -1] = 1.
            covered = bank > 0
            self.indices = covered.nonzero(as_tuple=True)[1].repeat_interleave(self.channels)*self.channels
            self.indices += torch.arange(self.channels).repeat(self.indices.numel()//self.channels)
            self.counts = covered.sum(0).repeat_interleave(self.channels).clamp_min(1)

    def stft(self, audio):
        b, c, self.length = audio.shape
        if c != self.channels:
            raise ValueError(f'Expected {self.channels} channels, got {c}')
        self.window = torch.hann_window(self.kwargs['win_length'], device=audio.device)
        z = torch.stft(audio.reshape(b*c, -1), **self.kwargs,
                       window=self.window, return_complex=True)
        self.spectrum = rearrange(z, '(b c) f t -> b (f c) t', b=b, c=c)
        x = torch.view_as_real(self.spectrum)
        if self.mel:
            return x.index_select(1, self.indices.to(x.device)).contiguous()
        return rearrange(x, 'b f t r -> b t (f r)').contiguous()

    def istft(self, output):
        masks = torch.view_as_complex(output.float().contiguous())
        if self.mel:
            b, n, _, t = masks.shape
            indices = self.indices.to(masks.device).view(1, 1, -1, 1).expand(b, n, -1, t)
            masks = masks.new_zeros(b, n, self.spectrum.shape[1], t).scatter_add_(2, indices, masks)
            masks = masks / self.counts.to(masks.device).view(1, 1, -1, 1)
        masked = self.spectrum.unsqueeze(1) * masks
        b, n = masked.shape[:2]
        masked = rearrange(masked, 'b n (f c) t -> (b n c) f t', c=self.channels)
        if self.zero_dc:
            masked[:, 0, :] = 0
        wave = torch.istft(masked, **self.kwargs, window=self.window, length=self.length)
        return wave.reshape(b, n, self.channels, self.length)


def make_processor(config, model_type):
    if model_type in ('bs_roformer', 'mel_band_roformer', 'bs_polarformer'):
        return RoformerProcessor(config, model_type)
    if model_type in ('mdx23c', 'segm_models'):
        return STFT(config.audio)
    raise ValueError(f'Unknown model type: {model_type}')
