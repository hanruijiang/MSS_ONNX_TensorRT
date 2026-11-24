# Original MSST author: Roman Solovyev (ZFTurbo).
"""Configuration, checkpoint loading and shared chunked inference."""
import numpy as np
import torch
import yaml
from ml_collections import ConfigDict
from models.preprocess import make_processor

MODEL_TYPES = ('mdx23c', 'segm_models', 'bs_roformer', 'mel_band_roformer', 'bs_polarformer')


def load_config(model_type, config_path):
    if model_type not in MODEL_TYPES:
        raise ValueError(f'Unknown model type: {model_type}')
    with open(config_path, encoding='utf-8') as f:
        # FullLoader supports the legacy !!python/tuple presets.
        config = ConfigDict(yaml.full_load(f))
    if model_type == 'bs_polarformer' and not config.model.get('use_pope', False):
        raise ValueError('BS PolarFormer requires use_pope: true')
    return config


def chunk_size(config):
    size = int(config.inference.get('chunk_size', config.audio.get('chunk_size', 0)))
    if size <= 0:
        raise ValueError('chunk_size must be positive')
    return size


def prefer_target_instrument(config):
    target = config.training.get('target_instrument')
    return [target] if target else list(config.training.instruments)


def load_core(model_type, config, checkpoint):
    """Load only the neural network; no checkpoint is needed for ONNX/TRT."""
    params = dict(config.model)
    if model_type == 'mdx23c':
        from models_without_stft.mdx23c_tfc_tdf_v3_no_stft import TFC_TDF_net
        model = TFC_TDF_net(config)
    elif model_type == 'segm_models':
        from models_without_stft.segm_models_no_stft import Segm_Models_Net
        model = Segm_Models_Net(config)
    elif model_type == 'bs_polarformer':
        from models.bs_roformer.bs_roformer import BSRoformer
        from models_without_stft.bs_polarformer_no_stft import BSRoformerCore, replace_pope_with_standard_attn
        params['flash_attn'] = False
        model = BSRoformer(**params)
    elif model_type == 'bs_roformer':
        from models_without_stft.bs_roformer_no_stft import BSRoformer
        params['flash_attn'] = False
        model = BSRoformer(**params)
    elif model_type == 'mel_band_roformer':
        from models_without_stft.mel_band_roformer_no_stft import MelBandRoformer
        params['flash_attn'] = False
        model = MelBandRoformer(**params)
    else:
        raise ValueError(f'Unknown model type: {model_type}')
    state = torch.load(checkpoint, map_location='cpu', weights_only=True)
    if 'state_dict' in state:
        state = state['state_dict']
    state = {k.removeprefix('module.'): v for k, v in state.items()}
    model.load_state_dict(state, strict=True)
    if model_type == 'bs_polarformer':
        model = BSRoformerCore(replace_pope_with_standard_attn(model))
    return model.eval()


class ONNXBackend:
    def __init__(self, path, device):
        import onnxruntime as ort
        device = torch.device(device)
        providers = ['CPUExecutionProvider']
        if device.type == 'cuda':
            if 'CUDAExecutionProvider' not in ort.get_available_providers():
                raise RuntimeError('CUDA provider unavailable; install onnxruntime-gpu or use --force_cpu')
            providers.insert(0, ('CUDAExecutionProvider', {'device_id': device.index or 0}))
        self.session = ort.InferenceSession(str(path), providers=providers)
        if len(self.session.get_inputs()) != 1 or len(self.session.get_outputs()) != 1:
            raise ValueError('Expected one spectral input and one output')
        self.input = self.session.get_inputs()[0]
        self.fixed_batch = self.input.shape[0] if isinstance(self.input.shape[0], int) else None

    def __call__(self, x):
        dtype = {'tensor(float)': np.float32, 'tensor(float16)': np.float16}[self.input.type]
        values = x.detach().cpu().numpy().astype(dtype, copy=False)
        for actual, expected in zip(values.shape, self.input.shape):
            if isinstance(expected, int) and actual != expected:
                raise ValueError(f'ONNX input shape {values.shape} does not match {self.input.shape}; check chunk size')
        out = self.session.run(None, {self.input.name: values})[0]
        return torch.from_numpy(out).to(x.device)


@torch.inference_mode()
def demix(config, model, mix, device, model_type, pbar=False, **kwargs):
    """Overlap-add fixed-size chunks; returns named stereo waveforms."""
    size = chunk_size(config)
    overlap = int(config.inference.num_overlap)
    batch = int(config.inference.batch_size)
    if not 1 <= overlap <= size or batch < 1:
        raise ValueError('Invalid num_overlap or batch_size')
    fixed_batch = getattr(model, 'fixed_batch', None)
    max_batch = getattr(model, 'max_batch', None)
    batch = fixed_batch or min(batch, max_batch or batch)
    mix = torch.as_tensor(mix, dtype=torch.float32, device='cpu')
    if mix.ndim != 2 or mix.shape[0] != 2 or mix.shape[1] == 0:
        raise ValueError('Expected nonempty stereo audio (2, samples)')
    length = mix.shape[-1]
    names = prefer_target_instrument(config)
    step = size // overlap
    fade = min(size // 10, size-step)
    window = torch.ones(size)
    if fade:
        window[:fade] = torch.linspace(0., 1., fade)
        window[-fade:] = torch.linspace(1., 0., fade)
    # Reflect context at both ends, when feasible; otherwise zero padding.
    border = size-step
    if border and length > border:
        padded = torch.nn.functional.pad(mix.unsqueeze(0), (border, border), mode='reflect')[0]
    elif border:
        padded = torch.nn.functional.pad(mix, (border, border))
    else:
        padded = mix
    total = padded.shape[-1]
    result = torch.zeros(len(names), 2, total)
    count = torch.zeros(total)
    processor = make_processor(config, model_type)
    starts = list(range(0, total, step))
    indices = range(0, len(starts), batch)
    if pbar:
        from tqdm.auto import tqdm
        indices = tqdm(indices, desc='Audio chunks')
    for i in indices:
        positions = starts[i:i+batch]
        chunks = []
        for start in positions:
            part = padded[:, start:start+size]
            chunks.append(torch.nn.functional.pad(part, (0, size-part.shape[-1])))
        if fixed_batch:
            chunks += [torch.zeros_like(chunks[0])] * (fixed_batch-len(chunks))
        x = processor.stft(torch.stack(chunks).to(device))
        y = processor.istft(model(x)).float().cpu()
        if y.shape[1:] != (len(names), 2, size):
            raise ValueError(f'Unexpected reconstructed shape: {tuple(y.shape)}')
        for j, start in enumerate(positions):
            end = min(start+size, total)
            n = end-start
            w = window.clone()
            if fade and start == 0:
                w[:fade] = 1
            if fade and end == total:
                w[-fade:] = 1
            result[..., start:end] += y[j, ..., :n] * w[:n]
            count[start:end] += w[:n]
    output = (result / count.clamp_min(1e-8))[..., border:border+length].numpy()
    return dict(zip(names, output))


def normalize_audio(audio):
    mono = audio.mean(0)
    mean, std = float(mono.mean()), max(float(mono.std()), 1e-8)
    return (audio-mean)/std, {'mean': mean, 'std': std}


def denormalize_audio(audio, params):
    return audio*params['std']+params['mean']
