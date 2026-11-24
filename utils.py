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
        return ConfigDict(yaml.safe_load(f))


def chunk_size(config):
    size = int(config.inference.chunk_size)
    if size <= 0:
        raise ValueError('chunk_size must be positive')
    return size


def inference_batch_size(config, model):
    batch = int(config.inference.batch_size)
    if batch < 1:
        raise ValueError('batch_size must be positive')
    return getattr(model, 'fixed_batch', None) or batch


def load_core(model_type, config, checkpoint):
    """Load the neural network checkpoint for ONNX export."""
    params = dict(config.model)
    if model_type == 'mdx23c':
        from models_without_stft.mdx23c_tfc_tdf_v3_no_stft import TFC_TDF_net
        model = TFC_TDF_net(config, len(config.inference.stems))
    elif model_type == 'segm_models':
        from models_without_stft.segm_models_no_stft import Segm_Models_Net
        model = Segm_Models_Net(config, len(config.inference.stems))
    elif model_type in ('bs_roformer', 'bs_polarformer', 'mel_band_roformer'):
        # Waveform-only options belong to the external processor.
        for key in ('stft_hop_length', 'stft_win_length', 'stft_normalized',
                    'zero_dc'):
            params.pop(key, None)
        if model_type == 'mel_band_roformer':
            from models_without_stft.mel_band_roformer_no_stft import MelBandRoformer
            model = MelBandRoformer(**params)
        else:
            from models_without_stft.bs_roformer_no_stft import BSRoformer
            params['use_pope'] = model_type == 'bs_polarformer'
            model = BSRoformer(**params)
    else:
        raise ValueError(f'Unknown model type: {model_type}')
    state = torch.load(checkpoint, map_location='cpu', weights_only=True)
    for key in ('state', 'state_dict', 'model_state_dict'):
        if isinstance(state, dict) and key in state:
            state = state[key]
    if not isinstance(state, dict):
        raise ValueError('Checkpoint must contain a model state dict')
    cleaned = {}
    for key, value in state.items():
        key = key.replace('_orig_mod.', '')
        while key.startswith('module.'):
            key = key.removeprefix('module.')
        if key in cleaned:
            raise ValueError(f'Duplicate checkpoint key after prefix removal: {key}')
        cleaned[key] = value
    model.load_state_dict(cleaned, strict=True)
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
        if device.type == 'cuda':
            if 'CUDAExecutionProvider' not in self.session.get_providers():
                raise RuntimeError('ONNX Runtime failed to initialize CUDAExecutionProvider')
            self.session.disable_fallback()
        if len(self.session.get_inputs()) != 1 or len(self.session.get_outputs()) != 1:
            raise ValueError('Expected one spectral input and one output')
        self.input = self.session.get_inputs()[0]
        self.output = self.session.get_outputs()[0]
        self.dtypes = {'tensor(float)': np.float32, 'tensor(float16)': np.float16}
        if self.input.type not in self.dtypes or self.output.type not in self.dtypes:
            raise ValueError('Expected float32 or float16 ONNX input and output')
        if not self.input.shape:
            raise ValueError('Expected a batched spectral input')
        self.fixed_batch = self.input.shape[0] if isinstance(self.input.shape[0], int) else None
        if self.fixed_batch is not None and self.fixed_batch < 1:
            raise ValueError('ONNX fixed batch must be positive')

    def validate_shape(self, shape):
        if len(shape) != len(self.input.shape) or any(
            isinstance(expected, int) and actual != expected
            for actual, expected in zip(shape, self.input.shape)
        ):
            raise ValueError(f'ONNX input shape {shape} does not match {self.input.shape}; check chunk size')

    def __call__(self, x):
        self.validate_shape(tuple(x.shape))
        values = x.detach().cpu().numpy().astype(self.dtypes[self.input.type], copy=False)
        out = self.session.run(None, {self.input.name: values})[0]
        return torch.from_numpy(out).to(x.device)


@torch.inference_mode()
def demix(config, model, mix, device, model_type, pbar=False):
    """Overlap-add fixed-size chunks; returns named stereo waveforms."""
    size = chunk_size(config)
    overlap = int(config.inference.num_overlap)
    if not 1 <= overlap <= size:
        raise ValueError('Invalid num_overlap')
    batch = inference_batch_size(config, model)
    fixed_batch = getattr(model, 'fixed_batch', None)
    mix = torch.as_tensor(mix, dtype=torch.float32, device='cpu')
    if mix.ndim != 2 or mix.shape[0] != 2 or mix.shape[1] == 0:
        raise ValueError('Expected nonempty stereo audio (2, samples)')
    length = mix.shape[-1]
    names = list(config.inference.stems)
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
            pad = size - part.shape[-1]
            if pad and part.shape[-1] > size // 2:
                part = torch.nn.functional.pad(part, (0, pad), mode='reflect')
            elif pad:
                part = torch.nn.functional.pad(part, (0, pad))
            chunks.append(part)
        if fixed_batch is not None and len(chunks) < fixed_batch:
            chunks += [torch.zeros_like(chunks[0])] * (fixed_batch-len(chunks))
        x = processor.stft(torch.stack(chunks).to(device))
        y = processor.istft(model(x)).float().cpu()
        if y.shape != (len(chunks), len(names), 2, size):
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
