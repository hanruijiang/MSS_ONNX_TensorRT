"""Vocal separation using ONNX Runtime spectral cores."""
import argparse
from pathlib import Path
import librosa
import numpy as np
import soundfile as sf
import torch
from utils import MODEL_TYPES, load_config, ONNXBackend, demix, normalize_audio, denormalize_audio, chunk_size, inference_batch_size
from models.preprocess import spectral_input_shape


def parse_args(dict_args=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model_type', choices=MODEL_TYPES, default='mdx23c')
    parser.add_argument('--config_path', required=dict_args is None)
    parser.add_argument('--input_folder', required=dict_args is None)
    parser.add_argument('--store_dir', required=dict_args is None)
    parser.add_argument('--onnx_model_path', required=dict_args is None)
    parser.add_argument('--force_cpu', action='store_true')
    parser.add_argument('--device_id', type=int, default=0, help='CUDA device index')
    parser.add_argument('--provider', choices=['auto', 'cpu', 'cuda', 'webgpu'], default='auto')
    parser.add_argument('--webgpu_graph_capture', action='store_true')
    parser.add_argument('--webgpu_io_binding', action='store_true')
    parser.add_argument('--webgpu_strict', action='store_true')
    parser.add_argument('--webgpu_layout', choices=['NCHW', 'NHWC'], default='NCHW')
    parser.add_argument('--webgpu_buffer_cache', choices=['disabled', 'lazyRelease', 'simple', 'bucket'], default='bucket')
    parser.add_argument('--chunk_size', type=int)
    parser.add_argument('--batch_size', type=int)
    parser.add_argument('--extract_instrumental', action='store_true')
    parser.add_argument('--use_tta', action='store_true')
    parser.add_argument('--disable_detailed_pbar', action='store_true')
    parser.add_argument('--flac_file', action='store_true')
    parser.add_argument('--pcm_type', choices=['PCM_16', 'PCM_24'], default='PCM_24')
    args = parser.parse_args(None if dict_args is None else [])
    if dict_args is not None:
        unknown = set(dict_args) - vars(args).keys()
        if unknown:
            raise ValueError(f'Unknown inference arguments: {", ".join(sorted(unknown))}')
        vars(args).update(dict_args)
        for name in ('config_path', 'input_folder', 'store_dir', 'onnx_model_path'):
            if not getattr(args, name):
                raise ValueError(f'--{name} is required')
    return args


def proc_folder(dict_args=None):
    args = parse_args(dict_args)
    config = load_config(args.model_type, args.config_path)
    if args.chunk_size is not None:
        config.inference.chunk_size = args.chunk_size
    if args.batch_size is not None:
        config.inference.batch_size = args.batch_size
    if args.force_cpu and args.provider not in ('auto', 'cpu'):
        raise ValueError('--force_cpu conflicts with explicit GPU provider')
    if args.provider == 'cuda' and not torch.cuda.is_available():
        raise RuntimeError('CUDA provider requires CUDA-enabled PyTorch for preprocessing')
    device = torch.device('cpu' if args.force_cpu or args.provider in ('cpu', 'webgpu')
                          or not torch.cuda.is_available() else f'cuda:{args.device_id}')
    model = ONNXBackend(args.onnx_model_path, device, provider=args.provider,
                        graph_capture=args.webgpu_graph_capture, io_binding=args.webgpu_io_binding,
                        strict_webgpu=args.webgpu_strict, layout=args.webgpu_layout,
                        buffer_cache=args.webgpu_buffer_cache)
    batch = inference_batch_size(config, model)
    model.validate_shape(spectral_input_shape(config, args.model_type, chunk_size(config), batch))
    output = Path(args.store_dir)
    output.mkdir(parents=True, exist_ok=True)
    rate = int(config.audio.get('sample_rate', 44100))
    paths = sorted(p for p in Path(args.input_folder).iterdir() if p.is_file()
                   and p.suffix.lower() in {'.wav','.flac','.mp3','.ogg','.m4a','.aac','.aiff'})
    if not paths:
        raise ValueError('No supported audio files found')
    if len({p.stem for p in paths}) != len(paths):
        raise ValueError('Input files must have unique stems to avoid overwriting outputs')
    written = []
    for path in paths:
        mix, _ = librosa.load(str(path), sr=rate, mono=False)
        if mix.ndim == 1:
            mix = np.stack([mix, mix])
        if mix.shape[0] != 2:
            raise ValueError(f'Expected mono or stereo audio: {path}')
        original = mix.copy()
        norm = None
        if config.inference.get('normalize', False):
            mix, norm = normalize_audio(mix)
        def separate(audio):
            return demix(config, model, audio, device, args.model_type,
                         pbar=not args.disable_detailed_pbar)
        stems = separate(mix)
        if args.use_tta:
            swapped, inverted = separate(mix[::-1].copy()), separate(-mix)
            stems = {k: (v+swapped[k][::-1]-inverted[k])/3 for k,v in stems.items()}
        if norm:
            stems = {k: denormalize_audio(v, norm) for k,v in stems.items()}
        if args.extract_instrumental:
            stems['instrumental'] = original-stems['vocals']
        for stem, audio in stems.items():
            ext = 'flac' if args.flac_file else 'wav'
            target = output / f'{path.stem}_{stem}.{ext}'
            sf.write(target, audio.T, rate, subtype=args.pcm_type if args.flac_file else 'FLOAT')
            written.append(str(target))
        print(f'Processed {path.name}')
    return written


if __name__ == '__main__':
    proc_folder()
