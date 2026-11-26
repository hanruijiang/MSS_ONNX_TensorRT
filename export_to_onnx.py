"""Export PyTorch checkpoints to spectral-core ONNX graphs."""
import argparse
from pathlib import Path
import torch
from utils import MODEL_TYPES, load_config, load_core, chunk_size
from models.preprocess import make_processor
from models_without_stft.optimization import optimize_model, prepare_positions


def export_model_to_onnx(config, model, model_type, output_path, opset_version=None, device='cpu', *,
                         optimization='off', group_size=4, shape_mode='dynamic',
                         export_chunk_size=None, batch_size=None):
    import onnx
    import onnxruntime as ort
    import numpy as np
    polar = model_type == 'bs_polarformer'
    opset = opset_version or (18 if polar else 17)
    if polar and opset < 18:
        raise ValueError('PolarFormer requires opset >= 18')
    output_path = str(output_path)
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    model = model.to(device=device, dtype=torch.float32).eval()
    if shape_mode not in ('static', 'dynamic'):
        raise ValueError('shape_mode must be static or dynamic')
    processor = make_processor(config, model_type)
    dynamic = shape_mode == 'dynamic'
    roformer = model_type in ('bs_roformer', 'mel_band_roformer', 'bs_polarformer')
    size = export_chunk_size if export_chunk_size is not None else chunk_size(config)
    if polar and dynamic and export_chunk_size is None:
        size = min(size, 65536)
    batch = batch_size if batch_size is not None else (2 if polar and dynamic else 1)
    if size <= 0 or batch <= 0:
        raise ValueError('chunk and batch size must be positive')
    if polar and shape_mode == 'dynamic' and not 2 <= batch <= 16:
        raise ValueError('Dynamic PolarFormer export requires example batch 2..16')
    x = processor.stft(torch.randn(batch, 2, size, device=device))
    names = ('stft_features', 'mask') if polar else ('input', 'output')
    with torch.no_grad():
        expected = model(x).detach().cpu().numpy()
        model, report = optimize_model(model, model_type, mode=optimization, group_size=group_size)
        if roformer and (dynamic or optimization == 'safe'):
            frames = x.shape[2] if model_type == 'mel_band_roformer' else x.shape[1]
            prepare_positions(model, frames, static_time=not dynamic, precompute=optimization == 'safe')
        print('Structural optimization:', report)
        if polar:
            torch.onnx.export(model, (x,), output_path, input_names=[names[0]], output_names=[names[1]],
                              opset_version=opset, dynamo=True, external_data=False,
                              dynamic_shapes={'x': {0: torch.export.Dim('batch', min=1, max=16),
                                                    1: torch.export.Dim('time_frames', min=1, max=4096)}} if shape_mode == 'dynamic' else None)
        else:
            axes = {names[0]: {0: 'batch'}, names[1]: {0: 'batch'}}
            if roformer:
                axes[names[0]][2 if model_type == 'mel_band_roformer' else 1] = 'time_frames'
                axes[names[1]][3] = 'time_frames'
            torch.onnx.export(model, (x,), output_path, input_names=[names[0]], output_names=[names[1]],
                              opset_version=opset, dynamo=False,
                              dynamic_axes=axes if dynamic else None)
    onnx.checker.check_model(output_path)
    session = ort.InferenceSession(output_path, providers=['CPUExecutionProvider'])
    actual = session.run(None, {names[0]: x.cpu().numpy()})[0]
    np.testing.assert_allclose(actual, expected, rtol=1e-2, atol=1e-3)
    if dynamic:
        probe_size = size + int(config.model.stft_hop_length) if roformer else size
        probe = processor.stft(torch.randn(2 if batch == 1 else 1, 2, probe_size, device=device))
        with torch.no_grad():
            reference = model(probe).cpu().numpy()
        np.testing.assert_allclose(session.run(None, {names[0]: probe.cpu().numpy()})[0],
                                   reference, rtol=1e-2, atol=1e-3)
    print(f'Exported and compared against PyTorch: {output_path}')


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--model_type', choices=MODEL_TYPES, required=True)
    p.add_argument('--config_path', required=True)
    p.add_argument('--checkpoint_path', required=True)
    p.add_argument('--output_path', required=True)
    p.add_argument('--opset_version', type=int)
    p.add_argument('--force_cpu', action='store_true')
    p.add_argument('--optimization', choices=['off', 'safe'], default='off')
    p.add_argument('--group_size', type=int, default=4)
    p.add_argument('--shape_mode', choices=['static', 'dynamic'], default='dynamic')
    p.add_argument('--chunk_size', type=int)
    p.add_argument('--batch_size', type=int)
    args = p.parse_args()
    config = load_config(args.model_type, args.config_path)
    device = 'cpu' if args.force_cpu or not torch.cuda.is_available() else 'cuda'
    model = load_core(args.model_type, config, args.checkpoint_path)
    export_model_to_onnx(config, model, args.model_type, args.output_path, args.opset_version, device,
                         optimization=args.optimization, group_size=args.group_size,
                         shape_mode=args.shape_mode, export_chunk_size=args.chunk_size,
                         batch_size=args.batch_size)


if __name__ == '__main__':
    main()
