"""Export PyTorch checkpoints to spectral-core ONNX graphs."""
import argparse
from pathlib import Path
from tempfile import TemporaryDirectory
import torch
from onnx_opset import DEFAULT_ONNX_OPSET_VERSION, require_onnx_support, require_opset
from utils import MODEL_TYPES, load_config, load_core, chunk_size, warn_if_not_close
from models.preprocess import make_processor
from models_without_stft.optimization import optimize_model, prepare_positions


def export_model_to_onnx(config, model, model_type, output_path, opset_version=None, device='cpu', *,
                         optimization='off', group_size=4, shape_mode='dynamic',
                         export_chunk_size=None, batch_size=None, quantization='none'):
    if opset_version is None:
        opset_version = DEFAULT_ONNX_OPSET_VERSION
    require_onnx_support(opset_version)
    options = dict(optimization=optimization, group_size=group_size, shape_mode=shape_mode,
                   export_chunk_size=export_chunk_size, batch_size=batch_size)
    if quantization == 'none':
        return _export_fp32(config, model, model_type, output_path, opset_version, device, **options)

    from convert_onnx_precision import QUANTIZATION_BITS, convert_precision
    from optimize_onnx import optimize_graph

    if quantization not in QUANTIZATION_BITS:
        raise ValueError(f'Unknown quantization: {quantization}')
    if optimization != 'safe':
        raise ValueError('Quantized export requires --optimization safe')
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(dir=output_path.parent) as tmp:
        fp32 = Path(tmp) / 'fp32.onnx'
        optimized = Path(tmp) / 'optimized.onnx'
        _export_fp32(config, model, model_type, fp32, opset_version, device, **options)
        optimize_graph(fp32, optimized, opset_version=opset_version)
        convert_precision(optimized, output_path, quantization=quantization, opset_version=opset_version)


def _export_fp32(config, model, model_type, output_path, opset_version, device, *,
                 optimization, group_size, shape_mode, export_chunk_size, batch_size):
    import onnx
    import onnxruntime as ort
    polar = model_type == 'bs_polarformer'
    output_path = str(output_path)
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    model = model.to(device=device, dtype=torch.float32).eval()
    if shape_mode not in ('static', 'dynamic'):
        raise ValueError('shape_mode must be static or dynamic')
    processor = make_processor(config, model_type)
    dynamic = shape_mode == 'dynamic'
    roformer = model_type in ('bs_roformer', 'mel_band_roformer', 'bs_polarformer')
    size = export_chunk_size if export_chunk_size is not None else chunk_size(config)
    # Short examples bound the attention buffers used by export and validation.
    if roformer and dynamic and export_chunk_size is None:
        size = min(size, 65536)
    batch = batch_size if batch_size is not None else (2 if polar and dynamic else 1)
    if size <= 0 or batch <= 0:
        raise ValueError('chunk and batch size must be positive')
    if polar and shape_mode == 'dynamic' and not 2 <= batch <= 16:
        raise ValueError('Dynamic PolarFormer export requires example batch 2..16')
    x = processor.stft(torch.randn(batch, 2, size, device=device))
    names = ('stft_features', 'mask') if polar else ('input', 'output')
    time_axis = 2 if model_type == 'mel_band_roformer' else 1
    export_input = x
    dynamic_shapes = None
    if dynamic:
        # torch.export specializes example dimensions of size one.
        if batch == 1:
            export_input = x.expand(2, *x.shape[1:]).contiguous()
        # A named dynamic hint lets Dynamo infer bounds imposed by CUDA kernels.
        axes = {0: torch.export.Dim('batch', min=1, max=16) if polar else 'batch'}
        if roformer:
            axes[time_axis] = torch.export.Dim('time_frames', min=1, max=4096 if polar else None)
        dynamic_shapes = (axes,)
    with torch.no_grad():
        expected = model(x).detach().cpu().numpy()
        model, report = optimize_model(model, model_type, mode=optimization, group_size=group_size)
        if roformer and (dynamic or optimization == 'safe'):
            prepare_positions(model, x.shape[time_axis], static_time=not dynamic, precompute=optimization == 'safe')
        print('Structural optimization:', report)
        program = torch.onnx.export(model, (export_input,), input_names=[names[0]], output_names=[names[1]],
                                    opset_version=opset_version, dynamo=True,
                                    dynamic_shapes=dynamic_shapes)
        require_opset(program.model.opset_imports.get(''), opset_version, 'Exporter output')
        program.save(output_path, external_data=False)
        del program
    onnx.checker.check_model(output_path)
    session = ort.InferenceSession(output_path, providers=['CPUExecutionProvider'])
    actual = session.run(None, {names[0]: x.cpu().numpy()})[0]
    passed = warn_if_not_close(actual, expected, label='ONNX export')
    if dynamic:
        probe_size = size + int(config.model.stft_hop_length) if roformer else size
        probe = processor.stft(torch.randn(2 if batch == 1 else 1, 2, probe_size, device=device))
        with torch.no_grad():
            reference = model(probe).cpu().numpy()
        probe_passed = warn_if_not_close(
            session.run(None, {names[0]: probe.cpu().numpy()})[0], reference,
            label='ONNX dynamic shape')
        passed = passed and probe_passed
    status = 'passed' if passed else 'warnings'
    print(f'Exported: {output_path} (numerical checks: {status})')


def main():
    from convert_onnx_precision import QUANTIZATION_BITS

    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--model_type', choices=MODEL_TYPES, required=True)
    p.add_argument('--config_path', required=True)
    p.add_argument('--checkpoint_path', required=True)
    p.add_argument('--output_path', required=True)
    p.add_argument('--opset_version', type=int, default=DEFAULT_ONNX_OPSET_VERSION,
                   help='Target ONNX opset for all stages (default: %(default)s)')
    p.add_argument('--force_cpu', action='store_true')
    p.add_argument('--optimization', choices=['off', 'safe'], default='off')
    p.add_argument('--group_size', type=int, default=4)
    p.add_argument('--shape_mode', choices=['static', 'dynamic'], default='dynamic')
    p.add_argument('--chunk_size', type=int)
    p.add_argument('--batch_size', type=int)
    p.add_argument('--quantization', choices=('none', *QUANTIZATION_BITS), default='none',
                   help='Apply INT8/INT4 feed-forward weight quantization (requires --optimization safe)')
    args = p.parse_args()
    if args.quantization != 'none' and args.optimization != 'safe':
        p.error('quantized export requires --optimization safe')
    config = load_config(args.model_type, args.config_path)
    device = 'cpu' if args.force_cpu or not torch.cuda.is_available() else 'cuda'
    model = load_core(args.model_type, config, args.checkpoint_path)
    export_model_to_onnx(config, model, args.model_type, args.output_path, args.opset_version, device,
                         optimization=args.optimization, group_size=args.group_size,
                         shape_mode=args.shape_mode, export_chunk_size=args.chunk_size,
                         batch_size=args.batch_size, quantization=args.quantization)


if __name__ == '__main__':
    main()
