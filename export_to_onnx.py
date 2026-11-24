"""Export PyTorch checkpoints to spectral-core ONNX graphs."""
import argparse
from pathlib import Path
import torch
from utils import MODEL_TYPES, load_config, load_core, chunk_size
from models.preprocess import make_processor


def export_model_to_onnx(config, model, model_type, output_path, opset_version=None, device='cpu'):
    import onnx
    import onnxruntime as ort
    import numpy as np
    polar = model_type == 'bs_polarformer'
    opset = opset_version or (18 if polar else 17)
    if polar and opset < 18:
        raise ValueError('PolarFormer requires opset >= 18')
    output_path = str(output_path)
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    model = model.to(device).eval()
    processor = make_processor(config, model_type)
    # PolarFormer has dynamic time; trace a smaller chunk to reduce export memory.
    size = min(chunk_size(config), 65536) if polar else chunk_size(config)
    x = processor.stft(torch.randn(2 if polar else 1, 2, size, device=device))
    names = ('stft_features', 'mask') if polar else ('input', 'output')
    with torch.no_grad():
        expected = model(x).detach().cpu().numpy()
        if polar:
            torch.onnx.export(model, (x,), output_path, input_names=[names[0]], output_names=[names[1]],
                              opset_version=opset, dynamo=True, external_data=False,
                              dynamic_shapes={'x': {0: torch.export.Dim('batch', min=1, max=16),
                                                    1: torch.export.Dim('time_frames', min=1, max=4096)}})
        else:
            torch.onnx.export(model, (x,), output_path, input_names=[names[0]], output_names=[names[1]],
                              opset_version=opset, dynamo=False,
                              dynamic_axes={names[0]: {0:'batch'}, names[1]: {0:'batch'}})
    onnx.checker.check_model(output_path)
    session = ort.InferenceSession(output_path, providers=['CPUExecutionProvider'])
    actual = session.run(None, {names[0]: x.cpu().numpy()})[0]
    np.testing.assert_allclose(actual, expected, rtol=1e-2, atol=1e-3)
    if polar:
        # A different batch/time shape catches accidentally specialized dynamic exports.
        probe = processor.stft(torch.randn(1, 2, size + 512, device=device))
        with torch.no_grad():
            reference = model(probe).cpu().numpy()
        actual = session.run(None, {names[0]: probe.cpu().numpy()})[0]
        np.testing.assert_allclose(actual, reference, rtol=1e-2, atol=1e-3)
    print(f'Exported and compared against PyTorch: {output_path}')


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--model_type', choices=MODEL_TYPES, required=True)
    p.add_argument('--config_path', required=True)
    p.add_argument('--checkpoint_path', required=True)
    p.add_argument('--output_path', required=True)
    p.add_argument('--opset_version', type=int)
    p.add_argument('--force_cpu', action='store_true')
    args = p.parse_args()
    config = load_config(args.model_type, args.config_path)
    device = 'cpu' if args.force_cpu or not torch.cuda.is_available() else 'cuda'
    model = load_core(args.model_type, config, args.checkpoint_path)
    export_model_to_onnx(config, model, args.model_type, args.output_path, args.opset_version, device)


if __name__ == '__main__':
    main()
