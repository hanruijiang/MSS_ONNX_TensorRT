"""Compare spectral-core outputs and measure synchronized Native WebGPU latency."""
import argparse
import time
import json
import numpy as np
import torch
from utils import ONNXBackend, MODEL_TYPES, load_config, chunk_size
from models.preprocess import make_processor


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--model_type', choices=MODEL_TYPES, required=True)
    p.add_argument('--config_path', required=True)
    p.add_argument('--reference', required=True, help='Original FP32 ONNX (CPU)')
    p.add_argument('--model', required=True, help='Candidate ONNX')
    p.add_argument('--provider', choices=['cpu', 'webgpu'], default='webgpu')
    p.add_argument('--chunk_size', type=int)
    p.add_argument('--batch_size', type=int, default=1)
    p.add_argument('--warmup', type=int, default=3)
    p.add_argument('--runs', type=int, default=10)
    p.add_argument('--layout', choices=['NCHW', 'NHWC'], default='NCHW')
    p.add_argument('--buffer_cache', choices=['disabled', 'lazyRelease', 'simple', 'bucket'], default='bucket')
    p.add_argument('--graph_capture', action='store_true')
    p.add_argument('--io_binding', action='store_true')
    p.add_argument('--strict_webgpu', action='store_true')
    p.add_argument('--profile', action='store_true', help='Save ORT kernel profile')
    p.add_argument('--rtol', type=float, default=1e-2)
    p.add_argument('--atol', type=float, default=1e-3)
    args = p.parse_args()
    if args.warmup < 0 or args.runs < 1 or args.batch_size < 1:
        p.error('Invalid warmup, runs, or batch size')
    config = load_config(args.model_type, args.config_path)
    size = args.chunk_size if args.chunk_size is not None else chunk_size(config)
    torch.manual_seed(0)
    processor = make_processor(config, args.model_type)
    x = processor.stft(torch.randn(args.batch_size, 2, size))
    ref = ONNXBackend(args.reference, 'cpu', provider='cpu')
    candidate = ONNXBackend(args.model, 'cpu', provider=args.provider,
                            graph_capture=args.graph_capture, io_binding=args.io_binding,
                            strict_webgpu=args.strict_webgpu, layout=args.layout,
                            buffer_cache=args.buffer_cache, profile=args.profile)
    expected, actual = ref(x), candidate(x)
    np.testing.assert_allclose(actual.numpy(), expected.numpy(), rtol=args.rtol, atol=args.atol)
    np.testing.assert_allclose(processor.istft(actual).numpy(), processor.istft(expected).numpy(),
                               rtol=args.rtol, atol=args.atol)
    for _ in range(args.warmup):
        candidate(x)
    times = []
    for _ in range(args.runs):
        start = time.perf_counter()
        candidate(x)
        times.append((time.perf_counter() - start) * 1000)
    print(json.dumps({'model': args.model, 'providers': candidate.session.get_providers(),
                      'chunk_size': size, 'batch': args.batch_size, 'input_shape': list(x.shape),
                      'layout': args.layout, 'capture': args.graph_capture, 'io_binding': args.io_binding or args.graph_capture,
                      'p50_ms': float(np.percentile(times, 50)), 'p95_ms': float(np.percentile(times, 95)),
                      'runs': args.runs,
                      'profile': candidate.session.end_profiling() if args.profile else None}, indent=2))


if __name__ == '__main__':
    main()
