"""Build a TensorRT 10 engine using the parsed ONNX input and config."""
import argparse
from pathlib import Path
from utils import MODEL_TYPES, load_config, chunk_size
from models.preprocess import spectral_input_shape


def export_to_tensorrt(onnx_path, model_type, config, output_path, fp16=False):
    import tensorrt as trt
    logger = trt.Logger(trt.Logger.WARNING)
    builder = trt.Builder(logger)
    network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
    parser = trt.OnnxParser(network, logger)
    if not parser.parse_from_file(str(onnx_path)):
        raise RuntimeError('\n'.join(str(parser.get_error(i)) for i in range(parser.num_errors)))
    if network.num_inputs != 1 or network.num_outputs != 1:
        raise ValueError('Expected one spectral input and one output')
    input_tensor = network.get_input(0)
    output_tensor = network.get_output(0)
    if any(t.dtype not in (trt.float32, trt.float16) for t in (input_tensor, output_tensor)):
        raise ValueError('Expected float32 or float16 TensorRT input and output')
    batch = int(config.inference.batch_size)
    if batch < 1:
        raise ValueError('batch_size must be positive')
    shape = list(spectral_input_shape(config, model_type, chunk_size(config)))
    declared = tuple(input_tensor.shape)
    if len(declared) != len(shape):
        raise ValueError(f'ONNX input rank {declared} does not match expected shape {shape}')
    if declared[0] > 0:
        shape[0] = declared[0]
    for a, b in zip(declared, shape):
        if a >= 0 and a != b:
            raise ValueError(f'ONNX input shape {declared} does not match expected shape {shape}')
    build_config = builder.create_builder_config()
    if any(d < 0 for d in declared):
        profile = builder.create_optimization_profile()
        lo, hi = shape.copy(), shape.copy()
        if declared[0] < 0:
            hi[0] = batch
        if not profile.set_shape(input_tensor.name, tuple(lo), tuple(hi), tuple(hi)):
            raise ValueError(f'Invalid optimization profile for {input_tensor.name}')
        if build_config.add_optimization_profile(profile) < 0:
            raise RuntimeError('Failed to register TensorRT profile')
    if fp16:
        build_config.set_flag(trt.BuilderFlag.FP16)
    serialized = builder.build_serialized_network(network, build_config)
    if serialized is None:
        raise RuntimeError('TensorRT engine build failed; check parser logs and GPU memory')
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_bytes(bytes(serialized))
    print(f'Saved TensorRT engine: {output_path}')


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--onnx_path', required=True)
    p.add_argument('--model_type', choices=MODEL_TYPES, required=True)
    p.add_argument('--config_path', required=True)
    p.add_argument('--output_path', required=True)
    p.add_argument('--fp16', action='store_true')
    p.add_argument('--chunk_size', type=int)
    p.add_argument('--batch_size', type=int)
    args = p.parse_args()
    config = load_config(args.model_type, args.config_path)
    if args.chunk_size is not None:
        config.inference.chunk_size = args.chunk_size
    if args.batch_size is not None:
        config.inference.batch_size = args.batch_size
    export_to_tensorrt(args.onnx_path, args.model_type, config, args.output_path, args.fp16)


if __name__ == '__main__':
    main()
