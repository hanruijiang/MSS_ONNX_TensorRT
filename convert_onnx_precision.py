"""Convert a spectral FP32 ONNX graph to mixed precision, keeping FP32 I/O."""
import argparse
from pathlib import Path
import onnx

# Preserve reduction/norm, softmax and positional nonlinearities conservatively.
# Individual sensitive nodes (e.g. phase construction or attention score MatMul)
# can be explicitly blocked after numerical comparison.
SENSITIVE = {
    'BatchNormalization', 'InstanceNormalization', 'LayerNormalization',
    'GroupNormalization', 'RMSNormalization', 'SimplifiedLayerNormalization',
    'ReduceL2', 'ReduceMean', 'ReduceSum', 'Sqrt', 'Pow', 'Div', 'Clip',
    'Softmax', 'LogSoftmax', 'Exp', 'Log', 'Softplus', 'Sin', 'Cos', 'Range',
}


def convert_precision(source, output, *, blocked_ops=(), blocked_nodes=()):
    from onnxconverter_common.float16 import convert_float_to_float16, DEFAULT_OP_BLOCK_LIST
    source, output = Path(source), Path(output)
    if source.resolve() == output.resolve():
        raise ValueError('Use a separate output path')
    model = onnx.load(str(source))
    if any(t.data_type == onnx.TensorProto.FLOAT16 for t in model.graph.initializer):
        raise ValueError('Expected an FP32 source model')
    known_names = {n.name for n in model.graph.node}
    unknown = set(blocked_nodes) - known_names
    if unknown:
        raise ValueError(f'Unknown node names: {sorted(unknown)}')
    model = convert_float_to_float16(
        model, keep_io_types=True,
        op_block_list=sorted(set(DEFAULT_OP_BLOCK_LIST) | SENSITIVE | set(blocked_ops)),
        node_block_list=list(blocked_nodes))
    model = onnx.shape_inference.infer_shapes(model)
    onnx.checker.check_model(model)
    output.parent.mkdir(parents=True, exist_ok=True)
    onnx.save(model, str(output))
    half_weights = sum(t.data_type == onnx.TensorProto.FLOAT16 for t in model.graph.initializer)
    print(f'Mixed precision graph: {output}; {half_weights} FP16 initializers')


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--input', required=True)
    p.add_argument('--output', required=True)
    p.add_argument('--keep_op', action='append', default=[])
    p.add_argument('--keep_node', action='append', default=[])
    args = p.parse_args()
    convert_precision(args.input, args.output, blocked_ops=args.keep_op, blocked_nodes=args.keep_node)


if __name__ == '__main__':
    main()
