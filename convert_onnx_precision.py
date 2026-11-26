"""Convert spectral-core compute kernels to FP16 with FP32 graph I/O."""
import argparse
from pathlib import Path

import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper


FP16_OPS = {'Conv', 'ConvTranspose', 'MatMul', 'Gemm', 'Gelu', 'Relu', 'Tanh', 'Sigmoid'}


def convert_precision(source, output, *, blocked_ops=(), blocked_nodes=()):
    source, output = Path(source), Path(output)
    if source.resolve() == output.resolve():
        raise ValueError('Use a separate output path')
    model = onnx.shape_inference.infer_shapes(onnx.load(str(source)))
    graph = model.graph
    initializers = {t.name: t for t in graph.initializer}
    types = {v.name: v.type.tensor_type.elem_type
             for v in (*graph.input, *graph.output, *graph.value_info)}
    types.update({t.name: t.data_type for t in graph.initializer})
    ops = FP16_OPS - set(blocked_ops)
    blocked = set(blocked_nodes)
    nodes, half, weights = [], {}, []
    generated_casts = set()
    original_nodes = list(graph.node)
    # Keep FP32 normalization, positional encoding and softmax; only compute
    # kernels receive half inputs. Reuse half values across adjacent kernels.
    used = {name for node in original_nodes for name in (*node.input, *node.output)}
    used.update(types)
    def unique(base):
        name = base
        while name in used:
            name += '_'
        used.add(name)
        return name

    def as_half(name):
        if name not in half:
            target = unique(name + '_fp16')
            if name in initializers:
                data = numpy_helper.to_array(initializers[name]).astype(np.float16)
                weights.append(numpy_helper.from_array(data, target))
            else:
                nodes.append(helper.make_node('Cast', [name], [target], to=TensorProto.FLOAT16))
                generated_casts.add(target)
            half[name] = target
        return half[name]

    converted = 0
    for node in original_nodes:
        bias_add = (node.op_type == 'Add' and 'Add' not in blocked_ops
                    and any(name in half for name in node.input)
                    and any(name in initializers for name in node.input))
        if (node.domain not in ('', 'ai.onnx') or (node.op_type not in ops and not bias_add)
                or node.name in blocked or len(node.output) != 1
                or types.get(node.output[0]) != TensorProto.FLOAT
                or any(types.get(name) != TensorProto.FLOAT for name in node.input if name)):
            nodes.append(node)
            continue
        out = node.output[0]
        for i, name in enumerate(node.input):
            if name:
                node.input[i] = as_half(name)
        target = unique(out + '_fp16')
        node.output[0] = target
        half[out] = target
        nodes.append(node)
        nodes.append(helper.make_node('Cast', [target], [out], to=TensorProto.FLOAT))
        generated_casts.add(out)
        converted += 1

    # Drop unused FP32 casts and weights while retaining graph outputs.
    live = {v.name for v in (*graph.input, *graph.output)}
    retained = []
    for node in reversed(nodes):
        if not (node.op_type == 'Cast' and node.output[0] in generated_casts) or any(name in live for name in node.output):
            retained.append(node)
            live.update(node.input)
    del graph.node[:]
    graph.node.extend(reversed(retained))
    kept_weights = [t for t in (*initializers.values(), *weights) if t.name in live]
    del graph.initializer[:]
    graph.initializer.extend(kept_weights)
    # Rebuild inferred types for renamed values.
    del graph.value_info[:]
    model = onnx.shape_inference.infer_shapes(model)
    onnx.checker.check_model(model)
    output.parent.mkdir(parents=True, exist_ok=True)
    onnx.save(model, str(output))
    print(f'Mixed precision graph: {output}; {converted} FP16 compute nodes')


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
