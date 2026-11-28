"""Convert or quantize spectral-core compute kernels with FP32 graph I/O."""
import argparse
from collections import Counter
import logging
from pathlib import Path
import re
from tempfile import TemporaryDirectory

import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper

from onnx_opset import require_model_opset, require_onnx_support


FP16_OPS = {'Conv', 'ConvTranspose', 'MatMul', 'Gemm', 'Gelu', 'Relu', 'Tanh', 'Sigmoid'}
QUANTIZATION_BITS = {'ffn_int8': 8, 'ffn_int4': 4}
ATTENTION_MIN_OPSET = 23


def _matches(node, op):
    return node is not None and not node.domain and node.op_type == op


def _attributes(node):
    return {attr.name: helper.get_attribute_value(attr) for attr in node.attribute}


def _attention(graph, tensors, types, initializers, blocked_ops, blocked_nodes, unique):
    """Match rank-four attention, including the optional MaxViT position bias."""
    producers = {name: node for node in graph.node for name in node.output}
    uses = Counter(name for node in graph.node for name in node.input)
    uses.update(value.name for value in graph.output)

    indices = {}

    def index(values):
        if values not in indices:
            name = unique('attention_indices')
            initializers[name] = helper.make_tensor(name, TensorProto.INT64, [len(values)], values)
            indices[values] = name
        return indices[values]

    replacements, removed = {}, set()
    for context in graph.node:
        if not _matches(context, 'MatMul'):
            continue
        probability, value = context.input
        softmax = producers.get(probability)
        if not _matches(softmax, 'Softmax') or _attributes(softmax).get('axis', -1) not in (-1, 3):
            continue
        logits = softmax.input[0]
        score = producers.get(logits)
        pattern = [softmax, context]
        intermediate = [probability, logits]
        bias = []
        if _matches(score, 'Add'):
            a, b = score.input
            if _matches(producers.get(b), 'MatMul'):
                a, b = b, a
            if types.get(b) != TensorProto.FLOAT:
                continue
            bias = [b]
            pattern.append(score)
            intermediate.append(a)
            score = producers.get(a)
        if not _matches(score, 'MatMul'):
            continue
        pattern.append(score)
        query, transposed_key = score.input
        transpose = producers.get(transposed_key)
        direct = _matches(transpose, 'Transpose') and _attributes(transpose).get('perm') == [0, 1, 3, 2]
        key = transpose.input[0] if direct else transposed_key
        if direct and uses[transposed_key] == 1:
            intermediate.append(transposed_key)
            pattern.append(transpose)
        if (any(uses[name] != 1 for name in intermediate)
                or any(node.op_type in blocked_ops or node.name in blocked_nodes for node in pattern)):
            continue
        if any(types.get(name) != TensorProto.FLOAT or name not in tensors
               or len(tensors[name].shape.dim) != 4 for name in (query, key, value, context.output[0])):
            continue
        query_dim = tensors[query].shape.dim[-1]
        value_dim = tensors[value].shape.dim[-1]
        query_width, value_width = query_dim.dim_value, value_dim.dim_value
        same_symbol = query_dim.dim_param and query_dim.dim_param == value_dim.dim_param
        if not (same_symbol or 0 < value_width <= query_width):
            continue
        prefix = []
        if not direct:
            # Some exporters apply the scale after transposing K.
            key = unique(transposed_key + '_attention_key')
            prefix.append(helper.make_node('Transpose', [transposed_key], [key], perm=[0, 1, 3, 2]))
            types[key] = TensorProto.FLOAT
        output = context.output[0]
        crop = None
        if query_width != value_width:
            # Match the V head width to Q and K for the CUDA Attention kernel.
            padded_value = unique(value + '_attention_value')
            pads = index((0, 0, 0, 0, 0, 0, 0, query_width - value_width))
            prefix.append(helper.make_node('Pad', [value, pads], [padded_value]))
            types[padded_value] = TensorProto.FLOAT
            value = padded_value
            output = unique(output + '_attention_context')
            types[output] = TensorProto.FLOAT
            crop = helper.make_node('Slice', [output, index((0,)), index((value_width,)), index((-1,))],
                                    context.output)
        # Q and K already include positional encoding and scaling.
        attention = helper.make_node('Attention', [query, key, value, *bias], [output],
                                     name=context.name, scale=1.0, softmax_precision=TensorProto.FLOAT)
        replacements[context.output[0]] = (prefix, attention, crop)
        removed.update(intermediate)
    return replacements, removed


def _linear_targets(graph):
    def in_feed_forward(node):
        name = node.name
        if ((name.startswith('/layers.') and '/net/' in name) or '/tdf/' in name or '/mlp/' in name):
            return True
        scopes = next((prop.value for prop in node.metadata_props if prop.key == 'pkg.torch.onnx.name_scopes'), '')
        return '.tdf.' in scopes or '.mlp.' in scopes or bool(re.search(r"'layers\.[\w.]*\.net[.']", scopes))

    weights = {weight.name: weight for weight in graph.initializer}
    linears = {node.output[0]: node for node in graph.node
               if not node.domain and node.op_type in ('MatMul', 'Gemm')
               and node.input[1] in weights and len(weights[node.input[1]].dims) == 2}
    targets = {name for name, node in linears.items() if in_feed_forward(node)}
    # Find PolarFormer FFNs by the two projections around GELU.
    producers = {name: node for node in graph.node for name in node.output}
    for name, down in linears.items():
        activation = down.input[0]
        if not (re.fullmatch(r'gelu(?:_\d+)?', activation) or _matches(producers.get(activation), 'Gelu')):
            continue
        pending, seen, upstream = [activation], set(), set()
        while pending:
            value = pending.pop()
            if value in seen:
                continue
            seen.add(value)
            node = producers.get(value)
            if value in linears:
                upstream.add(value)
            elif node is not None and not node.domain and node.op_type in ('Add', 'Mul', 'Div', 'Erf', 'Gelu'):
                pending.extend(node.input)
        if len(upstream) == 1:
            up_name = upstream.pop()
            up_dims = tuple(weights[linears[up_name].input[1]].dims)
            down_dims = tuple(weights[down.input[1]].dims)
            if up_dims == down_dims[::-1] or up_dims == down_dims:
                targets.update((name, up_name))
    return targets


def _quantize_linears(model, targets, bits, unique):
    from onnxruntime.quantization import QuantFormat
    from onnxruntime.quantization.matmul_nbits_quantizer import MatMulNBitsQuantizer

    graph = model.graph
    weights = {weight.name: weight for weight in graph.initializer}
    nodes, selected = [], set()
    for node in graph.node:
        target = node.output and node.output[0] in targets
        if not target or node.op_type not in ('MatMul', 'Gemm'):
            nodes.append(node)
            continue
        weight = weights[node.input[1]]
        if weight.data_type != TensorProto.FLOAT16:
            raise ValueError(f'Expected FP16 weight for {node.name}')
        if node.op_type == 'Gemm':
            attrs = _attributes(node)
            if attrs.get('transA', 0) or attrs.get('alpha', 1.0) != 1.0 or attrs.get('beta', 1.0) != 1.0:
                raise ValueError(f'Unsupported Gemm attributes for {node.name}')
            weight_name = weight.name
            if attrs.get('transB', 0):
                weight_name = unique(weight.name + '_transposed')
                graph.initializer.append(numpy_helper.from_array(numpy_helper.to_array(weight).T, weight_name))
            output = node.output[0]
            bias = node.input[2] if len(node.input) == 3 else ''
            product = unique(output + '_matmul') if bias else output
            nodes.append(helper.make_node('MatMul', [node.input[0], weight_name], [product], name=node.name))
            if bias:
                nodes.append(helper.make_node('Add', [product, bias], [output]))
        else:
            nodes.append(node)
        selected.add(node.name)
    if len(selected) != len(targets) or any(not name for name in selected):
        raise ValueError('FFN quantization requires distinct named FP16 projections; check keep filters')
    del graph.node[:]
    graph.node.extend(nodes)
    quantizer = MatMulNBitsQuantizer(
        model, bits=bits, block_size=64, is_symmetric=True, quant_format=QuantFormat.QOperator,
        op_types_to_quantize=('MatMul',),
        nodes_to_exclude=[node.name for node in nodes if node.op_type == 'MatMul' and node.name not in selected],
    )
    logger = logging.getLogger(MatMulNBitsQuantizer.__module__)
    level = logger.level
    try:
        logger.setLevel(logging.WARNING)
        quantizer.process()
    finally:
        logger.setLevel(level)
    model = quantizer.model.model
    quantized = [node for node in model.graph.node
                 if node.domain == 'com.microsoft' and node.op_type == 'MatMulNBits']
    if len(quantized) != len(selected) or any(_attributes(node).get('bits') != bits for node in quantized):
        raise ValueError(f'Unexpected INT{bits} projection coverage')
    return model


def convert_precision(source, output, *, blocked_ops=(), blocked_nodes=(), fuse_attention=False,
                      quantization='none', opset_version=None):
    source, output = Path(source), Path(output)
    if source.resolve() == output.resolve():
        raise ValueError('Use a separate output path')
    if quantization != 'none' and quantization not in QUANTIZATION_BITS:
        raise ValueError(f'Unknown quantization: {quantization}')
    bits = QUANTIZATION_BITS.get(quantization)
    model = onnx.load(str(source))
    opset_version = require_model_opset(model, opset_version, 'Precision conversion input')
    require_onnx_support(opset_version)
    attention_allowed = opset_version >= ATTENTION_MIN_OPSET
    if (fuse_attention or bits) and not attention_allowed:
        print(f'Skipping ONNX Attention fusion for opset {opset_version} (requires {ATTENTION_MIN_OPSET} or later)')
        fuse_attention = False
    if (fuse_attention or bits) and any(
            attr.type in (onnx.AttributeProto.GRAPH, onnx.AttributeProto.GRAPHS)
            for node in model.graph.node for attr in node.attribute):
        raise ValueError('Attention fusion and FFN quantization expect a flat graph')
    model = onnx.shape_inference.infer_shapes(model)
    graph = model.graph
    initializers = {t.name: t for t in graph.initializer}
    tensors = {v.name: v.type.tensor_type for v in (*graph.input, *graph.output, *graph.value_info)}
    types = {name: tensor.elem_type for name, tensor in tensors.items()}
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

    replacements, removed = {}, set()
    if attention_allowed and (fuse_attention or bits) and 'Attention' not in blocked_ops:
        replacements, removed = _attention(graph, tensors, types, initializers, blocked_ops, blocked, unique)
    if fuse_attention and not replacements:
        raise ValueError('No eligible attention found; use a safe FP32 export and check keep filters')
    targets = _linear_targets(graph) if bits else set()
    if bits and not targets:
        raise ValueError('No eligible feed-forward projections found; use a safe FP32 export')

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
    quantized_outputs = set()
    for node in original_nodes:
        if node.output and node.output[0] in removed:
            continue
        replacement = replacements.get(node.output[0]) if node.output else None
        crop = None
        if replacement is not None:
            prefix, node, crop = replacement
            nodes.extend(prefix)
        bias_add = (node.op_type == 'Add' and 'Add' not in blocked_ops
                    and any(name in half for name in node.input)
                    and any(name in initializers for name in node.input))
        if (node.domain not in ('', 'ai.onnx') or (node.op_type not in ops and not bias_add and replacement is None)
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
        nodes.append(node)
        if crop is not None:
            crop.input[0] = target
            out = crop.output[0]
            target = unique(out + '_fp16')
            crop.output[0] = target
            nodes.append(crop)
        if out in targets:
            quantized_outputs.add(target)
        half[out] = target
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
    if bits:
        if len(quantized_outputs) != len(targets):
            raise ValueError('FFN quantization requires FP16 projections; check keep filters')
        model = _quantize_linears(model, quantized_outputs, bits, unique)
    require_model_opset(model, opset_version, 'Precision conversion output')
    model = onnx.shape_inference.infer_shapes(model)
    onnx.checker.check_model(model, full_check=bool(bits))
    output.parent.mkdir(parents=True, exist_ok=True)
    if bits:
        with TemporaryDirectory(dir=output.parent) as tmp:
            staged = Path(tmp) / 'quantized.onnx'
            onnx.save(model, str(staged))
            staged.replace(output)
        print(f'Quantized {len(targets)} feed-forward projections to INT{bits} (block size 64)')
    else:
        onnx.save(model, str(output))
    print(f'Mixed precision graph: {output}; {converted} FP16 compute nodes')
    if replacements:
        print(f'Fused {len(replacements)} FP16 Attention nodes')


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--input', required=True)
    p.add_argument('--output', required=True)
    p.add_argument('--opset_version', type=int, help='Expected ONNX opset (default: read from input)')
    p.add_argument('--keep_op', action='append', default=[])
    p.add_argument('--keep_node', action='append', default=[])
    p.add_argument('--fuse_attention', action='store_true',
                   help='Fuse compatible attention for CUDA (requires ONNX opset 23 support)')
    p.add_argument('--quantization', choices=('none', *QUANTIZATION_BITS), default='none',
                   help='Apply INT8/INT4 feed-forward weight quantization and fuse compatible attention')
    args = p.parse_args()
    convert_precision(args.input, args.output, blocked_ops=args.keep_op, blocked_nodes=args.keep_node,
                      fuse_attention=args.fuse_attention, quantization=args.quantization,
                      opset_version=args.opset_version)


if __name__ == '__main__':
    main()
