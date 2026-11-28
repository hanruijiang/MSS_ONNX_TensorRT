"""Optimize FP32 ONNX graphs with ORT BASIC and shape inference."""
import argparse
from pathlib import Path
import tempfile
import onnx
import onnxruntime as ort

from onnx_opset import require_model_opset, require_onnx_support


def optimize_graph(source, output, *, opset_version=None):
    source, output = Path(source), Path(output)
    if source.resolve() == output.resolve():
        raise ValueError('Use a separate output path')
    opset_version = require_model_opset(onnx.load(str(source), load_external_data=False), opset_version,
                                        'ONNX optimization input')
    require_onnx_support(opset_version)
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=output.parent) as tmp:
        intermediate = Path(tmp) / 'basic.onnx'
        opts = ort.SessionOptions()
        opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_BASIC
        opts.optimized_model_filepath = str(intermediate)
        # BASIC runs before EP partitioning: constant folding and redundant nodes.
        ort.InferenceSession(str(source), sess_options=opts, providers=['CPUExecutionProvider'])
        graph = onnx.load(str(intermediate))
        try:
            require_model_opset(graph, opset_version, 'ONNX BASIC output')
        except ValueError as error:
            print(f'Skipping ONNX BASIC optimization ({error})')
            graph = onnx.load(str(source))
        graph = onnx.shape_inference.infer_shapes(graph)
        onnx.checker.check_model(graph)
        onnx.save(graph, str(output))
    print(f'Saved FP32 graph: {output} ({len(graph.graph.node)} nodes)')


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--input', required=True)
    p.add_argument('--output', required=True)
    p.add_argument('--opset_version', type=int, help='Expected ONNX opset (default: read from input)')
    args = p.parse_args()
    optimize_graph(args.input, args.output, opset_version=args.opset_version)


if __name__ == '__main__':
    main()
