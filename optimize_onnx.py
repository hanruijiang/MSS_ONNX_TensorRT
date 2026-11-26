"""Portable FP32 graph optimization; intentionally avoids CPU/GPU-specific fusions."""
import argparse
from pathlib import Path
import tempfile
import onnx
import onnxruntime as ort


def optimize_graph(source, output):
    source, output = Path(source), Path(output)
    if source.resolve() == output.resolve():
        raise ValueError('Use a separate output path')
    output.parent.mkdir(parents=True, exist_ok=True)
    onnx.checker.check_model(str(source))
    with tempfile.TemporaryDirectory(dir=output.parent) as tmp:
        intermediate = Path(tmp) / 'basic.onnx'
        opts = ort.SessionOptions()
        opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_BASIC
        opts.optimized_model_filepath = str(intermediate)
        # BASIC runs before EP partitioning: constant folding and redundant nodes.
        ort.InferenceSession(str(source), sess_options=opts, providers=['CPUExecutionProvider'])
        graph = onnx.load(str(intermediate))
        graph = onnx.shape_inference.infer_shapes(graph, strict_mode=True)
        onnx.checker.check_model(graph)
        onnx.save(graph, str(output))
    print(f'Optimized FP32 graph: {output} ({len(graph.graph.node)} nodes)')


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--input', required=True)
    p.add_argument('--output', required=True)
    args = p.parse_args()
    optimize_graph(args.input, args.output)


if __name__ == '__main__':
    main()
