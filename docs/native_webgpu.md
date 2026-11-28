# Native WebGPU

PyTorch FP32 STFT → ONNX spectral core → FP32 mask application / iSTFT.

## Export and optimization

Install `requirements-export.txt`, the model dependencies in README, and a compatible
`onnxruntime` / `onnxruntime-ep-webgpu` pair.

```sh
python export_to_onnx.py --model_type bs_roformer \
  --config_path configs/viperx/model_bs_roformer_ep_317_sdr_12.9755.yaml \
  --checkpoint_path weights/model_bs_roformer_ep_317_sdr_12.9755.ckpt \
  --output_path weights/bs_safe_fp32.onnx --optimization safe --group_size 4 \
  --shape_mode static --batch_size 1 --force_cpu

python optimize_onnx.py --input weights/bs_safe_fp32.onnx --output weights/bs_optimized_fp32.onnx
python convert_onnx_precision.py --input weights/bs_optimized_fp32.onnx --output weights/bs_optimized_fp16.onnx
```

- `--optimization off|safe`: safe packs bands and mask estimators, merges Roformer
  QKV and gate projections, folds Segm Conv-BN pairs, removes dropout and
  precomputes fixed RoPE/PoPE tables.
- `--group_size`: maximum equal-shape bands per packed MatMul (default 4).
- `--shape_mode static|dynamic`: static shapes for graph capture; dynamic batch for
  all models and dynamic time for Roformers. Dynamic PolarFormer traces with
  example batch 2–16 (default 2).
- `--chunk_size`, `--batch_size`: export dimensions. Convolution models require
  chunk lengths compatible with their encoder/decoder strides and windows.

ORT BASIC performs constant folding and graph cleanup, followed by ONNX shape
inference. Native WebGPU can apply additional session-level fusions and layout optimizations.

Mixed precision converts Conv, ConvTranspose, MatMul, Gemm, bias additions and
selected activation kernels to FP16. Graph I/O, normalization, softmax and the
position tables remain FP32. Use `--keep_op` or `--keep_node` to keep selected compute kernels in FP32.

## Inference

```sh
python inference.py --model_type bs_roformer \
  --config_path configs/viperx/model_bs_roformer_ep_317_sdr_12.9755.yaml \
  --input_folder input --store_dir output --onnx_model_path weights/bs_optimized_fp16.onnx \
  --provider webgpu --webgpu_layout NCHW
```

Set `--chunk_size` to match a static export. Use `--webgpu_graph_capture` for
static graphs that can run on WebGPU.

| Option | Values |
|---|---|
| `--webgpu_layout` | NCHW, NHWC |
| `--webgpu_buffer_cache` | bucket (default), simple, lazyRelease, disabled |
| `--webgpu_graph_capture` | Capture and replay fixed-shape inference |
| `--webgpu_strict` | Disable CPU EP fallback |

## Benchmark

```sh
python benchmark_webgpu.py --model_type bs_roformer \
  --config_path configs/viperx/model_bs_roformer_ep_317_sdr_12.9755.yaml \
  --reference weights/bs_safe_fp32.onnx --model weights/bs_optimized_fp16.onnx \
  --provider webgpu --graph_capture --batch_size 1
```

Reports warm p50/p95 core latency including GPU transfers and compares the
spectral and reconstructed waveform outputs with FP32. Compare chunk size, batch
size, precision, capture and NCHW/NHWC on the target adapter. Use `--profile`
to inspect attention kernels, layout transforms and CPU fallback in the ORT profile.

[Native WebGPU EP](https://onnxruntime.ai/docs/execution-providers/WebGPU-ExecutionProvider.html)
