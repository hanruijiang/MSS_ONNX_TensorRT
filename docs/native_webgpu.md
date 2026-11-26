# Native WebGPU spectral-core pipeline (experimental)

The waveform interface remains unchanged: PyTorch FP32 STFT → ONNX spectral core
→ FP32 complex mask application / iSTFT. No CUDA/Triton inference optimizations or
audio caching are introduced here. Use the original checkpoints, not packed weights.

## Export, optimize, convert

Install requirements-export.txt plus the model-specific extras in README; install
onnxscript for PolarFormer. For native WebGPU install a compatible
onnxruntime / onnxruntime-ep-webgpu pair (not onnxruntime-web). Do not mix CPU and
GPU ORT wheels in one environment.

Example (substitute model type, preset, checkpoint and valid chunk size):

```sh
python export_to_onnx.py --model_type bs_roformer \
  --config_path configs/viperx/model_bs_roformer_ep_317_sdr_12.9755.yaml \
  --checkpoint_path weights/model_bs_roformer_ep_317_sdr_12.9755.ckpt \
  --output_path weights/bs_safe_fp32.onnx --optimization safe --group_size 4 \
  --shape_mode static --batch_size 1 --force_cpu

python optimize_onnx.py --input weights/bs_safe_fp32.onnx --output weights/bs_optimized_fp32.onnx
python convert_onnx_precision.py --input weights/bs_optimized_fp32.onnx --output weights/bs_optimized_fp16.onnx
```

Export an additional baseline using --optimization off and identical shape arguments.
Dynamic export means dynamic batch for MDX23C, Segm, BS and MelBand; only PolarFormer
currently supports dynamic time as well. Dynamic PolarFormer traces example batch 2–16
(default 2); static export uses preset chunk size unless overridden with --chunk_size.
Convolution models require chunk sizes that satisfy the architecture's stride/window
constraints. Static export is recommended for capture. Large models may exceed the
single-file ONNX protobuf limit; external-data export/conversion is not implemented here.

Graph optimization uses ORT BASIC (portable pre-partition passes), followed by ONNX
shape inference and checker. It does not bake CPU-specific extended fusions or layout
changes into the model. Static shape-dependent positional expressions may be folded
where supported, but RoPE/PoPE precomputation is not guaranteed. BS/MelBand safe mode
expresses attention as MatMul/Softmax/MatMul. PolarFormer retains its original SDPA
and scale; for the preset dim_head=64, Q/K=128 after PoPE, V=64, scale=1/sqrt(64).
No MultiHeadAttention/FlashAttention fusion is asserted. Do not pad V or change scale
without numerical and performance tests.

FP16 conversion targets computation as well as weights. Graph inputs and outputs stay
FP32; the converter retains common normalization/reduction/softmax and positional
nonlinearities in FP32. Precision-sensitive neighboring ops may still need explicit
--keep_node or --keep_op flags. Inspect the graph and validate each candidate, including
PoPE phase construction at long sequences. This is a starting policy, not an approved
precision profile. CPU ORT may lack some FP16 kernels; use the target EP for validation.

## Native inference

```sh
python inference.py --model_type bs_roformer \
  --config_path configs/viperx/model_bs_roformer_ep_317_sdr_12.9755.yaml \
  --input_folder input --store_dir output --onnx_model_path weights/bs_optimized_fp16.onnx \
  --provider webgpu --webgpu_layout NCHW --webgpu_io_binding
```

Add --webgpu_graph_capture for static input/output shapes and compatible ORT builds.
Graph capture reuses the same session-owned device OrtValues and IOBinding; the input
is updated in place. ORT memory patterns are disabled when capture is requested.
--webgpu_strict disables CPU EP fallback (may reject shape-only nodes too). A normal
WebGPU run may still use CPU kernels. Capture requires all compute kernels on WebGPU;
failure is not silently retried as CPU. Select FP32 or FP16 via --onnx_model_path.
Default WebGPU buffer cache mode is bucket; compare simple, lazyRelease and disabled.
For convolution models compare --webgpu_layout NCHW versus NHWC. Native device buffers
are opaque, so no raw CUDA pointer or DLPack sharing is attempted. CPU STFT/iSTFT
still requires one upload and one readback per chunk; this is not zero-copy audio.

## Validation and benchmark

```sh
python benchmark_webgpu.py --model_type bs_roformer \
  --config_path configs/viperx/model_bs_roformer_ep_317_sdr_12.9755.yaml \
  --reference weights/bs_off_fp32.onnx --model weights/bs_optimized_fp16.onnx \
  --provider webgpu --graph_capture --batch_size 1
```

Use the matching chunk and batch for every static artifact. The benchmark compares
two distinct seeded spectra and reconstructed audio against the FP32 CPU reference,
then reports warm p50/p95 including transfers but excluding STFT/iSTFT. Use
--provider cpu to check portable FP32 graph transforms. Run the full suite with
python -m unittest discover -s tests -v. For real checkpoints/audio, compare mask
errors, SDR/SI-SDR against the original, cold session time, warm throughput, peak
GPU memory, CPU fallback node placement and end-to-end RTF. Sweep grouping (1,2,4,8),
valid chunk lengths, batch sizes, precision, capture, cache and layout on the same
GPU/driver/ORT version. Profile and validate actual kernels before enabling fusions.
No hardware performance results or production FP16 tolerances have been established.

Native provider API:
https://onnxruntime.ai/docs/execution-providers/WebGPU-ExecutionProvider.html
https://onnxruntime.ai/docs/api/python/api_summary.html
