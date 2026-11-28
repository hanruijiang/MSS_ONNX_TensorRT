# ONNX Runtime and model contracts

All five models use 44.1 kHz stereo. Mono input is duplicated to stereo. Outputs preserve the resampled length and input basename; `--extract_instrumental` writes mixture minus vocals after denormalization. MDX23C and Segm directly predict vocals and other; the Roformer presets directly predict vocals only. WAV output is floating point; FLAC uses PCM_16 or PCM_24.

| Model | Core input | Core output |
|---|---|---|
| MDX23C / Segm | B × 4 × F × T (channel-major real/imaginary) | estimated spectral planes |
| BS Roformer | B × T × (F × 4) | complex mask |
| MelBand Roformer | B × selected band-frequency-channel entries × T × 2 | per-band masks, averaged across overlapping bands |
| BS PolarFormer | B × T × 4100, `stft_features` | B × 1 × 2050 × T × 2, `mask` |

PolarFormer uses 2048 FFT/window, 512 hop, periodic Hann, centered STFT, unnormalized, frequency-major stereo interleaving. The complex mask multiplies the original spectrum; DC is zeroed by default in each channel for all three Roformer presets before iSTFT with explicit waveform length (`model.zero_dc` overrides this). The default chunk is 882000 samples (1723 frames), overlap 2, batch 1.

## Runtime selection

- `--onnx_model_path`: FP32 or mixed-precision spectral-core ONNX model.
- `--provider`: `auto`, `cpu`, `cuda`, or `webgpu`. See the [Native WebGPU guide](native_webgpu.md) for graph capture and I/O binding.
- `--device_id`: selects a single CUDA device (default 0); `--force_cpu` selects CPU.
- `--use_tta`: channel swap and polarity averaging for ONNX inference; slower.
- `--chunk_size`, `--batch_size`: override inference settings. Match the chunk size to the exported graph if it has a fixed time dimension. Fixed-batch graphs use the graph's batch size and pad the final batch. Dynamic-batch graphs use the configured batch size.

I/O binding is automatic on CPU, CUDA and Native WebGPU. The runtime accepts FP32 or FP16 graph inputs and outputs.

Roformer uses RoPE attention. PolarFormer uses PoPE.

## Memory and precision

If PolarFormer runs out of memory, try `--chunk_size 131072 --batch_size 1` with a dynamic graph or export a static graph with those settings. Shorter chunks can affect separation quality.

### CUDA FP16 Attention and weight quantization

For models with compatible attention on CUDA, convert an FP32 graph exported with `--optimization safe`.
The converter uses standard ONNX Attention with FP32 graph I/O and FP32 softmax.
Standard Attention fusion requires an input graph at opset 23 or later and is skipped below 23.
The converter preserves the input opset for FP16 and FFN quantization.

```sh
python convert_onnx_precision.py --input weights/bs_optimized_fp32.onnx \
  --output weights/bs_optimized_fp16_attention.onnx --fuse_attention

python inference.py --model_type bs_roformer \
  --config_path configs/viperx/model_bs_roformer_ep_317_sdr_12.9755.yaml \
  --input_folder input --store_dir output --onnx_model_path weights/bs_optimized_fp16_attention.onnx \
  --provider cuda --chunk_size 176400 --batch_size 1
```

Use these chunk and batch sizes when exporting a static graph for this preset.

Add `--quantization ffn_int8` or `--quantization ffn_int4` to `export_to_onnx.py`
with `--optimization safe`, or pass the same option to `convert_onnx_precision.py`
with an optimized FP32 ONNX input. Both modes quantize the weights of Roformer
FFNs, MDX23C TDF layers, or Segm Models MaxViT MLPs in symmetric blocks of 64.
They also fuse compatible attention automatically and keep FP32 graph inputs and outputs.
The quantized models use ONNX Runtime's `com.microsoft::MatMulNBits`.

## Configuration

`audio` defines the sample rate and convolutional-model STFT settings. `model` defines the neural network and Roformer STFT settings. `inference` defines `chunk_size`, `batch_size`, `num_overlap`, and the ordered output `stems`. Set `inference.normalize` to enable waveform normalization.

## Sources

- [PolarFormer model card and conversion scripts](https://huggingface.co/bgkb/bs_polarformer/tree/main)
- [MSST checkpoint releases](https://github.com/ZFTurbo/Music-Source-Separation-Training/releases)
