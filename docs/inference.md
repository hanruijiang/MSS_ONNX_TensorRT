# Backend and model contracts

All presets use 44.1 kHz stereo. Mono input is duplicated to stereo. Outputs preserve the resampled length and input basename; `--extract_instrumental` writes mixture minus vocals after denormalization. MDX23C and Segm directly predict vocals and other; the Roformer presets directly predict vocals only. WAV output is floating point; FLAC uses PCM_16 or PCM_24.

| Model | Core input | Core output |
|---|---|---|
| MDX23C / Segm | B × 4 × F × T (channel-major real/imaginary) | estimated spectral planes |
| BS Roformer | B × T × (F × 4) | complex mask |
| MelBand Roformer | B × selected band-frequency-channel entries × T × 2 | per-band masks, averaged across overlapping bands |
| BS PolarFormer | B × T × 4100, `stft_features` | B × 1 × 2050 × T × 2, `mask` |

PolarFormer uses 2048 FFT/window, 512 hop, periodic Hann, centered STFT, unnormalized, frequency-major stereo interleaving. The complex mask multiplies the original spectrum; DC is zeroed in each channel before iSTFT with explicit waveform length. `audio.hop_length` and `audio.dim_t` from training configs are not used for this model. The official inference chunk is 882000 samples (1723 frames), overlap 2; default batch 1 here to limit memory. The FP16-weight Hugging Face graph keeps FP32 external IO; it is not a true FP16 compute engine.

## Dependencies

The common requirements are used by the shared audio/configuration pipeline. Existing ONNX graphs need only the common requirements and exactly one ONNX Runtime distribution (CPU or GPU). TensorRT build/inference needs the common requirements plus a compatible TensorRT 10.x installation, without ONNX Runtime or the Python ONNX package. For checkpoint inference/export, install only the selected model extra described in the [README](../README.md). The Python ONNX package is needed for checkpoint export/checking, and ONNX Script only for the PolarFormer Dynamo export path. Model architecture imports are lazy; beartype and an explicit timm dependency are not needed.

## Runtime selection

- `--use_onnx --onnx_model_path ...`: uses CPU or CUDA Execution Provider. No checkpoint/model construction.
- `--use_tensorrt --tensorrt_model_path ...`: TensorRT 10, CUDA required; shapes/dtypes taken from engine tensors, execution on the selected PyTorch CUDA stream.
- Neither flag: PyTorch spectral core, requires `--start_check_point ...`. Optional `--use_compile` compiles this core only.
- `--use_tta`: channel swap and polarity averaging for all backends; slower.
- `--chunk_size`, `--batch_size`: override inference settings. Static ONNX axes must match the preset. TensorRT build fixes non-batch dimensions to the chosen chunk shape; use the same chunk size for inference. Dynamic batch is bounded by build profile. Final batches are padded for fixed-batch engines.

## Memory and numerical validation

PolarFormer attention memory grows with time frames; if 882000 samples is too large, try `--chunk_size 131072 --batch_size 1` for dynamic ONNX, or rebuild TensorRT with matching overrides. Shorter chunks can affect quality. FP16 TensorRT may have different numerical behavior: first build FP32 and compare reconstructed audio before enabling FP16. TensorRT parser compatibility depends on the installed release and graph operators. Do not assume a successfully parsed graph is numerically equivalent.

Suggested validation: compare PyTorch and ONNX on the same chunk, then ONNX and TensorRT on the same audio with matching settings, verify shape, finite values, residual consistency and SNR. Test mono, stereo, silence, short clips, final partial batches and multi-chunk songs. Report GPU, driver, CUDA, ORT/TRT versions, precision, batch and chunk size with any performance measurements.

## Sources

- [Reference inference-only repository](https://github.com/hanruijiang/Music-Source-Separation-Training/tree/cleanup/vocal-inference-only)
- [PolarFormer model card and conversion scripts](https://huggingface.co/bgkb/bs_polarformer/tree/main)
- [Original MSST checkpoint releases](https://github.com/ZFTurbo/Music-Source-Separation-Training/releases)
