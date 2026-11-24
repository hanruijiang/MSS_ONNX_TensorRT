# Backend and model contracts

All five models use 44.1 kHz stereo. Mono input is duplicated to stereo. Outputs preserve the resampled length and input basename; `--extract_instrumental` writes mixture minus vocals after denormalization. MDX23C and Segm directly predict vocals and other; the Roformer presets directly predict vocals only. WAV output is floating point; FLAC uses PCM_16 or PCM_24.

| Model | Core input | Core output |
|---|---|---|
| MDX23C / Segm | B × 4 × F × T (channel-major real/imaginary) | estimated spectral planes |
| BS Roformer | B × T × (F × 4) | complex mask |
| MelBand Roformer | B × selected band-frequency-channel entries × T × 2 | per-band masks, averaged across overlapping bands |
| BS PolarFormer | B × T × 4100, `stft_features` | B × 1 × 2050 × T × 2, `mask` |

PolarFormer uses 2048 FFT/window, 512 hop, periodic Hann, centered STFT, unnormalized, frequency-major stereo interleaving. The complex mask multiplies the original spectrum; DC is zeroed by default in each channel for all three Roformer presets before iSTFT with explicit waveform length (`model.zero_dc` overrides this). The default chunk is 882000 samples (1723 frames), overlap 2, batch 1. The FP16-weight Hugging Face graph keeps FP32 external IO; it is not a true FP16 compute engine.

## Runtime selection

- `--use_onnx --onnx_model_path ...`: uses CPU or CUDA Execution Provider. No checkpoint/model construction. CUDA requests require successful CUDA provider initialization; ordinary per-node CPU fallback may still occur.
- `--use_tensorrt --tensorrt_model_path ...`: TensorRT 10, CUDA required; shapes/dtypes taken from engine tensors, execution on the selected PyTorch CUDA stream.
- Neither flag: PyTorch spectral core, requires `--start_check_point ...`. Optional `--use_compile` compiles this core only.
- `--device_id`: selects a single CUDA device (default 0); `--force_cpu` selects CPU.
- `--use_tta`: channel swap and polarity averaging for all backends; slower.
- `--chunk_size`, `--batch_size`: override inference settings. Static ONNX axes must match the preset. TensorRT build fixes non-batch dimensions to the chosen chunk shape; use the same chunk size for inference. Dynamic batch is bounded by build profile. Final batches are padded for fixed-batch engines or to the profile minimum. Input rank, static dimensions and TensorRT profile bounds are checked before processing audio; supported external IO is FP32 or FP16.

The Rotary cores use explicit attention; PolarFormer uses scaled dot-product attention with PoPE positional encoding. Chunk tails longer than half a chunk use reflection padding; shorter tails use zero padding. Checkpoint loading accepts raw state dicts and `state`, `state_dict`, or `model_state_dict` wrappers, strips `module.` and `_orig_mod.` prefixes, and remains strict with `weights_only=True`.

## Memory and precision

PolarFormer attention memory grows with time frames; if 882000 samples is too large, try `--chunk_size 131072 --batch_size 1` for dynamic ONNX, or rebuild TensorRT with matching overrides. Shorter chunks can affect quality. FP16 TensorRT can produce different numerical results from FP32. TensorRT parser compatibility depends on the installed release and graph operators.

## Configuration

`audio` defines the sample rate and convolutional-model STFT settings. `model` defines the neural network and Roformer STFT settings. `inference` defines `chunk_size`, `batch_size`, `num_overlap`, and the ordered output `stems`. Set `inference.normalize` to enable waveform normalization.

## Sources

- [PolarFormer model card and conversion scripts](https://huggingface.co/bgkb/bs_polarformer/tree/main)
- [MSST checkpoint releases](https://github.com/ZFTurbo/Music-Source-Separation-Training/releases)
