# MSS ONNX — Vocal separation

Vocal separation inference with ONNX Runtime (CPU, CUDA or Native WebGPU). ONNX Runtime executes the neural network core; STFT, mask application and iSTFT run in PyTorch. Inference uses ONNX models; checkpoint and model architecture dependencies are needed only for export.

## Models

| Model | `--model_type` | Configuration | Pretrained checkpoint |
|---|---|---|---|
| MDX23C | `mdx23c` | [YAML](configs/config_vocals_mdx23c.yaml) | [Download](https://github.com/ZFTurbo/Music-Source-Separation-Training/releases/download/v1.0.0/model_vocals_mdx23c_sdr_10.17.ckpt) |
| Segm Models (VitLarge23) | `segm_models` | [YAML](configs/config_vocals_segm_models.yaml) | [Download](https://github.com/ZFTurbo/Music-Source-Separation-Training/releases/download/v1.0.0/model_vocals_segm_models_sdr_9.77.ckpt) |
| BS Roformer (viperx) | `bs_roformer` | [YAML](configs/viperx/model_bs_roformer_ep_317_sdr_12.9755.yaml) | [Download](https://github.com/TRvlvr/model_repo/releases/download/all_public_uvr_models/model_bs_roformer_ep_317_sdr_12.9755.ckpt) |
| MelBand Roformer (KimberleyJensen) | `mel_band_roformer` | [YAML](configs/KimberleyJensen/config_vocals_mel_band_roformer_kj.yaml) | [Download](https://huggingface.co/KimberleyJSN/melbandroformer/resolve/main/MelBandRoformer.ckpt) |
| BS PolarFormer (PoPE) | `bs_polarformer` | [YAML](configs/model_bs_polarformer_float16.yaml) | [Download](https://github.com/ZFTurbo/Music-Source-Separation-Training/releases/download/v1.0.20/model_bs_polarformer_float16.ckpt) |

## Installation

Python 3.10+; first install PyTorch for your CPU/CUDA platform.

```bash
pip install -r requirements-onnx.txt  # CPU ONNX Runtime
# OR for CUDA (do not install both ORT packages):
pip install -r requirements.txt onnxruntime-gpu
# For checkpoint-to-ONNX export:
pip install -r requirements-export.txt
# Native WebGPU:
pip install -r requirements-onnx.txt onnxruntime-ep-webgpu
# Export tools with CUDA ORT:
pip install -r requirements.txt "torch>=2.10" "onnx>=1.18" onnxscript onnxruntime-gpu
```

For ONNX export, additionally install **only** the selected model's architecture dependencies:

| Model | Extra packages for ONNX export |
|---|---|
| MDX23C | None |
| Segm Models | `pip install segmentation-models-pytorch==0.3.3` |
| BS Roformer / MelBand Roformer | `pip install rotary-embedding-torch==0.3.5` |
| BS PolarFormer | None |

Segm Models requires a torchvision build compatible with PyTorch.

For CUDA, ensure PyTorch, CUDA, cuDNN and ONNX Runtime are compatible. See the [inference guide](docs/inference.md).

## Inference

Use an exported spectral-core ONNX model. A [published PolarFormer FP32 graph](https://huggingface.co/bgkb/bs_polarformer/resolve/main/bs_polarformer.onnx) is also available.

```bash
mkdir -p weights input
curl -L --fail https://huggingface.co/bgkb/bs_polarformer/resolve/main/bs_polarformer.onnx -o weights/bs_polarformer.onnx

python inference.py --model_type bs_polarformer \
  --config_path configs/model_bs_polarformer_float16.yaml \
  --input_folder input --store_dir outputs --extract_instrumental \
  --onnx_model_path weights/bs_polarformer.onnx --force_cpu
```

Select `--provider cpu`, `cuda` or `webgpu`. The default `auto` selects CUDA when available, otherwise CPU.

## ONNX export

```bash
python export_to_onnx.py --model_type bs_roformer \
  --config_path configs/viperx/model_bs_roformer_ep_317_sdr_12.9755.yaml \
  --checkpoint_path weights/model_bs_roformer_ep_317_sdr_12.9755.ckpt \
  --output_path weights/bs_roformer.onnx --optimization safe --force_cpu
```

All models use the Dynamo exporter. Choose the target for the full export pipeline with
`--opset_version` (default 23). Standalone optimization and precision conversion inherit
the input graph's opset. Pass `--opset_version` to either command to require a specific
input opset. Standard Attention fusion is available from opset 23.

See the [inference guide](docs/inference.md) for chunk and batch settings, precision, audio output and tensor shapes.

## Native WebGPU optimization

See the [Native WebGPU guide](docs/native_webgpu.md) for safe export, graph optimization, FP16 conversion, I/O binding, graph capture and benchmarking.

## Tests

```bash
python -m unittest discover -s tests -v
```

The tests cover configuration loading, PolarFormer feature ordering, complex mask reconstruction, DC handling and short/partial chunks, ONNX argument validation and fixed/dynamic batch handling.

## Credits

MSST by [ZFTurbo](https://github.com/ZFTurbo/Music-Source-Separation-Training) (Roman Solovyev). PolarFormer ONNX conversion and interface by [bgkb/bs_polarformer](https://huggingface.co/bgkb/bs_polarformer). Roformer and PoPE by [lucidrains](https://github.com/lucidrains). See [LICENSE](LICENSE) for license terms.
