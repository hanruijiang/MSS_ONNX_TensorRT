# MSS ONNX — Vocal separation

Vocal separation inference with ONNX Runtime (CPU or CUDA). ONNX Runtime executes the neural network core; STFT, mask application and iSTFT run in PyTorch. Existing ONNX graphs require no checkpoint or model architecture dependencies. PyTorch model definitions are retained for checkpoint-to-ONNX export.

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
# For checkpoint-to-ONNX export with CPU ORT validation:
pip install -r requirements-export.txt
# OR, if using GPU ORT instead:
pip install -r requirements.txt "onnx>=1.16" onnxruntime-gpu
```

For ONNX export, additionally install **only** the selected model's architecture dependencies:

| Model | Extra packages for ONNX export |
|---|---|
| MDX23C | None |
| Segm Models | `pip install segmentation-models-pytorch==0.3.3` |
| BS Roformer / MelBand Roformer | `pip install rotary-embedding-torch==0.3.5` |
| BS PolarFormer | None |

For PolarFormer ONNX export, install `onnxscript` for the Dynamo exporter. Segm Models installs `timm==0.9.2` and torchvision through its pinned architecture dependency; use torchvision compatible with PyTorch. Existing ONNX graphs do not require model extras.

For CUDA, ensure PyTorch, CUDA, cuDNN and ONNX Runtime are compatible. See the [inference guide](docs/inference.md).

## Inference

Use the published [FP32 ONNX](https://huggingface.co/bgkb/bs_polarformer/resolve/main/bs_polarformer.onnx) or [FP16-weight ONNX](https://huggingface.co/bgkb/bs_polarformer/resolve/main/bs_polarformer_fp16.onnx), not the WebGPU variants.

```bash
mkdir -p weights input
curl -L --fail https://huggingface.co/bgkb/bs_polarformer/resolve/main/bs_polarformer.onnx -o weights/bs_polarformer.onnx

python inference.py --model_type bs_polarformer \
  --config_path configs/model_bs_polarformer_float16.yaml \
  --input_folder input --store_dir outputs --extract_instrumental \
  --onnx_model_path weights/bs_polarformer.onnx --force_cpu
```

For other models substitute the preset, model type and ONNX path. Omit `--force_cpu` to use CUDA when available. `--onnx_model_path` is required; no backend selection flag is needed.

## ONNX export

```bash
python export_to_onnx.py --model_type bs_roformer \
  --config_path configs/viperx/model_bs_roformer_ep_317_sdr_12.9755.yaml \
  --checkpoint_path weights/model_bs_roformer_ep_317_sdr_12.9755.ckpt \
  --output_path weights/bs_roformer.onnx --force_cpu
```

PolarFormer defaults to opset 18; other models default to opset 17. Export loads the checkpoint strictly and compares the output with ONNX Runtime.

See the [inference guide](docs/inference.md) for chunk and batch settings, precision, audio output and tensor shapes.

## Experimental Native WebGPU optimization

See [Native WebGPU spectral-core pipeline](docs/native_webgpu.md) for safe export, portable graph optimization, mixed precision conversion, Native WebGPU I/O binding, capture and validation. These are opt-in; existing CPU/CUDA commands are unchanged.

## Tests

```bash
python -m unittest discover -s tests -v
```

The tests cover configuration loading, PolarFormer feature ordering, complex mask reconstruction, DC handling and short/partial chunks, ONNX argument validation and fixed/dynamic batch handling.

## Credits

Original MSST: [ZFTurbo](https://github.com/ZFTurbo/Music-Source-Separation-Training) (Roman Solovyev); PolarFormer ONNX conversion and interface: [bgkb/bs_polarformer](https://huggingface.co/bgkb/bs_polarformer); Roformer and PoPE: [lucidrains](https://github.com/lucidrains). See [LICENSE](LICENSE) for license terms.
