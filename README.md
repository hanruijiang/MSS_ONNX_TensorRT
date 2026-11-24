# MSS ONNX / TensorRT — Vocal separation

人声分离推理与模型导出仓库。参考 [MSST vocal-inference-only](https://github.com/hanruijiang/Music-Source-Separation-Training/tree/cleanup/vocal-inference-only)，仅保留五个预训练版本，移除训练、验证、LoRA、GUI、临时 notebook 和其它模型预设。ONNX / TensorRT 仅运行神经网络核心；STFT、掩码和 iSTFT 在 PyTorch 侧完成。推理无需原始 checkpoint 或模型架构依赖。

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
# For TensorRT: install the matching NVIDIA TensorRT 10.x Python package.
# For checkpoint-to-ONNX export with CPU ORT validation:
pip install -r requirements-export.txt
# OR, if using GPU ORT instead:
pip install -r requirements.txt "onnx>=1.16" onnxruntime-gpu
```

For PyTorch checkpoint inference or ONNX export, additionally install **only** the selected model's architecture dependencies:

| Model | Extra packages for PyTorch checkpoint inference or export |
|---|---|
| MDX23C | None |
| Segm Models | `pip install segmentation-models-pytorch==0.3.3` |
| BS Roformer / MelBand Roformer | `pip install rotary-embedding-torch==0.3.5` |
| BS PolarFormer | `pip install PoPE-pytorch` |

For PolarFormer checkpoint-to-ONNX export, also run `pip install onnxscript` (the Dynamo exporter requires it). It is not needed for checkpoint inference or loading an existing ONNX graph. Segm Models 0.3.3 installs its own pinned `timm==0.9.2` and torchvision dependencies; use a torchvision build compatible with your PyTorch. To export all five presets, install all three model extras plus onnxscript. No model extras are needed to run existing ONNX graphs or build/run TensorRT engines. PoPE-pytorch must expose the checkpoint-compatible `PoPE.inv_freqs` and `PoPE.bias` attributes. Use mutually compatible PyTorch, ONNX, ONNX Script and ORT versions; no unverified version pins are implied.

For CUDA, ensure PyTorch, CUDA, cuDNN, ONNX Runtime and TensorRT are compatible. TensorRT engines are device/runtime specific. See [backend guide](docs/inference.md).

## Quick start: BS PolarFormer

Use the published [FP32 ONNX](https://huggingface.co/bgkb/bs_polarformer/resolve/main/bs_polarformer.onnx) or [FP16-weight ONNX](https://huggingface.co/bgkb/bs_polarformer/resolve/main/bs_polarformer_fp16.onnx), not the WebGPU variants.

```bash
mkdir -p weights input
curl -L --fail https://huggingface.co/bgkb/bs_polarformer/resolve/main/bs_polarformer.onnx -o weights/bs_polarformer.onnx

python inference.py --model_type bs_polarformer \
  --config_path configs/model_bs_polarformer_float16.yaml \
  --input_folder input --store_dir outputs --extract_instrumental \
  --use_onnx --onnx_model_path weights/bs_polarformer.onnx --force_cpu
```

Build an engine on the target GPU, then run:

```bash
python export_to_tensorrt.py --model_type bs_polarformer \
  --config_path configs/model_bs_polarformer_float16.yaml \
  --onnx_path weights/bs_polarformer.onnx --output_path weights/bs_polarformer.engine --fp16

python inference.py --model_type bs_polarformer \
  --config_path configs/model_bs_polarformer_float16.yaml \
  --input_folder input --store_dir outputs --extract_instrumental \
  --use_tensorrt --tensorrt_model_path weights/bs_polarformer.engine
```

For other models substitute the preset, model type and exported ONNX/engine path.

## Export from checkpoint

```bash
python export_to_onnx.py --model_type bs_roformer \
  --config_path configs/viperx/model_bs_roformer_ep_317_sdr_12.9755.yaml \
  --checkpoint_path weights/model_bs_roformer_ep_317_sdr_12.9755.ckpt \
  --output_path weights/bs_roformer.onnx --force_cpu
```

BS Roformer and BS PolarFormer share one spectral core; PolarFormer loads the original PoPE checkpoint strictly before replacing only PoPE attention with the ONNX-compatible implementation adapted from bgkb. PolarFormer defaults to opset 18; other models default to opset 17. Export uses strict checkpoint loading and compares the core output with ONNX Runtime. Only load trusted checkpoints. A prebuilt ONNX is sufficient for accelerated inference.

## Tests and validation

```bash
python -m unittest discover -s tests -v
```

The tests cover preset loading, PolarFormer feature ordering, complex mask / DC handling, and short/partial chunks without downloading weights. Real-weight ONNX parity and TensorRT engine builds on supported NVIDIA hardware are required before release; no benchmark or GPU validation is claimed.

## Credits

Original MSST: [ZFTurbo](https://github.com/ZFTurbo/Music-Source-Separation-Training) (Roman Solovyev); inference presets: [hanruijiang fork](https://github.com/hanruijiang/Music-Source-Separation-Training/tree/cleanup/vocal-inference-only); PolarFormer ONNX conversion and interface: [bgkb/bs_polarformer](https://huggingface.co/bgkb/bs_polarformer); Roformer and PoPE: [lucidrains](https://github.com/lucidrains). MIT license; preserve original notices. Checkpoint links intentionally point to their actual publishers.
