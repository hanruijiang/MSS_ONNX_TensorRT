"""Shared standard ONNX opset for model generation."""

DEFAULT_ONNX_OPSET_VERSION = 23


def require_onnx_support(opset_version):
    import onnx

    if not isinstance(opset_version, int) or opset_version < 1:
        raise ValueError(f'Expected a positive integer ONNX opset (got {opset_version})')
    supported = onnx.defs.onnx_opset_version()
    if supported < opset_version:
        raise RuntimeError(f'The installed onnx package supports through opset {supported}. '
                           f'The selected opset is {opset_version}.')


def require_opset(version, target, stage):
    if version != target:
        raise ValueError(f'{stage} requires ai.onnx opset {target} (got {version})')


def require_model_opset(model, target, stage):
    version = next((entry.version for entry in model.opset_import if entry.domain in ('', 'ai.onnx')), None)
    if version is None:
        raise ValueError(f'{stage} does not declare an ai.onnx opset')
    if target is not None:
        require_opset(version, target, stage)
    return version
