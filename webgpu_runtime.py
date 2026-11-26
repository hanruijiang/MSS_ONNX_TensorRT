"""Native ORT WebGPU plugin EP (not browser/JSEP).

The audio pipeline owns CPU tensors. One upload and one readback are still
needed per chunk; static device allocations are reused without rebinding.
"""
from threading import Lock
import numpy as np

_registration_lock = Lock()


def create_session(path, *, layout='NCHW', cache='bucket', capture=False, strict=False,
                   profile=False):
    import onnxruntime as ort
    if layout not in ('NCHW', 'NHWC') or cache not in ('disabled', 'lazyRelease', 'simple', 'bucket'):
        raise ValueError('Invalid WebGPU layout or cache mode')
    opts = ort.SessionOptions()
    opts.enable_profiling = profile
    opts.enable_mem_pattern = not capture
    if strict:
        opts.add_session_config_entry('session.disable_cpu_ep_fallback', '1')
    options = {'preferredLayout': layout, 'enableGraphCapture': str(int(capture)),
               'storageBufferCacheMode': cache, 'uniformBufferCacheMode': cache,
               'defaultBufferCacheMode': cache, 'queryResolveBufferCacheMode': cache}
    try:
        import onnxruntime_ep_webgpu as plugin
    except ImportError as exc:
        raise RuntimeError('Install compatible onnxruntime and onnxruntime-ep-webgpu packages') from exc
    name = plugin.get_ep_name()
    with _registration_lock:
        devices = [d for d in ort.get_ep_devices() if d.ep_name == name]
        if not devices:
            ort.register_execution_provider_library('mss_webgpu', plugin.get_library_path())
            devices = [d for d in ort.get_ep_devices() if d.ep_name == name]
    if not devices:
        raise RuntimeError('No native WebGPU EP device available')
    opts.add_provider_for_devices([devices[0]], options)
    session = ort.InferenceSession(str(path), sess_options=opts)
    if name not in session.get_providers():
        raise RuntimeError('Native WebGPU EP failed to initialize')
    session.disable_fallback()
    return session


def static_shape(meta):
    if not meta.shape or any(not isinstance(d, int) or d <= 0 for d in meta.shape):
        raise ValueError('WebGPU fixed I/O binding requires static input and output shapes')
    return tuple(meta.shape)


class StaticBinding:
    """One session-owned device input/output allocation per static spectral core."""
    def __init__(self, session, inp, out, dtypes):
        self.session = session
        self.shape = static_shape(inp)
        output_shape = static_shape(out)
        self.dtype = dtypes[inp.type]
        if not hasattr(session, 'create_ortvalue_from_shape_and_type'):
            raise RuntimeError('This ORT version lacks session-owned WebGPU OrtValue allocation')
        allocate = session.create_ortvalue_from_shape_and_type
        self.input_value = allocate(self.shape, self.dtype, 'webgpu', 0)
        self.output_value = allocate(output_shape, dtypes[out.type], 'webgpu', 0)
        self.binding = session.io_binding()
        self.binding.bind_ortvalue_input(inp.name, self.input_value)
        self.binding.bind_ortvalue_output(out.name, self.output_value)

    def run(self, values):
        if values.shape != self.shape:
            raise ValueError('Cannot change static WebGPU binding shape')
        self.input_value.update_inplace(np.ascontiguousarray(values, dtype=self.dtype))
        self.binding.synchronize_inputs()
        self.session.run_with_iobinding(self.binding)
        self.binding.synchronize_outputs()
        return self.binding.copy_outputs_to_cpu()[0]
