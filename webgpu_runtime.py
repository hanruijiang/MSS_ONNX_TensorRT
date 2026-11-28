"""Native WebGPU sessions and reusable device buffers."""
from threading import Lock

_registration_lock = Lock()


def create_session(path, *, layout='NCHW', cache='bucket', capture=False, strict=False,
                   profile=False):
    import onnxruntime as ort
    opts = ort.SessionOptions()
    opts.enable_profiling = profile
    opts.enable_mem_pattern = not capture
    if strict:
        opts.add_session_config_entry('session.disable_cpu_ep_fallback', '1')
    options = {'preferredLayout': layout, 'enableGraphCapture': str(int(capture)),
               'storageBufferCacheMode': cache, 'uniformBufferCacheMode': cache,
               'defaultBufferCacheMode': cache, 'queryResolveBufferCacheMode': cache}
    import onnxruntime_ep_webgpu as plugin
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


def _static_shape(meta):
    shape = tuple(meta.shape or ())
    if shape and all(isinstance(dim, int) and dim > 0 for dim in shape):
        return shape


def create_binding(session, inp, out, dtypes, *, capture):
    input_shape, output_shape = _static_shape(inp), _static_shape(out)
    if not input_shape or not output_shape:
        if capture:
            raise ValueError('WebGPU graph capture requires static input and output shapes')
        return None
    if not hasattr(session, 'create_ortvalue_from_shape_and_type'):
        if capture:
            raise RuntimeError('WebGPU graph capture requires ONNX Runtime with session-owned OrtValue support')
        return None
    return StaticBinding(session, inp, out, dtypes, input_shape, output_shape)


class StaticBinding:
    """One session-owned device input/output allocation per static spectral core."""
    def __init__(self, session, inp, out, dtypes, input_shape, output_shape):
        self.session = session
        self._lock = Lock()
        allocate = session.create_ortvalue_from_shape_and_type
        self.input_value = allocate(input_shape, dtypes[inp.type], 'webgpu', 0)
        self.output_value = allocate(output_shape, dtypes[out.type], 'webgpu', 0)
        self.binding = session.io_binding()
        self.binding.bind_ortvalue_input(inp.name, self.input_value)
        self.binding.bind_ortvalue_output(out.name, self.output_value)

    def run(self, values):
        with self._lock:
            self.input_value.update_inplace(values)
            self.binding.synchronize_inputs()
            self.session.run_with_iobinding(self.binding)
            self.binding.synchronize_outputs()
            return self.binding.copy_outputs_to_cpu()[0]
