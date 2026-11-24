"""TensorRT 10 runtime with dynamic shapes and PyTorch CUDA buffers."""
import torch


class TensorRTBackend:
    def __init__(self, path, device):
        import tensorrt as trt
        self.trt = trt
        self.device = torch.device(device)
        if self.device.type != 'cuda':
            raise ValueError('TensorRT requires CUDA')
        torch.cuda.set_device(self.device)
        self.logger = trt.Logger(trt.Logger.WARNING)
        self.runtime = trt.Runtime(self.logger)
        with open(path, 'rb') as f:
            self.engine = self.runtime.deserialize_cuda_engine(f.read())
        if self.engine is None:
            raise RuntimeError('Unable to deserialize TensorRT engine')
        self.context = self.engine.create_execution_context()
        if self.context is None:
            raise RuntimeError('Unable to create TensorRT execution context')
        names = [self.engine.get_tensor_name(i) for i in range(self.engine.num_io_tensors)]
        ins = [n for n in names if self.engine.get_tensor_mode(n) == trt.TensorIOMode.INPUT]
        outs = [n for n in names if self.engine.get_tensor_mode(n) == trt.TensorIOMode.OUTPUT]
        if len(ins) != 1 or len(outs) != 1:
            raise ValueError('Expected one spectral input and one output')
        self.input, self.output = ins[0], outs[0]
        for name in (self.input, self.output):
            self.dtype(name)
            if self.engine.get_tensor_location(name) != trt.TensorLocation.DEVICE:
                raise ValueError('Expected device-resident TensorRT IO')
            if self.engine.get_tensor_format(name) != trt.TensorFormat.LINEAR:
                raise ValueError('Expected linear TensorRT IO format')
        self.input_shape = tuple(self.engine.get_tensor_shape(self.input))
        if not self.input_shape:
            raise ValueError('Expected a batched spectral input')
        self.fixed_batch = self.input_shape[0] if self.input_shape[0] > 0 else None
        self.min_batch = self.fixed_batch or 1
        self.max_batch = self.fixed_batch
        self.profile = None
        if any(d < 0 for d in self.input_shape):
            lo, _, hi = self.engine.get_tensor_profile_shape(self.input, 0)
            self.profile = (tuple(lo), tuple(hi))
            self.min_batch, self.max_batch = int(lo[0]), int(hi[0])
            if self.min_batch == self.max_batch:
                self.fixed_batch = self.min_batch

    def validate_shape(self, shape):
        if len(shape) != len(self.input_shape) or any(
            expected >= 0 and actual != expected
            for actual, expected in zip(shape, self.input_shape)
        ):
            raise ValueError(f'TensorRT input shape {shape} does not match {self.input_shape}')
        if self.profile is not None:
            lo, hi = self.profile
            if any(not a <= x <= b for x, a, b in zip(shape, lo, hi)):
                raise ValueError(f'Input shape {shape} is outside TensorRT profile {lo}..{hi}')

    def dtype(self, name):
        trt = self.trt
        dtypes = {trt.float32: torch.float32, trt.float16: torch.float16}
        dtype = self.engine.get_tensor_dtype(name)
        if dtype not in dtypes:
            raise ValueError(f'Unsupported TensorRT dtype for {name}: {dtype}')
        return dtypes[dtype]

    def __call__(self, x):
        self.validate_shape(tuple(x.shape))
        x = x.to(device=self.device, dtype=self.dtype(self.input)).contiguous()
        if not self.context.set_input_shape(self.input, tuple(x.shape)):
            raise ValueError(f'Input shape {tuple(x.shape)} is outside the TensorRT profile')
        shape = tuple(self.context.get_tensor_shape(self.output))
        if any(d < 0 for d in shape):
            raise RuntimeError(f'Unresolved TensorRT output shape: {shape}')
        y = torch.empty(shape, device=self.device, dtype=self.dtype(self.output))
        if not self.context.set_tensor_address(self.input, x.data_ptr()):
            raise RuntimeError('Failed to bind TensorRT input')
        if not self.context.set_tensor_address(self.output, y.data_ptr()):
            raise RuntimeError('Failed to bind TensorRT output')
        stream = torch.cuda.current_stream(self.device)
        if not self.context.execute_async_v3(stream.cuda_stream):
            raise RuntimeError('TensorRT execution failed')
        stream.synchronize()
        return y
