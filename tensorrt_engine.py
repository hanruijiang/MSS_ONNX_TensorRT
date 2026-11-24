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
        shape = tuple(self.engine.get_tensor_shape(self.input))
        self.fixed_batch = shape[0] if shape[0] > 0 else None
        self.max_batch = self.fixed_batch
        if any(d < 0 for d in shape):
            lo, _, hi = self.engine.get_tensor_profile_shape(self.input, 0)
            self.max_batch = int(hi[0])
            if lo[0] == hi[0]:
                self.fixed_batch = int(lo[0])

    def dtype(self, name):
        trt = self.trt
        return {trt.float32: torch.float32, trt.float16: torch.float16}[self.engine.get_tensor_dtype(name)]

    def __call__(self, x):
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
