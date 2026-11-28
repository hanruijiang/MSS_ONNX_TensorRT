import unittest
from types import SimpleNamespace
from unittest.mock import patch

import torch

from inference import parse_args
from utils import ONNXBackend, inference_batch_size


class ONNXInferenceTests(unittest.TestCase):
    def setUp(self):
        self.args = dict(config_path='config.yaml', input_folder='input',
                         store_dir='output', onnx_model_path='model.onnx')

    def test_unknown_inference_argument(self):
        with self.assertRaisesRegex(ValueError, 'Unknown inference arguments'):
            parse_args(dict(self.args, unknown_option=True))

    def test_required_onnx_path(self):
        with self.assertRaisesRegex(ValueError, '--onnx_model_path is required'):
            parse_args(dict(self.args, onnx_model_path=None))
        with patch('sys.argv', ['inference.py', '--config_path', 'config.yaml',
                                '--input_folder', 'input', '--store_dir', 'output']):
            with self.assertRaises(SystemExit):
                parse_args()

    def test_fixed_and_dynamic_batch_sizes(self):
        config = SimpleNamespace(inference=SimpleNamespace(batch_size=3))
        self.assertEqual(inference_batch_size(config, SimpleNamespace(fixed_batch=2)), 2)
        self.assertEqual(inference_batch_size(config, SimpleNamespace(fixed_batch=None)), 3)
        config.inference.batch_size = 0
        with self.assertRaises(ValueError):
            inference_batch_size(config, SimpleNamespace(fixed_batch=None))

    def test_cpu_onnx_session(self):
        import onnxruntime as ort
        import numpy as np

        with patch.object(ort, 'InferenceSession') as session_factory:
            session = session_factory.return_value
            session.get_inputs.return_value = [SimpleNamespace(name='x', type='tensor(float)', shape=[None, 4])]
            session.get_outputs.return_value = [SimpleNamespace(name='y', type='tensor(float)')]
            binding = session.io_binding.return_value
            binding.get_outputs.return_value = [SimpleNamespace(numpy=lambda: np.ones((2, 4), dtype=np.float32))]
            backend = ONNXBackend('model.onnx', 'cpu')
            session_factory.assert_called_once_with('model.onnx', providers=['CPUExecutionProvider'])
            x = torch.ones(2, 4)
            torch.testing.assert_close(backend(x), x)
            self.assertEqual(binding.bind_cpu_input.call_args.args[0], 'x')
            self.assertEqual(binding.bind_cpu_input.call_args.args[1].dtype, np.float32)
            binding.bind_output.assert_called_once_with('y')
            session.run_with_iobinding.assert_called_once_with(binding)
            with self.assertRaises(ValueError):
                backend.validate_shape((2, 5))


if __name__ == '__main__':
    unittest.main()
