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

    def test_onnx_is_the_default_and_only_backend(self):
        args = parse_args(self.args)
        self.assertEqual(args.onnx_model_path, 'model.onnx')
        for option in ('use_onnx', 'use_tensorrt', 'use_compile',
                       'tensorrt_model_path', 'start_check_point'):
            with self.subTest(option=option), self.assertRaisesRegex(ValueError, 'Unknown inference arguments'):
                parse_args(dict(self.args, **{option: True}))

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
            session.run.side_effect = lambda _, inputs: [inputs['x']]
            backend = ONNXBackend('model.onnx', 'cpu')
            session_factory.assert_called_once_with('model.onnx', providers=['CPUExecutionProvider'])
            x = torch.ones(2, 4)
            torch.testing.assert_close(backend(x), x)
            self.assertEqual(session.run.call_args.args[1]['x'].dtype, np.float32)
            with self.assertRaises(ValueError):
                backend.validate_shape((2, 5))


if __name__ == '__main__':
    unittest.main()
