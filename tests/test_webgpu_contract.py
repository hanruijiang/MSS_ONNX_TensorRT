import unittest
from types import SimpleNamespace
from unittest.mock import MagicMock
import numpy as np
from webgpu_runtime import StaticBinding, static_shape


class WebGPUContractTests(unittest.TestCase):
    def test_dynamic_binding_rejected(self):
        with self.assertRaises(ValueError):
            static_shape(SimpleNamespace(shape=[None, 4]))

    def test_fixed_buffers_reused(self):
        session = MagicMock()
        inp = SimpleNamespace(name='input', shape=[1, 4], type='tensor(float)')
        out = SimpleNamespace(name='output', shape=[1, 4], type='tensor(float)')
        runner = StaticBinding(session, inp, out, {'tensor(float)': np.float32})
        for i in range(2):
            runner.run(np.full((1, 4), i, dtype=np.float32))
        self.assertEqual(session.create_ortvalue_from_shape_and_type.call_count, 2)
        self.assertEqual(session.io_binding.call_count, 1)
        self.assertEqual(runner.binding.bind_ortvalue_input.call_count, 1)
        self.assertEqual(runner.binding.bind_ortvalue_output.call_count, 1)
        self.assertEqual(runner.input_value.update_inplace.call_count, 2)
        self.assertEqual(session.run_with_iobinding.call_count, 2)
        with self.assertRaises(ValueError):
            runner.run(np.ones((2, 4), dtype=np.float32))


if __name__ == '__main__':
    unittest.main()
