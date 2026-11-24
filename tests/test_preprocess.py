import unittest
from pathlib import Path
import torch
import numpy as np
from models.preprocess import make_processor
from utils import load_config, demix, chunk_size, MODEL_TYPES

ROOT = Path(__file__).resolve().parents[1]
PRESETS = {
    'mdx23c': 'config_vocals_mdx23c.yaml',
    'segm_models': 'config_vocals_segm_models.yaml',
    'bs_roformer': 'viperx/model_bs_roformer_ep_317_sdr_12.9755.yaml',
    'mel_band_roformer': 'KimberleyJensen/config_vocals_mel_band_roformer_kj.yaml',
    'bs_polarformer': 'model_bs_polarformer_float16.yaml',
}


class SpectralTests(unittest.TestCase):
    def test_presets(self):
        self.assertEqual(set(MODEL_TYPES), set(PRESETS))
        for kind, name in PRESETS.items():
            cfg = load_config(kind, ROOT/'configs'/name)
            self.assertGreater(chunk_size(cfg), 0)
            self.assertEqual(cfg.audio.sample_rate, 44100)

    def test_polarformer_shape_and_dc(self):
        cfg = load_config('bs_polarformer', ROOT/'configs'/PRESETS['bs_polarformer'])
        proc = make_processor(cfg, 'bs_polarformer')
        audio = torch.randn(2, 2, 4096)
        features = proc.stft(audio)
        self.assertEqual(tuple(features.shape), (2, 9, 4100))
        mask = torch.zeros(2, 1, 2050, 9, 2)
        self.assertEqual(tuple(proc.istft(mask).shape), (2, 1, 2, 4096))
        self.assertEqual(proc.istft(mask).abs().max().item(), 0.)
        mask[..., 0] = 1.
        reference = torch.stft(audio.reshape(4, -1), 2048, 512,
                               window=torch.hann_window(2048), return_complex=True)
        reference[:, 0, :] = 0.
        expected = torch.istft(reference, 2048, 512, window=torch.hann_window(2048),
                               length=4096).reshape(2, 1, 2, 4096)
        torch.testing.assert_close(proc.istft(mask), expected)

    def test_short_and_tail_chunks(self):
        cfg = load_config('bs_polarformer', ROOT/'configs'/PRESETS['bs_polarformer'])
        cfg.inference.chunk_size = 4096
        cfg.inference.batch_size = 2
        def zero_mask(x):
            return x.new_zeros((x.shape[0], 1, 2050, x.shape[1], 2))
        for n in (7, 4096, 4321, 8193):
            out = demix(cfg, zero_mask, np.zeros((2, n), np.float32),
                        'cpu', 'bs_polarformer')['vocals']
            self.assertEqual(out.shape, (2, n))
            self.assertTrue(np.isfinite(out).all())
            self.assertEqual(float(np.abs(out).max()), 0.)


if __name__ == '__main__':
    unittest.main()
