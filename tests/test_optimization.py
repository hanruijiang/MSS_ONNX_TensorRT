import unittest
import copy
import torch
from torch import nn
from models_without_stft.roformer_common import BandSplit, MaskEstimator, Attend
from models_without_stft.optimization import PackedBandSplit, PackedMaskEstimator, optimize_model
from models_without_stft.bs_roformer_no_stft import BSRoformer


class OptimizationTests(unittest.TestCase):
    @torch.no_grad()
    def test_packing_preserves_order_and_dynamic_tokens(self):
        for group_size in (1, 2, 4):
            for batch, time in ((1, 3), (2, 7)):
                dims = (4, 8, 4, 12, 8, 4)
                band = BandSplit(8, dims).eval()
                mask = MaskEstimator(8, dims, 2).eval()
                x = torch.randn(batch, time, sum(dims))
                expected = band(x)
                actual = PackedBandSplit(band, group_size)(x)
                torch.testing.assert_close(actual, expected)
                torch.testing.assert_close(PackedMaskEstimator(mask, group_size)(actual), mask(expected))

    @torch.no_grad()
    def test_attention_matmul(self):
        attn = Attend().eval()
        q, k, v = [torch.randn(2, 3, 5, 8) for _ in range(3)]
        expected = attn(q, k, v)
        attn.use_matmul = True
        torch.testing.assert_close(attn(q, k, v), expected)

    @torch.no_grad()
    def test_polarformer_safe(self):
        model = BSRoformer(dim=8, depth=1, stereo=True, freqs_per_bands=(2, 2, 1),
                          dim_head=4, heads=2, dim_freqs_in=5, stft_n_fft=8,
                          time_transformer_depth=1, freq_transformer_depth=1,
                          use_pope=True).eval()
        x = torch.randn(2, 7, 20)
        expected = model(x)
        optimized, report = optimize_model(model, 'bs_polarformer')
        self.assertTrue(report)
        torch.testing.assert_close(optimized(x), expected, rtol=1e-4, atol=1e-5)
        self.assertEqual(optimize_model(optimized, 'bs_polarformer')[1], [])

    @torch.no_grad()
    def test_conv_bn_and_off(self):
        model = nn.Sequential(nn.Conv2d(2, 4, 3), nn.BatchNorm2d(4), nn.Dropout()).eval()
        x = torch.randn(2, 2, 8, 8)
        expected = model(x)
        self.assertEqual(optimize_model(model, 'segm_models', mode='off')[1], [])
        self.assertIsInstance(model[1], nn.BatchNorm2d)
        optimize_model(model, 'segm_models')
        self.assertIsInstance(model[1], nn.Identity)
        torch.testing.assert_close(model(x), expected)


if __name__ == '__main__':
    unittest.main()
