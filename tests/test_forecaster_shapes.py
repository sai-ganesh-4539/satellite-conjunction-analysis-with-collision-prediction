"""Shape and finiteness checks that do not need the CDM file."""

import os
import sys
import unittest

import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from collision_forecaster import MultiTargetForecaster, multitask_loss, summarize_71
from run_pipeline import FocalLoss


class ForecasterShapeTests(unittest.TestCase):
    def test_summary_has_71_features(self):
        feat = np.arange(12, dtype=np.float32).reshape(3, 4)
        # 4 channels * 5 stats + count = 21, so build 14 channels to hit 71.
        feat = np.random.default_rng(0).normal(size=(5, 14)).astype(np.float32)
        s = summarize_71(feat)
        self.assertEqual(s.shape, (71,))
        self.assertTrue(np.isfinite(s).all())

    def test_multitask_forward_is_finite(self):
        torch.manual_seed(0)
        model = MultiTargetForecaster("retnet", input_dim=14, d_model=32, n_heads=4, n_layers=1, max_len=8)
        x = torch.zeros(4, 8, 14)
        x[:, -3:, :] = torch.randn(4, 3, 14)
        out = model(x)
        y = torch.tensor([0.0, 1.0, 0.0, 1.0])
        loss = multitask_loss(out, torch.randn(4), torch.randn(4), y, FocalLoss())
        self.assertTrue(torch.isfinite(loss))
        step, _state = model.step(x[:, -1, :], timestep=0, states=None)
        self.assertEqual(tuple(step["cls"].shape), (4, 1))


if __name__ == "__main__":
    unittest.main()
