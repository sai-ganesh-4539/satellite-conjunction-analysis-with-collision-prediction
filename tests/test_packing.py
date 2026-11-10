"""Causal packing must hide padding from real CDM steps."""

import os
import sys
import unittest

import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from model import pack_real_first, RetNetForCollisionPrediction


class PackingTests(unittest.TestCase):
    def test_real_steps_move_to_the_front_in_order(self):
        x = torch.tensor([
            [[0.0, 0.0], [1.0, 2.0], [3.0, 4.0]],
            [[5.0, 6.0], [0.0, 0.0], [7.0, 8.0]],
        ])
        packed, lengths = pack_real_first(x)
        self.assertEqual(lengths.tolist(), [2, 2])
        self.assertTrue(torch.equal(packed[0, 0], x[0, 1]))
        self.assertTrue(torch.equal(packed[0, 1], x[0, 2]))
        self.assertTrue(torch.equal(packed[1, 0], x[1, 0]))
        self.assertTrue(torch.equal(packed[1, 1], x[1, 2]))

    def test_output_does_not_depend_on_padding_width(self):
        torch.manual_seed(0)
        model = RetNetForCollisionPrediction(input_dim=4, d_model=16, n_heads=4, n_layers=1, max_len=8)
        model.eval()
        real = torch.randn(2, 3, 4)
        left = torch.cat([torch.zeros(2, 2, 4), real], dim=1)
        right = torch.cat([real, torch.zeros(2, 4, 4)], dim=1)
        with torch.no_grad():
            a = model(left)
            b = model(right)
        self.assertTrue(torch.allclose(a, b, atol=1e-5))
        self.assertFalse(torch.isnan(a).any())


if __name__ == "__main__":
    unittest.main()
