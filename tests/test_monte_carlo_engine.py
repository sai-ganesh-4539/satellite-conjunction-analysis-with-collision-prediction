"""Checks for the B-plane probability engine. Run: python -m unittest tests.test_monte_carlo_engine"""

import math
import os
import sys
import unittest

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from monte_carlo_engine import (
    akella_alfriend_pc,
    bplane_miss,
    chan_pc,
    clopper_pearson_interval,
    foster_pc,
    hard_body_radius,
    isotropic_covariance,
    isotropic_pc,
    monte_carlo_pc,
    project_rtn_covariance_to_bplane,
    recommend_cam,
    wilson_interval,
)


class EngineTests(unittest.TestCase):
    def test_centered_disk_matches_closed_form(self):
        # P(||X|| <= R) = 1 - exp(-R^2 / (2 sigma^2)) when the mean is at the origin.
        pc = isotropic_pc(0.0, sigma_m=2.0, radius_m=2.0)
        self.assertAlmostEqual(pc, 1.0 - math.exp(-0.5), places=6)

    def test_foster_matches_isotropic_closed_form(self):
        mu = bplane_miss(30.0)
        cov = isotropic_covariance(15.0)
        quad = foster_pc(mu, cov, radius_m=6.0)
        exact = isotropic_pc(30.0, 15.0, 6.0)
        self.assertAlmostEqual(quad, exact, delta=2e-3)

    def test_chan_series_matches_chi_square_sum(self):
        mu = bplane_miss(12.0)
        cov = isotropic_covariance(5.0)
        series = chan_pc(mu, cov, radius_m=3.0, n_terms=40)
        exact = isotropic_pc(12.0, 5.0, 3.0)
        self.assertAlmostEqual(series, exact, delta=1e-3)

    def test_akella_is_close_when_hard_body_is_small(self):
        mu = bplane_miss(10.0)
        cov = isotropic_covariance(40.0)
        approx = akella_alfriend_pc(mu, cov, radius_m=1.5)
        exact = isotropic_pc(10.0, 40.0, 1.5)
        self.assertLess(abs(approx - exact), 5e-4)
        self.assertGreater(exact, 0.0)

    def test_monte_carlo_interval_covers_reference(self):
        mu = bplane_miss(20.0)
        cov = isotropic_covariance(10.0)
        exact = isotropic_pc(20.0, 10.0, 4.0)
        mc = monte_carlo_pc(mu, cov, 4.0, n_samples=250_000, seed=11)
        lo, hi = mc["wilson_95"]
        self.assertLessEqual(lo, exact)
        self.assertLessEqual(exact, hi)
        clo, chi = clopper_pearson_interval(mc["hits"], mc["n_samples"])
        self.assertLessEqual(clo, exact)
        self.assertLessEqual(exact, chi)
        self.assertGreater(mc["elapsed_s"], 0.0)

    def test_exclusion_radius_and_bplane_projection(self):
        self.assertEqual(hard_body_radius(2.0, 5.0), 7.0)
        cov = np.diag([4.0, 9.0, 16.0])
        # Velocity along-track: the B-plane drops the T component.
        projected = project_rtn_covariance_to_bplane(cov, v_rel_rtn=np.array([0.0, 1.0, 0.0]))
        self.assertEqual(projected.shape, (2, 2))
        # Radial variance is preserved; normal variance is preserved.
        self.assertAlmostEqual(projected[0, 0], 4.0, places=6)
        self.assertAlmostEqual(projected[1, 1], 16.0, places=6)

    def test_intervals_are_ordered(self):
        lo, hi = wilson_interval(3, 100)
        self.assertLess(lo, 0.03)
        self.assertGreater(hi, 0.03)
        lo, hi = clopper_pearson_interval(0, 50)
        self.assertEqual(lo, 0.0)
        self.assertGreater(hi, 0.0)

    def test_cam_uses_secular_along_track_drift(self):
        rec = recommend_cam(miss_m=1000.0, pc=1e-3, safe_miss_m=5000.0, horizon_s=86400.0)
        self.assertAlmostEqual(rec["delta_v_m_s"], 4000.0 / (3.0 * 86400.0), places=9)
        self.assertEqual(rec["axis"], "along-track")
        self.assertEqual(rec["safety"], "CRITICAL / ACTION REQUIRED")
        idle = recommend_cam(miss_m=8000.0, pc=1e-8)
        self.assertEqual(idle["delta_v_m_s"], 0.0)
        self.assertEqual(idle["safety"], "SAFE")

    def test_no_nan_on_zero_miss(self):
        pc = foster_pc(bplane_miss(0.0), isotropic_covariance(3.0), 1.0)
        self.assertTrue(np.isfinite(pc))


if __name__ == "__main__":
    unittest.main()
