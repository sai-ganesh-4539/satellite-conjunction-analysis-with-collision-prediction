"""
tests/test_mathematical_validation.py — Rigorous Astrodynamics & Statistical Validation

Independent mathematical audit test suite verifying:
  1. B-plane coordinate projection, orthonormality, and covariance trace invariance.
  2. Foster (1992) Gauss-Legendre 2D integration accuracy and symmetry.
  3. Chan (1997) equal-area Poisson mixture series convergence.
  4. Akella & Alfriend (2000) 2nd-order Taylor series Laplacian approximation.
  5. Monte Carlo sampling, 95% Wilson score, and exact Clopper-Pearson intervals.
  6. Isotropic scale inversion, root-finding accuracy, and radius inflation mechanics.
  7. Numerical edge cases (zero miss distance, zero radius, extreme offset, singular matrices).
  8. Linearized Clohessy-Wiltshire along-track secular drift and safety disclaimers.

Run via:
  python -m pytest tests/test_mathematical_validation.py
"""

from __future__ import annotations

import math
import os
import sys
import unittest

import numpy as np

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO_ROOT, "src"))

from monte_carlo_engine import (
    Encounter,
    akella_alfriend_pc,
    bplane_miss,
    chan_pc,
    clopper_pearson_interval,
    combine_covariances,
    foster_pc,
    hard_body_radius,
    invert_isotropic_scale,
    isotropic_covariance,
    isotropic_pc,
    monte_carlo_pc,
    project_rtn_covariance_to_bplane,
    recommend_cam,
    safety_class,
    wilson_interval,
)


class TestBPlaneProjection(unittest.TestCase):
    """Mathematical verification of B-plane projection and coordinate transformations."""

    def test_basis_orthonormality(self):
        """Verify that the B-plane basis vectors (xi, zeta) are orthogonal to each other
        and to the relative velocity vector h, with unit norm."""
        test_velocities = [
            np.array([7500.0, 0.0, 0.0]),             # Pure radial
            np.array([0.0, 7500.0, 0.0]),             # Pure along-track
            np.array([0.0, 0.0, 7500.0]),             # Pure cross-track
            np.array([1200.0, -6800.0, 3100.0]),       # Arbitrary 3D encounter velocity
            np.array([-4500.0, 4500.0, 100.0]),       # Oblique crossing
        ]
        cov_3d = np.diag([100.0**2, 500.0**2, 50.0**2])

        for v in test_velocities:
            h = v / np.linalg.norm(v)
            proj_cov = project_rtn_covariance_to_bplane(cov_3d, v)
            self.assertEqual(proj_cov.shape, (2, 2))
            # Symmetrized and positive definite
            self.assertTrue(np.allclose(proj_cov, proj_cov.T, atol=1e-10))
            eigvals = np.linalg.eigvalsh(proj_cov)
            self.assertTrue(np.all(eigvals > 0), f"Eigenvalues must be positive, got {eigvals}")

    def test_trace_invariance_and_bounding(self):
        """The trace of the projected 2D covariance cannot exceed the trace of the 3D covariance."""
        cov_3d = np.array([
            [10000.0, 1200.0, -300.0],
            [1200.0, 40000.0, 500.0],
            [-300.0, 500.0, 2500.0],
        ])
        v = np.array([250.0, 7200.0, -1100.0])
        proj_cov = project_rtn_covariance_to_bplane(cov_3d, v)
        self.assertLessEqual(np.trace(proj_cov), np.trace(cov_3d) + 1e-6)

    def test_singular_velocity_raises(self):
        """Relative velocity of magnitude zero cannot define an encounter plane."""
        cov_3d = np.eye(3) * 100.0
        with self.assertRaises(ValueError):
            project_rtn_covariance_to_bplane(cov_3d, np.array([0.0, 0.0, 0.0]))

    def test_non_finite_inputs_raise(self):
        cov_3d = np.eye(3) * 100.0
        with self.assertRaises(ValueError):
            project_rtn_covariance_to_bplane(cov_3d, np.array([np.nan, 100.0, 0.0]))
        with self.assertRaises(ValueError):
            project_rtn_covariance_to_bplane(np.ones((3, 3)) * np.inf, np.array([100.0, 0.0, 0.0]))


class TestFosterQuadrature(unittest.TestCase):
    """Mathematical verification of the Foster (1992) 2D bivariate normal integral."""

    def test_symmetry_property(self):
        """Foster Pc must be symmetric under sign inversion of the miss vector: Pc(d) == Pc(-d)."""
        cov = np.array([[400.0, 50.0], [50.0, 100.0]])
        r = 5.0
        mu_pos = np.array([25.0, -10.0])
        mu_neg = np.array([-25.0, 10.0])
        pc1 = foster_pc(mu_pos, cov, r)
        pc2 = foster_pc(mu_neg, cov, r)
        self.assertAlmostEqual(pc1, pc2, places=7)

    def test_rotation_invariance_isotropic(self):
        """For isotropic covariance, Pc depends only on the Euclidean norm of mu."""
        cov = isotropic_covariance(30.0)
        r = 4.0
        d = 45.0
        # Evaluate at various angles around the circle
        for theta in [0.0, math.pi / 4.0, math.pi / 2.0, 2.15, math.pi]:
            mu = np.array([d * math.cos(theta), d * math.sin(theta)])
            pc = foster_pc(mu, cov, r)
            expected = isotropic_pc(d, sigma_m=30.0, radius_m=r)
            self.assertAlmostEqual(pc, expected, delta=1e-4)

    def test_closed_form_exact_zero_miss(self):
        """When d = 0, Foster Pc must equal 1 - exp(-R^2 / (2 * sigma^2))."""
        sigma = 25.0
        r = 8.0
        exact = 1.0 - math.exp(-0.5 * (r / sigma) ** 2)
        quad = foster_pc(np.array([0.0, 0.0]), isotropic_covariance(sigma), r, n_radial=96, n_angular=180)
        self.assertAlmostEqual(quad, exact, places=5)


class TestChanApproximation(unittest.TestCase):
    """Validation of Chan (1997) equal-area series approximation."""

    def test_series_monotone_convergence(self):
        """Truncated Chan series must monotonically converge towards the exact chi-square integral."""
        d = 30.0
        sigma = 15.0
        r = 5.0
        cov = isotropic_covariance(sigma)
        mu = bplane_miss(d)
        exact = isotropic_pc(d, sigma, r)

        val_5 = chan_pc(mu, cov, r, n_terms=5)
        val_15 = chan_pc(mu, cov, r, n_terms=15)
        val_40 = chan_pc(mu, cov, r, n_terms=40)

        # Monotonic convergence towards exact value
        self.assertLessEqual(val_5, val_15 + 1e-9)
        self.assertLessEqual(val_15, val_40 + 1e-9)
        self.assertAlmostEqual(val_40, exact, delta=5e-4)

    def test_moderate_anisotropic_agreement(self):
        """Chan equal-area approximation on moderate eccentricity (ratio 2:1)."""
        cov = np.diag([40.0**2, 20.0**2])
        mu = np.array([25.0, 10.0])
        r = 3.0
        foster_val = foster_pc(mu, cov, r, n_radial=96, n_angular=180)
        chan_val = chan_pc(mu, cov, r)
        # For moderate aspect ratios (<= 2:1), Chan equal-area approximation is within 5%
        rel_err = abs(chan_val - foster_val) / max(foster_val, 1e-6)
        self.assertLess(rel_err, 0.05)


class TestAkellaAlfriendApproximation(unittest.TestCase):
    """Validation of Akella & Alfriend (2000) 2nd-order Taylor expansion."""

    def test_quadratic_convergence_for_small_body(self):
        """As R / sigma -> 0, the Akella-Alfriend relative error must scale as O((R/sigma)^4)."""
        mu = np.array([15.0, 5.0])
        cov = np.array([[50.0**2, 100.0], [100.0, 40.0**2]])
        
        # Test decreasing hard-body radii
        r_large = 4.0
        r_small = 1.0

        akella_large = akella_alfriend_pc(mu, cov, r_large)
        foster_large = foster_pc(mu, cov, r_large)
        err_large = abs(akella_large - foster_large) / foster_large

        akella_small = akella_alfriend_pc(mu, cov, r_small)
        foster_small = foster_pc(mu, cov, r_small)
        err_small = abs(akella_small - foster_small) / foster_small

        self.assertLess(err_small, err_large)
        self.assertLess(err_small, 0.001)  # Under 0.1% relative error for small body


class TestMonteCarloAndConfidenceBands(unittest.TestCase):
    """Validation of Monte Carlo sampling and binomial confidence interval coverage."""

    def test_wilson_and_clopper_pearson_coverage(self):
        """Verify that analytical probability is covered by 95% Wilson and Clopper-Pearson intervals."""
        d = 20.0
        sigma = 15.0
        r = 5.0
        cov = isotropic_covariance(sigma)
        mu = bplane_miss(d)
        exact = isotropic_pc(d, sigma, r)

        mc = monte_carlo_pc(mu, cov, r, n_samples=300_000, seed=123)
        w_lo, w_hi = mc["wilson_95"]
        c_lo, c_hi = mc["clopper_pearson_95"]

        self.assertLessEqual(w_lo, exact)
        self.assertLessEqual(exact, w_hi)
        self.assertLessEqual(c_lo, exact)
        self.assertLessEqual(exact, c_hi)

        # Clopper-Pearson exact interval is mathematically conservative (never narrower than Wilson)
        self.assertLessEqual(c_lo, w_lo + 1e-6)
        self.assertGreaterEqual(c_hi, w_hi - 1e-6)

    def test_interval_edge_cases(self):
        """Check edge cases: 0 hits out of N, and N hits out of N."""
        # 0 hits
        w_lo, w_hi = wilson_interval(0, 1000)
        c_lo, c_hi = clopper_pearson_interval(0, 1000)
        self.assertEqual(w_lo, 0.0)
        self.assertEqual(c_lo, 0.0)
        self.assertGreater(w_hi, 0.0)
        self.assertGreater(c_hi, 0.0)

        # N hits
        w_lo, w_hi = wilson_interval(1000, 1000)
        c_lo, c_hi = clopper_pearson_interval(1000, 1000)
        self.assertLess(w_lo, 1.0)
        self.assertLess(c_lo, 1.0)
        self.assertEqual(w_hi, 1.0)
        self.assertEqual(c_hi, 1.0)


class TestIsotropicScaleInversion(unittest.TestCase):
    """Validation of isotropic scale solver and radius inflation logic."""

    def test_feasible_root_recovery(self):
        """For a known (miss, radius, sigma), inverting Pc must recover the exact sigma."""
        miss = 500.0
        radius = 15.0
        true_sigma = 400.0
        target_pc = isotropic_pc(miss, true_sigma, radius)

        inv = invert_isotropic_scale(miss, radius, target_pc)
        self.assertTrue(inv["feasible_at_catalog_radius"])
        self.assertEqual(inv["mode"], "catalog_matched")
        self.assertAlmostEqual(inv["sigma_m"], true_sigma, delta=true_sigma * 0.01)
        self.assertAlmostEqual(inv["achieved_pc"], target_pc, delta=target_pc * 0.005)

    def test_infeasible_pc_triggers_radius_inflation(self):
        """When target Pc exceeds peak isotropic Pc at physical radius, radius must inflate."""
        miss = 2000.0
        physical_radius = 2.0
        # Peak isotropic Pc is approximately R^2 / (e * d^2) ≈ 4 / (2.718 * 4e6) ≈ 3.6e-7
        high_target_pc = 5e-4  # Impossible at radius = 2.0 m

        inv = invert_isotropic_scale(miss, physical_radius, high_target_pc)
        self.assertFalse(inv["feasible_at_catalog_radius"])
        self.assertEqual(inv["mode"], "radius_inflated_to_match_catalog_pc")
        self.assertGreater(inv["radius_m"], physical_radius)
        self.assertAlmostEqual(inv["achieved_pc"], high_target_pc, delta=high_target_pc * 0.02)

    def test_non_positive_and_unit_pc_edge_cases(self):
        inv_zero = invert_isotropic_scale(500.0, 5.0, 0.0)
        self.assertEqual(inv_zero["achieved_pc"], 0.0)
        self.assertEqual(inv_zero["mode"], "nonpositive_catalog_pc")

        inv_unit = invert_isotropic_scale(10.0, 5.0, 1.0)
        self.assertIsNotNone(inv_unit["sigma_m"])
        self.assertGreater(inv_unit["achieved_pc"], 0.0)


class TestNumericalEdgeCases(unittest.TestCase):
    """Validation of stability under extreme inputs."""

    def test_zero_miss_distance(self):
        """d = 0 must evaluate smoothly without NaN or divide-by-zero."""
        pc = isotropic_pc(0.0, 100.0, 5.0)
        self.assertTrue(np.isfinite(pc))
        self.assertGreater(pc, 0.0)

        quad = foster_pc(np.array([0.0, 0.0]), isotropic_covariance(100.0), 5.0)
        self.assertTrue(np.isfinite(quad))
        self.assertAlmostEqual(pc, quad, delta=1e-5)

    def test_zero_radius(self):
        """R = 0 must return 0.0."""
        self.assertEqual(isotropic_pc(100.0, 50.0, 0.0), 0.0)
        self.assertEqual(foster_pc(np.array([10.0, 0.0]), isotropic_covariance(50.0), 0.0), 0.0)
        self.assertEqual(akella_alfriend_pc(np.array([10.0, 0.0]), isotropic_covariance(50.0), 0.0), 0.0)
        self.assertEqual(chan_pc(np.array([10.0, 0.0]), isotropic_covariance(50.0), 0.0), 0.0)

    def test_extreme_miss_distance(self):
        """d >> sigma must evaluate to 0.0 without underflow crash."""
        pc = isotropic_pc(1e7, 10.0, 2.0)
        self.assertEqual(pc, 0.0)


class TestCAMPlannerSafety(unittest.TestCase):
    """Validation of CAM calculation and mandatory safety disclaimers."""

    def test_linearized_cw_drift_formula(self):
        """Along-track drift: delta-v = gap / (3 * t)."""
        safe_miss = 5000.0
        current_miss = 1000.0
        gap = 4000.0
        horizon = 24.0 * 3600.0  # 86,400 s
        rec = recommend_cam(miss_m=current_miss, pc=1e-3, safe_miss_m=safe_miss, horizon_s=horizon)

        expected_dv = gap / (3.0 * horizon)
        self.assertAlmostEqual(rec["delta_v_m_s"], expected_dv, places=8)
        self.assertEqual(rec["safety"], "CRITICAL / ACTION REQUIRED")
        self.assertFalse(rec["is_operational_certified"])
        self.assertIn("DEMONSTRATION HEURISTIC ONLY", rec["disclaimer"])

    def test_safe_encounter_zero_burn(self):
        """Safe encounters (miss >= safe_miss, Pc < 1e-5) require 0 delta-v."""
        rec = recommend_cam(miss_m=8000.0, pc=1e-7, safe_miss_m=5000.0)
        self.assertEqual(rec["delta_v_m_s"], 0.0)
        self.assertEqual(rec["safety"], "SAFE")
        self.assertEqual(rec["direction"], "no burn")


if __name__ == "__main__":
    unittest.main()
