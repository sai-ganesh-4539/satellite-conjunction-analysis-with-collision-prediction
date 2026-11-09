"""
B-plane conjunction geometry and collision-probability engine.

The public CDM extract in this repository stores miss distance, catalog Pc,
and exclusion radii. It does not store the 2x2 B-plane covariance. Every
routine here therefore takes covariance as an explicit argument. Matching a
catalog Pc requires a separate, labeled inversion step (see
``invert_isotropic_scale``); that inversion is not an orbit determination.
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
from scipy.special import erfinv
from scipy.stats import beta, ncx2

# Safety triage thresholds used by the flight tool. These are operational
# policy cuts, not a claim about maneuver authority.
PC_SAFE = 1e-5
PC_CRITICAL = 5e-4


@dataclass
class Encounter:
    """Planar conjunction at TCA.

    ``miss_xy_m`` is the primary-relative position in the B-plane (meters).
    At true TCA that vector is perpendicular to the relative velocity, so its
    Euclidean norm is the miss distance. ``cov_m2`` is the combined 2x2
    position covariance in the same frame. ``radius_m`` is the combined
    hard-body radius.
    """

    miss_xy_m: np.ndarray
    cov_m2: np.ndarray
    radius_m: float
    sat1_id: Optional[int] = None
    sat2_id: Optional[int] = None
    catalog_pc: Optional[float] = None
    notes: dict = field(default_factory=dict)

    def __post_init__(self):
        self.miss_xy_m = np.asarray(self.miss_xy_m, dtype=float).reshape(2)
        self.cov_m2 = np.asarray(self.cov_m2, dtype=float).reshape(2, 2)
        self.radius_m = float(self.radius_m)
        if self.radius_m < 0:
            raise ValueError("hard-body radius must be non-negative")
        sym = 0.5 * (self.cov_m2 + self.cov_m2.T)
        eig = np.linalg.eigvalsh(sym)
        if np.any(eig < -1e-8):
            raise ValueError(f"covariance is not positive semidefinite: eigenvalues {eig}")
        self.cov_m2 = sym + np.eye(2) * max(0.0, -float(eig.min()) + 1e-12)


def hard_body_radius(excl_1_m: float, excl_2_m: float) -> float:
    """Combined spherical hard-body radius from two exclusion radii (meters)."""
    r = float(excl_1_m) + float(excl_2_m)
    if not np.isfinite(r) or r < 0:
        raise ValueError(f"invalid exclusion radii: {excl_1_m}, {excl_2_m}")
    return r


def bplane_miss(miss_distance_m: float) -> np.ndarray:
    """Place a scalar miss distance on the B-plane xi axis.

    The catalog gives only the magnitude. The along-track coordinate of the
    miss vector is zero at TCA by the definition of closest approach, and the
    remaining in-plane angle is not observed in this extract, so the miss is
    aligned with the first B-plane axis.
    """
    d = float(miss_distance_m)
    if not np.isfinite(d) or d < 0:
        raise ValueError(f"miss distance must be finite and >= 0, got {miss_distance_m}")
    return np.array([d, 0.0], dtype=float)


def isotropic_covariance(sigma_m: float) -> np.ndarray:
    s2 = float(sigma_m) ** 2
    if not np.isfinite(s2) or s2 <= 0:
        raise ValueError("sigma must be finite and positive")
    return np.array([[s2, 0.0], [0.0, s2]], dtype=float)


def combine_covariances(cov_primary: np.ndarray, cov_secondary: np.ndarray) -> np.ndarray:
    """Independent-object combined covariance, C = C1 + C2, in one frame."""
    c1 = np.asarray(cov_primary, dtype=float).reshape(2, 2)
    c2 = np.asarray(cov_secondary, dtype=float).reshape(2, 2)
    return c1 + c2


def project_rtn_covariance_to_bplane(cov_rtn: np.ndarray, v_rel_rtn: np.ndarray) -> np.ndarray:
    """Project a 3x3 RTN covariance onto the plane normal to relative velocity.

    Returns the 2x2 covariance in an orthonormal basis of that plane
    (first axis prefers the radial direction when it is not parallel to v).
    """
    c = np.asarray(cov_rtn, dtype=float).reshape(3, 3)
    v = np.asarray(v_rel_rtn, dtype=float).reshape(3)
    nrm = np.linalg.norm(v)
    if nrm < 1e-12:
        raise ValueError("relative velocity is zero; the B-plane is undefined")
    h = v / nrm
    radial = np.array([1.0, 0.0, 0.0])
    xi = radial - np.dot(radial, h) * h
    if np.linalg.norm(xi) < 1e-8:
        xi = np.array([0.0, 1.0, 0.0]) - np.dot(np.array([0.0, 1.0, 0.0]), h) * h
    xi = xi / np.linalg.norm(xi)
    zeta = np.cross(h, xi)
    basis = np.stack([xi, zeta], axis=0)  # (2, 3)
    return basis @ c @ basis.T


def _principal_axes(mu: np.ndarray, cov: np.ndarray):
    """Rotate so covariance is diagonal. Returns mx, my, sx, sy in that frame."""
    eigval, eigvec = np.linalg.eigh(cov)
    eigval = np.clip(eigval, 1e-18, None)
    # Columns of eigvec are eigenvectors. Coordinates of the mean:
    m = eigvec.T @ np.asarray(mu, dtype=float).reshape(2)
    return float(m[0]), float(m[1]), float(np.sqrt(eigval[0])), float(np.sqrt(eigval[1]))


def isotropic_pc(miss_m: float, sigma_m: float, radius_m: float) -> float:
    """Exact Pc for N(miss on x, sigma^2 I) inside a circle of ``radius_m``.

    ||X||^2 / sigma^2 follows a noncentral chi-square with 2 degrees of
    freedom and noncentrality (miss/sigma)^2. This is the isotropic Foster
    integral, evaluated with the chi-square CDF rather than a grid.
    """
    radius_m = float(radius_m)
    sigma_m = float(sigma_m)
    miss_m = float(miss_m)
    if radius_m <= 0 or sigma_m <= 0:
        return 0.0
    if miss_m == 0.0:
        return float(1.0 - math.exp(-0.5 * (radius_m / sigma_m) ** 2))
    x = (radius_m / sigma_m) ** 2
    nc = (miss_m / sigma_m) ** 2
    return float(np.clip(ncx2.cdf(x, df=2, nc=nc), 0.0, 1.0))


def foster_pc(mu: np.ndarray, cov: np.ndarray, radius_m: float,
              n_radial: int = 64, n_angular: int = 128) -> float:
    """Foster (1992) integral of the bivariate normal over the hard-body disk.

    Quadrature is Gauss-Legendre in radius and a uniform periodic rule in
    angle, including the polar Jacobian r. For isotropic covariance the
    result matches ``isotropic_pc``.
    """
    radius_m = float(radius_m)
    if radius_m <= 0:
        return 0.0
    mu = np.asarray(mu, dtype=float).reshape(2)
    cov = np.asarray(cov, dtype=float).reshape(2, 2)
    sign, logdet = np.linalg.slogdet(cov)
    if sign <= 0:
        raise ValueError("covariance must be positive definite for the Foster integral")
    inv = np.linalg.inv(cov)
    # Map r from [-1, 1] onto [0, R]. Weight includes dr/dz = R/2 and Jacobian r.
    zr, wr = np.polynomial.legendre.leggauss(int(n_radial))
    r = 0.5 * radius_m * (zr + 1.0)
    wr = wr * (0.5 * radius_m) * r
    theta = (np.arange(n_angular) + 0.5) * (2.0 * math.pi / n_angular)
    dtheta = 2.0 * math.pi / n_angular
    ct, st = np.cos(theta), np.sin(theta)
    # points: (n_radial, n_angular, 2)
    xy = np.stack([r[:, None] * ct[None, :], r[:, None] * st[None, :]], axis=-1)
    delta = xy - mu
    quad = np.einsum("...i,ij,...j->...", delta, inv, delta)
    dens = np.exp(-0.5 * quad - 0.5 * logdet) / (2.0 * math.pi)
    return float(np.clip(np.sum(dens * wr[:, None]) * dtheta, 0.0, 1.0))


def chan_pc(mu: np.ndarray, cov: np.ndarray, radius_m: float, n_terms: int | None = None) -> float:
    """Chan (1997) equal-area approximation.

    The B-plane is whitened, the hard-body circle becomes an ellipse, and that
    ellipse is replaced by a circle of equal area. The isotropic integral is
    Chan's Poisson mixture of chi-square CDFs. The mixture sums to the
    noncentral chi-square CDF; ``n_terms`` forces the truncated series instead,
    which is what the unit tests check against the closed sum.
    """
    radius_m = float(radius_m)
    if radius_m <= 0:
        return 0.0
    mx, my, sx, sy = _principal_axes(mu, cov)
    r_eq = radius_m / math.sqrt(sx * sy)
    d_eq = math.hypot(mx / sx, my / sy)
    if n_terms is None:
        return isotropic_pc(d_eq, sigma_m=1.0, radius_m=r_eq)
    return _chan_isotropic_series(d_eq, sigma=1.0, radius=r_eq, n_terms=n_terms)


def _chan_isotropic_series(miss: float, sigma: float, radius: float, n_terms: int = 20) -> float:
    """P(||X|| <= R) for X ~ N((miss, 0), sigma^2 I), truncated Chan series."""
    if radius <= 0 or sigma <= 0:
        return 0.0
    lam = (miss / sigma) ** 2
    x = (radius / sigma) ** 2
    half_lam = 0.5 * lam
    half_x = 0.5 * x
    # log p_k = -λ/2 + k log(λ/2) - log(k!)
    log_pois = -half_lam
    # CDF of chi-square with df = 2(k+1) at x: 1 - exp(-x/2) * sum_{j=0}^{k} (x/2)^j / j!
    log_terms = []  # cumulative sum pieces of the incomplete series, in log space later
    acc = 0.0
    running = 0.0  # sum_{j=0}^{k} (x/2)^j / j!   kept incrementally
    log_fact_term = 0.0  # log( (x/2)^j / j! ) for j = 0
    running = 1.0
    total = 0.0
    for k in range(int(n_terms)):
        if k > 0:
            log_pois += math.log(half_lam + 1e-300) - math.log(k)
            log_fact_term += math.log(half_x + 1e-300) - math.log(k)
            running += math.exp(log_fact_term)
        chi_cdf = 1.0 - math.exp(-half_x) * running
        total += math.exp(log_pois) * chi_cdf
    return float(np.clip(total, 0.0, 1.0))


def akella_alfriend_pc(mu: np.ndarray, cov: np.ndarray, radius_m: float) -> float:
    """Akella & Alfriend (2000) small-body expansion through the Laplacian term.

    Pc ≈ Area * [φ(0) + (R^2 / 8) ∇²φ(0)], where φ is the B-plane normal
    density and the origin is the primary hard-body center. The correction is
    the integral of the second-order Taylor polynomial over the disk. It is
    accurate when R is small compared with the covariance lengths; it is not
    a substitute for the Foster integral when that assumption fails.
    """
    radius_m = float(radius_m)
    if radius_m <= 0:
        return 0.0
    mx, my, sx, sy = _principal_axes(mu, cov)
    quad = (mx / sx) ** 2 + (my / sy) ** 2
    phi = math.exp(-0.5 * quad) / (2.0 * math.pi * sx * sy)
    lap = phi * ((mx * mx) / (sx ** 4) - 1.0 / (sx ** 2) + (my * my) / (sy ** 4) - 1.0 / (sy ** 2))
    area = math.pi * radius_m ** 2
    pc = area * (phi + (radius_m ** 2) / 8.0 * lap)
    return float(np.clip(pc, 0.0, 1.0))


def wilson_interval(k: int, n: int, alpha: float = 0.05) -> tuple[float, float]:
    """Wilson score interval for a binomial proportion."""
    if n <= 0:
        raise ValueError("n must be positive")
    z = _z_alpha(alpha)
    phat = k / n
    den = 1.0 + z * z / n
    center = (phat + z * z / (2 * n)) / den
    half = z * math.sqrt(phat * (1 - phat) / n + z * z / (4 * n * n)) / den
    return float(max(0.0, center - half)), float(min(1.0, center + half))


def clopper_pearson_interval(k: int, n: int, alpha: float = 0.05) -> tuple[float, float]:
    """Exact Clopper-Pearson interval from beta quantiles."""
    if n <= 0:
        raise ValueError("n must be positive")
    k = int(k)
    n = int(n)
    lower = 0.0 if k <= 0 else float(beta.ppf(alpha / 2.0, k, n - k + 1))
    upper = 1.0 if k >= n else float(beta.ppf(1.0 - alpha / 2.0, k + 1, n - k))
    return lower, upper


def _z_alpha(alpha: float) -> float:
    """Normal quantile for a two-sided interval of tail mass ``alpha``."""
    return float(math.sqrt(2.0) * erfinv(1.0 - alpha))


def monte_carlo_pc(
    mu: np.ndarray,
    cov: np.ndarray,
    radius_m: float,
    n_samples: int = 1_000_000,
    seed: int = 42,
    alpha: float = 0.05,
    device: str = "cpu",
) -> dict:
    """Sample the B-plane normal and estimate Pc as the hit frequency.

    ``device='cuda'`` uses PyTorch when CUDA is available. The CPU path is
    NumPy vectorized sampling (Cholesky), which is the implementation whose
    wall clock is reported. A Python-level per-particle loop is not used.
    """
    if n_samples < 1:
        raise ValueError("n_samples must be >= 1")
    mu = np.asarray(mu, dtype=float).reshape(2)
    cov = np.asarray(cov, dtype=float).reshape(2, 2)
    radius_m = float(radius_m)
    chol = np.linalg.cholesky(cov)
    t0 = time.perf_counter()
    used = "cpu"
    if device == "cuda":
        import torch
        if torch.cuda.is_available():
            g = torch.Generator(device="cuda")
            g.manual_seed(int(seed))
            z = torch.randn((n_samples, 2), device="cuda", generator=g, dtype=torch.float64)
            samples = z @ torch.as_tensor(chol.T, device="cuda", dtype=torch.float64)
            samples = samples + torch.as_tensor(mu, device="cuda", dtype=torch.float64)
            hits = int(torch.count_nonzero(samples.square().sum(-1) <= radius_m ** 2).item())
            used = "cuda"
        else:
            hits = _numpy_hits(mu, chol, radius_m, n_samples, seed)
    else:
        hits = _numpy_hits(mu, chol, radius_m, n_samples, seed)
    elapsed = time.perf_counter() - t0
    pc = hits / n_samples
    w_lo, w_hi = wilson_interval(hits, n_samples, alpha)
    c_lo, c_hi = clopper_pearson_interval(hits, n_samples, alpha)
    return {
        "pc": float(pc),
        "hits": int(hits),
        "n_samples": int(n_samples),
        "wilson_95": [w_lo, w_hi],
        "clopper_pearson_95": [c_lo, c_hi],
        "elapsed_s": float(elapsed),
        "device": used,
        "samples_per_s": float(n_samples / elapsed) if elapsed > 0 else None,
    }


def _numpy_hits(mu, chol, radius_m, n_samples, seed) -> int:
    rng = np.random.default_rng(int(seed))
    z = rng.standard_normal((int(n_samples), 2))
    samples = mu + z @ chol.T
    return int(np.count_nonzero(samples[:, 0] ** 2 + samples[:, 1] ** 2 <= radius_m ** 2))


def safety_class(pc: float) -> str:
    if pc < PC_SAFE:
        return "SAFE"
    if pc < PC_CRITICAL:
        return "MONITOR / ELEVATED"
    return "CRITICAL / ACTION REQUIRED"


def invert_isotropic_scale(miss_m: float, radius_m: float, target_pc: float) -> dict:
    """Find an isotropic sigma, and if needed a larger radius, matching ``target_pc``.

    The function of sigma rises and then falls. The solver returns the
    larger-sigma root (the diffuse-covariance branch). If no sigma can reach
    ``target_pc`` at the supplied radius, the radius is increased to the
    smallest value that can, at the sigma which maximizes Pc. The result is
    tagged ``catalog_matched`` so callers do not treat it as a measured covariance.
    """
    miss_m = float(max(miss_m, 0.0))
    radius_m = float(max(radius_m, 0.0))
    target_pc = float(target_pc)
    out = {
        "feasible_at_catalog_radius": False,
        "sigma_m": None,
        "radius_m": radius_m,
        "achieved_pc": None,
        "mode": "catalog_matched",
    }
    if not np.isfinite(target_pc) or target_pc <= 0:
        out["mode"] = "nonpositive_catalog_pc"
        out["sigma_m"] = max(miss_m, radius_m, 1.0)
        out["achieved_pc"] = 0.0
        return out
    if target_pc >= 1:
        out["sigma_m"] = max(radius_m, 1e-3)
        out["achieved_pc"] = isotropic_pc(miss_m, out["sigma_m"], radius_m)
        return out

    sigma, achieved, ok = _invert_sigma(miss_m, radius_m, target_pc)
    if ok:
        out["feasible_at_catalog_radius"] = True
        out["sigma_m"] = sigma
        out["achieved_pc"] = achieved
        return out

    # Smallest radius whose peak Pc reaches the catalog value.
    lo, hi = radius_m, max(radius_m * 2.0, miss_m, 1.0)
    peak = _peak_pc(miss_m, hi)
    guard = 0
    while peak < target_pc and guard < 40:
        hi *= 2.0
        peak = _peak_pc(miss_m, hi)
        guard += 1
    for _ in range(50):
        mid = 0.5 * (lo + hi)
        if _peak_pc(miss_m, mid) >= target_pc:
            hi = mid
        else:
            lo = mid
    sigma, achieved, ok = _invert_sigma(miss_m, hi, target_pc)
    out["radius_m"] = float(hi)
    out["sigma_m"] = sigma
    out["achieved_pc"] = achieved
    out["mode"] = "radius_inflated_to_match_catalog_pc"
    return out


def _peak_pc(miss_m: float, radius_m: float) -> float:
    if radius_m <= 0:
        return 0.0
    if miss_m <= 1e-9:
        return 1.0
    sigmas = np.exp(np.linspace(math.log(max(radius_m, 1e-3) * 0.05), math.log(max(miss_m, radius_m) * 20.0), 48))
    return float(max(isotropic_pc(miss_m, float(s), radius_m) for s in sigmas))


def _invert_sigma(miss_m: float, radius_m: float, target_pc: float):
    if radius_m <= 0:
        return None, 0.0, False
    # Log-spaced grid, then bisection on the decreasing (large-sigma) side of the peak.
    lo_s = max(radius_m, 1e-3) * 0.02
    hi_s = max(miss_m, radius_m, 1.0) * 50.0
    grid = np.exp(np.linspace(math.log(lo_s), math.log(hi_s), 60))
    vals = np.array([isotropic_pc(miss_m, float(s), radius_m) for s in grid])
    k = int(np.argmax(vals))
    if vals[k] < target_pc * (1 - 1e-6):
        return None, float(vals[k]), False
    # Right branch: from the peak toward large sigma, Pc decreases toward 0.
    left, right = float(grid[k]), float(grid[-1])
    if isotropic_pc(miss_m, right, radius_m) > target_pc:
        right = right * 10.0
    for _ in range(60):
        mid = math.sqrt(left * right)  # geometric midpoint
        if isotropic_pc(miss_m, mid, radius_m) > target_pc:
            left = mid
        else:
            right = mid
    sigma = float(right)
    return sigma, float(isotropic_pc(miss_m, sigma, radius_m)), True


def encounter_from_catalog_row(row: dict, sigma_m: Optional[float] = None) -> Encounter:
    """Build an encounter from one CDM row.

    Hard-body radius is the sum of the two exclusion radii. If ``sigma_m`` is
    omitted, a screening covariance of 1 km (isotropic) is used and the note
    ``covariance_source='screening_default_1km'`` is set. Pass the result of
    ``invert_isotropic_scale`` explicitly when a catalog-matched scale is wanted.
    """
    miss = float(row["MIN_RNG"])
    r = hard_body_radius(row.get("SAT_1_EXCL_VOL", 1.0), row.get("SAT_2_EXCL_VOL", 1.0))
    if sigma_m is None:
        sigma_m = 1000.0
        source = "screening_default_1km"
    else:
        source = "caller_supplied"
    enc = Encounter(
        miss_xy_m=bplane_miss(miss),
        cov_m2=isotropic_covariance(sigma_m),
        radius_m=r,
        sat1_id=_maybe_int(row.get("SAT_1_ID")),
        sat2_id=_maybe_int(row.get("SAT_2_ID")),
        catalog_pc=_maybe_float(row.get("PC")),
        notes={"covariance_source": source, "min_rng_m": miss},
    )
    return enc


def _maybe_int(v):
    try:
        return int(v)
    except (TypeError, ValueError):
        return None


def _maybe_float(v):
    try:
        f = float(v)
    except (TypeError, ValueError):
        return None
    return f if np.isfinite(f) else None


def recommend_cam(
    miss_m: float,
    pc: float,
    safe_miss_m: float = 5000.0,
    horizon_s: float = 24.0 * 3600.0,
) -> dict:
    """Along-track avoidance impulse from the Clohessy-Wiltshire secular drift.

    A tangential impulse produces an along-track separation of about ``3 Δv t``
    (the mean-motion terms cancel). Radial and normal impulses oscillate and
    do not provide this secular separation, so the recommended axis is
    along-track. This extract has no relative-velocity sign and no absolute
    TCA, so the horizon is a 24-hour policy default and the burn sign is not
    observable. ``delta_v_m_s`` is zero when the miss already exceeds
    ``safe_miss_m``.
    """
    miss_m = float(miss_m)
    gap = max(0.0, float(safe_miss_m) - miss_m)
    dv = gap / (3.0 * float(horizon_s)) if gap > 0 else 0.0
    return {
        "safety": safety_class(float(pc)),
        "execution_window": "T-24 h policy default (absolute TCA is not in this CDM extract)",
        "delta_v_m_s": float(dv),
        "axis": "along-track",
        "direction": "prograde or retrograde (sign not observable in this extract)" if dv > 0 else "no burn",
        "safe_miss_m": float(safe_miss_m),
        "gap_m": float(gap),
        "assumption": "Clohessy-Wiltshire secular drift: along-track separation ≈ 3 * delta_v * t",
    }


def analytical_validation_cases() -> list[dict]:
    """Cases with a closed-form isotropic answer. Not a CARA certification set."""
    specs = [
        {"name": "centered_disk", "miss_m": 0.0, "sigma_m": 1.0, "radius_m": 1.0},
        {"name": "moderate_offset", "miss_m": 50.0, "sigma_m": 20.0, "radius_m": 5.0},
        {"name": "small_body", "miss_m": 10.0, "sigma_m": 25.0, "radius_m": 2.0},
        {"name": "anisotropic_rotated", "miss_m": None, "sigma_m": None, "radius_m": 4.0,
         "mu": np.array([8.0, -3.0]), "cov": np.array([[36.0, 8.0], [8.0, 16.0]])},
    ]
    cases = []
    for s in specs:
        if s.get("mu") is None:
            mu = bplane_miss(s["miss_m"])
            cov = isotropic_covariance(s["sigma_m"])
            exact = isotropic_pc(s["miss_m"], s["sigma_m"], s["radius_m"])
        else:
            mu = np.asarray(s["mu"], dtype=float)
            cov = np.asarray(s["cov"], dtype=float)
            exact = foster_pc(mu, cov, s["radius_m"], n_radial=96, n_angular=180)
        cases.append({
            "name": s["name"],
            "mu": mu,
            "cov": cov,
            "radius_m": s["radius_m"],
            "reference_pc": exact,
        })
    return cases


def run_validation(n_samples: int = 200_000, seed: int = 7) -> list[dict]:
    """Compare Monte Carlo hit rates with Foster, Chan, and Akella-Alfriend."""
    rows = []
    for case in analytical_validation_cases():
        mc = monte_carlo_pc(case["mu"], case["cov"], case["radius_m"], n_samples=n_samples, seed=seed)
        foster = foster_pc(case["mu"], case["cov"], case["radius_m"])
        chan = chan_pc(case["mu"], case["cov"], case["radius_m"])
        akella = akella_alfriend_pc(case["mu"], case["cov"], case["radius_m"])
        lo, hi = mc["wilson_95"]
        rows.append({
            "name": case["name"],
            "reference_pc": case["reference_pc"],
            "foster_pc": foster,
            "chan_pc": chan,
            "akella_alfriend_pc": akella,
            "monte_carlo_pc": mc["pc"],
            "wilson_95": mc["wilson_95"],
            "reference_inside_wilson_95": bool(lo <= case["reference_pc"] <= hi),
            "elapsed_s": mc["elapsed_s"],
            "n_samples": n_samples,
        })
    return rows

