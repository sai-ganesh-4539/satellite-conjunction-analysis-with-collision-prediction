# Comprehensive Pre-Merge Audit Report: Satellite Conjunction Assessment System

**Repository:** `https://github.com/sai-ganesh-4539/satellite-conjunction-analysis-with-collision-prediction`  
**Feature Branch:** `feature/operational-monte-carlo-surrogate`  
**Target Branch:** `main`  
**Audit Conducted:** November 10, 2026  
**Auditor:** Principal Astrodynamics & AI Pre-Merge Review Board  
**Overall Status:** **Conditionally Ready for User Approval (Do NOT merge to main automatically)**

---

## 1. Executive Summary

This pre-merge code review and scientific validation was performed on branch `feature/operational-monte-carlo-surrogate` against `main`. 

The objective was to audit every modified and added file, identify mathematical and physical flaws, eliminate data leakage risks, verify computational speedup claims, enforce operational safety disclaimers, and ensure absolute scientific integrity in research publications.

All 34 automated tests in `tests/` pass with zero failures. Critical scientific deficiencies around isotropic covariance calibration, CAM flight safety authority, and speedup framing have been fully resolved and documented.

---

## 2. File-by-File Audit Matrix

| File Path | Problem Identified | Severity | Evidence | Recommended Correction | Status |
| :--- | :--- | :---: | :--- | :--- | :---: |
| `src/monte_carlo_engine.py` | Isotropic covariance scale inversion presented without documenting circularity of MC agreement. | **Critical** | `invert_isotropic_scale` solves for $\sigma$ to match catalog $P_c$, and inflates radius in 71% of encounters. MC samples this fitted Gaussian, which verifies quadrature, not independent orbit determination. | Add explicit docstring and report disclaimers stating that Monte Carlo is numerical integration verification, not independent orbit determination. | **Fixed** |
| `src/monte_carlo_engine.py` | Collision Avoidance Maneuver (CAM) lacked operational flight disclaimers and safety flags. | **High** | `recommend_cam` returned linear along-track impulse without stating that 3D propagation and secondary screening are absent. | Added `is_operational_certified: False` and explicit multi-sentence safety disclaimer. | **Fixed** |
| `src/monte_carlo_engine.py` | B-plane coordinate projection lacked fallback for degenerate normal velocity vector. | **Medium** | When relative velocity was parallel to both radial and cross-track test directions, basis construction could fail. | Added 3rd orthogonal fallback vector `[0, 0, 1]` and verified basis orthonormality across all encounter angles. | **Fixed** |
| `orbit_guard.py` | CAM recommendation labeled as "ACTIONABLE FLIGHT DYNAMICS RECOMMENDATION" without safety warnings. | **Critical** | Output header implied operational flight execution authority without non-linear orbit propagation or secondary screening. | Added prominent flight safety warning banners and labeled as demonstration heuristic only. | **Fixed** |
| `orbit_guard.py` | Monte Carlo execution labeled as "GROUND-TRUTH HIGH-FIDELITY MONTE CARLO". | **High** | Implied independent physical truth from raw radar observations rather than 2D isotropic integration sampling. | Relabeled to "NUMERICAL MONTE CARLO INTEGRATION BENCHMARK (2D B-Plane Samples)". | **Fixed** |
| `orbit_guard.py` | Speedup comparison conflated 2D B-plane sampling with 3D numerical propagation. | **Medium** | RetNet inference was compared to 2D sampling while citing unverified literature 3D propagation times. | Profiled single-core CPU 2D B-plane MC ($44.62\text{ ms}$). RetNet recurrent achieves $67.3\times$ speedup ($0.66\text{ ms}$). Removed unverified 3D propagation runtime and >2,500x speedup claims. | **Fixed** |
| `src/model.py` | Dimension mismatch in `SwiGLUFFN` layer `w3`. | **High** | `self.w3 = nn.Linear(d_ff, d_model)` caused `RuntimeError: mat1 and mat2 shapes cannot be multiplied (56x128 and 170x128)`. | Corrected to `nn.Linear(d_model, d_ff, bias=False)`. Dimension tests verified. | **Fixed** |
| `src/model.py` | Token padding previously treated as physical zero-values. | **High** | Zero-padding after normalization matched mean feature values, injecting artificial observations. | Implemented `pack_real_first` and causal masked pooling over real observations. Verified by `test_packing.py`. | **Fixed** |
| `src/collision_forecaster.py` | Target values could potentially leak into earlier CDM sequence timesteps. | **High** | Sequence forecaster predicting final miss distance and $P_c$. | Verified that input prefix is strictly `g.iloc[:-1]` and target is `g.iloc[-1]`. Target $P_c$ is never an input feature. | **Verified Safe** |
| `scripts/benchmark_vs_monte_carlo.py` | Loky CPU core count detection hangs on Windows environments. | **High** | Scikit-learn tree models froze during multiprocessing initialization on Windows. | Forced `LOKY_MAX_CPU_COUNT=1` and `OMP_NUM_THREADS=1`. Execution verified. | **Fixed** |
| `scripts/generate_paper.py` | Hardcoded absolute Linux path `/home/z/...` causing `FileNotFoundError`. | **Medium** | Path to `research_paper.docx` pointed to external developer machine directory. | Replaced with `os.path.join(REPO_ROOT, "research_paper.docx")`. | **Fixed** |
| `RESEARCH_PAPER.md` & `research_paper.tex` | Speedup claims and Monte Carlo descriptions lacked explicit boundary disclosures. | **High** | Abstract claimed $>2,500\times$ speedup over Monte Carlo without verified numerical runtime. | Updated text to report measured single-core 2D B-plane MC ($44.62\text{ ms}$) speedups: RetNet recurrent $67.3\times$ ($0.66\text{ ms}$), Transformer $50.3\times$ ($0.89\text{ ms}$). Removed unverified 3D propagation and >2,500x claims. Disclosed 71% radius inflation. | **Fixed** |
| `README.md` | Outdated documentation referencing non-existent scripts (`train.py`), obsolete parameters, and lacking CLI guides. | **High** | README referenced 28 timesteps, RMSNorm, and did not document `orbit_guard.py` or `reproduce_all.py`. | Completely overhauled with accurate architecture, installation, reproduction commands, and CLI guide. | **Fixed** |
| `reproduce_all.py` | Submission bundle omitted new mathematical tests and audit reports. | **Medium** | `files_to_pack` did not include `AUDIT_REPORT.md`, `VALIDATION_REPORT.md`, and `test_mathematical_validation.py`. | Added all reports and tests to `files_to_pack`. | **Fixed** |

---

## 3. Detailed Audit Findings & Scientific Resolutions

### 3.1 Critical Finding 1: The Isotropic Covariance Inversion & Monte Carlo Circularity
- **The Issue:** Public Space-Track CDMs provide scalar miss distance and catalog $P_c$, but omit the 2x2 B-plane covariance matrix. The system inverts an isotropic $\sigma$ so that the analytical 2D Gaussian integral matches the catalog $P_c$. Monte Carlo sampling then draws from this fitted distribution and confirms numerical quadrature agreement with the analytical formula (93% of analytical points lie within the Wilson 95% confidence interval). Direct regression of log10(Pc) yields R2 <= 0.034 for Transformer and negative for RetNet due to unobserved covariance evolution.
- **Scientific Impact:** Presenting Monte Carlo agreement as an "independent validation of collision probability" is scientifically invalid. Monte Carlo is sampling the distribution that was explicitly calibrated to match the target. It verifies the **numerical integration accuracy**, NOT an independent physical orbit determination.
- **Resolution:**
  1. Updated all docstrings in `src/monte_carlo_engine.py` and `scripts/benchmark_vs_monte_carlo.py`.
  2. Updated `orbit_guard.py` display to clearly show: `[NOTE: Sampling verification of 2D isotropic Gaussian integral; not independent orbit determination]`.
  3. Disclosed in `RESEARCH_PAPER.md` and `VALIDATION_REPORT.md` that 71% of encounters required radius inflation because isotropic Gaussians cannot match the peak collision probabilities produced by real elongated covariance ellipsoids.

### 3.2 Critical Finding 2: Collision Avoidance Maneuver (CAM) Operational Authority
- **The Issue:** `orbit_guard.py` displayed "ACTIONABLE FLIGHT DYNAMICS RECOMMENDATION" and recommended impulsive burns ($\approx 0.08\text{--}0.5\text{ m/s}$) to achieve 5 km clearance based on a linearized along-track Clohessy-Wiltshire secular drift approximation.
- **Safety Impact:** In operational astrodynamics, executing an avoidance burn based solely on a linear 2-body approximation without non-linear gravity propagation, atmospheric drag covariance, or secondary conjunction screening risks deflecting the spacecraft into another debris object.
- **Resolution:**
  1. Renamed section to `CONJUNCTION ASSESSMENT ADVISORY (DEMONSTRATION HEURISTIC)`.
  2. Added prominent safety warning banners at startup and in the advisory block.
  3. Added `is_operational_certified: False` and explicit multi-sentence safety disclaimers to both `recommend_cam` and `compute_cam_recommendation`.

### 3.3 High Finding 3: Speedup Comparison & Removal of Unverified 3D Runtime Claims
- **The Issue:** Publications and CLI output previously cited an unverified $>2,500\times$ speedup over numerical 3D orbit propagation citing Foster (1992).
- **Scientific Reality:** Foster (1992) NASA JSC-25898 derived the 2D B-plane analytical probability integral, NOT numerical runtime. Citing Foster for numerical 3D runtime was invalid.
- **Resolution:**
  1. Removed all unverified 3D numerical orbit propagation runtime (~3,000–5,000 ms) and theoretical >2,500× speedup claims across all documentation.
  2. Standardized all timing on single-core CPU microbenchmarks:
     - 2D B-Plane Monte Carlo ($10^6$ particles): **$44.619\text{ ms}$** median ($44.689\text{ ms}$ mean).
     - RetNet Recurrent (O(1) step): **$0.663\text{ ms}$** median ($67.3\times$ speedup).
     - Standard Transformer: **$0.886\text{ ms}$** median ($50.3\times$ speedup).
     - HistGradientBoosting: **$1.286\text{ ms}$** median ($34.7\times$ speedup).

### 3.4 High Finding 4: Sequence Model Reality vs Tabular Tree Ensembles
- **The Issue:** Previous reports sought to position deep sequence models (RetNet / Transformer) as superior to tabular tree ensembles.
- **Audit Reality:**
  - On the 836 Space-Track encounters, HistGradientBoosting achieves **0.9086 AUC** and Random Forest achieves **0.9050 AUC**.
  - Standard Transformer achieves **0.8307 ± 0.0369 AUC**, and RetNet achieves **0.7506 ± 0.0484 AUC** (unified 5-fold CV).
  - Descriptive fold differences: Transformer minus RetNet $+0.0801 \pm 0.0458$ AUC; HistGB minus RetNet $+0.1580 \pm 0.0435$ AUC. Inferential p-values are omitted because standard $k$-fold CV violates sample independence (Dietterich 1998, 75% training overlap).
  - Tabular models under strict graph-component cross-validation with zero satellite overlap between train and test drop by 18 points (0.954 to 0.777) due to memorization of static satellite attributes, while regularized linear models maintain solid generalization ($0.823 \pm 0.051$). Note: Deep sequence models (Transformer, RetNet) have not been evaluated for unseen-satellite generalization under this grouped protocol.
- **Resolution:** All papers, documentation, and reports present these results transparently, highlighting the exact boundary conditions where tabular and sequence architectures excel.

---

## 4. Test Suite Execution & Verification

The expanded test suite was executed:
```bash
python -m pytest tests/
```

**Output:**
```
============================= test session starts =============================
platform win32 -- Python 3.11.9, pytest-8.3.3, pluggy-1.6.0
rootdir: E:\DL_fixed
collected 34 items

tests\test_forecaster_shapes.py ..                                       [  5%]
tests\test_mathematical_validation.py .....................              [ 67%]
tests\test_monte_carlo_engine.py .........                               [ 94%]
tests\test_packing.py ..                                                 [100%]

============================= 34 passed in 3.80s ==============================
```

All 34 tests passed cleanly.

---

## 5. Pre-Merge Sign-Off Checklist

- [x] Working branch is `feature/operational-monte-carlo-surrogate`.
- [x] Branch `main` has NOT been modified or merged (pristine at `111924d`).
- [x] Zero force-pushes executed.
- [x] Coordinate geometry and B-plane projection mathematically validated.
- [x] Foster 2D integration verified against analytical closed forms.
- [x] Akella-Alfriend and Chan approximations mathematically verified.
- [x] Wilson score and Clopper-Pearson confidence intervals verified.
- [x] Isotropic calibration circularity clearly disclosed across all files.
- [x] CAM planner safety hardened with prominent disclaimers.
- [x] Unverified 3D propagation runtime and >2,500x speedup claims removed; speedups measured against 2D B-plane MC.
- [x] Sequence padding (`pack_real_first`) verified against all-masked rows.
- [x] Model comparison results reported honestly without bias.
- [x] `VALIDATION_REPORT.md` written and committed.
- [x] `AUDIT_REPORT.md` written and committed.
- [x] All 34 pytest unit tests passing.
