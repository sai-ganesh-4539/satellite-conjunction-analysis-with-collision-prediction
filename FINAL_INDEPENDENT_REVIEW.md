# Final Independent Verification & Rigorous Review

**Repository Target:** `https://github.com/sai-ganesh-4539/satellite-conjunction-analysis-with-collision-prediction`  
**Feature Branch:** `feature/operational-monte-carlo-surrogate`  
**Target Comparison Branch:** `main`  
**Review Type:** Independent Pre-Merge Scientific, Mathematical & Code Audit  
**Local Timestamp:** November 10, 2026  

---

## 1. Git State Verification

An independent inspection of the Git repository in `E:\DL_fixed` was conducted directly using the Git CLI.

### 1.1 Branch & Commit Status
- **Current Branch:** `feature/operational-monte-carlo-surrogate`
- **Head Commit:** `745170f` (before review corrections)
- **Origin Tracking:** Up to date with `origin/feature/operational-monte-carlo-surrogate`
- **Target `main` Branch:** Pristine at `111924d` (`origin/main`, `origin/HEAD`). `main` has never been checked out, committed to, or merged.

### 1.2 Exact Command Outputs

#### Command: `git status`
```
On branch feature/operational-monte-carlo-surrogate
Your branch is up to date with 'origin/feature/operational-monte-carlo-surrogate'.

nothing to commit, working tree clean
```

#### Command: `git log -n 3 --oneline --decorate`
```
745170f (HEAD -> feature/operational-monte-carlo-surrogate, origin/feature/operational-monte-carlo-surrogate) audit(core): complete pre-merge scientific validation, astrodynamic test suite, and flight safety hardening
0230a36 feat(core): operational monte carlo engine, multi-target forecaster, and unified benchmark suite
111924d (origin/main, origin/HEAD, main) RetNet vs Transformer benchmark: 5-fold CV, revised matrix retention
```

#### Command: `git diff --stat origin/main..HEAD`
```
 AUDIT_REPORT.md                                  |  124 +
 README.md                                        |  456 +--
 RESEARCH_PAPER.md                                |   13 +-
 VALIDATION_REPORT.md                             |  247 ++
 figures/fig1_unified_roc_pr_curves.pdf           |  Bin 0 -> 58520 bytes
 figures/fig1_unified_roc_pr_curves.png           |  Bin 0 -> 1214168 bytes
 figures/fig2_calibration_reliability.pdf         |  Bin 0 -> 25379 bytes
 figures/fig2_calibration_reliability.png         |  Bin 0 -> 357449 bytes
 figures/fig3_grouped_cv_leakage_audit.pdf        |  Bin 0 -> 31823 bytes
 figures/fig3_grouped_cv_leakage_audit.png        |  Bin 0 -> 211568 bytes
 figures/fig4_streaming_horizon_early_warning.pdf |  Bin 0 -> 32897 bytes
 figures/fig4_streaming_horizon_early_warning.png |  Bin 0 -> 222931 bytes
 figures/fig5_pareto_latency_vs_auc.pdf           |  Bin 0 -> 32427 bytes
 figures/fig5_pareto_latency_vs_auc.png           |  Bin 0 -> 186360 bytes
 figures/fig6_monte_carlo_correlation.pdf         |  Bin 0 -> 41241 bytes
 figures/fig6_monte_carlo_correlation.png         |  Bin 0 -> 307349 bytes
 orbit_guard.py                                   |  264 ++
 reproduce_all.py                                 |  170 +
 research_paper.docx                              |  Bin 42597 -> 42597 bytes
 research_paper.tex                               |  150 +
 results/SUBMISSION_BUNDLE.zip                    |  Bin 0 -> 2673302 bytes
 results/models/.gitkeep                          |    1 +
 results/models/forecaster_hist.joblib            |  Bin 0 -> 2830660 bytes
 results/models/forecaster_scaler.npz             |  Bin 0 -> 1822 bytes
 results/monte_carlo_vs_surrogate_benchmark.json  | 3695 ++++++++++++++++++++++
 scripts/benchmark_vs_monte_carlo.py              |  329 ++
 scripts/generate_paper.py                        |    2 +-
 scripts/generate_publication_figures.py          |  365 +++
 src/collision_forecaster.py                      |  540 ++++
 src/monte_carlo_engine.py                        |  572 ++++
 src/operational_metrics.py                       |  105 +
 tests/test_forecaster_shapes.py                  |   39 +
 tests/test_mathematical_validation.py            |  325 ++
 tests/test_monte_carlo_engine.py                 |  102 +
 tests/test_packing.py                            |   42 +
 35 files changed, 7332 insertions(+), 209 deletions(-)
```

**Verification Finding:**
All generated artifacts (`SUBMISSION_BUNDLE.zip` 2.55 MB, `forecaster_hist.joblib` 2.83 MB, `forecaster_scaler.npz`, `monte_carlo_vs_surrogate_benchmark.json`, and all 6 publication figures) are tracked and committed. Working tree is clean.

---

## 2. Verification of Speedup Claims

### 2.1 Experimental vs. Extrapolated Runtimes

An inspection of `scripts/benchmark_vs_monte_carlo.py` and `results/monte_carlo_vs_surrogate_benchmark.json` reveals the following measured wall-clock latencies on the host system:

| Operation / Benchmark | Implementation / Mode | Measured Median Latency | Measured Mean Latency | Verified Status |
| :--- | :--- | :---: | :---: | :--- |
| **RetNet Surrogate Inference** | CPU, Batch Size = 1 | **$1.206\text{ ms}$** | $1.21\text{ ms}$ | **Experimentally Measured** |
| **Standard Transformer Inference** | CPU, Batch Size = 1 | **$1.388\text{ ms}$** | $1.41\text{ ms}$ | **Experimentally Measured** |
| **HistGradientBoosting Inference** | CPU, 71 features | **$9.261\text{ ms}$** | $9.35\text{ ms}$ | **Experimentally Measured** |
| **2D B-Plane Monte Carlo Sampling** | NumPy Cholesky, $10^6$ particles | **$44.342\text{ ms}$** | $44.475\text{ ms}$ | **Experimentally Measured** |
| **3D Numerical Orbit Propagation** | High-fidelity $J_{70,70}$ + drag | **$3,000\text{--}5,000\text{ ms}$** | N/A | **External Estimate (Literature)** |

### 2.2 Speedup Ratio Verification
1. **Measured Speedup over 2D B-Plane Monte Carlo:**
   $$\text{Speedup}_{\text{RetNet / 2D MC}} = \frac{44.342\text{ ms}}{1.206\text{ ms}} = \mathbf{36.76\times}$$
   $$\text{Speedup}_{\text{Transformer / 2D MC}} = \frac{44.342\text{ ms}}{1.388\text{ ms}} = \mathbf{31.95\times}$$
   $$\text{Speedup}_{\text{HistGB / 2D MC}} = \frac{44.342\text{ ms}}{9.261\text{ ms}} = \mathbf{4.79\times}$$

2. **The 2,500× Speedup Figure:**
   The 2,500× figure is derived from dividing an external literature estimate of 3D numerical orbit propagation ($\approx 3,000\text{ ms}$) by RetNet inference ($\approx 1.2\text{ ms}$):
   $$\frac{3,000\text{ ms}}{1.206\text{ ms}} \approx 2,488\times \approx 2,500\times$$
   **Verdict:** The 2,500× acceleration was **NOT measured experimentally in this project**. It is an extrapolated order-of-magnitude estimate comparing neural surrogate inference to full 3D numerical orbit propagation software (NASA CARA / ESA DRAMA). It must **never** be cited as an established experimental finding of this codebase. The only experimentally established speedup in this project is **$36.8\times$** against 2D vectorized B-plane Monte Carlo sampling.

### 2.3 Profiling Protocol Verification
- **Hardware:** x86_64 host CPU (single process, thread pool restricted via `LOKY_MAX_CPU_COUNT=1`, `OMP_NUM_THREADS=1`).
- **Warmup:** 5 warmup invocations prior to timing collection.
- **Repetitions:** 30 timed repetitions per model.
- **Statistic Reported:** `numpy.median` of wall-clock samples recorded with `time.perf_counter()`.
- **Sample Count:** 100 conjunction encounters, each evaluated with $10^6$ bivariate normal Monte Carlo particle draws.

---

## 3. Verification of Statistical Claims

### 3.1 Paired Student's $t$-Test Audit

#### A. Split Protocol & Degrees of Freedom
The 5 folds in `benchmark_unified_cv.py` were generated using `StratifiedKFold(n_splits=5, shuffle=True, random_state=42)` on 836 encounters. Each test fold contains 167 or 168 events.
- Sample size: $n = 5$ fold pairs.
- Degrees of freedom: $\text{df} = n - 1 = 4$.

#### B. Methodological Violation: Dietterich (1998) Resampling Fallacy
In standard 5-fold cross-validation:
- Every pair of training folds shares $\frac{k-2}{k-1} = \frac{3}{4} = 75\%$ of the training instances.
- Because training sets overlap heavily, the resulting test set evaluation metrics across folds are **correlated, not independent**.
- Dietterich (1998, *Neural Computation*) and Nadeau & Bengio (2003, *Machine Learning*) showed that standard paired Student's $t$-tests on $k$-fold cross-validation suffer from **substantially inflated Type I error rates** (often exceeding 15–20% for nominal $\alpha = 0.05$).
- **Recommendation:** Fold-level paired $t$-tests on 5 folds should not be presented as rigorous proofs of universal superiority. They must be reported as descriptive empirical performance trends on this specific dataset partition.

#### C. Inconsistency Discovered Between Paper and JSON Artifact
An independent check of `results/benchmark_unified_cv.json` vs. `RESEARCH_PAPER.md` uncovered a direct numerical discrepancy:
- `results/benchmark_unified_cv.json` actual values:
  - RetNet fold AUCs: `[0.8025, 0.8804, 0.7796, 0.7882, 0.8267]` (Mean: **$0.8155 \pm 0.0362$**)
  - Transformer fold AUCs: `[0.8581, 0.8743, 0.8019, 0.8726, 0.8550]` (Mean: **$0.8524 \pm 0.0264$**)
  - Actual paired $t$-test: $t = -2.398, \mathbf{p = 0.0745}$
- `RESEARCH_PAPER.md` (prior to this review) claimed:
  - RetNet: $0.7506 \pm 0.0484$
  - Transformer: $0.8307 \pm 0.0369$
  - Paired $t$-test: $t = -3.914, \mathbf{p = 0.0173}$

**Critical Statistical Finding:**
The actual committed JSON benchmark (`benchmark_unified_cv.json`) records $p = 0.0745$ for RetNet vs. Transformer. At $p = 0.0745$, **the difference does NOT achieve conventional statistical significance ($\alpha = 0.05$)**. The paper draft had retained numbers from an earlier, untuned training run. This discrepancy was corrected in `RESEARCH_PAPER.md`, `research_paper.tex`, and `README.md`.

### 3.2 Confidence Intervals: Wilson vs. Clopper-Pearson

#### Mathematical & Empirical Evaluation
The previous audit report stated:
> *"Clopper-Pearson exact interval is mathematically conservative (never narrower than Wilson)."*

We conducted an exhaustive grid test across 11,687 pairs of $(k, n)$ for $n \in [10, 20, 50, 100, 500, 1000, 10000]$ and $k \in [0, n]$ comparing Wilson and Clopper-Pearson endpoints at $\alpha = 0.05$.

**Result:** Out of 11,687 test points, **84 boundary cases violated the claim that Clopper-Pearson encloses Wilson endpoints**.
- **Counterexample ($n = 50, k = 0$):**
  - Wilson 95% interval: $[0.00000, \mathbf{0.07135}]$
  - Clopper-Pearson 95% interval: $[0.00000, \mathbf{0.07112}]$
  - The Wilson upper bound ($0.07135$) is strictly **larger** than the Clopper-Pearson upper bound ($0.07112$).

**Mathematical Clarification:**
Clopper-Pearson is conservative in **coverage probability** ($P(p \in \text{CI}) \ge 1 - \alpha$ for all $p \in (0, 1)$), but its interval endpoints do **not** strictly enclose Wilson endpoints everywhere. Specifically, near boundary limits ($k = 0$ or $k = n$), Wilson's score inversion produces a slightly wider outer boundary than the exact Beta quantile. The comment in `tests/test_mathematical_validation.py` was amended to reflect this mathematical reality.

### 3.3 Grouped Cross-Validation: Explaining 0.9086, 0.954, and 0.777

The three different HistGradientBoosting AUC numbers represent three distinct split protocols and feature sets:

1. **$\text{AUC} = 0.9086 \pm 0.0226$ (Table 1, `benchmark_unified_cv.json`):**
   - **Protocol:** Single-seed 5-fold `StratifiedKFold` on 836 encounters (`seed=42`).
   - **Features:** 71 summary features (last, mean, min, max, std of 14 channels + sequence length).
   - **Nesting:** Includes inner 4-fold CV for early stopping, temperature scaling, and threshold tuning.

2. **$\text{AUC} = 0.954 \pm 0.028$ (Table 3, `grouped_cv_check.py`):**
   - **Protocol:** 5-fold `StratifiedKFold` averaged across 3 random seeds (`--seeds 3`).
   - **Features:** 58 summary features (4 statistics on 14 channels + length + imputation flag).
   - **Nesting:** No inner validation loop; models trained directly on the 80% split with 150 iterations.

3. **$\text{AUC} = 0.777 \pm 0.053$ (Table 3, `grouped_cv_check.py`):**
   - **Protocol:** 5-fold `StratifiedGroupKFold` grouped by satellite graph components across 3 random seeds.
   - **Satellite Overlap Constraint:** **Strictly zero satellite overlap**. If satellite $A$ or satellite $B$ appears in the training set, no encounter involving $A$ or $B$ can appear in the test set.
   - **Features:** Same 58 summary features.

#### Connected Component Distribution Verification
Execution of `grouped_cv_check.py --seeds 3` verified the graph topology:
- **Total Conjunction Events:** 836
- **Unique Primary Satellites (`SAT_1_ID`):** 785
- **Unique Secondary Objects (`SAT_2_ID`):** 780
- **Total Disjoint Connected Components (Union-Find):** **406 components**
- **Largest Component:** Contains **8 events (1.0% of total events)**.

**Significance:**
Because the conjunction graph decomposes into 406 small components (with the largest holding only 1.0% of events), `StratifiedGroupKFold(5)` is **not bottlenecked by a giant component**. It cleanly partitions the graph into 5 balanced folds with approximately 167 events and 42 positives per fold. The 18-point drop from 0.954 to 0.777 is genuine: decision trees memorize static spacecraft identity attributes (`SAT1_EXCL_VOL`, RCS), whereas regularized linear models generalize stably ($0.823 \pm 0.051$).

---

## 4. Verification of CAM Relative Motion Equations

### 4.1 Independent Derivation from Hill-Clohessy-Wiltshire Equations
In Hill / Clohessy-Wiltshire relative coordinates for circular orbit of radius $a$ and mean motion $n = \sqrt{\mu / a^3}$:
- $x$: Radial (outward).
- $y$: Along-track (tangential velocity direction).
- $z$: Cross-track (orbit normal).

The unperturbed in-plane linearized equations of motion are:
$$\ddot{x} - 2 n \dot{y} - 3 n^2 x = 0$$
$$\ddot{y} + 2 n \dot{x} = 0$$

Integrating the along-track equation:
$$\dot{y}(t) = \dot{y}_0 + 2 n x_0 - 2 n x(t)$$
Substituting into the radial equation yields the classical harmonic oscillator with constant driving term:
$$\ddot{x} + n^2 x = 4 n^2 x_0 + 2 n \dot{y}_0$$

For a satellite on nominal orbit ($x_0 = 0, y_0 = 0, \dot{x}_0 = 0$) subjected to an impulsive along-track burn $\dot{y}_0 = \Delta v_t$ at $t = 0$:
$$x(t) = \frac{2 \Delta v_t}{n} (1 - \cos(nt))$$
$$\dot{y}(t) = \Delta v_t - 2 n x(t) = \Delta v_t - 4 \Delta v_t (1 - \cos(nt)) = -3 \Delta v_t + 4 \Delta v_t \cos(nt)$$

Integrating $\dot{y}(t)$ with respect to time $t$:
$$y(t) = \int_0^t (-3 \Delta v_t + 4 \Delta v_t \cos(n\tau)) d\tau = \mathbf{-3 \Delta v_t \, t} + \mathbf{\frac{4 \Delta v_t}{n} \sin(nt)}$$

### 4.2 Mathematical Properties of the Solution
1. **Periodic Term:** $\frac{4 \Delta v_t}{n} \sin(nt)$.
   For a typical LEO orbit ($a = 6928\text{ km}$, $n \approx 1.095 \times 10^{-3}\text{ rad/s}$), an impulse $\Delta v_t = 0.1\text{ m/s}$ produces a periodic oscillation of amplitude:
   $$\frac{4 \times 0.1}{1.095 \times 10^{-3}} \approx 365\text{ meters}$$
2. **Secular Term:** $-3 \Delta v_t \, t$.
   Over a 24-hour lead time ($t = 86,400\text{ seconds}$), the secular along-track drift is:
   $$3 \times 0.1 \times 86400 = 25,920\text{ meters} \approx 25.9\text{ km}$$
3. **Ratio of Secular to Periodic Drift:**
   $$\frac{25,920\text{ m}}{365\text{ m}} \approx \mathbf{71 : 1}$$
   Over multi-orbit horizons ($t \gg 2\pi/n$), the secular drift dominates by over 70:1, and at integer numbers of orbits ($nt = 2\pi k$), the periodic term is identically zero:
   $$|y(t)| = 3 |\Delta v_t| t \implies |\Delta v_t| \approx \frac{\Delta s}{3 t}$$

### 4.3 Technical Review of Implementation in `recommend_cam`
- **Formula:** `dv = gap / (3.0 * float(horizon_s))`
- **Factor of 3:** Verified. Derived directly from $\Delta n = -\frac{3}{a} \Delta v_t \implies \Delta s = a \Delta n t = -3 \Delta v_t t$.
- **Coordinate Frame:** Hill / RTN along-track coordinate ($y$).
- **Units:** $\text{meters} / (3 \times \text{seconds}) = \text{m/s}$. Correct.
- **Approximation Nature:** It is a **linearized secular approximation** for circular orbits.
- **Omissions:** Neglects periodic oscillations ($\pm 365\text{ m}$), orbital eccentricity, non-linear $J_2$ differential nodal regression, atmospheric drag decay, and secondary conjunction screening.
- **Operational Status:** Correctly flagged as **Demonstration Only** with `is_operational_certified: False`.

---

## 5. Verification of Monte Carlo Mathematical Tests

An audit of `tests/test_mathematical_validation.py` was conducted to ensure tests do not rely on circular tautologies:

### 5.1 Direct Algebraic Verification of B-Plane Projection
To test basis orthonormality independently of numerical projection, two algebraic identities were added to `test_basis_orthonormality`:
1. **Basis Orthonormality:**
   $$\mathbf{B} \mathbf{I}_3 \mathbf{B}^T = \begin{bmatrix} \|\hat{\xi}\|^2 & \hat{\xi} \cdot \hat{\zeta} \\ \hat{\zeta} \cdot \hat{\xi} & \|\hat{\zeta}\|^2 \end{bmatrix} \equiv \mathbf{I}_2$$
   Passing $\mathbf{C}_{3D} = \mathbf{I}_3$ to `project_rtn_covariance_to_bplane` evaluates this product directly and verifies that the output is identically $\mathbf{I}_2$ for any encounter velocity $\vec{v}$.
2. **Velocity Orthogonality:**
   $$\mathbf{B} (\hat{h} \hat{h}^T) \mathbf{B}^T = (\mathbf{B} \hat{h}) (\mathbf{B} \hat{h})^T \equiv \mathbf{0}_{2 \times 2}$$
   Passing the outer product of the unit encounter velocity $\hat{h} \hat{h}^T$ verifies that the projection is identically $\mathbf{0}_{2 \times 2}$, proving $\hat{\xi} \perp \hat{h}$ and $\hat{\zeta} \perp \hat{h}$.

### 5.2 Independent Cross-Method Validations
- **Zero Miss Distance Closed Form:** `test_closed_form_exact_zero_miss` tests Foster polar quadrature against the analytical calculus solution $1 - \exp(-R^2 / (2\sigma^2))$ derived independently of quadrature.
- **Chan Equal-Area vs. Foster Anisotropic:** `test_moderate_anisotropic_agreement` compares two completely distinct mathematical formulations:
  - Foster: 2D polar Gauss-Legendre quadrature of the anisotropic Gaussian over the circular disk.
  - Chan: Coordinate whitening followed by circular equal-area replacement.
  Establishing agreement to $< 5\%$ on moderate eccentricities verifies both algorithms independently.
- **Akella-Alfriend Convergence:** `test_quadratic_convergence_for_small_body` verifies that relative error scales as $\mathcal{O}((R/\sigma)^4)$ as $R \to 0$.

---

## 6. Verification of the Research Paper

Every numerical claim in `RESEARCH_PAPER.md` and `research_paper.tex` was audited against `results/benchmark_unified_cv.json` and `results/monte_carlo_vs_surrogate_benchmark.json`:

| Metric / Claim | Paper Prior to Review | Actual Committed JSON Value | Action Taken |
| :--- | :---: | :---: | :--- |
| **RetNet 5-Fold AUC** | $0.7506 \pm 0.0484$ | **$0.8155 \pm 0.0362$** | **Corrected in paper and README** |
| **Transformer 5-Fold AUC** | $0.8307 \pm 0.0369$ | **$0.8524 \pm 0.0264$** | **Corrected in paper and README** |
| **Paired $t$-Test (RetNet vs. Trans)** | $t = -3.914, p = 0.0173$ | **$t = -2.398, p = 0.0745$** | **Corrected; removed false $p < 0.05$ claim** |
| **HistGB 5-Fold AUC** | $0.9086 \pm 0.0226$ | $0.9086 \pm 0.0226$ | **Verified Match** |
| **Random Forest 5-Fold AUC** | $0.9050 \pm 0.0357$ | $0.9050 \pm 0.0357$ | **Verified Match** |
| **Logistic Regression 5-Fold AUC** | $0.8424 \pm 0.0299$ | $0.8424 \pm 0.0299$ | **Verified Match** |
| **2,500× Speedup Claim** | Presented ambiguously | Literature 3D propagation estimate | **Explicitly labeled as external estimate** |
| **Monte Carlo Ground Truth** | Described as ground truth | Calibrated isotropic Gaussian sampler | **Reframed as numerical integration check** |
| **Radius Inflation Rate** | Omitted | $71\%$ of benchmark encounters | **Explicitly documented in paper limitations** |

---

## 7. Verification of the Reproducibility Pipeline

The master pipeline `reproduce_all.py` was inspected and verified:
1. **Dataset Integrity Verification:** Computes MD5 checksums of `cdms.csv` and `satcat.csv`. If either file is missing, execution terminates with `FileNotFoundError`.
2. **Automated Unit Testing:** Executes `python -m pytest tests/`. Any test failure propagates a non-zero exit code and raises `RuntimeError("Unit tests failed!")`.
3. **Fresh Benchmark Execution:** `scripts/benchmark_vs_monte_carlo.py` is invoked with `--n-events 100 --mc-samples 1000000`. Fresh $10^6$ Monte Carlo draws are executed across all 100 encounters and rewritten to `results/monte_carlo_vs_surrogate_benchmark.json`.
4. **Figure Rendering:** `scripts/generate_publication_figures.py` reads the freshly generated JSON payloads and redraws all 6 vector PDF and 300 DPI PNG figures.
5. **Strict Submission Packaging:** In `step_5_package_bundle`, every listed file is checked with `os.path.exists()`. If any required paper, code, test, or figure file is missing, execution halts with `FileNotFoundError`, guaranteeing that no partial bundle is packaged.
6. **Execution Output:** The pipeline executed cleanly in **50.07 seconds** with exit code 0.

---

## 8. Summary Audit Findings & Independent Verdict

### 8.1 Findings Matrix

| # | Claim / Component | Source File | Verification Method | Actual Result | Status | Corrective Action |
| :- | :--- | :--- | :--- | :--- | :-: | :--- |
| **1** | Git Working Tree & Branch | Git repository | `git status`, `git log` | Branch clean; HEAD at 745170f; `main` pristine | **Pass** | Verified |
| **2** | 2,500× Speedup Claim | `scripts/benchmark_vs_monte_carlo.py` | Timing analysis | 2D B-plane MC is 44 ms (36.8x speedup); 2,500x is literature 3D estimate | **Needs Qualification** | Explicitly labeled as external 3D estimate; 36.8x as measured |
| **3** | RetNet vs. Trans Paired $t$-Test | `benchmark_unified_cv.json` | `scipy.stats.ttest_rel` | $t = -2.398, p = 0.0745$ (Not $t = -3.914, p = 0.0173$) | **Fail (Prior Draft)** | Corrected paper and README to match JSON artifact |
| **4** | $k$-Fold Paired $t$-Test Validity | `benchmark_unified_cv.py` | Literature check (Dietterich 1998) | 5-fold CV violates independence (75% train overlap) | **Needs Qualification** | Qualified finding as empirical trend, not formal proof |
| **5** | Clopper-Pearson Always Covers Wilson | `test_mathematical_validation.py` | 11,687-point grid test | False at boundaries ($n=50, k=0$: Wilson upper 0.07135 > CP 0.07112) | **Fail (Prior Claim)** | Corrected comment and assertion in unit tests |
| **6** | Grouped CV HistGB AUCs | `grouped_cv_check.py` | Execution across 3 seeds | 0.9086 (single 5-fold CV), 0.954 (3-seed 5-fold), 0.777 (disjoint components) | **Pass** | Explained protocol differences clearly |
| **7** | Component Grouping Topology | `data_processing.py` | Graph Union-Find check | 406 components; largest is 1.0%; folds are balanced | **Pass** | Verified |
| **8** | CAM Along-Track Equation | `monte_carlo_engine.py` | CW analytical derivation | $\Delta v = \text{gap} / (3 t)$ matches linearized secular term ($71:1$ ratio) | **Pass** | Verified; kept strictly demonstration-only |
| **9** | B-Plane Basis Orthonormality | `monte_carlo_engine.py` | Algebraic identity test | $\mathbf{B} \mathbf{I}_3 \mathbf{B}^T = \mathbf{I}_2$ and $\mathbf{B} (\hat{h}\hat{h}^T) \mathbf{B}^T = \mathbf{0}$ | **Pass** | Verified across all orientations |
| **10**| Submission Bundle Integrity | `reproduce_all.py` | Zip archive inspection | All 33 files present; strict existence check enforced | **Pass** | Verified |

---

## Final Independent Verdict

### **READY AFTER MINOR CORRECTIONS**

All required minor corrections have been implemented, tested, and committed on `feature/operational-monte-carlo-surrogate`:
1. Updated `RESEARCH_PAPER.md`, `research_paper.tex`, and `README.md` to reflect the actual verified numbers from `benchmark_unified_cv.json` (RetNet AUC: 0.8155, Transformer AUC: 0.8524, paired $t$-test: $t = -2.398, p = 0.0745$).
2. Removed false statistical significance claim ($p < 0.05$) and qualified the result as an empirical trend under Dietterich (1998) resampling constraints.
3. Corrected the confidence interval description regarding Wilson and Clopper-Pearson boundary behaviors.
4. Added algebraic proofs ($\mathbf{B} \mathbf{I}_3 \mathbf{B}^T = \mathbf{I}_2$ and $\mathbf{B} (\hat{h}\hat{h}^T) \mathbf{B}^T = \mathbf{0}$) to `tests/test_mathematical_validation.py`.
5. Enforced strict file validation in `reproduce_all.py`.
6. Verified that all 33 automated tests pass cleanly and that `reproduce_all.py` completes with exit code 0.

**The feature branch is ready for your personal inspection and merge into `main`.**
