# Final Independent Verification & Rigorous Pre-Merge Audit

**Repository Target:** `https://github.com/sai-ganesh-4539/satellite-conjunction-analysis-with-collision-prediction`  
**Feature Branch:** `feature/operational-monte-carlo-surrogate`  
**Target Comparison Branch:** `main` (completely pristine and untouched at `111924d`)  
**Audit Scope:** Independent Verification of `SUBMISSION_BUNDLE.zip`, Astrodynamic Engine, Statistical Integrity, Figure Redesign, and Pre-Merge Readiness  
**Local Timestamp:** November 10, 2026  

---

## 1. Executive Summary & Final Verdict

An exhaustive, adversarial verification was conducted across all source code, unit tests, benchmark artifacts, publication figures, and documentation in branch `feature/operational-monte-carlo-surrogate`.

Every discrepancy identified in the independent review of `SUBMISSION_BUNDLE.zip` has been verified, mathematically analyzed, and permanently resolved on this branch.

### **FINAL VERDICT: READY TO MERGE**

| Audit Domain | Initial Finding | Resolution & Verification Evidence | Status |
| :--- | :--- | :--- | :---: |
| **1. Bundle Reproducibility** | Extracted bundle failed with `ModuleNotFoundError: run_pipeline`; missing scripts, requirements, datasets | `FocalLoss` moved to `src/model.py`. Package list updated to 46 files including `requirements.txt`, `DATASET_RETRIEVAL.md`, raw CSVs, `benchmark_unified_cv.py`, and all scripts. Clean temp dir extraction verified: **34/34 tests pass (0 errors)**. | **RESOLVED (PASS)** |
| **2. Grouped CV Discrepancies** | Paper reported 0.777/0.802/0.823/0.954; benchmark JSON reported 0.705/0.721/0.779/0.891 | Disclosed that two protocols existed (836 events, 3 seeds, 58 features vs. 641 prefix events, 1 seed, 71 features). Exported `results/grouped_cv_leakage_audit.json` with all 15 fold-level scores. Reconciled `monte_carlo_vs_surrogate_benchmark.json` to embed the verified 3-seed protocol. | **RESOLVED (PASS)** |
| **3. Statistical Claims & t-Tests** | Stale $t = -3.914, p = 0.0173$ in reports; invalid Clopper-Pearson nesting claim | Reconciled all reports to verified benchmark values ($t = -2.398, p = 0.0745$). Removed false $p < 0.05$ claim; documented Dietterich (1998) resampling limitation. Added unit test proving Wilson upper bound can exceed Clopper-Pearson at zero hits. | **RESOLVED (PASS)** |
| **4. Benchmark Reporting & Monte Carlo** | Misleading ~0.94 $R^2$ claim in audit; fabricated $R^2 = 0.942$ in Fig 6; negative/near-zero continuous $P_c$ $R^2$ | Clarified that 93% refers to numerical quadrature inside Wilson 95% CI. Removed fake $R^2 \approx 0.94$. Refactored Fig 6 to plot real benchmark points from JSON. Clearly separated classification ($0.9086$ AUC) from underconstrained continuous $P_c$ regression. | **RESOLVED (PASS)** |
| **5. CAM Operational Language** | Conflicting CAM formulas; hardcoded 0.025 m/s burn; inappropriate "operational flight software" claims | Removed all operational flight claims; relabeled as research demonstration. Refactored `orbit_guard.py` to use linearized Clohessy-Wiltshire secular formula $\Delta v = \text{gap} / (3 t)$. Eliminated hardcoded standby burn. Documented simplified assumptions. | **RESOLVED (PASS)** |
| **6. Figure 5 Scientific Redesign** | Artificial AUC = 0.999 assigned to Monte Carlo and 3D propagation; "Flight Avionics Feasibility Envelope" | Excised unphysical AUC assignments (physical integrators are not classifiers). Redesigned Figure 5 into a two-panel figure: (a) ML Pareto Frontier (Classification AUC vs. Latency); (b) Paradigm Latency Comparison. Marked 3D propagation as external literature estimate with distinct hatched visual styling. Replaced flight claims with "Low-Latency Regime (< 3 ms CPU)". | **RESOLVED (PASS)** |
| **7. Latency Harmonization** | Transformer latency discrepancy (1.388 ms vs 1.728 ms); Monte Carlo discrepancy (47.8 ms vs 41.95 ms) | Reconciled all texts, tables, and figures to verified microbenchmarks: Transformer 1.728 ms (1.251 ms pipeline), MC 41.948 ms median (43.609 ms mean). Fully documented measurement protocol (single-core CPU, batch size 1, 100 reps, 10 warmups, excluding disk I/O and SATCAT merging). | **RESOLVED (PASS)** |
| **8. Artifact-Driven Figures** | Hardcoded/fallback values in plotting script | Figures 1, 2, 3, 4, 5, 6 strictly load from `results/` JSON artifacts with zero fallback; explicit `FileNotFoundError` raised if missing. | **RESOLVED (PASS)** |
| **9. Transformer Architecture** | Clarification on causal masking and encoder prefix appropriateness | Documented `TransformerBaseline.forward()` causal attention: `pack_real_first`, `generate_square_subsequent_mask(T)`, `is_causal=True`. Justified causal encoder over observed prefix window $\mathcal{S}_k = \{\mathbf{x}_1, \dots, \mathbf{x}_k\}$ without future lookahead leakage. | **RESOLVED (PASS)** |
| **10. Unified CV Reproducibility** | Missing script provenance for `benchmark_unified_cv.json` | Updated `src/benchmark_unified_cv.py` to write `results/benchmark_unified_cv.json` directly. Added script to `reproduce_all.py` bundle packaging and added optional `--rerun-unified-cv` flag. | **RESOLVED (PASS)** |

---

## 2. Itemized Verification & Root Cause Analysis

### 2.1 Issue 1: Submission Bundle Reproducibility

#### A. Root Cause Analysis
- `src/collision_forecaster.py` and `tests/test_forecaster_shapes.py` imported `from run_pipeline import FocalLoss`.
- `run_pipeline.py` was located at repository root, but was omitted from the previous bundle file list.
- When extracted into an isolated clean environment, pytest failed immediately upon importing `tests/test_forecaster_shapes.py` with:
  `ModuleNotFoundError: No module named 'run_pipeline'`.
- Additionally, `scripts/benchmark_vs_monte_carlo.py`, `scripts/generate_publication_figures.py`, `src/benchmark_unified_cv.py`, `requirements.txt`, and the source CSV files were missing from the previous archive.

#### B. Corrections Implemented
1. **Architectural Decoupling:** Relocated `class FocalLoss(nn.Module)` directly into `src/model.py`.
2. **Import Updates:** Updated `src/collision_forecaster.py`, `src/benchmark_unified_cv.py`, and `tests/test_forecaster_shapes.py` to import `from model import FocalLoss`.
3. **Comprehensive Bundle Packaging:** Updated `reproduce_all.py` (`step_5_package_bundle`) to package 46 essential files:
   - All core libraries in `src/` (`model.py`, `monte_carlo_engine.py`, `collision_forecaster.py`, `operational_metrics.py`, `data_processing.py`, `config.py`, `benchmark_unified_cv.py`).
   - Root execution scripts (`run_pipeline.py`, `orbit_guard.py`, `grouped_cv_check.py`, `reproduce_all.py`, `download_cdms.py`).
   - Benchmark scripts in `scripts/` (`benchmark_vs_monte_carlo.py`, `generate_publication_figures.py`).
   - Test suite in `tests/` (`test_mathematical_validation.py`, `test_monte_carlo_engine.py`, `test_forecaster_shapes.py`, `test_packing.py`).
   - Datasets (`data/cdms.csv`, `data/satcat.csv`) and `DATASET_RETRIEVAL.md`.
   - Dependencies (`requirements.txt`, including `torch`, `scipy`, `scikit-learn`, `matplotlib`, `pytest`, `joblib`).
   - Benchmark JSON artifacts (`results/benchmark_unified_cv.json`, `results/monte_carlo_vs_surrogate_benchmark.json`, `results/grouped_cv_leakage_audit.json`).
   - Documentation and reports (`README.md`, `RESEARCH_PAPER.md`, `research_paper.tex`, `research_paper.docx`, `AUDIT_REPORT.md`, `VALIDATION_REPORT.md`, `FINAL_INDEPENDENT_REVIEW.md`, `PRESENTATION_GUIDE.md`).
   - All 6 vector publication figures in PNG and PDF.

#### C. Verification in Clean Isolated Directory
The newly generated `results/SUBMISSION_BUNDLE.zip` was extracted into a fresh directory (`C:\Users\Hp\AppData\Local\Temp\bundle_test_clean`) and tested:
```
============================= test session starts =============================
platform win32 -- Python 3.11.9, pytest-8.3.3, pluggy-1.6.0
rootdir: C:\Users\Hp\AppData\Local\Temp\bundle_test_clean
collected 34 items

tests\test_forecaster_shapes.py ..                                       [  5%]
tests\test_mathematical_validation.py .....................              [ 67%]
tests\test_monte_carlo_engine.py .........                               [ 94%]
tests\test_packing.py ..                                                 [100%]

============================= 34 passed in 4.11s ==============================
Return code: 0
```
Result: **100% standalone reproducibility verified with zero missing modules or datasets.**

---

### 2.2 Issue 2: Resolution of Grouped CV Discrepancies

#### A. Root Cause Analysis
Two separate cross-validation protocols were historically executed in different scripts without clear protocol distinction:
1. **Protocol A (`grouped_cv_check.py`):**
   - **Dataset:** All 836 conjunction events.
   - **Feature Space:** 58 summary features (last, mean, min, max of 14 channels, plus range trend and CDM sequence count).
   - **Split Protocol:** 5-fold cross-validation evaluated across **3 random seeds** (seeds 0, 1, 2) $\to$ **15 test folds per scheme**.
   - **Results:**
     - Stratified: HistGB $0.954 \pm 0.028$, Random Forest $0.922 \pm 0.022$, LogReg $0.833 \pm 0.035$
     - Group SAT 1: HistGB $0.957 \pm 0.025$, Random Forest $0.924 \pm 0.028$, LogReg $0.839 \pm 0.038$
     - Group SAT 2: HistGB $0.950 \pm 0.026$, Random Forest $0.923 \pm 0.027$, LogReg $0.836 \pm 0.022$
     - Group Component: HistGB **$0.777 \pm 0.053$**, Random Forest **$0.802 \pm 0.045$**, LogReg **$0.823 \pm 0.051$**
2. **Protocol B (original helper inside `scripts/benchmark_vs_monte_carlo.py`):**
   - **Dataset:** 641 prefix forecast events (excluding events with sequence length $< 2$).
   - **Feature Space:** 71 summary features (`batch.summary`).
   - **Split Protocol:** Single-seed (`seed=42`) 5-fold cross-validation $\to$ **5 folds**.
   - **Results:**
     - Stratified: HistGB $0.891 \pm 0.034$, Random Forest $0.893 \pm 0.035$, LogReg $0.782 \pm 0.044$
     - Group Component: HistGB **$0.705 \pm 0.056$**, Random Forest **$0.721 \pm 0.056$**, LogReg **$0.779 \pm 0.032$**

Because `scripts/benchmark_vs_monte_carlo.py` saved its single-seed 641-event check under `"grouped_cv"` in `monte_carlo_vs_surrogate_benchmark.json`, while Table III in `RESEARCH_PAPER.md` and Figure 3 cited the 836-event 3-seed results, an apparent discrepancy existed.

#### B. Corrections Implemented
1. **Artifact Generation:** Updated `grouped_cv_check.py` to write `results/grouped_cv_leakage_audit.json` containing complete metadata, event counts (836), unique satellite counts (785 SAT1, 780 SAT2, 406 components), random seeds (`[0, 1, 2]`), and all 15 individual fold AUC and PR-AUC scores.
2. **Benchmark JSON Synchronization:** Updated `scripts/benchmark_vs_monte_carlo.py` to import and embed `results/grouped_cv_leakage_audit.json` directly under `"grouped_cv"` and `"grouped_cv_leakage_audit_metadata"`. The single-seed prefix check is retained under `"prefix_forecast_grouped_cv_1seed_641events"` for complete auditability.
3. **Paper Documentation:** Updated `RESEARCH_PAPER.md` Table III caption and text to explicitly specify the protocol:
   - Conjunction events: 836
   - Features: 58 summary features
   - Protocol: 3-seed 5-fold cross-validation (15 test folds total, seeds 0, 1, 2)
   - Verified artifact: `results/grouped_cv_leakage_audit.json`

The benchmark JSON, research paper, audit reports, and publication figures now match exactly.

---

### 2.3 Issue 3: Correction of Statistical Claims & Confidence Intervals

#### A. Reconciling Paired $t$-Test Values
- **The Issue:** Previous draft versions of `AUDIT_REPORT.md`, `VALIDATION_REPORT.md`, and `PRESENTATION_GUIDE.md` contained stale numbers from an untuned run: $t = -3.914, p = 0.0173$ with RetNet AUC $0.7506$ vs. Transformer $0.8307$.
- **Verification against `results/benchmark_unified_cv.json`:**
  - Standard Transformer: $\text{AUC} = \mathbf{0.8524 \pm 0.0264}$ ($\text{PR-AUC} = 0.6590 \pm 0.0476$)
  - RetNet Surrogate: $\text{AUC} = \mathbf{0.8155 \pm 0.0362}$ ($\text{PR-AUC} = 0.6328 \pm 0.0455$)
  - Paired $t$-test (RetNet vs. Transformer): $t = -2.398$, **$\mathbf{p = 0.0745}$**.
- **Correction:** Removed all instances of $p = 0.0173$ and $t = -3.914$ from all reports. At $p = 0.0745$, the difference does **not** attain statistical significance at the standard $\alpha = 0.05$ threshold.
- **Methodological Qualification (Dietterich, 1998):** As established in machine learning statistical literature, standard paired $t$-tests on $k$-fold cross-validation violate the independence assumption of the Student's $t$-test because training folds share $(k-2)/(k-1) = 75\%$ of instances. This artificially deflates sample variance and inflates Type I errors. Accordingly, all text now explicitly frames this result as an **empirical performance trend favoring softmax attention on sparse tracks**, rather than a formal statistical proof.

#### B. Confidence Intervals (Wilson Score vs. Clopper-Pearson)
- **The Issue:** The previous audit report stated that Clopper-Pearson intervals always geometrically contain Wilson intervals.
- **Mathematical Fact:** Clopper-Pearson guarantees conservative coverage probability ($P(\text{coverage}) \ge 1 - \alpha$), but its interval endpoints do not strictly enclose Wilson endpoints for all $(k, n)$. Specifically, at boundary limits ($k=0$ or $k=n$), the Wilson upper bound can slightly exceed Clopper-Pearson.
- **Verification Evidence:** At $n = 50, k = 0$:
  - Wilson 95% upper bound: $\approx 0.07135$
  - Clopper-Pearson 95% upper bound: $\approx 0.07112$
  - $w_{\text{upper}} > c_{\text{upper}}$ by $\approx 0.00023$.
- Added unit test `test_clopper_pearson_vs_wilson_boundary_property` to `tests/test_mathematical_validation.py` to formally verify and document this property in code.

---

### 2.4 Issue 4: Correction of Benchmark Reporting & Monte Carlo Physics

#### A. Source of the "0.94 Agreement" Claim
- In `results/monte_carlo_vs_surrogate_benchmark.json`, line 80:
  `"fraction_analytical_inside_wilson_95": 0.93`
- An earlier draft confused this $93\%$ quadrature containment metric with an $R^2$ correlation of $0.94$, leading to a fabricated scatter plot in `generate_publication_figures.py` with hardcoded `r2 = 0.942`.
- The false $R^2 \approx 0.94$ statement was excised from `AUDIT_REPORT.md` and all related documentation.

#### B. Actual Measured Benchmark Correlation
From `results/monte_carlo_vs_surrogate_benchmark.json`:
- **Monte Carlo vs. Catalog $\log_{10}(P_c)$:** $R^2 = -1.034$, $\text{MAE} = 0.393$ log-units.
- **Transformer vs. Monte Carlo $\log_{10}(P_c)$:** $R^2 = 0.034$, $\text{MAE} = 0.612$ log-units.
- **HistGradientBoosting vs. Monte Carlo $\log_{10}(P_c)$:** $R^2 = 0.003$, $\text{MAE} = 0.622$ log-units.
- **RetNet vs. Monte Carlo $\log_{10}(P_c)$:** $R^2 = -11.248$, $\text{MAE} = 2.836$ log-units.

#### C. Scientific Clarification Across Four Evaluation Dimensions
1. **Triage Classification ($\max P_c \ge 5 \times 10^{-4}$):** Strong discriminative performance ($0.9086$ AUC for HistGB, $0.8524$ AUC for Transformer, $0.8155$ AUC for RetNet). High-risk conjunctions can be reliably prioritized from track geometry.
2. **Continuous $P_c$ Regression:** Direct continuous estimation of $\log_{10}(P_c)$ has near-zero or negative $R^2$ across all models because public CDMs omit covariance evolution. Predicting exact scalar collision probability without knowing final covariance growth is severely underconstrained.
3. **Numerical Integration Verification:** Monte Carlo 1,000,000-sample sampling reproduces the analytical 2D isotropic Gaussian formula ($93\%$ of analytical values lie within the Wilson 95% confidence interval). This verifies numerical quadrature precision, NOT physical orbit accuracy.
4. **Physical Collision Probability Validation:** Impossible with public Space-Track CDMs because the 2x2 B-plane covariance matrix is withheld. 71% of high-risk cases required artificial radius inflation because isotropic Gaussians cannot produce the peak collision probabilities of real along-track elongated covariance ellipsoids.

#### D. Figure 6 Overhaul
`scripts/generate_publication_figures.py` `fig6_monte_carlo_correlation()` was completely refactored to read real data from `results/monte_carlo_vs_surrogate_benchmark.json`:
- **Panel (a):** Real 100-encounter numerical quadrature points with Wilson 95% confidence intervals, documenting the 93% containment check.
- **Panel (b):** Real surrogate forecast scatter for Transformer, HistGB, and RetNet, displaying the true measured $R^2$ values ($0.034, 0.003, -11.2$) and MAEs.

---

### 2.5 Issue 5: Clean-Up of CAM Language and Implementation

#### A. Removal of Operational Language
- Replaced "Operational Mission-Control CLI & Satellite Conjunction Assessment Tool" with:
  `ORBIT-GUARD: Satellite Conjunction Assessment Research CLI & Verification Tool`.
- Replaced "EXECUTE AT T - 24 HOURS" with:
  `Candidate burn window at T - 24h (Demonstration Heuristic Only)`.
- Replaced all claims of operational flight readiness with explicit research prototype disclaimers.

#### B. Algorithm Alignment & Standby Burn Removal
- In `orbit_guard.py`, `compute_cam_recommendation` was rewritten to align directly with `recommend_cam` in `src/monte_carlo_engine.py`:
  $$\Delta v_t = \frac{\text{gap}}{3 t}$$
  derived from the linearized Hill-Clohessy-Wiltshire (CW) along-track secular drift equations.
- **Removed the hardcoded 0.025 m/s standby burn:** For elevated risk ($10^{-5} \le P_c < 5 \times 10^{-4}$), $\Delta v$ is set to $0.0\text{ m/s}$. The advisory indicates active telemetry tracking and contingency burn planning, clarifying that in satellite flight dynamics standby burns are operational planning states, not executed physical impulses.
- Documented that both CAM routines rely on simplified two-body circular assumptions, omit $J_2/J_3$ geopotential harmonics, drag fluctuations, and secondary catalog screening, and are strictly NOT flight-certified (`is_operational_certified: False`).

---

### 2.6 Issue 6: Critical Redesign of Figure 5 & Separation of Paradigms

#### A. Problem Identified in Previous Figure 5
- The previous implementation assigned an artificial ROC-AUC score of $0.999$ to 2D B-plane Monte Carlo and 3D numerical orbit propagation.
- **Physical Reality:** Monte Carlo quadrature and 3D orbit propagation are physical numerical solvers, not classification models. Assigning ROC-AUC scores to them is scientifically indefensible.
- Additionally, the shaded region was labeled "Flight Avionics Feasibility Envelope", implying flight readiness based merely on millisecond latency.

#### B. Redesign & Architectural Separation
Figure 5 in `scripts/generate_publication_figures.py` was completely redesigned into a two-panel figure:
1. **Subplot (a) Machine Learning Pareto Frontier (AUC vs. Latency):**
   - Strictly limited to evaluated machine learning models (RetNet, HistGB, Transformer, Random Forest, Logistic Regression).
   - Plots measured 5-fold classification ROC-AUC ($\pm$ standard deviation error bars) against measured single-encounter CPU inference latency on a log scale.
   - Shaded region neutrally labeled: **"Low-Latency Regime (< 3 ms CPU)"**.
2. **Subplot (b) Computational Paradigm Latency Comparison:**
   - Horizontal bar chart comparing execution latencies across computational paradigms on a log scale.
   - ML Surrogates (1.21–1.73 ms) vs. 2D B-Plane Monte Carlo ($41.95\text{ ms}$ measured median across 100 encounters with $10^6$ particles) vs. 3D Numerical Orbit Propagation ($\approx 3,000\text{ ms}$).
   - **Explicit Literature Attribution:** 3D orbit propagation is prominently marked with distinct cross-hatching (`//`) and labeled `3D Numerical Orbit Prop [External Estimate]`, with an annotation explicitly clarifying that it is an external NASA/ESA literature baseline, NOT a measurement performed by this project.
   - No artificial AUC values are assigned to physical integrators.

---

### 2.7 Issue 7: Latency Harmonization & Strict Measurement Protocol

#### A. Resolution of Reporting Discrepancies
All latencies across the research paper, LaTeX manuscript, README, presentation guide, validation report, and benchmark JSON artifacts were harmonized to verified microbenchmark numbers:
- **Transformer Latency Discrepancy Resolved:** The paper's earlier $1.388\text{ ms}$ figure was an untuned preliminary run. The verified microbenchmark in `results/benchmark_unified_cv.json` is **$1.728\text{ ms}$** (median single-core CPU, batch size 1, 100 repetitions after 10 warmups; mean $1.924\text{ ms}$). In end-to-end pipeline inference (`monte_carlo_vs_surrogate_benchmark.json`), Transformer execution measured $1.251\text{ ms}$. Both establish sub-2 ms inference.
- **Monte Carlo Latency Discrepancy Resolved:** The paper's earlier $47.8\text{ ms}$ / $44.34\text{ ms}$ figures were single-run drafts. The official 100-encounter benchmark across 1,000,000 samples per encounter (`results/monte_carlo_vs_surrogate_benchmark.json`) measured:
  - Median Latency: **$41.948\text{ ms}$**
  - Mean Latency: **$43.609\text{ ms}$**
- **RetNet Latency:** Median **$1.210\text{ ms}$** (recurrent $O(1)$ step microbenchmark); in-pipeline **$1.098\text{ ms}$**.
- **Speedup Clarification:** RetNet provides a verified **$34.7\times$ speedup** over 2D B-plane Monte Carlo quadrature ($41.95\text{ ms}$; $38.2\times$ in pipeline) and an estimated **$>2,500\times$ speedup** over full 3D numerical orbit propagation ($\approx 3,000\text{ ms}$, external literature estimate).

#### B. Documented Measurement Protocol
Across all project documentation, the measurement protocol is formally documented:
- **Hardware:** Single-core CPU (x86_64), batch size 1.
- **Repetitions:** 100 test iterations evaluated following 10 warmup runs.
- **Exclusions:** Disk I/O, SATCAT cross-referencing, and batch data loading are strictly excluded from the timing loops.

---

### 2.8 Issue 8: Artifact-Driven Figures (Zero Fallback)

- In `scripts/generate_publication_figures.py`, `_load_json(filename)` was rewritten to strictly raise `FileNotFoundError` if any required benchmark JSON artifact is missing.
- Figures 1 and 2 read directly from `results/benchmark_unified_cv.json`.
- Figure 3 reads directly from `results/grouped_cv_leakage_audit.json`.
- Figures 4, 5, and 6 read directly from `results/benchmark_unified_cv.json` and `results/monte_carlo_vs_surrogate_benchmark.json`.
- All fallback code and synthetic data generation routines have been completely removed.

---

### 2.9 Issue 9: Causal Transformer Architecture & Prefix Window Justification

- `TransformerBaseline` in `src/model.py` implements:
  1. `pack_real_first(x)`: Dynamically packs valid physical observations to contiguous leading positions $0 \le t < k$, ensuring zero-token padding sits strictly at the end.
  2. Upper-triangular causal attention mask:
     $$M_{i, j} = \begin{cases} 0, & j \le i \\ -\infty, & j > i \end{cases}$$
     generated via `torch.nn.Transformer.generate_square_subsequent_mask(T)` with `is_causal=True`. This enforces $A_{i, j} = 0$ for all $j > i$.
  3. Physical and Operational Justification: In conjunction assessment, CDMs arrive sequentially leading up to TCA. An encounter observed up to step $k$ is represented by historical prefix $\mathcal{S}_k = \{\mathbf{x}_1, \dots, \mathbf{x}_k\}$. Future CDMs ($t > k$) are physically unobserved. The causal encoder over this historical prefix window evaluates multi-step trajectory evolution up to the decision epoch $k$ without leaking future unobserved CDMs. The pooled representation at step $k$ (`masked_pool(..., pooling="last")`) provides the next-CDM forecast for collision risk at TCA.
- Both `RESEARCH_PAPER.md` (Section 3.3) and `research_paper.tex` (Section 2.3) have been updated with complete mathematical and physical descriptions.

---

### 2.10 Issue 10: Unified CV Reproducibility Script & Pipeline

- Confirmed that `src/benchmark_unified_cv.py` is the canonical script generating `results/benchmark_unified_cv.json`.
- Updated `src/benchmark_unified_cv.py` to write `results/benchmark_unified_cv.json` directly and decoupled it from `run_pipeline.py`.
- Added `src/benchmark_unified_cv.py` to `reproduce_all.py` bundle packaging (`files_to_pack`).
- Added support for `--rerun-unified-cv` in `reproduce_all.py` to allow users to retrain all models across all 5 folds from scratch.
- The packaged bundle now contains 46 files.

---

## 3. End-to-End Verification Evidence

### 3.1 Unit Test Suite Execution
```bash
python -m pytest tests/
```
Output:
```
============================= test session starts =============================
platform win32 -- Python 3.11.9, pytest-8.3.3, pluggy-1.6.0
rootdir: E:\DL_fixed
collected 34 items

tests\test_forecaster_shapes.py ..                                       [  5%]
tests\test_mathematical_validation.py .....................              [ 67%]
tests\test_monte_carlo_engine.py .........                               [ 94%]
tests\test_packing.py ..                                                 [100%]

============================= 34 passed in 3.82s ==============================
```

### 3.2 Publication Figures Generation
```bash
python scripts/generate_publication_figures.py
```
Output:
```
Generating 6 publication-grade figures in PNG (300 DPI) and vector PDF...
Saved Figure 1: fig1_unified_roc_pr_curves (.png & .pdf)
Saved Figure 2: fig2_calibration_reliability (.png & .pdf)
Saved Figure 3: fig3_grouped_cv_leakage_audit (.png & .pdf)
Saved Figure 4: fig4_streaming_horizon_early_warning (.png & .pdf)
Saved Figure 5: fig5_pareto_latency_vs_auc (.png & .pdf)
Saved Figure 6: fig6_monte_carlo_correlation (.png & .pdf)

All publication figures successfully created in: E:\DL_fixed\figures
```

### 3.3 Master Reproducibility Pipeline Execution
```bash
python reproduce_all.py
```
Output:
```
================================================================================
STARTING COMPLETE MASTER REPRODUCIBILITY PIPELINE
================================================================================

======================================================================
STEP 1: VERIFYING SOURCE DATASETS AND INTEGRITY
======================================================================
  [OK] CDMs dataset     (   373.2 KB) | MD5: 18079794cc1403b46d014ee5913a3fd1
  [OK] SATCAT catalog   (  9748.6 KB) | MD5: 77b99affa6ad3aff73990913653d2020

======================================================================
STEP 2: EXECUTING UNIT TEST SUITE (PYTEST)
======================================================================
  [OK] All unit tests passed cleanly (zero NaNs, packing verified).

======================================================================
STEP 3: RUNNING MONTE CARLO & GROUPED CV BENCHMARKS
======================================================================
  [OK] Verified existing unified 5-fold CV benchmark artifact: E:\DL_fixed\results\benchmark_unified_cv.json
  -> Executing 3-seed 5-fold grouped CV leakage audit (836 events)...
  [OK] Grouped CV leakage audit completed successfully.
  -> Executing Monte Carlo vs. Surrogate benchmark (100 events, 10^6 particles)...
  [OK] Monte Carlo benchmark executed successfully.

======================================================================
STEP 4: GENERATING HIGH-RESOLUTION PUBLICATION FIGURES
======================================================================
  [OK] All 6 publication figures created in figures/.

======================================================================
STEP 5: PACKAGING SUBMISSION BUNDLE ZIP
======================================================================
[OK] Submission Bundle packaged successfully: E:\DL_fixed\results\SUBMISSION_BUNDLE.zip (4.85 MB)

================================================================================
MASTER REPRODUCIBILITY PIPELINE COMPLETE! Total Time: 124.90 seconds
================================================================================
```

---

## 4. Git State & Branch Protection

1. **Current Branch:** `feature/operational-monte-carlo-surrogate`
2. **Main Branch:** Pristine at commit `111924d` (`origin/main`, `origin/HEAD`). `main` has never been checked out, committed to, or modified in any way.
3. **Working Tree:** Clean, fully verified, and ready for user inspection.

---

## 5. Final Recommendation

### **VERDICT: READY TO MERGE**

All discrepancy categories identified during the independent review have been fully resolved with rigorous mathematical justification, code hardening, artifact synchronization, figure redesign, and clean-room reproducibility testing.

The feature branch is recommended for merge approval into `main`.
