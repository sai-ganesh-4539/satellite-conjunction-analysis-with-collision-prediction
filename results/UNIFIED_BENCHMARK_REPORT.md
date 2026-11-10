# Unified 5-Fold Cross-Validation Benchmark Report
## Space Debris Conjunction Assessment: Surrogate vs. Baseline Evaluation

**Methodological Integrity Guarantee:**
- **Splits:** 5-Fold Stratified Cross-Validation across all 836 conjunction events (210 high-risk, 626 low-risk).
- **Zero Test Leakage:** Each outer fold has separate inner Train (60%), Validation (20%), and held-out Test (20%).
- **Zero Scaler Leakage:** Standardization fit strictly on inner-train non-padded observations.
- **Probabilistic Calibration:** Temperature scaling fit on validation logits via NLL minimization.
- **Safety Threshold:** Operating threshold $\tau_{\text{tuned}}$ tuned on validation to target $\ge 90\%$ recall, then tested strictly out-of-fold.

---

### Table 1: Comprehensive Model Comparison (Mean ± Std across 5 Test Folds)

| Model Class | Model | AUC-ROC | PR-AUC (AP) | Test Accuracy ($\tau=0.5$) | Test F1 ($\tau=0.5$) | Brier Score ($\downarrow$) | ECE ($\downarrow$) | Measured Latency (Batch=1) |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Classical Physics** | **2D B-Plane Monte Carlo ($10^6$ particles)** | Reference | Reference | N/A | N/A | N/A | N/A | 45.28 ms (Measured) |
| Naive Baseline | **Majority Class** | 0.5000 ± 0.0000 | 0.2512 ± 0.0006 | 0.7488 ± 0.0006 | 0.0000 ± 0.0000 | 0.1881 ± 0.0003 | 0.0003 ± 0.0006 | Not benchmarked |
| Linear Tabular | **Logistic Regression** | 0.8424 ± 0.0299 | 0.6700 ± 0.0653 | 0.7727 ± 0.0436 | 0.6107 ± 0.0396 | 0.1602 ± 0.0289 | 0.1449 ± 0.0377 | 0.094 ms |
| Tree Ensemble | **Random Forest** | 0.9050 ± 0.0357 | 0.7957 ± 0.0684 | 0.8361 ± 0.0401 | 0.6815 ± 0.0617 | 0.1119 ± 0.0198 | 0.1054 ± 0.0235 | 11.19 ms |
| Gradient Boosting | **HistGradientBoosting** | 0.9086 ± 0.0226 | 0.8121 ± 0.0439 | 0.8744 ± 0.0296 | 0.7361 ± 0.0714 | 0.1028 ± 0.0203 | 0.0950 ± 0.0160 | 1.10 ms |
| Deep Sequence | **Standard Transformer** | 0.8307 ± 0.0369 | 0.6440 ± 0.0633 | 0.7655 ± 0.0499 | 0.6091 ± 0.0536 | 0.1586 ± 0.0162 | 0.1444 ± 0.0117 | 0.78 ms |
| Deep Surrogate | **RetNet (Proposed)** | 0.7506 ± 0.0484 | 0.5349 ± 0.0722 | 0.7308 ± 0.0251 | 0.5114 ± 0.0458 | 0.1819 ± 0.0178 | 0.1466 ± 0.0296 | 0.57 ms |

---

### Key Scientific Findings

1. **Fold-Level Descriptive Differences on AUC-ROC (5 Outer Test Folds):**
   - Transformer minus RetNet: mean diff = $+0.0801 \pm 0.0457$ AUC (sample std, $N-1=4$).
   - HistGB minus RetNet: mean diff = $+0.1580 \pm 0.0436$ AUC (sample std, $N-1=4$).
   - *Note on Statistical Reporting:* Fold differences are reported as mean $\pm$ sample standard deviation ($s$, $\text{ddof}=1$, $N-1=4$; population standard deviations across the 5 folds are $\sigma = 0.0409$ and $\sigma = 0.0390$). Inferential p-values from paired t-tests are intentionally omitted because 5-fold cross-validation involves 75% training set overlap between folds, violating the sample independence assumption required for valid hypothesis testing (Dietterich, 1998).
2. **Computational Speedups vs 2D B-Plane Monte Carlo:**
   - RetNet_Recurrent: **80.0x faster** than 1M-particle Monte Carlo quadrature.
   - Transformer: **58.3x faster** than 1M-particle Monte Carlo quadrature.
   - HistGradientBoosting: **41.3x faster** than 1M-particle Monte Carlo quadrature.
   - LogisticRegression: **482.5x faster** than 1M-particle Monte Carlo quadrature.
   - RandomForest: **4.0x faster** than 1M-particle Monte Carlo quadrature.
