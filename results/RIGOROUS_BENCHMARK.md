# Comprehensive Triage Benchmark Report
## Satellite Conjunction Assessment: Classical Time Series, Tabular Ensembles, & Deep Models

**Methodology & Protocol:**
- **Dataset:** 836 unique conjunction encounter pairs (3,351 raw CDMs merged with Space-Track SATCAT).
- **Zero Leakage:** Instantaneous collision probability ($P_c$) is strictly excluded from features.
- **Protocol:** 5-Fold Stratified Cross-Validation with nested validation splits for calibration and threshold selection.
- **Models Benchmarked:** Naive Baseline (Majority Class), Stochastic Model (Markov Chain), Statistical Model (Autoregressive / ARIMA), Linear Tabular (Logistic Regression), Tree Ensembles (Random Forest, HistGradientBoosting), Deep Sequence Models (Standard Transformer, RetNet Surrogate).
- **Architecture Updates:** Evaluated with revised `model.py` featuring real-step packing (`pack_real_first`), masked temporal pooling (`masked_pool`), and Sun et al. (2023) multi-scale matrix retention.

---

### Table 1: Complete Model Comparison under 5-Fold Cross-Validation (Mean ± Std)

| Model Category | Model Architecture | AUC-ROC | PR-AUC (Average Precision) | Accuracy ($\tau=0.5$) | F1-Score ($\tau=0.5$) | Brier Score ($\downarrow$) | ECE ($\downarrow$) | Measured Latency (Batch=1) |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Naive Baseline** | **Majority Class** | 0.5000 ± 0.0000 | 0.2512 ± 0.0006 | 0.7488 ± 0.0006 | 0.0000 ± 0.0000 | 0.1881 ± 0.0003 | 0.0003 ± 0.0006 | Not benchmarked |
| **Stochastic Model** | **Markov Chain (MC)** | 0.5777 ± 0.0197 | 0.2849 ± 0.0091 | 0.7488 ± 0.0006 | 0.0000 ± 0.0000 | 0.1830 ± 0.0028 | **0.0179 ± 0.0118** | Not benchmarked |
| **Statistical Model** | **Autoregressive / ARIMA** | 0.7838 ± 0.0481 | 0.6351 ± 0.0828 | 0.7308 ± 0.0384 | 0.5750 ± 0.0544 | 0.1835 ± 0.0242 | 0.1917 ± 0.0335 | Not benchmarked |
| **Linear Tabular** | **Logistic Regression** | 0.8424 ± 0.0299 | 0.6700 ± 0.0653 | 0.7727 ± 0.0436 | 0.6107 ± 0.0396 | 0.1602 ± 0.0289 | 0.1449 ± 0.0377 | 0.094 ms |
| **Tree Ensemble** | **Random Forest (200 trees)** | 0.9050 ± 0.0357 | 0.7957 ± 0.0684 | 0.8361 ± 0.0401 | 0.6815 ± 0.0617 | 0.1119 ± 0.0198 | 0.1054 ± 0.0235 | 11.19 ms |
| **Gradient Boosting**| **HistGradientBoosting** | **0.9086 ± 0.0226** | **0.8121 ± 0.0439** | **0.8744 ± 0.0296** | **0.7361 ± 0.0714** | **0.1028 ± 0.0203** | 0.0950 ± 0.0160 | 1.10 ms |
| **Deep Sequence** | **Standard Transformer** | 0.8307 ± 0.0369 | 0.6440 ± 0.0633 | 0.7655 ± 0.0499 | 0.6091 ± 0.0536 | 0.1586 ± 0.0162 | 0.1444 ± 0.0117 | 0.78 ms |
| **Deep Surrogate** | **RetNet (Proposed)** | 0.7506 ± 0.0484 | 0.5349 ± 0.0722 | 0.7308 ± 0.0251 | 0.5114 ± 0.0458 | 0.1819 ± 0.0178 | 0.1466 ± 0.0296 | **0.57 ms** (Recurrent) |

---

### Table 2: Operational Triage Performance (Validation Target $\ge 90\%$ Recall)

| Model | Operating Threshold ($\tau$) | Test Safety Recall | Test Precision | Flagged for Detailed Run (%) | Screened Out Safely (%) |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Logistic Regression** | $0.31 \pm 0.08$ | 85.71% ± 4.99% | 44.22% ± 5.15% | 48.7% | 51.3% |
| **Random Forest** | $0.30 \pm 0.04$ | **92.38% ± 4.86%** | 50.15% ± 5.25% | 46.3% | 53.7% |
| **HistGradientBoosting** | $0.13 \pm 0.19$ | 84.29% ± 5.55% | **61.24% ± 10.72%** | **34.6%** | **65.4%** |
| **Standard Transformer** | $0.35 \pm 0.05$ | 88.57% ± 5.91% | 41.64% ± 4.55% | 53.4% | 46.6% |
| **RetNet Surrogate (Ours)**| $0.30 \pm 0.05$ | 84.29% ± 7.16% | 34.33% ± 4.32% | 61.7% | 38.3% |

*Screening Formula:* $\text{Flag Rate} = (\text{Prevalence} \times \text{Recall}) / \text{Precision}$.

---

### Table 3: Satellite Overlap Audit (Grouped Cross-Validation across 3 Seeds)

| Grouping Scheme | Leakage Constraint | HistGradientBoosting AUC | Random Forest AUC | Logistic Regression AUC |
| :--- | :--- | :---: | :---: | :---: |
| **Stratified (Standard)** | Random encounters | $0.954 \pm 0.028$ | $0.922 \pm 0.022$ | $0.833 \pm 0.035$ |
| **Group by SAT 1** | Target satellite unseen in test | $0.957 \pm 0.025$ | $0.924 \pm 0.028$ | $0.839 \pm 0.038$ |
| **Group by SAT 2** | Chaser satellite unseen in test | $0.950 \pm 0.026$ | $0.923 \pm 0.027$ | $0.836 \pm 0.022$ |
| **Group by Component (Union-Find)** | **Zero satellite overlap across splits** | **$0.777 \pm 0.053$** | **$0.802 \pm 0.045$** | **$0.823 \pm 0.051$** |

---

### Table 4: Streaming Horizon Analysis (AUC vs. Observations Received)

| Step ($k$) | CDM Update Stage | HistGradientBoosting AUC | Logistic Regression AUC |
| :---: | :--- | :---: | :---: |
| $k = 1$ | Initial Detection (7 days pre-TCA) | $0.8543 \pm 0.0356$ | $0.6717 \pm 0.0306$ |
| $k = 2$ | Second Radar Track Update | $0.9293 \pm 0.0243$ | $0.7899 \pm 0.0338$ |
| $k = 3$ | Third Radar Track Update | $0.9243 \pm 0.0259$ | $0.8035 \pm 0.0410$ |
| $k = 4$ | Late-stage conjunction tracking | $0.9324 \pm 0.0276$ | $0.8228 \pm 0.0375$ |

---

### Table 5: Subgroup Analysis by Track Length

| Subgroup | Number of Encounters | Positive Rate | HistGB AUC | RetNet AUC | Transformer AUC |
| :--- | :---: | :---: | :---: | :---: | :---: |
| Short ($N \le 3$ CDMs) | 506 (60.5%) | 13.4% | $0.8948 \pm 0.0528$ | $0.742 \pm 0.051$ | $0.825 \pm 0.038$ |
| Long ($N \ge 4$ CDMs) | 330 (39.5%) | 43.0% | $0.9254 \pm 0.0214$ | $0.768 \pm 0.054$ | $0.838 \pm 0.042$ |
