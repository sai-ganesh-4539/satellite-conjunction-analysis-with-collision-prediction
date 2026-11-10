# Satellite Conjunction Assessment via Physics-Enriched Sequence and Tabular Models: A Comparative Study of Triage Screening

**Authors:** Sai Ganesh, Department of Computer Science & Engineering  
**Dataset:** 3,351 Conjunction Data Messages (CDMs) across 836 unique orbital encounters fused with 70,000+ Space-Track SATCAT records  
**Repository:** `E:\DL_fixed`

---

## Abstract

As the catalog of tracked space objects expands, satellite operations face an escalating volume of conjunction warnings. Operational workflows rely on high-fidelity numerical orbit propagation and Monte Carlo covariance sampling to compute collision probabilities ($P_c$), which is computationally intensive and difficult to scale across all pairwise combinations in crowded orbital shells. In this work, we evaluate machine learning triage classifiers to identify high-risk conjunction events ($\max P_c \ge 5 \times 10^{-4}$) directly from sequences of Conjunction Data Messages (CDMs) enriched with orbital physics from the Space-Track Satellite Catalog (SATCAT). 

We benchmark a diverse hierarchy of models under a unified 5-fold cross-validation protocol on 836 conjunction events: naive baselines (Majority Class), classical stochastic and statistical time series models (Markov Chain, Autoregressive/ARIMA), linear regression, tree ensembles (Random Forest, HistGradientBoosting), and deep sequence architectures (Standard Transformer, Retentive Network / RetNet). Tabular gradient boosting achieves the highest discriminative performance ($\text{AUC-ROC} = 0.9086 \pm 0.0226$, $\text{PR-AUC} = 0.8121 \pm 0.0439$). Among sequence models, RetNet with Sun et al. (2023) multi-scale matrix retention achieves $\text{AUC-ROC} = 0.8155 \pm 0.0362$, while the Standard Transformer achieves $\text{AUC-ROC} = 0.8524 \pm 0.0264$. A paired two-tailed $t$-test ($n = 5, t = -2.398, p = 0.0745$) indicates a trend favoring Transformer attention over linear retention on sparse sequences (though falling short of the conventional $\alpha = 0.05$ significance threshold), with RetNet maintaining constant $O(1)$ recurrent memory and sub-2 ms latency.

Analysis of sequence length shows that 59.5% of encounters contain only 1 to 3 CDM updates (median length = 3.0), leaving sequence models with limited temporal signal to model. Subgroup experiments demonstrate that while gradient boosting maintains strong performance across both short ($N \le 3, \text{AUC} = 0.8948$) and longer histories ($N \ge 4, \text{AUC} = 0.9254$), deep sequence models remain near $\text{AUC} \approx 0.814\text{--}0.818$. Furthermore, streaming horizon analysis reveals that the primary geometric risk signals are largely resolved by the second observation ($k = 2, \text{AUC} = 0.9293$). At validation-tuned operating thresholds targeting $\ge 90\%$ recall, out-of-fold test recall reached $\approx 84\%\text{--}88\%$ across models (with Random Forest reaching $92.4\%$). In an operational triage configuration, gradient boosting safely screens out $65.4\%$ of non-critical events, while RetNet screens out $38.3\%$. Furthermore, under strict graph-component cross-validation with zero satellite overlap between train and test, tree models experience an 18-point drop while regularized linear models maintain solid generalization ($\text{AUC} = 0.823 \pm 0.051$), demonstrating that tree models partially rely on static spacecraft identity features. These findings delineate the boundary conditions where tabular and sequence architectures operate effectively in space situational awareness.

Finally, we implement a vectorized 1,000,000-sample B-plane Monte Carlo simulator and evaluate surrogate speedups. We document that public CDMs lack 2x2 B-plane covariance matrices, requiring an isotropic scale calibration against catalog $P_c$ under which 71% of high-risk cases require radius inflation due to real-world along-track covariance elongation; the resulting Monte Carlo simulations serve as numerical sampling verification of analytical quadrature, rather than independent orbit determination. Computational profiling reveals that RetNet inference ($1.21\text{ ms}$) provides a $15\times\text{--}40\times$ acceleration over 2D B-plane Monte Carlo sampling ($48\text{ ms}$) and an estimated $>2,500\times$ speedup over full 3D numerical orbit propagation ($3,000\text{--}5,000\text{ ms}$). An illustrative Collision Avoidance Maneuver (CAM) module is provided for demonstration, with explicit flight safety disclaimers.

---

## 1. Introduction

With more than 36,000 trackable objects exceeding 10 cm in Earth orbit and mega-constellations deploying thousands of active satellites, conjunction assessment (CA) has become a primary bottleneck for space situational awareness (SSA). Satellite operators receive daily streams of Conjunction Data Messages (CDMs) produced by the 18th Space Defense Squadron via Space-Track.org. Each CDM encapsulates state vectors, relative miss distance, positional covariance matrices, and estimated collision probability ($P_c$) for two orbiting objects at the Time of Closest Approach (TCA).

Operational spaceflight safety typically manages risk through a multi-tiered pipeline:
1. **Screening Volumes:** Coarse geometric filters identify object pairs passing within predefined thresholds (e.g., 5 km miss distance in LEO).
2. **Analytical Approximations:** Two-dimensional Foster or Akella formulas compute short-term $P_c$ under linear relative motion and constant Gaussian covariance.
3. **High-Fidelity Numerical Propagation:** When covariance geometry is non-linear or atmospheric drag uncertainty is high, Monte Carlo simulations propagate thousands to millions of sampled trajectories to compute numerical collision probabilities.

Because high-fidelity propagation across hundreds of thousands of candidate encounters is computationally demanding, automated machine learning triage classifiers have been proposed to prioritize high-risk events. However, existing literature presents several methodological challenges:
- **Direct Target Leakage:** Multiple studies include instantaneous $P_c$ from Space-Track within input features while predicting peak encounter $P_c$, resulting in trivial near-perfect accuracy that fails in operational deployment.
- **Absence of Classical & Tabular Baselines:** Deep sequence models are often published without comparison against classical time-series models (such as Markov Chains and ARIMA) or gradient-boosted decision trees.
- **Uncalibrated Probabilities and Symmetric Thresholds:** Standard models often report metrics at an uncalibrated default threshold ($\tau = 0.5$). In satellite safety, false negatives risk spacecraft collision, requiring decision thresholds tuned specifically for high recall with calibrated posterior probabilities.
- **Satellite Overlap Leakage:** Evaluating conjunction events with standard randomized splits allows the same satellite (and its static RCS or exclusion volume) to appear in both train and test splits, artificially inflating decision tree scores.

### Contributions
To address these issues systematically, this paper provides:
1. **A Leak-Free Astrodynamics Benchmark:** We extract 3,351 CDMs across 836 unique encounters, strictly excluding $P_c$ from features and incorporating 14 orbital mechanics features from the Space-Track Satellite Catalog (SATCAT).
2. **Comprehensive Model Comparison:** We evaluate eight architectures spanning naive baselines, Markov Chain state transitions, Autoregressive/ARIMA time-series models, Logistic Regression, Random Forest, HistGradientBoosting, Standard Transformer, and RetNet.
3. **Unified 5-Fold Cross-Validation:** All models are evaluated under the same 5-fold cross-validation with nested validation for temperature scaling calibration and high-recall threshold tuning.
4. **Padding-Insensitive Sequence Modeling:** Sequences are dynamically packed (`pack_real_first`) so that causal attention operates strictly over real physical observations without zero-token padding corruption or mask-induced NaN instabilities.
5. **Multi-Scheme Grouped CV Audit:** We evaluate models under strict graph-component grouping (Union-Find) ensuring zero satellite overlap across splits, revealing that tree models drop from 0.954 to 0.777 while regularized linear models generalize robustly (0.823).
6. **Streaming Horizon & Subgroup Experiments:** We investigate how model performance evolves across observation steps ($k = 1, 2, 3, 4$) and analyze performance on short ($N \le 3$) versus longer ($N \ge 4$) observation histories.

---

## 2. Problem Formulation & Feature Space

### 2.1 Conjunction Data Messages (CDMs)
A CDM sequence represents an encounter between a primary satellite and a secondary space object (payload, rocket body, or debris) over a multi-day monitoring window preceding TCA. Let an encounter $i$ be represented by:
$$\mathcal{S}_i = \{\mathbf{x}_1, \mathbf{x}_2, \dots, \mathbf{x}_{T_i}\}, \quad \mathbf{x}_t \in \mathbb{R}^D$$
where $T_i$ is the number of updates received and $D = 14$ is the feature dimension.

### 2.2 Prediction Target
Because Space-Track publishes CDMs only when initial screening exceeds $P_c \ge 10^{-4}$, the operational task is formulated as identifying whether the encounter will escalate to high collision risk requiring avoidance maneuver planning:
$$y_i = \begin{cases} 1, & \text{if } \max_{t \in [1, T_i]} P_c(t) \ge 5 \times 10^{-4} \\ 0, & \text{otherwise} \end{cases}$$
Across the 836 encounters, 210 events (25.12%) are labeled positive, reflecting a 1:3 class imbalance. The instantaneous value $P_c(t)$ is strictly excluded from $\mathbf{x}_t$.

### 2.3 Input Features (Physics-Enriched via SATCAT)
Features at each observation step include:
- `log_min_rng`: $\log_{10}(\text{MIN\_RNG})$ (miss distance at TCA in meters).
- `delta_min_rng`: Consecutive change in miss distance ($\Delta \text{RNG}_t / 100\text{ m}$).
- `range_ratio`: Normalized range relative to initial observation ($\text{RNG}_t / \text{RNG}_1$).
- `norm_step_pos`: Normalized observation step ($t / T_i$).
- `delta_apogee`: $\log_{10} |\text{Apogee}_1 - \text{Apogee}_2|$ (orbital shell altitude difference in km).
- `delta_perigee`: $\log_{10} |\text{Perigee}_1 - \text{Perigee}_2|$ (km).
- `delta_inclination`: $|\text{Inc}_1 - \text{Inc}_2|$ (plane crossing difference in degrees).
- `delta_period`: $|\text{Period}_1 - \text{Period}_2|$ (period difference in minutes).
- `sat1_rcs`, `sat2_rcs`: Radar cross section categories (0: Small, 1: Medium, 2: Large).
- `sat1_type`, `sat2_type`: Object classification (0: Debris, 1: Payload, 2: Rocket Body).
- `sat1_excl_vol`, `sat2_excl_vol`: Spacecraft hard-body exclusion radii (meters).

---

## 3. Evaluated Model Architectures

### 3.1 Naive & Classical Time-Series Models
- **Majority Class:** Predicts constant empirical training prevalence ($P(Y=1) \approx 0.251$).
- **Markov Chain (MC):** Discretizes miss distance into three states: Safe ($r \ge 3\text{ km}$), Warning ($500\text{ m} \le r < 3\text{ km}$), and Critical ($r < 500\text{ m}$). Computes an empirical $3 \times 3$ transition probability matrix $\mathbf{P}(S_t \mid S_{t-1})$ and assigns risk from historical state labels.
- **Autoregressive / ARIMA Model:** Fits an autoregressive model on the miss distance sequence ($\text{RNG}_t$), capturing the final miss distance, minimum distance, trend slope, and observation length, calibrated via logistic regression.
- **Logistic Regression:** Regularized linear model fit on 71 summary features (last step, mean, min, max, trend, and length).

### 3.2 Tree Ensembles
- **Random Forest:** 200 balanced decision trees trained on tabular summary features.
- **HistGradientBoosting:** Histogram-based gradient boosting trees with balanced sample weighting.

### 3.3 Deep Sequence Models
- **Standard Transformer:** 3-layer Transformer encoder with causal multi-head self-attention (4 heads, $d_{\text{model}} = 64, d_{\text{ff}} = 256$, dropout = 0.1, 155,201 parameters). Padded tokens are gathered to the end via `pack_real_first`.
- **RetNet Surrogate:** 3-layer Retentive Network using Sun et al. (2023) multi-scale matrix retention with decay parameters $\gamma \in \{0.90, 0.95, 0.98, 0.99\}$, SwiGLU feed-forward networks, and $d_{\text{model}} = 64$ (104,801 parameters). RetNet supports both parallel batch training and recurrent single-step inference ($S_t = \gamma S_{t-1} + K_t^T V_t$).

---

## 4. Experimental Results

### Table 1: Model Comparison under Unified 5-Fold Cross-Validation (Mean ± Std)
*All metrics represent held-out test fold performance across identical 5 folds on 836 events.*

| Model Category | Model Architecture | AUC-ROC | PR-AUC (Average Precision) | Accuracy ($\tau=0.5$) | F1-Score ($\tau=0.5$) | Brier Score ($\downarrow$) | ECE ($\downarrow$) | Measured Latency (Batch=1) |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Classical Physics** | **Monte Carlo ($10^6$ sims, 2D B-Plane)** | Reference | Reference | N/A | N/A | N/A | N/A | 47.8 ms (Local CPU) / 3,000–5,000 ms (3D Prop) |
| **Naive Baseline** | **Majority Class** | 0.5000 ± 0.0000 | 0.2512 ± 0.0006 | 0.7488 ± 0.0006 | 0.0000 ± 0.0000 | 0.1881 ± 0.0003 | 0.0003 ± 0.0006 | < 0.05 ms |
| **Stochastic Model** | **Markov Chain (MC)** | 0.5777 ± 0.0197 | 0.2849 ± 0.0091 | 0.7488 ± 0.0006 | 0.0000 ± 0.0000 | 0.1830 ± 0.0028 | 0.0179 ± 0.0118 | < 0.10 ms |
| **Statistical Model** | **Autoregressive / ARIMA** | 0.7838 ± 0.0481 | 0.6351 ± 0.0828 | 0.7308 ± 0.0384 | 0.5750 ± 0.0544 | 0.1835 ± 0.0242 | 0.1917 ± 0.0335 | < 0.15 ms |
| **Linear Tabular** | **Logistic Regression** | 0.8424 ± 0.0299 | 0.6700 ± 0.0653 | 0.7727 ± 0.0436 | 0.6107 ± 0.0396 | 0.1602 ± 0.0289 | 0.1449 ± 0.0377 | 0.16 ms |
| **Tree Ensemble** | **Random Forest (200 trees)** | 0.9050 ± 0.0357 | 0.7957 ± 0.0684 | 0.8361 ± 0.0401 | 0.6815 ± 0.0617 | 0.1119 ± 0.0198 | 0.1054 ± 0.0235 | 7.77 ms |
| **Gradient Boosting**| **HistGradientBoosting** | **0.9086 ± 0.0226** | **0.8121 ± 0.0439** | **0.8744 ± 0.0296** | **0.7361 ± 0.0714** | **0.1028 ± 0.0203** | **0.0950 ± 0.0160** | 1.56 ms |
| **Deep Sequence** | **Standard Transformer** | 0.8524 ± 0.0264 | 0.6590 ± 0.0476 | 0.7858 ± 0.0531 | 0.6227 ± 0.0471 | 0.1475 ± 0.0152 | 0.1372 ± 0.0210 | 1.39 ms |
| **Deep Surrogate** | **RetNet (Proposed)** | 0.8155 ± 0.0362 | 0.6328 ± 0.0455 | 0.7679 ± 0.0414 | 0.5583 ± 0.0749 | 0.1522 ± 0.0187 | 0.0960 ± 0.0228 | 1.21 ms |

---

### Table 2: Safety-Critical Triage Performance (Validation Target $\ge 90\%$ Recall)
*The operating threshold $\tau$ is tuned on inner validation folds to target $\ge 90\%$ recall, and evaluated out-of-fold on test sets.*

| Model | Operating Threshold ($\tau$) | Test Safety Recall | Test Precision | Test F1 Score | Flagged for Detailed Run (%) | Screened Out Safely (%) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Logistic Regression** | $0.31 \pm 0.08$ | 85.71% ± 4.99% | 44.22% ± 5.15% | 0.5804 ± 0.0399 | 48.7% | 51.3% |
| **Random Forest** | $0.30 \pm 0.04$ | **92.38% ± 4.86%** | 50.15% ± 5.25% | 0.6495 ± 0.0544 | 46.3% | 53.7% |
| **HistGradientBoosting** | $0.13 \pm 0.19$ | 84.29% ± 5.55% | **61.24% ± 10.72%** | **0.7034 ± 0.0741** | **34.6%** | **65.4%** |
| **Standard Transformer** | $0.35 \pm 0.06$ | 84.76% ± 6.14% | 45.05% ± 4.56% | 0.5851 ± 0.0330 | 47.3% | 52.7% |
| **RetNet Surrogate (Ours)** | $0.20 \pm 0.11$ | 85.24% ± 5.91% | 40.92% ± 3.37% | 0.5513 ± 0.0306 | 52.3% | 47.7% |

*Note on Screening Rates:* The proportion of events flagged for follow-up is calculated from test predictions ($\text{Flag Rate} = \frac{\text{Prevalence} \times \text{Recall}}{\text{Precision}}$). HistGradientBoosting achieves the highest screening rate, safely eliminating 65.4% of candidate encounters while capturing 84.3% of true high-risk events.

---

### Table 3: Satellite Overlap Audit (Grouped Cross-Validation)
To evaluate sensitivity to satellite identity leakage, models were benchmarked across four split schemes across 3 random seeds:

| Grouping Scheme | Leakage Constraint | HistGradientBoosting AUC | Random Forest AUC | Logistic Regression AUC |
| :--- | :--- | :---: | :---: | :---: |
| **Stratified (Standard)** | None (random encounters) | $0.954 \pm 0.028$ | $0.922 \pm 0.022$ | $0.833 \pm 0.035$ |
| **Group by SAT 1** | Target satellite unseen in test | $0.957 \pm 0.025$ | $0.924 \pm 0.028$ | $0.839 \pm 0.038$ |
| **Group by SAT 2** | Chaser satellite unseen in test | $0.950 \pm 0.026$ | $0.923 \pm 0.027$ | $0.836 \pm 0.022$ |
| **Group by Component (Union-Find)** | **Zero satellite overlap across splits** | **$0.777 \pm 0.053$** | **$0.802 \pm 0.045$** | **$0.823 \pm 0.051$** |

**Finding:** When completely isolated from familiar satellites (`group_component`), tree ensembles drop by $\approx 18$ points (0.954 to 0.777). Decision trees partially memorized static satellite attributes (`SAT1_EXCL_VOL`, RCS, orbital inclinations) that correlate with recurring high-traffic constellations. Regularized linear regression, by contrast, generalizes with high stability ($0.823 \pm 0.051$).

---

## 5. Sequence Dynamics & Subgroup Analysis

### 5.1 Streaming Horizon Analysis: AUC vs. Number of Observed CDMs
To test how early a triage decision can be made as CDMs arrive sequentially:

| Observation Horizon | Number of CDMs Used ($k$) | HistGradientBoosting AUC | Logistic Regression AUC |
| :---: | :---: | :---: | :---: |
| Initial Detection | $k = 1$ | $0.8543 \pm 0.0356$ | $0.6717 \pm 0.0306$ |
| 2nd Radar Update | $k = 2$ | $0.9293 \pm 0.0243$ | $0.7899 \pm 0.0338$ |
| 3rd Radar Update | $k = 3$ | $0.9243 \pm 0.0259$ | $0.8035 \pm 0.0410$ |
| 4th Radar Update | $k = 4$ | $0.9324 \pm 0.0276$ | $0.8228 \pm 0.0375$ |

**Finding:** At the very first update ($k = 1$), HistGB achieves $\text{AUC} = 0.854$. By the second update ($k = 2$), performance stabilizes at $\text{AUC} \approx 0.929$. Subsequent observations provide incremental refinement, indicating that primary orbital geometry signals are resolved early in the tracking campaign.

### 5.2 Subgroup Analysis by Sequence Length
We partitioned the 836 encounters into short ($N \le 3$ CDMs) and extended ($N \ge 4$ CDMs) histories:

| Subgroup | Number of Events | High-Risk Prevalence | HistGB AUC | RetNet AUC | Transformer AUC |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Short Histories ($N \le 3$)** | 506 (60.5%) | 13.4% (68 pos) | $0.8948 \pm 0.0528$ | $0.742 \pm 0.051$ | $0.825 \pm 0.038$ |
| **Long Histories ($N \ge 4$)** | 330 (39.5%) | 43.0% (142 pos) | $0.9254 \pm 0.0214$ | $0.768 \pm 0.054$ | $0.838 \pm 0.042$ |
| **Extended Histories ($N \ge 6$)** | 201 (24.0%) | 56.7% (114 pos) | $0.9101 \pm 0.0296$ | — | — |

---

## 6. Discussion & Limitations

### 6.1 Statistical Significance & Model Comparison
Across the 5 outer test folds:
- **RetNet vs. Transformer:** A paired two-tailed $t$-test yields $t = -2.398, p = 0.0745$ ($n = 5$, 4 degrees of freedom). While the Transformer shows higher mean discrimination ($0.8524$ vs. $0.8155$), this difference does not attain formal statistical significance at the standard $\alpha = 0.05$ threshold. Furthermore, as established by Dietterich (1998), standard paired $t$-tests on $k$-fold cross-validation violate sample independence due to overlapping training sets (each pair of training sets shares 75% of events), which inflates Type I error rates. Consequently, this difference must be characterized as an empirical performance trend on sparse sequences rather than a proven universal inequality.
- **HistGB vs. RetNet:** Paired $t$-test yields $t = -8.771, p = 0.0009$, confirming that gradient-boosted trees on summary features strongly outperform linear sequence retention on this small-sample regime.
- **Why Softmax Attention Outperforms Linear Matrix Retention on Sparse CDMs:** In Sun et al. (2023), retention uses linear decay-weighted dot products without softmax normalization across timesteps:
  $$o_n = \sum_{i \le n} \gamma^{n-i} (q_n \cdot k_i) v_i$$
  In dense, long natural language contexts ($L > 1024$), decaying linear attention excels. However, in satellite conjunctions, sequences are extremely short (median 3 CDMs, maximum 14), with sporadic updates separated by 12–24 hours. Under these conditions, the Transformer's bounded softmax normalization $\text{softmax}(QK^T / \sqrt{d})$ provides critical sharpening, allowing the model to focus sharply on the single critical update where miss distance dropped precipitously.
- **RetNet's Trade-Off:** While RetNet has lower discrimination on sparse short tracks ($0.751$ vs $0.831$), it retains 32.6% fewer parameters (104,801 vs 155,201) and constant $O(1)$ memory, making it well-suited for streaming edge hardware on flight avionics.

### 6.2 Latency and Computational Complexity
Measured median CPU latencies at batch size 1 are:
- RetNet Recurrent: $1.210\text{ ms}$
- HistGradientBoosting: $1.557\text{ ms}$
- Standard Transformer: $1.728\text{ ms}$
- Random Forest: $7.772\text{ ms}$

The $0.3\text{--}0.5\text{ ms}$ difference between RetNet, HistGB, and Transformer is within framework overhead. For short sequences ($L \le 14$), quadratic attention costs only $14^2 = 196$ dot products per head, so $O(L^2)$ attention is not an operational bottleneck at this scale.

### 6.3 Limitations & Boundary Conditions
1. **Absence of Full B-Plane Covariance in Public CDMs:** Space-Track public CDMs omit the full 2x2 B-plane covariance matrix. To establish a standardized numerical baseline, an isotropic scale $\sigma$ was calibrated against catalog $P_c$. In 71% of high-risk encounters, achieving the catalog $P_c$ within an isotropic Gaussian required inflating the hard-body radius beyond physical dimensions (median inflation $3.8\times$), reflecting the physical reality that real-world encounters feature highly elongated, anisotropic covariances along the orbit track. Consequently, Monte Carlo simulations in this pipeline serve as numerical integration sampling verification of the calibrated Gaussian, not independent orbit determination.
2. **Collision Avoidance Maneuver (CAM) Heuristic:** The CAM module in `orbit_guard.py` implements a linearized Clohessy-Wiltshire along-track secular drift calculation for illustrative mission-control demonstration. Real-world flight operations strictly require high-fidelity numerical ephemeris propagation (including $J_{70,70}$ gravity, atmospheric drag covariance, SRP), attitude slew constraints, thruster dispersion budgets, and mandatory catalog-wide secondary conjunction screening to ensure maneuvers do not induce secondary collisions.
3. **Sequence Length Sparsity:** In our 836-encounter catalog, 59.5% of events contain only 1 to 3 CDM updates. Sequence models (RetNet and Transformer) have minimal temporal context to leverage, allowing tabular tree ensembles on summary statistics to dominate discriminative performance.
4. **Single-Epoch Observation Window:** The Space-Track CDMs fall within a concentrated observational window, leaving generalization across different solar storm cycles unexamined.
5. **Exclusion Volume Feature Importance:** `SAT1_EXCL_VOL` appears with high permutation importance in tree models. Because exclusion volume is an input in Space-Track's internal $P_c$ calculation, its predictive value partly reflects the physical definition of the target label, contributing to the 18-point drop observed when evaluating on unseen satellites.

---

## 7. Conclusion

On 836 satellite conjunction events under unified 5-fold cross-validation, tabular gradient boosting ($\text{AUC} = 0.9086$) outperformed both deep sequence architectures (Transformer $\text{AUC} = 0.8524$, RetNet $\text{AUC} = 0.8155$). Sequence models provided no detectable advantage over tabular ensembles, primarily because 59.5% of events have $\le 3$ updates and risk is strongly governed by static orbital shell geometry. At validation-tuned operating thresholds targeting $\ge 90\%$ recall, models achieved $\approx 84\%\text{--}88\%$ out-of-fold test recall, with gradient boosting functioning as the most effective triage filter by safely screening out $65.4\%$ of events.

---

## References

1. Vaswani, A., et al. "Attention is all you need." *NeurIPS*, 2017.
2. Sun, Y., et al. "Retentive Network: A successor to Transformer for large language models." *arXiv:2307.08621*, 2023.
3. Touvron, H., et al. "LLaMA: Open and efficient foundation language models." *arXiv:2302.13971*, 2023.
4. Lin, T.-Y., et al. "Focal loss for dense object detection." *ICCV*, 2017.
5. Foster, J. L., & Estes, H. S. "A Parametric Analysis of Orbital Debris Collision Probability." *NASA JSC-25898*, 1992.
6. Akella, M. R., & Alfriend, K. T. "Probability of collision between space objects." *JGCD*, 2000.
7. Box, G. E., & Jenkins, G. M. *Time Series Analysis: Forecasting and Control*. Holden-Day, 1976.
8. Kemeny, J. G., & Snell, J. L. *Finite Markov Chains*. Springer, 1976.
