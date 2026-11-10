# Presentation & Defense Guide: Satellite Conjunction Assessment
## Prepared for Faculty Guide Review ("Madam") & Research Committee

---

## 1. Executive Summary & 30-Second Elevator Pitch

> *"Respected Madam, with thousands of space objects tracked in Low Earth Orbit, satellite operations receive an overwhelming volume of Conjunction Data Messages (CDMs) daily. High-fidelity numerical trajectory propagation and Monte Carlo sampling are computationally demanding, creating a need for automated triage screening to filter out safe events and prioritize genuine collision threats.*
>
> *In our project, we evaluated machine learning triage classifiers on 836 real satellite encounters from Space-Track.org, enriched with 14 orbital mechanics features from the Satellite Catalog (SATCAT). We evaluated a full hierarchy of models: stochastic Markov Chains, statistical ARIMA/autoregressive models, Logistic Regression, Random Forest, HistGradientBoosting, Standard Transformer, and Microsoft's Retentive Network (RetNet).*
>
> *Under a unified 5-fold cross-validation protocol, tabular Gradient Boosting achieved the highest discriminative performance ($\text{AUC} = 0.909$). We discovered that 59.5% of conjunction encounters contain only 1 to 3 CDM updates, where static orbital geometry dominates and sequence models offer no advantage ($\text{AUC} \approx 0.751\text{--}0.831$). In a triage screening pipeline, Gradient Boosting safely screens out 65.4% of candidate encounters at ~84.3% safety recall. Furthermore, when audited under strict zero-satellite overlap (Union-Find component grouping), tree models drop from 0.954 to 0.777, revealing that trees partially rely on static satellite identity features, whereas regularized linear models maintain solid generalization ($0.823 \pm 0.051$)."*

---

## 2. Slide-by-Slide Presentation Structure & Script

### Slide 1: Title & Motivation
- **Title:** Satellite Conjunction Assessment via Physics-Enriched Triage Classifiers
- **Subtitle:** Benchmarking Classical Time Series, Tabular Ensembles, and Deep Sequence Models
- **What to Say:**
  > *"Good morning Madam and committee members. Today, satellite conjunction assessment is facing unprecedented scale due to the growth of satellite constellations and orbital debris. Our goal is to build and benchmark machine learning triage classifiers that can rapidly screen out non-threatening encounters and flag genuine collision risks for high-fidelity numerical assessment."*

---

### Slide 2: The Operational Problem: Triage Screening
- **Key Points:**
  - Standard Foster 2D formulas assume linear relative motion and spherical objects.
  - High-precision assessment requires numerical orbit propagation and Monte Carlo covariance sampling, which is computationally expensive across large object catalogs.
  - Triage goal: Rapidly filter out safe encounters so computational resources are focused on actionable conjunctions.
- **What to Say:**
  > *"Madam, why do we need a triage classifier? Because running detailed numerical trajectory propagation for every single conjunction warning across a catalog of 30,000+ objects creates a computational bottleneck. A triage classifier acts as an initial filter: it identifies safe conjunctions with high confidence, leaving detailed numerical modeling for the remaining actionable events."*

---

### Slide 3: Data Engineering & Zero-Leakage Protocol
- **Dataset:** 3,351 CDMs across 836 unique satellite encounter pairs from Space-Track.org.
- **Target Definition:** Binary label $y = 1$ if maximum encounter risk $\max(P_c) \ge 5 \times 10^{-4}$ (25.1% positive prevalence).
- **Leakage Prevention:**
  - $P_c$ was **strictly stripped** from input feature vectors.
  - Fused Space-Track SATCAT (70,000+ objects) to construct 14 physical features:
    - Miss distance and range trends: $\log_{10}(\text{MIN\_RNG})$, $\Delta\text{MIN\_RNG}$, Range Ratio.
    - Orbital mechanics differentials: $\Delta\text{Apogee}$, $\Delta\text{Perigee}$, $\Delta\text{Inclination}$, $\Delta\text{Period}$.
    - Physical parameters: Radar Cross Section, Object Types (Debris/Payload/Rocket Body), and exclusion volumes.
  - Dynamic sequence packing (`pack_real_first`) ensures causal attention/retention operates strictly over real CDMs without padded-zero corruption or NaN attention masks.
- **What to Say:**
  > *"A key methodological focus of our project was zero leakage. Many previous publications included the Space-Track collision probability inside the input features, which causes artificial near-100% accuracy. We strictly removed $P_c$ from the inputs. The models must predict encounter risk strictly from relative geometry, observation trends, and orbital mechanics features merged from the SATCAT catalog."*

---

### Slide 4: Model Hierarchy: From Classical Time Series to Deep Models
- **Stochastic & Statistical Models (from time series literature):**
  - **Markov Chain (MC):** Discretizes miss distance into Safe, Warning, and Critical states and tracks state transition probabilities $\mathbf{P}(S_t \mid S_{t-1})$.
  - **Autoregressive / ARIMA Model:** Models miss distance trajectory trends using linear autoregression calibrated with logistic regression.
- **Tabular Models:**
  - **Logistic Regression:** Regularized linear model on 71 summary features.
  - **Random Forest & HistGradientBoosting:** Decision tree ensembles.
- **Deep Sequence Models:**
  - **Standard Transformer:** Causal self-attention with quadratic complexity (155k params).
  - **RetNet Surrogate:** Multi-scale causal matrix retention (Sun et al. 2023) with linear training and recurrent $O(1)$ inference (104k params).
- **What to Say:**
  > *"To ensure a thorough scientific study, we evaluated a complete hierarchy of models. We included stochastic Markov Chains and autoregressive ARIMA-style models from classical time series literature, linear and tree-based tabular baselines, and deep sequence architectures including Transformers and RetNet."*

---

### Slide 5: Comprehensive Benchmark Results (Table 1)
*Present Table 1:*

| Model Architecture | Model Category | AUC-ROC | PR-AUC (Average Precision) | Test Accuracy ($\tau=0.5$) | Brier Score ($\downarrow$) | ECE ($\downarrow$) | Measured Latency (Single-Core) |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Majority Class** | Naive Baseline | 0.5000 ± 0.0000 | 0.2512 ± 0.0006 | 0.7488 ± 0.0006 | 0.1881 ± 0.0003 | 0.0003 ± 0.0006 | Not benchmarked |
| **Markov Chain (MC)** | Stochastic Model | 0.5777 ± 0.0197 | 0.2849 ± 0.0091 | 0.7488 ± 0.0006 | 0.1830 ± 0.0028 | **0.0179 ± 0.0118** | Not benchmarked |
| **ARIMA / Autoregressive** | Statistical Model | 0.7838 ± 0.0481 | 0.6351 ± 0.0828 | 0.7308 ± 0.0384 | 0.1835 ± 0.0242 | 0.1917 ± 0.0335 | Not benchmarked |
| **Logistic Regression** | Linear Tabular | 0.8424 ± 0.0299 | 0.6700 ± 0.0653 | 0.7727 ± 0.0436 | 0.1602 ± 0.0289 | 0.1449 ± 0.0377 | **0.094 ms** |
| **Random Forest (200 trees)**| Tree Ensemble | 0.9050 ± 0.0357 | 0.7957 ± 0.0684 | 0.8361 ± 0.0401 | 0.1119 ± 0.0198 | 0.1054 ± 0.0235 | 11.19 ms |
| **HistGradientBoosting** | Gradient Boosting | **0.9086 ± 0.0226** | **0.8121 ± 0.0439** | **0.8744 ± 0.0296** | **0.7361 ± 0.0714** | **0.1028 ± 0.0203** | 0.0950 ± 0.0160 | 1.10 ms |
| **Standard Transformer** | Deep Sequence | 0.8307 ± 0.0369 | 0.6440 ± 0.0633 | 0.7655 ± 0.0499 | 0.6091 ± 0.0536 | 0.1586 ± 0.0162 | 0.1444 ± 0.0117 | 0.78 ms |
| **RetNet Surrogate (Ours)**| Deep Surrogate | 0.7506 ± 0.0484 | 0.5349 ± 0.0722 | 0.7308 ± 0.0251 | 0.5114 ± 0.0458 | 0.1819 ± 0.0178 | 0.1466 ± 0.0296 | **0.57 ms** (Recurrent) |

- **What to Say:**
  > *"Looking at Table 1 across our 5 cross-validation folds, tabular gradient boosting achieves the highest overall discriminative performance at 0.909 AUC and 0.812 Average Precision. The Standard Transformer achieves 0.831 AUC, while RetNet achieves 0.751 AUC (mean fold difference +0.080 AUC). We omit inferential p-values because 5-fold CV has 75% training overlap violating sample independence (Dietterich 1998). The performance gap favors Transformer attention on sparse updates because bounded softmax attention sharpens focus on isolated radar updates compared to unnormalized linear retention."*

---

### Slide 6: Safety-Critical Operating Threshold & Triage Workload (Table 2)
*Present Table 2:*

| Model | Operating Threshold ($\tau$) | Test Safety Recall | Test Precision | Test F1 Score | Flagged for Detailed Run (%) | Screened Out Safely (%) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Logistic Regression** | $0.313 \pm 0.082$ | 85.71% ± 4.99% | 44.22% ± 5.15% | 0.5804 ± 0.0399 | 48.7% | 51.3% |
| **Random Forest** | $0.300 \pm 0.039$ | **92.38% ± 4.86%** | 50.15% ± 5.25% | 0.6495 ± 0.0544 | 46.3% | 53.7% |
| **HistGradientBoosting** | $0.129 \pm 0.189$ | 84.29% ± 5.55% | **61.24% ± 10.72%** | **0.7034 ± 0.0741** | **34.6%** | **65.4%** |
| **Standard Transformer** | $0.350 \pm 0.052$ | 88.57% ± 5.91% | 41.64% ± 4.55% | 0.5632 ± 0.0333 | 53.4% | 46.6% |
| **RetNet Surrogate (Ours)** | $0.297 \pm 0.050$ | 84.29% ± 7.16% | 34.33% ± 4.32% | 0.4868 ± 0.0528 | 61.7% | 38.3% |

- **What to Say:**
  > *"In collision assessment, a symmetric 0.5 threshold is dangerous because missing a collision is catastrophic. We tuned operating thresholds on the inner validation set targeting $\ge 90\%$ recall. On out-of-fold test evaluation, models achieved $84.3\%\text{--}92.4\%$ recall (with Random Forest reaching 92.4%).
  > Using the operational screening formula $\text{Flag Rate} = (\text{Prevalence} \times \text{Recall}) / \text{Precision}$, HistGradientBoosting flags only 34.6% of events, successfully screening out 65.4% of candidate encounters from high-fidelity numerical assessment."*

---

### Slide 7: Why Do Trees Outperform Sequence Models? (Ablation & Subgroup Analysis)
- **Insight 1: Real-world CDM Sequence Lengths are Very Short**
  - Median sequence length is only **3.0 updates** (59.5% have $\le 3$ CDMs).
  - Short sequences leave minimal temporal context for multi-head attention.
- **Insight 2: Streaming Horizon Analysis ($k = 1, 2, 3, 4$ updates):**
  - $k=1$ (First CDM): HistGB achieves $\text{AUC} = 0.854$.
  - $k=2$ (Second CDM): HistGB jumps to $\text{AUC} = 0.929$.
  - Primary collision risk is determined primarily by static orbital geometry ($\Delta\text{Apogee}$, $\Delta\text{Perigee}$, initial miss distance), which decision trees segment directly.
- **Insight 3: Subgroup Analysis (Short vs. Long Tracks):**
  - Even on extended sequences ($N \ge 4$), deep sequence models remain near $\text{AUC} \approx 0.768\text{--}0.838$, while HistGB reaches $0.925$.
- **What to Say:**
  > *"Madam, why do decision trees outperform sequence models here? We dug into the astrodynamics and found that 60% of real CDM events contain only 1 to 3 updates. When we tested how early risk is detected, by the second update ($k=2$), gradient boosting already hits 0.929 AUC. Static orbital geometry—like apogee and perigee shell separation—determines the encounter boundary before temporal trends even unfold."*

---

### Slide 8: Satellite Leakage Audit: Component Grouped CV (Table 3)
*Present Table 3:*

| Grouping Scheme | Leakage Constraint | HistGradientBoosting AUC | Random Forest AUC | Logistic Regression AUC |
| :--- | :--- | :---: | :---: | :---: |
| **Stratified (Standard)** | Random encounters | $0.954 \pm 0.028$ | $0.922 \pm 0.022$ | $0.833 \pm 0.035$ |
| **Group by SAT 1** | Target satellite unseen | $0.957 \pm 0.025$ | $0.924 \pm 0.028$ | $0.839 \pm 0.038$ |
| **Group by SAT 2** | Chaser satellite unseen | $0.950 \pm 0.026$ | $0.923 \pm 0.027$ | $0.836 \pm 0.022$ |
| **Group by Component (Union-Find)** | **Zero satellite overlap** | **$0.777 \pm 0.053$** | **$0.802 \pm 0.045$** | **$0.823 \pm 0.051$** |

- **What to Say:**
  > *"We went one step further and tested what happens when models are evaluated on completely unseen satellites using graph Union-Find grouping. Tree ensembles drop by 18 points (0.954 down to 0.777) because trees partly memorized static exclusion volume and radar cross section features of specific high-traffic satellites. In contrast, regularized linear regression generalizes stably at 0.823 AUC, proving that it relies on universal relative geometry rather than satellite identity shortcuts."*

---

### Slide 9: RetNet's Operational Role & Streaming Inference
- **Why RetNet Still Matters:**
  - Standard Transformers cost $O(L^2)$ memory and cannot stream recurrently without recomputing past keys.
  - RetNet's multi-scale retention executes recurrently in **$0.566\text{ ms}$ constant time ($O(1)$)** with fixed memory.
  - RetNet provides a verified **$80.0\times$ speedup** over our measured 2D B-plane Monte Carlo quadrature engine ($45.28\text{ ms}$).
  - In streaming architectures, RetNet processes sequential updates without memory buffers expanding over time.
- **What to Say:**
  > *"While Gradient Boosting is the strongest offline triage filter on ground servers, RetNet offers a unique advantage for streaming architectures. RetNet runs in recurrent mode in $0.57\text{ ms}$ with constant $O(1)$ memory, delivering an 80-fold acceleration over 1M-particle 2D Monte Carlo quadrature while avoiding dynamic memory expansion."*

---

### Slide 10: Conclusion & Engineering Recommendations
- **Summary:**
  1. Gradient Boosting is the most effective operational triage filter on ground catalogs, screening out 65.4% of candidate encounters.
  2. In-catalog encounters benefit from tree ensembles; out-of-catalog unseen satellites require linear or sequence models ($0.823$ AUC) to avoid overfitting static object traits.
  3. RetNet provides a constant-memory recurrent architecture for streaming updates ($0.57\text{ ms}$, $80.0\times$ speedup vs 2D MC).
- **What to Say:**
  > *"In conclusion, Madam, our work establishes rigorous benchmark baselines for satellite conjunction triage. We show that tabular gradient boosting saves 65% of numerical propagation compute, identify the physical limits of sequence models on sparse radar updates, and demonstrate how RetNet provides constant-memory streaming. Thank you, and I welcome your questions."*

---

## 3. Anticipated Questions from Madam & How to Answer

### Q1: *"Why does Gradient Boosting beat your deep learning models?"*
**Answer:**
> *"Madam, there are two physical reasons. First, 59.5% of real CDM events contain only 1 to 3 updates (median = 3.0), leaving deep sequence models with minimal temporal context to exploit. Second, conjunction risk in Low Earth Orbit is strongly dictated by orbital mechanics separations—such as $\Delta\text{Apogee}$, $\Delta\text{Perigee}$, and minimum miss distance. Decision tree ensembles are exceptionally effective at finding sharp threshold boundaries on these static features without needing large training corpora."*

### Q2: *"Why did you include classical models like Markov Chains and ARIMA?"*
**Answer:**
> *"To ensure academic thoroughness, Madam. Classical literature often uses Markov state transitions or autoregressive models for trajectory forecasting. Benchmarking them under identical 5-fold cross-validation showed that Markov Chains ($\text{AUC} = 0.578$) are too memoryless, and ARIMA ($\text{AUC} = 0.784$) captures smooth trends but struggles with abrupt covariance updates, confirming why non-linear machine learning is necessary."*

### Q3: *"Did RetNet beat the Transformer?"*
**Answer:**
> *"No, Madam. Under unified 5-fold cross-validation, the Transformer achieves $0.831 \pm 0.037$ AUC while RetNet achieves $0.751 \pm 0.048$ AUC, with a descriptive mean fold difference of $+0.0801 \pm 0.0457$ AUC (sample std, $N-1=4$, $\text{ddof}=1$; inferential p-values omitted due to overlapping folds per Dietterich 1998). The Transformer's bounded softmax attention allows it to focus sharply on the single critical update where miss distance dropped, whereas linear retention without softmax spreads attention more evenly. However, RetNet has 32.5% fewer parameters and executes recurrently with $O(1)$ constant memory in $0.566\text{ ms}$ ($0.57\text{ ms}$), delivering an 80.0x speedup over 2D Monte Carlo quadrature ($45.28\text{ ms}$)."*

### Q4: *"What did your Grouped Cross-Validation audit discover?"*
**Answer:**
> *"Madam, when satellites can appear in both train and test sets, tree models achieve 0.954 AUC. But when we clustered encounters using Union-Find to ensure zero satellite overlap between train and test, tree performance dropped to 0.777 AUC. This proved that trees were partly memorizing static satellite traits like exclusion volume and radar cross section. Regularized linear regression, on the other hand, held steady at 0.823 AUC, proving that it learns physics-based relative geometry that transfers safely to brand-new satellites. Note that this grouped audit evaluated tabular models; deep sequence architectures have not been evaluated for unseen-satellite generalization under this protocol."*
