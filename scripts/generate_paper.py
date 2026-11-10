"""
Generate research paper as DOCX using python-docx.
Satellite Conjunction Assessment via Physics-Enriched Sequence and Tabular Models: A Comparative Study of Triage Screening
Matches RESEARCH_PAPER.md and research_paper.tex exactly.
"""
import sys
import os

from docx import Document
from docx.shared import Inches, Pt, RGBColor
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.enum.table import WD_TABLE_ALIGNMENT
from docx.oxml import OxmlElement, parse_xml
from docx.oxml.ns import nsdecls, qn

def set_cell_background(cell, fill_hex):
    shading_elm = parse_xml(f'<w:shd {nsdecls("w")} w:fill="{fill_hex}"/>')
    cell._tc.get_or_add_tcPr().append(shading_elm)

def style_table(table, col_widths=None):
    table.alignment = WD_TABLE_ALIGNMENT.CENTER
    for i, row in enumerate(table.rows):
        # Prevent row split across pages
        trPr = row._tr.get_or_add_trPr()
        trPr.append(parse_xml(f'<w:cantSplit {nsdecls("w")}/>'))
        if i == 0:
            # Header repeat
            trPr.append(parse_xml(f'<w:tblHeader {nsdecls("w")}/>'))
            for cell in row.cells:
                set_cell_background(cell, "EAEAEA")
                for p in cell.paragraphs:
                    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
                    for run in p.runs:
                        run.bold = True
                        run.font.name = "Times New Roman"
                        run.font.size = Pt(9.5)
        else:
            for cell in row.cells:
                for p in cell.paragraphs:
                    for run in p.runs:
                        run.font.name = "Times New Roman"
                        run.font.size = Pt(9.5)
        if i % 2 == 1 and i > 0:
            for cell in row.cells:
                set_cell_background(cell, "F9F9F9")

    if col_widths:
        for row in table.rows:
            for idx, width in enumerate(col_widths):
                if idx < len(row.cells):
                    row.cells[idx].width = width

def build_docx(output_path):
    doc = Document()

    # Page Margins: 1 inch all around
    sections = doc.sections
    for s in sections:
        s.top_margin = Inches(1.0)
        s.bottom_margin = Inches(1.0)
        s.left_margin = Inches(1.0)
        s.right_margin = Inches(1.0)

    # Styles
    normal_style = doc.styles['Normal']
    font = normal_style.font
    font.name = 'Times New Roman'
    font.size = Pt(11)
    normal_style.paragraph_format.line_spacing = 1.15
    normal_style.paragraph_format.space_after = Pt(4)

    for i in range(1, 4):
        hs = doc.styles[f'Heading {i}']
        hs.font.name = 'Times New Roman'
        hs.font.color.rgb = RGBColor(0, 0, 0)
        hs.paragraph_format.space_before = Pt(8)
        hs.paragraph_format.space_after = Pt(3)

    doc.styles['Heading 1'].font.size = Pt(14)
    doc.styles['Heading 1'].font.bold = True
    doc.styles['Heading 2'].font.size = Pt(12)
    doc.styles['Heading 2'].font.bold = True
    doc.styles['Heading 3'].font.size = Pt(11)
    doc.styles['Heading 3'].font.italic = True

    # Title
    title = doc.add_paragraph()
    title.alignment = WD_ALIGN_PARAGRAPH.CENTER
    run_t = title.add_run("Satellite Conjunction Assessment via Physics-Enriched Sequence and Tabular Models: A Comparative Study of Triage Screening")
    run_t.bold = True
    run_t.font.size = Pt(16)
    title.paragraph_format.space_after = Pt(6)

    # Authors
    authors = doc.add_paragraph()
    authors.alignment = WD_ALIGN_PARAGRAPH.CENTER
    run_a = authors.add_run("Sai Ganesh\nDepartment of Computer Science & Engineering\n")
    run_a.font.size = Pt(11)
    run_meta = authors.add_run("Dataset: 3,351 Conjunction Data Messages (CDMs) across 836 unique orbital encounters fused with 70,000+ Space-Track SATCAT records\nRepository Target: E:\\DL_fixed (feature/operational-monte-carlo-surrogate)")
    run_meta.font.size = Pt(9.5)
    run_meta.italic = True
    authors.paragraph_format.space_after = Pt(12)

    # Abstract
    doc.add_heading("Abstract", level=1)
    doc.add_paragraph(
        "As the catalog of tracked space objects expands beyond 36,000 bodies, satellite operations face an escalating volume of conjunction warnings. "
        "Operational workflows rely on high-fidelity numerical orbit propagation and Monte Carlo covariance sampling to compute collision probabilities (Pc), "
        "which is computationally intensive and difficult to scale across all pairwise combinations in crowded orbital shells. In this work, we evaluate "
        "machine learning triage classifiers to identify high-risk conjunction events (max Pc >= 5e-4) directly from sequences of Conjunction Data Messages (CDMs) "
        "enriched with orbital physics from the Space-Track Satellite Catalog (SATCAT)."
    )
    doc.add_paragraph(
        "We benchmark a diverse hierarchy of models under a unified 5-fold cross-validation protocol on 836 conjunction events: naive baselines (Majority Class), "
        "classical stochastic and statistical time series models (Markov Chain, Autoregressive/ARIMA), linear regression, tree ensembles (Random Forest, HistGradientBoosting), "
        "and deep sequence architectures (Standard Transformer, Retentive Network / RetNet). Tabular gradient boosting achieves the highest discriminative performance "
        "(AUC-ROC = 0.9086 +- 0.0226, PR-AUC = 0.8121 +- 0.0439). Among sequence models, the causal Standard Transformer achieves AUC-ROC = 0.8307 +- 0.0369, while "
        "RetNet with Sun et al. (2023) multi-scale matrix retention achieves AUC-ROC = 0.7506 +- 0.0484. Descriptive fold differences across the 5 outer test folds "
        "indicate an empirical performance difference favoring softmax attention on sparse sequences (+0.0801 +- 0.0457 mean AUC over RetNet, sample standard deviation "
        "with N - 1 = 4 degrees of freedom, ddof = 1; population sigma = 0.0409; inferential p-values omitted due to training fold overlap per Dietterich, 1998), "
        "with RetNet maintaining constant O(1) recurrent memory and sub-millisecond single-step latency (0.57 ms)."
    )
    doc.add_paragraph(
        "Analysis of sequence length shows that 59.5% of encounters contain only 1 to 3 CDM updates (median length = 3.0), leaving sequence models with limited "
        "temporal signal to model. Subgroup experiments demonstrate that while gradient boosting maintains strong performance across both short (N <= 3, AUC = 0.8948) "
        "and longer histories (N >= 4, AUC = 0.9254), deep sequence models remain near AUC ~ 0.75-0.83. Furthermore, streaming horizon analysis reveals that the primary "
        "geometric risk signals are largely resolved by the second observation (k = 2, AUC = 0.9293). At validation-tuned operating thresholds targeting >= 90% recall, "
        "out-of-fold test recall reached 84.3%-92.4% across models. In a triage screening configuration, gradient boosting safely screens out 65.4% of non-critical "
        "events, while RetNet screens out 38.3%. Furthermore, under strict graph-component cross-validation with zero satellite overlap between train and test, tree "
        "models experience an 18-point drop while regularized linear models maintain solid generalization (AUC = 0.823 +- 0.051), demonstrating that tree models "
        "partially rely on static spacecraft identity features. These findings delineate the boundary conditions where tabular and sequence architectures operate effectively in space situational awareness."
    )
    doc.add_paragraph(
        "Finally, we implement a vectorized 1,000,000-sample B-plane Monte Carlo simulator and evaluate surrogate speedups. We document that public CDMs lack 2x2 B-plane "
        "covariance matrices, requiring an isotropic scale calibration against catalog Pc under which 71% of high-risk cases require radius inflation due to real-world "
        "along-track covariance elongation; the resulting Monte Carlo simulations serve as numerical sampling verification of analytical quadrature, rather than independent "
        "orbit determination. Standardized single-core CPU profiling reveals that RetNet recurrent inference (0.566 ms median, 0.57 ms) and Transformer inference (0.776 ms median, 0.78 ms) "
        "achieve verified 80.0x and 58.3x speedups over 1,000,000-sample 2D B-plane Monte Carlo quadrature (45.283 ms median, 45.247 ms mean), with HistGradientBoosting (1.095 ms) "
        "achieving a 41.3x speedup. An illustrative Collision Avoidance Maneuver (CAM) module is provided for demonstration, with explicit flight safety disclaimers."
    )

    # 1. Introduction
    doc.add_heading("1. Introduction", level=1)
    doc.add_paragraph(
        "With more than 36,000 trackable objects exceeding 10 cm in Earth orbit and mega-constellations deploying thousands of active satellites, conjunction assessment (CA) "
        "has become a primary bottleneck for space situational awareness (SSA). Satellite operators receive daily streams of Conjunction Data Messages (CDMs) produced by the "
        "18th Space Defense Squadron via Space-Track.org. Each CDM encapsulates state vectors, relative miss distance, positional covariance matrices, and estimated collision "
        "probability (Pc) for two orbiting objects at the Time of Closest Approach (TCA)."
    )
    doc.add_paragraph("Operational spaceflight safety typically manages risk through a multi-tiered pipeline:")
    doc.add_paragraph("1. Screening Volumes: Coarse geometric filters identify object pairs passing within predefined thresholds (e.g., 5 km miss distance in LEO).")
    doc.add_paragraph("2. Analytical Approximations: Two-dimensional Foster or Akella formulas compute short-term Pc under linear relative motion and constant Gaussian covariance.")
    doc.add_paragraph("3. High-Fidelity Numerical Propagation: When covariance geometry is non-linear or atmospheric drag uncertainty is high, Monte Carlo simulations propagate thousands to millions of sampled trajectories to compute numerical collision probabilities.")
    doc.add_paragraph(
        "Because high-fidelity propagation across hundreds of thousands of candidate encounters is computationally demanding, automated machine learning triage classifiers "
        "have been proposed to prioritize high-risk events. However, existing literature presents several methodological challenges:"
    )
    doc.add_paragraph("- Direct Target Leakage: Multiple studies include instantaneous Pc from Space-Track within input features while predicting peak encounter Pc, resulting in trivial near-perfect accuracy that fails in operational deployment.")
    doc.add_paragraph("- Absence of Classical & Tabular Baselines: Deep sequence models are often published without comparison against classical time-series models (such as Markov Chains and ARIMA) or gradient-boosted decision trees.")
    doc.add_paragraph("- Uncalibrated Probabilities and Symmetric Thresholds: Standard models often report metrics at an uncalibrated default threshold (tau = 0.5). In satellite safety, false negatives risk spacecraft collision, requiring decision thresholds tuned specifically for high recall with calibrated posterior probabilities.")
    doc.add_paragraph("- Satellite Overlap Leakage: Evaluating conjunction events with standard randomized splits allows the same satellite (and its static RCS or exclusion volume) to appear in both train and test splits, artificially inflating decision tree scores.")

    doc.add_heading("Contributions", level=2)
    doc.add_paragraph("To address these issues systematically, this paper provides:")
    doc.add_paragraph("1. A Leak-Free Astrodynamics Benchmark: We extract 3,351 CDMs across 836 unique encounters, strictly excluding Pc from features and incorporating 14 orbital mechanics features from the Space-Track Satellite Catalog (SATCAT).")
    doc.add_paragraph("2. Comprehensive Model Comparison: We evaluate eight architectures spanning naive baselines, Markov Chain state transitions, Autoregressive/ARIMA time-series models, Logistic Regression, Random Forest, HistGradientBoosting, Standard Transformer, and RetNet.")
    doc.add_paragraph("3. Unified 5-Fold Cross-Validation: All models are evaluated under the same 5-fold cross-validation with nested validation for temperature scaling calibration and high-recall threshold tuning.")
    doc.add_paragraph("4. Padding-Insensitive Sequence Modeling: Sequences are dynamically packed (pack_real_first) so that causal attention operates strictly over real physical observations without zero-token padding corruption or mask-induced NaN instabilities. Causal upper-triangular masks strictly prevent future lookahead.")
    doc.add_paragraph("5. Multi-Scheme Grouped CV Audit: We evaluate models under strict graph-component grouping (Union-Find) ensuring zero satellite overlap across splits, revealing that tree models drop from 0.954 to 0.777 while regularized linear models generalize robustly (0.823).")
    doc.add_paragraph("6. Streaming Horizon & Subgroup Experiments: We investigate how model performance evolves across observation steps (k = 1, 2, 3, 4) and analyze performance on short (N <= 3) versus longer (N >= 4) observation histories.")
    doc.add_paragraph("7. Empirically Measured Monte Carlo Speedup: We measure standardized single-core CPU execution times for all models, documenting that RetNet (0.57 ms) and Transformer (0.78 ms) achieve verified 80.0x and 58.3x speedups over 2D B-plane Monte Carlo quadrature (45.28 ms for 10^6 particles), with HistGradientBoosting (1.10 ms) achieving a 41.3x speedup.")

    # 2. Problem Formulation & Feature Space
    doc.add_heading("2. Problem Formulation & Feature Space", level=1)
    doc.add_heading("2.1 Conjunction Data Messages (CDMs)", level=2)
    doc.add_paragraph(
        "A CDM sequence represents an encounter between a primary satellite and a secondary space object (payload, rocket body, or debris) over a multi-day monitoring window preceding TCA. "
        "Let an encounter i be represented by S_i = {x_1, x_2, ..., x_{T_i}}, where x_t in R^D, T_i is the number of updates received, and D = 14 is the feature dimension."
    )
    doc.add_heading("2.2 Prediction Target", level=2)
    doc.add_paragraph(
        "Because Space-Track publishes CDMs only when initial screening exceeds Pc >= 1e-4, the operational task is formulated as identifying whether the encounter will escalate "
        "to high collision risk requiring avoidance maneuver planning: y_i = 1 if max_{t in [1, T_i]} Pc(t) >= 5e-4, else 0. "
        "Across the 836 encounters, 210 events (25.12%) are labeled positive, reflecting a 1:3 class imbalance. The instantaneous value Pc(t) is strictly excluded from x_t."
    )
    doc.add_heading("2.3 Input Features (Physics-Enriched via SATCAT)", level=2)
    doc.add_paragraph("Features at each observation step include:")
    doc.add_paragraph("- log_min_rng: log10(MIN_RNG) (miss distance at TCA in meters).")
    doc.add_paragraph("- delta_min_rng: Consecutive change in miss distance (Delta RNG_t / 100 m).")
    doc.add_paragraph("- range_ratio: Normalized range relative to initial observation (RNG_t / RNG_1).")
    doc.add_paragraph("- norm_step_pos: Normalized observation step (t / T_i).")
    doc.add_paragraph("- delta_apogee: log10|Apogee_1 - Apogee_2| (orbital shell altitude difference in km).")
    doc.add_paragraph("- delta_perigee: log10|Perigee_1 - Perigee_2| (km).")
    doc.add_paragraph("- delta_inclination: |Inc_1 - Inc_2| (plane crossing difference in degrees).")
    doc.add_paragraph("- delta_period: |Period_1 - Period_2| (period difference in minutes).")
    doc.add_paragraph("- sat1_rcs, sat2_rcs: Radar cross section categories (0: Small, 1: Medium, 2: Large).")
    doc.add_paragraph("- sat1_type, sat2_type: Object classification (0: Debris, 1: Payload, 2: Rocket Body).")
    doc.add_paragraph("- sat1_excl_vol, sat2_excl_vol: Spacecraft hard-body exclusion radii (meters).")

    # 3. Evaluated Model Architectures
    doc.add_heading("3. Evaluated Model Architectures", level=1)
    doc.add_heading("3.1 Naive & Classical Time-Series Models", level=2)
    doc.add_paragraph("- Majority Class: Predicts constant empirical training prevalence (P(Y=1) ~ 0.251).")
    doc.add_paragraph("- Markov Chain (MC): Discretizes miss distance into three states: Safe (r >= 3 km), Warning (500 m <= r < 3 km), and Critical (r < 500 m). Computes an empirical 3x3 transition probability matrix P(S_t | S_{t-1}) and assigns risk from historical state labels.")
    doc.add_paragraph("- Autoregressive / ARIMA Model: Fits an autoregressive model on the miss distance sequence (RNG_t), capturing the final miss distance, minimum distance, trend slope, and observation length, calibrated via logistic regression.")
    doc.add_paragraph("- Logistic Regression: Regularized linear model fit on 71 summary features (last step, mean, min, max, trend, and length).")

    doc.add_heading("3.2 Tree Ensembles", level=2)
    doc.add_paragraph("- Random Forest: 200 balanced decision trees trained on tabular summary features.")
    doc.add_paragraph("- HistGradientBoosting: Histogram-based gradient boosting trees with balanced sample weighting.")

    doc.add_heading("3.3 Deep Sequence Models", level=2)
    doc.add_paragraph(
        "- Standard Transformer: 3-layer Transformer encoder (TransformerBaseline) with strict causal multi-head self-attention (4 heads, d_model = 64, d_ff = 256, dropout = 0.1, 155,201 parameters). "
        "Sequences are dynamically structured via pack_real_first so that valid observations occupy positions 0 <= t < k and padding sits strictly at the end. Attention weights are constrained by an "
        "upper-triangular causal mask via torch.nn.Transformer.generate_square_subsequent_mask(T) with is_causal=True, ensuring A_{i, j} = 0 for all j > i. Over the observed historical prefix window "
        "S_k = {x_1, ..., x_k}, historical CDMs are past physical telemetry tracks received chronologically leading up to TCA. Applying a causal encoder over this observed prefix evaluates temporal "
        "trajectory convergence up to the decision epoch k without leaking future unobserved CDMs (t > k). The pooled representation at the last observed step k maps to the binary collision risk triage classification at TCA."
    )
    doc.add_paragraph(
        "- RetNet Surrogate: 3-layer Retentive Network using Sun et al. (2023) multi-scale matrix retention with decay parameters gamma in {0.90, 0.95, 0.98, 0.99}, SwiGLU feed-forward networks, and d_model = 64 "
        "(104,801 parameters). RetNet supports both parallel batch training and recurrent single-step inference (S_t = gamma S_{t-1} + K_t^T V_t)."
    )

    # 4. Experimental Results
    doc.add_heading("4. Experimental Results", level=1)
    doc.add_heading("4.1 Comprehensive 5-Fold Cross-Validation Performance", level=2)
    doc.add_paragraph("Table 1 summarizes held-out test fold performance across identical 5 folds on 836 conjunction events.")

    # Table 1
    t1 = doc.add_table(rows=10, cols=9)
    headers_t1 = ["Model Category", "Model Architecture", "AUC-ROC", "PR-AUC", "Accuracy", "F1-Score", "Brier", "ECE", "Measured Latency"]
    for j, h in enumerate(headers_t1):
        t1.rows[0].cells[j].text = h
    rows_data_t1 = [
        ["Classical Physics", "Monte Carlo (10^6 particles)", "Reference", "Reference", "N/A", "N/A", "N/A", "N/A", "45.28 ms (Measured)"],
        ["Naive Baseline", "Majority Class", "0.5000 +- 0.0000", "0.2512 +- 0.0006", "0.7488 +- 0.0006", "0.0000 +- 0.0000", "0.1881", "0.0003", "Not benchmarked"],
        ["Stochastic Model", "Markov Chain (MC)", "0.5777 +- 0.0197", "0.2849 +- 0.0091", "0.7488 +- 0.0006", "0.0000 +- 0.0000", "0.1830", "0.0179", "Not benchmarked"],
        ["Statistical Model", "Autoregressive / ARIMA", "0.7838 +- 0.0481", "0.6351 +- 0.0828", "0.7308 +- 0.0384", "0.5750 +- 0.0544", "0.1835", "0.1917", "Not benchmarked"],
        ["Linear Tabular", "Logistic Regression", "0.8424 +- 0.0299", "0.6700 +- 0.0653", "0.7727 +- 0.0436", "0.6107 +- 0.0396", "0.1602", "0.1449", "0.094 ms"],
        ["Tree Ensemble", "Random Forest (200 trees)", "0.9050 +- 0.0357", "0.7957 +- 0.0684", "0.8361 +- 0.0401", "0.6815 +- 0.0617", "0.1119", "0.1054", "11.19 ms"],
        ["Gradient Boosting", "HistGradientBoosting", "0.9086 +- 0.0226", "0.8121 +- 0.0439", "0.8744 +- 0.0296", "0.7361 +- 0.0714", "0.1028", "0.0950", "1.10 ms"],
        ["Deep Sequence", "Standard Transformer", "0.8307 +- 0.0369", "0.6440 +- 0.0633", "0.7655 +- 0.0499", "0.6091 +- 0.0536", "0.1586", "0.1444", "0.78 ms"],
        ["Deep Surrogate", "RetNet (Proposed)", "0.7506 +- 0.0484", "0.5349 +- 0.0722", "0.7308 +- 0.0251", "0.5114 +- 0.0458", "0.1819", "0.1466", "0.57 ms (Recurrent)"]
    ]
    for i, rdata in enumerate(rows_data_t1):
        for j, val in enumerate(rdata):
            t1.rows[i+1].cells[j].text = val
    style_table(t1)
    p_cap1 = doc.add_paragraph("Table 1: Comprehensive 5-fold cross-validation performance comparison across 836 encounters (mean +- std across 5 test folds).")
    p_cap1.italic = True
    p_cap1.paragraph_format.space_after = Pt(8)

    # Table 2: Triage Performance
    doc.add_heading("4.2 Safety-Critical Triage Performance (Validation Target >= 90% Recall)", level=2)
    doc.add_paragraph(
        "In satellite collision avoidance, missing a true collision risk is catastrophic. The operating decision threshold tau was tuned on inner validation "
        "folds to target >= 90% recall, and evaluated strictly out-of-fold on the test splits. Table 2 summarizes triage performance."
    )
    t2 = doc.add_table(rows=6, cols=7)
    headers_t2 = ["Model", "Threshold (tau)", "Test Recall", "Test Precision", "Test F1", "Flagged (%)", "Screened Out (%)"]
    for j, h in enumerate(headers_t2):
        t2.rows[0].cells[j].text = h
    rows_data_t2 = [
        ["Logistic Regression", "0.313 +- 0.082", "85.71% +- 4.99%", "44.22% +- 5.15%", "0.5804 +- 0.0399", "48.7%", "51.3%"],
        ["Random Forest", "0.300 +- 0.039", "92.38% +- 4.86%", "50.15% +- 5.25%", "0.6495 +- 0.0544", "46.3%", "53.7%"],
        ["HistGradientBoosting", "0.129 +- 0.189", "84.29% +- 5.55%", "61.24% +- 10.72%", "0.7034 +- 0.0741", "34.6%", "65.4%"],
        ["Standard Transformer", "0.350 +- 0.052", "88.57% +- 5.91%", "41.64% +- 4.55%", "0.5632 +- 0.0333", "53.4%", "46.6%"],
        ["RetNet Surrogate (Ours)", "0.297 +- 0.050", "84.29% +- 7.16%", "34.33% +- 4.32%", "0.4868 +- 0.0528", "61.7%", "38.3%"]
    ]
    for i, rdata in enumerate(rows_data_t2):
        for j, val in enumerate(rdata):
            t2.rows[i+1].cells[j].text = val
    style_table(t2)
    p_cap2 = doc.add_paragraph("Table 2: Safety-critical triage performance at validation-tuned operating thresholds targeting >= 90% recall.")
    p_cap2.italic = True
    p_cap2.paragraph_format.space_after = Pt(8)

    # Table 3: Satellite Overlap Audit
    doc.add_heading("4.3 Satellite Identity Leakage Audit (Grouped Cross-Validation)", level=2)
    doc.add_paragraph(
        "To evaluate sensitivity to satellite identity leakage, models were benchmarked across four split schemes across 3 random seeds (seeds 0, 1, 2; 5 folds each = 15 test folds total). "
        "Evaluated on 836 conjunction events using 58 summary features. Complete fold-level artifacts are saved in results/grouped_cv_leakage_audit.json."
    )
    t3 = doc.add_table(rows=5, cols=5)
    headers_t3 = ["Grouping Scheme", "Leakage Constraint", "HistGB AUC", "Random Forest AUC", "Logistic Regression AUC"]
    for j, h in enumerate(headers_t3):
        t3.rows[0].cells[j].text = h
    rows_data_t3 = [
        ["Stratified (Standard)", "None (random encounters)", "0.954 +- 0.028", "0.922 +- 0.022", "0.833 +- 0.035"],
        ["Group by SAT 1", "Target satellite unseen in test", "0.957 +- 0.025", "0.924 +- 0.028", "0.839 +- 0.038"],
        ["Group by SAT 2", "Chaser satellite unseen in test", "0.950 +- 0.026", "0.923 +- 0.027", "0.836 +- 0.022"],
        ["Group by Component (Union-Find)", "Zero satellite overlap across splits", "0.777 +- 0.053", "0.802 +- 0.045", "0.823 +- 0.051"]
    ]
    for i, rdata in enumerate(rows_data_t3):
        for j, val in enumerate(rdata):
            t3.rows[i+1].cells[j].text = val
    style_table(t3)
    p_cap3 = doc.add_paragraph("Table 3: Satellite overlap audit across 3 random seeds (15 folds). Finding: Tree ensembles drop ~18 points under component grouping, while linear models generalize stably.")
    p_cap3.italic = True
    p_scope = doc.add_paragraph()
    r_scope = p_scope.add_run(
        "Scope of Grouped Cross-Validation Audit: This connected-component grouped cross-validation audit evaluated tabular models (HistGradientBoosting, Random Forest, Logistic Regression) only. "
        "Deep sequence architectures (Standard Transformer and RetNet) have not been evaluated for unseen-satellite generalization under the full multi-seed connected-component grouping protocol. "
        "The grouped audit must not be construed as validating unseen-satellite generalization for deep architectures."
    )
    r_scope.font.size = Pt(9.5)
    r_scope.italic = True
    p_scope.paragraph_format.space_after = Pt(8)

    # 5. Sequence Dynamics & Subgroup Analysis
    doc.add_heading("5. Sequence Dynamics & Subgroup Analysis", level=1)
    doc.add_heading("5.1 Streaming Horizon Analysis: AUC vs. Number of Observed CDMs", level=2)
    doc.add_paragraph("To test how early a triage decision can be made as CDMs arrive sequentially:")
    t4 = doc.add_table(rows=5, cols=4)
    headers_t4 = ["Observation Horizon", "Number of CDMs Used (k)", "HistGradientBoosting AUC", "Logistic Regression AUC"]
    for j, h in enumerate(headers_t4):
        t4.rows[0].cells[j].text = h
    rows_data_t4 = [
        ["Initial Detection", "k = 1", "0.8543 +- 0.0356", "0.6717 +- 0.0306"],
        ["2nd Radar Update", "k = 2", "0.9293 +- 0.0243", "0.7899 +- 0.0338"],
        ["3rd Radar Update", "k = 3", "0.9243 +- 0.0259", "0.8035 +- 0.0410"],
        ["4th Radar Update", "k = 4", "0.9324 +- 0.0276", "0.8228 +- 0.0375"]
    ]
    for i, rdata in enumerate(rows_data_t4):
        for j, val in enumerate(rdata):
            t4.rows[i+1].cells[j].text = val
    style_table(t4)
    p_cap4 = doc.add_paragraph("Table 4: Progressive classification performance as tracking updates accumulate over time.")
    p_cap4.italic = True
    p_cap4.paragraph_format.space_after = Pt(8)

    doc.add_heading("5.2 Subgroup Analysis by Sequence Length", level=2)
    doc.add_paragraph("We partitioned the 836 encounters into short (N <= 3 CDMs) and extended (N >= 4 CDMs) histories:")
    t5 = doc.add_table(rows=4, cols=6)
    headers_t5 = ["Subgroup", "Number of Events", "High-Risk Prevalence", "HistGB AUC", "RetNet AUC", "Transformer AUC"]
    for j, h in enumerate(headers_t5):
        t5.rows[0].cells[j].text = h
    rows_data_t5 = [
        ["Short Histories (N <= 3)", "506 (60.5%)", "13.4% (68 pos)", "0.8948 +- 0.0528", "0.742 +- 0.051", "0.825 +- 0.038"],
        ["Long Histories (N >= 4)", "330 (39.5%)", "43.0% (142 pos)", "0.9254 +- 0.0214", "0.768 +- 0.054", "0.838 +- 0.042"],
        ["Extended Histories (N >= 6)", "201 (24.0%)", "56.7% (114 pos)", "0.9101 +- 0.0296", "-", "-"]
    ]
    for i, rdata in enumerate(rows_data_t5):
        for j, val in enumerate(rdata):
            t5.rows[i+1].cells[j].text = val
    style_table(t5)
    p_cap5 = doc.add_paragraph("Table 5: Subgroup evaluation by observation history length.")
    p_cap5.italic = True
    p_cap5.paragraph_format.space_after = Pt(8)

    # 6. Discussion & Limitations
    doc.add_heading("6. Discussion & Limitations", level=1)
    doc.add_heading("6.1 Model Comparison & Descriptive Fold Differences", level=2)
    doc.add_paragraph(
        "Across the 5 outer test folds (reported as mean +- sample standard deviation with N - 1 = 4 degrees of freedom, ddof = 1):\n"
        "- RetNet vs. Transformer: The Transformer demonstrates higher mean discrimination (0.8307 +- 0.0369 vs. 0.7506 +- 0.0484, descriptive mean fold difference "
        "+0.0801 +- 0.0457 AUC, with population sigma = 0.0409). As established by Dietterich (1998), standard paired t-tests on k-fold cross-validation violate sample "
        "independence due to overlapping training sets (each pair of training sets shares 75% of events), which inflates Type I error rates. Consequently, inferential "
        "p-values are intentionally omitted, and this difference is reported as an empirical descriptive performance gap under the evaluated cross-validation splits.\n"
        "- HistGB vs. RetNet: Tabular gradient-boosted trees achieve a descriptive mean fold difference of +0.1580 +- 0.0436 AUC over RetNet (with population sigma = 0.0390), "
        "confirming that gradient-boosted trees on summary features strongly outperform linear sequence retention on this small-sample regime.\n"
        "- Why Softmax Attention Outperforms Linear Matrix Retention on Sparse CDMs: In Sun et al. (2023), retention uses linear decay-weighted dot products without softmax "
        "normalization across timesteps: o_n = sum_{i <= n} gamma^{n-i} (q_n . k_i) v_i. In dense, long natural language contexts (L > 1024), decaying linear attention excels. "
        "However, in satellite conjunctions, sequences are extremely short (median 3 CDMs, maximum 14), with sporadic updates separated by 12-24 hours. Under these conditions, "
        "the Transformer's bounded softmax normalization softmax(QK^T / sqrt(d)) provides critical sharpening, allowing the model to focus sharply on the single critical update where miss distance dropped precipitously.\n"
        "- RetNet's Trade-Off: While RetNet has lower discrimination on sparse short tracks (0.7506 vs 0.8307), it retains 32.5% fewer parameters (104,801 vs 155,201) and constant O(1) memory, "
        "executing recurrently in 0.566 ms (0.57 ms)."
    )

    doc.add_heading("6.2 Latency and Computational Complexity", level=2)
    doc.add_paragraph(
        "Measured median single-encounter CPU latencies at batch size 1 (10 warmup runs, 100 repetitions on single-core CPU) are:\n"
        "- Logistic Regression: 0.094 ms (0.09385 ms; mean 0.094 ms)\n"
        "- RetNet Recurrent (O(1) step): 0.566 ms (0.57 ms; mean 0.684 ms; Parallel L=14: 0.847 ms)\n"
        "- Standard Transformer: 0.776 ms (0.78 ms; mean 1.065 ms)\n"
        "- HistGradientBoosting: 1.095 ms (1.10 ms; mean 1.319 ms)\n"
        "- Random Forest (200 trees): 11.194 ms (11.19 ms; mean 10.992 ms)\n"
        "- 2D B-Plane Monte Carlo (10^6 particles): 45.283 ms median (45.28 ms; 45.247 ms mean across 100 encounters)"
    )
    doc.add_paragraph(
        "Speedup Analysis:\n"
        "Relative to the project's own 2D B-plane Monte Carlo quadrature engine (45.28 ms for 10^6 particles), the measured speedups are:\n"
        "- RetNet Recurrent: 80.0x faster\n"
        "- Standard Transformer: 58.3x faster\n"
        "- HistGradientBoosting: 41.3x faster\n"
        "- Logistic Regression: 482.5x faster\n"
        "- Random Forest: 4.0x faster\n\n"
        "All measurements were performed under a strictly controlled single-core benchmark protocol. Disk I/O, SATCAT cross-referencing, and batch data loading were strictly excluded from the timing loops."
    )

    doc.add_heading("6.3 Limitations & Boundary Conditions", level=2)
    doc.add_paragraph(
        "1. Absence of Full B-Plane Covariance in Public CDMs: Space-Track public CDMs omit the full 2x2 B-plane covariance matrix. To establish a standardized numerical baseline, "
        "an isotropic scale sigma was calibrated against catalog Pc. In 71% of high-risk encounters, achieving the catalog Pc within an isotropic Gaussian required inflating the hard-body radius "
        "beyond physical dimensions (median inflation 3.8x), reflecting the physical reality that real-world encounters feature highly elongated, anisotropic covariances along the orbit track. "
        "Consequently, Monte Carlo simulations in this pipeline serve as numerical integration sampling verification of the calibrated Gaussian, not independent orbit determination.\n"
        "2. Collision Avoidance Maneuver (CAM) Heuristic: The CAM module in orbit_guard.py implements a linearized Clohessy-Wiltshire along-track secular drift calculation for illustrative demonstration. "
        "Real-world flight operations strictly require high-fidelity numerical ephemeris propagation (including J_{70,70} gravity, atmospheric drag covariance, SRP), attitude slew constraints, "
        "thruster dispersion budgets, and mandatory catalog-wide secondary conjunction screening to ensure maneuvers do not induce secondary collisions.\n"
        "3. Sequence Length Sparsity: In our 836-encounter catalog, 59.5% of events contain only 1 to 3 CDM updates. Sequence models (RetNet and Transformer) have minimal temporal context to leverage, "
        "allowing tabular tree ensembles on summary statistics to dominate discriminative performance.\n"
        "4. Single-Epoch Observation Window: The Space-Track CDMs fall within a concentrated observational window, leaving generalization across different solar storm cycles unexamined.\n"
        "5. Exclusion Volume Feature Importance: SAT1_EXCL_VOL appears with high permutation importance in tree models. Because exclusion volume is an input in Space-Track's internal Pc calculation, "
        "its predictive value partly reflects the physical definition of the target label, contributing to the 18-point drop observed when evaluating on unseen satellites."
    )

    # 7. Conclusion
    doc.add_heading("7. Conclusion", level=1)
    doc.add_paragraph(
        "On 836 satellite conjunction events under unified 5-fold cross-validation, tabular gradient boosting (AUC = 0.9086) outperformed deep sequence architectures "
        "(Transformer AUC = 0.8307, RetNet AUC = 0.7506). Sequence models provided no detectable advantage over tabular ensembles, primarily because 59.5% of events have <= 3 updates "
        "and risk is strongly governed by static orbital shell geometry. At validation-tuned operating thresholds targeting >= 90% recall, models achieved 84.3%-92.4% out-of-fold test recall, "
        "with gradient boosting functioning as the most effective triage filter by safely screening out 65.4% of events."
    )

    # References
    doc.add_heading("References", level=1)
    refs = [
        "[1] A. Vaswani et al., \"Attention is all you need,\" Advances in Neural Information Processing Systems, 2017.",
        "[2] Y. Sun et al., \"Retentive Network: A successor to Transformer for large language models,\" Microsoft Research, arXiv:2307.08621, 2023.",
        "[3] H. Touvron et al., \"LLaMA: Open and efficient foundation language models,\" arXiv:2302.13971, 2023.",
        "[4] T.-Y. Lin et al., \"Focal loss for dense object detection,\" IEEE International Conference on Computer Vision, 2017.",
        "[5] J. L. Foster and H. S. Estes, \"A Parametric Analysis of Orbital Debris Collision Probability,\" NASA JSC-25898, 1992.",
        "[6] M. R. Akella and K. T. Alfriend, \"Probability of collision between space objects,\" Journal of Guidance, Control, and Dynamics, vol. 23, no. 4, pp. 769-772, 2000.",
        "[7] K. Chan, \"Collision probability for satellites in close encounters,\" Space Flight, 1997.",
        "[8] J. G. Kemeny and J. L. Snell, Finite Markov Chains. Springer, 1976.",
        "[9] G. E. Box and G. M. Jenkins, Time Series Analysis: Forecasting and Control. Holden-Day, 1976.",
        "[10] T. G. Dietterich, \"Approximate statistical tests for comparing supervised classification learning algorithms,\" Neural Computation, vol. 10, no. 7, pp. 1895-1923, 1998."
    ]
    for r in refs:
        p = doc.add_paragraph(r)
        p.paragraph_format.space_after = Pt(3)

    doc.save(output_path)
    print(f"Successfully generated full manuscript: {output_path}")

if __name__ == "__main__":
    out = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "research_paper.docx")
    build_docx(out)
