"""
baselines.py  —  TIER 1 SANITY CHECK

Trains deliberately simple models on EXACTLY the same train/test split as
the deep models, with the same metrics. Purpose:

    1. Majority class      -> the "do nothing" floor. Shows why accuracy
                              alone lies on imbalanced data (~84% accuracy,
                              F1 = 0).
    2. Logistic regression -> linear floor on flattened features.
    3. Random Forest       -> classical ML, no sequence structure needed.
    4. Gradient boosting   -> usually the strongest classical baseline.

How to read the result (the decision rule):
    - Deep models must beat the BEST classical baseline by more than the
      seed noise (~ +-0.03-0.04 AUC, from your 3-seed study) to justify
      their extra complexity.
    - If RetNet/Transformer CANNOT beat gradient boosting, report it —
      that is a real finding, and far better than being asked live at the
      review "did you compare against a simple model?".

Usage (from the project root):
    python src/baselines.py
"""

import argparse
import json
import os

import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    average_precision_score,
    confusion_matrix,
)

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

SEED = 42
DEEP_COLORS = {"retnet": "#1f77b4", "transformer": "#d62728"}
CLASSIC_COLOR = "#7f7f7f"
DEEP_LABELS = {"retnet": "RetNet (ours)", "transformer": "Transformer"}

plt.rcParams.update({
    "figure.dpi": 110,
    "savefig.dpi": 200,
    "font.size": 11,
    "axes.grid": True,
    "grid.alpha": 0.3,
    "axes.spines.top": False,
    "axes.spines.right": False,
})


# ─── Helpers ──────────────────────────────────────────────────────

def flatten(X: torch.Tensor) -> np.ndarray:
    """(N, T, F) -> (N, T*F). Classical models get ALL timesteps as plain
    features — no sequence structure, which is exactly the point."""
    return X.reshape(X.shape[0], -1).numpy()


def metrics_at(y_true, probs, thr=0.5) -> dict:
    preds = (probs >= thr).astype(int)
    return {
        "auc_roc": round(float(roc_auc_score(y_true, probs)), 4),
        "avg_precision": round(float(average_precision_score(y_true, probs)), 4),
        "accuracy": round(float(accuracy_score(y_true, preds)), 4),
        "precision": round(float(precision_score(y_true, preds, zero_division=0)), 4),
        "recall": round(float(recall_score(y_true, preds, zero_division=0)), 4),
        "f1": round(float(f1_score(y_true, preds, zero_division=0)), 4),
        "confusion_matrix": confusion_matrix(y_true, preds).tolist(),
    }


# ─── Main ─────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Tier-1 sanity baselines")
    parser.add_argument("--data_dir", type=str, default="data/processed")
    parser.add_argument("--checkpoint_dir", type=str, default="checkpoints")
    parser.add_argument("--output_dir", type=str, default="figures")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    np.random.seed(SEED)

    # --- Data (identical splits to the deep models) ---
    print("Loading data...")
    X_train, y_train = torch.load(os.path.join(args.data_dir, "train.pt"), weights_only=False)
    X_test, y_test = torch.load(os.path.join(args.data_dir, "test.pt"), weights_only=False)
    y_train, y_test = y_train.numpy(), y_test.numpy()

    F_train = flatten(X_train)
    F_test = flatten(X_test)
    print(f"  Train: {F_train.shape[0]} events x {F_train.shape[1]} flattened features")
    print(f"  Test : {F_test.shape[0]} events ({int(y_test.sum())} high-risk)")

    results = {}

    # --- 1. Majority class (predict ALL low-risk) ---
    pos_rate = float(y_train.mean())
    probs = np.full(len(y_test), pos_rate)
    results["Majority class"] = metrics_at(y_test, probs)
    print("\n[1/4] Majority class ... done (always predicts LOW-risk)")

    # --- 2. Logistic regression ---
    print("[2/4] Logistic regression ...", end=" ", flush=True)
    lr = LogisticRegression(max_iter=5000, class_weight="balanced", random_state=SEED)
    lr.fit(F_train, y_train)
    results["Logistic regression"] = metrics_at(y_test, lr.predict_proba(F_test)[:, 1])
    print("done")

    # --- 3. Random forest ---
    print("[3/4] Random forest ...", end=" ", flush=True)
    rf = RandomForestClassifier(
        n_estimators=500, class_weight="balanced", random_state=SEED, n_jobs=-1
    )
    rf.fit(F_train, y_train)
    results["Random forest"] = metrics_at(y_test, rf.predict_proba(F_test)[:, 1])
    print("done")

    # --- 4. Gradient boosting ---
    print("[4/4] Gradient boosting ...", end=" ", flush=True)
    gb = HistGradientBoostingClassifier(
        max_iter=300, learning_rate=0.1, class_weight="balanced", random_state=SEED
    )
    gb.fit(F_train, y_train)
    results["Gradient boosting"] = metrics_at(y_test, gb.predict_proba(F_test)[:, 1])
    print("done")

    # --- Add deep-model test metrics from train.py history files ---
    # Avg precision for deep models is not saved by train.py; pull it from
    # evaluate.py's summary if that has been run, otherwise show "-".
    deep_ap = {}
    eval_summary_path = os.path.join(args.output_dir, "evaluation_summary.json")
    if os.path.exists(eval_summary_path):
        with open(eval_summary_path) as f:
            for name, m in json.load(f).items():
                deep_ap[DEEP_LABELS[name]] = m["test@0.5"].get("average_precision")

    for name in ["retnet", "transformer"]:
        path = os.path.join(args.checkpoint_dir, f"history_{name}.json")
        if os.path.exists(path):
            with open(path) as f:
                tm = json.load(f)["test_metrics"]
            results[DEEP_LABELS[name]] = {
                "auc_roc": round(tm["auc_roc"], 4),
                "avg_precision": deep_ap.get(DEEP_LABELS[name]),
                "accuracy": round(tm["accuracy"], 4),
                "precision": round(tm["precision"], 4),
                "recall": round(tm["recall"], 4),
                "f1": round(tm["f1"], 4),
                "confusion_matrix": tm["confusion_matrix"],
            }

    # --- Print + save comparison table ---
    def fmt(v):
        return f"{v:>9.4f}" if isinstance(v, (int, float)) else f"{'-':>9}"

    print("\n" + "=" * 86)
    print(f"{'MODEL COMPARISON — TEST SET':^86}")
    print("=" * 86)
    print(f"{'Model':<22}{'AUC-ROC':>9}{'AvgPrec':>9}{'Acc':>8}{'Prec':>8}{'Rec':>8}{'F1':>8}")
    print("-" * 86)
    for name, m in results.items():
        print(
            f"{name:<22}{fmt(m['auc_roc'])}{fmt(m['avg_precision'])}"
            f"{m['accuracy']:>8.4f}{m['precision']:>8.4f}{m['recall']:>8.4f}{m['f1']:>8.4f}"
        )
    print("=" * 86)

    with open(os.path.join(args.output_dir, "baselines_summary.json"), "w") as f:
        json.dump(results, f, indent=2)

    # --- Markdown for the report ---
    lines = [
        "## Baseline Comparison (Test Set)\n",
        "| Model | AUC-ROC | Avg Precision | Accuracy | Precision | Recall | F1 |",
        "|---|---|---|---|---|---|---|",
    ]
    for name, m in results.items():
        ap = f"{m['avg_precision']:.3f}" if isinstance(m['avg_precision'], (int, float)) else "-"
        lines.append(
            f"| {name} | {m['auc_roc']:.3f} | {ap} | "
            f"{m['accuracy']:.3f} | {m['precision']:.3f} | {m['recall']:.3f} | {m['f1']:.3f} |"
        )
    lines += [
        "",
        "*All models use the identical train/test split (seed 42). Classical models "
        "receive the same 12x15 CDM sequences flattened into 180 features. Deep-model "
        "numbers are from `train.py` (best checkpoints).*",
    ]
    with open(os.path.join(args.output_dir, "BASELINES.md"), "w") as f:
        f.write("\n".join(lines) + "\n")

    # --- Bar chart: AUC + F1 for every model ---
    names = list(results.keys())
    auc_vals = [results[n]["auc_roc"] for n in names]
    f1_vals = [results[n]["f1"] for n in names]
    colors = [
        DEEP_COLORS.get("retnet") if n == DEEP_LABELS["retnet"]
        else DEEP_COLORS.get("transformer") if n == DEEP_LABELS["transformer"]
        else CLASSIC_COLOR
        for n in names
    ]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12.6, 4.8), constrained_layout=True)
    y = np.arange(len(names))
    for ax, vals, title in [(ax1, auc_vals, "AUC-ROC"), (ax2, f1_vals, "F1-score")]:
        ax.barh(y, vals, color=colors, alpha=0.9)
        ax.set_yticks(y, names, fontsize=10)
        ax.invert_yaxis()
        ax.set_xlim(0, 1)
        ax.set_xlabel(title)
        ax.set_title(title + " — Test Set")
        for i, v in enumerate(vals):
            ax.text(v + 0.01, i, f"{v:.3f}", va="center", fontsize=9)
    fig.suptitle("Deep models vs classical baselines\n(blue/red = deep, gray = classical)", fontsize=12)
    fig.savefig(os.path.join(args.output_dir, "baseline_comparison.png"))
    plt.close(fig)

    # --- Verdict ---
    classic = [results[n]["auc_roc"] for n in results if n not in DEEP_LABELS.values()]
    best_classic = max(classic) if classic else None
    print(f"\nSaved: {args.output_dir}/BASELINES.md, baselines_summary.json, baseline_comparison.png")

    if best_classic is not None:
        print(f"\nBest classical AUC: {best_classic:.4f}")
        for name in DEEP_LABELS.values():
            if name in results:
                diff = results[name]["auc_roc"] - best_classic
                verdict = "BEATS classical ML" if diff > 0.03 else (
                    "COMPARABLE to classical ML" if diff > -0.01 else "LOSES to classical ML")
                print(f"  {name}: {results[name]['auc_roc']:.4f} ({diff:+.4f} vs best classical) -> {verdict}")


if __name__ == "__main__":
    main()
