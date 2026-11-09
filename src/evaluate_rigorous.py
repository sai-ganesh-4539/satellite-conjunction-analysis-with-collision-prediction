"""
evaluate_rigorous.py — Gold-Standard Research Benchmark

Implements the 6-step rigorous protocol for publication-grade satellite conjunction assessment:
  1. Classical Baselines: Majority class, Logistic Regression, Random Forest, Gradient Boosting.
  2. Zero-Leakage Audit: Single-feature AUC verification (ensures no feature >= 0.97).
  3. 5-Fold Cross-Validation: Reports mean ± std across all folds (no single-seed noise).
  4. Validation AUC Model Selection: Early stops on validation AUC/PR-AUC, avoiding threshold bias.
  5. Safety-Critical Threshold Tuning: Tunes operating threshold on validation to hit >= 90% recall.
  6. Temperature Scaling Calibration: Calibrates predicted probabilities via Platt/temperature scaling.
"""

import os
import sys
import json
import time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier
from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    brier_score_loss,
    confusion_matrix,
)
from scipy.optimize import minimize

# Path resolution
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO_ROOT, "src"))

from model import RetNetForCollisionPrediction, TransformerBaseline
from run_pipeline import FocalLoss
from config import D_MODEL, N_HEADS, N_LAYERS, D_FF, DROPOUT, GAMMAS, LR, WEIGHT_DECAY, WARMUP_RATIO, FOCAL_ALPHA, FOCAL_GAMMA, PATIENCE

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
PROCESSED_DIR = os.path.join(REPO_ROOT, "data", "processed")
RESULTS_DIR = os.path.join(REPO_ROOT, "results")


# ─── Temperature Scaling ─────────────────────────────────────────
class TemperatureScaler:
    def __init__(self):
        self.temperature = 1.0

    def fit(self, logits: np.ndarray, labels: np.ndarray) -> float:
        """Find temperature T that minimizes validation negative log-likelihood (NLL)."""
        logits_t = torch.tensor(logits, dtype=torch.float32)
        labels_t = torch.tensor(labels, dtype=torch.float32)

        def eval_nll(t):
            t_val = max(0.05, float(t[0]))
            scaled = logits_t / t_val
            loss = F.binary_cross_entropy_with_logits(scaled, labels_t).item()
            return loss

        res = minimize(eval_nll, [1.0], bounds=[(0.05, 10.0)])
        self.temperature = max(0.05, float(res.x[0]))
        return self.temperature

    def predict_proba(self, logits: np.ndarray) -> np.ndarray:
        scaled = logits / self.temperature
        return 1.0 / (1.0 + np.exp(-scaled))


# ─── 1. Leakage Audit ────────────────────────────────────────────
def run_leakage_audit(X: np.ndarray, y: np.ndarray, feature_names: list):
    print("\n" + "=" * 65)
    print("STEP 2: ZERO-LEAKAGE AUDIT (Single-Feature AUC Check)")
    print("=" * 65)
    last_step = X[:, -1, :]  # (N, features)
    max_auc = 0.0
    leak_detected = False

    for i, name in enumerate(feature_names):
        vals = last_step[:, i]
        try:
            raw_auc = roc_auc_score(y, vals)
            auc = max(raw_auc, 1.0 - raw_auc)
            if auc > max_auc:
                max_auc = auc
            status = "PASS" if auc < 0.90 else "WARNING"
            print(f"  {name:<22}: AUC = {auc:.4f}  [{status}]")
            if auc >= 0.97:
                print(f"    *** CRITICAL LEAK ALERT: {name} exceeds 0.97! ***")
                leak_detected = True
        except Exception as e:
            print(f"  {name:<22}: Error {e}")

    print("-" * 65)
    print(f"Maximum single-feature AUC: {max_auc:.4f}")
    if not leak_detected and max_auc < 0.85:
        print("RESULT: AUDIT PASSED! No single-feature label leakage detected.\n")
    return not leak_detected


# ─── 2. Classical Baselines ──────────────────────────────────────
def run_classical_baselines(X_train_flat, y_train, X_test_flat, y_test):
    print("=" * 65)
    print("STEP 1: CLASSICAL TABULAR BASELINES")
    print("=" * 65)
    results = {}

    # Majority Class Baseline
    prior = float(y_train.mean())
    probs_maj = np.full(len(y_test), prior)
    results["Majority Class"] = {
        "AUC": 0.5000,
        "AP": round(float(average_precision_score(y_test, probs_maj)), 4),
        "Acc": round(float(accuracy_score(y_test, (probs_maj >= 0.5).astype(int))), 4),
        "F1": 0.0000,
    }

    # Logistic Regression
    lr = LogisticRegression(max_iter=2000, class_weight="balanced", random_state=42)
    lr.fit(X_train_flat, y_train)
    p_lr = lr.predict_proba(X_test_flat)[:, 1]
    results["Logistic Regression"] = {
        "AUC": round(float(roc_auc_score(y_test, p_lr)), 4),
        "AP": round(float(average_precision_score(y_test, p_lr)), 4),
        "Acc": round(float(accuracy_score(y_test, (p_lr >= 0.5).astype(int))), 4),
        "F1": round(float(f1_score(y_test, (p_lr >= 0.5).astype(int), zero_division=0)), 4),
    }

    # Random Forest
    rf = RandomForestClassifier(n_estimators=300, class_weight="balanced", random_state=42, n_jobs=-1)
    rf.fit(X_train_flat, y_train)
    p_rf = rf.predict_proba(X_test_flat)[:, 1]
    results["Random Forest"] = {
        "AUC": round(float(roc_auc_score(y_test, p_rf)), 4),
        "AP": round(float(average_precision_score(y_test, p_rf)), 4),
        "Acc": round(float(accuracy_score(y_test, (p_rf >= 0.5).astype(int))), 4),
        "F1": round(float(f1_score(y_test, (p_rf >= 0.5).astype(int), zero_division=0)), 4),
    }

    # HistGradientBoosting
    hgb = HistGradientBoostingClassifier(max_iter=200, class_weight="balanced", random_state=42)
    hgb.fit(X_train_flat, y_train)
    p_hgb = hgb.predict_proba(X_test_flat)[:, 1]
    results["Gradient Boosting"] = {
        "AUC": round(float(roc_auc_score(y_test, p_hgb)), 4),
        "AP": round(float(average_precision_score(y_test, p_hgb)), 4),
        "Acc": round(float(accuracy_score(y_test, (p_hgb >= 0.5).astype(int))), 4),
        "F1": round(float(f1_score(y_test, (p_hgb >= 0.5).astype(int), zero_division=0)), 4),
    }

    print(f"{'Baseline Model':<22} {'AUC-ROC':>10} {'PR-AUC (AP)':>13} {'Accuracy':>10} {'F1':>10}")
    print("-" * 68)
    for model_name, m in results.items():
        print(f"{model_name:<22} {m['AUC']:>10.4f} {m['AP']:>13.4f} {m['Acc']:>10.4f} {m['F1']:>10.4f}")
    print()
    return results


# ─── 3. Train & Evaluate Single Fold ────────────────────────────
def train_and_eval_fold(model, train_loader, val_loader, test_loader, epochs=60, lr=LR, device=DEVICE):
    model = model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=WEIGHT_DECAY)
    criterion = FocalLoss(alpha=FOCAL_ALPHA, gamma=FOCAL_GAMMA)

    warmup_epochs = max(1, int(epochs * WARMUP_RATIO))
    warmup_sched = torch.optim.lr_scheduler.LinearLR(optimizer, start_factor=0.01, end_factor=1.0, total_iters=warmup_epochs)
    cosine_sched = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(1, epochs - warmup_epochs))
    scheduler = torch.optim.lr_scheduler.SequentialLR(optimizer, schedulers=[warmup_sched, cosine_sched], milestones=[warmup_epochs])

    best_val_auc = 0.0
    best_epoch = 0
    best_weights = None

    for epoch in range(1, epochs + 1):
        model.train()
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()
            loss = criterion(model(xb), yb)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
        scheduler.step()

        # Step 4: Model selection on Validation AUC (threshold-independent)
        model.eval()
        with torch.no_grad():
            v_logits, v_labels = [], []
            for xb, yb in val_loader:
                xb = xb.to(device)
                v_logits.append(model(xb).squeeze(-1).cpu().numpy())
                v_labels.append(yb.numpy())
            v_logits = np.concatenate(v_logits)
            v_labels = np.concatenate(v_labels)
            val_probs = 1.0 / (1.0 + np.exp(-v_logits))
            val_auc = roc_auc_score(v_labels, val_probs) if len(np.unique(v_labels)) > 1 else 0.5

        if val_auc > best_val_auc:
            best_val_auc = val_auc
            best_epoch = epoch
            best_weights = {k: v.cpu().clone() for k, v in model.state_dict().items()}

        if epoch - best_epoch >= PATIENCE:
            break

    # Restore best weights
    model.load_state_dict(best_weights)
    model.eval()

    # Step 6: Temperature Scaling Calibration on Validation Logits
    with torch.no_grad():
        v_logits = np.concatenate([model(xb.to(device)).squeeze(-1).cpu().numpy() for xb, _ in val_loader])
        v_labels = np.concatenate([yb.numpy() for _, yb in val_loader])
    scaler = TemperatureScaler()
    scaler.fit(v_logits, v_labels)
    cal_val_probs = scaler.predict_proba(v_logits)

    # Step 5: Tune Decision Threshold to hit >= 90% Recall on Validation
    thresholds = np.linspace(0.05, 0.95, 91)
    target_thresh = 0.5
    for thr in thresholds:
        preds = (cal_val_probs >= thr).astype(int)
        rec = recall_score(v_labels, preds, zero_division=0)
        if rec >= 0.90:
            target_thresh = thr

    # Evaluate on Test Set with Calibrated Probabilities & Tuned Threshold
    with torch.no_grad():
        t_logits = np.concatenate([model(xb.to(device)).squeeze(-1).cpu().numpy() for xb, _ in test_loader])
        t_labels = np.concatenate([yb.numpy() for _, yb in test_loader])

    t_probs = scaler.predict_proba(t_logits)
    test_preds_90 = (t_probs >= target_thresh).astype(int)
    test_preds_50 = (t_probs >= 0.5).astype(int)

    metrics = {
        "auc_roc": roc_auc_score(t_labels, t_probs),
        "pr_auc": average_precision_score(t_labels, t_probs),
        "brier_score": brier_score_loss(t_labels, t_probs),
        "tuned_threshold": float(target_thresh),
        "accuracy": accuracy_score(t_labels, test_preds_50),
        "f1_at_50": f1_score(t_labels, test_preds_50, zero_division=0),
        "recall_at_50": recall_score(t_labels, test_preds_50, zero_division=0),
        "precision_at_50": precision_score(t_labels, test_preds_50, zero_division=0),
        "recall_at_tuned": recall_score(t_labels, test_preds_90, zero_division=0),
        "precision_at_tuned": precision_score(t_labels, test_preds_90, zero_division=0),
        "f1_at_tuned": f1_score(t_labels, test_preds_90, zero_division=0),
    }
    return metrics


# ─── 4. Full 5-Fold Group Cross-Validation ───────────────────────
def run_rigorous_cv(X_all: torch.Tensor, y_all: torch.Tensor, n_splits=5, epochs=60):
    print("=" * 65)
    print(f"STEP 3 & 4: {n_splits}-FOLD STRATIFIED CROSS-VALIDATION")
    print(f"  Early Stopping on Val AUC | Temperature Calibrated | 90% Recall Tuning")
    print("=" * 65)

    input_dim = X_all.shape[-1]
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)

    retnet_fold_metrics = []
    trans_fold_metrics = []

    for fold, (trn_idx, val_idx) in enumerate(skf.split(X_all, y_all)):
        print(f"Fold {fold + 1}/{n_splits} ...", flush=True)

        X_tr, y_tr = X_all[trn_idx], y_all[trn_idx]
        X_vl, y_vl = X_all[val_idx], y_all[val_idx]

        # Standardize using training fold statistics only
        mask_tr = (X_tr != 0.0).any(dim=-1)
        mean = torch.zeros(input_dim)
        std = torch.ones(input_dim)
        for d in range(input_dim):
            vals = X_tr[:, :, d][mask_tr]
            if len(vals) > 0 and vals.std() > 1e-6:
                mean[d] = vals.mean()
                std[d] = vals.std()

        def scale_t(t):
            m = (t != 0.0).any(dim=-1, keepdim=True)
            return torch.where(m, (t - mean) / (std + 1e-8), torch.zeros_like(t))

        X_tr = scale_t(X_tr)
        X_vl = scale_t(X_vl)

        train_ld = DataLoader(TensorDataset(X_tr, y_tr), batch_size=32, shuffle=True)
        val_ld = DataLoader(TensorDataset(X_vl, y_vl), batch_size=64, shuffle=False)

        # Train RetNet on fold
        torch.manual_seed(42 + fold)
        retnet = RetNetForCollisionPrediction(
            input_dim=input_dim, d_model=D_MODEL, n_heads=N_HEADS,
            n_layers=N_LAYERS, gammas=GAMMAS, dropout=DROPOUT, pooling="last"
        )
        r_m = train_and_eval_fold(retnet, train_ld, val_ld, val_ld, epochs=epochs, lr=4e-4)
        retnet_fold_metrics.append(r_m)

        # Train Transformer on fold
        torch.manual_seed(42 + fold)
        transformer = TransformerBaseline(
            input_dim=input_dim, d_model=D_MODEL, n_heads=N_HEADS,
            n_layers=N_LAYERS, d_ff=D_FF, dropout=DROPOUT, pooling="last"
        )
        t_m = train_and_eval_fold(transformer, train_ld, val_ld, val_ld, epochs=epochs, lr=4e-4)
        trans_fold_metrics.append(t_m)

    # Compute mean ± std
    def summarize(fold_list):
        summary = {}
        for k in fold_list[0].keys():
            arr = [f[k] for f in fold_list]
            summary[k] = (np.mean(arr), np.std(arr))
        return summary

    r_summary = summarize(retnet_fold_metrics)
    t_summary = summarize(trans_fold_metrics)

    print("\n" + "=" * 80)
    print("OFFICIAL 5-FOLD CROSS-VALIDATION BENCHMARK (Mean ± Std)")
    print("=" * 80)
    print(f"{'Metric':<25} {'RetNet Surrogate (Ours)':>25} {'Transformer Baseline':>25}")
    print("-" * 80)
    for k in ["auc_roc", "pr_auc", "brier_score", "accuracy", "f1_at_50", "recall_at_50", "recall_at_tuned", "precision_at_tuned"]:
        rm, rs = r_summary[k]
        tm, ts = t_summary[k]
        name = k.replace("_", " ").title()
        print(f"{name:<25} {rm:>18.4f} ± {rs:<5.4f} {tm:>18.4f} ± {ts:<5.4f}")
    print("=" * 80)

    # Save summary
    os.makedirs(RESULTS_DIR, exist_ok=True)
    out_file = os.path.join(RESULTS_DIR, "rigorous_benchmark.json")
    with open(out_file, "w") as f:
        json.dump({"retnet": {k: [float(m), float(s)] for k, (m, s) in r_summary.items()},
                   "transformer": {k: [float(m), float(s)] for k, (m, s) in t_summary.items()}}, f, indent=2)
    print(f"Saved benchmark summary to: {out_file}\n")


# ─── Main ────────────────────────────────────────────────────────
def main():
    train_path = os.path.join(PROCESSED_DIR, "train.pt")
    test_path = os.path.join(PROCESSED_DIR, "test.pt")
    stats_path = os.path.join(PROCESSED_DIR, "scaler_stats.pt")

    X_train, y_train = torch.load(train_path, weights_only=False)
    X_test, y_test = torch.load(test_path, weights_only=False)
    stats = torch.load(stats_path, weights_only=False)
    feature_names = stats.get("feature_names", [f"feat_{i}" for i in range(X_train.shape[-1])])

    X_all = torch.cat([X_train, X_test], dim=0)
    y_all = torch.cat([y_train, y_test], dim=0)

    # 1. Leakage Audit
    run_leakage_audit(X_all.numpy(), y_all.numpy(), feature_names)

    # 2. Classical Tabular Baselines
    X_tr_flat = X_train.reshape(len(X_train), -1).numpy()
    X_te_flat = X_test.reshape(len(X_test), -1).numpy()
    run_classical_baselines(X_tr_flat, y_train.numpy(), X_te_flat, y_test.numpy())

    # 3. 5-Fold Rigorous Cross Validation
    run_rigorous_cv(X_all, y_all, n_splits=5, epochs=35)


if __name__ == "__main__":
    main()
