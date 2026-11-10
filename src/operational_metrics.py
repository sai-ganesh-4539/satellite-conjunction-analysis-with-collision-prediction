"""Calibration, screening rate, and threshold selection shared by the reports."""

from __future__ import annotations

import numpy as np
from sklearn.metrics import brier_score_loss


def expected_calibration_error(y_true: np.ndarray, y_prob: np.ndarray, n_bins: int = 10) -> float:
    y_true = np.asarray(y_true, dtype=float)
    y_prob = np.asarray(y_prob, dtype=float)
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    n = len(y_true)
    if n == 0:
        return float("nan")
    for i in range(n_bins):
        lo, hi = edges[i], edges[i + 1]
        if i == n_bins - 1:
            mask = (y_prob >= lo) & (y_prob <= hi)
        else:
            mask = (y_prob >= lo) & (y_prob < hi)
        if not np.any(mask):
            continue
        ece += (mask.sum() / n) * abs(y_true[mask].mean() - y_prob[mask].mean())
    return float(ece)


def reliability_diagram(y_true: np.ndarray, y_prob: np.ndarray, n_bins: int = 10) -> dict:
    y_true = np.asarray(y_true, dtype=float)
    y_prob = np.asarray(y_prob, dtype=float)
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    centers, acc, conf, counts = [], [], [], []
    for i in range(n_bins):
        lo, hi = edges[i], edges[i + 1]
        if i == n_bins - 1:
            mask = (y_prob >= lo) & (y_prob <= hi)
        else:
            mask = (y_prob >= lo) & (y_prob < hi)
        centers.append(0.5 * (lo + hi))
        counts.append(int(mask.sum()))
        if np.any(mask):
            acc.append(float(y_true[mask].mean()))
            conf.append(float(y_prob[mask].mean()))
        else:
            acc.append(float("nan"))
            conf.append(float("nan"))
    return {
        "bin_center": centers,
        "accuracy": acc,
        "confidence": conf,
        "count": counts,
        "ece": expected_calibration_error(y_true, y_prob, n_bins),
        "brier": float(brier_score_loss(y_true, np.clip(y_prob, 0.0, 1.0))) if len(y_true) else float("nan"),
    }


def fit_temperature(logits: np.ndarray, labels: np.ndarray) -> float:
    """Scalar temperature minimizing Bernoulli NLL on validation logits."""
    from scipy.optimize import minimize_scalar

    z = np.asarray(logits, dtype=float).reshape(-1)
    y = np.asarray(labels, dtype=float).reshape(-1)

    def nll(log_t):
        t = np.exp(log_t)
        p = 1.0 / (1.0 + np.exp(-z / t))
        p = np.clip(p, 1e-8, 1 - 1e-8)
        return float(-(y * np.log(p) + (1 - y) * np.log(1 - p)).mean())

    fit = minimize_scalar(nll, bounds=(-3.0, 3.0), method="bounded")
    return float(np.exp(fit.x))


def apply_temperature(logits: np.ndarray, temperature: float) -> np.ndarray:
    z = np.asarray(logits, dtype=float) / float(temperature)
    return 1.0 / (1.0 + np.exp(-z))


def threshold_for_recall(y_true: np.ndarray, y_prob: np.ndarray, target_recall: float = 0.90) -> float:
    """Lowest threshold on this set whose recall is at least ``target_recall``.

    If no threshold reaches the target, the threshold that recalls every
    positive (just below the smallest positive score, or 0) is returned.
    """
    y_true = np.asarray(y_true).astype(int)
    y_prob = np.asarray(y_prob, dtype=float)
    pos = np.sort(y_prob[y_true == 1])
    if len(pos) == 0:
        return 1.0
    # Drop at most the bottom (1 - target) fraction of positive scores.
    n_drop = int(np.floor((1.0 - target_recall) * len(pos)))
    n_drop = min(n_drop, len(pos) - 1)
    return float(pos[n_drop])


def screening_flag_rate(prevalence: float, recall: float, precision: float) -> float:
    """Fraction of events sent on for a detailed run.

    Flag rate = (prevalence * recall) / precision, which is TP+FP over N
    when precision and recall describe the same operating point.
    """
    if precision <= 0:
        return float("nan")
    return float(prevalence * recall / precision)
