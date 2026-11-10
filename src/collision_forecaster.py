"""
Multi-horizon forecaster for the last CDM in a conjunction track.

Each event is one primary/secondary pair sorted by CDM_ID. The last message
is the prediction target (miss distance, catalog Pc, and the triage class).
Earlier messages are the only inputs. Catalog Pc is never a feature.

Sequence models are the RetNet and Transformer already in ``model.py``, with
the classification head replaced by miss, Pc, and class heads. The tabular
model is a 71-feature HistGradientBoosting ensemble: five statistics on each
of the 14 geometry channels, plus the number of CDMs seen.
"""

from __future__ import annotations

import os
import sys
import time
from dataclasses import dataclass

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, r2_score, roc_auc_score
from sklearn.model_selection import GroupShuffleSplit, StratifiedKFold

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO_ROOT, "src"))

from config import (  # noqa: E402
    D_FF, D_MODEL, DROPOUT, FOCAL_ALPHA, FOCAL_GAMMA, GAMMAS, N_HEADS, N_LAYERS,
    PC_THRESHOLD, SEQ_LEN,
)
from data_processing import (  # noqa: E402
    DEFAULT_DATA_CSV, DEFAULT_SATCAT_CSV, FEATURE_NAMES, load_and_merge_cdms, satellite_components,
)
from model import FocalLoss, RetNetForCollisionPrediction, TransformerBaseline  # noqa: E402
from monte_carlo_engine import PC_CRITICAL, PC_SAFE, safety_class  # noqa: E402

MODEL_DIR = os.path.join(REPO_ROOT, "results", "models")
N_SUMMARY = 14 * 5 + 1  # 71


@dataclass
class ForecastBatch:
    X: np.ndarray            # (N, T, F) left-padded raw features
    summary: np.ndarray      # (N, 71)
    log_miss: np.ndarray
    log_pc: np.ndarray
    y: np.ndarray
    miss_m: np.ndarray
    pc: np.ndarray
    sat1: np.ndarray
    sat2: np.ndarray
    component: np.ndarray
    n_prefix: np.ndarray
    event_id: np.ndarray
    prefixes: list


def _sequence_matrix(g: pd.DataFrame) -> np.ndarray:
    """14 geometry channels. Depends only on rows inside ``g``."""
    rng = g["MIN_RNG"].to_numpy(dtype=float)
    n = len(g)
    log_rng = np.log10(np.clip(rng, 1.0, None))
    prev = np.concatenate([[rng[0]], rng[:-1]])
    d_rng = np.clip((rng - prev) / 100.0, -10.0, 10.0)
    ratio = rng / (rng[0] + 1.0)
    pos = np.arange(n, dtype=float) / float(SEQ_LEN)
    feat = np.stack([
        log_rng, d_rng, ratio, pos,
        g["DELTA_APOGEE"].to_numpy(float),
        g["DELTA_PERIGEE"].to_numpy(float),
        g["DELTA_INCLINATION"].to_numpy(float),
        g["DELTA_PERIOD"].to_numpy(float),
        g["SAT1_RCS_NUM"].to_numpy(float),
        g["SAT2_RCS_NUM"].to_numpy(float),
        g["SAT1_TYPE_NUM"].to_numpy(float),
        g["SAT2_TYPE_NUM"].to_numpy(float),
        g["SAT1_EXCL_VOL"].to_numpy(float),
        g["SAT2_EXCL_VOL"].to_numpy(float),
    ], axis=-1).astype(np.float32)
    if feat.shape[1] != len(FEATURE_NAMES):
        raise RuntimeError("feature layout drifted from FEATURE_NAMES")
    return feat


def summarize_71(feat: np.ndarray) -> np.ndarray:
    """last, mean, min, max, std of each channel, then the CDM count."""
    last = feat[-1]
    return np.concatenate([
        last, feat.mean(0), feat.min(0), feat.max(0), feat.std(0),
        np.array([feat.shape[0]], dtype=np.float32),
    ]).astype(np.float32)


def _pad_left(feat: np.ndarray, seq_len: int = SEQ_LEN) -> np.ndarray:
    n, f = feat.shape
    if n >= seq_len:
        return feat[-seq_len:].astype(np.float32)
    out = np.zeros((seq_len, f), np.float32)
    out[-n:] = feat
    return out


def load_forecast_events(cdm_path=DEFAULT_DATA_CSV, satcat_path=DEFAULT_SATCAT_CSV) -> ForecastBatch:
    """One row per pair with at least two CDMs and a finite final Pc."""
    df = load_and_merge_cdms(cdm_path, satcat_path)
    xs, summaries, log_miss, log_pc, ys = [], [], [], [], []
    miss, pcs, s1s, s2s, npre, eids, prefixes = [], [], [], [], [], [], []
    for (s1, s2), g in df.groupby(["SAT_1_ID", "SAT_2_ID"], sort=False):
        g = g.sort_values("CDM_ID")
        if len(g) < 2:
            continue
        final = g.iloc[-1]
        pc = final["PC"]
        if not np.isfinite(pc):
            continue
        feat = _sequence_matrix(g.iloc[:-1])
        prefixes.append(feat)
        xs.append(_pad_left(feat))
        summaries.append(summarize_71(feat))
        miss_m = float(final["MIN_RNG"])
        pc = float(pc)
        log_miss.append(np.log10(max(miss_m, 1.0)))
        log_pc.append(np.log10(np.clip(pc, 1e-12, 1.0)))
        ys.append(1.0 if pc >= PC_THRESHOLD else 0.0)
        miss.append(miss_m)
        pcs.append(pc)
        s1s.append(s1)
        s2s.append(s2)
        npre.append(feat.shape[0])
        eids.append(f"{int(s1)}_{int(s2)}")
    sat1 = np.array(s1s)
    sat2 = np.array(s2s)
    return ForecastBatch(
        X=np.stack(xs).astype(np.float32),
        summary=np.stack(summaries).astype(np.float32),
        log_miss=np.asarray(log_miss, np.float32),
        log_pc=np.asarray(log_pc, np.float32),
        y=np.asarray(ys, np.float32),
        miss_m=np.asarray(miss, np.float64),
        pc=np.asarray(pcs, np.float64),
        sat1=sat1,
        sat2=sat2,
        component=satellite_components(sat1, sat2),
        n_prefix=np.asarray(npre, np.int32),
        event_id=np.asarray(eids),
        prefixes=prefixes,
    )


def scale_sequences(X: np.ndarray, mean: np.ndarray, std: np.ndarray) -> np.ndarray:
    real = (np.abs(X).sum(-1, keepdims=True) > 0)
    return np.where(real, (X - mean) / (std + 1e-8), 0.0).astype(np.float32)


def fit_scaler(X: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    real = np.abs(X).sum(-1) > 0
    mean = np.zeros(X.shape[-1], np.float32)
    std = np.ones(X.shape[-1], np.float32)
    for d in range(X.shape[-1]):
        v = X[:, :, d][real]
        if len(v) and float(v.std()) > 1e-6:
            mean[d] = float(v.mean())
            std[d] = float(v.std())
    return mean, std


class _Heads(nn.Module):
    def __init__(self, d_model: int, dropout: float):
        super().__init__()
        self.miss = nn.Linear(d_model, 2)
        self.pc = nn.Linear(d_model, 2)
        self.cls = nn.Linear(d_model, 1)
        self.drop = nn.Dropout(dropout)

    def forward(self, h):
        h = self.drop(h)
        return {"miss": self.miss(h), "pc": self.pc(h), "cls": self.cls(h)}


class MultiTargetForecaster(nn.Module):
    def __init__(self, kind: str, input_dim: int, d_model=D_MODEL, n_heads=N_HEADS,
                 n_layers=N_LAYERS, dropout=DROPOUT, max_len=SEQ_LEN):
        super().__init__()
        if kind == "retnet":
            backbone = RetNetForCollisionPrediction(
                input_dim, d_model=d_model, n_heads=n_heads, n_layers=n_layers,
                gammas=GAMMAS, dropout=dropout, pooling="last", retention_mode="matrix",
                max_len=max_len,
            )
        elif kind == "transformer":
            backbone = TransformerBaseline(
                input_dim, d_model=d_model, n_heads=n_heads, n_layers=n_layers,
                d_ff=D_FF, dropout=dropout, pooling="last", max_len=max_len,
            )
        else:
            raise ValueError(kind)
        backbone.classifier = nn.Identity()
        self.kind = kind
        self.backbone = backbone
        self.heads = _Heads(d_model, dropout)

    def forward(self, x):
        return self.heads(self.backbone(x))

    def step(self, x_t, timestep: int, states=None):
        if self.kind != "retnet":
            raise NotImplementedError("O(1) streaming state is the RetNet path")
        h, states = self.backbone.forward_recurrent(x_t, timestep=timestep, states=states)
        return self.heads(h), states


def multitask_loss(out: dict, log_miss, log_pc, y, focal: FocalLoss) -> torch.Tensor:
    cls = focal(out["cls"], y)
    mu_m = out["miss"][:, 0]
    huber = F.huber_loss(mu_m, log_miss)
    mu_p = out["pc"][:, 0]
    log_var = out["pc"][:, 1].clamp(-6.0, 4.0)
    nll = 0.5 * (log_var + (log_pc - mu_p) ** 2 / log_var.exp()).mean()
    return cls + 0.5 * huber + 0.5 * nll


def _predict_torch(model, X, device, batch=256) -> dict:
    model.eval()
    miss, logv_m, pc, logv_p, logit = [], [], [], [], []
    with torch.inference_mode():
        for i in range(0, len(X), batch):
            xb = torch.from_numpy(X[i:i + batch]).to(device)
            out = model(xb)
            miss.append(out["miss"][:, 0].cpu().numpy())
            logv_m.append(out["miss"][:, 1].clamp(-6, 4).cpu().numpy())
            pc.append(out["pc"][:, 0].cpu().numpy())
            logv_p.append(out["pc"][:, 1].clamp(-6, 4).cpu().numpy())
            logit.append(out["cls"].squeeze(-1).cpu().numpy())
    mu_m = np.concatenate(miss)
    mu_p = np.concatenate(pc)
    sig_m = np.sqrt(np.exp(np.concatenate(logv_m)))
    sig_p = np.sqrt(np.exp(np.concatenate(logv_p)))
    logits = np.concatenate(logit)
    return _pack_prediction(mu_m, sig_m, mu_p, sig_p, 1.0 / (1.0 + np.exp(-logits)), logits)


def _pack_prediction(mu_m, sig_m, mu_p, sig_p, prob, logits) -> dict:
    miss = np.power(10.0, mu_m)
    pc = np.clip(np.power(10.0, mu_p), 1e-12, 1.0)
    return {
        "miss_m": miss,
        "miss_lo_m": np.power(10.0, mu_m - sig_m),
        "miss_hi_m": np.power(10.0, mu_m + sig_m),
        "pc": pc,
        "pc_lo": np.clip(np.power(10.0, mu_p - sig_p), 1e-12, 1.0),
        "pc_hi": np.clip(np.power(10.0, mu_p + sig_p), 1e-12, 1.0),
        "log10_pc": mu_p,
        "class_prob": np.clip(prob, 0.0, 1.0),
        "logits": logits,
        "safety": [safety_class(float(p)) for p in pc],
    }


class HistGBForecaster:
    """Median and 16/84 percentile regressors, plus a classifier on the same 71 features."""

    def __init__(self, seed: int = 42):
        self.seed = seed
        self.clf = HistGradientBoostingClassifier(max_iter=200, learning_rate=0.08, random_state=seed)
        self.miss = {
            q: HistGradientBoostingRegressor(
                loss="quantile", quantile=q, max_iter=200, learning_rate=0.08, random_state=seed,
            )
            for q in (0.16, 0.50, 0.84)
        }
        self.pc = {
            q: HistGradientBoostingRegressor(
                loss="quantile", quantile=q, max_iter=200, learning_rate=0.08, random_state=seed,
            )
            for q in (0.16, 0.50, 0.84)
        }

    def fit(self, summary, log_miss, log_pc, y):
        self.clf.fit(summary, y.astype(int))
        for model, target in (
            (self.miss, log_miss),
            (self.pc, log_pc),
        ):
            for reg in model.values():
                reg.fit(summary, target)
        return self

    def predict(self, summary) -> dict:
        mu_m = self.miss[0.50].predict(summary)
        mu_p = self.pc[0.50].predict(summary)
        sig_m = np.maximum(self.miss[0.84].predict(summary) - self.miss[0.16].predict(summary), 1e-3) / 2.0
        sig_p = np.maximum(self.pc[0.84].predict(summary) - self.pc[0.16].predict(summary), 1e-3) / 2.0
        prob = self.clf.predict_proba(summary)[:, 1]
        logits = np.log(np.clip(prob, 1e-6, 1 - 1e-6) / np.clip(1 - prob, 1e-6, 1 - 1e-6))
        return _pack_prediction(mu_m, sig_m, mu_p, sig_p, prob, logits)


def _train_net(kind, X_train, train_targets, X_val, val_targets, device, epochs, seed) -> MultiTargetForecaster:
    torch.manual_seed(seed)
    model = MultiTargetForecaster(kind, X_train.shape[-1]).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-5)
    focal = FocalLoss(FOCAL_ALPHA, FOCAL_GAMMA)
    log_miss, log_pc, y = train_targets
    va_miss, va_pc, va_y = val_targets
    best, best_state, stale = float("inf"), None, 0
    n = len(X_train)
    rng = np.random.default_rng(seed)
    xb_val = torch.from_numpy(X_val).to(device)
    t_miss = torch.from_numpy(va_miss).to(device)
    t_pc = torch.from_numpy(va_pc).to(device)
    t_y = torch.from_numpy(va_y).to(device)
    for _epoch in range(epochs):
        model.train()
        order = rng.permutation(n)
        for i in range(0, n, 64):
            b = order[i:i + 64]
            xb = torch.from_numpy(X_train[b]).to(device)
            loss = multitask_loss(
                model(xb),
                torch.from_numpy(log_miss[b]).to(device),
                torch.from_numpy(log_pc[b]).to(device),
                torch.from_numpy(y[b]).to(device),
                focal,
            )
            opt.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
        model.eval()
        with torch.no_grad():
            vloss = float(multitask_loss(model(xb_val), t_miss, t_pc, t_y, focal))
        if vloss < best - 1e-4:
            best = vloss
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            stale = 0
        else:
            stale += 1
            if stale >= 4:
                break
    if best_state is not None:
        model.load_state_dict(best_state)
    model.eval()
    return model


def train_one_network(kind, X_train, train_targets, X_val, val_targets, device, epochs, seed):
    return _train_net(kind, X_train, train_targets, X_val, val_targets, device, epochs, seed)


def regression_report(pred: dict, batch: ForecastBatch, idx: np.ndarray) -> dict:
    pc_hat = pred["pc"]
    pc = batch.pc[idx]
    miss_hat = pred["miss_m"]
    miss = batch.miss_m[idx]
    log_hat = np.log10(np.clip(pc_hat, 1e-12, 1))
    log_true = np.log10(np.clip(pc, 1e-12, 1))
    spearman = _spearman(log_hat, log_true)
    y = batch.y[idx]
    prob = pred["class_prob"]
    auc = float(roc_auc_score(y, prob)) if len(np.unique(y)) > 1 else float("nan")
    return {
        "n": int(len(idx)),
        "pc_r2": float(r2_score(log_true, log_hat)),
        "pc_mae_log10": float(mean_absolute_error(log_true, log_hat)),
        "pc_mae": float(mean_absolute_error(pc, pc_hat)),
        "spearman_log10_pc": spearman,
        "miss_mae_m": float(mean_absolute_error(miss, miss_hat)),
        "miss_r2_log10": float(r2_score(np.log10(np.clip(miss, 1, None)), np.log10(np.clip(miss_hat, 1, None)))),
        "class_auc": auc,
        "safety_counts": {k: int(sum(s == k for s in pred["safety"])) for k in (
            "SAFE", "MONITOR / ELEVATED", "CRITICAL / ACTION REQUIRED")},
    }


def _spearman(a, b) -> float:
    ra = pd.Series(a).rank().to_numpy()
    rb = pd.Series(b).rank().to_numpy()
    if ra.std() == 0 or rb.std() == 0:
        return float("nan")
    return float(np.corrcoef(ra, rb)[0, 1])


def component_split(batch: ForecastBatch, seed: int = 42, test_size: float = 0.2):
    splitter = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=seed)
    train, test = next(splitter.split(batch.X, batch.y, batch.component))
    inner = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=seed)
    tr, va = next(inner.split(batch.X[train], batch.y[train], batch.component[train]))
    return train[tr], train[va], test


def save_bundle(path_stem, hist, retnet, transformer, mean, std, meta: dict):
    os.makedirs(os.path.dirname(path_stem), exist_ok=True)
    import joblib
    joblib.dump({"hist": hist, "mean": mean, "std": std, "meta": meta}, path_stem + "_hist.joblib")
    torch.save(retnet.state_dict(), path_stem + "_retnet.pt")
    torch.save(transformer.state_dict(), path_stem + "_transformer.pt")
    np.savez(path_stem + "_scaler.npz", mean=mean, std=std, feature_names=np.array(FEATURE_NAMES))


def load_bundle(path_stem, device="cpu"):
    import joblib
    blob = joblib.load(path_stem + "_hist.joblib")
    scaler = np.load(path_stem + "_scaler.npz", allow_pickle=True)
    retnet = MultiTargetForecaster("retnet", len(FEATURE_NAMES))
    transformer = MultiTargetForecaster("transformer", len(FEATURE_NAMES))
    retnet.load_state_dict(torch.load(path_stem + "_retnet.pt", map_location=device, weights_only=True))
    transformer.load_state_dict(torch.load(path_stem + "_transformer.pt", map_location=device, weights_only=True))
    retnet.to(device).eval()
    transformer.to(device).eval()
    return {
        "hist": blob["hist"],
        "retnet": retnet,
        "transformer": transformer,
        "mean": scaler["mean"],
        "std": scaler["std"],
        "meta": blob["meta"],
    }


def train_forecasters(batch: ForecastBatch, epochs: int = 8, seed: int = 42, device=None) -> dict:
    """Fit all three models on a satellite-component-disjoint split and save them."""
    device = device or torch.device("cpu")
    tr, va, te = component_split(batch, seed=seed)
    mean, std = fit_scaler(batch.X[tr])
    Xtr = scale_sequences(batch.X[tr], mean, std)
    Xva = scale_sequences(batch.X[va], mean, std)
    Xte = scale_sequences(batch.X[te], mean, std)
    hist = HistGBForecaster(seed).fit(batch.summary[tr], batch.log_miss[tr], batch.log_pc[tr], batch.y[tr])
    retnet = train_one_network(
        "retnet", Xtr, (batch.log_miss[tr], batch.log_pc[tr], batch.y[tr]),
        Xva, (batch.log_miss[va], batch.log_pc[va], batch.y[va]), device, epochs, seed,
    )
    transformer = train_one_network(
        "transformer", Xtr, (batch.log_miss[tr], batch.log_pc[tr], batch.y[tr]),
        Xva, (batch.log_miss[va], batch.log_pc[va], batch.y[va]), device, epochs, seed + 1,
    )
    preds = {
        "HistGradientBoosting": hist.predict(batch.summary[te]),
        "RetNet": _predict_torch(retnet, Xte, device),
        "Transformer": _predict_torch(transformer, Xte, device),
    }
    metrics = {name: regression_report(pred, batch, te) for name, pred in preds.items()}
    # Overlap check: no satellite id shared between train and test.
    train_ids = set(batch.sat1[tr]).union(batch.sat2[tr])
    test_ids = set(batch.sat1[te]).union(batch.sat2[te])
    meta = {
        "seed": seed,
        "epochs": epochs,
        "n_train": int(len(tr)),
        "n_val": int(len(va)),
        "n_test": int(len(te)),
        "shared_satellites_train_test": int(len(train_ids & test_ids)),
        "pc_safe": PC_SAFE,
        "pc_critical": PC_CRITICAL,
        "target": "last CDM miss distance and catalog Pc; inputs are earlier CDMs only",
        "metrics": metrics,
    }
    stem = os.path.join(MODEL_DIR, "forecaster")
    save_bundle(stem, hist, retnet, transformer, mean, std, meta)
    return {"models": {"hist": hist, "retnet": retnet, "transformer": transformer},
            "mean": mean, "std": std, "meta": meta, "split": (tr, va, te), "preds": preds, "device": device}


def histgb_grouped_auc(batch: ForecastBatch, scheme: str, seed: int = 0) -> float:
    from sklearn.model_selection import StratifiedGroupKFold
    if scheme == "stratified":
        splits = StratifiedKFold(5, shuffle=True, random_state=seed).split(batch.summary, batch.y.astype(int))
    else:
        groups = {"group_sat1": batch.sat1, "group_sat2": batch.sat2, "group_component": batch.component}[scheme]
        splits = StratifiedGroupKFold(5, shuffle=True, random_state=seed).split(
            batch.summary, batch.y.astype(int), groups)
    scores = []
    y = batch.y.astype(int)
    for tr, te in splits:
        if len(np.unique(y[te])) < 2:
            continue
        clf = HistGradientBoostingClassifier(max_iter=150, learning_rate=0.08, random_state=seed)
        clf.fit(batch.summary[tr], y[tr])
        scores.append(roc_auc_score(y[te], clf.predict_proba(batch.summary[te])[:, 1]))
    return float(np.mean(scores)) if scores else float("nan")


def horizon_curve(batch: ForecastBatch, ks=(1, 2, 3, 4), seed: int = 0) -> dict:
    """AUC of a HistGB classifier that has seen only the first k prefix CDMs.

    The label is still the held-out final CDM. Events shorter than k are left
    out of that step, so the populations differ and the curve is not a paired test.
    """
    out = {}
    y = batch.y.astype(int)
    for k in ks:
        rows, labels, groups = [], [], []
        for i in range(len(batch.y)):
            feat = batch.prefixes[i]
            if len(feat) < k:
                continue
            rows.append(summarize_71(feat[:k]))
            labels.append(y[i])
            groups.append(batch.component[i])
        if len(rows) < 40 or len(set(labels)) < 2:
            out[str(k)] = {"auc": None, "n": len(rows)}
            continue
        F = np.stack(rows)
        labels = np.asarray(labels)
        groups = np.asarray(groups)
        from sklearn.model_selection import StratifiedGroupKFold
        aucs = []
        splitter = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=seed)
        for tr, te in splitter.split(F, labels, groups):
            if len(np.unique(labels[te])) < 2:
                continue
            clf = HistGradientBoostingClassifier(max_iter=120, learning_rate=0.08, random_state=seed)
            clf.fit(F[tr], labels[tr])
            aucs.append(roc_auc_score(labels[te], clf.predict_proba(F[te])[:, 1]))
        out[str(k)] = {"auc_mean": float(np.mean(aucs)) if aucs else None,
                       "auc_std": float(np.std(aucs)) if aucs else None,
                       "n": int(len(labels)), "folds": [float(a) for a in aucs]}
    return out


def median_latency_ms(fn, repeats: int = 30, warmup: int = 5) -> float:
    for _ in range(warmup):
        fn()
    samples = []
    for _ in range(repeats):
        t0 = time.perf_counter()
        fn()
        samples.append((time.perf_counter() - t0) * 1000.0)
    return float(np.median(samples))


def bundle_stem() -> str:
    return os.path.join(MODEL_DIR, "forecaster")
