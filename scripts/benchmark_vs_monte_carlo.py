"""
Head-to-head timing of the B-plane Monte Carlo engine and the three surrogates.

The CDM extract has no covariance. For each encounter the isotropic scale is
chosen so the exact integral equals the catalog Pc (inflating the hard-body
radius only when the catalog exclusion radii cannot reach that Pc). The Monte
Carlo column is therefore a sampling estimate of the catalog probability, not
an independent orbit determination. Latency is still a direct wall-clock
comparison on this machine.

Usage:
  python scripts/benchmark_vs_monte_carlo.py --n-events 100 --mc-samples 1000000 --epochs 8
"""

from __future__ import annotations

import argparse
import json
import os
os.environ["LOKY_MAX_CPU_COUNT"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"
import sys
import time

import numpy as np
import torch
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import StratifiedGroupKFold, StratifiedKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from scipy.stats import ttest_rel

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO_ROOT, "src"))

from collision_forecaster import (  # noqa: E402
    bundle_stem, component_split, histgb_grouped_auc, horizon_curve, load_bundle,
    load_forecast_events, median_latency_ms, regression_report, scale_sequences,
    train_forecasters, _predict_torch,
)
from monte_carlo_engine import (  # noqa: E402
    bplane_miss, invert_isotropic_scale, isotropic_covariance, isotropic_pc, monte_carlo_pc,
)
from operational_metrics import (  # noqa: E402
    apply_temperature, expected_calibration_error, fit_temperature, reliability_diagram,
    screening_flag_rate, threshold_for_recall,
)
from sklearn.metrics import precision_score, recall_score, brier_score_loss, roc_auc_score


def _spearman(a, b) -> float:
    import pandas as pd
    ra = pd.Series(np.asarray(a, float)).rank().to_numpy()
    rb = pd.Series(np.asarray(b, float)).rank().to_numpy()
    if ra.std() == 0 or rb.std() == 0:
        return float("nan")
    return float(np.corrcoef(ra, rb)[0, 1])


def _metrics(y_true_pc, y_hat_pc) -> dict:
    yt = np.log10(np.clip(y_true_pc, 1e-12, 1))
    yh = np.log10(np.clip(y_hat_pc, 1e-12, 1))
    return {
        "r2_log10_pc": float(r2_score(yt, yh)),
        "mae_log10_pc": float(mean_absolute_error(yt, yh)),
        "mae_pc": float(mean_absolute_error(y_true_pc, y_hat_pc)),
        "spearman_log10_pc": _spearman(yt, yh),
    }


def grouped_audit(batch, seed: int = 0) -> dict:
    y = batch.y.astype(int)
    schemes = {
        "stratified": None,
        "group_sat1": batch.sat1,
        "group_sat2": batch.sat2,
        "group_component": batch.component,
    }
    makers = {
        "LogReg": lambda s: make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000)),
        "RandomForest": lambda s: RandomForestClassifier(120, random_state=s, n_jobs=1),
        "HistGradientBoosting": lambda s: HistGradientBoostingClassifier(
            max_iter=120, learning_rate=0.08, random_state=s),
    }
    report = {}
    X = batch.summary
    for scheme, groups in schemes.items():
        report[scheme] = {}
        for name, make in makers.items():
            aucs = []
            if scheme == "stratified":
                splits = StratifiedKFold(5, shuffle=True, random_state=seed).split(X, y)
            else:
                splits = StratifiedGroupKFold(5, shuffle=True, random_state=seed).split(X, y, groups)
            for tr, te in splits:
                if len(np.unique(y[te])) < 2:
                    continue
                model = make(seed).fit(X[tr], y[tr])
                aucs.append(float(roc_auc_score(y[te], model.predict_proba(X[te])[:, 1])))
            report[scheme][name] = {
                "auc_mean": float(np.mean(aucs)) if aucs else None,
                "auc_std": float(np.std(aucs)) if aucs else None,
                "folds": aucs,
            }
    return report


def classification_operating_point(y, prob) -> dict:
    """Threshold chosen on this same set. This is descriptive, not a nested test."""
    thr = threshold_for_recall(y, prob, 0.90)
    pred = (prob >= thr).astype(int)
    prev = float(np.mean(y))
    rec = float(recall_score(y, pred, zero_division=0))
    prec = float(precision_score(y, pred, zero_division=0))
    return {
        "threshold": thr,
        "recall": rec,
        "precision": prec,
        "flag_rate": screening_flag_rate(prev, rec, prec),
        "prevalence": prev,
        "brier": float(brier_score_loss(y, np.clip(prob, 0, 1))),
        "ece": expected_calibration_error(y, prob),
        "note": "Threshold was fit on the same scores it is scored on. Nested test numbers live in results/benchmark_unified_cv.json.",
    }


def run(n_events: int, mc_samples: int, epochs: int, seed: int, retrain: bool) -> dict:
    t_all = time.perf_counter()
    batch = load_forecast_events()
    stem = bundle_stem()
    if retrain or not os.path.exists(stem + "_hist.joblib"):
        print(f"Training forecasters for up to {epochs} epochs...")
        trained = train_forecasters(batch, epochs=epochs, seed=seed)
        heldout_metrics = trained["meta"]["metrics"]
        split = trained["split"]
    else:
        print("Reusing saved forecasters.")
        heldout_metrics = None
        split = component_split(batch, seed=seed)
    device = torch.device("cpu")
    bundle = load_bundle(stem, device=device)
    if heldout_metrics is None:
        heldout_metrics = bundle["meta"].get("metrics")

    _tr, _va, te = split
    rng = np.random.default_rng(seed)
    pool = te if len(te) >= min(20, n_events) else np.arange(len(batch.y))
    take = min(n_events, len(pool))
    chosen = np.sort(rng.choice(pool, size=take, replace=False))

    rows = []
    mc_times = []
    n_inflated = 0
    for n_done, i in enumerate(chosen, start=1):
        miss = float(batch.miss_m[i])
        pc = float(batch.pc[i])
        last = batch.prefixes[i][-1]
        radius = float(last[12] + last[13])
        inv = invert_isotropic_scale(miss, radius, pc)
        if inv["mode"] == "radius_inflated_to_match_catalog_pc":
            n_inflated += 1
        sigma = float(inv["sigma_m"]) if inv.get("sigma_m") is not None else max(1000.0, miss * 2.0)
        used_radius = float(inv["radius_m"])
        mc = monte_carlo_pc(
            bplane_miss(miss), isotropic_covariance(sigma), used_radius,
            n_samples=mc_samples, seed=int(seed + i),
        )
        mc_times.append(mc["elapsed_s"])
        analytical = isotropic_pc(miss, sigma, used_radius)
        screening = isotropic_pc(miss, 1000.0, radius)
        xs = scale_sequences(batch.X[i:i + 1], bundle["mean"], bundle["std"])
        pred_r = _predict_torch(bundle["retnet"], xs, device)
        pred_t = _predict_torch(bundle["transformer"], xs, device)
        pred_h = bundle["hist"].predict(batch.summary[i:i + 1])
        rows.append({
            "event_id": str(batch.event_id[i]),
            "catalog_pc": pc,
            "catalog_miss_m": miss,
            "catalog_radius_m": radius,
            "matched_radius_m": used_radius,
            "matched_sigma_m": sigma,
            "match_mode": inv["mode"],
            "monte_carlo_pc": mc["pc"],
            "monte_carlo_wilson_95": mc["wilson_95"],
            "analytical_matched_pc": analytical,
            "screening_1km_pc": screening,
            "mc_elapsed_ms": mc["elapsed_s"] * 1000.0,
            "reference_inside_wilson": bool(mc["wilson_95"][0] <= analytical <= mc["wilson_95"][1]),
            "retnet_pc": float(pred_r["pc"][0]),
            "transformer_pc": float(pred_t["pc"][0]),
            "histgb_pc": float(pred_h["pc"][0]),
            "retnet_miss_m": float(pred_r["miss_m"][0]),
            "transformer_miss_m": float(pred_t["miss_m"][0]),
            "histgb_miss_m": float(pred_h["miss_m"][0]),
        })
        if n_done % 10 == 0 or n_done == take:
            print(f"  Monte Carlo {n_done}/{take}  last {mc['elapsed_s']*1000:.1f} ms  "
                  f"hits {mc['hits']}/{mc_samples}")

    catalog = np.array([r["catalog_pc"] for r in rows])
    mc_pc = np.array([r["monte_carlo_pc"] for r in rows])
    agreement = {
        "monte_carlo_vs_catalog": _metrics(catalog, mc_pc),
        "RetNet_vs_monte_carlo": _metrics(mc_pc, np.array([r["retnet_pc"] for r in rows])),
        "Transformer_vs_monte_carlo": _metrics(mc_pc, np.array([r["transformer_pc"] for r in rows])),
        "HistGradientBoosting_vs_monte_carlo": _metrics(mc_pc, np.array([r["histgb_pc"] for r in rows])),
        "fraction_analytical_inside_wilson_95": float(np.mean([r["reference_inside_wilson"] for r in rows])),
        "fraction_radius_inflated": float(n_inflated / max(1, len(rows))),
    }

    # Single-encounter latency, synchronized with canonical benchmark.
    canonical_lat_file = os.path.join(REPO_ROOT, "results", "empirical_latency_benchmark.json")
    if os.path.exists(canonical_lat_file):
        with open(canonical_lat_file, "r", encoding="utf-8") as f:
            can_lat = json.load(f)
        latency = {
            "RetNet_ms": can_lat["RetNet Recurrent O(1) step"]["median"],
            "RetNet_parallel_ms": can_lat["RetNet Parallel (seq L=14)"]["median"],
            "Transformer_ms": can_lat["Transformer Baseline"]["median"],
            "HistGradientBoosting_ms": can_lat["HistGradientBoosting"]["median"],
            "LogisticRegression_ms": can_lat["Logistic Regression"]["median"],
            "RandomForest_ms": can_lat["Random Forest"]["median"],
            "MonteCarlo_ms_median": can_lat["2D B-Plane Monte Carlo (10^6 samples)"]["median"],
            "MonteCarlo_ms_mean": can_lat["2D B-Plane Monte Carlo (10^6 samples)"]["mean"],
            "mc_samples": mc_samples,
            "RetNet_speedup_vs_mc": can_lat["speedups_vs_2d_monte_carlo"]["RetNet_Recurrent"],
            "Transformer_speedup_vs_mc": can_lat["speedups_vs_2d_monte_carlo"]["Transformer"],
            "HistGradientBoosting_speedup_vs_mc": can_lat["speedups_vs_2d_monte_carlo"]["HistGradientBoosting"],
            "canonical_source": "results/empirical_latency_benchmark.json (10 warmup, 100 repeats, single-core CPU, batch size 1)",
        }
    else:
        xs = scale_sequences(batch.X[chosen[:1]], bundle["mean"], bundle["std"])
        xt = torch.from_numpy(xs)
        latency = {
            "RetNet_ms": median_latency_ms(lambda: bundle["retnet"](xt)),
            "Transformer_ms": median_latency_ms(lambda: bundle["transformer"](xt)),
            "HistGradientBoosting_ms": median_latency_ms(lambda: bundle["hist"].predict(batch.summary[chosen[:1]])),
            "MonteCarlo_ms_median": float(np.median(mc_times) * 1000.0),
            "MonteCarlo_ms_mean": float(np.mean(mc_times) * 1000.0),
            "mc_samples": mc_samples,
        }
        for name in ("RetNet_ms", "Transformer_ms", "HistGradientBoosting_ms"):
            latency[name.replace("_ms", "_speedup_vs_mc")] = latency["MonteCarlo_ms_median"] / latency[name]

    print("Component-grouped tabular audit...")
    audit = grouped_audit(batch, seed=seed)
    audit_file = os.path.join(REPO_ROOT, "results", "grouped_cv_leakage_audit.json")
    full_audit_payload = None
    if os.path.exists(audit_file):
        try:
            with open(audit_file, "r", encoding="utf-8") as f:
                full_audit_payload = json.load(f)
        except Exception:
            pass

    if full_audit_payload and "with_length_features" in full_audit_payload:
        comp_schemes = full_audit_payload["with_length_features"]["schemes"]["group_component"]
        h = comp_schemes["HistGB"]["fold_aucs"]
        lin = comp_schemes["LogReg"]["fold_aucs"]
        grouped_cv_export = {
            scheme: {
                m: {
                    "auc_mean": dat["auc_mean"],
                    "auc_std": dat["auc_std"],
                    "folds": dat["fold_aucs"]
                }
                for m, dat in mdict.items()
            }
            for scheme, mdict in full_audit_payload["with_length_features"]["schemes"].items()
        }
    else:
        h = audit["group_component"]["HistGradientBoosting"]["folds"]
        lin = audit["group_component"]["LogReg"]["folds"]
        grouped_cv_export = audit

    paired = {}
    if len(h) == len(lin) and len(h) >= 2:
        t_stat, p_value = ttest_rel(h, lin)
        paired = {"histgb_vs_logreg_component_folds": {"t_stat": float(t_stat), "p_value": float(p_value),
                                                       "n_folds": len(h)}}

    print("Streaming-horizon curve...")
    horizons = horizon_curve(batch, seed=seed)

    # Held-out classification scores for calibration figures.
    te_x = scale_sequences(batch.X[te], bundle["mean"], bundle["std"])
    heldout_preds = {
        "RetNet": _predict_torch(bundle["retnet"], te_x, device),
        "Transformer": _predict_torch(bundle["transformer"], te_x, device),
        "HistGradientBoosting": bundle["hist"].predict(batch.summary[te]),
    }
    y_te = batch.y[te]
    # Temperature on a slice of the test set would leak. Use the validation split.
    _tr, va, _te = split
    va_x = scale_sequences(batch.X[va], bundle["mean"], bundle["std"])
    calibration = {}
    reliability = {}
    operating = {}
    for name, pred_te in heldout_preds.items():
        if name == "HistGradientBoosting":
            pred_va = bundle["hist"].predict(batch.summary[va])
        elif name == "RetNet":
            pred_va = _predict_torch(bundle["retnet"], va_x, device)
        else:
            pred_va = _predict_torch(bundle["transformer"], va_x, device)
        temperature = fit_temperature(pred_va["logits"], batch.y[va])
        prob = apply_temperature(pred_te["logits"], temperature)
        calibration[name] = {
            "temperature": temperature,
            "ece": expected_calibration_error(y_te, prob),
            "brier": float(brier_score_loss(y_te, np.clip(prob, 0, 1))),
            "auc": float(roc_auc_score(y_te, prob)) if len(np.unique(y_te)) > 1 else None,
        }
        reliability[name] = reliability_diagram(y_te, prob)
        operating[name] = classification_operating_point(y_te, prob)
        # Keep the scores so figures can draw one ROC without refitting.
        calibration[name]["y"] = [float(v) for v in y_te]
        calibration[name]["prob"] = [float(v) for v in prob]

    # Held-out regression already computed at training time when available.
    if heldout_metrics is None:
        heldout_metrics = {name: regression_report(pred, batch, te) for name, pred in heldout_preds.items()}

    payload = {
        "caveat": (
            "This CDM extract has no B-plane covariance. matched_sigma_m is solved so the "
            "isotropic integral equals catalog Pc. Monte Carlo agreement with catalog Pc "
            "checks the sampler. Surrogate agreement with Monte Carlo is agreement with that "
            "catalog label, up to sampling noise. screening_1km_pc uses the catalog exclusion "
            "radii and a 1 km default sigma and is not fit to catalog Pc."
        ),
        "n_forecast_events": int(len(batch.y)),
        "n_positive": int(batch.y.sum()),
        "n_benchmark_events": int(len(rows)),
        "mc_samples": mc_samples,
        "epochs": epochs,
        "heldout_regression": heldout_metrics,
        "agreement": agreement,
        "latency_ms": latency,
        "grouped_cv": grouped_cv_export,
        "grouped_cv_leakage_audit_metadata": full_audit_payload.get("metadata") if full_audit_payload else None,
        "prefix_forecast_grouped_cv_1seed_641events": audit,
        "paired_tests": paired,
        "horizon": horizons,
        "calibration": {k: {kk: vv for kk, vv in d.items() if kk not in ("y", "prob")}
                        for k, d in calibration.items()},
        "reliability": reliability,
        "operating_point_same_set": operating,
        "scores": {k: {"y": calibration[k]["y"], "prob": calibration[k]["prob"]} for k in calibration},
        "encounters": rows,
        "wall_clock_s": time.perf_counter() - t_all,
        "machine_note": "CPU timings from this process. CUDA was not required.",
    }
    out = os.path.join(REPO_ROOT, "results", "monte_carlo_vs_surrogate_benchmark.json")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    with open(out, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
    print(f"Wrote {out}")
    print(json.dumps({"latency_ms": latency, "agreement": agreement}, indent=2))
    return payload


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-events", type=int, default=100)
    ap.add_argument("--mc-samples", type=int, default=1_000_000)
    ap.add_argument("--epochs", type=int, default=8)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--retrain", action="store_true")
    run(ap.parse_args().n_events, ap.parse_args().mc_samples, ap.parse_args().epochs,
        ap.parse_args().seed, ap.parse_args().retrain)


if __name__ == "__main__":
    main()
