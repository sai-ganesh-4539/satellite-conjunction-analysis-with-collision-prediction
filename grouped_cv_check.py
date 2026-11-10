"""
grouped_cv_check.py -- does the tabular result survive proper grouping?

Runs LR / RandomForest / HistGradientBoosting under:
  stratified      plain StratifiedKFold   (what your benchmark effectively used:
                                           every event is its own pair, so
                                           "grouped by pair" == no grouping)
  group_sat1      no SAT_1 object shared between train and test
  group_sat2      no SAT_2 object shared between train and test
  group_component no satellite shared at all (skipped if one component dominates)
and with / without event-length information, plus single-feature leak probes.

Usage:  python grouped_cv_check.py [cdms.csv] [satcat.csv] [--seeds 3]
"""
import os
import sys
import json
import argparse
os.environ["LOKY_MAX_CPU_COUNT"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"

REPO_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(REPO_ROOT, "src"))

import numpy as np
from sklearn.model_selection import StratifiedKFold, StratifiedGroupKFold
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score, average_precision_score
from data_processing import build_events, DEFAULT_DATA_CSV, DEFAULT_SATCAT_CSV

STEP_POS = 3      # index of norm_step_pos (encodes sequence length)


def summarize(X, n, drop_length):
    T = X.shape[1]
    real = (X != 0).any(-1)
    Xn = np.where(real[..., None], X, np.nan)
    n_eff = np.minimum(n, T)
    first = X[np.arange(len(X)), T - n_eff]
    blocks = [X[:, -1], np.nanmean(Xn, 1), np.nanmin(Xn, 1), np.nanmax(Xn, 1),
              (X[:, -1, 0] - first[:, 0])[:, None]]          # range trend
    if drop_length:
        blocks[:4] = [np.delete(b, STEP_POS, axis=1) for b in blocks[:4]]
    else:
        blocks.append(n_eff[:, None].astype(float))
    return np.nan_to_num(np.concatenate(blocks, axis=1))


def models():
    return {
        "LogReg": lambda s: make_pipeline(StandardScaler(), LogisticRegression(max_iter=3000)),
        "RandomForest": lambda s: RandomForestClassifier(200, random_state=s, n_jobs=1),
        "HistGB": lambda s: HistGradientBoostingClassifier(random_state=s),
    }


def splitter(scheme, y, ev, seed):
    X0 = np.zeros(len(y))
    if scheme == "stratified":
        return StratifiedKFold(5, shuffle=True, random_state=seed).split(X0, y)
    g = {"group_sat1": ev["sat1"], "group_sat2": ev["sat2"], "group_component": ev["component"]}[scheme]
    return StratifiedGroupKFold(5, shuffle=True, random_state=seed).split(X0, y, g)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cdm", nargs="?", default=DEFAULT_DATA_CSV)
    ap.add_argument("satcat", nargs="?", default=DEFAULT_SATCAT_CSV)
    ap.add_argument("--seeds", type=int, default=3)
    a = ap.parse_args()

    ev = build_events(a.cdm, a.satcat)
    y, n = ev["y"].astype(int), ev["seq_lengths"]

    print("\nSingle-feature probes (AUC; ~0.5 = no signal):")
    for name, v in [("seq_length", n), ("satcat_imputed_flag", ev["satcat_missing"])]:
        if np.ptp(v) > 0:
            auc = roc_auc_score(y, v)
            print(f"  {name:22s} {max(auc, 1 - auc):.3f}")

    big = np.bincount(ev["component"]).max() / len(y)
    schemes = ["stratified", "group_sat1", "group_sat2"]
    if big < 0.5:
        schemes.append("group_component")
    else:
        print(f"\n(group_component skipped: one component holds {big:.0%} of events)")

    audit_payload = {
        "metadata": {
            "dataset_cdm": str(a.cdm),
            "dataset_satcat": str(a.satcat),
            "n_events": int(len(y)),
            "n_unique_sat1": int(len(set(ev["sat1"]))),
            "n_unique_sat2": int(len(set(ev["sat2"]))),
            "n_components": int(len(set(ev["component"]))),
            "largest_component_fraction": float(big),
            "n_seeds": int(a.seeds),
            "seeds": list(range(a.seeds)),
            "k_folds": 5,
            "total_folds_per_scheme": int(a.seeds * 5),
            "script": "grouped_cv_check.py",
        },
        "with_length_features": {
            "feature_description": "Last observation, mean, min, max across 14 channels + range trend + sequence length count",
            "schemes": {}
        },
        "without_length_features": {
            "feature_description": "Last observation, mean, min, max across 14 channels (excluding norm_step_pos and length count) + range trend",
            "schemes": {}
        }
    }

    for drop_length in (False, True):
        F = summarize(ev["X"], n, drop_length)
        block_key = "without_length_features" if drop_length else "with_length_features"
        audit_payload[block_key]["n_features"] = int(F.shape[1])
        audit_payload[block_key]["schemes"] = {}
        print(f"\n=== {'WITHOUT' if drop_length else 'WITH'} event-length features "
              f"({F.shape[1]} features) ===")
        print(f"{'scheme':16s} {'model':13s} {'AUC':>15s} {'PR-AUC':>15s}")
        for scheme in schemes:
            audit_payload[block_key]["schemes"][scheme] = {}
            for mname, make in models().items():
                aucs, aps = [], []
                for seed in range(a.seeds):
                    for tr, te in splitter(scheme, y, ev, seed):
                        if len(set(y[te])) < 2:
                            continue
                        m = make(seed).fit(F[tr], y[tr])
                        p = m.predict_proba(F[te])[:, 1]
                        aucs.append(float(roc_auc_score(y[te], p)))
                        aps.append(float(average_precision_score(y[te], p)))
                mean_auc, std_auc = float(np.mean(aucs)), float(np.std(aucs))
                mean_ap, std_ap = float(np.mean(aps)), float(np.std(aps))
                audit_payload[block_key]["schemes"][scheme][mname] = {
                    "auc_mean": mean_auc,
                    "auc_std": std_auc,
                    "pr_auc_mean": mean_ap,
                    "pr_auc_std": std_ap,
                    "fold_aucs": aucs,
                    "fold_pr_aucs": aps,
                }
                print(f"{scheme:16s} {mname:13s} {mean_auc:.3f} +/- {std_auc:.3f}  "
                      f"{mean_ap:.3f} +/- {std_ap:.3f}")

    out_json = os.path.join(REPO_ROOT, "results", "grouped_cv_leakage_audit.json")
    os.makedirs(os.path.dirname(out_json), exist_ok=True)
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(audit_payload, f, indent=2)
    print(f"\nWrote fold-level grouped CV audit artifact to: {out_json}")

    print("\nRead it like this: if AUC drops a lot from 'stratified' to 'group_*', the 0.91 was "
          "partly object identity. If it drops a lot WITHOUT length features, the model is "
          "partly reading event length (more CDMs -> higher max Pc), not geometry.")


if __name__ == "__main__":
    main()
