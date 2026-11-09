"""
data_processing.py  (revised)

Changes vs. the previous version
  * build_events() returns arrays + metadata so CV scripts can group properly.
  * Satellite-level groups: one node per NORAD id regardless of whether it appears
    as SAT_1 or SAT_2 (the old graph used "S1_"/"S2_" prefixes, so a satellite seen
    in both roles was never connected, and cluster_ids was never used anyway).
  * Default train/val/test split is GROUP-aware (by SAT_1_ID) instead of a plain
    stratified split.
  * Saves meta.npz (seq_lengths, sat ids, component ids, SATCAT-missing flag).
  * Records which events had SATCAT orbital fields imputed (missingness can leak).
  * Padding: padded rows are all-zero AFTER scaling, which equals the feature mean.
    Models MUST mask them: mask = (x.abs().sum(-1) > 0).
"""

import os
import sys
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold, train_test_split

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO_ROOT, "src"))

try:
    from config import SEQ_LEN, PC_THRESHOLD, TRAIN_RATIO, VAL_RATIO, RANDOM_SEED
except ImportError:
    SEQ_LEN, PC_THRESHOLD = 14, 5e-4
    TRAIN_RATIO, VAL_RATIO, RANDOM_SEED = 0.7, 0.15, 42

DEFAULT_DATA_CSV = os.path.join(REPO_ROOT, "data", "cdms.csv")
DEFAULT_SATCAT_CSV = os.path.join(REPO_ROOT, "data", "satcat.csv")
DEFAULT_PROCESSED_DIR = os.path.join(REPO_ROOT, "data", "processed")

FEATURE_NAMES = [
    "log_min_rng", "delta_min_rng", "range_ratio", "norm_step_pos",
    "delta_apogee", "delta_perigee", "delta_inclination", "delta_period",
    "sat1_rcs", "sat2_rcs", "sat1_type", "sat2_type",
    "sat1_excl_vol", "sat2_excl_vol",
]


def load_and_merge_cdms(cdm_path, satcat_path):
    print(f"Loading CDMs from {cdm_path} ...")
    df = pd.read_csv(cdm_path)
    df.columns = [c.strip().upper() for c in df.columns]

    df["CDM_ID"] = pd.to_numeric(df["CDM_ID"], errors="coerce").fillna(0)
    df["MIN_RNG"] = pd.to_numeric(df["MIN_RNG"], errors="coerce")
    df["PC"] = pd.to_numeric(df["PC"], errors="coerce")
    df = df.dropna(subset=["MIN_RNG", "SAT_1_ID", "SAT_2_ID"]).reset_index(drop=True)

    rcs_map = {"SMALL": 0.0, "MEDIUM": 1.0, "LARGE": 2.0}
    type_map = {"DEBRIS": 0.0, "PAYLOAD": 1.0, "ROCKET BODY": 2.0}
    norm = lambda s: s.astype(str).str.strip().str.upper()
    df["SAT1_RCS_NUM"] = norm(df["SAT1_RCS"]).map(rcs_map).fillna(0.0)
    df["SAT2_RCS_NUM"] = norm(df["SAT2_RCS"]).map(rcs_map).fillna(0.0)
    df["SAT1_TYPE_NUM"] = norm(df["SAT1_OBJECT_TYPE"]).map(type_map).fillna(-1.0)
    df["SAT2_TYPE_NUM"] = norm(df["SAT2_OBJECT_TYPE"]).map(type_map).fillna(-1.0)
    df["SAT1_EXCL_VOL"] = pd.to_numeric(df["SAT_1_EXCL_VOL"], errors="coerce").fillna(1.0)
    df["SAT2_EXCL_VOL"] = pd.to_numeric(df["SAT_2_EXCL_VOL"], errors="coerce").fillna(1.0)

    if os.path.exists(satcat_path):
        print(f"Merging SATCAT from {satcat_path} ...")
        sc = pd.read_csv(satcat_path)
        sc.columns = [c.strip().upper() for c in sc.columns]
        sc["NORAD_CAT_ID"] = pd.to_numeric(sc["NORAD_CAT_ID"], errors="coerce")
        sc = sc.drop_duplicates(subset=["NORAD_CAT_ID"]).set_index("NORAD_CAT_ID")
        cols = ["APOGEE", "PERIGEE", "INCLINATION", "PERIOD", "RCSVALUE"]
        for c in cols:
            if c in sc.columns:
                sc[c] = pd.to_numeric(sc[c], errors="coerce")
        df = df.merge(sc[cols].add_prefix("S1_"), left_on="SAT_1_ID", right_index=True, how="left")
        df = df.merge(sc[cols].add_prefix("S2_"), left_on="SAT_2_ID", right_index=True, how="left")

        # Imputation can leak (e.g. fresh debris missing from SATCAT) -> keep a flag
        df["SATCAT_MISSING"] = df[["S1_APOGEE", "S1_PERIGEE", "S2_APOGEE", "S2_PERIGEE"]].isna().any(axis=1).astype(float)

        df["DELTA_APOGEE"] = np.log10(np.clip(np.abs(df["S1_APOGEE"] - df["S2_APOGEE"]).fillna(500.0), 1.0, None))
        df["DELTA_PERIGEE"] = np.log10(np.clip(np.abs(df["S1_PERIGEE"] - df["S2_PERIGEE"]).fillna(500.0), 1.0, None))
        df["DELTA_INCLINATION"] = np.abs(df["S1_INCLINATION"] - df["S2_INCLINATION"]).fillna(10.0)
        df["DELTA_PERIOD"] = np.clip(np.abs(df["S1_PERIOD"] - df["S2_PERIOD"]).fillna(5.0), 0.0, 50.0)
        print(f"  SATCAT fields imputed for {df['SATCAT_MISSING'].mean():.1%} of CDMs")
    else:
        print("  Notice: satcat.csv not found, using constant orbital proxies.")
        df["SATCAT_MISSING"] = 0.0
        df["DELTA_APOGEE"], df["DELTA_PERIGEE"] = 2.0, 2.0
        df["DELTA_INCLINATION"], df["DELTA_PERIOD"] = 10.0, 5.0

    df = df.sort_values(["SAT_1_ID", "SAT_2_ID", "CDM_ID"]).reset_index(drop=True)
    return df


def satellite_components(sat1, sat2):
    """Union-find over satellites (one node per NORAD id, role-independent)."""
    parent = {}

    def find(a):
        parent.setdefault(a, a)
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        return a

    for a, b in zip(sat1, sat2):
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[ra] = rb
    roots = {r: i for i, r in enumerate(sorted({find(x) for x in parent}))}
    return np.array([roots[find(a)] for a in sat1], dtype=np.int32)


def build_events(csv_path=DEFAULT_DATA_CSV, satcat_path=DEFAULT_SATCAT_CSV,
                 seq_len=SEQ_LEN, pc_threshold=PC_THRESHOLD):
    """Returns dict: X (N,T,F) raw/unscaled, y, seq_lengths, sat1, sat2,
    component, satcat_missing."""
    df = load_and_merge_cdms(csv_path, satcat_path)
    X, y, lens, s1s, s2s, miss = [], [], [], [], [], []

    for (s1, s2), g in df.groupby(["SAT_1_ID", "SAT_2_ID"]):
        max_pc = g["PC"].max()
        if pd.isna(max_pc):
            continue
        n = len(g)
        rng = g["MIN_RNG"].values
        log_rng = np.log10(np.clip(rng, 1.0, None))
        pos = np.arange(n) / max(1, n - 1)
        d_rng = np.clip(np.diff(np.concatenate([[rng[0]], rng])) / 100.0, -10.0, 10.0)
        ratio = rng / (rng[0] + 1.0)
        feat = np.stack([
            log_rng, d_rng, ratio, pos,
            g["DELTA_APOGEE"].values, g["DELTA_PERIGEE"].values,
            g["DELTA_INCLINATION"].values, g["DELTA_PERIOD"].values,
            g["SAT1_RCS_NUM"].values, g["SAT2_RCS_NUM"].values,
            g["SAT1_TYPE_NUM"].values, g["SAT2_TYPE_NUM"].values,
            g["SAT1_EXCL_VOL"].values, g["SAT2_EXCL_VOL"].values,
        ], axis=-1).astype(np.float32)
        if n >= seq_len:
            feat = feat[-seq_len:]
        else:
            feat = np.concatenate([np.zeros((seq_len - n, feat.shape[1]), np.float32), feat], axis=0)
        X.append(feat)
        y.append(1.0 if max_pc >= pc_threshold else 0.0)
        lens.append(n); s1s.append(s1); s2s.append(s2)
        miss.append(float(g["SATCAT_MISSING"].max()))

    ev = dict(
        X=np.array(X, np.float32), y=np.array(y, np.float32),
        seq_lengths=np.array(lens, np.int32),
        sat1=np.array(s1s), sat2=np.array(s2s),
        satcat_missing=np.array(miss, np.float32),
    )
    ev["component"] = satellite_components(ev["sat1"], ev["sat2"])
    sizes = np.bincount(ev["component"])
    print(f"Events: {len(ev['y'])}  | unique SAT_1: {len(set(s1s))}  unique SAT_2: {len(set(s2s))}")
    print(f"Satellite-connected components: {len(sizes)}  "
          f"(largest holds {sizes.max() / len(ev['y']):.1%} of events)")
    return ev


def grouped_split(y, groups, test_frac, seed):
    """One stratified, group-disjoint split ~test_frac (via StratifiedGroupKFold)."""
    k = max(2, int(round(1.0 / test_frac)))
    cv = StratifiedGroupKFold(n_splits=k, shuffle=True, random_state=seed)
    train, test = next(cv.split(np.zeros(len(y)), y, groups))
    return train, test


def build_dataset(csv_path=DEFAULT_DATA_CSV, satcat_path=DEFAULT_SATCAT_CSV,
                  output_dir=DEFAULT_PROCESSED_DIR, seq_len=SEQ_LEN,
                  pc_threshold=PC_THRESHOLD, train_ratio=TRAIN_RATIO,
                  val_ratio=VAL_RATIO, random_seed=RANDOM_SEED, group_split=True):
    import torch  # only needed for saving tensors
    ev = build_events(csv_path, satcat_path, seq_len, pc_threshold)
    X, y = ev["X"], ev["y"]
    n_pos = int(y.sum())
    print(f"  Sequence shape: {X.shape}  High-risk: {n_pos} ({n_pos / len(y):.1%})")

    idx = np.arange(len(X))
    test_frac = 1.0 - train_ratio - val_ratio
    if group_split:
        groups = ev["sat1"]       # no SAT_1 object shared across splits
        rest, test_idx = grouped_split(y, groups, test_frac, random_seed)
        val_frac_of_rest = val_ratio / (train_ratio + val_ratio)
        tr, va = grouped_split(y[rest], groups[rest], val_frac_of_rest, random_seed)
        train_idx, val_idx = rest[tr], rest[va]
        shared = len(set(ev["sat2"][train_idx]) & set(ev["sat2"][test_idx]))
        print(f"  Group split by SAT_1_ID. SAT_2 objects shared train/test: {shared} "
              f"(use grouped_cv_check.py to see how much this matters)")
    else:
        train_idx, tmp = train_test_split(idx, train_size=train_ratio, random_state=random_seed, stratify=y)
        val_idx, test_idx = train_test_split(tmp, train_size=val_ratio / (1 - train_ratio),
                                             random_state=random_seed, stratify=y[tmp])

    X_train, X_val, X_test = X[train_idx], X[val_idx], X[test_idx]
    mask = (X_train != 0.0).any(axis=-1)
    mean = np.zeros(X.shape[-1], np.float32)
    std = np.ones(X.shape[-1], np.float32)
    for d in range(X.shape[-1]):
        v = X_train[:, :, d][mask]
        if len(v) and v.std() > 1e-6:
            mean[d], std[d] = v.mean(), v.std()

    def scale(t):
        m = (t != 0.0).any(axis=-1, keepdims=True)
        return np.where(m, (t - mean) / (std + 1e-8), 0.0).astype(np.float32)

    os.makedirs(output_dir, exist_ok=True)
    for name, ix, Xs in [("train", train_idx, scale(X_train)), ("val", val_idx, scale(X_val)),
                         ("test", test_idx, scale(X_test))]:
        torch.save((torch.FloatTensor(Xs), torch.FloatTensor(y[ix])), os.path.join(output_dir, f"{name}.pt"))
        print(f"  {name:5s}: {len(ix)} events ({int(y[ix].sum())} pos)")
    torch.save({"mean": mean, "std": std, "feature_names": FEATURE_NAMES},
               os.path.join(output_dir, "scaler_stats.pt"))
    np.savez(os.path.join(output_dir, "meta.npz"),
             seq_lengths=ev["seq_lengths"], sat1=ev["sat1"], sat2=ev["sat2"],
             component=ev["component"], satcat_missing=ev["satcat_missing"],
             train_idx=train_idx, val_idx=val_idx, test_idx=test_idx)
    print(f"Saved datasets + meta.npz to {output_dir}/")


if __name__ == "__main__":
    a = sys.argv
    build_dataset(a[1] if len(a) > 1 else DEFAULT_DATA_CSV,
                  a[2] if len(a) > 2 else DEFAULT_SATCAT_CSV,
                  a[3] if len(a) > 3 else DEFAULT_PROCESSED_DIR)
