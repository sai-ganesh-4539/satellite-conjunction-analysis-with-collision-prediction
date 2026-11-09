# Project Review — Findings & Fixes (Sept 2026)

Full check of the DL project: every file inspected, pipeline executed end-to-end
on your real `data/cdms.csv` (2,708 CDMs). Results below.

**What works after fixes:** `data_processing.py` → `train.py --model retnet`
and `--model transformer` both run to completion. Smoke test (8 epochs, CPU):
RetNet Test AUC **0.823** / F1 **0.579**; Transformer Test AUC **0.842** / F1 **0.556**.

---

## CRITICAL — must fix before any training means anything

### C1. The zip contained the OLD, broken `data_processing.py`
It still references `OBJECT1_ID / OBJECT2_ID / COLLISION_PROBABILITY` — columns
that do not exist in your CSV. Running it crashes with
`KeyError: 'OBJECT1_ID'` (reproduced on your exact zip). Your CSV uses
`SAT_1_ID / SAT_2_ID / PC`. The fixed version that was promised in your chat
session never made it into this zip.
**Fixed:** rewritten for your real columns.

### C2. The planned task is impossible with this data — redefined
Space-Track only publishes CDMs for events **already above the 1e-4 screening
threshold**. In your 2,708 CDMs, **every single non-null PC value is ≥ 1e-4**
(min = 0.000100008). So the planned label `Pc > 1e-4` gives ~89% positive and
no real negatives (the remaining 11% are just missing Pc values, 295 rows).
There is no "high-risk vs low-risk" classification to learn here.
**Fixed:** task redefined to *"will this event peak at HIGH risk
(event max Pc ≥ 1e-3)?"* → 15.9% positive, 1:5.3 imbalance — a real, defensible
task consistent with the Limmer et al. (2022) collision-avoidance literature.
Update your report's problem statement accordingly.

### C3. Label leakage in the original design
Original: Pc was a per-timestep feature AND the label was `max(Pc) of the same
sequence` → the model can just read Pc off the last timestep. You would get
~100% accuracy that means nothing and would collapse under questioning.
**Fixed:** **Pc removed from features.** Features now: log(min range), event
position index, RCS (both objects), object types (one-hot), TCA hour/day
(cyclical). The model must predict peak risk from geometry/catalog proxies.

---

## HIGH — wrong results, not crashes

### H1. `StandardScaler` fit on ALL data before the split (leakage)
Test-set statistics influenced training normalization. Inflates metrics.
**Fixed:** scaler fit on train split only, applied to val/test.

### H2. Retention math did not match the RetNet paper (your main contribution!)
Three deviations in `MultiScaleRetention`:
1. **Decay was power-law** `lag^(-γ)` instead of the paper's **exponential**
   `γ^lag` — and it inverts the multi-scale semantics: in the old code a higher
   γ meant *faster* decay; in the paper γ→1 means *slower* decay / longer memory.
   Your README/report claim ("0.9 = fast, 0.995 = slow") was backwards in code.
2. **Element-wise `k*v` then `*q`** instead of the paper's scalar dot-product
   `(q_n · k_s) v_s` — not equivalent to retention at all.
3. Missing the `1/√d_head` scaling.
**Fixed:** parallel form `(QKᵀ/√d ⊙ D)V` with `D[i,j] = γ^(i-j)` for i≥j.
Be ready to explain this formula at your review — it is now the paper's Eq. (2).

### H3. `train.py` crashed immediately: `train(device=...)` vs `device_str`
The CLI passed `device=` but the function signature is `device_str=` →
`TypeError` on first run. You never reached this because C1 crashed first.
**Fixed.** (Also: no random seeding existed → non-reproducible runs. Fixed with
`set_seed(42)`, `--seed` flag added.)

---

## MEDIUM

| # | Issue | Fix / note |
|---|-------|-----------|
| M1 | SEQ_LEN=25 but real events have median 4, max 11 CDMs → **100% of samples were ≥56% zero-padding**; the "7 days × 4 readings/day" narrative does not match the data (all CDMs of an event share the exact same TCA) | SEQ_LEN=12, padding at start, latest CDM at last position |
| M2 | `hours_to_tca` computed vs wall-clock "now" → non-deterministic feature, changes every run, useless across train/test | Removed; replaced by TCA cyclical features + event position index |
| M3 | `CREATED` column in your CSV is corrupted (values like `46:41.0` — Excel mangled the timestamp). CDM issue-time is lost | Re-download with `format/json` if you need true issue times; meanwhile CDM_ID ordering used as proxy |
| M4 | Focal loss computed on sigmoid probs (`binary_cross_entropy`) — numerically unstable | Now logits-based (`binary_cross_entropy_with_logits`); default alpha 0.75 ≈ your imbalance (old 0.5 = no weighting) |
| M5 | `download_cdms.py`: missing `import os` → `NameError` on run; hardcoded `E:\DL\` paths | Fixed; added credential guard + .env instructions |
| M6 | `requirements.txt` missing `requests`, `python-dotenv` (needed by downloader) | Added |
| M7 | Random split could put the same object PAIR in train and test | Split now by event group/pair — no pair spans two splits |
| M8 | No training history saved → nothing to plot for the report | `checkpoints/history_{model}.json` now saved (loss/F1/AUC curves) |

## LOW (hygiene)

- **CRLF line endings** everywhere in the zip → normalized to LF (avoids git diff noise).
- `EMERGENCY_REPORTABLE` is constant `Y` → useless as feature (dropped).
- `SAT_1_EXCL_VOL` values {1, 5, 3} mirror `SAT1_OBJECT_TYPE` exactly — it is
  redundant; kept out of features.
- 295 CDMs have `PC = NaN` → dropped at load (they cannot produce labels).
- Files that are **empty placeholders**: `src/evaluate.py`,
  `src/feature_engineering.py`, `notebooks/train_colab.ipynb` (0 bytes).
- `README.md` still lists `parse_tle.py`, `build_dataset.py` (don't exist) and
  describes features (relative velocity vectors, altitude diff, RAAN diff) that
  **do not exist in your downloaded data**. Either implement them (re-download
  CDMs with full fields) or correct the README — right now it overpromises.
- No LSTM baseline exists despite the plan mentioning comparisons; the
  Transformer baseline is the only comparator (fine if professor approves).
- `check_cdm.py`, `main.py` had hardcoded `E:\DL` paths (won't run elsewhere).
- ⚠️ A previous chat mentioned an **exposed Space-Track password in the repo** —
  verify this is removed + rotate the password. `.env` must stay in `.gitignore`.

---

## Verified pipeline (after fixes)

```
python src/data_processing.py
  -> 735 events x 12 timesteps x 15 features, 15.9% positive
python src/train.py --model retnet       --epochs 100   # full run on Colab GPU
python src/train.py --model transformer  --epochs 100
# 8-epoch CPU smoke test: RetNet AUC 0.823 / F1 0.579; Transformer AUC 0.842 / F1 0.556
```

## Recommended next steps (priority order)

1. Re-download CDMs with **full fields** (`class/cdm`, not `cdm_public` summary)
   to unlock miss-distance RNT, relative velocity and covariance features —
   these are the features your report already promises.
2. Write `evaluate.py`: ROC curves, confusion matrix heatmap, training curves
   (from `history_*.json`), threshold sweep — you need these for Review 2.
3. Fill `feature_engineering.py` or delete it from the repo (empty files look bad).
4. Update README + report: 12 timesteps, new feature table, redefined task,
   honest data limitations (no sub-threshold CDMs publicly available).
5. Build the Colab notebook (GPU training + both model comparison table).
