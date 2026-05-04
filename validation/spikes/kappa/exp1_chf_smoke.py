"""κ.exp1 — real CHF cohort end-to-end smoke fit.

Reads `~/Downloads/filtered_data_2012.csv` (94,098 patients × 64 cols, real
3-state competing risk: HF admission / death / censored), does minimal
preprocessing (drop ID + date, binary-encode 3 string columns, median impute),
then fits CompetingRiskForest(n_estimators=100) on Mac CPU via the default
flat-tree path.

Goal: get a wall-clock baseline on the *real* dataset before scoping Plan 3.
Replaces synthetic-n=100k as the headline benchmark per
`project_real_dataset_chf.md`.

Run: uv run python -u validation/spikes/kappa/exp1_chf_smoke.py
"""

from __future__ import annotations

import time as _time
from pathlib import Path

import numpy as np
import pandas as pd

from comprisk import CompetingRiskForest, concordance_index_cr
from comprisk.metrics import compute_uno_weights, concordance_index_uno_cr

CSV_PATH = Path.home() / "Downloads" / "filtered_data_2012.csv"

TIME_COL = "survival_time_with_competing_risk_death"
STATUS_COL = "outcome_cv_hf_with_competing_risk_death"
# Leakage-safe drops:
#   - euid: ID
#   - echo_date: raw date
#   - event_cv_hf_admission_icd10_post: outcome restated (Event+ count = 13,536
#     matches cause-1 exactly), trivial leakage
#   - follow_up_days: total observation window — equals survival_time for
#     censored + cause-2 (death) subjects, exceeds it for cause-1 (HF) only.
#     Encodes "did this person stay observed long enough to die or get
#     censored?" → time leakage especially for death C-index.
# (event_cv_hf_admission_icd10_PRIOR stays — integer 0/1 prior-history flag.)
DROP_COLS = ["euid", "echo_date", "event_cv_hf_admission_icd10_post", "follow_up_days"]
STATUS_MAP = {"Event -": 0, "Event +": 1, "Competing risk (death)": 2}
SEX_MAP = {"Female": 0, "Male": 1}


def load_chf() -> tuple[np.ndarray, np.ndarray, np.ndarray, list[str]]:
    """Return (X, time, event, feature_names) for the CHF cohort."""
    print(f"[load] reading {CSV_PATH}", flush=True)
    df = pd.read_csv(CSV_PATH, na_values=["NA"])
    print(f"[load] raw shape: {df.shape}", flush=True)

    time_arr = df[TIME_COL].to_numpy(dtype=np.float64)
    event_arr = df[STATUS_COL].map(STATUS_MAP).to_numpy(dtype=np.int64)
    if np.any(pd.isna(event_arr)):
        raise ValueError(f"unmapped status values: {df[STATUS_COL].unique()}")

    df = df.drop(columns=[*DROP_COLS, TIME_COL, STATUS_COL])
    df["demographics_birth_sex"] = df["demographics_birth_sex"].map(SEX_MAP)

    leftover_str = [
        c for c in df.columns if df[c].dtype.kind in ("O", "U") or str(df[c].dtype) == "string"
    ]
    if leftover_str:
        raise ValueError(f"unexpected string columns after encoding: {leftover_str}")

    medians = df.median(numeric_only=True)
    n_imputed = df.isna().sum().sum()
    df = df.fillna(medians)
    print(f"[load] median-imputed {n_imputed:,} cells across {df.shape[1]} cols", flush=True)

    X = df.to_numpy(dtype=np.float64)
    feature_names = list(df.columns)
    print(f"[load] X={X.shape}, p={len(feature_names)}", flush=True)
    print(
        f"[load] event distribution: censored={int((event_arr == 0).sum())}, "
        f"HF={int((event_arr == 1).sum())}, death={int((event_arr == 2).sum())}",
        flush=True,
    )
    print(f"[load] time range: [{time_arr.min():.0f}, {time_arr.max():.0f}] days", flush=True)
    return X, time_arr, event_arr, feature_names


def main() -> None:
    X, time_arr, event_arr, feature_names = load_chf()

    n, p = X.shape
    n_estimators = 100

    rng = np.random.default_rng(42)
    perm = rng.permutation(n)
    n_train = int(0.8 * n)
    train_idx, test_idx = perm[:n_train], perm[n_train:]
    X_tr, t_tr, e_tr = X[train_idx], time_arr[train_idx], event_arr[train_idx]
    X_te, t_te, e_te = X[test_idx], time_arr[test_idx], event_arr[test_idx]
    print(
        f"\n[split] 80/20 holdout: train n={len(train_idx):,}, test n={len(test_idx):,}",
        flush=True,
    )
    print(
        f"[split] test event distribution: censored={int((e_te == 0).sum())}, "
        f"HF={int((e_te == 1).sum())}, death={int((e_te == 2).sum())}",
        flush=True,
    )

    # Dump cleaned full table + indices for R-side rfSRC apples-to-apples bench.
    dump_dir = Path("/tmp")
    full_df = pd.DataFrame(X, columns=feature_names)
    full_df["time"] = time_arr
    full_df["status"] = event_arr
    full_df.to_parquet(dump_dir / "chf_2012_clean.parquet")
    np.savetxt(dump_dir / "chf_2012_train_idx.txt", train_idx, fmt="%d")
    np.savetxt(dump_dir / "chf_2012_test_idx.txt", test_idx, fmt="%d")
    print("[dump] wrote /tmp/chf_2012_clean.parquet + train/test idx", flush=True)

    print(
        f"\n[fit] CompetingRiskForest(n_estimators={n_estimators}, n_jobs=-1, device='cpu') "
        f"on n_train={n_train:,} p={p}",
        flush=True,
    )
    forest = CompetingRiskForest(
        n_estimators=n_estimators,
        n_jobs=-1,
        random_state=42,
    )

    t0 = _time.perf_counter()
    forest.fit(X_tr, t_tr, e_tr)
    fit_wall = _time.perf_counter() - t0
    print(
        f"[fit] wall: {fit_wall:.2f}s ({fit_wall / n_estimators * 1000:.1f}ms/tree avg)", flush=True
    )

    t0 = _time.perf_counter()
    risk_c1 = forest.predict_risk(X_te, cause=1)
    risk_c2 = forest.predict_risk(X_te, cause=2)
    pred_wall = _time.perf_counter() - t0
    print(f"[predict] holdout predict_risk x2 wall: {pred_wall:.2f}s", flush=True)

    c1_h = concordance_index_cr(e_te, t_te, risk_c1, cause=1)
    c2_h = concordance_index_cr(e_te, t_te, risk_c2, cause=2)

    # Uno IPCW weights computed on the test cohort (G = KM-of-censoring fit on
    # test (time, event)); same weight vector reused for both causes.
    uno_w = compute_uno_weights(t_te, e_te)
    c1_u = concordance_index_uno_cr(e_te, t_te, risk_c1, cause=1, weights=uno_w)
    c2_u = concordance_index_uno_cr(e_te, t_te, risk_c2, cause=2, weights=uno_w)

    print("\n=== summary ===", flush=True)
    print(
        f"  fit wall:           {fit_wall:.2f}s ({fit_wall / n_estimators * 1000:.1f}ms/tree)",
        flush=True,
    )
    print(f"  features used:      {p}", flush=True)
    print("  holdout C-index    Harrell's    Uno IPCW", flush=True)
    print(f"    cause-1 (HF):    {c1_h:.4f}      {c1_u:.4f}", flush=True)
    print(f"    cause-2 (death): {c2_h:.4f}      {c2_u:.4f}", flush=True)
    print(f"  effective device:   {forest._effective_device_}", flush=True)


if __name__ == "__main__":
    main()
