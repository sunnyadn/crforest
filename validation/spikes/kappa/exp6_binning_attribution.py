"""κ.exp6 — test 'binning quantization causes Q1 early-death gap' hypothesis.

κ.exp5 found rfSRC outperforms comprisk on early-death pairs (Q1: t<66 days,
gap −0.035) but comprisk reverses at late deaths (Q4: t≥1419 days, gap +0.009).
Hypothesis: comprisk's default 256-bin histogram quantizes extreme feature
values present in early-death (high-acuity) patients; rfSRC's exact split
candidates retain sub-bin distinctions.

Test: refit comprisk with n_bins ∈ {256, 512, 1024} on seed=42, re-run the
quartile diagnostic. If hypothesis is right, Q1 gap should shrink monotonically
with more bins.

Run: uv run python -u validation/spikes/kappa/exp6_binning_attribution.py
"""

from __future__ import annotations

import time as _time
from pathlib import Path

import numpy as np
import pandas as pd

from comprisk import CompetingRiskForest, concordance_index_cr

CLEAN_PARQUET = Path("/tmp/chf_2012_clean.parquet")
TRAIN_IDX = Path("/tmp/chf_2012_train_idx.txt")
TEST_IDX = Path("/tmp/chf_2012_test_idx.txt")
RF_RISKS = Path("/tmp/chf_2012_rfsrc_risks_multiseed.parquet")

# comprisk caps n_bins at 256 (uint8 bin index); test reverse direction:
# at coarser binning (64, 128), if hypothesis is right the Q1 gap should
# WORSEN (gap becomes more negative).
N_BINS_LIST = [256, 128, 64]


def stratified_gap(
    e_te: np.ndarray,
    t_te: np.ndarray,
    risk_cr: np.ndarray,
    risk_rf: np.ndarray,
    n_strata: int = 4,
) -> pd.DataFrame:
    case_mask = e_te == 2
    t_cases = t_te[case_mask]
    qs = np.quantile(t_cases, np.linspace(0, 1, n_strata + 1))
    rows = []
    for k in range(n_strata):
        lo, hi = qs[k], qs[k + 1]
        if k < n_strata - 1:
            in_stratum = (t_te >= lo) & (t_te < hi) & (e_te == 2)
        else:
            in_stratum = (t_te >= lo) & (t_te <= hi) & (e_te == 2)
        e_strat = e_te.copy()
        e_strat[(e_te == 2) & ~in_stratum] = 0
        c_cr = concordance_index_cr(e_strat, t_te, risk_cr, cause=2)
        c_rf = concordance_index_cr(e_strat, t_te, risk_rf, cause=2)
        rows.append(
            {
                "stratum": k + 1,
                "t_lo": float(lo),
                "t_hi": float(hi),
                "n_cases": int(in_stratum.sum()),
                "c_cr": c_cr,
                "c_rf": c_rf,
                "gap": c_cr - c_rf,
            }
        )
    return pd.DataFrame(rows)


def main() -> None:
    df = pd.read_parquet(CLEAN_PARQUET)
    train_idx = np.loadtxt(TRAIN_IDX, dtype=np.int64)
    test_idx = np.loadtxt(TEST_IDX, dtype=np.int64)
    feature_cols = [c for c in df.columns if c not in ("time", "status")]
    X = df[feature_cols].to_numpy(dtype=np.float64)
    t = df["time"].to_numpy(dtype=np.float64)
    e = df["status"].to_numpy(dtype=np.int64)
    X_tr, t_tr, e_tr = X[train_idx], t[train_idx], e[train_idx]
    X_te, t_te, e_te = X[test_idx], t[test_idx], e[test_idx]

    # Load rfSRC seed=42 risks (the reference comparison)
    rf_long = pd.read_parquet(RF_RISKS)
    rf_seed42 = rf_long[rf_long["seed"] == 42].set_index("test_idx").reindex(test_idx)
    rf_risk2 = rf_seed42["risk2"].to_numpy(dtype=np.float64)

    print(
        f"[load] train n={len(train_idx):,}, test n={len(test_idx):,}, p={len(feature_cols)}",
        flush=True,
    )
    print("[ref]  rfSRC seed=42 risks loaded for comparison\n", flush=True)

    summary_rows = []
    for n_bins in N_BINS_LIST:
        print(f"=== n_bins={n_bins} ===", flush=True)
        f = CompetingRiskForest(n_estimators=100, n_jobs=-1, random_state=42, n_bins=n_bins)
        t0 = _time.perf_counter()
        f.fit(X_tr, t_tr, e_tr)
        wall = _time.perf_counter() - t0
        cr_risk2 = f.predict_risk(X_te, cause=2)
        print(f"  fit wall: {wall:.2f}s", flush=True)
        strat = stratified_gap(e_te, t_te, cr_risk2, rf_risk2)
        for _, r in strat.iterrows():
            print(
                f"    Q{int(r['stratum'])} [{r['t_lo']:.0f}-{r['t_hi']:.0f}, n={int(r['n_cases'])}]: "
                f"cr={r['c_cr']:.4f}  rf={r['c_rf']:.4f}  gap={r['gap']:+.4f}",
                flush=True,
            )
        for _, r in strat.iterrows():
            summary_rows.append(
                {
                    "n_bins": n_bins,
                    "stratum": int(r["stratum"]),
                    "gap": r["gap"],
                    "c_cr": r["c_cr"],
                    "c_rf": r["c_rf"],
                }
            )
        print(flush=True)

    print("=" * 80)
    print(" Summary: gap (cr-rf) per stratum, varying n_bins")
    print("=" * 80)
    pivot = pd.DataFrame(summary_rows).pivot(index="stratum", columns="n_bins", values="gap")
    print(pivot.to_string(float_format=lambda v: f"{v:+.4f}"))

    print("\nHypothesis check (reverse direction: does coarser binning worsen Q1 gap?):")
    q1_256 = pivot.loc[1, 256]
    q1_64 = pivot.loc[1, 64]
    if q1_64 < q1_256 - 0.005:
        print(f"  Q1 gap WORSENED from {q1_256:+.4f} (n_bins=256) to {q1_64:+.4f} (n_bins=64).")
        print("  → Binning quantization hypothesis SUPPORTED (coarser bins → bigger Q1 gap).")
    elif abs(q1_64 - q1_256) < 0.003:
        print(
            f"  Q1 gap essentially unchanged across {64}-{256} bins ({q1_64:+.4f} vs {q1_256:+.4f})."
        )
        print("  → Binning quantization hypothesis NOT supported in this range.")
    else:
        print(f"  Q1 gap moved from {q1_256:+.4f} (256) to {q1_64:+.4f} (64) — pattern unclear.")


if __name__ == "__main__":
    main()
