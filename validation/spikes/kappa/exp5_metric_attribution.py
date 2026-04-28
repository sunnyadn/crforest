"""κ.exp5 — distinguish 'Harrell artifact' vs 'Uno masks real difference'.

Stratify death (cause-2) C-index by death time. Two competing hypotheses
explain the observed 0.015 Harrell gap + 0.001 Uno gap:

  H1: models truly equivalent; Harrell's gap is a censoring-related artifact
      (pair-count accumulation over heavy-censoring data).
      Prediction: gap is small at every time stratum.

  H2: models actually disagree on a subset of pairs; Uno IPCW's 1/G(t)^2
      weighting downweights that subset (probably late-time deaths where
      G(t) is small) and so the gap collapses.
      Prediction: gap is concentrated at one time stratum (likely late).

Diagnostic: split the test cohort by death time quartile and compute
cause-2 Harrell C-index on each stratum, for each method. The gap pattern
across strata distinguishes H1 from H2.

Run: uv run python -u validation/spikes/kappa/exp5_metric_attribution.py
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from crforest import concordance_index_cr
from crforest.metrics import compute_uno_weights, concordance_index_uno_cr

CLEAN_PARQUET = Path("/tmp/chf_2012_clean.parquet")
TEST_IDX = Path("/tmp/chf_2012_test_idx.txt")
CR_RISKS = Path("/tmp/chf_2012_crforest_risks_multiseed.parquet")
RF_RISKS = Path("/tmp/chf_2012_rfsrc_risks_multiseed.parquet")
SEEDS = [42, 43, 44, 45, 46]


def stratify_cindex(
    e_te: np.ndarray,
    t_te: np.ndarray,
    risk_cr: np.ndarray,
    risk_rf: np.ndarray,
    cause: int,
    n_strata: int = 4,
) -> pd.DataFrame:
    """Cause-specific Harrell's C-index per death-time stratum.

    Pairs (i, j) are bucketed by t_i quartile (i is the case with event=cause).
    For each bucket, computes Harrell's C separately for crforest and rfSRC
    risk vectors. Allows seeing where the two methods agree/disagree.
    """
    case_mask = e_te == cause
    t_cases = t_te[case_mask]
    if len(t_cases) == 0:
        return pd.DataFrame()

    # Quartile boundaries on case event times.
    qs = np.quantile(t_cases, np.linspace(0, 1, n_strata + 1))
    rows = []
    for k in range(n_strata):
        lo, hi = qs[k], qs[k + 1]
        # Cases in this stratum: include lower bound, include upper bound on last bucket.
        if k < n_strata - 1:
            in_stratum = (t_te >= lo) & (t_te < hi) & (e_te == cause)
        else:
            in_stratum = (t_te >= lo) & (t_te <= hi) & (e_te == cause)
        # For the C-index of THIS stratum: keep all controls (j) but only
        # cases (i) from this stratum. We do this by zeroing out cases outside
        # the stratum from the event vector while keeping their times.
        e_strat = e_te.copy()
        e_strat[(e_te == cause) & ~in_stratum] = 0  # treat as censored
        c_cr = concordance_index_cr(e_strat, t_te, risk_cr, cause=cause)
        c_rf = concordance_index_cr(e_strat, t_te, risk_rf, cause=cause)
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
    test_idx = np.loadtxt(TEST_IDX, dtype=np.int64)
    t_te = df["time"].to_numpy(dtype=np.float64)[test_idx]
    e_te = df["status"].to_numpy(dtype=np.int64)[test_idx]
    print(
        f"[load] test n={len(test_idx):,}, "
        f"deaths={int((e_te == 2).sum())}, HFs={int((e_te == 1).sum())}",
        flush=True,
    )

    cr_long = pd.read_parquet(CR_RISKS)
    rf_long = pd.read_parquet(RF_RISKS)

    def get_risks(method_long: pd.DataFrame, seed: int) -> tuple[np.ndarray, np.ndarray]:
        sub = method_long[method_long["seed"] == seed].set_index("test_idx").reindex(test_idx)
        return (sub["risk1"].to_numpy(dtype=np.float64), sub["risk2"].to_numpy(dtype=np.float64))

    # --- Stratified diagnostic, averaged over 5 seeds ---
    all_strat = []
    for seed in SEEDS:
        _, cr2 = get_risks(cr_long, seed)
        _, rf2 = get_risks(rf_long, seed)
        s = stratify_cindex(e_te, t_te, cr2, rf2, cause=2, n_strata=4)
        s["seed"] = seed
        all_strat.append(s)
    strat = pd.concat(all_strat, ignore_index=True)

    print("\n" + "=" * 88)
    print(" Death (cause-2) Harrell C-index per death-time quartile")
    print(" (cases outside the stratum are treated as censored; controls = full test cohort)")
    print("=" * 88)
    agg = (
        strat.groupby("stratum")
        .agg(
            t_lo=("t_lo", "first"),
            t_hi=("t_hi", "first"),
            n_cases=("n_cases", "first"),
            cr_mean=("c_cr", "mean"),
            cr_std=("c_cr", "std"),
            rf_mean=("c_rf", "mean"),
            rf_std=("c_rf", "std"),
            gap_mean=("gap", "mean"),
            gap_std=("gap", "std"),
        )
        .reset_index()
    )
    print(
        f"{'stratum':<8}{'t range (days)':<22}{'n_cases':>8}"
        f"{'cr mean ± std':>20}{'rf mean ± std':>20}{'gap (cr-rf)':>15}"
    )
    print("-" * 88)
    for _, r in agg.iterrows():
        print(
            f"  Q{int(r['stratum']):<5}"
            f"[{r['t_lo']:>5.0f}, {r['t_hi']:>5.0f}]      "
            f"{int(r['n_cases']):>8}"
            f"  {r['cr_mean']:.4f} ± {r['cr_std']:.4f}"
            f"  {r['rf_mean']:.4f} ± {r['rf_std']:.4f}"
            f"  {r['gap_mean']:>+8.4f} ± {r['gap_std']:.4f}"
        )
    print("-" * 88)

    # --- Compare to overall (un-stratified) for reference ---
    overall_cr_h = []
    overall_rf_h = []
    overall_cr_u = []
    overall_rf_u = []
    uno_w = compute_uno_weights(t_te, e_te)
    for seed in SEEDS:
        _, cr2 = get_risks(cr_long, seed)
        _, rf2 = get_risks(rf_long, seed)
        overall_cr_h.append(concordance_index_cr(e_te, t_te, cr2, cause=2))
        overall_rf_h.append(concordance_index_cr(e_te, t_te, rf2, cause=2))
        overall_cr_u.append(concordance_index_uno_cr(e_te, t_te, cr2, cause=2, weights=uno_w))
        overall_rf_u.append(concordance_index_uno_cr(e_te, t_te, rf2, cause=2, weights=uno_w))
    print("\n  Overall (no stratification):")
    print(
        f"    Death Harrell:   cr={np.mean(overall_cr_h):.4f}  "
        f"rf={np.mean(overall_rf_h):.4f}  gap={np.mean(overall_cr_h) - np.mean(overall_rf_h):+.4f}"
    )
    print(
        f"    Death Uno IPCW:  cr={np.mean(overall_cr_u):.4f}  "
        f"rf={np.mean(overall_rf_u):.4f}  gap={np.mean(overall_cr_u) - np.mean(overall_rf_u):+.4f}"
    )

    print("\n" + "=" * 88)
    print(" Interpretation")
    print("=" * 88)
    gap_q1 = agg.loc[agg["stratum"] == 1, "gap_mean"].iloc[0]
    gap_q4 = agg.loc[agg["stratum"] == 4, "gap_mean"].iloc[0]
    print(f"  Q1 (early-death) gap: {gap_q1:+.4f}")
    print(f"  Q4 (late-death) gap:  {gap_q4:+.4f}")
    if abs(gap_q4) > 2 * abs(gap_q1):
        print("  → late-death gap >> early-death gap. H2 (real model difference at late times)")
        print("    is supported. Uno IPCW masks the gap because 1/G(t)^2 downweights late pairs.")
    elif abs(gap_q1) > 2 * abs(gap_q4):
        print("  → early-death gap >> late-death gap. (Unexpected pattern.)")
    elif abs(gap_q1) < 0.005 and abs(gap_q4) < 0.005:
        print("  → all strata gap ≈ 0. H1 (Harrell artifact, not real difference) supported:")
        print("    aggregate Harrell gap accumulates from many ~0 contributions, not localized.")
    else:
        print("  → gap distribution mixed; look at per-stratum table above for actual pattern.")


if __name__ == "__main__":
    main()
