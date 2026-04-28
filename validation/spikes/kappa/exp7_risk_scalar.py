"""κ.exp7 — test 'risk scalar choice causes Q1 gap' hypothesis.

κ.exp5 found rfSRC outperforms crforest on early-death pairs. κ.exp6
refuted the binning-quantization hypothesis. New candidate: crforest's
`predict_risk(cause=k) = CIF_k(at last time)` saturates near 1 for
high-risk subjects, while rfSRC's `predicted[,k] = sum_t CHF_k(t)` does
not — integrated CHF preserves rank ordering at the extreme tail where
early-death (high-risk) subjects live.

Test: refit crforest seed=42, build a mortality-style scalar from
predict_chf (integrate cause-2 CHF over time), re-run Q1-Q4 diagnostic
against the same rfSRC seed=42 risks. If Q1 gap shrinks, the original
0.035 gap was risk-scalar artifact, not algorithm difference.

Run: uv run python -u validation/spikes/kappa/exp7_risk_scalar.py
"""

from __future__ import annotations

import time as _time
from pathlib import Path

import numpy as np
import pandas as pd

from crforest import CompetingRiskForest, concordance_index_cr

CLEAN_PARQUET = Path("/tmp/chf_2012_clean.parquet")
TRAIN_IDX = Path("/tmp/chf_2012_train_idx.txt")
TEST_IDX = Path("/tmp/chf_2012_test_idx.txt")
RF_RISKS = Path("/tmp/chf_2012_rfsrc_risks_multiseed.parquet")


def stratified_gap(
    e_te: np.ndarray,
    t_te: np.ndarray,
    risk_cr: np.ndarray,
    risk_rf: np.ndarray,
    label: str,
) -> pd.DataFrame:
    case_mask = e_te == 2
    qs = np.quantile(t_te[case_mask], np.linspace(0, 1, 5))
    rows = []
    for k in range(4):
        lo, hi = qs[k], qs[k + 1]
        if k < 3:
            in_stratum = (t_te >= lo) & (t_te < hi) & (e_te == 2)
        else:
            in_stratum = (t_te >= lo) & (t_te <= hi) & (e_te == 2)
        e_strat = e_te.copy()
        e_strat[(e_te == 2) & ~in_stratum] = 0
        c_cr = concordance_index_cr(e_strat, t_te, risk_cr, cause=2)
        c_rf = concordance_index_cr(e_strat, t_te, risk_rf, cause=2)
        rows.append(
            {
                "label": label,
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

    rf_long = pd.read_parquet(RF_RISKS)
    rf_seed42 = rf_long[rf_long["seed"] == 42].set_index("test_idx").reindex(test_idx)
    rf_risk2 = rf_seed42["risk2"].to_numpy(dtype=np.float64)
    print(
        f"[ref] rfSRC seed=42 mean predict={rf_risk2.mean():.2f} (integrated CHF mortality)\n",
        flush=True,
    )

    print("Fitting crforest seed=42 (n_bins=256, default config)...", flush=True)
    f = CompetingRiskForest(n_estimators=100, n_jobs=-1, random_state=42)
    t0 = _time.perf_counter()
    f.fit(X_tr, t_tr, e_tr)
    print(f"  fit wall: {_time.perf_counter() - t0:.2f}s\n", flush=True)

    # --- Risk scalar A: predict_risk = CIF[-1] (current default) ---
    risk_cif_last = f.predict_risk(X_te, cause=2)

    # --- Risk scalar B: integrated cause-2 CHF (mortality-style, mirrors rfSRC) ---
    chf = f.predict_chf(X_te)  # shape (n_test, n_causes, n_times)
    # cause 2 → index 1; sum over time grid (= rfSRC's "predicted" convention)
    risk_chf_int = chf[:, 1, :].sum(axis=-1)
    # Also the trapezoidal-integrated version, just for completeness:
    times = f.unique_times_
    risk_chf_trapz = np.trapezoid(chf[:, 1, :], times, axis=-1)

    print(
        f"[scalar A] CIF[last]:     mean={risk_cif_last.mean():.4f}  "
        f"max={risk_cif_last.max():.4f}  saturation@1.0: "
        f"{(risk_cif_last > 0.99).mean() * 100:.1f}% subjects",
        flush=True,
    )
    print(
        f"[scalar B] sum(CHF_2):    mean={risk_chf_int.mean():.4f}  "
        f"max={risk_chf_int.max():.2f}  std={risk_chf_int.std():.2f}",
        flush=True,
    )
    print(
        f"[scalar C] trapz(CHF_2):  mean={risk_chf_trapz.mean():.2f}  "
        f"max={risk_chf_trapz.max():.2f}\n",
        flush=True,
    )

    # --- Stratified diagnostic for each risk scalar ---
    s_a = stratified_gap(e_te, t_te, risk_cif_last, rf_risk2, "CIF[last]")
    s_b = stratified_gap(e_te, t_te, risk_chf_int, rf_risk2, "sum(CHF_2)")
    s_c = stratified_gap(e_te, t_te, risk_chf_trapz, rf_risk2, "trapz(CHF_2)")
    all_strat = pd.concat([s_a, s_b, s_c], ignore_index=True)

    print("=" * 90)
    print(" Death (cause-2) Harrell C-index per quartile, varying crforest risk scalar")
    print("=" * 90)
    print(
        f"{'scalar':<14}{'Q1 gap':>11}{'Q2 gap':>11}{'Q3 gap':>11}{'Q4 gap':>11}"
        f"{'Q1 cr':>9}{'Q1 rf':>9}{'Q4 cr':>9}{'Q4 rf':>9}"
    )
    print("-" * 90)
    for label in ("CIF[last]", "sum(CHF_2)", "trapz(CHF_2)"):
        sub = all_strat[all_strat["label"] == label].set_index("stratum")
        print(
            f"{label:<14}"
            f"{sub.loc[1, 'gap']:>+11.4f}{sub.loc[2, 'gap']:>+11.4f}"
            f"{sub.loc[3, 'gap']:>+11.4f}{sub.loc[4, 'gap']:>+11.4f}"
            f"{sub.loc[1, 'c_cr']:>9.4f}{sub.loc[1, 'c_rf']:>9.4f}"
            f"{sub.loc[4, 'c_cr']:>9.4f}{sub.loc[4, 'c_rf']:>9.4f}"
        )
    print("-" * 90)

    # Overall Harrell for sanity
    from crforest.metrics import compute_uno_weights, concordance_index_uno_cr

    uno_w = compute_uno_weights(t_te, e_te)
    print("\n  Overall Harrell / Uno on full test set:")
    for label, risk in [
        ("CIF[last]", risk_cif_last),
        ("sum(CHF_2)", risk_chf_int),
        ("trapz(CHF_2)", risk_chf_trapz),
    ]:
        c_h = concordance_index_cr(e_te, t_te, risk, cause=2)
        c_u = concordance_index_uno_cr(e_te, t_te, risk, cause=2, weights=uno_w)
        print(f"    {label:<14}  Harrell={c_h:.4f}  Uno={c_u:.4f}")
    rf_h = concordance_index_cr(e_te, t_te, rf_risk2, cause=2)
    rf_u = concordance_index_uno_cr(e_te, t_te, rf_risk2, cause=2, weights=uno_w)
    print(f"    {'rfSRC predict':<14}  Harrell={rf_h:.4f}  Uno={rf_u:.4f}")

    print("\n" + "=" * 90)
    print(" Verdict")
    print("=" * 90)
    q1_cif = all_strat[(all_strat.label == "CIF[last]") & (all_strat.stratum == 1)]["gap"].iloc[0]
    q1_chf = all_strat[(all_strat.label == "sum(CHF_2)") & (all_strat.stratum == 1)]["gap"].iloc[0]
    sat_pct = (risk_cif_last > 0.99).mean() * 100
    print(f"  CIF[last] saturation @ 1.0: {sat_pct:.1f}% of test subjects")
    print(f"  Q1 gap with CIF[last]:    {q1_cif:+.4f}")
    print(f"  Q1 gap with sum(CHF_2):   {q1_chf:+.4f}")
    if abs(q1_chf) < abs(q1_cif) - 0.005:
        print(f"  → Q1 gap shrunk by {abs(q1_cif) - abs(q1_chf):.4f}. Risk-scalar choice is")
        print("    a real contributor: CIF[last] saturates at high-risk tail, losing rank info.")
    elif abs(q1_chf - q1_cif) < 0.003:
        print("  → Q1 gap essentially unchanged. Risk-scalar choice is NOT the mechanism.")
        print("    The early-death disagreement reflects an actual model difference.")
    else:
        print(f"  → Q1 gap moved {q1_cif:+.4f} → {q1_chf:+.4f}; pattern unclear.")


if __name__ == "__main__":
    main()
