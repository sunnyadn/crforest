"""κ.exp3 — apples-to-apples crforest vs rfSRC summary.

Reads `/tmp/chf_2012_rfsrc_risk.parquet` (rfSRC test predictions from exp2.R),
re-runs crforest fit-and-predict on the same train/test split, then scores
BOTH risk vectors with the SAME `concordance_index_cr` + `concordance_index_uno_cr`
implementations from `crforest.metrics`.

Run: uv run python -u validation/spikes/kappa/exp3_compare.py
"""

from __future__ import annotations

import time as _time
from pathlib import Path

import numpy as np
import pandas as pd

from crforest import CompetingRiskForest, concordance_index_cr
from crforest.metrics import compute_uno_weights, concordance_index_uno_cr

CLEAN_PARQUET = Path("/tmp/chf_2012_clean.parquet")
RFSRC_RISK = Path("/tmp/chf_2012_rfsrc_risk.parquet")
RFSRC_CIF = Path("/tmp/chf_2012_rfsrc_cif.parquet")
TRAIN_IDX = Path("/tmp/chf_2012_train_idx.txt")
TEST_IDX = Path("/tmp/chf_2012_test_idx.txt")


def main() -> None:
    df = pd.read_parquet(CLEAN_PARQUET)
    train_idx = np.loadtxt(TRAIN_IDX, dtype=np.int64)
    test_idx = np.loadtxt(TEST_IDX, dtype=np.int64)
    feature_cols = [c for c in df.columns if c not in ("time", "status")]

    X_full = df[feature_cols].to_numpy(dtype=np.float64)
    t_full = df["time"].to_numpy(dtype=np.float64)
    e_full = df["status"].to_numpy(dtype=np.int64)

    X_tr, t_tr, e_tr = X_full[train_idx], t_full[train_idx], e_full[train_idx]
    X_te, t_te, e_te = X_full[test_idx], t_full[test_idx], e_full[test_idx]
    print(
        f"[load] train n={len(train_idx):,}, test n={len(test_idx):,}, p={len(feature_cols)}",
        flush=True,
    )

    # --- crforest fit + predict ---
    print("\n[crforest] fitting 100 trees, default config, n_jobs=-1, device='cpu'", flush=True)
    forest = CompetingRiskForest(n_estimators=100, n_jobs=-1, random_state=42)
    t0 = _time.perf_counter()
    forest.fit(X_tr, t_tr, e_tr)
    cr_fit = _time.perf_counter() - t0
    t0 = _time.perf_counter()
    cr_risk1 = forest.predict_risk(X_te, cause=1)
    cr_risk2 = forest.predict_risk(X_te, cause=2)
    cr_pred = _time.perf_counter() - t0
    print(f"[crforest] fit {cr_fit:.2f}s, predict {cr_pred:.2f}s", flush=True)

    # --- Full CIF on rfSRC's time grid for direct curve comparison ---
    cif_long = pd.read_parquet(RFSRC_CIF)
    n_test = int(cif_long["subj"].max()) + 1
    nt = int(cif_long["t_idx"].max()) + 1
    assert n_test == len(test_idx), f"n_test mismatch: cif={n_test}, idx={len(test_idx)}"
    rf_times = cif_long.loc[cif_long["subj"] == 0, "t_value"].to_numpy(dtype=np.float64)
    rf_cif1 = cif_long["cif1"].to_numpy(dtype=np.float64).reshape(n_test, nt)
    rf_cif2 = cif_long["cif2"].to_numpy(dtype=np.float64).reshape(n_test, nt)

    cr_cif_all = forest.predict_cif(X_te, times=rf_times)  # (n_test, n_causes, nt)
    cr_cif1 = cr_cif_all[:, 0, :]
    cr_cif2 = cr_cif_all[:, 1, :]
    print(
        f"[cif-grid] using rfSRC time.interest: nt={nt}, range=[{rf_times.min():.0f}, "
        f"{rf_times.max():.0f}] days",
        flush=True,
    )

    # --- rfSRC predictions (loaded from R-side dump) ---
    rfsrc_df = pd.read_parquet(RFSRC_RISK)
    # Sanity: rfSRC dumped test_idx in same order; verify alignment.
    assert (rfsrc_df["test_idx"].to_numpy() == test_idx).all(), "rfSRC test idx misaligned"
    rf_risk1 = rfsrc_df["risk_cause1"].to_numpy(dtype=np.float64)
    rf_risk2 = rfsrc_df["risk_cause2"].to_numpy(dtype=np.float64)

    # --- Score both with the SAME crforest metric implementations ---
    uno_w = compute_uno_weights(t_te, e_te)

    def score(risk1: np.ndarray, risk2: np.ndarray) -> tuple[float, float, float, float]:
        c1_h = concordance_index_cr(e_te, t_te, risk1, cause=1)
        c2_h = concordance_index_cr(e_te, t_te, risk2, cause=2)
        c1_u = concordance_index_uno_cr(e_te, t_te, risk1, cause=1, weights=uno_w)
        c2_u = concordance_index_uno_cr(e_te, t_te, risk2, cause=2, weights=uno_w)
        return c1_h, c2_h, c1_u, c2_u

    cr_c1h, cr_c2h, cr_c1u, cr_c2u = score(cr_risk1, cr_risk2)
    rf_c1h, rf_c2h, rf_c1u, rf_c2u = score(rf_risk1, rf_risk2)

    # --- Side-by-side print ---
    print("\n" + "=" * 70)
    print(" apples-to-apples: crforest vs rfSRC on real CHF cohort (n=75k/19k)")
    print("=" * 70)
    print(f"{'metric':<32}{'crforest':>14}{'rfSRC':>14}{'Δ (cr − rf)':>14}")
    print("-" * 70)
    print(f"{'fit wall (s)':<32}{cr_fit:>14.2f}{235.38:>14.2f}{cr_fit - 235.38:>+14.2f}")
    print(f"{'speedup (×)':<32}{'':>14}{'':>14}{235.38 / cr_fit:>13.2f}×")
    print(f"{'predict wall (s)':<32}{cr_pred:>14.2f}{6.31:>14.2f}{cr_pred - 6.31:>+14.2f}")
    print(
        f"{'C-index Harrell cause-1 (HF)':<32}{cr_c1h:>14.4f}{rf_c1h:>14.4f}"
        f"{cr_c1h - rf_c1h:>+14.4f}"
    )
    print(
        f"{'C-index Harrell cause-2 (death)':<32}{cr_c2h:>14.4f}{rf_c2h:>14.4f}"
        f"{cr_c2h - rf_c2h:>+14.4f}"
    )
    print(
        f"{'C-index Uno IPCW cause-1':<32}{cr_c1u:>14.4f}{rf_c1u:>14.4f}{cr_c1u - rf_c1u:>+14.4f}"
    )
    print(
        f"{'C-index Uno IPCW cause-2':<32}{cr_c2u:>14.4f}{rf_c2u:>14.4f}{cr_c2u - rf_c2u:>+14.4f}"
    )
    print("=" * 70)

    # --- CIF curve direct comparison ---
    print("\n" + "=" * 70)
    print(" CIF curve comparison (per-subject × per-time, on rfSRC grid)")
    print("=" * 70)
    for cause, cr_cif, rf_cif in [(1, cr_cif1, rf_cif1), (2, cr_cif2, rf_cif2)]:
        diff = cr_cif - rf_cif
        absdiff = np.abs(diff)
        flat_cr = cr_cif.flatten()
        flat_rf = rf_cif.flatten()
        # Pearson on flattened (n_test * nt) pairs.
        pearson_flat = np.corrcoef(flat_cr, flat_rf)[0, 1]
        # Per-time Pearson averaged.
        per_time_pearson = np.array(
            [
                np.corrcoef(cr_cif[:, j], rf_cif[:, j])[0, 1]
                for j in range(nt)
                if cr_cif[:, j].std() > 0 and rf_cif[:, j].std() > 0
            ]
        )
        print(f"\n  cause-{cause}:")
        print(f"    mean |Δ CIF|:           {absdiff.mean():.4f}")
        print(f"    p50 |Δ CIF|:            {np.percentile(absdiff, 50):.4f}")
        print(f"    p95 |Δ CIF|:            {np.percentile(absdiff, 95):.4f}")
        print(f"    max |Δ CIF|:            {absdiff.max():.4f}")
        print(f"    mean signed Δ (cr−rf):  {diff.mean():+.4f}  (<0 = crforest underestimates)")
        print(f"    Pearson (full flat):    {pearson_flat:.4f}")
        print(
            f"    mean per-time Pearson:  {per_time_pearson.mean():.4f}  "
            f"(min={per_time_pearson.min():.4f})"
        )
        # Spot times: 1y, 3y, 5y → find nearest indices
        for tgt_days, label in [(365, "1y"), (1095, "3y"), (1825, "5y")]:
            j = int(np.searchsorted(rf_times, tgt_days))
            if j >= nt:
                j = nt - 1
            cr_at_t = cr_cif[:, j]
            rf_at_t = rf_cif[:, j]
            r = np.corrcoef(cr_at_t, rf_at_t)[0, 1]
            print(
                f"    @ {label} (t={rf_times[j]:.0f}d):    "
                f"mean cr={cr_at_t.mean():.4f}, mean rf={rf_at_t.mean():.4f}, "
                f"Pearson={r:.4f}"
            )
    print("=" * 70)

    # --- Risk score correlation (the scalar we use for C-index) ---
    print("\n" + "=" * 70)
    print(" predict_risk scalar correlation (= CIF at last time)")
    print("=" * 70)
    for cause, cr_r, rf_r in [(1, cr_risk1, rf_risk1), (2, cr_risk2, rf_risk2)]:
        from scipy.stats import spearmanr

        pearson = np.corrcoef(cr_r, rf_r)[0, 1]
        spearman = spearmanr(cr_r, rf_r).statistic
        print(
            f"  cause-{cause}:  Pearson={pearson:.4f}  Spearman={spearman:.4f}  "
            f"(cr mean={cr_r.mean():.4f}, rf mean={rf_r.mean():.4f})"
        )
    print("=" * 70)


if __name__ == "__main__":
    main()
