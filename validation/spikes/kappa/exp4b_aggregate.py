"""κ.exp4b — aggregate crforest + rfSRC multi-seed dumps, answer:
is the 0.016 death C-index gap within either method's seed-to-seed noise?

Loads:
  /tmp/chf_2012_crforest_risks_multiseed.parquet  (from exp4a, Mac)
  /tmp/chf_2012_crforest_walls_multiseed.parquet
  /tmp/chf_2012_rfsrc_risks_multiseed.parquet     (from exp4_multiseed_rfsrc.R, win)
  /tmp/chf_2012_rfsrc_walls_multiseed.parquet

Scores both prediction sets with the SAME `concordance_index_cr` +
`concordance_index_uno_cr`, then reports within-method std + cross-method gap.

Run: uv run python -u validation/spikes/kappa/exp4b_aggregate.py
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
CR_WALLS = Path("/tmp/chf_2012_crforest_walls_multiseed.parquet")
RF_RISKS = Path("/tmp/chf_2012_rfsrc_risks_multiseed.parquet")
RF_WALLS = Path("/tmp/chf_2012_rfsrc_walls_multiseed.parquet")
SEEDS = [42, 43, 44, 45, 46]


def main() -> None:
    df = pd.read_parquet(CLEAN_PARQUET)
    test_idx = np.loadtxt(TEST_IDX, dtype=np.int64)
    t = df["time"].to_numpy(dtype=np.float64)
    e = df["status"].to_numpy(dtype=np.int64)
    t_te, e_te = t[test_idx], e[test_idx]
    print(f"[load] test n={len(test_idx):,}", flush=True)

    uno_w = compute_uno_weights(t_te, e_te)

    def score(risk1: np.ndarray, risk2: np.ndarray) -> dict:
        return {
            "c1_h": concordance_index_cr(e_te, t_te, risk1, cause=1),
            "c2_h": concordance_index_cr(e_te, t_te, risk2, cause=2),
            "c1_u": concordance_index_uno_cr(e_te, t_te, risk1, cause=1, weights=uno_w),
            "c2_u": concordance_index_uno_cr(e_te, t_te, risk2, cause=2, weights=uno_w),
        }

    def load_method(method: str, risks_path: Path, walls_path: Path) -> pd.DataFrame:
        risks = pd.read_parquet(risks_path)
        walls = pd.read_parquet(walls_path)
        rows = []
        for seed in SEEDS:
            # Reindex by test_idx so risk vectors line up with t_te/e_te.
            sub = risks[risks["seed"] == seed].set_index("test_idx").reindex(test_idx)
            assert not sub["risk1"].isna().any(), f"{method} seed={seed} missing test rows"
            sc = score(
                sub["risk1"].to_numpy(dtype=np.float64), sub["risk2"].to_numpy(dtype=np.float64)
            )
            sc["seed"] = seed
            sc["fit_wall"] = float(walls.loc[walls["seed"] == seed, "fit_wall"].iloc[0])
            sc["method"] = method
            rows.append(sc)
        return pd.DataFrame(rows)

    cr_df = load_method("crforest", CR_RISKS, CR_WALLS)
    rf_df = load_method("rfsrc", RF_RISKS, RF_WALLS)
    all_df = pd.concat([cr_df, rf_df], ignore_index=True)

    # --- per-seed table ---
    print("\n" + "=" * 80)
    print(" per-seed C-index (apples-to-apples; same metric on both prediction sets)")
    print("=" * 80)
    print(
        all_df[["method", "seed", "fit_wall", "c1_h", "c2_h", "c1_u", "c2_u"]].to_string(
            index=False, float_format=lambda v: f"{v:.4f}"
        )
    )

    # --- within-method statistics ---
    print("\n" + "=" * 80)
    print(" within-method statistics (n=5 seeds)")
    print("=" * 80)
    for method in ("crforest", "rfsrc"):
        sub = all_df[all_df["method"] == method]
        print(f"\n  {method}:")
        for col in ("c1_h", "c2_h", "c1_u", "c2_u"):
            v = sub[col]
            print(
                f"    {col}: mean={v.mean():.4f}  std={v.std(ddof=1):.4f}  "
                f"min={v.min():.4f}  max={v.max():.4f}  range={v.max() - v.min():.4f}"
            )
        wall = sub["fit_wall"]
        print(
            f"    fit_wall: mean={wall.mean():.2f}s  std={wall.std(ddof=1):.2f}s  "
            f"range=[{wall.min():.2f}, {wall.max():.2f}]"
        )

    # --- cross-method gap vs within-method std ---
    print("\n" + "=" * 80)
    print(" cross-method gap vs within-method seed std")
    print("=" * 80)
    print(
        f"{'metric':<10}{'cr mean':>10}{'rf mean':>10}{'Δ (cr-rf)':>12}"
        f"{'cr std':>10}{'rf std':>10}{'pooled SE':>11}{'|Δ|/SE':>10}"
    )
    print("-" * 83)
    for col in ("c1_h", "c2_h", "c1_u", "c2_u"):
        cr_mean = cr_df[col].mean()
        rf_mean = rf_df[col].mean()
        cr_std = cr_df[col].std(ddof=1)
        rf_std = rf_df[col].std(ddof=1)
        # SE of Δ assuming independent samples of n=5 each.
        pooled_se = np.sqrt(cr_std**2 / 5 + rf_std**2 / 5)
        gap = cr_mean - rf_mean
        ratio = abs(gap) / pooled_se if pooled_se > 0 else float("nan")
        print(
            f"{col:<10}{cr_mean:>10.4f}{rf_mean:>10.4f}{gap:>+12.4f}"
            f"{cr_std:>10.4f}{rf_std:>10.4f}{pooled_se:>11.4f}{ratio:>10.2f}"
        )
    print("-" * 83)
    print("|Δ|/SE < 2 → gap consistent with seed noise; > 2 → likely structural difference.")

    # --- speedup summary ---
    print("\n" + "=" * 80)
    print(" wall-time comparison")
    print("=" * 80)
    cr_wall = cr_df["fit_wall"].mean()
    rf_wall = rf_df["fit_wall"].mean()
    print(f"  crforest mean fit:  {cr_wall:.2f}s  (Mac CPU, 100 trees)")
    print(f"  rfsrc    mean fit:  {rf_wall:.2f}s  (win 28-thread, 100 trees)")
    print(f"  speedup (rf/cr):    {rf_wall / cr_wall:.2f}× crforest faster on its own machine")
    print("  NOTE: speedup is cross-machine (Mac M-series vs win i7+OpenBLAS) — for")
    print("        canonical benchmarks both should run on the same machine.")


if __name__ == "__main__":
    main()
