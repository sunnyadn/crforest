"""κ.exp4d — canonical-machine aggregator (win i7-14700K, 5 seeds, 100 trees).

Loads:
  /tmp/chf_2012_clean.parquet                         # shared data
  /tmp/chf_2012_test_idx.txt
  /tmp/chf_2012_comprisk_win_risks.parquet            # exp4c (cpu + cuda × 5 seeds × 2 risk scalars)
  /tmp/chf_2012_comprisk_win_walls.parquet
  /tmp/chf_2012_rfsrc_risks_multiseed.parquet         # exp4_multiseed_rfsrc (rf.cores=16)
  /tmp/chf_2012_rfsrc_walls_multiseed.parquet

Reports same-machine apples-to-apples (1) wall-time, (2) Harrell + Uno
C-index for HF/death under both `cif_last` and `integrated_chf` comprisk
risk scalars. This is the table that goes into the paper.

Run on Mac after the win bench finishes and after rsync of win:/tmp/* back:
  rsync win:/tmp/chf_2012_comprisk_win_*.parquet /tmp/
  rsync win:/tmp/chf_2012_rfsrc_*multiseed.parquet /tmp/
  uv run python -u validation/spikes/kappa/exp4d_win_aggregate.py
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from comprisk import concordance_index_cr
from comprisk.metrics import compute_uno_weights, concordance_index_uno_cr

CLEAN_PARQUET = Path("/tmp/chf_2012_clean.parquet")
TEST_IDX = Path("/tmp/chf_2012_test_idx.txt")
CR_WIN_RISKS = Path("/tmp/chf_2012_comprisk_win_risks.parquet")
CR_WIN_WALLS = Path("/tmp/chf_2012_comprisk_win_walls.parquet")
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

    # --- comprisk win (5 seeds × {cpu,cuda} × {cif,chf}) ---
    cr_long = pd.read_parquet(CR_WIN_RISKS)
    cr_walls = pd.read_parquet(CR_WIN_WALLS)
    rows = []
    for device in ("cpu", "cuda"):
        for risk_kind in ("cif", "chf"):
            for seed in SEEDS:
                sub = (
                    cr_long[(cr_long["device"] == device) & (cr_long["seed"] == seed)]
                    .set_index("test_idx")
                    .reindex(test_idx)
                )
                col1 = f"risk1_{risk_kind}"
                col2 = f"risk2_{risk_kind}"
                sc = score(
                    sub[col1].to_numpy(dtype=np.float64), sub[col2].to_numpy(dtype=np.float64)
                )
                wall = float(
                    cr_walls.loc[
                        (cr_walls["device"] == device) & (cr_walls["seed"] == seed), "fit_wall"
                    ].iloc[0]
                )
                rows.append(
                    {
                        "method": f"comprisk-{device}-{risk_kind}",
                        "device": device,
                        "scalar": risk_kind,
                        "seed": seed,
                        "fit_wall": wall,
                        **sc,
                    }
                )

    # --- rfSRC (5 seeds, predicted = integrated CHF mortality) ---
    rf_long = pd.read_parquet(RF_RISKS)
    rf_walls = pd.read_parquet(RF_WALLS)
    for seed in SEEDS:
        sub = rf_long[rf_long["seed"] == seed].set_index("test_idx").reindex(test_idx)
        sc = score(sub["risk1"].to_numpy(dtype=np.float64), sub["risk2"].to_numpy(dtype=np.float64))
        wall = float(rf_walls.loc[rf_walls["seed"] == seed, "fit_wall"].iloc[0])
        rows.append(
            {
                "method": "rfsrc",
                "device": "cpu",
                "scalar": "rfsrc-predict",
                "seed": seed,
                "fit_wall": wall,
                **sc,
            }
        )

    df_all = pd.DataFrame(rows)

    # --- Aggregate ---
    print("\n" + "=" * 100)
    print(" Canonical bench: same-machine win i7-14700K (28 thread, 24 GB WSL, RTX 5070 Ti)")
    print(" 5 seeds × 100 trees × 75k train / 19k test (real CHF cohort)")
    print("=" * 100)
    agg = (
        df_all.groupby(["method", "device", "scalar"])
        .agg(
            wall_mean=("fit_wall", "mean"),
            wall_std=("fit_wall", "std"),
            c1_h_mean=("c1_h", "mean"),
            c1_h_std=("c1_h", "std"),
            c2_h_mean=("c2_h", "mean"),
            c2_h_std=("c2_h", "std"),
            c1_u_mean=("c1_u", "mean"),
            c1_u_std=("c1_u", "std"),
            c2_u_mean=("c2_u", "mean"),
            c2_u_std=("c2_u", "std"),
        )
        .reset_index()
    )
    print(
        f"{'method':<28}{'wall (s)':>16}{'HF Harrell':>14}{'death Harrell':>16}"
        f"{'HF Uno':>12}{'death Uno':>14}"
    )
    print("-" * 100)
    for _, r in agg.iterrows():
        print(
            f"{r['method']:<28}"
            f"{r['wall_mean']:>9.2f}±{r['wall_std']:.2f} "
            f"{r['c1_h_mean']:>9.4f}±{r['c1_h_std']:.4f}"
            f"{r['c2_h_mean']:>9.4f}±{r['c2_h_std']:.4f}"
            f"{r['c1_u_mean']:>7.4f}±{r['c1_u_std']:.4f}"
            f"{r['c2_u_mean']:>7.4f}±{r['c2_u_std']:.4f}"
        )
    print("-" * 100)

    # --- Speedup table ---
    rf_wall = agg.loc[agg["method"] == "rfsrc", "wall_mean"].iloc[0]
    print("\n  Speedup vs rfSRC (same machine, 100 trees):")
    for _, r in agg.iterrows():
        if r["method"] != "rfsrc":
            print(
                f"    {r['method']:<28}: {rf_wall / r['wall_mean']:.2f}× faster "
                f"({r['wall_mean']:.2f}s vs {rf_wall:.2f}s)"
            )

    # --- Cross-method gap analysis (apples-to-apples scalar) ---
    print("\n" + "=" * 100)
    print(" Cross-method gap with matched risk scalar (integrated CHF for both)")
    print(" (the apples-to-apples comparison: same scalar, same machine)")
    print("=" * 100)
    for col, label in (
        ("c1_h", "HF Harrell"),
        ("c2_h", "Death Harrell"),
        ("c1_u", "HF Uno IPCW"),
        ("c2_u", "Death Uno IPCW"),
    ):
        # comprisk cpu integrated_chf
        sub_cr = df_all[(df_all["method"] == "comprisk-cpu-chf")]
        sub_rf = df_all[(df_all["method"] == "rfsrc")]
        cr_mean = sub_cr[col].mean()
        rf_mean = sub_rf[col].mean()
        cr_std = sub_cr[col].std(ddof=1)
        rf_std = sub_rf[col].std(ddof=1)
        pooled_se = np.sqrt(cr_std**2 / 5 + rf_std**2 / 5)
        gap = cr_mean - rf_mean
        ratio = abs(gap) / pooled_se if pooled_se > 0 else float("nan")
        print(
            f"  {label:<18}  cr_chf={cr_mean:.4f}±{cr_std:.4f}  "
            f"rf={rf_mean:.4f}±{rf_std:.4f}  Δ={gap:+.4f}  |Δ|/SE={ratio:5.2f}"
        )

    # --- Compare scalar choice within comprisk ---
    print("\n" + "=" * 100)
    print(" comprisk scalar comparison (same model, same machine, same seeds)")
    print("=" * 100)
    for col, label in (
        ("c1_h", "HF Harrell"),
        ("c2_h", "Death Harrell"),
        ("c1_u", "HF Uno IPCW"),
        ("c2_u", "Death Uno IPCW"),
    ):
        cif = df_all[df_all["method"] == "comprisk-cpu-cif"][col]
        chf = df_all[df_all["method"] == "comprisk-cpu-chf"][col]
        gap = cif.mean() - chf.mean()
        print(
            f"  {label:<18}  CIF[last]={cif.mean():.4f}  "
            f"integrated_CHF={chf.mean():.4f}  Δ(cif-chf)={gap:+.4f}"
        )

    # --- CPU vs GPU bit-equivalence sanity check ---
    print("\n" + "=" * 100)
    print(" comprisk CPU vs CUDA cross-backend ranking agreement (per seed)")
    print("=" * 100)
    for seed in SEEDS:
        cpu_sub = (
            cr_long[(cr_long["device"] == "cpu") & (cr_long["seed"] == seed)]
            .set_index("test_idx")
            .reindex(test_idx)
        )
        cuda_sub = (
            cr_long[(cr_long["device"] == "cuda") & (cr_long["seed"] == seed)]
            .set_index("test_idx")
            .reindex(test_idx)
        )
        # Pearson on CIF cause-2 risk
        r = np.corrcoef(cpu_sub["risk2_cif"], cuda_sub["risk2_cif"])[0, 1]
        print(f"  seed={seed}: Pearson(cpu_risk2, cuda_risk2) = {r:.6f}")


if __name__ == "__main__":
    main()
