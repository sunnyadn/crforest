"""Post-hoc unified C-index scoring for the n75k_path_b bench output.

The R harness emits rfSRC's native `1 - err.rate` (Ishwaran integrated
mortality) as `harrell_c1` / `harrell_c2`. The README's existing
0.8642/0.8643 row uses concordance_index_cr (Wolbers cause-specific)
scored on the dumped predicted risk vector. This script reads the
per-cell /tmp/rfsrc_n75k_seed*_cores*_risk.parquet dumps, scores them
under the unified metric, and writes back an enriched parquet plus a
side-by-side native vs unified summary.

Run on win after n75k_path_b.py finishes:
  ~/.local/bin/uv run --extra dev python -u \\
    validation/comparisons/_n75k_path_b_rescore.py
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from comprisk import concordance_index_cr

BENCH_PARQUET = Path("/tmp/n75k_path_b.parquet")
CHF_PARQUET = Path("/tmp/chf_2012_clean.parquet")
TEST_IDX_FILE = Path("/tmp/chf_2012_test_idx.txt")


def main() -> None:
    df = pd.read_parquet(BENCH_PARQUET)
    chf = pd.read_parquet(CHF_PARQUET)
    test_idx = np.loadtxt(TEST_IDX_FILE, dtype=np.int64)
    t_te = chf["time"].to_numpy(dtype=np.float64)[test_idx]
    e_te = chf["status"].to_numpy(dtype=np.int64)[test_idx]

    df["c1_unified"] = np.nan
    df["c2_unified"] = np.nan

    for i, row in df.iterrows():
        if row.get("lib") != "rfsrc":
            continue
        risk_path = row.get("risk_path")
        if not isinstance(risk_path, str) or not Path(risk_path).exists():
            continue
        risk_df = pd.read_parquet(risk_path).set_index("test_idx").reindex(test_idx)
        r1 = risk_df["risk1"].to_numpy(dtype=np.float64)
        r2 = risk_df["risk2"].to_numpy(dtype=np.float64)
        df.at[i, "c1_unified"] = concordance_index_cr(e_te, t_te, r1, cause=1)
        df.at[i, "c2_unified"] = concordance_index_cr(e_te, t_te, r2, cause=2)

    # comprisk cells use concordance_index_cr already → c1_unified == harrell_c1.
    cr_mask = df["lib"] == "comprisk"
    df.loc[cr_mask, "c1_unified"] = df.loc[cr_mask, "harrell_c1"]
    df.loc[cr_mask, "c2_unified"] = df.loc[cr_mask, "harrell_c2"]

    df.to_parquet(BENCH_PARQUET)
    print(f"[dump] wrote {BENCH_PARQUET} with c1_unified / c2_unified")

    df["cell"] = df.apply(
        lambda r: (
            "rfsrc_off"
            if (r["lib"] == "rfsrc" and r["rf_cores"] == 1)
            else "rfsrc_on"
            if r["lib"] == "rfsrc"
            else "comprisk"
        ),
        axis=1,
    )
    print("\n## Native (rfSRC err.rate) vs unified (concordance_index_cr) C-index\n")
    cols = ["fit_wall", "peak_rss_gb", "harrell_c1", "c1_unified", "harrell_c2", "c2_unified"]
    print(df.groupby("cell")[cols].agg(["mean", "std"]).round(4).to_string())

    print("\n## OMP-on vs OMP-off output equivalence (per-seed unified C, rfSRC)\n")
    rf = df[df["lib"] == "rfsrc"].copy()
    paired = rf.pivot_table(index="seed", columns="rf_cores", values="c1_unified")
    print(paired.round(5).to_string())
    if 1 in paired.columns and paired.columns.size > 1:
        other = next(c for c in paired.columns if c != 1)
        delta = (paired[1] - paired[other]).abs()
        print(f"\nmax |Δ| (OMP-off vs OMP-on, same seed): {delta.max():.6f}")


if __name__ == "__main__":
    main()
