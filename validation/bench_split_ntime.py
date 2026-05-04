"""ε §6.2 gate: split_ntime=N vs split_ntime=None vs rfSRC.

Inputs:
- parquet-coarse: results from `validation run ... --split-ntime <N>`
- parquet-none:   results from `validation run ... --split-ntime None`

Gate (PRD §6.2, matching ``validation/report.py``):
- bias:   |median(ΔC_seed)| < 0.01 on every dataset
- spread: IQR(ΔC_seed) <= IQR of rfSRC's own seed-to-seed C on the same seeds

Also reports the per-seed ``median(|ΔC_seed|)`` as a diagnostic (noise+bias
combined) alongside the gate columns.
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
import pandas as pd


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--parquet-coarse", required=True, type=Path)
    ap.add_argument("--parquet-none", required=True, type=Path)
    ap.add_argument(
        "--label",
        default="coarse",
        help="short label for the coarse run (e.g. 'ntime=30'); appears in report headings",
    )
    args = ap.parse_args()

    dco = pd.read_parquet(args.parquet_coarse)
    dno = pd.read_parquet(args.parquet_none)

    rows = []
    for dataset, gco in dco.groupby("dataset"):
        gno = dno[dno["dataset"] == dataset]
        delta_co = gco["c_comprisk"].values - gco["c_rfsrc"].values
        delta_no = gno["c_comprisk"].values - gno["c_rfsrc"].values
        rfsrc_c = gco["c_rfsrc"].values
        rows.append(
            dict(
                dataset=dataset,
                n_seeds=len(gco),
                abs_median_delta_c=float(abs(np.median(delta_co))),
                iqr_delta_c=float(np.quantile(delta_co, 0.75) - np.quantile(delta_co, 0.25)),
                rfsrc_seed_iqr_c=float(np.quantile(rfsrc_c, 0.75) - np.quantile(rfsrc_c, 0.25)),
                abs_median_delta_c_none=float(abs(np.median(delta_no))),
                median_abs_delta_c=float(np.median(np.abs(delta_co))),
            )
        )

    df = pd.DataFrame(rows)
    df["pass_bias"] = df["abs_median_delta_c"] < 0.01
    df["pass_iqr"] = df["iqr_delta_c"] <= df["rfsrc_seed_iqr_c"]
    df["pass"] = df["pass_bias"] & df["pass_iqr"]

    out = Path("validation/reports") / f"{time.strftime('%Y-%m-%d')}-split-ntime-alignment.md"
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w") as f:
        f.write(f"# ε §6.2 alignment — split_ntime={args.label} vs rfSRC\n\n")
        f.write(f"Source: `{args.parquet_coarse.name}` vs `{args.parquet_none.name}`\n\n")
        f.write(
            f"| dataset | n | \\|median ΔC\\| ({args.label}) | \\|median ΔC\\| (None) | "
            "IQR ΔC | rfSRC seed IQR | median \\|ΔC\\| | pass bias | pass IQR | PASS |\n"
        )
        f.write("|---|---|---|---|---|---|---|---|---|---|\n")
        for _, r in df.iterrows():
            f.write(
                f"| {r['dataset']} | {r['n_seeds']} | {r['abs_median_delta_c']:.4f} | "
                f"{r['abs_median_delta_c_none']:.4f} | "
                f"{r['iqr_delta_c']:.4f} | {r['rfsrc_seed_iqr_c']:.4f} | "
                f"{r['median_abs_delta_c']:.4f} | "
                f"{'PASS' if r['pass_bias'] else 'FAIL'} | "
                f"{'PASS' if r['pass_iqr'] else 'FAIL'} | "
                f"{'✅' if r['pass'] else '❌'} |\n"
            )
        overall = bool(df["pass"].all())
        f.write(f"\n**Overall §6.2 gate:** {'PASS' if overall else 'FAIL'}\n")
    print(f"wrote {out}")
    print(df.to_string(index=False))
    if not bool(df["pass"].all()):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
