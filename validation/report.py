"""Aggregate per-seed results into per-dataset summary and markdown report."""

from __future__ import annotations

from dataclasses import asdict
from pathlib import Path

import numpy as np
import pandas as pd

from validation.runner import SeedResult

PASS_THRESHOLD = 0.01
REPORT_COLUMNS = [
    "dataset",
    "n_seeds",
    "median_c_comprisk",
    "median_c_rfsrc",
    "median_delta_c",
    "iqr_delta_c",
    "max_abs_delta_c",
    "pass",
]


def results_to_df(results: list[SeedResult]) -> pd.DataFrame:
    """Convert a list of SeedResult to a tidy DataFrame."""
    return pd.DataFrame([asdict(r) for r in results])


def summarize(df: pd.DataFrame) -> pd.DataFrame:
    """Return a per-dataset summary from a SeedResult DataFrame."""
    grouped = df.groupby("dataset", sort=False)
    out = grouped.agg(
        n_seeds=("seed", "size"),
        median_c_comprisk=("c_comprisk", "median"),
        median_c_rfsrc=("c_rfsrc", "median"),
        median_delta_c=("delta_c", "median"),
        iqr_delta_c=("delta_c", lambda s: float(np.quantile(s, 0.75) - np.quantile(s, 0.25))),
        max_abs_delta_c=("delta_c", lambda s: float(np.max(np.abs(s)))),
    ).reset_index()
    out["pass"] = out["median_delta_c"].abs() < PASS_THRESHOLD
    return out[REPORT_COLUMNS]


def write_report(
    summary: pd.DataFrame,
    path: Path,
    run_date: str,
    commit: str,
    n_seeds: int,
) -> None:
    """Write the markdown report to ``path``."""
    path = Path(path)
    lines = [
        "# comprisk vs randomForestSRC — paired-seed validation",
        "",
        f"Run: {run_date}",
        f"Seeds per dataset: {n_seeds}",
        f"comprisk commit: {commit}",
        "",
        "| Dataset | Seeds | Median C_cr | Median C_rfSRC | Median ΔC | IQR ΔC | Max |ΔC| | Pass |",
        "|---------|-------|-------------|----------------|-----------|--------|----------|------|",
    ]
    for _, row in summary.iterrows():
        mark = "✓" if bool(row["pass"]) else "⚠ follow-up"
        lines.append(
            f"| {row['dataset']} | {int(row['n_seeds'])} "
            f"| {row['median_c_comprisk']:.3f} "
            f"| {row['median_c_rfsrc']:.3f} "
            f"| {row['median_delta_c']:+.3f} "
            f"| {row['iqr_delta_c']:.3f} "
            f"| {row['max_abs_delta_c']:.3f} "
            f"| {mark} |"
        )
    path.write_text("\n".join(lines) + "\n")
