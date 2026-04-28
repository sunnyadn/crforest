"""One-shot: convert CSVs from ``gen_datasets.R`` into parquet.

Run AFTER ``Rscript validation/gen_datasets.R``. Idempotent.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

DATASETS = ["pbc", "follic", "hd"]


def convert_one(name: str, data_dir: Path) -> Path:
    csv = data_dir / f"{name}.csv"
    pq = data_dir / f"{name}.parquet"
    df = pd.read_csv(csv)
    feature_cols = [c for c in df.columns if c not in ("time", "event")]
    df[feature_cols] = df[feature_cols].astype(np.float64)
    df["time"] = df["time"].astype(np.float64)
    df["event"] = df["event"].astype(np.int64)
    df.to_parquet(pq, index=False)
    return pq


def main() -> None:
    data_dir = Path(__file__).resolve().parent / "data"
    for name in DATASETS:
        pq = convert_one(name, data_dir)
        df = pd.read_parquet(pq)
        causes = sorted(df["event"].unique().tolist())
        print(f"wrote {pq.name}: {len(df)} rows, events={causes}")


if __name__ == "__main__":
    main()
