"""One-shot: generate stratified 80/20 splits for each dataset across 100 seeds.

Output: validation/splits/<dataset>.parquet with columns
  seed (int32), sample_id (int32), fold ("train"|"test").
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from validation.config import DATASETS
from validation.datasets import load
from validation.splits import make_splits

N_SEEDS = 100
TEST_FRAC = 0.2
_OUT_DIR = Path(__file__).resolve().parent / "splits"


def generate_for(name: str) -> pd.DataFrame:
    _, _, event = load(name)
    rows = []
    for seed in range(N_SEEDS):
        train, test = make_splits(len(event), event, seed=seed, test_frac=TEST_FRAC)
        rows.append(
            pd.DataFrame(
                {
                    "seed": np.full(len(train), seed, dtype=np.int32),
                    "sample_id": train.astype(np.int32),
                    "fold": "train",
                }
            )
        )
        rows.append(
            pd.DataFrame(
                {
                    "seed": np.full(len(test), seed, dtype=np.int32),
                    "sample_id": test.astype(np.int32),
                    "fold": "test",
                }
            )
        )
    return pd.concat(rows, ignore_index=True)


def main() -> None:
    _OUT_DIR.mkdir(parents=True, exist_ok=True)
    for name in DATASETS:
        df = generate_for(name)
        out = _OUT_DIR / f"{name}.parquet"
        df.to_parquet(out, index=False)
        print(f"wrote {out.name}: {len(df)} rows, {df['seed'].nunique()} seeds")


if __name__ == "__main__":
    main()
