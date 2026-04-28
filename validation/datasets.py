"""Dataset loaders for the validation harness.

Every dataset has columns ``x0..x{p-1}``, ``time``, ``event`` where
``event == 0`` means censored and ``event >= 1`` is cause of event.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

_DATA_DIR = Path(__file__).resolve().parent / "data"


def load(name: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return ``(X, time, event)`` for the named dataset."""
    pq = _DATA_DIR / f"{name}.parquet"
    if not pq.exists():
        raise FileNotFoundError(
            f"{pq} missing; run the vendoring scripts per validation/gen_datasets.R"
        )
    df = pd.read_parquet(pq)
    feature_cols = [c for c in df.columns if c not in ("time", "event")]
    X = df[feature_cols].to_numpy(dtype=np.float64)
    time = df["time"].to_numpy(dtype=np.float64)
    event = df["event"].to_numpy(dtype=np.int64)
    return X, time, event
