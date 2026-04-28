"""Stratified 80/20 split generator for paired-seed validation."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

_SPLITS_DIR = Path(__file__).resolve().parent / "splits"


def make_splits(
    n: int,
    event: np.ndarray,
    seed: int,
    test_frac: float = 0.2,
) -> tuple[np.ndarray, np.ndarray]:
    """Return ``(train_idx, test_idx)`` stratified on ``event``.

    Every distinct value in ``event`` appears in both folds; raises
    ``ValueError`` if any stratum has fewer than 2 samples.
    """
    if len(event) != n:
        raise ValueError(f"len(event)={len(event)} does not match n={n}")
    rng = np.random.default_rng(seed)
    train_idx: list[int] = []
    test_idx: list[int] = []
    for code in np.unique(event):
        idx = np.where(event == code)[0]
        if len(idx) < 2:
            raise ValueError(f"stratum event={int(code)} has {len(idx)} samples; need >= 2")
        shuffled = rng.permutation(idx)
        n_test = max(1, round(len(shuffled) * test_frac))
        test_idx.extend(shuffled[:n_test].tolist())
        train_idx.extend(shuffled[n_test:].tolist())
    return (
        np.sort(np.asarray(train_idx, dtype=np.int64)),
        np.sort(np.asarray(test_idx, dtype=np.int64)),
    )


def load(dataset: str) -> list[tuple[np.ndarray, np.ndarray]]:
    """Read committed ``splits/<dataset>.parquet`` as a list of splits.

    Returns ``(train_idx, test_idx)`` tuples in ascending seed order.
    ``out[i]`` corresponds to the i-th smallest seed value, not necessarily seed ``i``.
    """
    pq = _SPLITS_DIR / f"{dataset}.parquet"
    if not pq.exists():
        raise FileNotFoundError(f"{pq} missing; run `uv run python -m validation.gen_splits`")
    df = pd.read_parquet(pq)
    seeds = sorted(df["seed"].unique().tolist())
    out: list[tuple[np.ndarray, np.ndarray]] = []
    for s in seeds:
        sub = df[df["seed"] == s]
        train = sub.loc[sub["fold"] == "train", "sample_id"].to_numpy(dtype=np.int64)
        test = sub.loc[sub["fold"] == "test", "sample_id"].to_numpy(dtype=np.int64)
        out.append((np.sort(train), np.sort(test)))
    return out
