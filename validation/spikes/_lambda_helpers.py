"""Shared helpers for λ-sprint and κ-sprint spike scripts.

Three patterns recurring across exp1/4/5/6/9 (and likely future spikes):

  * Real CHF parquet+idx loader — same /tmp paths, varies only by whether
    test split is needed.
  * 2-cause Weibull DGP make_synthetic — used by exp1/5 with identical
    Weibull shapes 1.2/0.9 and intercepts -3.0/-3.5.
  * Four-metric C-index block (HF/Death × Harrell/Uno) — used by exp6/9.

Spike scripts can import from this module via:
    sys.path.insert(0, str(Path(__file__).parents[1]))
    from _lambda_helpers import load_chf, make_synthetic, score_four_cindex
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

CHF_CLEAN_PARQUET = Path("/tmp/chf_2012_clean.parquet")
CHF_TRAIN_IDX = Path("/tmp/chf_2012_train_idx.txt")
CHF_TEST_IDX = Path("/tmp/chf_2012_test_idx.txt")


def load_chf(*, with_test: bool = False):
    """Load /tmp/chf_2012_*.parquet+idx files. Returns train arrays plus
    test arrays when with_test=True."""
    df = pd.read_parquet(CHF_CLEAN_PARQUET)
    train_idx = np.loadtxt(CHF_TRAIN_IDX, dtype=np.int64)
    feature_cols = [c for c in df.columns if c not in ("time", "status")]
    X = df[feature_cols].to_numpy(dtype=np.float64)
    t = df["time"].to_numpy(dtype=np.float64)
    e = df["status"].to_numpy(dtype=np.int64)
    if not with_test:
        return X[train_idx], t[train_idx], e[train_idx], len(feature_cols)
    test_idx = np.loadtxt(CHF_TEST_IDX, dtype=np.int64)
    return (
        X[train_idx],
        t[train_idx],
        e[train_idx],
        X[test_idx],
        t[test_idx],
        e[test_idx],
        len(feature_cols),
    )


def make_synthetic(n: int, p: int, seed: int):
    """3-state competing-risks Weibull DGP — 5 informative + (p-5) noise.
    Cause-1 shape 1.2, cause-2 shape 0.9, censoring rate 0.06."""
    rng = np.random.default_rng(seed)
    X = rng.standard_normal(size=(n, p))
    beta1 = np.zeros(p)
    beta1[: min(5, p)] = [0.8, 0.4, -0.3, 0.0, 0.0][: min(5, p)]
    beta2 = np.zeros(p)
    if p >= 5:
        beta2[3:5] = [-0.5, 0.6]
    lam1 = np.exp(-3.0 + X @ beta1)
    lam2 = np.exp(-3.5 + X @ beta2)
    u1 = rng.uniform(size=n)
    u2 = rng.uniform(size=n)
    t1 = (-np.log(u1) / lam1) ** (1.0 / 1.2)
    t2 = (-np.log(u2) / lam2) ** (1.0 / 0.9)
    c = rng.exponential(scale=1.0 / 0.06, size=n)
    times = np.minimum.reduce([t1, t2, c])
    event = np.where(times == t1, 1, np.where(times == t2, 2, 0)).astype(np.int64)
    return X.astype(np.float64), times.astype(np.float64), event


def score_four_cindex(forest, X_te, t_te, e_te, w_te) -> dict:
    """Compute HF/Death × Harrell/Uno on a fitted forest. Returns a dict
    with keys hf_harrell, death_harrell, hf_uno, death_uno."""
    from comprisk import concordance_index_cr
    from comprisk.metrics import concordance_index_uno_cr

    r1 = forest.predict_risk(X_te, cause=1)
    r2 = forest.predict_risk(X_te, cause=2)
    return {
        "hf_harrell": concordance_index_cr(e_te, t_te, r1, cause=1),
        "death_harrell": concordance_index_cr(e_te, t_te, r2, cause=2),
        "hf_uno": concordance_index_uno_cr(e_te, t_te, r1, cause=1, weights=w_te),
        "death_uno": concordance_index_uno_cr(e_te, t_te, r2, cause=2, weights=w_te),
    }
