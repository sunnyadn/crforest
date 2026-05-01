"""Synthetic 2-cause Weibull competing-risks DGP shared by Python and R harnesses.

Identical to the DGP in validation/profile_fit.py and bench/dgp.R; centralizing here
so future bench runs can import a single source of truth. Seed 20260417 is the
canonical bench seed and must match dgp.R for cross-language reproducibility.
"""

from __future__ import annotations

import numpy as np


def make_synthetic_cr(
    n: int, p: int, seed: int = 20260417
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    X = rng.standard_normal(size=(n, p)).astype(np.float64)
    beta_1 = np.zeros(p)
    beta_2 = np.zeros(p)
    beta_1[:5] = np.array([0.8, 0.4, -0.3, 0.0, 0.0])[:5]
    beta_2[:5] = np.array([0.0, 0.0, 0.0, -0.5, 0.6])[:5]
    if p >= 10:
        beta_2[5:10] = np.array([0.5, -0.4, 0.3, 0.0, -0.6])
        beta_1[5:10] = np.array([0.0, 0.3, -0.5, 0.4, 0.0])
    alpha = (1.2, 0.9)
    intercept_1, intercept_2 = -3.0, -3.5
    censor_rate = 0.06
    lam1 = np.exp(intercept_1 + X @ beta_1)
    lam2 = np.exp(intercept_2 + X @ beta_2)
    u1 = rng.uniform(size=n)
    u2 = rng.uniform(size=n)
    t1 = (-np.log(u1) / lam1) ** (1.0 / alpha[0])
    t2 = (-np.log(u2) / lam2) ** (1.0 / alpha[1])
    c = rng.exponential(scale=1.0 / censor_rate, size=n)
    times = np.minimum.reduce([t1, t2, c])
    event = np.where(times == t1, 1, np.where(times == t2, 2, 0)).astype(np.int64)
    return X, times, event
