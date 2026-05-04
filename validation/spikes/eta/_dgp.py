"""Frozen Weibull CR DGP for the η perf-ceiling spike.

Mirrors validation/spikes/zeta/run_comprisk.py:weibull_cr (same constants),
extended to n=100000. Kept in its own module so all η experiments load
data through one path.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

DATA_DIR = Path(__file__).parent / "data"

P = 60
N_INFORMATIVE = 10
BETA_SCALE = 0.5
SHAPE_1 = 1.2
SHAPE_2 = 0.9
CENS_RATE = 0.06


def weibull_cr(n: int, p: int, seed: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, p))
    beta = rng.normal(scale=BETA_SCALE, size=(p,))
    beta[N_INFORMATIVE:] = 0.0
    lp = X @ beta
    u = rng.uniform(size=n)
    t1 = (-np.log(u) * np.exp(-lp)) ** (1 / SHAPE_1)
    u = rng.uniform(size=n)
    t2 = (-np.log(u) * np.exp(-lp)) ** (1 / SHAPE_2)
    cens = rng.exponential(scale=1 / CENS_RATE, size=n)
    t_event = np.minimum(t1, t2)
    time_obs = np.minimum(t_event, cens)
    event = np.where(time_obs < cens, np.where(t1 < t2, 1, 2), 0).astype(np.int64)
    return X, time_obs, event


def parquet_path(n: int, seed: int) -> Path:
    return DATA_DIR / f"weibull_n{n}_s{seed}.parquet"


def ensure(n: int, seed: int) -> Path:
    DATA_DIR.mkdir(exist_ok=True)
    p = parquet_path(n, seed)
    if p.exists():
        return p
    print(f"[gen] writing {p.name} ({n} rows × {P} cols)…", flush=True)
    X, t, e = weibull_cr(n, P, seed)
    cols = {f"X_{i}": X[:, i] for i in range(P)}
    cols["time"] = t
    cols["event"] = e
    pd.DataFrame(cols).to_parquet(p)
    return p


def load(n: int, seed: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    ensure(n, seed)
    df = pd.read_parquet(parquet_path(n, seed))
    X = df[[f"X_{i}" for i in range(P)]].to_numpy()
    return X, df["time"].to_numpy(), df["event"].to_numpy()
