"""One-shot synthetic competing-risks dataset.

Writes ``validation/data/synthetic.parquet``. 2-cause Weibull cause-specific
hazards with 5 informative + 5 noise features. See spec §4 for DGP details.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

SEED = 20260417
N = 1000
P = 10
ALPHA = (1.2, 0.9)
BETA_1 = np.array([0.8, 0.4, -0.3, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
BETA_2 = np.array([0.0, 0.0, 0.0, -0.5, 0.6, 0.0, 0.0, 0.0, 0.0, 0.0])
INTERCEPT_1 = -3.0
INTERCEPT_2 = -3.5
CENSOR_RATE = 0.06


def generate_synthetic() -> pd.DataFrame:
    rng = np.random.default_rng(SEED)
    X = rng.standard_normal(size=(N, P))
    lam1 = np.exp(INTERCEPT_1 + X @ BETA_1)
    lam2 = np.exp(INTERCEPT_2 + X @ BETA_2)
    u1 = rng.uniform(size=N)
    u2 = rng.uniform(size=N)
    t1 = (-np.log(u1) / lam1) ** (1.0 / ALPHA[0])
    t2 = (-np.log(u2) / lam2) ** (1.0 / ALPHA[1])
    c = rng.exponential(scale=1.0 / CENSOR_RATE, size=N)
    times = np.minimum.reduce([t1, t2, c])
    event = np.where(times == t1, 1, np.where(times == t2, 2, 0)).astype(np.int64)
    df = pd.DataFrame(X, columns=[f"x{i}" for i in range(P)])
    df["time"] = times
    df["event"] = event
    return df


def main() -> None:
    out = Path(__file__).resolve().parent / "data" / "synthetic.parquet"
    out.parent.mkdir(parents=True, exist_ok=True)
    df = generate_synthetic()
    df.to_parquet(out, index=False)
    print(f"wrote {out} ({len(df)} rows, {(df['event'] == 0).mean():.1%} censored)")


if __name__ == "__main__":
    main()
