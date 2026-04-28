"""ε perf exit-gate: fit wall vs split_ntime on the gamma-sprint Weibull workload.

Runs at n ∈ {5000, 20000, 50000}, both split_ntime ∈ {None, <chosen default>}.
Fits log-log (t = beta * n^alpha), extrapolates to n=100_000, compares against
split_ntime=None baseline for exit-gate assessment.
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np

from crforest import CompetingRiskForest


def weibull_cr(n: int, p: int, seed: int):
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, p))
    beta = rng.normal(scale=0.5, size=(p,))
    beta[10:] = 0.0
    lp = X @ beta
    u = rng.uniform(size=n)
    t1 = (-np.log(u) * np.exp(-lp)) ** (1 / 1.2)
    u = rng.uniform(size=n)
    t2 = (-np.log(u) * np.exp(-lp)) ** (1 / 0.9)
    cens = rng.exponential(scale=1 / 0.06, size=n)
    t_event = np.minimum(t1, t2)
    time_obs = np.minimum(t_event, cens)
    event = np.where(time_obs < cens, np.where(t1 < t2, 1, 2), 0).astype(np.int64)
    return X, time_obs, event


def _time_fit(n: int, p: int, split_ntime) -> float:
    X, t, e = weibull_cr(n, p, seed=0)
    f = CompetingRiskForest(
        n_estimators=500,
        random_state=0,
        n_jobs=-1,
        split_ntime=split_ntime,
    )
    start = time.perf_counter()
    f.fit(X, t, e)
    return time.perf_counter() - start


def _extrapolate(ns: list[int], walls: list[float]) -> tuple[float, float, float]:
    """Fit log(t) = alpha*log(n) + beta. Returns (alpha, beta, t_at_100k)."""
    logn = np.log(ns)
    logt = np.log(walls)
    slope, intercept = np.polyfit(logn, logt, deg=1)
    t100k = float(np.exp(intercept + slope * np.log(100_000)))
    return float(slope), float(intercept), t100k


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--split-ntime",
        type=lambda s: None if s.lower() == "none" else int(s),
        action="append",
        default=None,
        help="repeatable; e.g. --split-ntime 50 --split-ntime None",
    )
    args = ap.parse_args()
    configs = args.split_ntime or [None, 50]
    ns = [5_000, 20_000, 50_000]
    p = 60

    # warm numba at small n
    print("warming numba at n=500...", flush=True)
    _ = _time_fit(500, p, split_ntime=configs[0])
    print("warmup done", flush=True)

    all_rows = []
    for sn in configs:
        walls = []
        for n in ns:
            w = _time_fit(n, p, split_ntime=sn)
            walls.append(w)
            print(f"split_ntime={sn} n={n}: {w:.1f} s", flush=True)
        slope, intercept, t100k = _extrapolate(ns, walls)
        all_rows.append(
            dict(split_ntime=sn, ns=ns, walls=walls, alpha=slope, beta=intercept, t_100k=t100k)
        )

    baseline = all_rows[0]
    out = Path("validation/reports") / f"{time.strftime('%Y-%m-%d')}-split-ntime-fit-bench.md"
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w") as f:
        f.write("# ε perf bench — fit wall vs split_ntime\n\n")
        f.write("gamma-sprint Weibull workload, p=60, 500 trees, n_jobs=-1.\n\n")
        f.write(
            "| split_ntime | n=5k | n=20k | n=50k | alpha | extrap n=100k | speedup vs baseline |\n"
        )
        f.write("|---|---|---|---|---|---|---|\n")
        for r in all_rows:
            speedup = baseline["t_100k"] / r["t_100k"]
            f.write(
                f"| {r['split_ntime']} | {r['walls'][0]:.1f}s | {r['walls'][1]:.1f}s | "
                f"{r['walls'][2]:.1f}s | {r['alpha']:.2f} | {r['t_100k'] / 60:.1f} min | "
                f"{speedup:.2f}x |\n"
            )
        f.write("\n**PRD §6.1 target:** ≤ 3 min at n=100k.\n")
        f.write("**Sprint exit-gate:** ≥ 6x vs split_ntime=None baseline (ship ≥ 4x).\n")
    print(f"wrote {out}", flush=True)


if __name__ == "__main__":
    main()
