"""Iota.exp5 — formal 100-tree ensemble wall measurement at n=100k on RTX 5070 Ti.

Final benchmark for Plan 2's GPU sprint (post-9d.4). 5 reps for variance signal.
Comparison baseline: Plan 1 CPU at n=100k, n_jobs=10 = 87.7s (memory record).
"""

from __future__ import annotations

import time

import numpy as np


def main():
    import cupy as cp

    from crforest import CompetingRiskForest

    rng = np.random.default_rng(0)
    n, p = 100_000, 8
    X = rng.uniform(size=(n, p))
    t = rng.exponential(1.0, n) + 0.1
    e = rng.integers(0, 3, n)

    # Warm
    CompetingRiskForest(
        n_estimators=4,
        device="cuda",
        random_state=0,
    ).fit(X[:5000], t[:5000], e[:5000])

    walls = []
    for k in range(5):
        t0 = time.perf_counter()
        CompetingRiskForest(
            n_estimators=100,
            device="cuda",
            random_state=k,
        ).fit(X, t, e)
        cp.cuda.runtime.deviceSynchronize()
        walls.append(time.perf_counter() - t0)

    walls_sorted = sorted(walls)
    median = walls_sorted[len(walls) // 2]
    minimum = walls_sorted[0]
    maximum = walls_sorted[-1]
    print(f"100-tree walls: {[f'{w:.2f}' for w in walls]} s", flush=True)
    print(f"min:    {minimum:.2f} s", flush=True)
    print(f"median: {median:.2f} s", flush=True)
    print(f"max:    {maximum:.2f} s", flush=True)
    print("Plan 1 CPU baseline: 87.7 s (n_jobs=10)", flush=True)
    print(f"Speedup vs Plan 1 CPU (median): {87.7 / median:.2f}x", flush=True)

    pool = cp.get_default_memory_pool()
    print(
        f"peak GPU mem (used_bytes after final fit): {pool.used_bytes() / 1e9:.2f} GB", flush=True
    )
    print(f"pool total_bytes: {pool.total_bytes() / 1e9:.2f} GB", flush=True)


if __name__ == "__main__":
    main()
