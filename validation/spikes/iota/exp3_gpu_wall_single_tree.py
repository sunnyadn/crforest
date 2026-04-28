"""Iota.exp3 — single-tree wall on GPU at n=100k.

Spec exit gate: <= 100 ms on RTX 5070 Ti.
"""

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
    f = CompetingRiskForest(
        n_estimators=1,
        device="cuda",
        random_state=0,
    ).fit(X[:5000], t[:5000], e[:5000])

    # Measure single-tree wall
    walls = []
    for k in range(5):
        t0 = time.perf_counter()
        CompetingRiskForest(
            n_estimators=1,
            device="cuda",
            random_state=k,
        ).fit(X, t, e)
        cp.cuda.runtime.deviceSynchronize()
        walls.append(time.perf_counter() - t0)
    median = sorted(walls)[len(walls) // 2]
    print(f"single-tree walls: {[f'{w * 1000:.1f}' for w in walls]} ms", flush=True)
    print(f"median: {median * 1000:.1f} ms", flush=True)
    print("target: 100 ms", flush=True)
    print(f"GATE:    {'PASS' if median <= 0.100 else 'FAIL'}", flush=True)

    # Memory headroom: peak allocated bytes
    pool = cp.get_default_memory_pool()
    print(f"peak GPU mem: {pool.used_bytes() / 1e9:.2f} GB", flush=True)


if __name__ == "__main__":
    main()
