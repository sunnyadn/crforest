"""κ.exp8b — n=250k cuda variance follow-up.

exp8 saw cuda at n=250k synthetic 100 trees with 2 seeds: 136.46s vs 47.52s
(2.87× spread; CPU std at same cell was 0.9s). One run is an outlier or
the GPU backend is genuinely unstable here.

This script:
  1. One throwaway cuda fit at n=250k to fully warm the GPU memory pool
  2. 5 measured seeds × {cpu, cuda} at n=250k, ntree=100, p=10
  3. Per-fit GPU memory pool used+total bytes (cuda only)

Output: /tmp/gpu_n250k_variance.parquet

Run: ssh win 'export PATH=$HOME/.local/bin:$PATH && cd ~/crforest && \\
       PYTHONUNBUFFERED=1 uv run --extra gpu --extra dev \\
       python -u validation/spikes/kappa/exp8b_gpu_n250k_variance.py \\
       2>&1 | tee /tmp/exp8b_gpu_n250k_variance.log'
"""

from __future__ import annotations

import time as _time
from pathlib import Path

import numpy as np
import pandas as pd

from crforest import CompetingRiskForest

OUT_WALLS = Path("/tmp/gpu_n250k_variance.parquet")
N = 250_000
P = 10
NTREE = 100
SEEDS = [42, 43, 44, 45, 46]
DEVICES = ["cpu", "cuda"]


def make_synthetic(n: int, seed: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    X = rng.standard_normal(size=(n, P))
    beta1 = np.array([0.8, 0.4, -0.3, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    beta2 = np.array([0.0, 0.0, 0.0, -0.5, 0.6, 0.0, 0.0, 0.0, 0.0, 0.0])
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


def gpu_mem_bytes() -> tuple[int, int]:
    try:
        import cupy as cp

        pool = cp.get_default_memory_pool()
        return pool.used_bytes(), pool.total_bytes()
    except Exception:
        return 0, 0


def fit_and_time(
    X: np.ndarray,
    t: np.ndarray,
    e: np.ndarray,
    *,
    device: str,
    seed: int,
) -> dict:
    n_jobs = 1 if device == "cuda" else -1
    f = CompetingRiskForest(
        n_estimators=NTREE,
        n_jobs=n_jobs,
        random_state=seed,
        device=device,
    )
    t0 = _time.perf_counter()
    f.fit(X, t, e)
    if device == "cuda":
        import cupy as cp

        cp.cuda.runtime.deviceSynchronize()
    wall = _time.perf_counter() - t0
    used, total = gpu_mem_bytes()
    return {
        "device": device,
        "seed": seed,
        "wall": wall,
        "effective_device": f._effective_device_,
        "gpu_used_gb": used / 1e9,
        "gpu_total_gb": total / 1e9,
    }


def main() -> None:
    print(f"[gen] synthetic n={N:,} p={P}, seed=20260417 for X (fixed)", flush=True)
    X, t, e = make_synthetic(N, seed=20260417)
    print(
        f"[gen] event distribution: cause0={(e == 0).mean():.1%} cause1={(e == 1).mean():.1%} cause2={(e == 2).mean():.1%}",
        flush=True,
    )

    print("\n[warmup-cuda] one throwaway fit at full n=250k...", flush=True)
    warm = fit_and_time(X, t, e, device="cuda", seed=999)
    print(
        f"  warm wall={warm['wall']:.2f}s pool used={warm['gpu_used_gb']:.2f} GB total={warm['gpu_total_gb']:.2f} GB",
        flush=True,
    )

    rows = []
    for device in DEVICES:
        for seed in SEEDS:
            print(f"\n[fit] device={device} seed={seed}...", flush=True)
            row = fit_and_time(X, t, e, device=device, seed=seed)
            print(
                f"  wall={row['wall']:.2f}s pool used={row['gpu_used_gb']:.2f} GB",
                flush=True,
            )
            rows.append(row)

    df = pd.DataFrame(rows)
    df.to_parquet(OUT_WALLS)
    print(f"\n[dump] {OUT_WALLS} ({len(df)} rows)", flush=True)

    print("\n=== Summary ===", flush=True)
    print(df.to_string(index=False), flush=True)
    print("\nPer-device:", flush=True)
    agg = df.groupby("device")["wall"].agg(["count", "mean", "std", "min", "max"])
    print(agg.to_string(), flush=True)


if __name__ == "__main__":
    main()
