"""κ.exp8 — GPU vs CPU scaling sweep on win.

Goal: find the (ntree, n) where cuda backend stops being slower than cpu
backend on this canonical machine (i7-14700K 28-thread + RTX 5070 Ti).

Background: κ.exp4d showed cuda 25.57s vs cpu 22.54s at real CHF
n=75k × 100 trees — GPU loses. iota.exp5 showed cuda 16.6s vs cpu
87.7s at synthetic n=100k × 100 trees — GPU wins by 5.29×.
The gap between these two is what we're mapping out.

Two sweeps:
  A. Real CHF (n=75k, p=58) × ntree ∈ {100, 200, 500} × {cpu, cuda} × 2 seeds
  B. Synthetic competing-risks (p=10) × n ∈ {100k, 250k} × ntree=100
                                       × {cpu, cuda} × 2 seeds

Output: /tmp/gpu_scaling_walls.parquet with one row per fit.

Run: ssh win 'export PATH=$HOME/.local/bin:$PATH && cd ~/crforest && \\
       PYTHONUNBUFFERED=1 uv run --extra gpu --extra dev \\
       python -u validation/spikes/kappa/exp8_gpu_scaling.py \\
       2>&1 | tee /tmp/exp8_gpu_scaling.log'
"""

from __future__ import annotations

import time as _time
from pathlib import Path

import numpy as np
import pandas as pd

from crforest import CompetingRiskForest

CLEAN_PARQUET = Path("/tmp/chf_2012_clean.parquet")
TRAIN_IDX = Path("/tmp/chf_2012_train_idx.txt")
OUT_WALLS = Path("/tmp/gpu_scaling_walls.parquet")

REAL_NTREES = [100, 200, 500]
SYNTH_NS = [100_000, 250_000]
SEEDS = [42, 43]
DEVICES = ["cpu", "cuda"]


def make_synthetic(n: int, seed: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """3-state competing-risks DGP, p=10 (5 informative + 5 noise)."""
    rng = np.random.default_rng(seed)
    p = 10
    X = rng.standard_normal(size=(n, p))
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


def load_real() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    df = pd.read_parquet(CLEAN_PARQUET)
    train_idx = np.loadtxt(TRAIN_IDX, dtype=np.int64)
    feature_cols = [c for c in df.columns if c not in ("time", "status")]
    X = df[feature_cols].to_numpy(dtype=np.float64)
    t = df["time"].to_numpy(dtype=np.float64)
    e = df["status"].to_numpy(dtype=np.int64)
    return X[train_idx], t[train_idx], e[train_idx]


def fit_one(
    X: np.ndarray,
    t: np.ndarray,
    e: np.ndarray,
    *,
    n_estimators: int,
    device: str,
    seed: int,
) -> tuple[float, str]:
    n_jobs = 1 if device == "cuda" else -1
    f = CompetingRiskForest(
        n_estimators=n_estimators,
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
    return wall, f._effective_device_


def warmup_cuda(X: np.ndarray, t: np.ndarray, e: np.ndarray) -> None:
    print("[warmup] cuda compile + first-fit on 5k slice...", flush=True)
    fit_one(X[:5000], t[:5000], e[:5000], n_estimators=4, device="cuda", seed=0)


def main() -> None:
    rows = []

    print("\n=== Sweep A: real CHF, ntree sweep ===", flush=True)
    Xr, tr, er = load_real()
    print(f"[real] n={len(Xr):,} p={Xr.shape[1]}", flush=True)

    print("[warmup] cuda...", flush=True)
    warmup_cuda(Xr, tr, er)

    for ntree in REAL_NTREES:
        for device in DEVICES:
            for seed in SEEDS:
                print(
                    f"[real] ntree={ntree} device={device} seed={seed} fitting...",
                    flush=True,
                )
                wall, eff = fit_one(
                    Xr,
                    tr,
                    er,
                    n_estimators=ntree,
                    device=device,
                    seed=seed,
                )
                print(f"  wall={wall:.2f}s effective={eff}", flush=True)
                rows.append(
                    {
                        "sweep": "real_chf",
                        "n": len(Xr),
                        "p": Xr.shape[1],
                        "ntree": ntree,
                        "device": device,
                        "seed": seed,
                        "wall": wall,
                        "effective_device": eff,
                    }
                )

    print("\n=== Sweep B: synthetic, n sweep at ntree=100 ===", flush=True)
    for n in SYNTH_NS:
        Xs, ts, es = make_synthetic(n, seed=20260417)
        print(f"[synth] n={n:,} p={Xs.shape[1]}", flush=True)
        for device in DEVICES:
            for seed in SEEDS:
                print(
                    f"[synth] n={n} device={device} seed={seed} fitting...",
                    flush=True,
                )
                wall, eff = fit_one(
                    Xs,
                    ts,
                    es,
                    n_estimators=100,
                    device=device,
                    seed=seed,
                )
                print(f"  wall={wall:.2f}s effective={eff}", flush=True)
                rows.append(
                    {
                        "sweep": "synth",
                        "n": n,
                        "p": Xs.shape[1],
                        "ntree": 100,
                        "device": device,
                        "seed": seed,
                        "wall": wall,
                        "effective_device": eff,
                    }
                )

    df = pd.DataFrame(rows)
    df.to_parquet(OUT_WALLS)
    print(f"\n[dump] {OUT_WALLS} ({len(df)} rows)", flush=True)

    print("\n=== Summary: cuda/cpu wall ratio (>1.0 = GPU slower) ===", flush=True)
    agg = (
        df.groupby(["sweep", "n", "ntree", "device"])["wall"].agg(["mean", "std"]).unstack("device")
    )
    agg["ratio_cuda_over_cpu"] = agg[("mean", "cuda")] / agg[("mean", "cpu")]
    print(agg.to_string(), flush=True)


if __name__ == "__main__":
    main()
