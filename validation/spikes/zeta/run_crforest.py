"""ζ benchmark — comprisk side: Weibull CR data gen and fit timing.

Usage:
    uv run python run_comprisk.py --step gen
    uv run python run_comprisk.py --step bench
    uv run python run_comprisk.py --self-test
"""

from __future__ import annotations

import argparse
import resource
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

from comprisk import CompetingRiskForest

HERE = Path(__file__).resolve().parent
DATA_DIR = HERE / "data"
TIMINGS_DIR = HERE / "timings"

N_LADDER = [5000, 20000, 50000]
SEEDS = [0, 1, 2]
P = 60
N_INFORMATIVE = 10
BETA_SCALE = 0.5
SHAPE_1 = 1.2
SHAPE_2 = 0.9
CENS_RATE = 0.06


def weibull_cr(n: int, p: int, seed: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Mirror of validation/benches/bench_fit.py weibull_cr (frozen here)."""
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


def _parquet_path(n: int, seed: int) -> Path:
    return DATA_DIR / f"weibull_n{n}_s{seed}.parquet"


def gen_all() -> None:
    DATA_DIR.mkdir(exist_ok=True)
    for n in N_LADDER:
        for s in SEEDS:
            p_path = _parquet_path(n, s)
            if p_path.exists():
                print(f"[gen] skip {p_path.name} (exists)", flush=True)
                continue
            X, t, e = weibull_cr(n, P, s)
            cols = {f"X_{i}": X[:, i] for i in range(P)}
            cols["time"] = t
            cols["event"] = e
            pd.DataFrame(cols).to_parquet(p_path)
            print(f"[gen] wrote {p_path.name} ({n} rows)", flush=True)


def self_test() -> None:
    """Gen determinism: generating the same (n, seed) twice produces identical arrays."""
    X1, t1, e1 = weibull_cr(5000, P, seed=0)
    X2, t2, e2 = weibull_cr(5000, P, seed=0)
    assert np.array_equal(X1, X2), "X differs across calls"
    assert np.array_equal(t1, t2), "time differs across calls"
    assert np.array_equal(e1, e2), "event differs across calls"
    print("[self-test] gen is deterministic on seed — OK", flush=True)


def _peak_rss_mb() -> float:
    """Absolute peak RSS of this process so far, in MB.

    ru_maxrss is a non-decreasing high-water mark, so per-fit deltas collapse
    to zero after the first fit. We record watermark-to-date instead; compare.py
    uses the n=50k average (the largest point in the ladder, so the watermark
    at n=50k reflects the peak memory of that workload).
    """
    raw = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    if sys.platform == "darwin":
        return raw / (1024 * 1024)
    return raw / 1024


def _load(n: int, seed: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    t = pd.read_parquet(_parquet_path(n, seed))
    X = t[[f"X_{i}" for i in range(P)]].to_numpy()
    return X, t["time"].to_numpy(), t["event"].to_numpy()


def _fit_once(n: int, seed: int) -> tuple[float, float, float | None]:
    X, t, e = _load(n, seed)
    forest = CompetingRiskForest(
        n_estimators=500,
        min_samples_leaf=15,
        max_features=8,
        nsplit=10,
        splitrule="logrankCR",
        split_ntime=50,
        random_state=seed,
        n_jobs=-1,
    )
    t0 = time.perf_counter()
    forest.fit(X, t, e)
    fit_wall = time.perf_counter() - t0
    rss_mb = _peak_rss_mb()

    predict_wall: float | None = None
    if n == 50000 and seed == 0:
        t0 = time.perf_counter()
        forest.predict_cif(X)
        predict_wall = time.perf_counter() - t0
    return fit_wall, rss_mb, predict_wall


def bench_all() -> None:
    TIMINGS_DIR.mkdir(exist_ok=True)
    # Warm-up pass to absorb numba JIT. Untimed. Discarded.
    print("[bench] warm-up (n=5000, seed=0)…", flush=True)
    _fit_once(5000, 0)

    rows = []
    for n in N_LADDER:
        for s in SEEDS:
            print(f"[bench] lib=comprisk n={n} seed={s} start", flush=True)
            fit_wall, rss_mb, pred_wall = _fit_once(n, s)
            print(
                f"[bench] lib=comprisk n={n} seed={s} fit_wall={fit_wall:.2f}s "
                f"peak_rss={rss_mb:.1f}MB predict_wall={pred_wall}",
                flush=True,
            )
            rows.append(
                {
                    "n": n,
                    "seed": s,
                    "fit_wall_s": fit_wall,
                    "peak_rss_mb": rss_mb,
                    "predict_wall_s": pred_wall,
                }
            )
    out = TIMINGS_DIR / "comprisk_timings.parquet"
    pd.DataFrame(rows).to_parquet(out)
    print(f"[bench] wrote {out}", flush=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--step", choices=["gen", "bench"])
    ap.add_argument("--self-test", action="store_true")
    args = ap.parse_args()

    if args.self_test:
        self_test()
        return
    if args.step == "gen":
        gen_all()
        return
    if args.step == "bench":
        bench_all()
        return
    ap.print_help()
    sys.exit(2)


if __name__ == "__main__":
    main()
