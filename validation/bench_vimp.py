"""Benchmark for P3d VIMP: time compute_importance on a large synthetic workload.

Invoke via ``python -m validation bench-vimp ...`` (see validation/__main__.py).
See docs/superpowers/specs/2026-04-21-p3d-permutation-vimp-design.md §5.
"""

from __future__ import annotations

import resource
import sys
import time

import numpy as np

from crforest.forest import CompetingRiskForest
from validation.datasets import load as load_dataset


def _peak_rss_mb() -> float:
    usage = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    # On Linux ru_maxrss is KB; on macOS it's bytes. Normalize to MB.
    if sys.platform == "darwin":
        return usage / (1024 * 1024)
    return usage / 1024


def run(dataset: str, n: int, n_repeats: int, seed: int, n_jobs: int = -1) -> dict:
    if dataset != "synthetic":
        raise ValueError(f"Only 'synthetic' is supported; got {dataset}")
    X_all, time_all, event_all = load_dataset("synthetic")
    if len(X_all) < n:
        raise ValueError(
            f"Synthetic has {len(X_all)} rows; need {n}. "
            f"Pass --n <= {len(X_all)} or regenerate "
            f"validation/data/synthetic.parquet at a larger size "
            f"(see validation/gen_synthetic.py)."
        )
    rng = np.random.RandomState(seed)
    idx = rng.choice(len(X_all), size=n, replace=False)
    X = X_all[idx]
    time_arr = time_all[idx]
    event = event_all[idx]

    # 80/20 train/eval split
    n_eval = n // 5
    perm = rng.permutation(n)
    train_idx, eval_idx = perm[n_eval:], perm[:n_eval]

    t0 = time.perf_counter()
    forest = CompetingRiskForest(
        n_estimators=500, max_depth=15, random_state=seed, n_jobs=n_jobs
    ).fit(X[train_idx], time_arr[train_idx], event[train_idx])
    fit_secs = time.perf_counter() - t0

    y_eval = np.rec.fromarrays([time_arr[eval_idx], event[eval_idx]], names=["time", "event"])

    t0 = time.perf_counter()
    df = forest.compute_importance(X[eval_idx], y_eval, n_repeats=n_repeats, random_state=seed)
    vimp_secs = time.perf_counter() - t0

    return {
        "dataset": dataset,
        "n": n,
        "n_features": X.shape[1],
        "n_trees": 500,
        "n_repeats": n_repeats,
        "n_jobs": n_jobs,
        "fit_secs": fit_secs,
        "vimp_secs": vimp_secs,
        "peak_rss_mb": _peak_rss_mb(),
        "vimp_head": df.head(5).to_dict(orient="records"),
    }
