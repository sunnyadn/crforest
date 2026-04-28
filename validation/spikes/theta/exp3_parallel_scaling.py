"""θ.3 — 100-tree joblib parallel scaling for the njit-everything POC.

Production (η.exp2) parallel efficiency at 10 cores = 11% — n_jobs=10
is SLOWER than n_jobs=5 due to GIL contention from Python-side per-node
work in find_best_split_hist's dispatcher.

θ.1's njit-everything kernel runs entirely inside one nogil call per
tree. If parallelism works, 100 trees should scale near-linearly with
cores.

This experiment runs ntree=100 at n_jobs ∈ {1, 2, 5, 10} and reports
parallel efficiency. Compares headline ensemble wall to production
(η.exp2 baseline at n_jobs=10 = 261s).
"""

from __future__ import annotations

import os
import sys
import time
from pathlib import Path

import numpy as np
from joblib import Parallel, delayed

sys.path.insert(0, str(Path(__file__).parent.parent / "eta"))
sys.path.insert(0, str(Path(__file__).parent))

from _build_tree_njit import build_tree_njit
from _dgp import load

from crforest._binning import apply_bins, fit_bin_edges
from crforest._time_grid import coarsen_time_grid, fit_time_grid

LOG_DIR = Path(__file__).parent / "logs"
LOG_DIR.mkdir(exist_ok=True)
LOG_PATH = LOG_DIR / "exp3_parallel_scaling.log"

N = 100_000
NTREE = 100
SEED = 0
SPLIT_NTIME = 50
N_BINS = 256


def _print(msg: str, fp) -> None:
    print(msg, flush=True)
    fp.write(msg + "\n")
    fp.flush()


def _build_one(args):
    X_binned, t_idx_split, e_int, seed = args
    nodes, ev_counts, _at_risk = build_tree_njit(
        X_binned,
        t_idx_split,
        e_int,
        n_bins=N_BINS,
        n_causes=2,
        n_time_bins=SPLIT_NTIME,
        min_samples_split=30,
        min_samples_leaf=15,
        max_depth=-1,
        mtry=8,
        nsplit=10,
        splitrule_code=0,
        cause=1,
        seed=seed,
    )
    return len(nodes), len(ev_counts)


def main() -> None:
    fp = open(LOG_PATH, "w")
    _print(f"[θ.3] dataset weibull n={N} seed={SEED} p=60 ntree={NTREE}", fp)

    X, t, e = load(N, SEED)

    bin_edges = fit_bin_edges(X, n_bins=N_BINS)
    X_binned = apply_bins(X, bin_edges)
    time_grid_full = fit_time_grid(t, e, max_points=200)
    n_time_bins_full = len(time_grid_full)
    t_idx_full = np.clip(
        np.searchsorted(time_grid_full, t, side="right") - 1,
        0,
        n_time_bins_full - 1,
    ).astype(np.int32)
    full_to_split = coarsen_time_grid(time_grid_full, SPLIT_NTIME)
    t_idx_split = full_to_split[t_idx_full]
    e_int = e.astype(np.int32)

    cpu = os.cpu_count() or 1
    _print(f"[θ.3] cpu_count = {cpu}", fp)

    # Warm
    print("[θ.3] warming njit...", flush=True)
    build_tree_njit(
        X_binned[:1000],
        t_idx_split[:1000],
        e_int[:1000],
        n_bins=N_BINS,
        n_causes=2,
        n_time_bins=SPLIT_NTIME,
        min_samples_split=30,
        min_samples_leaf=15,
        max_depth=-1,
        mtry=8,
        nsplit=10,
        splitrule_code=0,
        cause=1,
        seed=42,
    )

    args_list = [(X_binned, t_idx_split, e_int, s) for s in range(NTREE)]

    walls_by_jobs = {}
    for nj in [1, 2, 5, 10]:
        t0 = time.perf_counter()
        Parallel(n_jobs=nj, prefer="threads")(delayed(_build_one)(a) for a in args_list)
        wall = time.perf_counter() - t0
        walls_by_jobs[nj] = wall
        speedup_vs_serial = walls_by_jobs[1] / wall if nj > 1 else 1.0
        ideal = walls_by_jobs[1] / nj
        eff = ideal / wall * 100
        _print(
            f"  n_jobs={nj:3d}  wall={wall:6.2f}s  speedup={speedup_vs_serial:4.2f}x  "
            f"parallel_eff={eff:4.0f}%",
            fp,
        )

    base = walls_by_jobs[10]
    _print(f"\n[θ.3] best ensemble wall (n_jobs=10) = {base:.2f}s", fp)

    # Comparisons to η baselines
    PROD_NJOBS_10 = 261.4  # η.exp2 baseline n_jobs=10
    PROD_NJOBS_5 = 192.3  # η.exp2 best operating point currently
    EXP6_NJIT_CANDMASK = 140.0  # η.exp6 patched

    _print(
        f"[θ.3] vs production n_jobs=10 ({PROD_NJOBS_10}s) = {PROD_NJOBS_10 / base:5.1f}x faster",
        fp,
    )
    _print(
        f"[θ.3] vs production n_jobs=5  ({PROD_NJOBS_5}s) = {PROD_NJOBS_5 / base:5.1f}x faster", fp
    )
    _print(
        f"[θ.3] vs η.exp6 njit-cand-mask ({EXP6_NJIT_CANDMASK}s) = {EXP6_NJIT_CANDMASK / base:5.1f}x faster",
        fp,
    )

    fp.close()


if __name__ == "__main__":
    main()
