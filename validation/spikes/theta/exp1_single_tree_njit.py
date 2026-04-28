"""θ.1 — single-tree njit POC wall on n=100k.

Compares:
  - production single-tree wall (from η.exp4 = 3.0s on this hardware)
  - θ.1 njit-only single-tree wall (this script)
  - kernel-only floor estimate (η.exp4 = 0.03s)

If θ.1 lands ≤0.5s, the C-path kernel rewrite is viable. If it lands
~1s or above, the engineering effort vs payoff is similar to B and we
should pick B as a more conservative scope.
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent / "eta"))
sys.path.insert(0, str(Path(__file__).parent))

from _build_tree_njit import build_tree_njit
from _dgp import load

from crforest._binning import apply_bins, fit_bin_edges
from crforest._time_grid import coarsen_time_grid, fit_time_grid

LOG_DIR = Path(__file__).parent / "logs"
LOG_DIR.mkdir(exist_ok=True)
LOG_PATH = LOG_DIR / "exp1_single_tree_njit.log"

N = 100_000
SEED = 0
SPLIT_NTIME = 50
N_BINS = 256


def _print(msg: str, fp) -> None:
    print(msg, flush=True)
    fp.write(msg + "\n")
    fp.flush()


def main() -> None:
    fp = open(LOG_PATH, "w")
    _print(f"[θ.1] dataset weibull n={N} seed={SEED} p=60", fp)

    X, t, e = load(N, SEED)

    # Prep inputs
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

    n_causes = 2
    _print(f"[θ.1] X_binned shape={X_binned.shape} dtype={X_binned.dtype}", fp)
    _print(f"[θ.1] split_ntime={SPLIT_NTIME} n_causes={n_causes}", fp)

    # Warm njit kernel with a tiny call
    _print("[θ.1] warming njit kernel (small n)...", fp)
    build_tree_njit(
        X_binned[:1000],
        t_idx_split[:1000],
        e_int[:1000],
        n_bins=N_BINS,
        n_causes=n_causes,
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
    _print("[θ.1] warmup done", fp)

    # Time the full n single-tree build
    REPS = 3
    walls = []
    for r in range(REPS):
        t0 = time.perf_counter()
        nodes, ev_counts, _at_risk = build_tree_njit(
            X_binned,
            t_idx_split,
            e_int,
            n_bins=N_BINS,
            n_causes=n_causes,
            n_time_bins=SPLIT_NTIME,
            min_samples_split=30,
            min_samples_leaf=15,
            max_depth=-1,
            mtry=8,
            nsplit=10,
            splitrule_code=0,
            cause=1,
            seed=r,
        )
        wall = time.perf_counter() - t0
        walls.append(wall)
        _print(f"[θ.1] rep {r}: wall={wall:6.3f}s  nodes={len(nodes)}  leaves={len(ev_counts)}", fp)

    median = float(np.median(walls))
    _print(f"\n[θ.1] median single-tree wall = {median:.3f}s", fp)

    # Anchors from η spike
    PROD_SINGLE = 3.01  # η.exp4 single-tree wall
    KERNEL_FLOOR = 0.03  # η.exp4 estimate
    _print(f"[θ.1] vs production single-tree (η.exp4) = {PROD_SINGLE / median:5.1f}x faster", fp)
    _print(f"[θ.1] vs kernel-only floor    (η.exp4) = {median / KERNEL_FLOOR:5.1f}x slower", fp)

    # Decision matrix
    if median <= 0.3:
        _print("[θ.1] DECISION: C is viable (≤0.3s gate hit)", fp)
    elif median <= 1.0:
        _print("[θ.1] DECISION: C marginal — between 0.3-1.0s; B is safer", fp)
    else:
        _print("[θ.1] DECISION: C unviable in this form (>1.0s)", fp)

    fp.close()


if __name__ == "__main__":
    main()
