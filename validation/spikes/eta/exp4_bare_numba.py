"""η spike — Experiment 4: bare-numba single-tree floor at n=100k.

Two layers of "floor":
  (1) Single-tree wall in current crforest with n_jobs=1 ntree=1.
      Tells us: per-tree wall in the production engine.
  (2) Repeated calls to ``find_best_split_hist_batched`` (an existing
      ``@njit nogil cache`` kernel) on the full n=100k root node. Sums
      across an estimated nodes-per-tree to give a "kernel-only floor"
      lower bound: every microsecond ABOVE this floor is Python /
      orchestration / leaf-CIF / sparse-encoding overhead, not kernel.

Output: logs/exp4_bare_numba.log with:
  - production single-tree wall
  - kernel-only-floor estimate
  - ratio = headroom available to a kernel-rewrite sprint
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

import numpy as np
from _dgp import load

from crforest import CompetingRiskForest
from crforest._binning import apply_bins, fit_bin_edges
from crforest._hist_splits import find_best_split_hist_batched
from crforest._time_grid import coarsen_time_grid, fit_time_grid

LOG_DIR = Path(__file__).parent / "logs"
LOG_DIR.mkdir(exist_ok=True)
LOG_PATH = LOG_DIR / "exp4_bare_numba.log"

N = 100_000
SEED = 0
SPLIT_NTIME = 50
N_BINS = 256
SPLITRULE_CODE = 0  # 0=logrankCR, 1=logrank


def _print(msg: str, fp) -> None:
    print(msg, flush=True)
    fp.write(msg + "\n")
    fp.flush()


def main() -> None:
    fp = open(LOG_PATH, "w")
    _print(f"[exp4] dataset weibull n={N} seed={SEED} p=60", fp)

    X, t, e = load(N, SEED)
    n, p = X.shape

    # Warmup small fit.
    _print("[exp4] warmup (small fit, n_jobs=1, ntree=2)…", fp)
    CompetingRiskForest(
        n_estimators=2,
        min_samples_leaf=15,
        max_features=8,
        nsplit=10,
        splitrule="logrankCR",
        split_ntime=SPLIT_NTIME,
        random_state=0,
        n_jobs=1,
    ).fit(X[:2000], t[:2000], e[:2000])

    # ────────────── (1) Production single-tree wall ──────────────
    _print("\n=== (1) production single-tree wall (n_jobs=1, ntree=1) ===", fp)
    forest = CompetingRiskForest(
        n_estimators=1,
        min_samples_leaf=15,
        max_features=8,
        nsplit=10,
        splitrule="logrankCR",
        split_ntime=SPLIT_NTIME,
        random_state=SEED,
        n_jobs=1,
    )
    t0 = time.perf_counter()
    forest.fit(X, t, e)
    one_tree = time.perf_counter() - t0
    _print(f"[exp4] single-tree wall      = {one_tree:6.2f}s", fp)

    import os

    cpu = os.cpu_count() or 1
    _print(
        f"[exp4] cpu_count = {cpu}; "
        f"projected 100-tree wall (perfect parallel) = {one_tree * 100 / cpu:6.2f}s",
        fp,
    )

    # ────────────── (2) Kernel-only floor ──────────────
    _print("\n=== (2) kernel-only floor (find_best_split_hist_batched on full root) ===", fp)

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
    n_time_bins_split = SPLIT_NTIME
    e_int = e.astype(np.int32)

    n_causes = 2
    REPS = 5

    def _bench(mtry: int, label: str) -> float:
        # X_binned must be (n_node, mtry) per kernel signature
        selected = np.arange(mtry, dtype=np.int64)
        Xb = np.ascontiguousarray(X_binned[:, selected])
        candidate_mask = np.ones((mtry, N_BINS - 1), dtype=np.bool_)
        # warm
        find_best_split_hist_batched(
            Xb[:1000],
            t_idx_split[:1000],
            e_int[:1000],
            N_BINS,
            n_causes,
            n_time_bins_split,
            15,
            SPLITRULE_CODE,
            1,
            candidate_mask,
        )
        t0 = time.perf_counter()
        for _ in range(REPS):
            find_best_split_hist_batched(
                Xb,
                t_idx_split,
                e_int,
                N_BINS,
                n_causes,
                n_time_bins_split,
                15,
                SPLITRULE_CODE,
                1,
                candidate_mask,
            )
        per = (time.perf_counter() - t0) / REPS
        _print(f"[exp4] kernel @ full root, {label:13s} = {per * 1000:7.1f} ms", fp)
        return per

    kernel_full_root_all = _bench(p, "all 60 feat")
    kernel_full_root_mtry = _bench(8, "8 (mtry) feat")

    # Tree-cost projection.
    # Binary tree, min_leaf=15, n=100k. Per-level work ≈ kernel_full_root_mtry
    # because each level processes the same n samples partitioned across nodes
    # (ignoring constant-feature short-circuits). Tree depth ≈ log2(n / min_leaf).
    est_depth = int(np.log2(n / 15)) + 1
    est_kernel_only = kernel_full_root_mtry * est_depth
    _print(f"[exp4] est tree depth = {est_depth}", fp)
    _print(
        f"[exp4] est KERNEL-ONLY single-tree wall  = {est_kernel_only:6.2f}s "
        f"(={kernel_full_root_mtry * 1000:.1f}ms × {est_depth} levels)",
        fp,
    )

    if est_kernel_only > 0:
        headroom = one_tree / est_kernel_only
        _print(f"[exp4] HEADROOM ratio = single_tree_wall / kernel_only = {headroom:.1f}x", fp)
        _print(
            f"       (i.e. {(1 - 1 / headroom) * 100:.0f}% of single-tree wall is non-kernel)", fp
        )

    fp.close()


if __name__ == "__main__":
    main()
