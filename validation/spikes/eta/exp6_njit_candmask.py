"""η spike — Experiment 6: njit-fy the candidate-mask Python loop.

Findings from exp1/exp4:
  - 99% of single-tree wall is non-kernel
  - parallel efficiency = 17% (181s vs 30s perfect-parallel at 10 cores)
  - top hot spot: 5M Python-side calls to ``_observed_bins_sorted_ascending``
    inside ``find_best_split_hist``'s batched-path nsplit loop

This experiment monkey-patches ``find_best_split_hist`` to build the
candidate mask in ONE njit call per node (instead of mtry Python-loop
iterations × 5M total calls).

Note: uses numba's ``np.random.seed(seed)`` per node — RNG draws will
NOT bit-match numpy MT19937, so this is a perf-only test, not a
correctness preservation. The split decisions and tree structure differ;
fit wall is still a fair comparison because the per-node algorithmic
work is the same shape.
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np
from numba import njit

sys.path.insert(0, str(Path(__file__).parent))

from _dgp import load

LOG_DIR = Path(__file__).parent / "logs"
LOG_DIR.mkdir(exist_ok=True)
LOG_PATH = LOG_DIR / "exp6_njit_candmask.log"

N = 100_000
NTREE = 100
SEED = 0


@njit(cache=True, nogil=True)
def _build_cand_mask_njit(bin_sub, n_bins, nsplit, seed):
    """Build (mtry, n_bins-1) candidate mask in one njit call.

    Replaces the Python ``for f in range(mtry):`` loop in
    ``find_best_split_hist``'s use_batched path. Same algorithm:
      1. observed bins per feature (bincount + walk for nonzero)
      2. exclude max bin
      3. SWOR draw of k = min(nsplit, n_valid) candidates
    """
    np.random.seed(seed)
    n_node, mtry = bin_sub.shape
    cand_mask = np.zeros((mtry, n_bins - 1), dtype=np.bool_)
    counts = np.zeros(n_bins, dtype=np.int64)
    observed = np.empty(n_bins, dtype=np.int64)
    for f in range(mtry):
        # bincount inline
        counts[:] = 0
        for i in range(n_node):
            counts[bin_sub[i, f]] += 1
        # walk for nonzero
        n_obs = 0
        for b in range(n_bins):
            if counts[b] > 0:
                observed[n_obs] = b
                n_obs += 1
        if n_obs < 2:
            continue
        n_valid = n_obs - 1
        k = nsplit if nsplit < n_valid else n_valid
        # Fisher-Yates partial SWOR over observed[:n_valid]
        for i in range(k):
            j = i + (np.random.randint(0, n_valid - i) if (n_valid - i) > 0 else 0)
            tmp = observed[i]
            observed[i] = observed[j]
            observed[j] = tmp
            cand_mask[f, observed[i]] = True
    return cand_mask


def _make_patched_find_best_split_hist():
    """Return a Python find_best_split_hist that delegates cand-mask building
    to _build_cand_mask_njit in the use_batched + nsplit>0 path.

    Other paths fall through to the original.
    """
    import crforest._hist_splits as hs

    orig = hs.find_best_split_hist
    batched = hs.find_best_split_hist_batched
    counter = [0]  # per-node monotonic counter for seeding

    def patched(
        X_binned_node,
        time_indices_node,
        event_node,
        selected_features,
        n_bins,
        n_causes,
        n_time_bins,
        min_samples_leaf,
        *,
        splitrule="logrankCR",
        cause=1,
        nsplit=0,
        rng=None,
        use_batched=False,
        skip_nsplit_rng_when_deterministic=False,
    ):
        if not (use_batched and nsplit > 0):
            # Defer to original for any path we don't optimize.
            return orig(
                X_binned_node,
                time_indices_node,
                event_node,
                selected_features,
                n_bins,
                n_causes,
                n_time_bins,
                min_samples_leaf,
                splitrule=splitrule,
                cause=cause,
                nsplit=nsplit,
                rng=rng,
                use_batched=use_batched,
                skip_nsplit_rng_when_deterministic=skip_nsplit_rng_when_deterministic,
            )

        # Fast batched-with-nsplit path.
        bin_sub = np.ascontiguousarray(X_binned_node[:, selected_features])
        # Per-node seed: derive from rng state (cheap proxy).
        seed = (rng.tomaxint() ^ counter[0]) & 0x7FFFFFFF if rng is not None else counter[0]
        counter[0] += 1
        cand_mask = _build_cand_mask_njit(bin_sub, n_bins, nsplit, seed)

        splitrule_code = 0 if splitrule == "logrankCR" else 1
        f_sel, bin_idx, stat = batched(
            bin_sub,
            time_indices_node,
            event_node,
            n_bins,
            n_causes,
            n_time_bins,
            min_samples_leaf,
            splitrule_code,
            cause,
            cand_mask,
        )
        if f_sel == -1:
            return -1, 0, 0.0
        return int(selected_features[f_sel]), int(bin_idx), float(stat)

    return patched, orig


def _print(msg: str, fp) -> None:
    print(msg, flush=True)
    fp.write(msg + "\n")
    fp.flush()


def _fit_once() -> float:
    from crforest import CompetingRiskForest

    X, t, e = load(N, SEED)
    forest = CompetingRiskForest(
        n_estimators=NTREE,
        min_samples_leaf=15,
        max_features=8,
        nsplit=10,
        splitrule="logrankCR",
        split_ntime=50,
        random_state=SEED,
        n_jobs=-1,
    )
    t0 = time.perf_counter()
    forest.fit(X, t, e)
    return time.perf_counter() - t0


def _warmup() -> None:
    from crforest import CompetingRiskForest

    X, t, e = load(N, SEED)
    CompetingRiskForest(
        n_estimators=4,
        min_samples_leaf=15,
        max_features=8,
        nsplit=10,
        splitrule="logrankCR",
        split_ntime=50,
        random_state=0,
        n_jobs=1,
    ).fit(X[:2000], t[:2000], e[:2000])


def main() -> None:
    fp = open(LOG_PATH, "w")
    _print(f"[exp6] dataset weibull n={N} seed={SEED} p=60 ntree={NTREE}", fp)

    _warmup()

    # Warm njit kernel
    dummy = np.zeros((1000, 8), dtype=np.uint8)
    _build_cand_mask_njit(dummy, 256, 10, 1)

    _print("[exp6] baseline (current main)…", fp)
    base = _fit_once()
    _print(f"[exp6] BASELINE wall = {base:6.2f}s", fp)

    patched_fn, orig = _make_patched_find_best_split_hist()
    import crforest._hist_splits as hs
    import crforest._hist_tree as ht  # also imports it

    hs.find_best_split_hist = patched_fn
    ht.find_best_split_hist = patched_fn
    try:
        _print("[exp6] patched (njit cand-mask builder)…", fp)
        patched_wall = _fit_once()
        _print(f"[exp6] PATCHED  wall = {patched_wall:6.2f}s", fp)
    finally:
        hs.find_best_split_hist = orig
        ht.find_best_split_hist = orig

    saving = base - patched_wall
    speedup = base / patched_wall if patched_wall > 0 else float("inf")
    _print(f"[exp6] SAVING  = {saving:+6.2f}s ({saving / base * 100:+.1f}% of baseline)", fp)
    _print(f"[exp6] SPEEDUP = {speedup:.2f}x", fp)
    fp.close()


if __name__ == "__main__":
    main()
