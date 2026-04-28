"""η spike — Experiment 5: njit-ify ``_observed_bins_sorted_ascending``.

Exp1 revealed it owns 37% of fit wall (87s/237s) on n=100k ntree=100,
called ~5M times, currently pure Python. Single most concentrated
hotspot. This experiment monkey-patches in an ``@njit`` version (same
algorithm: bincount + flatnonzero) and re-times.

If the swap shrinks fit wall by ~30%, that's the cheapest possible
v1.0 win — not a kernel rewrite, just a missing decorator equivalent.
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
LOG_PATH = LOG_DIR / "exp5_njit_obs_bins.log"

N = 100_000
NTREE = 100
SEED = 0


@njit(cache=True, nogil=True)
def _observed_bins_njit(column: np.ndarray, n_bins: int) -> np.ndarray:
    counts = np.bincount(column, minlength=n_bins)
    # numba's np.flatnonzero is fine; equivalent to np.nonzero(c)[0]
    return np.flatnonzero(counts).astype(np.int64)


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
    _print(f"[exp5] dataset weibull n={N} seed={SEED} p=60 ntree={NTREE}", fp)

    _warmup()

    # Warm njit kernel
    dummy = np.zeros(1000, dtype=np.uint8)
    _observed_bins_njit(dummy, 256)

    # Baseline
    _print("[exp5] baseline (current main)…", fp)
    base_wall = _fit_once()
    _print(f"[exp5] BASELINE wall = {base_wall:6.2f}s", fp)

    # Patched
    import crforest._hist_splits as hs

    orig = hs._observed_bins_sorted_ascending
    hs._observed_bins_sorted_ascending = _observed_bins_njit
    try:
        _print("[exp5] patched (_observed_bins_sorted_ascending → @njit)…", fp)
        patched_wall = _fit_once()
        _print(f"[exp5] PATCHED  wall = {patched_wall:6.2f}s", fp)
    finally:
        hs._observed_bins_sorted_ascending = orig

    saving = base_wall - patched_wall
    speedup = base_wall / patched_wall if patched_wall > 0 else float("inf")
    _print(f"[exp5] SAVING  = {saving:+6.2f}s ({saving / base_wall * 100:+.1f}% of baseline)", fp)
    _print(f"[exp5] SPEEDUP = {speedup:.2f}x", fp)
    fp.close()


if __name__ == "__main__":
    main()
