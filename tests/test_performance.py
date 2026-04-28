"""Performance gate: default mode >=2x faster than reference on PBC.

This is a regression detector, not a stretch-goal benchmark. PBC is
small (n=276); per-node histogram allocation overhead eats into the
speedup. On larger datasets (e.g. synthetic n=10k, PRD target workload
n=100k), the histogram advantage grows well past 5x. Keep the per-
commit gate at a value that comfortably passes current implementation
and catches genuine regressions (a broken default path typically slows
to 1x or worse).
"""

import time as time_mod

import pytest
from validation.datasets import load as load_dataset

from crforest.forest import CompetingRiskForest


def _fit_time(X, time_arr, event, mode: str, n_trees: int, n_jobs: int = 1) -> float:
    t0 = time_mod.perf_counter()
    # Pin cpu: defensive anchor against a future v1.1 auto→cuda flip
    # (cuda forces n_jobs=1 and would collapse this speedup).
    CompetingRiskForest(
        n_estimators=n_trees,
        mode=mode,
        random_state=0,
        n_jobs=n_jobs,
        device="cpu",
    ).fit(X, time_arr, event)
    return time_mod.perf_counter() - t0


@pytest.mark.perf
def test_default_mode_faster_than_reference_pbc():
    X, time_arr, event = load_dataset("pbc")

    # Warm-up JIT for numba kernels so the first-call compile cost is
    # excluded from the measured default fit.
    CompetingRiskForest(n_estimators=5, mode="default", random_state=0, n_jobs=1, device="cpu").fit(
        X, time_arr, event
    )

    t_ref = _fit_time(X, time_arr, event, mode="reference", n_trees=100)
    t_def = _fit_time(X, time_arr, event, mode="default", n_trees=100)
    speedup = t_ref / t_def

    assert speedup >= 2.0, (
        f"default mode speedup = {speedup:.2f}x "
        f"(t_ref={t_ref:.2f}s, t_def={t_def:.2f}s; expected ≥2x)"
    )


@pytest.mark.slow
@pytest.mark.perf
def test_parallel_speedup_default_mode_pbc():
    """Opt-in gate: n_jobs=2 is at least 1.3x faster than n_jobs=1 on PBC
    default-mode fit. Slow-marked so it only runs under ``pytest -m slow``;
    threading speedup is noisy even on multi-core hardware under load.
    The 1.3x threshold catches regressions like an accidental revert of
    ``nogil=True`` on the histogram kernels (which would collapse speedup
    to ~1.0x) while tolerating normal measurement variance."""
    X, time_arr, event = load_dataset("pbc")

    # Warm JIT so compile cost is excluded from both measurements.
    CompetingRiskForest(n_estimators=5, mode="default", random_state=0, n_jobs=1, device="cpu").fit(
        X, time_arr, event
    )

    t1 = _fit_time(X, time_arr, event, mode="default", n_trees=200, n_jobs=1)
    t2 = _fit_time(X, time_arr, event, mode="default", n_trees=200, n_jobs=2)
    speedup = t1 / t2

    assert speedup >= 1.3, (
        f"parallel speedup = {speedup:.2f}x "
        f"(t_serial={t1:.2f}s, t_parallel={t2:.2f}s; expected ≥1.3x on 2 threads)"
    )
