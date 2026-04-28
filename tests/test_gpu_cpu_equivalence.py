"""Cross-equivalence between cpu and cuda backends — seed-variance framing.

CPU and GPU twins use different RNG streams (numba MT inside njit vs numpy
default_rng on CPU side) and different node orderings (DFS vs BFS), so the
cross-backend gap on raw CIF is dominated by ensemble variance from RNG-stream
divergence. We test the invariant that *actually* matters: cross-backend gap
must not exceed same-backend seed-shuffle gap by more than a small margin.

A future regression that breaks numerical parity between backends would
inflate the cross gap above the same-backend floor — caught here. RNG-stream
differences alone do not move the cross gap relative to the same-backend
floor.

Pinned 2026-04-26 on RTX 5070 Ti, cupy 14.x. Re-measure with
validation/spikes/iota/exp2_measure_gpu_cpu_gap.py if either backend's
algorithm changes materially.
"""

from __future__ import annotations

import numpy as np
import pytest

pytestmark = pytest.mark.gpu


def _run(device, seed, X, time, event):
    from crforest import CompetingRiskForest

    f = CompetingRiskForest(
        n_estimators=20,
        device=device,
        random_state=seed,
        n_jobs=1,
    ).fit(X, time, event)
    return f.predict_cif(X[:200])


def test_cpu_cuda_gap_within_seed_variance_margin():
    rng = np.random.default_rng(0)
    n, p = 100_000, 8
    X = rng.uniform(size=(n, p))
    time = rng.uniform(0.1, 10.0, n)
    event = rng.integers(0, 3, n)

    cif_cpu_0 = _run("cpu", 0, X, time, event)
    cif_cpu_1 = _run("cpu", 1, X, time, event)
    cif_gpu_0 = _run("cuda", 0, X, time, event)
    cif_gpu_1 = _run("cuda", 1, X, time, event)

    cpu_seed_shift = float(np.max(np.abs(cif_cpu_0 - cif_cpu_1)))
    gpu_seed_shift = float(np.max(np.abs(cif_gpu_0 - cif_gpu_1)))
    cross_gap = float(np.max(np.abs(cif_cpu_0 - cif_gpu_0)))
    same_backend_floor = max(cpu_seed_shift, gpu_seed_shift)

    # Margin: cross-backend gap must not exceed same-backend floor by more than 50%.
    # Calibrated against measurement (cpu_seed=0.221, gpu_seed=0.206, cross=0.226 → ratio 1.02).
    margin = 1.5
    assert cross_gap <= same_backend_floor * margin, (
        f"cpu↔cuda cross gap {cross_gap:.3e} exceeds "
        f"{margin}x same-backend floor {same_backend_floor:.3e} "
        f"(cpu seed shift {cpu_seed_shift:.3e}, gpu seed shift {gpu_seed_shift:.3e}). "
        "One of the backends has drifted algorithmically."
    )
