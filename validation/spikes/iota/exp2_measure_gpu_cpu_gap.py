"""Iota.exp2 — pin cpu↔cuda equivalence threshold via same-backend seed variance.

CPU and GPU twins use different RNG streams (numba MT inside njit vs numpy
default_rng on CPU side) and different node orderings (DFS vs BFS), so we
compare predictions on a held-out X. The cross-backend gap is dominated by
ensemble variance from RNG-stream divergence, not algorithmic difference.

The right invariant is: cross-backend gap should be no worse than same-backend
seed-shuffle gap (with a small margin). A future regression that breaks
backend parity would inflate the cross gap above the same-backend floor.
"""

from __future__ import annotations

import numpy as np


def _max_diff(a, b):
    return float(np.max(np.abs(a - b)))


def _run(device, seed):
    from comprisk import CompetingRiskForest

    rng = np.random.default_rng(0)
    n, p = 100_000, 8
    X = rng.uniform(size=(n, p))
    time = rng.uniform(0.1, 10.0, n)
    event = rng.integers(0, 3, n)

    f = CompetingRiskForest(
        n_estimators=20,
        device=device,
        random_state=seed,
        n_jobs=1,
    ).fit(X, time, event)
    return f.predict_cif(X[:200])


def main():
    cif_cpu_0 = _run("cpu", 0)
    cif_cpu_1 = _run("cpu", 1)
    cif_gpu_0 = _run("cuda", 0)
    cif_gpu_1 = _run("cuda", 1)

    rows = [
        ("cpu(0) vs cpu(0)  [sanity zero]", cif_cpu_0, cif_cpu_0),
        ("cpu(0) vs cpu(1)  [cpu seed shift]", cif_cpu_0, cif_cpu_1),
        ("cuda(0) vs cuda(1) [gpu seed shift]", cif_gpu_0, cif_gpu_1),
        ("cpu(0) vs cuda(0)  [cross]", cif_cpu_0, cif_gpu_0),
    ]
    print(f"{'comparison':40} {'max':>10}  {'median':>10}  {'p95':>10}", flush=True)
    for label, a, b in rows:
        diff = np.abs(a - b)
        max_d = float(np.max(diff))
        med_d = float(np.median(diff))
        p95_d = float(np.quantile(diff, 0.95))
        print(f"{label:40} {max_d:.3e}  {med_d:.3e}  {p95_d:.3e}", flush=True)


if __name__ == "__main__":
    main()
