"""Isolated micro-bench of the split kernels at n ∈ {5k, 20k}.

ε Day 1 measurement sub-task. Emits a dated report under
validation/reports/ with per-kernel per-n wall time and a go/no-go
assessment for the fused batched kernel.
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np

from crforest._hist_splits import _best_split_in_feature, _node_histograms, find_best_split_hist


def _make_node_inputs(n: int, p: int, seed: int, n_bins: int, n_time_bins: int):
    rng = np.random.default_rng(seed)
    X_binned = rng.integers(0, n_bins, size=(n, p), dtype=np.uint8)
    t_idx = rng.integers(0, n_time_bins, size=n, dtype=np.int32)
    event = rng.integers(0, 3, size=n, dtype=np.int64)
    selected = np.arange(p, dtype=np.int64)
    return X_binned, t_idx, event, selected


def _time_fn(fn, reps: int = 5) -> tuple[float, float]:
    samples = []
    for _ in range(reps):
        t0 = time.perf_counter()
        fn()
        samples.append(time.perf_counter() - t0)
    samples.sort()
    return samples[len(samples) // 2], samples[0]


def main() -> None:
    n_bins = 32
    n_causes = 2
    n_time_bins = 200
    mtry = 8

    # warm numba
    X_binned, t_idx, event, sel = _make_node_inputs(1000, mtry, 0, n_bins, n_time_bins)
    bin_sub = np.ascontiguousarray(X_binned[:, sel])
    event_hist, at_risk = _node_histograms(bin_sub, t_idx, event, n_bins, n_causes, n_time_bins)
    mask = np.ones(n_bins - 1, dtype=np.bool_)
    _best_split_in_feature(event_hist[0], at_risk[0], 1000, 3, mask)
    _ = find_best_split_hist(X_binned, t_idx, event, sel, n_bins, n_causes, n_time_bins, 3)

    rows = []
    for n in (5_000, 20_000):
        X_binned, t_idx, event, sel = _make_node_inputs(n, mtry, 1, n_bins, n_time_bins)
        bin_sub = np.ascontiguousarray(X_binned[:, sel])

        med_hist, min_hist = _time_fn(
            lambda bin_sub=bin_sub, t_idx=t_idx, event=event: _node_histograms(
                bin_sub, t_idx, event, n_bins, n_causes, n_time_bins
            ),
            reps=5,
        )
        event_hist, at_risk = _node_histograms(bin_sub, t_idx, event, n_bins, n_causes, n_time_bins)
        med_split, min_split = _time_fn(
            lambda event_hist=event_hist, at_risk=at_risk, n=n: [
                _best_split_in_feature(event_hist[f], at_risk[f], n, 3, mask) for f in range(mtry)
            ],
            reps=5,
        )
        med_end, min_end = _time_fn(
            lambda X_binned=X_binned, t_idx=t_idx, event=event, sel=sel: find_best_split_hist(
                X_binned, t_idx, event, sel, n_bins, n_causes, n_time_bins, 3
            ),
            reps=5,
        )
        rows.append(
            dict(
                n=n,
                hist_median_ms=med_hist * 1000,
                hist_min_ms=min_hist * 1000,
                split_median_ms=med_split * 1000,
                split_min_ms=min_split * 1000,
                end_to_end_median_ms=med_end * 1000,
                end_to_end_min_ms=min_end * 1000,
                hist_pct=med_hist / med_end * 100,
                split_pct=med_split / med_end * 100,
            )
        )

    out = Path("validation/reports") / f"{time.strftime('%Y-%m-%d')}-split-kernel-bench.md"
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w") as f:
        f.write("# ε Day 1 — split kernel micro-bench (baseline)\n\n")
        f.write(
            f"Setup: p={mtry}, n_bins={n_bins}, n_causes={n_causes}, n_time_bins={n_time_bins}.\n\n"
        )
        f.write(
            "| n | _node_histograms (ms) | _best_split_in_feature x mtry (ms) | find_best_split_hist (ms) | hist % | split % |\n"
        )
        f.write("|---|---|---|---|---|---|\n")
        for r in rows:
            f.write(
                f"| {r['n']} | {r['hist_median_ms']:.2f} | {r['split_median_ms']:.2f} | "
                f"{r['end_to_end_median_ms']:.2f} | {r['hist_pct']:.1f}% | {r['split_pct']:.1f}% |\n"
            )
        f.write(
            "\n**Decision rule.** If `split %` is the dominant term at n=20k (e.g. ≥ 55%), the fused kernel + `split_ntime=30` combo is strongly favored: both levers attack the dominant cost. If `hist %` dominates, time-grid coarsening alone is the higher-value lever for Day 2.\n"
        )
    print(f"wrote {out}")
    for r in rows:
        print(r)


if __name__ == "__main__":
    sys.exit(main())
