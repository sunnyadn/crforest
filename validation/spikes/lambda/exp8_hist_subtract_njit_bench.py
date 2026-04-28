"""λ.exp8 — njit hist-subtract vs from-scratch micro-benchmark.

exp7 validated bit-equivalence in pure numpy. This spike measures the
actual njit speedup on real-shape histograms (real CHF: p=58, n_bins=32,
n_causes=2, n_time_bins=50, varying n_node from 100 to 50_000).

Compares:
  (a) from-scratch: iterate n_node samples × p features, accumulate
      uint32 atomics. O(n × p) integer adds.
  (b) subtract: H_child = H_parent - H_sibling. O(p × n_bins × n_causes
      × n_time_bins) integer subtractions, INDEPENDENT of n.

For real CHF level-d nodes (n_node ranging 5k → 50 → 5), subtract path
is O(constant) regardless of n_node, while from-scratch grows linearly.
At n_node=10_000 we expect subtract to be 100×+ faster.

Run: ssh win 'export PATH=$HOME/.local/bin:$PATH && cd ~/crforest && \\
       PYTHONUNBUFFERED=1 uv run --extra dev \\
       python -u validation/spikes/lambda/exp8_hist_subtract_njit_bench.py'
(or run on Mac since no GPU needed)
"""

from __future__ import annotations

import time as _time

import numpy as np
from numba import njit


@njit(cache=True, nogil=True)
def hist_from_scratch_njit(
    bin_sub: np.ndarray,  # (n_node, mtry) uint8
    t_idx: np.ndarray,  # (n_node,) int32
    event: np.ndarray,  # (n_node,) int64
    n_bins: int,
    n_causes: int,
    n_time_bins: int,
):
    """Mirror of find_best_split_hist_batched's histogram phase."""
    n_node, mtry = bin_sub.shape
    event_hist = np.zeros((mtry, n_bins, n_causes, n_time_bins), dtype=np.uint32)
    n_at = np.zeros((mtry, n_bins, n_time_bins), dtype=np.uint32)
    for i in range(n_node):
        t = t_idx[i]
        e = event[i]
        for f in range(mtry):
            b = bin_sub[i, f]
            n_at[f, b, t] += 1
            if e > 0:
                event_hist[f, b, e - 1, t] += 1
    return event_hist, n_at


@njit(cache=True, nogil=True)
def hist_subtract_njit(
    parent_eh: np.ndarray,  # (mtry, n_bins, n_causes, n_time_bins) uint32
    parent_nat: np.ndarray,  # (mtry, n_bins, n_time_bins) uint32
    sibling_eh: np.ndarray,
    sibling_nat: np.ndarray,
):
    """child_eh = parent_eh - sibling_eh (uint32 element-wise).
    Same for n_at. Output dtype matches input."""
    eh_out = np.empty_like(parent_eh)
    nat_out = np.empty_like(parent_nat)
    mtry, n_bins, n_causes, n_time_bins = parent_eh.shape
    for f in range(mtry):
        for b in range(n_bins):
            for t in range(n_time_bins):
                nat_out[f, b, t] = parent_nat[f, b, t] - sibling_nat[f, b, t]
                for k in range(n_causes):
                    eh_out[f, b, k, t] = parent_eh[f, b, k, t] - sibling_eh[f, b, k, t]
    return eh_out, nat_out


def make_split(
    n_total: int,
    n_left_frac: float,
    mtry: int,
    n_bins: int,
    n_causes: int,
    n_time_bins: int,
    seed: int,
):
    rng = np.random.default_rng(seed)
    n_left = int(n_total * n_left_frac)
    bin_sub = rng.integers(0, n_bins, size=(n_total, mtry), dtype=np.uint8)
    t_idx = rng.integers(0, n_time_bins, size=n_total, dtype=np.int32)
    event = rng.integers(0, n_causes + 1, size=n_total, dtype=np.int64)
    return bin_sub, t_idx, event, n_left


def main() -> None:
    # Real CHF shape parameters.
    mtry = 8  # sqrt(58) ≈ 7.6
    n_bins = 32
    n_causes = 2
    n_time_bins = 50

    # Warm njit compile.
    print("[warmup] njit compile...", flush=True)
    bw, tw, ew, _ = make_split(100, 0.5, mtry, n_bins, n_causes, n_time_bins, 0)
    eh, nat = hist_from_scratch_njit(bw, tw, ew, n_bins, n_causes, n_time_bins)
    eh_l, nat_l = hist_from_scratch_njit(bw[:50], tw[:50], ew[:50], n_bins, n_causes, n_time_bins)
    _ = hist_subtract_njit(eh, nat, eh_l, nat_l)
    print("[warmup] done", flush=True)

    # n_node sweep — what we'd see across tree levels for real CHF (n=75k, depth ~15).
    n_nodes = [100, 1_000, 5_000, 10_000, 25_000, 50_000]
    n_left_fracs = [0.5, 0.2, 0.5, 0.1, 0.5, 0.5]  # vary balance

    print(
        f"\n{'n_node':>10} {'frac':>6} {'fresh_us':>10} {'sub_us':>10} {'speedup':>10} {'bit_eq':>8}",
        flush=True,
    )
    print("=" * 60, flush=True)

    for n_total, frac in zip(n_nodes, n_left_fracs, strict=False):
        bs, ti, ev, n_left = make_split(n_total, frac, mtry, n_bins, n_causes, n_time_bins, 1)
        # Compute parent + left from scratch (the "would have computed anyway" path).
        eh_p, nat_p = hist_from_scratch_njit(bs, ti, ev, n_bins, n_causes, n_time_bins)
        eh_l, nat_l = hist_from_scratch_njit(
            bs[:n_left], ti[:n_left], ev[:n_left], n_bins, n_causes, n_time_bins
        )

        # Time: from-scratch for the larger child (right = n_total - n_left)
        # vs subtract.
        right_bs = bs[n_left:]
        right_ti = ti[n_left:]
        right_ev = ev[n_left:]

        # Median of 5 timed runs each.
        fresh_walls = []
        for _ in range(5):
            t0 = _time.perf_counter()
            eh_r_fresh, nat_r_fresh = hist_from_scratch_njit(
                right_bs, right_ti, right_ev, n_bins, n_causes, n_time_bins
            )
            fresh_walls.append(_time.perf_counter() - t0)
        fresh_us = float(np.median(fresh_walls)) * 1e6

        sub_walls = []
        for _ in range(5):
            t0 = _time.perf_counter()
            eh_r_sub, nat_r_sub = hist_subtract_njit(eh_p, nat_p, eh_l, nat_l)
            sub_walls.append(_time.perf_counter() - t0)
        sub_us = float(np.median(sub_walls)) * 1e6

        speedup = fresh_us / sub_us if sub_us > 0 else float("inf")
        bit_eq = bool(
            np.array_equal(eh_r_fresh, eh_r_sub) and np.array_equal(nat_r_fresh, nat_r_sub)
        )

        print(
            f"{n_total:>10} {frac:>6.1f} {fresh_us:>10.1f} {sub_us:>10.1f} {speedup:>10.1f} {bit_eq!s:>8}",
            flush=True,
        )

    print("\nDecode:", flush=True)
    print("  fresh_us = njit from-scratch histogram for one child (the larger half)", flush=True)
    print("  sub_us   = njit uint32 subtraction parent - sibling", flush=True)
    print("  speedup  = how much faster subtract is on the larger child", flush=True)
    print("\nObservations:", flush=True)
    print(
        "  - subtract is O(mtry * n_bins * n_causes * n_time_bins) = constant in n_node",
        flush=True,
    )
    print(
        "  - fresh is O(n_node * mtry); ~linear in n_node",
        flush=True,
    )
    print(
        "  - speedup grows linearly with n_node; root-level levels see biggest wins",
        flush=True,
    )


if __name__ == "__main__":
    main()
