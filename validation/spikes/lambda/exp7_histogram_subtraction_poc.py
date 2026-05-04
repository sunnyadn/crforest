"""λ.exp7 — histogram subtraction trick POC for competing-risks RF.

LightGBM/XGBoost/sklearn HistGradientBoosting all use the histogram-
subtraction trick: after splitting a parent node, compute the histogram
for the smaller child only; the larger child = parent - smaller. Saves
~50% of histogram work per tree level.

Crforest's find_best_split_hist (CPU njit) and histogram_kernel_per_level
(GPU) currently compute every active-node histogram from scratch per
level. Applying the subtraction trick is the cheapest paper-grade
algorithmic optimization available.

POC scope: validate that uint32 (event_hist, at_risk) histograms support
exact subtraction with bit-equivalence to from-scratch computation.
Mathematically obvious (sample partitioning is disjoint, atomicAdd is
commutative integer accumulation), but worth pinning before integration.

Tests:
  1. event_hist[parent] == event_hist[left] + event_hist[right]
  2. n_at[parent] == n_at[left] + n_at[right]
  3. event_hist[right_via_subtract] == event_hist[right_from_scratch]
  4. n_at[right_via_subtract] == n_at[right_from_scratch]
  5. Repeat across 5 splits with varying split bin / sample size

Run: ssh win 'export PATH=$HOME/.local/bin:$PATH && cd ~/comprisk && \\
       PYTHONUNBUFFERED=1 uv run --extra dev \\
       python -u validation/spikes/lambda/exp7_histogram_subtraction_poc.py'
"""

from __future__ import annotations

import sys

import numpy as np


def histogram_from_samples(
    X_binned: np.ndarray,
    t_idx: np.ndarray,
    event: np.ndarray,
    sample_idx: np.ndarray,
    n_bins: int,
    n_causes: int,
    n_time_bins: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Pure-numpy reference: per-(feature, bin, cause, time) event count
    + per-(feature, bin, time) at-risk count. Mirrors the per-node logic
    in histogram_kernel_per_level and find_best_split_hist."""
    p = X_binned.shape[1]
    event_hist = np.zeros((p, n_bins, n_causes, n_time_bins), dtype=np.uint32)
    n_at = np.zeros((p, n_bins, n_time_bins), dtype=np.uint32)
    for s in sample_idx:
        s = int(s)
        for f in range(p):
            b = int(X_binned[s, f])
            tt = int(t_idx[s])
            ev = int(event[s])
            n_at[f, b, tt] += 1
            if 1 <= ev <= n_causes:
                event_hist[f, b, ev - 1, tt] += 1
    return event_hist, n_at


def make_data(n: int, p: int, n_bins: int, n_causes: int, n_time_bins: int, seed: int):
    rng = np.random.default_rng(seed)
    X = rng.integers(0, n_bins, size=(n, p), dtype=np.uint8)
    t_idx = rng.integers(0, n_time_bins, size=n, dtype=np.int32)
    event = rng.integers(0, n_causes + 1, size=n, dtype=np.int32)
    return X, t_idx, event


def run_case(
    case_name: str, X, t_idx, event, *, n_bins, n_causes, n_time_bins, split_feat, split_bin
):
    n = X.shape[0]
    parent_idx = np.arange(n, dtype=np.int64)
    # Split parent at (split_feat, split_bin): left = X[:, split_feat] <= split_bin
    left_mask = X[:, split_feat] <= split_bin
    left_idx = parent_idx[left_mask]
    right_idx = parent_idx[~left_mask]

    parent_eh, parent_nat = histogram_from_samples(
        X, t_idx, event, parent_idx, n_bins, n_causes, n_time_bins
    )
    left_eh, left_nat = histogram_from_samples(
        X, t_idx, event, left_idx, n_bins, n_causes, n_time_bins
    )
    right_eh_scratch, right_nat_scratch = histogram_from_samples(
        X, t_idx, event, right_idx, n_bins, n_causes, n_time_bins
    )

    # Test 1+2: parent == left + right
    sum_eh = (left_eh.astype(np.int64) + right_eh_scratch.astype(np.int64)).astype(np.uint32)
    sum_nat = (left_nat.astype(np.int64) + right_nat_scratch.astype(np.int64)).astype(np.uint32)
    add_ok = np.array_equal(parent_eh, sum_eh) and np.array_equal(parent_nat, sum_nat)

    # Test 3+4: right_via_subtract == right_from_scratch
    right_eh_sub = (parent_eh.astype(np.int64) - left_eh.astype(np.int64)).astype(np.uint32)
    right_nat_sub = (parent_nat.astype(np.int64) - left_nat.astype(np.int64)).astype(np.uint32)
    sub_ok = np.array_equal(right_eh_sub, right_eh_scratch) and np.array_equal(
        right_nat_sub, right_nat_scratch
    )

    n_left = int(left_idx.size)
    n_right = int(right_idx.size)
    smaller = min(n_left, n_right)
    larger = max(n_left, n_right)
    saving = 1 - smaller / max(n, 1)  # fraction of work saved by subtraction
    status = "OK" if (add_ok and sub_ok) else "FAIL"
    print(
        f"  [{case_name}] {status}  parent={n} → left={n_left}, right={n_right}  "
        f"saving={saving:.0%}  smaller_child_only={smaller}",
        flush=True,
    )
    return add_ok and sub_ok


def main() -> None:
    print("Validating uint32 histogram subtraction = bit-equivalent...\n", flush=True)

    # Smaller scenarios for fast iteration; the math is the same at any size.
    n_bins, n_causes, n_time_bins = 16, 2, 8
    cases = [
        # name, n, p, split_feat, split_bin, seed
        ("balanced", 2000, 4, 0, 7, 0),
        ("imbalanced-low", 2000, 4, 0, 1, 1),
        ("imbalanced-high", 2000, 4, 0, 14, 2),
        ("all-left", 1000, 4, 0, 15, 3),  # split_bin>=max → all-left
        ("all-right", 1000, 4, 0, -1, 4),  # split_bin=-1 → all-right (none ≤ -1)
        ("realistic-mid", 5000, 8, 3, 7, 5),
        ("p=58-style", 3000, 58, 22, 12, 6),  # match real CHF feature width
    ]

    all_ok = True
    for name, n, p, sf, sb, seed in cases:
        X, t_idx, event = make_data(n, p, n_bins, n_causes, n_time_bins, seed)
        ok = run_case(
            name,
            X,
            t_idx,
            event,
            n_bins=n_bins,
            n_causes=n_causes,
            n_time_bins=n_time_bins,
            split_feat=sf,
            split_bin=sb,
        )
        all_ok = all_ok and ok

    print(f"\n=== POC: {'PASS' if all_ok else 'FAIL'} ===", flush=True)
    print(
        "\nIntegration plan after POC PASS:\n"
        "  1. Per-node parent_histogram cache: store event_hist + n_at after\n"
        "     a node's split decision so its children can subtract.\n"
        "  2. CPU njit (find_best_split_hist): compute smaller-child histogram\n"
        "     from scratch; recover larger-child via uint32 subtraction.\n"
        "  3. GPU (histogram_kernel_per_level): same logic, with parent\n"
        "     histograms persisted on device across levels.\n"
        "  4. Memory tradeoff: caching ~3-5 MB per active node × ~32 nodes/level\n"
        "     × 15 levels = ~1.5-2.5 GB additional GPU pool. Acceptable on A6000\n"
        "     48GB; tight on 5070 Ti 16GB at large n.\n"
        "  5. Expected per-tree wall reduction: ~30-40% (CPU);\n"
        "     ~10-15% (GPU, since histogram is smaller fraction of GPU wall).",
        flush=True,
    )
    sys.exit(0 if all_ok else 1)


if __name__ == "__main__":
    main()
