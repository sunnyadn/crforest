"""Tests for the nogil numba tree-descent kernel.

Verifies ``predict_leaf_indices`` produces identical leaf indices to a
pure-NumPy reference descent, across both dtype specializations
(reference-mode float64 thresholds and histogram-mode uint8 binned
features + int64 bin-index splits).
"""

from __future__ import annotations

import numpy as np

from comprisk._binning import apply_bins, fit_bin_edges
from comprisk._hist_tree import _flatten_tree_hist, build_tree_hist
from comprisk._time_grid import fit_time_grid
from comprisk._tree import _flatten_tree, build_tree
from comprisk._tree_flat import FlatTree, predict_leaf_indices


def _numpy_descend(flat: FlatTree, X: np.ndarray) -> np.ndarray:
    """Pure-NumPy reference descent used as the test oracle.

    Returns the compact leaf index (via ``leaf_idx_of_node``) to match the
    post-δ.2 kernel contract.
    """
    n_samples = X.shape[0]
    sample_arange = np.arange(n_samples)
    current = np.zeros(n_samples, dtype=np.int64)
    while True:
        leaves = flat.is_leaf_flags[current]
        if leaves.all():
            break
        go_left = X[sample_arange, flat.features[current]] <= flat.split_values[current]
        next_nodes = np.where(go_left, flat.left_children[current], flat.right_children[current])
        current = np.where(leaves, current, next_nodes)
    return flat.leaf_idx_of_node[current]


def _small_dataset(seed: int = 0):
    rng = np.random.RandomState(seed)
    n, d = 200, 4
    X = rng.randn(n, d)
    time = rng.exponential(scale=1.0, size=n)
    event = rng.randint(0, 3, size=n).astype(np.int64)
    return X, time, event


def test_descend_nogil_matches_numpy_reference_mode():
    X, time, event = _small_dataset()
    unique_times = np.sort(np.unique(time))
    tree = build_tree(
        X,
        time,
        event,
        n_causes=2,
        max_depth=5,
        min_samples_split=4,
        min_samples_leaf=2,
        unique_times=unique_times,
        max_features=None,
        rng=None,
        splitrule="logrankCR",
        cause=1,
        cause_weights=None,
        nsplit=0,
    )
    flat = _flatten_tree(tree)
    np.testing.assert_array_equal(predict_leaf_indices(flat, X), _numpy_descend(flat, X))


def test_descend_nogil_matches_numpy_histogram_mode():
    X, time, event = _small_dataset()
    bin_edges = fit_bin_edges(X, n_bins=64)
    X_binned = apply_bins(X, bin_edges)
    time_grid = fit_time_grid(time, event, max_points=50)
    n_time_bins = len(time_grid)
    t_idx = np.clip(
        np.searchsorted(time_grid, time, side="right") - 1,
        0,
        n_time_bins - 1,
    ).astype(np.int32)
    tree = build_tree_hist(
        X_binned,
        t_idx,
        event,
        n_causes=2,
        n_bins=64,
        n_time_bins=n_time_bins,
        max_depth=5,
        min_samples_split=4,
        min_samples_leaf=2,
        max_features=None,
        rng=None,
        splitrule="logrankCR",
        cause=1,
        nsplit=0,
    )
    flat = _flatten_tree_hist(tree)
    np.testing.assert_array_equal(
        predict_leaf_indices(flat, X_binned), _numpy_descend(flat, X_binned)
    )


def test_descend_nogil_handles_root_only_tree():
    """Tree with only a root leaf should return all zeros."""
    flat = FlatTree(
        features=np.zeros(1, dtype=np.int64),
        split_values=np.zeros(1, dtype=np.float64),
        left_children=np.zeros(1, dtype=np.int64),
        right_children=np.zeros(1, dtype=np.int64),
        is_leaf_flags=np.ones(1, dtype=bool),
        leaf_table=np.zeros((1, 2, 3), dtype=np.float64),
        leaf_idx_of_node=np.zeros(1, dtype=np.int64),
    )
    X = np.random.RandomState(0).randn(10, 3)
    np.testing.assert_array_equal(predict_leaf_indices(flat, X), np.zeros(10, dtype=np.int64))
