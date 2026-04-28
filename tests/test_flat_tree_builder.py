"""Unit tests for the flat-array tree builder."""

from __future__ import annotations

import numpy as np
import pytest

from crforest._tree_flat import FlatTree


def test_flat_tree_from_arrays_constructs_root_only_leaf():
    # Trivial tree: single leaf at the root.
    features = np.array([0], dtype=np.int64)
    split_values = np.array([0], dtype=np.int64)
    left_children = np.array([0], dtype=np.int64)
    right_children = np.array([0], dtype=np.int64)
    is_leaf_flags = np.array([True])
    leaf_table = np.array([[[0.1, 0.3, 0.5]]], dtype=np.float64)  # (1 leaf, 1 cause, 3 time bins)
    leaf_idx_of_node = np.array([0], dtype=np.int64)

    flat = FlatTree.from_arrays(
        features=features,
        split_values=split_values,
        left_children=left_children,
        right_children=right_children,
        is_leaf_flags=is_leaf_flags,
        leaf_table=leaf_table,
        leaf_idx_of_node=leaf_idx_of_node,
    )

    assert isinstance(flat, FlatTree)
    assert flat.features is features
    assert flat.leaf_table.shape == (1, 1, 3)


def test_flat_tree_from_arrays_validates_shape_consistency():
    # n_nodes=2 in features but is_leaf_flags has 3 entries → mismatch.
    with pytest.raises(ValueError, match="length"):
        FlatTree.from_arrays(
            features=np.array([0, 0], dtype=np.int64),
            split_values=np.array([0, 0], dtype=np.int64),
            left_children=np.array([0, 0], dtype=np.int64),
            right_children=np.array([0, 0], dtype=np.int64),
            is_leaf_flags=np.array([True, False, False]),
            leaf_table=np.empty((1, 1, 1), dtype=np.float64),
            leaf_idx_of_node=np.array([0, -1], dtype=np.int64),
        )


def test_build_flat_tree_root_only_when_n_below_min_split():
    from crforest._flat_tree_builder import build_flat_tree

    # n=10 < min_samples_split=30 → root-only leaf.
    rng = np.random.default_rng(0)
    n = 10
    X_binned = rng.integers(0, 256, size=(n, 4), dtype=np.uint8)
    t_idx = rng.integers(0, 5, size=n).astype(np.int32)
    event = rng.integers(0, 3, size=n).astype(np.int32)

    flat = build_flat_tree(
        X_binned,
        t_idx,
        t_idx,
        event,
        bootstrap_indices=np.arange(n, dtype=np.int32),
        n_bins=256,
        n_causes=2,
        n_time_bins_split=5,
        n_time_bins_full=5,
        min_samples_split=30,
        min_samples_leaf=5,
        max_depth=-1,
        max_features=2,
        nsplit=10,
        splitrule_code=0,
        cause=1,
        seed=0,
    )

    assert flat.is_leaf_flags.shape == (1,)
    assert flat.is_leaf_flags[0]
    assert flat.leaf_idx_of_node[0] == 0
    assert flat.leaf_table.shape == (1, 2, 5)


def test_build_flat_tree_overflow_guard_produces_valid_leaves():
    """Stack-overflow guard must coerce current node to a leaf (not orphan
    its children). Triggered by an undersized N_max_nodes via tiny
    min_samples_leaf forcing very deep splits.

    This is a regression test for the C1 issue from Task 2 code review.
    """
    from crforest._flat_tree_builder import build_flat_tree

    rng = np.random.default_rng(0)
    n = 200
    n_features = 3
    X_binned = rng.integers(0, 256, size=(n, n_features), dtype=np.uint8)
    t_idx = rng.integers(0, 10, size=n).astype(np.int32)
    event = rng.integers(0, 3, size=n).astype(np.int32)

    flat = build_flat_tree(
        X_binned,
        t_idx,
        t_idx,
        event,
        bootstrap_indices=np.arange(n, dtype=np.int32),
        n_bins=256,
        n_causes=2,
        n_time_bins_split=10,
        n_time_bins_full=10,
        # Aggressive params: min_samples_leaf=2 → many splits → many nodes
        min_samples_split=4,
        min_samples_leaf=2,
        max_depth=-1,
        max_features=3,
        nsplit=10,
        splitrule_code=0,
        cause=1,
        seed=0,
    )

    # Invariant: every internal node points at children that exist and are
    # either valid leaves OR valid internal nodes (no orphan slots that
    # are zero-init internal-node placeholders).
    n_nodes = flat.is_leaf_flags.shape[0]
    for i in range(n_nodes):
        if not flat.is_leaf_flags[i]:
            li = int(flat.left_children[i])
            ri = int(flat.right_children[i])
            assert 0 <= li < n_nodes, f"node {i} left_child {li} out of range"
            assert 0 <= ri < n_nodes, f"node {i} right_child {ri} out of range"
            assert li != i and ri != i, f"node {i} self-points"
            # Children must NOT be all-zero unless they are leaves.
            if not flat.is_leaf_flags[li]:
                # Internal child: must have a meaningful feature OR
                # different children (not 0/0).
                assert (
                    int(flat.features[li]) != 0
                    or int(flat.left_children[li]) != 0
                    or int(flat.right_children[li]) != 0
                ), f"orphan node detected at left_child {li}"
            if not flat.is_leaf_flags[ri]:
                assert (
                    int(flat.features[ri]) != 0
                    or int(flat.left_children[ri]) != 0
                    or int(flat.right_children[ri]) != 0
                ), f"orphan node detected at right_child {ri}"

    # Predict — must not infinite-loop.
    from crforest._tree_flat import predict_with_flat

    cif = predict_with_flat(flat, X_binned)
    assert cif.shape == (n, 2, 10)
    assert np.isfinite(cif).all()


def test_build_flat_tree_splits_with_clear_signal():
    """Planted-signal sanity: feature 0 perfectly separates time-to-event,
    so the root split's chosen feature should be feature 0 even though
    other features are present and randomly distributed."""
    from crforest._flat_tree_builder import build_flat_tree

    n = 200
    n_features = 4
    rng = np.random.default_rng(42)

    X_binned = rng.integers(0, 256, size=(n, n_features), dtype=np.uint8)
    X_binned[:100, 0] = rng.integers(0, 50, size=100, dtype=np.uint8)
    X_binned[100:, 0] = rng.integers(200, 256, size=100, dtype=np.uint8)
    t_idx = np.empty(n, dtype=np.int32)
    t_idx[:100] = 0  # all early
    t_idx[100:] = 9  # all late
    event = np.ones(n, dtype=np.int32)  # all cause-1 events

    flat = build_flat_tree(
        X_binned,
        t_idx,
        t_idx,
        event,
        bootstrap_indices=np.arange(n, dtype=np.int32),
        n_bins=256,
        n_causes=2,
        n_time_bins_split=10,
        n_time_bins_full=10,
        min_samples_split=30,
        min_samples_leaf=15,
        max_depth=-1,
        max_features=4,
        nsplit=10,
        splitrule_code=0,
        cause=1,
        seed=42,
    )

    # Should make at least one split (root + 2 children = 3 nodes minimum).
    assert flat.is_leaf_flags.shape[0] >= 3, f"expected ≥3 nodes, got {flat.is_leaf_flags.shape[0]}"
    assert int(flat.is_leaf_flags.sum()) >= 2, "expected ≥2 leaves"
    # Root's chosen feature should be 0 (the only informative one).
    assert flat.features[0] == 0, f"root feature {flat.features[0]} != 0"


def test_build_flat_tree_within_lib_p95_stable_across_seeds():
    """Two flat-tree forests at adjacent seeds should produce CIFs within
    the same-lib seed-to-seed noise band on a small synthetic dataset.
    Acts as a within-lib stability gate for the new builder."""
    from crforest._flat_tree_builder import build_flat_tree
    from crforest._tree_flat import predict_with_flat

    n = 500
    n_features = 5
    rng = np.random.default_rng(0)
    X_binned = rng.integers(0, 256, size=(n, n_features), dtype=np.uint8)
    t_idx = rng.integers(0, 10, size=n).astype(np.int32)
    event = rng.integers(0, 3, size=n).astype(np.int32)
    bootstrap = rng.choice(n, size=n, replace=True).astype(np.int32)

    def _fit_seed(seed):
        return build_flat_tree(
            X_binned,
            t_idx,
            t_idx,
            event,
            bootstrap_indices=bootstrap,
            n_bins=256,
            n_causes=2,
            n_time_bins_split=10,
            n_time_bins_full=10,
            min_samples_split=30,
            min_samples_leaf=15,
            max_depth=-1,
            max_features=3,
            nsplit=5,
            splitrule_code=0,
            cause=1,
            seed=seed,
        )

    flat_a = _fit_seed(11)
    flat_b = _fit_seed(13)
    cif_a = predict_with_flat(flat_a, X_binned)
    cif_b = predict_with_flat(flat_b, X_binned)

    # CIFs are in [0, 1]; same-lib noise on adjacent seeds is typically
    # under 0.40 at p95 on small data. Loose bound.
    p95 = float(np.percentile(np.abs(cif_a - cif_b), 95))
    assert p95 < 0.50, f"within-lib p95 |ΔCIF| = {p95:.3f} too large"
