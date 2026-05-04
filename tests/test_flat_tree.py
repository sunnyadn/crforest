"""δ.2 compact FlatTree.leaf_table invariants."""

from __future__ import annotations

import numpy as np


def _build_3_leaf_hist_tree():
    """Return a hand-built histogram tree with 2 internal + 3 leaf nodes.

    Topology (DFS pre-order yields: root, left-leaf, right-inner, mid-leaf, right-leaf):

                 root (feature=0, bin<=3)
                /                        \\
             left-leaf                   right (feature=1, bin<=7)
                                        /                       \\
                                     mid-leaf                 right-leaf
    """
    from comprisk._hist_tree import HistTreeNode
    from comprisk._sparse_leaves import to_sparse_at_risk, to_sparse_event_counts

    def mkleaf(seed: int) -> HistTreeNode:
        rng = np.random.default_rng(seed)
        ec_dense = np.zeros((2, 20), dtype=np.uint32)
        ec_dense[0, rng.integers(0, 20)] = rng.integers(1, 5)
        ar_dense = np.full(20, 10 - seed, dtype=np.uint32)
        return HistTreeNode(
            is_leaf=True,
            event_counts_sparse=to_sparse_event_counts(ec_dense),
            at_risk_sparse=to_sparse_at_risk(ar_dense),
            _n_causes=2,
            _n_time_bins=20,
        )

    left = mkleaf(seed=1)
    mid = mkleaf(seed=2)
    right = mkleaf(seed=3)
    inner_right = HistTreeNode(feature=1, bin_idx=7, left=mid, right=right)
    root = HistTreeNode(feature=0, bin_idx=3, left=left, right=inner_right)
    return root


def test_leaf_idx_of_node_maps_leaves_in_dfs_order() -> None:
    """leaf_idx_of_node[node] == k iff node is the k-th leaf in DFS order.

    For internal nodes it is -1.
    """
    from comprisk._hist_tree import _flatten_tree_hist

    root = _build_3_leaf_hist_tree()
    flat = _flatten_tree_hist(root)
    assert flat.leaf_idx_of_node.shape == (flat.features.shape[0],)
    assert flat.leaf_idx_of_node.dtype == np.int64

    leaf_positions = np.flatnonzero(flat.is_leaf_flags)
    for k, node_idx in enumerate(leaf_positions):
        assert flat.leaf_idx_of_node[node_idx] == k, (
            f"leaf at node {node_idx} should map to leaf index {k}, "
            f"got {flat.leaf_idx_of_node[node_idx]}"
        )
    internal_positions = np.flatnonzero(~flat.is_leaf_flags)
    for node_idx in internal_positions:
        assert flat.leaf_idx_of_node[node_idx] == -1


def test_leaf_table_compact_shape() -> None:
    """leaf_table is shape (n_leaves, n_causes, n_time_bins)."""
    from comprisk._hist_tree import _flatten_tree_hist

    root = _build_3_leaf_hist_tree()
    flat = _flatten_tree_hist(root)
    n_leaves = int(flat.is_leaf_flags.sum())
    assert flat.leaf_table.shape == (n_leaves, 2, 20)
    assert flat.leaf_table.shape[0] == 3  # three leaves in our fixture


def test_predict_with_flat_uses_leaf_idx_map() -> None:
    """predict_tree_hist returns one row per input X row, shape (n, n_causes, n_time_bins)."""
    from comprisk._hist_tree import _flatten_tree_hist, predict_tree_hist

    root = _build_3_leaf_hist_tree()
    # Three samples chosen to hit each leaf (feature 0 <= 3 → left; > 3 and
    # feature 1 <= 7 → mid; else → right).
    X = np.array(
        [
            [2, 0],  # left leaf
            [5, 3],  # mid leaf
            [5, 12],  # right leaf
        ],
        dtype=np.uint8,
    )
    out = predict_tree_hist(root, X)
    assert out.shape == (3, 2, 20)

    flat = _flatten_tree_hist(root)
    # DFS order yields leaves as: left, mid, right. So sample_idx matches leaf_idx.
    for sample_idx in range(3):
        expected = flat.leaf_table[sample_idx]
        assert np.array_equal(out[sample_idx], expected)
