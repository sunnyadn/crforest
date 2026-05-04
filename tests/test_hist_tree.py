"""Tests for histogram-mode trees: node dataclass, leaf counts, predict."""

import numpy as np
import pytest

from comprisk._binning import apply_bins, fit_bin_edges
from comprisk._hist_tree import (
    HistTreeNode,
    _leaf_counts,
    build_tree_hist,
    predict_tree_hist,
)


def _make_binned_data(n=60, p=3, n_bins=8, seed=42):
    """Generate random binned X, time, event arrays for histogram-split tests.

    Returns a dict with keys: X, X_binned, bin_edges, time, event, t_idx,
    n_bins, n_time_bins, p, selected (np.arange(p, dtype=int64)).
    """
    rng = np.random.default_rng(seed)
    X = rng.uniform(0, 10, size=(n, p))
    time = rng.uniform(1.0, 10.0, n).astype(np.float64)
    event = rng.integers(0, 3, n).astype(np.int64)
    event[0] = 1
    event[1] = 2
    bin_edges = fit_bin_edges(X, n_bins=n_bins)
    X_binned = apply_bins(X, bin_edges)
    unique_t = np.sort(np.unique(time))
    t_idx = (np.searchsorted(unique_t, time, side="right") - 1).astype(np.int32)
    return {
        "X": X,
        "X_binned": X_binned,
        "bin_edges": bin_edges,
        "time": time,
        "event": event,
        "t_idx": t_idx,
        "n_bins": n_bins,
        "n_time_bins": len(unique_t),
        "p": p,
        "selected": np.arange(p, dtype=np.int64),
    }


def test_hist_tree_node_defaults():
    node = HistTreeNode()
    assert node.is_leaf is False
    assert node.feature == -1
    assert node.bin_idx == 0
    assert node.left is None
    assert node.right is None
    assert node.event_counts_sparse is None
    assert node.at_risk_sparse is None


def test_leaf_counts_shapes_and_dtypes():
    t_idx = np.array([0, 1, 0, 2], dtype=np.int32)
    ev = np.array([1, 2, 0, 1], dtype=np.int64)
    event_counts, at_risk = _leaf_counts(t_idx, ev, n_causes=2, n_time_bins=3)
    assert event_counts.shape == (2, 3)
    assert event_counts.dtype == np.uint32
    assert at_risk.shape == (3,)
    assert at_risk.dtype == np.uint32
    # Event counts
    assert event_counts[0, 0] == 1  # cause 1 at t=0
    assert event_counts[1, 1] == 1  # cause 2 at t=1
    assert event_counts[0, 2] == 1  # cause 1 at t=2
    assert event_counts[0].sum() == 2
    assert event_counts[1].sum() == 1
    # Samples at times: t=0: 2, t=1: 1, t=2: 1 => reverse cumsum = (4, 2, 1)
    np.testing.assert_array_equal(at_risk, [4, 2, 1])


def test_leaf_cif_from_counts_matches_aalen_johansen():
    from comprisk._estimators import aalen_johansen, aalen_johansen_from_counts

    t_idx = np.array([0, 1, 1, 2], dtype=np.int32)
    ev = np.array([1, 2, 1, 0], dtype=np.int64)
    event_counts, at_risk = _leaf_counts(t_idx, ev, n_causes=2, n_time_bins=3)
    cif_hist = aalen_johansen_from_counts(event_counts, at_risk, n_causes=2)

    time = np.array([0.0, 1.0, 1.0, 2.0])
    event = np.array([1, 2, 1, 0])
    unique_times = np.array([0.0, 1.0, 2.0])
    cif_ref = aalen_johansen(time, event, unique_times, n_causes=2)

    np.testing.assert_allclose(cif_hist, cif_ref, atol=1e-10)


def test_build_tree_hist_single_leaf_when_too_small():
    X_binned = np.array([[0], [1]], dtype=np.uint8)
    t_idx = np.array([0, 1], dtype=np.int32)
    ev = np.array([1, 0], dtype=np.int64)
    tree = build_tree_hist(
        X_binned,
        t_idx,
        ev,
        n_causes=1,
        n_bins=2,
        n_time_bins=2,
        max_depth=5,
        min_samples_split=5,
        min_samples_leaf=1,
        max_features=None,
        rng=np.random.RandomState(0),
    )
    assert tree.is_leaf is True
    assert tree.event_counts_sparse is not None
    assert tree.at_risk_sparse is not None


def test_build_tree_hist_splits_separable_data():
    # 10 samples: bin 0 cause-1 at t=0, bin 1 censored at t=1
    X_binned = np.zeros((10, 1), dtype=np.uint8)
    X_binned[5:, 0] = 1
    t_idx = np.concatenate([np.zeros(5, dtype=np.int32), np.ones(5, dtype=np.int32)])
    ev = np.concatenate([np.ones(5, dtype=np.int64), np.zeros(5, dtype=np.int64)])
    tree = build_tree_hist(
        X_binned,
        t_idx,
        ev,
        n_causes=1,
        n_bins=2,
        n_time_bins=2,
        max_depth=3,
        min_samples_split=2,
        min_samples_leaf=1,
        max_features=None,
        rng=np.random.RandomState(0),
    )
    assert tree.is_leaf is False
    assert tree.feature == 0
    assert tree.bin_idx == 0
    assert tree.left.is_leaf is True
    assert tree.right.is_leaf is True


def test_build_tree_hist_respects_max_depth():
    X_binned = np.array([[0], [1]] * 10, dtype=np.uint8)
    t_idx = np.array([0, 1] * 10, dtype=np.int32)
    ev = np.array([1, 0] * 10, dtype=np.int64)
    tree = build_tree_hist(
        X_binned,
        t_idx,
        ev,
        n_causes=1,
        n_bins=2,
        n_time_bins=2,
        max_depth=0,
        min_samples_split=2,
        min_samples_leaf=1,
        max_features=None,
        rng=np.random.RandomState(0),
    )
    assert tree.is_leaf is True  # depth 0 ⇒ immediate leaf


def test_predict_tree_hist_shape_and_dtype():
    X_binned = np.zeros((8, 1), dtype=np.uint8)
    X_binned[4:, 0] = 1
    t_idx = np.concatenate([np.zeros(4, dtype=np.int32), np.ones(4, dtype=np.int32)])
    ev = np.concatenate([np.ones(4, dtype=np.int64), np.zeros(4, dtype=np.int64)])
    tree = build_tree_hist(
        X_binned,
        t_idx,
        ev,
        n_causes=1,
        n_bins=2,
        n_time_bins=2,
        max_depth=3,
        min_samples_split=2,
        min_samples_leaf=1,
        max_features=None,
        rng=np.random.RandomState(0),
    )
    preds = predict_tree_hist(tree, X_binned)
    assert preds.shape == (8, 1, 2)
    assert preds.dtype == np.float64


def test_predict_tree_hist_returns_leaf_cif():
    # Build a tree that splits samples into left (all events) vs right (all censored)
    X_binned = np.zeros((10, 1), dtype=np.uint8)
    X_binned[5:, 0] = 1
    t_idx = np.concatenate([np.zeros(5, dtype=np.int32), np.ones(5, dtype=np.int32)])
    ev = np.concatenate([np.ones(5, dtype=np.int64), np.zeros(5, dtype=np.int64)])
    tree = build_tree_hist(
        X_binned,
        t_idx,
        ev,
        n_causes=1,
        n_bins=2,
        n_time_bins=2,
        max_depth=3,
        min_samples_split=2,
        min_samples_leaf=1,
        max_features=None,
        rng=np.random.RandomState(0),
    )
    preds = predict_tree_hist(tree, X_binned)
    # Left leaf: 5 events at t=0 ⇒ CIF jumps to 1.0 at t=0
    assert preds[0, 0, 0] == 1.0
    # Right leaf: 5 censored at t=1 ⇒ CIF stays 0
    assert preds[5, 0, 0] == 0.0
    assert preds[5, 0, 1] == 0.0


def test_predict_tree_hist_caches_flat_tree():
    X_binned = np.array([[0], [1]], dtype=np.uint8)
    t_idx = np.array([0, 1], dtype=np.int32)
    ev = np.array([1, 0], dtype=np.int64)
    tree = build_tree_hist(
        X_binned,
        t_idx,
        ev,
        n_causes=1,
        n_bins=2,
        n_time_bins=2,
        max_depth=5,
        min_samples_split=5,
        min_samples_leaf=1,
        max_features=None,
        rng=np.random.RandomState(0),
    )
    predict_tree_hist(tree, X_binned)
    flat_first = tree._flat
    assert flat_first is not None
    predict_tree_hist(tree, X_binned)
    assert tree._flat is flat_first  # cache reused, not rebuilt


def test_leaf_chf_from_counts_matches_nelson_aalen_cs():
    """Default-mode CHF materialization agrees with the reference NA-CS."""
    from comprisk._estimators import nelson_aalen_cs, nelson_aalen_from_counts

    t_idx = np.array([0, 1, 1, 2], dtype=np.int32)
    ev = np.array([1, 2, 1, 0], dtype=np.int64)
    event_counts, at_risk = _leaf_counts(t_idx, ev, n_causes=2, n_time_bins=3)
    chf_hist = nelson_aalen_from_counts(event_counts, at_risk, n_causes=2)

    time = np.array([0.0, 1.0, 1.0, 2.0])
    event = np.array([1, 2, 1, 0])
    unique_times = np.array([0.0, 1.0, 2.0])
    chf_ref = nelson_aalen_cs(time, event, unique_times, n_causes=2)

    np.testing.assert_allclose(chf_hist, chf_ref, atol=1e-10)


def test_predict_tree_hist_chf_single_leaf():
    """Forced-leaf histogram tree CHF equals the root nelson_aalen_from_counts."""
    from comprisk._estimators import nelson_aalen_from_counts
    from comprisk._hist_tree import build_tree_hist, predict_tree_hist_chf

    X_binned = np.array([[0], [1], [0], [1]], dtype=np.uint8)
    t_idx = np.array([0, 1, 2, 1], dtype=np.int32)
    ev = np.array([1, 2, 1, 0], dtype=np.int64)
    tree = build_tree_hist(
        X_binned,
        t_idx,
        ev,
        n_causes=2,
        n_bins=2,
        n_time_bins=3,
        max_depth=5,
        min_samples_split=100,
        min_samples_leaf=1,
        max_features=None,
        rng=np.random.RandomState(0),
    )
    assert tree.is_leaf

    chf_pred = predict_tree_hist_chf(tree, X_binned)
    chf_expected = nelson_aalen_from_counts(tree.event_counts_dense, tree.at_risk_dense, n_causes=2)

    assert chf_pred.shape == (4, 2, 3)
    for i in range(4):
        np.testing.assert_allclose(chf_pred[i], chf_expected, atol=1e-12)


def test_predict_tree_hist_chf_caches_flat_tree():
    X_binned = np.array([[0], [1]], dtype=np.uint8)
    t_idx = np.array([0, 1], dtype=np.int32)
    ev = np.array([1, 0], dtype=np.int64)
    tree = build_tree_hist(
        X_binned,
        t_idx,
        ev,
        n_causes=1,
        n_bins=2,
        n_time_bins=2,
        max_depth=5,
        min_samples_split=5,  # forces the root to be a leaf
        min_samples_leaf=1,
        max_features=None,
        rng=np.random.RandomState(0),
    )
    # Lazy initial state
    assert tree._chf is None
    assert tree._flat_chf is None

    from comprisk._hist_tree import predict_tree_hist_chf

    predict_tree_hist_chf(tree, X_binned)

    flat_first = tree._flat_chf
    assert flat_first is not None
    assert tree._chf is not None  # materialized on first predict

    predict_tree_hist_chf(tree, X_binned)
    assert tree._flat_chf is flat_first  # cache reused, not rebuilt


def test_build_tree_hist_accepts_splitrule_logrank():
    from comprisk._time_grid import fit_time_grid

    rng = np.random.default_rng(2)
    n = 50
    X = rng.standard_normal((n, 3))
    time = rng.uniform(0.1, 5.0, size=n)
    event = rng.integers(0, 3, size=n).astype(np.int64)

    edges = fit_bin_edges(X, n_bins=32)
    X_binned = apply_bins(X, edges).astype(np.uint8)
    grid = fit_time_grid(time, event, max_points=40)
    t_idx = np.clip(np.searchsorted(grid, time, side="right") - 1, 0, len(grid) - 1).astype(
        np.int32
    )

    tree = build_tree_hist(
        X_binned,
        t_idx,
        event,
        n_causes=2,
        n_bins=32,
        n_time_bins=len(grid),
        max_depth=3,
        min_samples_split=4,
        min_samples_leaf=2,
        max_features=None,
        rng=None,
        splitrule="logrank",
        cause=1,
    )
    assert tree is not None


def test_build_tree_hist_nsplit_zero_matches_pre_p3a5_structure():
    """nsplit=0 must preserve the exhaustive histogram-build tree structure."""
    d = _make_binned_data(n=50, p=3, n_bins=16, seed=20)

    tree_pre = build_tree_hist(
        d["X_binned"],
        d["t_idx"],
        d["event"],
        n_causes=2,
        n_bins=d["n_bins"],
        n_time_bins=d["n_time_bins"],
        max_depth=3,
        min_samples_split=4,
        min_samples_leaf=1,
        max_features=None,
        rng=None,
    )
    tree_ns0 = build_tree_hist(
        d["X_binned"],
        d["t_idx"],
        d["event"],
        n_causes=2,
        n_bins=d["n_bins"],
        n_time_bins=d["n_time_bins"],
        max_depth=3,
        min_samples_split=4,
        min_samples_leaf=1,
        max_features=None,
        rng=None,
        nsplit=0,
    )

    def preorder(n):
        if n.is_leaf:
            return [("leaf",)]
        return [("split", n.feature, n.bin_idx), *preorder(n.left), *preorder(n.right)]

    assert preorder(tree_pre) == preorder(tree_ns0)


def test_build_tree_hist_nsplit_positive_requires_rng():
    """nsplit > 0 without rng must raise."""
    d = _make_binned_data(n=40, p=2, n_bins=8, seed=21)

    with pytest.raises(ValueError, match="nsplit > 0 requires an rng"):
        build_tree_hist(
            d["X_binned"],
            d["t_idx"],
            d["event"],
            n_causes=2,
            n_bins=d["n_bins"],
            n_time_bins=d["n_time_bins"],
            max_depth=2,
            min_samples_split=4,
            min_samples_leaf=1,
            max_features=None,
            rng=None,
            nsplit=2,
        )
