"""Round-trip and dtype-selection tests for the δ.2 sparse leaf rep."""

from __future__ import annotations

import numpy as np
import pytest


@pytest.mark.parametrize("seed", list(range(10)))
def test_event_counts_roundtrip_random(seed: int) -> None:
    """to_sparse_event_counts -> to_dense_event_counts must round-trip bit-exactly."""
    from comprisk._sparse_leaves import (
        to_dense_event_counts,
        to_sparse_event_counts,
    )

    rng = np.random.default_rng(seed)
    n_causes = rng.integers(1, 5)
    n_time_bins = rng.integers(10, 400)
    sparsity = rng.choice([0.5, 0.9, 0.99, 1.0])

    dense = np.zeros((n_causes, n_time_bins), dtype=np.uint32)
    mask = rng.random((n_causes, n_time_bins)) > sparsity
    dense[mask] = rng.integers(1, 100, size=mask.sum(), dtype=np.uint32)

    sparse = to_sparse_event_counts(dense)
    back = to_dense_event_counts(sparse, n_causes, n_time_bins)
    assert np.array_equal(back, dense)
    assert back.dtype == np.uint32


@pytest.mark.parametrize(
    "max_val,expected_dtype",
    [
        (0, np.uint8),
        (1, np.uint8),
        (254, np.uint8),
        (255, np.uint8),
        (256, np.uint16),
        (65_535, np.uint16),
    ],
)
def test_event_counts_dtype_boundary(max_val: int, expected_dtype) -> None:
    """δ.3 dtype selector: uint8 if max ≤ 255 else uint16."""
    from comprisk._sparse_leaves import to_sparse_event_counts

    dense = np.zeros((2, 50), dtype=np.uint32)
    if max_val > 0:
        dense[0, 0] = max_val
    sparse = to_sparse_event_counts(dense)
    assert sparse.values.dtype == expected_dtype, (
        f"max={max_val}: expected {expected_dtype}, got {sparse.values.dtype}"
    )


def test_event_counts_all_zero() -> None:
    """Degenerate case: all-zero leaf returns empty COO arrays."""
    from comprisk._sparse_leaves import (
        to_dense_event_counts,
        to_sparse_event_counts,
    )

    dense = np.zeros((2, 100), dtype=np.uint32)
    sparse = to_sparse_event_counts(dense)
    assert len(sparse.cause) == 0
    assert len(sparse.time) == 0
    assert len(sparse.values) == 0
    back = to_dense_event_counts(sparse, 2, 100)
    assert np.array_equal(back, dense)


@pytest.mark.parametrize("seed", list(range(10)))
def test_at_risk_roundtrip_random(seed: int) -> None:
    """at_risk step-function encoding round-trips bit-exactly."""
    from comprisk._sparse_leaves import to_dense_at_risk, to_sparse_at_risk

    rng = np.random.default_rng(seed)
    n_time_bins = rng.integers(10, 400)
    n_samples = rng.integers(1, 100)
    decrements = np.sort(rng.integers(0, n_time_bins, size=n_samples))
    dense = np.zeros(n_time_bins, dtype=np.uint32)
    remaining = n_samples
    last_t = 0
    for t in decrements:
        dense[last_t : t + 1] = remaining
        remaining -= 1
        last_t = t + 1
    if last_t < n_time_bins:
        dense[last_t:] = remaining

    sparse = to_sparse_at_risk(dense)
    back = to_dense_at_risk(sparse, n_time_bins)
    assert np.array_equal(back, dense), (
        f"seed={seed}: round-trip differs at indices {np.where(back != dense)[0][:10]}"
    )
    assert back.dtype == np.uint32


def test_at_risk_monotone_decreasing_constant_segments() -> None:
    """Hand-constructed leaf: breakpoints at indices 0, 3, 6."""
    from comprisk._sparse_leaves import to_dense_at_risk, to_sparse_at_risk

    dense = np.array([5, 5, 5, 3, 3, 3, 1, 1, 1, 1], dtype=np.uint32)
    sparse = to_sparse_at_risk(dense)
    assert np.array_equal(sparse.time, np.array([0, 3, 6], dtype=np.uint16))
    assert np.array_equal(sparse.values, np.array([5, 3, 1], dtype=np.uint16))
    back = to_dense_at_risk(sparse, 10)
    assert np.array_equal(back, dense)


def test_hist_tree_node_lazy_dense_properties() -> None:
    """HistTreeNode.event_counts_dense materializes from sparse on access."""
    from comprisk._hist_tree import HistTreeNode
    from comprisk._sparse_leaves import to_sparse_at_risk, to_sparse_event_counts

    dense_ec = np.zeros((2, 100), dtype=np.uint32)
    dense_ec[0, 10] = 3
    dense_ec[1, 50] = 1
    dense_ar = np.full(100, 20, dtype=np.uint32)
    dense_ar[10:] = 19
    dense_ar[50:] = 18

    node = HistTreeNode(is_leaf=True)
    node.event_counts_sparse = to_sparse_event_counts(dense_ec)
    node.at_risk_sparse = to_sparse_at_risk(dense_ar)
    node._n_causes = 2
    node._n_time_bins = 100

    ec = node.event_counts_dense
    ar = node.at_risk_dense
    assert np.array_equal(ec, dense_ec)
    assert np.array_equal(ar, dense_ar)
    # Cached: second access returns the same object (no recomputation).
    assert node.event_counts_dense is ec
    assert node.at_risk_dense is ar


def test_pickle_size_regression_small_fixture() -> None:
    """Pickle size regression guard for both flat-tree (default) and
    sparse-leaf (equivalence='rfsrc') paths.

    Default-mode FlatTree stores dense float64 CIF tables — much larger
    than HistTreeNode's sparse rep, but bounded. A pickle ballooning
    beyond ~20 MB on this fixture would indicate accidental duplicate-
    storage or O(n²) growth in the leaf rep.

    equivalence='rfsrc' (HistTreeNode) keeps the sparse-leaf rep with
    pickle <= 500 KB. A regression to dense storage would push that to
    ~2 MB.

    Post-delta.2 baseline for the rfsrc fixture is ~335 KB dominated by
    numpy-array pickle overhead on ~1000 small leaves.
    """
    import pickle

    from comprisk import CompetingRiskForest

    n = 200
    p = 5
    rng = np.random.default_rng(0)
    X = rng.standard_normal((n, p))
    time = rng.uniform(0.1, 10.0, size=n)
    event = rng.integers(0, 3, size=n)

    # Default mode (FlatTree, dense leaf_table)
    flat_forest = CompetingRiskForest(n_estimators=20, random_state=0, n_jobs=1)
    flat_forest.fit(X, time=time, event=event)
    flat_pkl = pickle.dumps(flat_forest)
    assert len(flat_pkl) <= 20_000_000, (
        f"FlatTree default-mode pickle = {len(flat_pkl)} bytes (>20 MB); "
        "indicates dense-blob duplication or unexpected leaf storage growth"
    )

    # equivalence='rfsrc' (HistTreeNode, sparse leaves)
    rfsrc_forest = CompetingRiskForest(
        n_estimators=20,
        random_state=0,
        n_jobs=1,
        equivalence="rfsrc",
        bootstrap=True,
    )
    rfsrc_forest.fit(X, time=time, event=event)
    rfsrc_pkl = pickle.dumps(rfsrc_forest)
    assert len(rfsrc_pkl) <= 500_000, (
        f"HistTreeNode (equivalence='rfsrc') pickle = {len(rfsrc_pkl)} bytes "
        f"(>500 KB); sparse-leaf rep regressed to dense"
    )
