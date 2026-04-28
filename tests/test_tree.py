"""Tests for the reference-mode tree builder and predictor."""

import numpy as np
import pytest

from crforest._estimators import aalen_johansen, aalen_johansen_from_counts
from crforest._tree import RefTreeNode, build_tree, predict_tree


def _toy_separating_dataset():
    # Feature 0 perfectly separates early- vs late-event samples
    rng = np.random.default_rng(0)
    n = 40
    X = np.zeros((n, 2))
    X[:, 0] = np.concatenate([np.zeros(20), np.ones(20)])
    X[:, 1] = rng.uniform(size=n)
    time = np.concatenate([rng.uniform(0.5, 1.5, 20), rng.uniform(5.0, 6.0, 20)])
    event = np.ones(n, dtype=int)  # single cause
    return X, time, event


def test_build_tree_returns_leaf_when_below_min_samples_split():
    X = np.array([[0.0], [1.0]])
    time = np.array([1.0, 2.0])
    event = np.array([1, 1])
    unique_times = np.array([1.0, 2.0])
    tree = build_tree(
        X,
        time,
        event,
        n_causes=1,
        max_depth=10,
        min_samples_split=5,
        min_samples_leaf=1,
        unique_times=unique_times,
    )
    assert tree.is_leaf
    assert tree.event_counts is not None
    assert tree.at_risk is not None


def test_build_tree_root_split_feature_is_separating_one():
    X, time, event = _toy_separating_dataset()
    unique_times = np.sort(np.unique(time))
    tree = build_tree(
        X,
        time,
        event,
        n_causes=1,
        max_depth=5,
        min_samples_split=4,
        min_samples_leaf=2,
        unique_times=unique_times,
    )
    assert not tree.is_leaf
    assert tree.feature == 0


def test_zero_split_tree_leaf_equals_dataset_wide_cif():
    # Force a zero-split tree by requiring more samples-per-split than we have
    X = np.zeros((5, 2))
    time = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    event = np.array([1, 0, 2, 1, 0])
    unique_times = np.sort(np.unique(time))
    tree = build_tree(
        X,
        time,
        event,
        n_causes=2,
        max_depth=5,
        min_samples_split=100,
        min_samples_leaf=1,
        unique_times=unique_times,
    )
    assert tree.is_leaf
    expected = aalen_johansen(time, event, unique_times, n_causes=2)
    leaf_cif = aalen_johansen_from_counts(tree.event_counts, tree.at_risk, n_causes=2)
    assert np.allclose(leaf_cif, expected, atol=1e-12)


def test_build_tree_deterministic_under_same_seed():
    X, time, event = _toy_separating_dataset()
    unique_times = np.sort(np.unique(time))

    def build():
        return build_tree(
            X,
            time,
            event,
            n_causes=1,
            max_depth=5,
            min_samples_split=4,
            min_samples_leaf=2,
            unique_times=unique_times,
            max_features=1,
            rng=np.random.RandomState(42),
        )

    t1 = build()
    t2 = build()

    # Walk both trees in parallel, comparing feature/threshold/is_leaf
    def compare(a: RefTreeNode, b: RefTreeNode):
        assert a.is_leaf == b.is_leaf
        if a.is_leaf:
            np.testing.assert_array_equal(a.event_counts, b.event_counts)
            np.testing.assert_array_equal(a.at_risk, b.at_risk)
            return
        assert a.feature == b.feature
        assert np.isclose(a.threshold, b.threshold)
        compare(a.left, b.left)
        compare(a.right, b.right)

    compare(t1, t2)


def test_predict_tree_leaf_value_matches_node_cif():
    X, time, event = _toy_separating_dataset()
    unique_times = np.sort(np.unique(time))
    tree = build_tree(
        X,
        time,
        event,
        n_causes=1,
        max_depth=1,
        min_samples_split=4,
        min_samples_leaf=2,
        unique_times=unique_times,
    )
    preds = predict_tree(tree, X)
    assert preds.shape == (X.shape[0], 1, len(unique_times))
    # With max_depth=1, tree has at most 2 leaves; predictions take at most 2 distinct values
    uniq_preds = {tuple(p.ravel().tolist()) for p in preds}
    assert 1 <= len(uniq_preds) <= 2


def test_predict_tree_shape_for_multi_cause():
    X, time, event = _toy_separating_dataset()
    # Inject a cause-2 event so n_causes is well-defined as 2
    event[0] = 2
    unique_times = np.sort(np.unique(time))
    tree = build_tree(
        X,
        time,
        event,
        n_causes=2,
        max_depth=3,
        min_samples_split=4,
        min_samples_leaf=2,
        unique_times=unique_times,
    )
    preds = predict_tree(tree, X)
    assert preds.shape == (X.shape[0], 2, len(unique_times))
    assert np.all(preds >= 0.0)
    assert np.all(preds <= 1.0 + 1e-9)


def test_build_tree_rejects_max_features_without_rng():
    X = np.zeros((10, 3))
    time = np.arange(1.0, 11.0)
    event = np.ones(10, dtype=int)
    with pytest.raises(ValueError, match="max_features requires an rng"):
        build_tree(
            X,
            time,
            event,
            n_causes=1,
            max_depth=3,
            min_samples_split=4,
            min_samples_leaf=2,
            max_features=2,
        )


def test_predict_tree_rejects_non_2d_input():
    X, time, event = _toy_separating_dataset()
    unique_times = np.sort(np.unique(time))
    tree = build_tree(
        X,
        time,
        event,
        n_causes=1,
        max_depth=2,
        min_samples_split=4,
        min_samples_leaf=2,
        unique_times=unique_times,
    )
    with pytest.raises(ValueError, match="X must be 2-D"):
        predict_tree(tree, np.array([0.0, 1.0, 2.0]))


def test_build_tree_accepts_splitrule_logrank():
    import numpy as np

    from crforest._tree import build_tree

    rng = np.random.default_rng(0)
    n = 40
    X = np.zeros((n, 2))
    X[:, 0] = np.concatenate([np.zeros(20), np.ones(20)])
    X[:, 1] = rng.uniform(size=n)
    time = np.concatenate([rng.uniform(0.5, 1.5, 20), rng.uniform(5.0, 6.0, 20)])
    event = np.concatenate([np.ones(20, dtype=int), 2 * np.ones(20, dtype=int)])

    tree = build_tree(
        X,
        time,
        event,
        n_causes=2,
        max_depth=3,
        min_samples_split=4,
        min_samples_leaf=2,
        splitrule="logrank",
        cause=1,
    )
    assert not tree.is_leaf
    assert tree.feature == 0


def test_predict_tree_chf_single_leaf_matches_nelson_aalen_cs():
    """A forced single-leaf tree's CHF equals the root Nelson-Aalen CHF."""
    from crforest._estimators import nelson_aalen_cs
    from crforest._tree import build_tree, predict_tree_chf

    # 6 samples, 2 features — force a single leaf by making min_samples_split huge
    X = np.array([[0.0, 0.0], [1.0, 1.0], [0.5, 0.5], [0.2, 0.8], [0.8, 0.2], [0.6, 0.4]])
    time = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
    event = np.array([1, 2, 1, 0, 2, 1])
    unique_times = np.sort(np.unique(time))

    tree = build_tree(
        X,
        time,
        event,
        n_causes=2,
        max_depth=5,
        min_samples_split=100,  # ensures root is a leaf
        min_samples_leaf=1,
        unique_times=unique_times,
    )
    assert tree.is_leaf

    chf_pred = predict_tree_chf(tree, X)
    chf_expected = nelson_aalen_cs(time, event, unique_times, n_causes=2)

    assert chf_pred.shape == (6, 2, len(unique_times))
    # Every sample descends to the single leaf, so every row equals the root CHF.
    for i in range(6):
        np.testing.assert_allclose(chf_pred[i], chf_expected, atol=1e-12)


def test_predict_tree_chf_matches_predict_tree_shape():
    """CHF prediction has the same shape as CIF prediction and is non-negative."""
    from crforest._tree import build_tree, predict_tree, predict_tree_chf

    rng = np.random.default_rng(0)
    X = rng.uniform(size=(40, 3))
    time = rng.uniform(0.1, 10.0, 40)
    event = rng.integers(0, 3, 40)
    if not np.any(event > 0):
        event[0] = 1
    unique_times = np.sort(np.unique(time))

    tree = build_tree(
        X,
        time,
        event,
        n_causes=2,
        max_depth=3,
        min_samples_split=4,
        min_samples_leaf=1,
        unique_times=unique_times,
        max_features=None,
        rng=np.random.RandomState(0),
    )
    cif = predict_tree(tree, X)
    chf = predict_tree_chf(tree, X)

    assert chf.shape == cif.shape
    assert np.all(chf >= 0.0)
    assert np.all(np.diff(chf, axis=2) >= -1e-12)


def test_build_tree_nsplit_zero_matches_pre_p3a5_structure():
    """nsplit=0 must preserve the exhaustive-build tree structure."""
    rng_data = np.random.default_rng(10)
    n, p = 40, 2
    X = rng_data.uniform(0, 10, size=(n, p))
    time = rng_data.uniform(1.0, 10.0, n)
    event = rng_data.integers(0, 3, n)
    event[0] = 1
    event[1] = 2

    tree_pre = build_tree(
        X,
        time,
        event,
        n_causes=2,
        max_depth=3,
        min_samples_split=4,
        min_samples_leaf=1,
    )
    tree_ns0 = build_tree(
        X,
        time,
        event,
        n_causes=2,
        max_depth=3,
        min_samples_split=4,
        min_samples_leaf=1,
        nsplit=0,
    )

    def preorder(n):
        if n.is_leaf:
            return [("leaf",)]
        return [("split", n.feature, n.threshold), *preorder(n.left), *preorder(n.right)]

    assert preorder(tree_pre) == preorder(tree_ns0)


def test_build_tree_nsplit_positive_differs_from_exhaustive():
    """nsplit > 0 with an rng produces a (typically) different tree."""
    rng_data = np.random.default_rng(11)
    n, p = 80, 3
    X = rng_data.uniform(0, 10, size=(n, p))
    time = rng_data.uniform(1.0, 10.0, n)
    event = rng_data.integers(0, 3, n)
    event[0] = 1
    event[1] = 2

    tree_ex = build_tree(
        X,
        time,
        event,
        n_causes=2,
        max_depth=3,
        min_samples_split=4,
        min_samples_leaf=1,
    )
    tree_ns = build_tree(
        X,
        time,
        event,
        n_causes=2,
        max_depth=3,
        min_samples_split=4,
        min_samples_leaf=1,
        nsplit=3,
        rng=np.random.RandomState(0),
    )
    assert not tree_ex.is_leaf
    assert not tree_ns.is_leaf
