"""Integration tests for the new flat-tree default-mode forest path."""

from __future__ import annotations

import numpy as np

from crforest import CompetingRiskForest


def _toy_dataset(n=300, p=4, seed=0):
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, p))
    time = rng.exponential(scale=2.0, size=n)
    event = rng.integers(0, 3, size=n)
    return X, time, event


def test_default_path_fit_returns_predictions_in_unit_interval():
    X, time, event = _toy_dataset()
    forest = CompetingRiskForest(
        n_estimators=10,
        min_samples_leaf=15,
        max_features=2,
        nsplit=5,
        splitrule="logrankCR",
        split_ntime=5,
        random_state=42,
        n_jobs=1,
    ).fit(X, time, event)
    cif = forest.predict_cif(X)
    assert cif.shape == (X.shape[0], forest.n_causes_, len(forest.time_grid_))
    # CIFs are non-negative; values in [0, 1].
    assert (cif >= 0).all()
    assert (cif <= 1).all()


def test_default_path_uses_flat_tree_representation():
    X, time, event = _toy_dataset()
    forest = CompetingRiskForest(
        n_estimators=2,
        min_samples_leaf=15,
        max_features=2,
        random_state=0,
        n_jobs=1,
    ).fit(X, time, event)
    # New default mode produces FlatTree directly (no HistTreeNode in trees_).
    from crforest._tree_flat import FlatTree

    assert all(isinstance(t, FlatTree) for t in forest.trees_)


def test_equivalence_preset_still_uses_old_path():
    """Ensure equivalence='rfsrc' continues to produce HistTreeNode (Plan
    1 must NOT touch that code path)."""
    X, time, event = _toy_dataset(n=200)
    forest = CompetingRiskForest(
        n_estimators=2,
        min_samples_leaf=15,
        bootstrap=True,
        random_state=0,
        equivalence="rfsrc",
        n_jobs=1,
    ).fit(X, time, event)
    from crforest._hist_tree import HistTreeNode

    assert all(isinstance(t, HistTreeNode) for t in forest.trees_)
