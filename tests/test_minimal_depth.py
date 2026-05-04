"""Unit tests for minimal-depth variable selection (SUN-42)."""

from __future__ import annotations

import numpy as np

from crforest import CompetingRiskForest


def _toy(n=200, p=4, seed=0, n_causes=2):
    rng = np.random.RandomState(seed)
    X = rng.randn(n, p)
    time = rng.uniform(0.1, 10, n)
    event = rng.randint(0, n_causes + 1, n).astype(np.int64)
    return X, time, event


def _fit(seed=0, n_jobs=1, equivalence=None, mode=None, **kw):
    X, time, event = _toy(seed=seed)
    kwargs = dict(
        n_estimators=20, max_depth=4, min_samples_leaf=5, random_state=seed, n_jobs=n_jobs
    )
    if equivalence is not None:
        kwargs["equivalence"] = equivalence
    if mode is not None:
        kwargs["mode"] = mode
    kwargs.update(kw)
    forest = CompetingRiskForest(**kwargs).fit(
        X, np.array(list(zip(event, time, strict=False)), dtype=[("event", "i8"), ("time", "f8")])
    )
    return forest


def test_schema():
    forest = _fit(seed=0)
    df = forest.minimal_depth()
    assert list(df.columns) == ["feature", "mean_min_depth", "threshold", "selected"]
    assert len(df) == forest.n_features_in_
    # sorted ascending by mean_min_depth
    assert (df["mean_min_depth"].values[:-1] <= df["mean_min_depth"].values[1:]).all()


def test_walker_flat_tree_finds_root_split():
    from crforest._minimal_depth import _walk_min_depth
    from crforest._tree_flat import FlatTree

    forest = _fit(seed=0)
    tree = forest.trees_[0]
    # FlatTree path is the default
    assert isinstance(tree, FlatTree)
    res = _walk_min_depth(tree, n_features=forest.n_features_in_)
    assert res.min_depth_per_feature.shape == (forest.n_features_in_,)
    assert res.min_depth_per_feature.dtype == np.int32
    # At least one feature must be the root split (depth 0)
    assert res.min_depth_per_feature.min() == 0
    # Every value is in [0, D_T + 1]
    assert (res.min_depth_per_feature >= 0).all()
    assert (res.min_depth_per_feature <= res.max_depth + 1).all()


def test_ishwaran_threshold_handcomputed():
    """Depth-2 toy tree: 1 internal node at depth 0, 2 internals at depth 1.

    p = 4, so (1 - 1/p) = 3/4. Cumulative internal counts cumL = [1, 3, 3].
    P(md > 0) = (3/4)^1 = 0.75
    P(md > 1) = (3/4)^3 = 27/64 = 0.421875
    P(md > 2) = (3/4)^3 = 27/64 = 0.421875
    E[md] = 0.75 + 0.421875 + 0.421875 = 1.59375

    Sanity: P(md=0)=0.25, P(md=1)=0.328125, P(md=2)=0, P(md=3)=0.421875
            -> 0*0.25 + 1*0.328125 + 2*0 + 3*0.421875 = 1.59375
    """
    from crforest._minimal_depth import _ishwaran_expected_md

    L = np.array([1, 2], dtype=np.int64)
    expected = 1.59375
    got = _ishwaran_expected_md(L, max_depth_T=2, n_features=4)
    assert abs(got - expected) < 1e-12, f"got {got}, expected {expected}"


def test_ishwaran_threshold_pure_stump():
    """Pure stump (D_T = 0, no internals): expected md = 1.0."""
    from crforest._minimal_depth import _ishwaran_expected_md

    L = np.array([], dtype=np.int64)
    got = _ishwaran_expected_md(L, max_depth_T=0, n_features=4)
    # cumL_full = [0]; P(md>0) = (3/4)^0 = 1.0; sum = 1.0
    assert abs(got - 1.0) < 1e-12
