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
