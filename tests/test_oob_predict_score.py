"""Public API tests for forest.predict_oob_risk() and forest.oob_score()."""

from __future__ import annotations

import numpy as np
import pytest

from crforest import CompetingRiskForest
from crforest.metrics import concordance_index_cr


def _toy(n=200, p=4, seed=0, n_causes=2):
    rng = np.random.RandomState(seed)
    X = rng.randn(n, p)
    time = rng.uniform(0.1, 10, n)
    event = rng.randint(0, n_causes + 1, n).astype(np.int64)
    return X, time, event


def _fit(n=200, p=4, n_estimators=20, bootstrap=True, seed=0):
    X, time, event = _toy(n=n, p=p, seed=seed)
    forest = CompetingRiskForest(
        n_estimators=n_estimators,
        max_depth=4,
        min_samples_leaf=10,
        bootstrap=bootstrap,
        random_state=seed,
        n_jobs=1,
    )
    forest.fit(X, time, event)
    return forest, X, time, event


def test_predict_oob_risk_shape_and_dtype():
    forest, X, _time, _event = _fit()
    risk = forest.predict_oob_risk(cause=1)
    assert risk.shape == (X.shape[0],)
    assert risk.dtype == np.float64
    finite = np.isfinite(risk)
    # All training rows should be OOB for at least one tree at n_estimators=20,
    # so every entry is finite.
    assert finite.all(), f"non-finite OOB risks: {(~finite).sum()} rows"


def test_predict_oob_risk_requires_bootstrap():
    forest, _X, _time, _event = _fit(bootstrap=False)
    with pytest.raises(ValueError, match="bootstrap=True"):
        forest.predict_oob_risk(cause=1)


def test_predict_oob_risk_cause_validation():
    forest, _X, _time, _event = _fit()
    n_causes = forest.n_causes_
    with pytest.raises(ValueError, match="cause"):
        forest.predict_oob_risk(cause=0)
    with pytest.raises(ValueError, match="cause"):
        forest.predict_oob_risk(cause=n_causes + 1)


def test_predict_oob_risk_matches_manual_aggregation():
    """OOB risk should equal pred[cause-1] / count from the existing primitive."""
    from crforest._importance import _ensemble_oob_predictions

    forest, X, _time, _event = _fit(n=120, p=3, n_estimators=15, seed=1)
    causes = list(range(1, forest.n_causes_ + 1))
    pred, count = _ensemble_oob_predictions(
        forest,
        X,
        causes=causes,
        bin_edges=getattr(forest, "bin_edges_", None),
        time_grid=forest.unique_times_,
    )
    expected_cause1 = pred[0] / np.maximum(count, 1)
    actual_cause1 = forest.predict_oob_risk(cause=1)
    np.testing.assert_allclose(actual_cause1, expected_cause1, rtol=1e-12, atol=1e-12)


def test_oob_score_returns_finite_float():
    forest, _X, _time, _event = _fit()
    score = forest.oob_score(cause=1)
    assert isinstance(score, float)
    assert np.isfinite(score)
    # IID synthetic with random labels — score sits near 0.5 but discrimination
    # is unconstrained at small n; just require [0, 1].
    assert 0.0 <= score <= 1.0


def test_oob_score_matches_external_concordance():
    """oob_score(cause=k) == concordance_index_cr(event, time, predict_oob_risk(k))."""
    forest, _X, time, event = _fit(seed=2)
    for cause in range(1, forest.n_causes_ + 1):
        risk = forest.predict_oob_risk(cause=cause)
        expected = concordance_index_cr(event, time, risk, cause=cause)
        actual = forest.oob_score(cause=cause)
        assert actual == pytest.approx(expected, rel=0, abs=1e-12)


def test_oob_score_requires_bootstrap():
    forest, _X, _time, _event = _fit(bootstrap=False)
    with pytest.raises(ValueError, match="bootstrap=True"):
        forest.oob_score(cause=1)
