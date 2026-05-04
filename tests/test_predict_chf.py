"""Tests for CompetingRiskForest.predict_chf (Nelson-Aalen cause-specific CHF)."""

import numpy as np
import pytest
from sklearn.exceptions import NotFittedError

from comprisk.forest import CompetingRiskForest


def _toy_data(n=60, seed=0):
    rng = np.random.default_rng(seed)
    X = rng.uniform(size=(n, 3))
    time = rng.uniform(0.1, 10.0, n)
    event = rng.integers(0, 3, n)
    # Ensure both causes present
    if not np.any(event == 1):
        event[0] = 1
    if not np.any(event == 2):
        event[1] = 2
    return X, time, event


@pytest.mark.parametrize("mode", ["default", "reference"])
def test_predict_chf_shape(mode):
    X, time, event = _toy_data()
    f = CompetingRiskForest(n_estimators=5, mode=mode, random_state=0).fit(X, time, event)
    chf = f.predict_chf(X)
    assert chf.shape == (X.shape[0], f.n_causes_, len(f.unique_times_))
    assert chf.dtype == np.float64


@pytest.mark.parametrize("mode", ["default", "reference"])
def test_predict_chf_non_negative(mode):
    X, time, event = _toy_data()
    f = CompetingRiskForest(n_estimators=5, mode=mode, random_state=0).fit(X, time, event)
    chf = f.predict_chf(X)
    assert np.all(chf >= 0.0)


@pytest.mark.parametrize("mode", ["default", "reference"])
def test_predict_chf_monotone_nondecreasing(mode):
    X, time, event = _toy_data()
    f = CompetingRiskForest(n_estimators=5, mode=mode, random_state=0).fit(X, time, event)
    chf = f.predict_chf(X)
    assert np.all(np.diff(chf, axis=2) >= -1e-9)


@pytest.mark.parametrize("mode", ["default", "reference"])
def test_predict_chf_times_interpolation_before_grid_is_zero(mode):
    """Sampling CHF at a time before any observed event gives 0."""
    X, time, event = _toy_data()
    f = CompetingRiskForest(n_estimators=5, mode=mode, random_state=0).fit(X, time, event)
    t_before = np.array([min(time) - 1.0])
    chf = f.predict_chf(X, times=t_before)
    assert chf.shape == (X.shape[0], f.n_causes_, 1)
    np.testing.assert_allclose(chf, 0.0)


@pytest.mark.parametrize("mode", ["default", "reference"])
def test_predict_chf_times_interpolation_after_grid_plateaus(mode):
    """Sampling CHF at a time beyond the last observed event equals CHF at the
    last grid point (right-continuous step function)."""
    X, time, event = _toy_data()
    f = CompetingRiskForest(n_estimators=5, mode=mode, random_state=0).fit(X, time, event)
    t_beyond = np.array([float(f.unique_times_[-1]) + 10.0])
    chf_beyond = f.predict_chf(X, times=t_beyond)
    chf_full = f.predict_chf(X)
    np.testing.assert_allclose(chf_beyond[:, :, 0], chf_full[:, :, -1], atol=1e-12)


def test_predict_chf_requires_fit():
    f = CompetingRiskForest()
    with pytest.raises(NotFittedError):
        f.predict_chf(np.zeros((2, 3)))


def test_predict_chf_wrong_n_features_raises():
    X, time, event = _toy_data()
    f = CompetingRiskForest(n_estimators=3, random_state=0).fit(X, time, event)
    with pytest.raises(ValueError, match="n_features"):
        f.predict_chf(np.zeros((2, X.shape[1] + 1)))


@pytest.mark.parametrize("mode", ["default", "reference"])
def test_predict_chf_times_interpolation_mid_grid_is_left_neighbor(mode):
    """Right-continuous step: a time strictly between grid[k] and grid[k+1]
    returns the value at grid[k], pinning the `side="right"` / `-1` offset
    convention in _make_time_projection."""
    X, time, event = _toy_data()
    f = CompetingRiskForest(n_estimators=5, mode=mode, random_state=0).fit(X, time, event)
    grid = f.unique_times_
    k = len(grid) // 2
    t_mid = np.array([(grid[k] + grid[k + 1]) / 2.0])
    chf_mid = f.predict_chf(X, times=t_mid)
    chf_full = f.predict_chf(X)
    np.testing.assert_allclose(chf_mid[:, :, 0], chf_full[:, :, k], atol=1e-12)
