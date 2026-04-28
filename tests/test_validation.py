"""Tests for input validation at fit-time."""

import numpy as np
import pytest

from crforest._validation import check_inputs


def _valid_inputs():
    X = np.array([[0.0, 1.0], [1.0, 2.0], [2.0, 3.0], [3.0, 4.0]])
    time = np.array([1.0, 2.0, 3.0, 4.0])
    event = np.array([1, 0, 2, 1])
    return X, time, event


def test_accepts_valid_inputs():
    X, time, event = _valid_inputs()
    X_out, time_out, event_out, n_causes = check_inputs(X, time, event)
    assert X_out.dtype == np.float64
    assert time_out.dtype == np.float64
    assert event_out.dtype == np.int64
    assert n_causes == 2


def test_rejects_wrong_X_ndim():
    _, time, event = _valid_inputs()
    with pytest.raises(ValueError, match="X must be 2-D"):
        check_inputs(np.array([1.0, 2.0, 3.0, 4.0]), time, event)


def test_rejects_length_mismatch():
    X, time, event = _valid_inputs()
    with pytest.raises(ValueError, match="length"):
        check_inputs(X, time[:3], event)


def test_rejects_negative_times():
    X, time, event = _valid_inputs()
    time[0] = -1.0
    with pytest.raises(ValueError, match="non-negative"):
        check_inputs(X, time, event)


def test_rejects_nan_times():
    X, time, event = _valid_inputs()
    time[0] = np.nan
    with pytest.raises(ValueError, match="finite"):
        check_inputs(X, time, event)


def test_rejects_inf_times():
    X, time, event = _valid_inputs()
    time[0] = np.inf
    with pytest.raises(ValueError, match="non-finite"):
        check_inputs(X, time, event)


def test_rejects_non_integer_events():
    X, time, _ = _valid_inputs()
    with pytest.raises(ValueError, match="integer"):
        check_inputs(X, time, np.array([1.5, 0.0, 2.0, 1.0]))


def test_rejects_negative_non_integer_float_event():
    # Both "integer-valued" and "non-negative" checks fail; integer check
    # must fire first.
    X, time, _ = _valid_inputs()
    with pytest.raises(ValueError, match="integer"):
        check_inputs(X, time, np.array([-1.5, 0.0, 2.0, 1.0]))


def test_rejects_negative_events():
    X, time, _ = _valid_inputs()
    with pytest.raises(ValueError, match="non-negative"):
        check_inputs(X, time, np.array([-1, 0, 2, 1]))


def test_rejects_all_censored():
    X, time, _ = _valid_inputs()
    with pytest.raises(ValueError, match="at least one event"):
        check_inputs(X, time, np.array([0, 0, 0, 0]))


def test_rejects_non_contiguous_causes():
    X, time, _ = _valid_inputs()
    # causes {1, 3} — missing 2
    with pytest.raises(ValueError, match="contiguous"):
        check_inputs(X, time, np.array([1, 0, 3, 1]))


def test_accepts_single_cause():
    X = np.zeros((3, 2))
    time = np.array([1.0, 2.0, 3.0])
    event = np.array([1, 0, 1])
    _, _, _, n_causes = check_inputs(X, time, event)
    assert n_causes == 1


def test_rejects_empty_inputs():
    X = np.empty((0, 2))
    time = np.array([], dtype=np.float64)
    event = np.array([], dtype=np.int64)
    with pytest.raises(ValueError, match="at least one row"):
        check_inputs(X, time, event)


def test_nan_in_X_rejected():
    X, time, event = _valid_inputs()
    X[0, 0] = np.nan
    with pytest.raises(ValueError, match="NaN"):
        check_inputs(X, time, event)


def test_fit_raises_on_too_many_causes() -> None:
    """n_causes > 255 raises ValueError at fit (δ.3 uint8 cause dtype cap)."""
    from crforest import CompetingRiskForest

    n = 200
    rng = np.random.default_rng(0)
    X = rng.standard_normal((n, 3))
    time = rng.uniform(0.1, 10.0, size=n)
    # 300 distinct positive event codes.
    event = rng.integers(1, 301, size=n)
    forest = CompetingRiskForest(n_estimators=2, random_state=0)
    with pytest.raises(ValueError, match=r"competing causes|n_causes"):
        forest.fit(X, time=time, event=event)


def test_fit_raises_on_too_many_time_bins() -> None:
    """time_grid > 65_535 raises ValueError at fit (δ.3 uint16 time-index cap)."""
    from crforest import CompetingRiskForest

    n = 200
    rng = np.random.default_rng(0)
    X = rng.standard_normal((n, 3))
    time = rng.uniform(0.1, 10.0, size=n)
    event = rng.integers(0, 3, size=n)
    forest = CompetingRiskForest(n_estimators=2, time_grid=70_000, random_state=0)
    with pytest.raises(ValueError, match=r"time_grid|time bins"):
        forest.fit(X, time=time, event=event)
