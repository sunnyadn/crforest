"""Tests for shared event-time quantile grid."""

import numpy as np
import pytest

from comprisk._time_grid import fit_time_grid


def test_fit_time_grid_uses_all_event_times_when_few():
    time = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    event = np.array([1, 0, 1, 0, 1])  # events at t=1, 3, 5
    grid = fit_time_grid(time, event, max_points=200)
    np.testing.assert_array_equal(grid, [1.0, 3.0, 5.0])


def test_fit_time_grid_ignores_censoring_times():
    time = np.array([1.0, 2.0, 3.0])
    event = np.array([0, 0, 0])  # all censored
    with pytest.raises(ValueError, match="no event"):
        fit_time_grid(time, event, max_points=200)


def test_fit_time_grid_quantile_when_many_events():
    rng = np.random.default_rng(2)
    n = 500
    time = rng.uniform(0, 100, size=n)
    event = np.ones(n, dtype=np.int64)
    grid = fit_time_grid(time, event, max_points=50)
    assert len(grid) <= 50
    assert np.all(np.diff(grid) > 0)  # strictly increasing after dedupe
    assert grid[0] == time.min()
    assert grid[-1] == time.max()


def test_fit_time_grid_deduplicates_quantile_ties():
    # 50% of events tied at t=1, 50% at t=2 -> quantile grid should dedupe
    time = np.concatenate([np.full(100, 1.0), np.full(100, 2.0)])
    event = np.ones(200, dtype=np.int64)
    grid = fit_time_grid(time, event, max_points=50)
    # Only two unique event times exist, grid should be exactly [1.0, 2.0]
    np.testing.assert_array_equal(grid, [1.0, 2.0])


def test_fit_time_grid_returns_sorted_float64():
    time = np.array([5.0, 1.0, 3.0])
    event = np.array([1, 1, 1])
    grid = fit_time_grid(time, event)
    assert grid.dtype == np.float64
    np.testing.assert_array_equal(grid, [1.0, 3.0, 5.0])
