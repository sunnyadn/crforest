"""Unit tests for coarsen_time_grid (ε sprint Task 1.2)."""

from __future__ import annotations

import numpy as np
import pytest

from crforest._time_grid import coarsen_time_grid


def test_identity_when_ntime_ge_grid_length() -> None:
    grid = np.array([0.1, 0.5, 1.2, 3.0, 7.5], dtype=np.float64)
    for ntime in (5, 6, 100):
        full_to_split = coarsen_time_grid(grid, ntime)
        assert np.array_equal(full_to_split, np.arange(len(grid), dtype=np.int32))


def test_ntime_one_collapses_to_single_bin() -> None:
    grid = np.linspace(0.0, 10.0, 200)
    full_to_split = coarsen_time_grid(grid, ntime=1)
    assert full_to_split.shape == (200,)
    assert full_to_split.dtype == np.int32
    assert np.all(full_to_split == 0)


def test_monotonic_full_to_split() -> None:
    grid = np.linspace(0.0, 10.0, 200)
    full_to_split = coarsen_time_grid(grid, ntime=30)
    diffs = np.diff(full_to_split)
    assert np.all(diffs >= 0), f"full_to_split must be non-decreasing; diffs={diffs}"


def test_full_to_split_range_bounded_by_ntime() -> None:
    grid = np.linspace(0.0, 10.0, 200)
    ntime = 30
    full_to_split = coarsen_time_grid(grid, ntime=ntime)
    assert full_to_split.min() >= 0
    assert full_to_split.max() < ntime


def test_full_to_split_covers_every_bin() -> None:
    """Every coarse bin receives at least one full-grid index when grid is uniform."""
    grid = np.linspace(0.0, 10.0, 200)
    ntime = 30
    full_to_split = coarsen_time_grid(grid, ntime=ntime)
    assert set(np.unique(full_to_split).tolist()) == set(range(ntime))


def test_pathological_clustering_triggers_uniform_fallback_with_warning() -> None:
    """If quantile bucketing produces < ntime//2 distinct break points, fall back to uniform
    time-range coarsening and emit a UserWarning.
    """
    grid = np.concatenate([np.zeros(195), np.linspace(1.0, 2.0, 5)]).astype(np.float64)
    grid = np.sort(grid)
    with pytest.warns(UserWarning, match="pathological"):
        full_to_split = coarsen_time_grid(grid, ntime=30)
    assert full_to_split.shape == grid.shape
