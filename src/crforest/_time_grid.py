"""Shared event-time quantile grid for compact leaf storage."""

from __future__ import annotations

import warnings

import numpy as np


def fit_time_grid(time: np.ndarray, event: np.ndarray, max_points: int = 200) -> np.ndarray:
    """Quantile grid over event times (where ``event > 0``), capped at ``max_points``.

    CIFs only change at event times, so placing grid points at quantiles of
    event times maximizes fidelity per point. Censored times are ignored.

    Parameters
    ----------
    time : array-like, shape (n,)
    event : array-like, shape (n,)
    max_points : int
        Upper bound on the grid length. Fewer points may be returned if
        there are fewer unique event times or if quantile dedupe collapses
        ties.

    Returns
    -------
    grid : ndarray, sorted float64, length <= max_points
    """
    time = np.asarray(time, dtype=np.float64)
    event = np.asarray(event)
    event_times = time[event > 0]
    if len(event_times) == 0:
        raise ValueError("no event times in training data (all censored)")
    unique_event_times = np.unique(event_times)
    if len(unique_event_times) <= max_points:
        return unique_event_times
    qs = np.linspace(0.0, 1.0, max_points)
    grid = np.quantile(unique_event_times, qs)
    grid = np.unique(grid)  # dedupe after quantile
    return grid


def coarsen_time_grid(time_grid: np.ndarray, ntime: int) -> np.ndarray:
    """Quantile-bucket ``time_grid`` into ``ntime`` coarse bins.

    The split-search kernels use the coarse grid (``ntime`` bins) to cut
    the inner loop work by ``len(time_grid) / ntime``. Leaves still use
    the full grid for CIF / CHF output.

    Returns
    -------
    full_to_split : (len(time_grid),) int32
        For each full-grid index ``i``, the coarse bin it falls into.
        Monotone non-decreasing. Values in ``[0, ntime)``.

    If ``ntime >= len(time_grid)``, returns an identity mapping (no
    coarsening). If quantile bucketing produces fewer than
    ``ntime // 2`` distinct break points (pathological clustering),
    emits a ``UserWarning`` and falls back to uniform time-range
    coarsening via ``np.linspace`` on the grid values.
    """
    n_full = len(time_grid)
    if ntime >= n_full:
        return np.arange(n_full, dtype=np.int32)

    qs = np.linspace(0.0, 1.0, ntime + 1)
    break_values = np.quantile(time_grid, qs)
    break_indices = np.searchsorted(time_grid, break_values, side="left").astype(np.int64)
    break_indices[-1] = n_full
    distinct = len(np.unique(break_indices))
    if distinct < ntime // 2:
        warnings.warn(
            "pathological clustering in time_grid; falling back to uniform time-range coarsening.",
            UserWarning,
            stacklevel=2,
        )
        edges = np.linspace(time_grid[0], time_grid[-1], ntime + 1)
        return np.clip(np.searchsorted(edges, time_grid, side="right") - 1, 0, ntime - 1).astype(
            np.int32
        )

    full_to_split = np.empty(n_full, dtype=np.int32)
    for j in range(ntime):
        full_to_split[int(break_indices[j]) : int(break_indices[j + 1])] = j
    return full_to_split
