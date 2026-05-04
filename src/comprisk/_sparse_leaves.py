"""Sparse at-rest storage for per-leaf event counts and at-risk counts.

Each histogram-mode leaf carries its counts in two compact dataclasses:

- ``SparseEventCounts`` — COO triplets ``(cause, time, values)`` for the
  non-zero cells of an ``(n_causes, n_time_bins)`` uint32 array. Profiling
  measured 99.75% sparsity on realistic workloads, so a dense ``(2, 200)``
  array (1600 bytes) becomes ~4k + 40 header bytes for typical k <= 20.
- ``SparseAtRisk`` — step-function encoding of the monotone non-increasing
  at-risk vector. Stores ``(time, values)`` only at breakpoints; materializes
  back to dense via forward-fill.

Densification promotes to uint32 so downstream arithmetic dtype in
``_estimators`` is unchanged, preserving bit-identity.

Dtype selection happens at ``to_sparse_*``:
- ``event_counts.values`` — ``uint8`` if max cell ≤ 255 else ``uint16``.
- ``at_risk.values`` — ``uint16`` if max ≤ 65_535 else ``uint32``.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class SparseEventCounts:
    cause: np.ndarray  # (k,) uint8
    time: np.ndarray  # (k,) uint16
    values: np.ndarray  # (k,) uint8 or uint16


@dataclass(frozen=True)
class SparseAtRisk:
    time: np.ndarray  # (m,) uint16
    values: np.ndarray  # (m,) uint16 or uint32


def to_sparse_event_counts(dense: np.ndarray) -> SparseEventCounts:
    """COO-compress a ``(n_causes, n_time_bins)`` uint32 dense array."""
    if dense.ndim != 2:
        raise ValueError(f"dense must be 2-D (n_causes, n_time_bins); got ndim={dense.ndim}")
    causes, times = np.nonzero(dense)
    vals = dense[causes, times]
    max_val = int(vals.max()) if len(vals) > 0 else 0
    values_dtype = np.uint8 if max_val <= 255 else np.uint16
    return SparseEventCounts(
        cause=causes.astype(np.uint8),
        time=times.astype(np.uint16),
        values=vals.astype(values_dtype),
    )


def to_dense_event_counts(sparse: SparseEventCounts, n_causes: int, n_time_bins: int) -> np.ndarray:
    """Reconstruct the dense ``(n_causes, n_time_bins)`` uint32 array."""
    dense = np.zeros((n_causes, n_time_bins), dtype=np.uint32)
    if len(sparse.cause) > 0:
        dense[sparse.cause, sparse.time] = sparse.values
    return dense


def to_sparse_at_risk(dense: np.ndarray) -> SparseAtRisk:
    """Step-function compress a monotone non-increasing at-risk vector.

    Stores (time, value) at every index where ``dense[t] != dense[t - 1]``,
    including the implicit ``dense[0]`` breakpoint.
    """
    if dense.ndim != 1:
        raise ValueError(f"dense must be 1-D; got ndim={dense.ndim}")
    if len(dense) == 0:
        return SparseAtRisk(
            time=np.empty(0, dtype=np.uint16),
            values=np.empty(0, dtype=np.uint16),
        )
    change_mask = np.empty(len(dense), dtype=bool)
    change_mask[0] = True
    change_mask[1:] = dense[1:] != dense[:-1]
    times = np.flatnonzero(change_mask).astype(np.uint16)
    vals = dense[times]
    max_val = int(vals.max()) if len(vals) > 0 else 0
    values_dtype = np.uint16 if max_val <= 65_535 else np.uint32
    return SparseAtRisk(time=times, values=vals.astype(values_dtype))


def to_dense_at_risk(sparse: SparseAtRisk, n_time_bins: int) -> np.ndarray:
    """Reconstruct the dense ``(n_time_bins,)`` uint32 array via forward fill."""
    if len(sparse.time) == 0:
        return np.zeros(n_time_bins, dtype=np.uint32)
    # Encoder guarantees times[0] == 0, so widths sum to n_time_bins.
    times = sparse.time
    widths = np.empty(len(times), dtype=np.int64)
    widths[:-1] = np.diff(times)
    widths[-1] = n_time_bins - int(times[-1])
    return np.repeat(sparse.values, widths).astype(np.uint32, copy=False)
