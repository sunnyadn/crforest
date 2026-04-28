"""Quantile-based feature binning for histogram-mode trees.

``fit_bin_edges`` computes per-feature *midpoint* edges on training X —
midpoints between consecutive unique values (lossless case) or interior
quantile cutpoints (lossy case). ``apply_bins`` uses those edges to map
any X to uint8 bin indices via ``searchsorted(side="right")``. The
midpoint convention matches reference-mode routing (splits at midpoints
of consecutive unique values), so default and reference modes produce
identical training-split decisions AND identical test-sample routing.

Bin count for feature j = ``len(edges[j]) + 1``.
"""

from __future__ import annotations

import numpy as np


def fit_bin_edges(X: np.ndarray, n_bins: int = 256) -> list[np.ndarray]:
    """Per-feature midpoint edges for histogram binning.

    If a feature has ``n_uniques <= n_bins``, edges are the midpoints
    between consecutive unique values (``n_uniques - 1`` edges giving
    ``n_uniques`` bins — matches the split-candidate count reference mode
    evaluates).

    If ``n_uniques > n_bins``, edges are ``n_bins - 1`` interior quantile
    cutpoints over unique values, giving exactly ``n_bins`` bins.

    Returns
    -------
    edges : list[ndarray]
        ``edges[j]`` is a sorted 1-D float64 array of midpoint/cutpoint
        thresholds. ``len(edges[j])`` ≤ ``n_bins - 1``.
    """
    if not (2 <= n_bins <= 256):
        raise ValueError(f"n_bins must be in [2, 256]; got {n_bins}")
    X = np.asarray(X, dtype=np.float64)
    if X.ndim != 2:
        raise ValueError(f"X must be 2-D; got ndim={X.ndim}")
    p = X.shape[1]
    edges: list[np.ndarray] = []
    for j in range(p):
        uniq = np.unique(X[:, j])
        if len(uniq) <= n_bins:
            # Midpoints between consecutive unique values (empty array if
            # only one unique value, producing a single-bin feature).
            edges.append((uniq[:-1] + uniq[1:]) / 2.0)
        else:
            # n_bins - 1 interior quantile cutpoints.
            qs = np.linspace(0.0, 1.0, n_bins + 1)[1:-1]
            edges.append(np.quantile(uniq, qs))
    return edges


def apply_bins(X: np.ndarray, edges: list[np.ndarray]) -> np.ndarray:
    """Map a (n, p) float array to (n, p) uint8 bin indices using ``edges``.

    ``bin_idx = searchsorted(edges[j], X[:, j], side="right")`` clipped
    into ``[0, n_bins - 1]`` where ``n_bins = len(edges[j]) + 1``.
    """
    X = np.asarray(X, dtype=np.float64)
    if X.ndim != 2:
        raise ValueError(f"X must be 2-D; got ndim={X.ndim}")
    if X.shape[1] != len(edges):
        raise ValueError(f"X has {X.shape[1]} features but edges has {len(edges)}")
    n, p = X.shape
    out = np.zeros((n, p), dtype=np.uint8)
    for j, e in enumerate(edges):
        n_bins = len(e) + 1
        if n_bins > 256:
            raise ValueError(
                f"edges[{j}] has {len(e)} entries, implying {n_bins} bins (must be ≤ 256)"
            )
        idx = np.searchsorted(e, X[:, j], side="right")
        out[:, j] = np.clip(idx, 0, n_bins - 1).astype(np.uint8)
    return out


def bin_to_threshold(edges: list[np.ndarray], feature: int, bin_idx: int) -> float:
    """Upper boundary of the bin, in original feature units.

    Bin ``b`` covers values ``v`` in the half-open interval from the
    previous edge up to and including ``edges[feature][b]``. The first bin
    has no finite lower boundary; the last bin has no finite upper boundary
    and this function returns ``+inf`` for it.
    """
    col_edges = edges[feature]
    n_bins = len(col_edges) + 1
    if bin_idx < 0 or bin_idx >= n_bins:
        raise ValueError(f"bin_idx={bin_idx} out of range [0, {n_bins - 1}]")
    if bin_idx == n_bins - 1:
        return float("inf")
    return float(col_edges[bin_idx])
