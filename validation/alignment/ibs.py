"""IPCW-weighted Integrated Brier Score for competing-risks CIF predictions.

Single scalar summary per (lib, seed) that complements the pointwise CIF gap:
whereas |ΔCIF| measures pointwise agreement between two CIF surfaces, IBS
integrates prediction error against observed events with IPCW weighting, so
two libs with equal |ΔCIF| distributions but differently-aligned errors
against events can still disagree in IBS. Adding IBS to the gate gives an
orthogonal defensibility signal (squared-error + event-alignment) on top of
the raw |ΔCIF| view.

Formula at time t for cause-1 competing risks:

    BS(t) = (1/n) Σ_i w_i(t) · (1{T_i ≤ t, δ_i = 1} - F_1(t|X_i))²

where IPCW weights come from the Kaplan-Meier estimate G of the censoring
distribution on the training set:

    w_i(t) = 1 / G(T_i-)      if T_i ≤ t and δ_i ≠ 0
           = 1 / G(t-)        if T_i > t
           = 0                if T_i ≤ t and δ_i = 0

IBS = mean over ref_grid of BS(t) (uniform time weighting, trapezoidal
integration, normalized by (t_max - t_min)).
"""

from __future__ import annotations

import numpy as np


def _kaplan_meier_censoring(
    train_time: np.ndarray, train_event: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """KM estimator of the censoring distribution.

    Treats ``train_event == 0`` rows as the event of interest (censored) for
    the KM; returns ``(unique_times, G)`` where ``G[k]`` is G(unique_times[k]).
    """
    cens = (train_event == 0).astype(np.float64)
    order = np.argsort(train_time)
    t_sorted = train_time[order]
    c_sorted = cens[order]
    unique_t, inv = np.unique(t_sorted, return_inverse=True)
    n = len(t_sorted)
    # Number at risk just before each unique time = number of obs with T >= t.
    at_risk = n - np.searchsorted(t_sorted, unique_t, side="left")
    events = np.bincount(inv, weights=c_sorted, minlength=len(unique_t))
    # Guard against at_risk=0 (shouldn't happen by construction but be safe).
    factors = np.where(at_risk > 0, 1.0 - events / np.maximum(at_risk, 1), 1.0)
    G = np.cumprod(factors)
    return unique_t, G


def _G_left(unique_t: np.ndarray, G: np.ndarray, t: np.ndarray) -> np.ndarray:
    """Left-continuous KM lookup: G(t-) = G at the largest unique_t strictly less than t.

    For t ≤ unique_t[0] returns 1.0 (censoring survival is 1 before first obs).
    """
    t = np.atleast_1d(t)
    idx = np.searchsorted(unique_t, t, side="left") - 1
    below = t <= unique_t[0]
    idx_clipped = np.clip(idx, 0, len(G) - 1)
    out = G[idx_clipped]
    return np.where(below, 1.0, out)


def _brier_vectorized(
    cif_grid: np.ndarray,
    ref_grid: np.ndarray,
    test_time: np.ndarray,
    test_event: np.ndarray,
    unique_t: np.ndarray,
    G: np.ndarray,
) -> np.ndarray:
    """BS(t) for each t in ref_grid. cif_grid is (n_test, n_ref). Returns (n_ref,)."""
    inv_G_T = 1.0 / np.clip(_G_left(unique_t, G, test_time), 1e-12, None)  # (n_test,)
    inv_G_ref = 1.0 / np.clip(_G_left(unique_t, G, ref_grid), 1e-12, None)  # (n_ref,)

    t_le = test_time[:, None] <= ref_grid[None, :]  # (n_test, n_ref)

    # w(i,t) = inv_G_T[i] if T_i≤t and δ≠0; inv_G_ref[t] if T_i>t; 0 if T_i≤t and δ=0.
    w = np.where(
        t_le,
        np.where((test_event != 0)[:, None], inv_G_T[:, None], 0.0),
        inv_G_ref[None, :],
    )
    y = (t_le & (test_event == 1)[:, None]).astype(np.float64)
    return (w * (y - cif_grid) ** 2).mean(axis=0)


def compute_ibs(
    cif_grid: np.ndarray,
    ref_grid: np.ndarray,
    test_time: np.ndarray,
    test_event: np.ndarray,
    train_time: np.ndarray,
    train_event: np.ndarray,
) -> float:
    """Integrated Brier Score for cause-1 CIF predictions on ref_grid.

    Integration is truncated at ``tau = max(train_time[train_event != 0])`` —
    the last observed event time on the training set. Beyond tau the censoring
    KM estimator G(t) is unreliable (often collapses toward 0), which makes
    IPCW weights 1/G(t) explode; datasets with heavy late censoring (e.g.
    follic) produce meaningless IBS blow-ups without truncation. The
    truncation convention matches scikit-survival's ``integrated_brier_score``.
    """
    unique_t, G = _kaplan_meier_censoring(train_time, train_event)

    train_event_times = train_time[train_event != 0]
    if len(train_event_times) == 0:
        raise ValueError("training set has no events — cannot compute IBS")
    tau = float(train_event_times.max())
    mask = ref_grid <= tau
    if mask.sum() < 2:
        raise ValueError(
            f"ref_grid has <2 points under tau={tau}; "
            f"ref_grid[0..-1]={ref_grid[0]:.3f}..{ref_grid[-1]:.3f}"
        )
    grid = ref_grid[mask]
    cif_trunc = cif_grid[:, mask]

    bs = _brier_vectorized(cif_trunc, grid, test_time, test_event, unique_t, G)
    t_range = float(grid[-1] - grid[0])
    if t_range <= 0:
        return float(bs.mean())
    # Inline trapezoidal rule — portable across numpy 1.24+ (np.trapezoid is 2.0+,
    # np.trapz is deprecated in 2.0+).
    trapz = float(np.sum(0.5 * (bs[1:] + bs[:-1]) * np.diff(grid)))
    return trapz / t_range
