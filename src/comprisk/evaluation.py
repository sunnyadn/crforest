"""Competing-risks evaluation: time-dep AUC + Brier/IBS + iAUC.

Modelled on R ``riskRegression::Score`` for competing risks. One entry
point :func:`score_cr` accepts a dict of model name to (n_test,
n_eval_times) CIF probability matrix and returns per-(model, time) AUC
and Brier with optional bootstrap CIs, plus integrated AUC and IBS.

References
----------
Blanche, P., Dartigues, J.-F., Jacqmin-Gadda, H. (2013). "Estimating and
comparing time-dependent areas under receiver operating characteristic
curves for censored event times with competing risks." *Statistics in
Medicine* 32(30): 5381-5397.

Gerds, T.A., Schumacher, M. (2006). "Consistent estimation of the
expected Brier score in general survival models with right-censored
event times." *Biometrical Journal* 48(6): 1029-1040.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass

import numpy as np
import pandas as pd
from joblib import Parallel, delayed

from comprisk.metrics import _ghat_minus

__all__ = ["ScoreResult", "score_cr"]


@dataclass
class ScoreResult:
    """Container for :func:`score_cr` output.

    Attributes
    ----------
    auc : pandas.DataFrame
        Columns ``model, times, AUC, lower, upper``. ``lower``/``upper``
        are ``NaN`` when ``n_bootstrap == 0``.
    brier : pandas.DataFrame
        Columns ``model, times, Brier, lower, upper``.
    iauc : pandas.DataFrame
        Columns ``model, iAUC, lower, upper``.
    ibs : pandas.DataFrame
        Columns ``model, IBS, lower, upper``.
    """

    auc: pd.DataFrame
    brier: pd.DataFrame
    iauc: pd.DataFrame
    ibs: pd.DataFrame


# ---------------------------------------------------------------------------
# KM-of-censoring under CR with the events-first tie convention
# ---------------------------------------------------------------------------


def _km_censoring_cr(time: np.ndarray, event: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """KM survivor of the censoring distribution under competing risks.

    Treats every observation with ``event != 0`` as a non-censoring
    "event" (i.e. cause-of-interest *and* competing events both shrink
    the at-risk set without contributing to the censoring decrement).
    At a tied time ``t`` the events-first convention places non-
    censoring events just before ``t``, so the censoring decrement uses
    ``n_risk - d_event`` in the denominator.
    """
    n = len(time)
    if n == 0:
        return np.empty(0, dtype=float), np.empty(0, dtype=float)

    order = np.argsort(time, kind="stable")
    t_sorted = time[order]
    e_sorted = event[order]
    is_event = (e_sorted != 0).astype(np.int64)
    is_cens = (e_sorted == 0).astype(np.int64)

    t_unique, first_idx = np.unique(t_sorted, return_index=True)
    bounds = np.append(first_idx, n)

    G = np.empty(t_unique.shape[0], dtype=float)
    surv = 1.0
    for k in range(t_unique.shape[0]):
        s, e = int(bounds[k]), int(bounds[k + 1])
        n_risk = n - s
        d_event = int(is_event[s:e].sum())
        c = int(is_cens[s:e].sum())
        if c > 0:
            denom = n_risk - d_event
            surv = 0.0 if denom <= 0 else surv * (1.0 - c / denom)
        G[k] = surv
    return t_unique, G


def _ghat_at(t_unique: np.ndarray, G: np.ndarray, query: np.ndarray) -> np.ndarray:
    """Right-continuous KM survivor at each ``query`` time. ``1`` below
    the first knot; equal to ``G[k]`` for the largest knot ``<= q``."""
    if t_unique.size == 0:
        return np.ones(query.shape, dtype=float)
    idx = np.searchsorted(t_unique, query, side="right")
    out = np.ones(query.shape, dtype=float)
    has = idx > 0
    out[has] = G[idx[has] - 1]
    return out


# ---------------------------------------------------------------------------
# Per-time AUC + Brier for one model on one (sub)sample
# ---------------------------------------------------------------------------


def _per_time_auc_brier(
    probs: np.ndarray,
    time: np.ndarray,
    event: np.ndarray,
    eval_times: np.ndarray,
    t_unique: np.ndarray,
    G: np.ndarray,
    cause: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Time-dep AUC (Blanche 2013) + IPCW Brier (Gerds-Schumacher 2006,
    CR-extended) for one model.

    Parameters
    ----------
    probs : (n, T) ndarray
        Predicted CIF for ``cause`` at each ``eval_times[k]``.
    time, event : (n,) ndarrays
    eval_times : (T,) ndarray
    t_unique, G : ndarrays
        KM-censoring estimator (sorted unique knots + survivor).
    cause : int
        Cause of interest.

    Returns
    -------
    (auc, brier) : (T,) ndarrays
    """
    n, T = probs.shape
    auc_out = np.full(T, np.nan)
    brier_out = np.full(T, np.nan)

    G_left_T = _ghat_minus(t_unique, G, time)
    G_at_eval = _ghat_at(t_unique, G, eval_times)

    is_known = event != 0
    is_cause = event == cause

    for k in range(T):
        t = float(eval_times[k])
        p = probs[:, k]
        G_t = float(G_at_eval[k])
        if not np.isfinite(G_t) or G_t <= 0.0:
            continue

        # is_case: experienced cause-of-interest by t
        # is_cont: still at risk past t (event-free past t)
        # is_other_event: experienced any event by t with known status (cases or competing-by-t)
        is_case = is_cause & (time <= t)
        is_cont = time > t
        is_event_by_t = is_known & (time <= t)

        # Case-side weight 1/G(T_i^-); guard against zero G_left_T.
        # Subjects censored before t (event==0 & time<=t) contribute zero.
        with np.errstate(divide="ignore", invalid="ignore"):
            inv_GleftT = np.where(G_left_T > 0, 1.0 / G_left_T, 0.0)

        # ---- IPCW Brier ----
        # Three contribution paths (subjects censored before t are dropped):
        #   case        : (1 - p)^2 / G(T_i^-)
        #   competing   : (0 - p)^2 / G(T_i^-)
        #   control     : (0 - p)^2 / G(t)
        is_comp_by_t = is_event_by_t & ~is_cause
        sq_case = np.where(is_case, (1.0 - p) ** 2 * inv_GleftT, 0.0)
        sq_comp = np.where(is_comp_by_t, p**2 * inv_GleftT, 0.0)
        sq_cont = np.where(is_cont, p**2 / G_t, 0.0)
        brier_out[k] = float(sq_case.sum() + sq_comp.sum() + sq_cont.sum()) / n

        # ---- IPCW AUC (Blanche 2013, cumulative-cases / dynamic-controls) ----
        # Controls are everyone whose status at t is observed and is NOT
        # a cause-of-interest event by t. Two control sub-populations:
        #   - T_j > t (event-free past t)              weight 1 / G(t)
        #   - competing event at time <= t             weight 1 / G(T_j^-)
        case_idx = np.flatnonzero(is_case)
        cont_surv_idx = np.flatnonzero(is_cont)
        cont_comp_idx = np.flatnonzero(is_comp_by_t)
        if case_idx.size == 0 or (cont_surv_idx.size + cont_comp_idx.size) == 0:
            continue

        w_i = inv_GleftT[case_idx]
        if w_i.sum() <= 0:
            continue

        cont_idx = np.concatenate([cont_surv_idx, cont_comp_idx])
        w_j = np.empty(cont_idx.size, dtype=float)
        w_j[: cont_surv_idx.size] = 1.0 / G_t
        w_j[cont_surv_idx.size :] = inv_GleftT[cont_comp_idx]

        # Sort-based concordance count weighted by w_i, w_j.
        p_i = p[case_idx]
        p_j = p[cont_idx]
        order = np.argsort(p_j, kind="stable")
        p_j_sorted = p_j[order]
        w_j_sorted = w_j[order]
        cum_w = np.concatenate(([0.0], np.cumsum(w_j_sorted)))

        left = np.searchsorted(p_j_sorted, p_i, side="left")
        right = np.searchsorted(p_j_sorted, p_i, side="right")
        W_lt = cum_w[left]
        W_eq = cum_w[right] - cum_w[left]

        denom = float(w_i.sum() * w_j.sum())
        if denom <= 0:
            continue
        numer = float((w_i * (W_lt + 0.5 * W_eq)).sum())
        auc_out[k] = numer / denom

    return auc_out, brier_out


def _trap_mean(times: np.ndarray, vals: np.ndarray) -> float:
    """Trapezoidal time-average of ``vals`` over ``times`` (NaN-safe).

    Returns ``NaN`` if fewer than two finite values remain or the time
    span is zero.
    """
    times = np.asarray(times, dtype=float)
    vals = np.asarray(vals, dtype=float)
    mask = np.isfinite(vals) & np.isfinite(times)
    times = times[mask]
    vals = vals[mask]
    if times.size < 2:
        return float("nan")
    span = float(times[-1] - times[0])
    if span <= 0:
        return float("nan")
    return float(np.trapezoid(vals, times) / span)


# ---------------------------------------------------------------------------
# Bootstrap workhorse
# ---------------------------------------------------------------------------


def _bootstrap_one(
    seed: int,
    pred_arrs: list,
    time: np.ndarray,
    event: np.ndarray,
    eval_times: np.ndarray,
    cause: int,
) -> np.ndarray:
    """One bootstrap iteration. Refits KM-censoring on the resample.

    Returns a ``(n_models, T, 2)`` array stacking ``[auc, brier]`` along
    the last axis.
    """
    rng = np.random.default_rng(seed)
    n = time.size
    idx = rng.integers(0, n, size=n)
    t_b = time[idx]
    e_b = event[idx]
    tu_b, G_b = _km_censoring_cr(t_b, e_b)
    out = np.full((len(pred_arrs), eval_times.size, 2), np.nan)
    for m, p in enumerate(pred_arrs):
        a, b = _per_time_auc_brier(p[idx], t_b, e_b, eval_times, tu_b, G_b, cause)
        out[m, :, 0] = a
        out[m, :, 1] = b
    return out


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------


def score_cr(
    predictions: Mapping[str, np.ndarray],
    test_time: np.ndarray,
    test_event: np.ndarray,
    eval_times: np.ndarray,
    *,
    cause: int = 1,
    metrics: Sequence[str] = ("auc", "brier"),
    n_bootstrap: int = 0,
    confidence_level: float = 0.95,
    n_jobs: int = -1,
    random_state: int | None = None,
) -> ScoreResult:
    """Competing-risks time-dep AUC, Brier, IBS, iAUC.

    One-call replacement for the AUC/Brier block of R
    ``riskRegression::Score`` in CR mode. Accepts an arbitrary number of
    candidate models as a dict of name to (n_test, n_eval_times) CIF
    probability matrix at the cause of interest.

    Parameters
    ----------
    predictions : Mapping[str, ndarray]
        ``model_name -> CIF`` array of shape ``(n_test, n_eval_times)``
        evaluated at the same ``eval_times``.
    test_time : array-like of float
        Observed time per test subject.
    test_event : array-like of int
        Event code per test subject. ``0`` is censoring; ``cause`` is
        the cause of interest; any other positive integer is a competing
        event.
    eval_times : array-like of float
        Times at which the metrics are evaluated. Must align with the
        columns of every prediction matrix.
    cause : int, default 1
        Cause of interest.
    metrics : sequence of str, default ``("auc", "brier")``
        Subset of ``{"auc", "brier"}``. iAUC is reported when ``"auc"``
        is present; IBS when ``"brier"`` is present.
    n_bootstrap : int, default 0
        Number of bootstrap resamples for 95% CIs. ``0`` skips CI
        computation and leaves ``lower``/``upper`` as ``NaN``.
    confidence_level : float, default 0.95
    n_jobs : int, default -1
        Number of parallel workers for the bootstrap loop.
    random_state : int or None, default None
        Seed for the bootstrap.

    Returns
    -------
    ScoreResult
    """
    test_time = np.asarray(test_time, dtype=float)
    test_event = np.asarray(test_event)
    eval_times = np.asarray(eval_times, dtype=float)
    n = test_time.size
    T = eval_times.size

    if test_event.size != n:
        raise ValueError("test_time and test_event must have the same length")
    if T == 0:
        raise ValueError("eval_times must be non-empty")
    if not np.all(np.diff(eval_times) > 0):
        raise ValueError("eval_times must be strictly increasing")

    metric_set = {m.lower() for m in metrics}
    bad = metric_set - {"auc", "brier"}
    if bad:
        raise ValueError(f"unknown metrics: {sorted(bad)}; expected subset of {{'auc','brier'}}")
    if not metric_set:
        raise ValueError("metrics must be non-empty")

    if not (0.0 < confidence_level < 1.0):
        raise ValueError("confidence_level must be in (0, 1)")

    if not predictions:
        raise ValueError("predictions must contain at least one model")

    model_names: list[str] = list(predictions.keys())
    pred_arrs: list[np.ndarray] = []
    for name in model_names:
        arr = np.asarray(predictions[name], dtype=float)
        if arr.shape != (n, T):
            raise ValueError(f"predictions[{name!r}] shape {arr.shape} != expected ({n}, {T})")
        pred_arrs.append(arr)

    tu, G = _km_censoring_cr(test_time, test_event)

    # Point estimates
    n_m = len(model_names)
    auc_pt = np.full((n_m, T), np.nan)
    brier_pt = np.full((n_m, T), np.nan)
    for m, p in enumerate(pred_arrs):
        a, b = _per_time_auc_brier(p, test_time, test_event, eval_times, tu, G, cause)
        auc_pt[m] = a
        brier_pt[m] = b

    # Bootstrap CIs
    auc_lo = np.full_like(auc_pt, np.nan)
    auc_hi = np.full_like(auc_pt, np.nan)
    brier_lo = np.full_like(brier_pt, np.nan)
    brier_hi = np.full_like(brier_pt, np.nan)
    iauc_lo = np.full(n_m, np.nan)
    iauc_hi = np.full(n_m, np.nan)
    ibs_lo = np.full(n_m, np.nan)
    ibs_hi = np.full(n_m, np.nan)

    if n_bootstrap > 0:
        ss = np.random.SeedSequence(random_state)
        seeds = ss.generate_state(n_bootstrap)
        boot = Parallel(n_jobs=n_jobs)(
            delayed(_bootstrap_one)(
                int(seeds[b]), pred_arrs, test_time, test_event, eval_times, cause
            )
            for b in range(n_bootstrap)
        )
        boot = np.stack(boot, axis=0)  # (B, n_m, T, 2)
        alpha = 1.0 - confidence_level
        lo_q = 100.0 * alpha / 2.0
        hi_q = 100.0 * (1.0 - alpha / 2.0)
        auc_b = boot[..., 0]
        brier_b = boot[..., 1]
        with np.errstate(invalid="ignore"):
            auc_lo = np.nanpercentile(auc_b, lo_q, axis=0)
            auc_hi = np.nanpercentile(auc_b, hi_q, axis=0)
            brier_lo = np.nanpercentile(brier_b, lo_q, axis=0)
            brier_hi = np.nanpercentile(brier_b, hi_q, axis=0)

        iauc_b = np.array(
            [[_trap_mean(eval_times, auc_b[b, m]) for m in range(n_m)] for b in range(n_bootstrap)]
        )
        ibs_b = np.array(
            [
                [_trap_mean(eval_times, brier_b[b, m]) for m in range(n_m)]
                for b in range(n_bootstrap)
            ]
        )
        with np.errstate(invalid="ignore"):
            iauc_lo = np.nanpercentile(iauc_b, lo_q, axis=0)
            iauc_hi = np.nanpercentile(iauc_b, hi_q, axis=0)
            ibs_lo = np.nanpercentile(ibs_b, lo_q, axis=0)
            ibs_hi = np.nanpercentile(ibs_b, hi_q, axis=0)

    # Pack DataFrames
    auc_rows = []
    brier_rows = []
    iauc_rows = []
    ibs_rows = []
    for m, name in enumerate(model_names):
        for k in range(T):
            auc_rows.append(
                {
                    "model": name,
                    "times": float(eval_times[k]),
                    "AUC": float(auc_pt[m, k]),
                    "lower": float(auc_lo[m, k]),
                    "upper": float(auc_hi[m, k]),
                }
            )
            brier_rows.append(
                {
                    "model": name,
                    "times": float(eval_times[k]),
                    "Brier": float(brier_pt[m, k]),
                    "lower": float(brier_lo[m, k]),
                    "upper": float(brier_hi[m, k]),
                }
            )
        iauc_rows.append(
            {
                "model": name,
                "iAUC": _trap_mean(eval_times, auc_pt[m]),
                "lower": float(iauc_lo[m]),
                "upper": float(iauc_hi[m]),
            }
        )
        ibs_rows.append(
            {
                "model": name,
                "IBS": _trap_mean(eval_times, brier_pt[m]),
                "lower": float(ibs_lo[m]),
                "upper": float(ibs_hi[m]),
            }
        )

    auc_cols = ["model", "times", "AUC", "lower", "upper"]
    brier_cols = ["model", "times", "Brier", "lower", "upper"]
    iauc_cols = ["model", "iAUC", "lower", "upper"]
    ibs_cols = ["model", "IBS", "lower", "upper"]
    auc_df = (
        pd.DataFrame(auc_rows, columns=auc_cols)
        if "auc" in metric_set
        else pd.DataFrame(columns=auc_cols)
    )
    brier_df = (
        pd.DataFrame(brier_rows, columns=brier_cols)
        if "brier" in metric_set
        else pd.DataFrame(columns=brier_cols)
    )
    iauc_df = (
        pd.DataFrame(iauc_rows, columns=iauc_cols)
        if "auc" in metric_set
        else pd.DataFrame(columns=iauc_cols)
    )
    ibs_df = (
        pd.DataFrame(ibs_rows, columns=ibs_cols)
        if "brier" in metric_set
        else pd.DataFrame(columns=ibs_cols)
    )
    return ScoreResult(auc=auc_df, brier=brier_df, iauc=iauc_df, ibs=ibs_df)
