"""Gray's K-sample test for equality of cumulative incidence functions.

Gray's test (Gray, 1988) is the cumulative-incidence analogue of the log-rank
test for survival data. It compares the cause-specific cumulative incidence
functions ``F_{1,g}(t)`` across ``K >= 2`` groups under right-censored
competing-risks observation. The null hypothesis is that the CIF for the
cause of interest is the same across all groups.

The test statistic is built from a (K-1)-dimensional score vector ``S`` and
a (K-1) x (K-1) covariance matrix ``V`` accumulated over the unique observed
times. ``S`` measures, for each non-pivot group, the weighted divergence of
the group's cause-1 hazard from the pooled subdistribution-hazard prediction.
``V`` is the covariance estimator obtained from counting-process martingale
theory by tracking how each group's cumulative-influence row contributes to
score variance through cause-1 events and through the censoring induced by
competing events. The Wald-type quadratic form ``T = S^T V^{-1} S`` is
asymptotically chi-square with ``K-1`` degrees of freedom under the null.

This module is a clean-room implementation written directly from the
mathematical statement of Gray's procedure plus standard counting-process
martingale theory; no GPL-licensed third-party source code (Fortran, R, or
otherwise) was consulted while writing it. Variable names follow the
statistical literature.

References
----------
Gray, R.J. (1988). "A class of K-sample tests for comparing the cumulative
incidence of a competing risk." *Annals of Statistics* 16(3):1141-1154.

Andersen, P.K., Borgan, O., Gill, R.D., Keiding, N. (1993). *Statistical
Models Based on Counting Processes*. Springer.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.stats import chi2

__all__ = ["GrayTestResult", "gray_test"]


@dataclass
class GrayTestResult:
    """Outcome of :func:`gray_test`.

    Attributes
    ----------
    stat : float
        Wald-type chi-square statistic ``S^T V^{-1} S``.
    pvalue : float
        Upper-tail p-value from a chi-square distribution with ``df``
        degrees of freedom.
    df : int
        Degrees of freedom, equal to ``K - 1`` for ``K`` groups.
    score : np.ndarray of shape (K-1,)
        Score vector for the non-pivot groups.
    var : np.ndarray of shape (K-1, K-1)
        Estimated covariance matrix of ``score``.
    n_groups : int
        Number of distinct groups ``K``.
    rho : float
        Weight exponent ``rho`` actually used; the per-time weight is
        ``(1 - Fbar(t-))^rho``.
    """

    stat: float
    pvalue: float
    df: int
    score: np.ndarray
    var: np.ndarray
    n_groups: int
    rho: float


def _coerce_inputs(
    time: np.ndarray,
    event: np.ndarray,
    group: np.ndarray,
    cause: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, int]:
    """Validate inputs and reduce them to canonical ``m, group_idx`` arrays.

    Returns sorted ``time`` (ascending), the recoded status array
    ``m in {0, 1, 2}`` (0 = censored, 1 = cause-of-interest, 2 = competing
    event), the integer group index ``group_idx in {0, ..., K-1}``, and
    ``K``.
    """
    time = np.asarray(time, dtype=np.float64).ravel()
    event = np.asarray(event).ravel()
    group = np.asarray(group).ravel()
    if not (time.size == event.size == group.size):
        raise ValueError(
            "time, event, and group must have the same length; "
            f"got {time.size}, {event.size}, {group.size}"
        )
    if time.size == 0:
        raise ValueError("inputs are empty")

    # Map group labels (possibly strings) to {0, ..., K-1} via stable np.unique.
    _unique_groups, group_idx = np.unique(group, return_inverse=True)
    n_groups = int(_unique_groups.size)
    if n_groups < 2:
        raise ValueError(f"Gray's test requires at least 2 groups; got {n_groups}")

    # Recode event into m: 0 censored, 1 = cause-of-interest, 2 = any other event.
    event_int = np.asarray(event)
    m = np.zeros(event_int.shape[0], dtype=np.int64)
    is_event = event_int != 0
    is_cause = is_event & (event_int == cause)
    m[is_cause] = 1
    m[is_event & ~is_cause] = 2

    # Sort by time so we can sweep unique times in ascending order.
    order = np.argsort(time, kind="mergesort")
    return time[order], m[order], group_idx.astype(np.int64)[order], n_groups


def _per_time_counts(
    time_sorted: np.ndarray,
    m_sorted: np.ndarray,
    group_idx_sorted: np.ndarray,
    n_groups: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Reduce row-level data to per-(unique-time, group) tabulations.

    Returns
    -------
    unique_times : (T,) float
        Sorted distinct observed times.
    cause1_count : (T, K) int
        Cause-of-interest event counts per time and group.
    cause2_count : (T, K) int
        Competing event counts per time and group.
    censored_count : (T, K) int
        Censored counts per time and group; needed to drain the risk set
        even when no event happens.
    """
    unique_times, inverse = np.unique(time_sorted, return_inverse=True)
    n_times = unique_times.size

    cause1 = np.zeros((n_times, n_groups), dtype=np.int64)
    cause2 = np.zeros((n_times, n_groups), dtype=np.int64)
    cens = np.zeros((n_times, n_groups), dtype=np.int64)

    # np.add.at gives unbuffered scatter-add for repeated (t, g) pairs.
    is_c1 = m_sorted == 1
    is_c2 = m_sorted == 2
    is_cs = m_sorted == 0
    np.add.at(cause1, (inverse[is_c1], group_idx_sorted[is_c1]), 1)
    np.add.at(cause2, (inverse[is_c2], group_idx_sorted[is_c2]), 1)
    np.add.at(cens, (inverse[is_cs], group_idx_sorted[is_cs]), 1)
    return unique_times, cause1, cause2, cens


def gray_test(
    time,
    event,
    group,
    *,
    cause: int = 1,
    rho: float = 0.0,
) -> GrayTestResult:
    """Gray's K-sample test for equality of cumulative incidence functions.

    Tests the null hypothesis that the cumulative incidence function for the
    cause of interest is the same across all groups, against the alternative
    that at least one group differs. Censored observations and competing
    events are handled via the subdistribution-hazard formulation of
    Gray (1988).

    Parameters
    ----------
    time : array-like of shape (n,)
        Observed times (event or censoring), non-negative.
    event : array-like of shape (n,)
        Status code per subject. ``0`` denotes censoring; any positive code
        is an event. The code matching ``cause`` is the cause of interest;
        any other positive code is treated as a competing event.
    group : array-like of shape (n,)
        Group labels. May be integer- or string-typed; labels are sorted
        and recoded to ``{0, ..., K-1}``. Requires ``K >= 2``.
    cause : int, default 1
        Event code identifying the cause of interest.
    rho : float, default 0.0
        Power-of-pooled-survival weight exponent. The per-time weight is
        ``W(t) = (1 - Fbar(t-))^rho``. ``rho = 0`` recovers the original
        Gray test.

    Returns
    -------
    GrayTestResult
        Statistic, p-value, degrees of freedom, score vector, score
        covariance, group count, and the ``rho`` actually used.

    Raises
    ------
    ValueError
        If inputs have inconsistent lengths, are empty, or fewer than
        2 distinct groups are present.

    References
    ----------
    Gray, R.J. (1988). "A class of K-sample tests for comparing the
    cumulative incidence of a competing risk." *Annals of Statistics*
    16(3):1141-1154.
    """
    time_sorted, m_sorted, group_idx_sorted, n_groups = _coerce_inputs(time, event, group, cause)
    unique_times, cause1_count, cause2_count, cens_count = _per_time_counts(
        time_sorted, m_sorted, group_idx_sorted, n_groups
    )

    # Per-group running quantities (left-continuous = "minus" pre-jump).
    n_total_per_group = np.bincount(group_idx_sorted, minlength=n_groups).astype(np.float64)
    at_risk_g = n_total_per_group.copy()  # Y_g(t-): updated to Y_g(t) after each step.
    surv_g = np.ones(n_groups, dtype=np.float64)  # S_g(t-) and S_g(t)
    cif_g = np.zeros(n_groups, dtype=np.float64)  # F_g(t-) and F_g(t)

    # Pooled cause-1 CIF, also left/right continuous.
    pooled_cif_minus = 0.0  # Fbar(t-)
    # Score and influence accumulators.
    n_contrast = n_groups - 1  # last group is the contrast pivot.
    score = np.zeros(n_contrast, dtype=np.float64)
    influence = np.zeros((n_contrast, n_groups), dtype=np.float64)  # C_{i,k}
    var3 = np.zeros(n_groups, dtype=np.float64)  # var3_k
    var2 = np.zeros((n_contrast, n_groups), dtype=np.float64)  # var2_{i,k}
    cov = np.zeros((n_contrast, n_contrast), dtype=np.float64)  # V_{i,j} (i >= j)

    n_times = unique_times.size
    for t_idx in range(n_times):
        d1 = cause1_count[t_idx]  # shape (K,) cause-1 events per group
        d2 = cause2_count[t_idx]  # shape (K,) competing events per group
        nc = cens_count[t_idx]  # shape (K,) censorings per group

        total_d1 = int(d1.sum())
        total_d2 = int(d2.sum())

        # Snapshot pre-jump quantities.
        surv_minus = surv_g.copy()
        cif_minus = cif_g.copy()
        Y_minus = at_risk_g.copy()
        active = Y_minus > 0  # groups still being followed at this time

        # If censoring-only tied time: drain risk set, leave everything else
        # unchanged and skip score/variance updates.
        if total_d1 == 0 and total_d2 == 0:
            at_risk_g -= (d1 + d2 + nc).astype(np.float64)
            continue

        # Update group-level KM survival and CIF (right-continuous).
        leaving = (d1 + d2).astype(np.float64)
        with np.errstate(invalid="ignore", divide="ignore"):
            ratio_drop = np.where(active, (Y_minus - leaving) / np.where(active, Y_minus, 1.0), 1.0)
        surv_g = surv_minus * ratio_drop
        with np.errstate(invalid="ignore", divide="ignore"):
            inc = np.where(
                active, surv_minus * d1.astype(np.float64) / np.where(active, Y_minus, 1.0), 0.0
            )
        cif_g = cif_minus + inc

        # Pooled denominators using *pre-jump* Y, S, F.
        with np.errstate(invalid="ignore", divide="ignore"):
            y_over_s = np.where(
                active & (surv_minus > 0), Y_minus / np.where(surv_minus > 0, surv_minus, 1.0), 0.0
            )
        tr = float(y_over_s.sum())  # cause-1 hazard pool
        tq = float((y_over_s * (1.0 - cif_minus)).sum())  # subdistribution pool

        # Per-time weight.
        weight = 1.0 if rho == 0.0 else (1.0 - pooled_cif_minus) ** rho

        # ---- Score update (only K-1 non-pivot groups). ----
        if total_d1 > 0 and tq > 0.0:
            for g in range(n_contrast):
                if not active[g]:
                    continue
                expected_g = (
                    (total_d1 * Y_minus[g] * (1.0 - cif_minus[g]) / (surv_minus[g] * tq))
                    if surv_minus[g] > 0
                    else 0.0
                )
                score[g] += weight * (d1[g] - expected_g)

        # ---- Per-time hazard influence matrix A(t), only when needed. ----
        # A is K x K; rows i correspond to groups (1..K-1) and columns k to
        # all groups (1..K). We only need rows 0..K-2 and columns 0..K-1.
        # We compute A on the fly; it is symmetric so we only need rows.
        # Build a vector u_k = Y_k / S_k(t-) for active groups, else 0.
        u = np.where(active & (surv_minus > 0), y_over_s, 0.0)  # shape (K,)
        # A_{i,j} = -W * u_i * u_j / TR for i != j
        # A_{i,i} =  W * u_i * (1 - u_i / TR)
        # Computing only the influence rows we need (length K-1 by K).
        if tr > 0.0:
            A_rows = -weight * np.outer(u[:n_contrast], u) / tr  # (K-1, K)
            # Diagonal correction: A_{i,i} = W * u_i * (1 - u_i / TR)
            for i in range(n_contrast):
                A_rows[i, i] = weight * u[i] * (1.0 - u[i] / tr)
        else:
            A_rows = np.zeros((n_contrast, n_groups), dtype=np.float64)

        # ---- Cumulative influence matrix C update (only when D_1 > 0). ----
        if total_d1 > 0 and tr > 0.0 and pooled_cif_minus < 1.0:
            denom_c = tr * (1.0 - pooled_cif_minus)
            if denom_c > 0.0:
                influence += A_rows * (total_d1 / denom_c)

        # ---- Variance contributions (a) cause-1 events at this time. ----
        if total_d1 > 0 and tr > 0.0:
            for k in range(n_groups):
                if not active[k]:
                    continue
                if surv_g[k] > 0.0:
                    z_k = 1.0 - (1.0 - (pooled_cif_minus + total_d1 / tr)) / surv_g[k]
                else:
                    z_k = 1.0
                if total_d1 <= 1:
                    finite_corr = 1.0
                else:
                    denom_fc = tr * surv_minus[k] - 1.0
                    finite_corr = 1.0 - (total_d1 - 1) / denom_fc if denom_fc > 0.0 else 1.0
                if Y_minus[k] <= 0.0:
                    continue
                sigma_k = finite_corr * surv_minus[k] * total_d1 / (tr * Y_minus[k])
                var3[k] += z_k * z_k * sigma_k
                # ai_k = A_{i,k} - z_k * C_{i,k} for i in 0..K-2.
                ai_col = A_rows[:, k] - z_k * influence[:, k]  # shape (K-1,)
                var2[:, k] += ai_col * z_k * sigma_k
                # V_{i,j} += ai_k * aj_k * sigma_k, j <= i.
                cov += np.outer(ai_col, ai_col) * sigma_k

        # ---- Variance contributions (b) competing events at this time. ----
        if total_d2 > 0:
            for k in range(n_groups):
                if d2[k] <= 0:
                    continue
                if surv_g[k] <= 0.0:
                    continue
                # Use right-continuous pooled CIF (after the cause-1 jump
                # at this same time) to be consistent with surv_g being
                # right-continuous.
                pooled_cif_now = pooled_cif_minus + (
                    total_d1 / tr if (total_d1 > 0 and tr > 0.0) else 0.0
                )
                z_k = (1.0 - pooled_cif_now) / surv_g[k]
                d2k = int(d2[k])
                if d2k <= 1:
                    finite_corr = 1.0
                else:
                    denom_fc = Y_minus[k] - 1.0
                    finite_corr = 1.0 - (d2k - 1) / denom_fc if denom_fc > 0.0 else 1.0
                Yk = Y_minus[k]
                if Yk <= 0.0:
                    continue
                sigma_k = finite_corr * surv_minus[k] ** 2 * d2k / (Yk * Yk)
                var3[k] += z_k * z_k * sigma_k
                ti_col = z_k * influence[:, k]  # shape (K-1,)
                var2[:, k] -= ti_col * z_k * sigma_k
                cov += np.outer(ti_col, ti_col) * sigma_k

        # ---- Update pooled cause-1 CIF (left -> right continuous). ----
        if total_d1 > 0 and tr > 0.0:
            pooled_cif_minus = pooled_cif_minus + total_d1 / tr

        # ---- Drain risk set: Y_g(t) = Y_g(t-) - d1_g - d2_g - cens_g. ----
        at_risk_g = at_risk_g - (d1 + d2 + nc).astype(np.float64)

    # ---- Final V update: cross-row terms from C and var2/var3. ----
    # V_{i,j} += sum_k ( C_{i,k}*C_{j,k}*var3_k
    #                  + C_{i,k}*var2_{j,k}
    #                  + C_{j,k}*var2_{i,k} )
    # Vectorised:
    cov = cov + influence @ (influence * var3).T
    cov = cov + influence @ var2.T
    cov = cov + var2 @ influence.T

    # Symmetrise (V is symmetric in i, j by construction).
    cov = 0.5 * (cov + cov.T)

    # ---- Test statistic ----
    # Clean-room implementation derived from Gray (1988) and counting-process
    # martingale theory; no GPL source consulted.
    df = n_groups - 1
    try:
        stat = float(score @ np.linalg.solve(cov, score))
    except np.linalg.LinAlgError:
        # Fall back to pseudo-inverse on near-singular V.
        stat = float(score @ np.linalg.pinv(cov) @ score)
    if not np.isfinite(stat) or stat < 0.0:
        stat = 0.0
    pvalue = float(chi2.sf(stat, df))

    return GrayTestResult(
        stat=stat,
        pvalue=pvalue,
        df=df,
        score=score,
        var=cov,
        n_groups=n_groups,
        rho=float(rho),
    )
