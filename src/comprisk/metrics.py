"""Concordance-style metrics for competing-risks survival models.

This module provides three public functions:

* :func:`concordance_index_cr` - Wolbers cause-specific concordance for
  competing risks (unweighted).
* :func:`compute_uno_weights` - per-observation IPCW weights based on the
  Kaplan-Meier estimator of the censoring distribution, with optional
  ESS-truncation gating.
* :func:`concordance_index_uno_cr` - IPCW (Uno) cause-specific concordance
  for competing risks.

References
----------
Wolbers, M., Koller, M.T., Witteman, J.C.M., Schemper, M. (2009).
"Concordance for prognostic models with competing risks."
*Biostatistics* 10(4): 715-727.

Uno, H., Cai, T., Pencina, M.J., D'Agostino, R.B., Wei, L.J. (2011).
"On the C-statistics for evaluating overall adequacy of risk prediction
procedures with censored survival data."
*Statistics in Medicine* 30(10): 1105-1117.

Cole, S.R., Hernan, M.A. (2008). "Constructing inverse probability
weights for marginal structural models."
*American Journal of Epidemiology* 168(6): 656-664.
"""

from __future__ import annotations

import math

import numpy as np

__all__ = [
    "compute_uno_weights",
    "concordance_index_cr",
    "concordance_index_uno_cr",
]


# Tolerance used for time-tie and estimate-tie detection in the Uno IPCW
# concordance. Implementation choice: any value in the range 1e-12..1e-6
# would be defensible for IEEE-754 double comparisons; 1e-9 was chosen
# empirically. Not from any specific publication.
_EPS_T = 1e-9


def concordance_index_cr(event, time, estimate, cause: int = 1) -> float:
    """Cause-specific concordance index for competing risks (Wolbers, 2009).

    A pair ``(i, j)`` with ``event[i] == cause`` is comparable iff
    ``time[j] > time[i]`` and subject ``j`` did not experience a competing
    event at or before ``time[i]``.  For each comparable pair the estimate
    of subject ``i`` is compared to the estimate of subject ``j``: a higher
    estimate at ``i`` is concordant, a lower one is discordant, equal
    estimates count as a half-concordance (tie).

    Parameters
    ----------
    event : array-like of int
        Event/cause code per subject. ``0`` denotes censoring; ``cause``
        is the cause of interest; any other positive integer is a
        competing event.
    time : array-like of float
        Observed time (event or censoring) per subject.
    estimate : array-like of float
        Predicted risk score for the cause of interest. Higher values
        should indicate higher risk.
    cause : int, default 1
        The cause of interest.

    Returns
    -------
    float
        The cause-specific concordance index.  Returns ``0.5`` when there
        are no comparable pairs or no events of ``cause``.

    Raises
    ------
    ValueError
        If ``cause`` is strictly larger than every observed event code.
    """
    event = np.asarray(event)
    time = np.asarray(time, dtype=float)
    estimate = np.asarray(estimate, dtype=float)
    n = len(event)
    if n == 0:
        return 0.5

    pos_events = event[event > 0]
    if pos_events.size > 0:
        max_cause = int(pos_events.max())
        if cause > max_cause:
            available = sorted({int(c) for c in pos_events.tolist()})
            raise ValueError(f"cause={cause} is not present; available causes are {available}")

    case_idx = np.flatnonzero(event == cause)
    if case_idx.size == 0:
        return 0.5

    # Build the comparable mask of shape (n_case, n).
    t_i = time[case_idx][:, None]
    t_j = time[None, :]
    e_j = event[None, :]

    # j is comparable with case i iff t_j > t_i AND not (j had a competing
    # event at or before t_i).
    competing_blocked = (e_j != 0) & (e_j != cause) & (t_j <= t_i)
    comparable = (t_j > t_i) & ~competing_blocked

    p_i = estimate[case_idx][:, None]
    p_j = estimate[None, :]

    concordant = int(np.sum(comparable & (p_j < p_i)))
    discordant = int(np.sum(comparable & (p_j > p_i)))
    tied = int(np.sum(comparable & (p_j == p_i)))

    denom = concordant + discordant + tied
    if denom == 0:
        return 0.5
    return (concordant + 0.5 * tied) / denom


# ---------------------------------------------------------------------------
# Kaplan-Meier of the censoring distribution
# ---------------------------------------------------------------------------


def _km_censor_fit(time: np.ndarray, event: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Kaplan-Meier of the censoring distribution under a competing-risks
    "events-first" tie convention.

    At each unique knot ``t_k`` with ``d_one`` cause-1 events, ``d_other``
    other-status occurrences (true censoring + competing events), and
    ``n_risk`` subjects at risk just before ``t_k``::

        if d_other > 0:
            denom = n_risk - d_one
            surv *= 0 if denom <= 0 else (1 - d_other / denom)

    Returns the unique sorted knots and the post-update KM survivor at
    each knot. For binary-status (cause-1 only) data this collapses to
    the textbook KM-of-censoring (Kaplan & Meier 1958) with the
    events-first tie convention used by the R ``survival`` package and
    most CR-IPCW reference implementations. The convention is rarely
    spelled out in published papers — it is implementation lore that
    became de-facto standard.
    """
    n = len(time)
    if n == 0:
        return np.empty(0, dtype=float), np.empty(0, dtype=float)

    order = np.argsort(time, kind="stable")
    t_sorted = time[order]
    e_sorted = event[order]
    is_one = (e_sorted == 1).astype(np.int64)

    t_unique, first_idx = np.unique(t_sorted, return_index=True)
    # Append the n boundary so np.diff gives group widths.
    bounds = np.append(first_idx, n)

    G = np.empty(t_unique.shape[0], dtype=float)
    surv = 1.0
    for k in range(t_unique.shape[0]):
        s, e = int(bounds[k]), int(bounds[k + 1])
        n_risk = n - s
        d_one = int(is_one[s:e].sum())
        d_other = (e - s) - d_one
        if d_other > 0:
            denom = n_risk - d_one
            surv = 0.0 if denom <= 0 else surv * (1.0 - d_other / denom)
        G[k] = surv
    return t_unique, G


def _ghat_minus(t_unique: np.ndarray, G: np.ndarray, query: np.ndarray) -> np.ndarray:
    """Left-limit of the KM survivor at each ``query`` time.

    For a query time ``q``: if ``q`` is at or below ``t_unique[0]`` the
    left-limit is ``1.0``; otherwise it equals ``G`` at the largest knot
    strictly less than ``q``.
    """
    if t_unique.size == 0:
        return np.ones(query.shape, dtype=float)
    idx = np.searchsorted(t_unique, query, side="left")
    out = np.ones(query.shape, dtype=float)
    has_predecessor = idx > 0
    out[has_predecessor] = G[idx[has_predecessor] - 1]
    return out


def _choose_gmin_auto(g_event: np.ndarray, ess_frac: float, ess_min: int, eps: float) -> float:
    """ESS-truncation lower clip for the IPCW G distribution.

    Operates on the event-only G values: chooses the smallest sorted
    ``g_sorted[k]`` at which the upper-tail effective sample size
    ``(Σ w)^2 / Σ w^2`` (with ``w = 1 / max(g, eps)^2``) reaches the
    target ``max(ess_min, ceil(ess_frac * d))``.
    """
    g = g_event[np.isfinite(g_event)]
    d = g.size
    if d <= 1:
        return 0.0
    if g.min() >= 1.0 - 1e-12:
        return 0.0

    g_sorted = np.sort(g)
    w_desc = 1.0 / np.maximum(g_sorted, eps) ** 2

    ess_target = max(ess_min, math.ceil(ess_frac * d))
    ess_target = min(ess_target, d)

    cum_w = np.concatenate(([0.0], np.cumsum(w_desc)))
    cum_w2 = np.concatenate(([0.0], np.cumsum(w_desc * w_desc)))

    # Search k = 0 ... d - ess_target inclusive: the upper-tail set
    # g_sorted[k:] has size d - k >= ess_target.
    for k in range(0, d - ess_target + 1):
        sum_w = cum_w[d] - cum_w[k]
        sum_w2 = cum_w2[d] - cum_w2[k]
        if sum_w2 > 0:
            ess_k = (sum_w * sum_w) / sum_w2
            if math.isfinite(ess_k) and ess_k >= ess_target:
                return float(g_sorted[k])
    # Fall-through: pick the boundary k.
    return float(g_sorted[d - ess_target])


def compute_uno_weights(
    time,
    event,
    *,
    gmin: float | str = "auto",
    ess_frac: float = 0.20,
    ess_min: int = 20,
    eps: float = 1e-12,
    eps_keep: float | None = None,
) -> np.ndarray:
    """Per-observation IPCW weights using the KM censoring estimator.

    For each subject ``i`` the weight is ``1 / G(time[i]^-)^2`` (Uno,
    2011) where ``G`` is the KM-of-censoring under a competing-risks
    events-first tie convention. Subjects whose left-limit ``G`` falls
    below the chosen lower clip ``gmin`` keep the data row alive with a
    tiny ``eps_keep`` weight rather than being silently dropped.

    Parameters
    ----------
    time : array-like of float
        Observed time per subject.
    event : array-like of int
        Event/cause code per subject. ``0`` denotes censoring.
    gmin : float | {"auto", "none"}, default "auto"
        Lower clip for the censoring survivor. ``"none"`` disables
        gating; ``"auto"`` chooses an ESS-stable clip from the event-only
        ``G`` distribution (Cole & Hernán, 2008). A non-finite or
        negative numeric value is treated as ``0.0``.
    ess_frac : float, default 0.20
        Effective-sample-size fraction target for ``gmin="auto"``.
    ess_min : int, default 20
        Effective-sample-size minimum target for ``gmin="auto"``.
    eps : float, default 1e-12
        Floor used when squaring ``G`` to avoid division by zero.
    eps_keep : float, optional
        Weight assigned to gated-out rows. Defaults to
        ``np.finfo(np.float64).eps``.

    Returns
    -------
    np.ndarray
        ``float64`` array of length ``n``.
    """
    time = np.asarray(time, dtype=float)
    event = np.asarray(event)
    n = time.size
    if n == 0:
        return np.zeros(0, dtype=np.float64)

    eps_keep = float(np.finfo(np.float64).eps) if eps_keep is None else float(eps_keep)

    event_f = np.asarray(event, dtype=float)
    miss = np.isnan(time) | np.isnan(event_f)

    # KM survivor of censoring + left-limit at each subject's time.
    t_unique, G_at = _km_censor_fit(time, event)
    G_left = _ghat_minus(t_unique, G_at, time)
    G_gate = np.where(miss, np.nan, G_left)

    # Decide gmin.
    if isinstance(gmin, str):
        if gmin == "none":
            gmin_used = 0.0
        elif gmin == "auto":
            mask_ev = (event != 0) & ~miss
            G_event = G_gate[mask_ev]
            gmin_used = _choose_gmin_auto(G_event, ess_frac, ess_min, eps)
        else:
            raise ValueError(f"unknown gmin={gmin!r}; expected float, 'auto', or 'none'")
    else:
        gmin_used = float(gmin)
        if not math.isfinite(gmin_used) or gmin_used < 0:
            gmin_used = 0.0

    w = np.zeros(n, dtype=np.float64)
    valid = ~np.isnan(G_gate)
    keep = valid & (G_gate >= gmin_used)
    drop = valid & ~keep

    if keep.any():
        Gsafe = np.maximum(G_gate[keep], eps)
        w[keep] = 1.0 / (Gsafe * Gsafe)
    if drop.any():
        w[drop] = eps_keep
    return w


# ---------------------------------------------------------------------------
# Uno IPCW cause-specific concordance
# ---------------------------------------------------------------------------


def concordance_index_uno_cr(
    event,
    time,
    estimate,
    *,
    cause: int,
    weights: np.ndarray,
) -> float:
    """Cause-specific Uno IPCW concordance index for competing risks.

    Combines the Wolbers (2009) cause-specific pair structure with
    inverse-probability-of-censoring weighting (Uno, 2011). Returns the
    raw ratio ``numerator / denominator`` (not ``1 - num/denom``).

    Parameters
    ----------
    event, time, estimate : array-like
        Per-subject event code, observed time, and predicted risk score
        for the cause of interest.
    cause : int
        Cause of interest (keyword-only).
    weights : np.ndarray
        IPCW weights, e.g. as produced by :func:`compute_uno_weights`.

    Returns
    -------
    float
        Concordance value, or ``NaN`` if there are fewer than two
        retained observations or the denominator is zero.
    """
    event = np.asarray(event)
    time = np.asarray(time, dtype=float)
    estimate = np.asarray(estimate, dtype=float)
    weights = np.asarray(weights, dtype=float)

    keep_mask = (
        (weights != 0)
        & ~np.isnan(time)
        & ~np.isnan(estimate)
        & ~np.isnan(np.asarray(event, dtype=float))
    )
    t = time[keep_mask]
    e = event[keep_mask]
    p = estimate[keep_mask]
    w = weights[keep_mask]
    n = t.size
    if n < 2:
        return float("nan")

    w1 = np.sqrt(w)
    is_case = e == cause
    is_cens = e == 0
    is_comp = (e > 0) & (e != cause)

    case_indices = np.flatnonzero(is_case)
    comp_indices = np.flatnonzero(is_comp)

    numer = 0.0
    denom = 0.0

    # ---- Branch A: case (i) vs comparator (j: t[j] > t[i] OR (t[j] == t[i]
    # AND j is censored)). Self is excluded automatically (case has e!=0,
    # so it is not in is_cens; the strict t[j] > t[i] handles the rest).
    for i in case_indices:
        ti = t[i]
        pi = p[i]
        wi = w[i]
        mask_j = (t > ti) | ((t == ti) & is_cens)
        if not mask_j.any():
            continue
        n_pairs = int(mask_j.sum())
        denom += 2.0 * wi * n_pairs
        p_j = p[mask_j]
        less = int((p_j < pi).sum())
        eq = int((p_j == pi).sum())
        numer += 2.0 * wi * less + wi * eq

    # ---- Branch B: tied case-times.
    if case_indices.size >= 2:
        order = np.argsort(t[case_indices], kind="stable")
        sorted_case = case_indices[order]
        sorted_t = t[sorted_case]
        d_total = sorted_case.size
        i_grp = 0
        while i_grp < d_total:
            j_grp = i_grp
            while j_grp + 1 < d_total and abs(sorted_t[j_grp + 1] - sorted_t[i_grp]) <= _EPS_T:
                j_grp += 1
            d = j_grp - i_grp + 1
            if d >= 2:
                grp = sorted_case[i_grp : j_grp + 1]
                w_grp = w[grp]
                p_grp = p[grp]
                sumW = float(w_grp.sum())

                # Tie mass: sum over rank-tied subgroups (within this
                # equal-time group) of (c - 1) * Σw over the subgroup.
                p_order = np.argsort(p_grp, kind="stable")
                sorted_p = p_grp[p_order]
                tieMass = 0.0
                k = 0
                while k < d:
                    L = k
                    while d > L + 1 and abs(sorted_p[L + 1] - sorted_p[L]) <= _EPS_T:
                        L += 1
                    c = L - k + 1
                    if c >= 2:
                        rank_w_sum = float(w_grp[p_order[k : L + 1]].sum())
                        tieMass += (c - 1) * rank_w_sum
                    k = L + 1

                denomTie = (d - 1) * sumW
                denom += denomTie
                numer += 0.5 * denomTie + 0.5 * tieMass
            i_grp = j_grp + 1

    # ---- Branch C: case (i) vs competing (j: time[j] <= time[i]).
    # Pair contribution multiplies BOTH endpoints by sqrt(w) — i.e. the
    # pair weight is sqrt(w_i) * sqrt(w_j), giving a symmetric form in i
    # and j. This differs from Branch A (asymmetric, weighted by w_i
    # alone) and is the convention used by the reference implementation
    # we benchmark against. The form is a natural extension of Uno
    # (2011) IPCW to the Wolbers (2009) competing-pair structure but
    # the specific sqrt-sqrt symmetrisation is an implementation choice
    # rather than a directly-cited published formula.
    if comp_indices.size > 0:
        t_comp = t[comp_indices]
        p_comp = p[comp_indices]
        w1_comp = w1[comp_indices]
        for i in case_indices:
            mask = t_comp <= t[i]
            if not mask.any():
                continue
            wj1 = w1_comp[mask]
            p_j = p_comp[mask]
            wi1 = w1[i]
            denom += 2.0 * wi1 * float(wj1.sum())
            less_term = float((wj1 * (p_j < p[i])).sum())
            eq_term = float((wj1 * (p_j == p[i])).sum())
            numer += 2.0 * wi1 * less_term + wi1 * eq_term

    if denom <= 0:
        return float("nan")
    return float(numer / denom)
