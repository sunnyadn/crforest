"""Aalen-Johansen cumulative incidence with Greenwood-corrected variance.

Non-parametric estimation of the cause-specific cumulative incidence function
(CIF) under right-censored competing-risks data. The point estimator is the
classical Aalen-Johansen estimator (Aalen, 1978; Aalen & Johansen, 1978),
which generalises the Kaplan-Meier survival estimator to multi-state data.
The pointwise variance is the discrete-time finite-sample form with a
Greenwood-style tie correction (Pepe, 1991).

This module is a clean-room implementation written from the mathematical
specification only; no third-party survival-analysis source code (Fortran,
R, or otherwise) was consulted while writing it. Variable names follow the
statistical literature rather than any specific implementation.

References
----------
Aalen, O.O. (1978). "Nonparametric inference for a family of counting
processes." *Annals of Statistics* 6(4):701-726.

Aalen, O.O., Johansen, S. (1978). "An empirical transition matrix for
non-homogeneous Markov chains based on censored observations."
*Scandinavian Journal of Statistics* 5(3):141-150.

Pepe, M.S. (1991). "Inference for events with dependent risks in multiple
endpoint studies." *Biometrics* 47(3):1003-1014.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

__all__ = ["CIFCurve", "CumulativeIncidence"]


@dataclass
class CIFCurve:
    """Cause-specific cumulative incidence step function.

    Stores the CIF jump times and corresponding step values plus pointwise
    variance estimates. The step function is right-continuous: the CIF at a
    query time ``q`` is the value at the latest jump time ``<= q``, or zero
    if ``q`` lies before the first jump.

    Attributes
    ----------
    times : np.ndarray of shape (n_jumps,)
        Distinct event times for the cause of interest, in ascending order.
    cif : np.ndarray of shape (n_jumps,)
        Cumulative incidence values at the jump times.
    var : np.ndarray of shape (n_jumps,)
        Pointwise variance estimates at the jump times.
    """

    times: np.ndarray = field(default_factory=lambda: np.zeros(0))
    cif: np.ndarray = field(default_factory=lambda: np.zeros(0))
    var: np.ndarray = field(default_factory=lambda: np.zeros(0))

    def evaluate(self, t: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Evaluate the right-continuous step function at query times.

        Parameters
        ----------
        t : array-like
            Query times.

        Returns
        -------
        cif_at_t : np.ndarray
            CIF values at the query times. Zero below the first jump,
            value at the last jump for queries beyond the last jump.
        var_at_t : np.ndarray
            Pointwise variance at the query times, with the same step
            convention.
        """
        t = np.asarray(t, dtype=np.float64)
        if self.times.size == 0:
            return np.zeros_like(t), np.zeros_like(t)
        # Right-continuous: index of latest jump time <= q.
        idx = np.searchsorted(self.times, t, side="right") - 1
        cif_out = np.where(idx >= 0, self.cif[np.clip(idx, 0, None)], 0.0)
        var_out = np.where(idx >= 0, self.var[np.clip(idx, 0, None)], 0.0)
        return cif_out, var_out


def _aalen_johansen_curves(
    time: np.ndarray, event: np.ndarray, cause_codes: list[int]
) -> dict[int, CIFCurve]:
    """Compute Aalen-Johansen CIF curves for one stratum, all causes.

    Implements the recursion in the module docstring. A single forward pass
    over the unique event times jointly tracks the Kaplan-Meier survival
    ``surv`` and per-cause cumulative incidence ``cif_k``, plus three running
    accumulators for the Pepe (1991) variance.

    Returns a dict mapping cause code to ``CIFCurve``.
    """
    # --- group counts at unique event times -----------------------------------
    is_event = event > 0
    event_times = np.unique(time[is_event])
    if event_times.size == 0:
        return {k: CIFCurve() for k in cause_codes}

    # at_risk[j] = number of subjects with time >= event_times[j]
    sort_idx = np.argsort(time, kind="mergesort")
    sorted_time = time[sort_idx]
    # n_at_or_after(t) = n - searchsorted(sorted_time, t, side='left')
    at_risk = (sorted_time.size - np.searchsorted(sorted_time, event_times, side="left")).astype(
        np.float64
    )

    # d_k[j] = number of cause-k events at event_times[j]
    n_t = event_times.size
    cause_event_counts: dict[int, np.ndarray] = {}
    t_idx = np.searchsorted(event_times, time)
    in_grid = (t_idx < n_t) & is_event
    for k in cause_codes:
        mask = in_grid & (event == k)
        cause_event_counts[k] = np.bincount(t_idx[mask], minlength=n_t).astype(np.float64)
    d_total = sum(cause_event_counts.values())

    # --- forward recursion ----------------------------------------------------
    # Per-cause running CIF jump records (one append per cause-k event time).
    jump_times: dict[int, list[float]] = {k: [] for k in cause_codes}
    jump_cif: dict[int, list[float]] = {k: [] for k in cause_codes}
    jump_var: dict[int, list[float]] = {k: [] for k in cause_codes}

    # Per-cause running CIF and variance accumulators.
    # Clean-room implementation of the Aalen-Johansen-with-Pepe-variance
    # recursion: written from the mathematical specification only, no GPL
    # source consulted. Each cause maintains its own (acc_ff, acc_fb, acc_bb)
    # so that the variance treatment is symmetric across causes.
    cif_running: dict[int, float] = {k: 0.0 for k in cause_codes}
    acc_ff: dict[int, float] = {k: 0.0 for k in cause_codes}
    acc_fb: dict[int, float] = {k: 0.0 for k in cause_codes}
    acc_bb: dict[int, float] = {k: 0.0 for k in cause_codes}

    surv = 1.0  # left-continuous KM survival, S(t-)

    for j in range(n_t):
        y = at_risk[j]
        d = d_total[j]
        if y <= 0 or d <= 0:
            continue

        surv_left = surv  # S(t-)
        # Update per-cause CIFs using S(t-).
        cif_left = dict(cif_running)  # snapshot F(t-)
        for k in cause_codes:
            d_k = cause_event_counts[k][j]
            if d_k > 0:
                cif_running[k] = cif_left[k] + surv_left * d_k / y
        # Update KM survival.
        surv = surv_left * (y - d) / y

        # Variance accumulator updates use the post-jump F(t) and S(t).
        for k in cause_codes:
            d_k = cause_event_counts[k][j]
            if d_k <= 0:
                continue

            # Greenwood-style finite-population correction.
            gw_factor = 1.0 if d_k <= 1 or y <= 1 else 1.0 - (d_k - 1.0) / (y - 1.0)
            sigma_k = surv_left * surv_left * gw_factor * d_k / (y * y)

            # Update accumulators for every other cause m != k (the "other
            # cause at this time" branch).
            for m in cause_codes:
                if m == k:
                    continue
                if surv > 0.0:
                    a_other = cif_running[m] / surv
                    b_other = 1.0 / surv
                    acc_ff[m] += a_other * a_other * sigma_k
                    acc_fb[m] += a_other * b_other * sigma_k
                    acc_bb[m] += b_other * b_other * sigma_k

            # Self-cause branch (k itself).
            if surv > 0.0:
                a_self = 1.0 + cif_running[k] / surv
                b_self = 1.0 / surv
            else:
                a_self = 1.0
                b_self = 0.0
            acc_ff[k] += a_self * a_self * sigma_k
            acc_fb[k] += a_self * b_self * sigma_k
            acc_bb[k] += b_self * b_self * sigma_k

        # Emit a jump record for each cause that had an event at this time.
        for k in cause_codes:
            if cause_event_counts[k][j] > 0:
                f_now = cif_running[k]
                v_now = acc_ff[k] + f_now * f_now * acc_bb[k] - 2.0 * f_now * acc_fb[k]
                jump_times[k].append(float(event_times[j]))
                jump_cif[k].append(float(f_now))
                jump_var[k].append(float(v_now))

    return {
        k: CIFCurve(
            times=np.asarray(jump_times[k], dtype=np.float64),
            cif=np.asarray(jump_cif[k], dtype=np.float64),
            var=np.asarray(jump_var[k], dtype=np.float64),
        )
        for k in cause_codes
    }


class CumulativeIncidence:
    """Aalen-Johansen cumulative incidence estimator with Pepe variance.

    Estimates the cause-specific cumulative incidence function and pointwise
    variance for one or more competing event types, optionally stratified by
    a grouping variable.

    Parameters
    ----------
    cause_codes : list of int, optional
        Positive integer event codes to fit. If ``None`` (default) the codes
        are inferred from the data as the sorted unique positive values of
        ``event``.

    Attributes
    ----------
    curves_ : dict
        Maps ``(group, cause)`` to a :class:`CIFCurve`. When ``group`` is
        not supplied, keys take the form ``(None, cause)``.

    Notes
    -----
    Event coding: ``0`` denotes censoring; positive integers index the
    competing causes. The estimator is consistent under independent
    right-censoring with discrete or continuous event-time distributions
    (Aalen & Johansen, 1978).
    """

    def __init__(self, cause_codes: list[int] | None = None) -> None:
        self.cause_codes = cause_codes

    # ------------------------------------------------------------------
    # fit
    # ------------------------------------------------------------------
    def fit(
        self,
        time: np.ndarray | None = None,
        event: np.ndarray | None = None,
        *,
        group: np.ndarray | None = None,
    ) -> CumulativeIncidence:
        """Fit the estimator.

        Parameters
        ----------
        time : array-like of shape (n,)
            Observed times.
        event : array-like of shape (n,)
            Event indicators (``0`` = censored, positive integers = causes).
        group : array-like of shape (n,), optional
            Grouping variable. When supplied, a separate CIF is fit per
            ``(group, cause)`` pair.

        Returns
        -------
        self : CumulativeIncidence
            Fitted estimator. ``self.curves_`` exposes the fitted curves.
        """
        if time is None or event is None:
            raise ValueError("`time` and `event` are required.")

        time_arr = np.asarray(time, dtype=np.float64)
        event_arr = np.asarray(event)

        if time_arr.shape != event_arr.shape:
            raise ValueError("`time` and `event` must have the same shape.")

        # Determine the cause codes to fit.
        if self.cause_codes is None:
            event_int = event_arr.astype(np.int64, copy=False)
            causes = sorted(int(c) for c in np.unique(event_int) if c > 0)
        else:
            causes = list(self.cause_codes)
            event_int = event_arr.astype(np.int64, copy=False)

        # Determine grouping.
        if group is None:
            group_arr: np.ndarray | None = None
            group_keys: list = [None]
        else:
            group_arr = np.asarray(group)
            group_keys = sorted({g.item() if hasattr(g, "item") else g for g in group_arr})

        curves: dict[tuple, CIFCurve] = {}
        for g in group_keys:
            mask = np.ones(time_arr.shape[0], dtype=bool) if group_arr is None else group_arr == g
            stratum_curves = _aalen_johansen_curves(time_arr[mask], event_int[mask], causes)
            for k, curve in stratum_curves.items():
                curves[(g, k)] = curve

        self.curves_ = curves
        return self

    # ------------------------------------------------------------------
    # timepoints
    # ------------------------------------------------------------------
    def timepoints(self, t) -> tuple[np.ndarray, np.ndarray]:
        """Evaluate every fitted curve at a common set of query times.

        Curves are returned in a deterministic order: sorted by
        ``(str(group), cause)`` so that single-group fits and grouped fits
        share the same key convention.

        Parameters
        ----------
        t : array-like
            Query times.

        Returns
        -------
        est : np.ndarray of shape (n_curves, n_t)
            Cumulative incidence at the query times.
        var : np.ndarray of shape (n_curves, n_t)
            Pointwise variance at the query times.
        """
        if not hasattr(self, "curves_"):
            raise RuntimeError("CumulativeIncidence must be fit before timepoints().")

        t_arr = np.asarray(t, dtype=np.float64)
        keys = sorted(self.curves_.keys(), key=lambda gk: (str(gk[0]), gk[1]))
        self._timepoints_keys_ = keys

        n_curves = len(keys)
        n_t = t_arr.shape[0]
        est = np.zeros((n_curves, n_t), dtype=np.float64)
        var = np.zeros((n_curves, n_t), dtype=np.float64)
        for i, key in enumerate(keys):
            cif_at, var_at = self.curves_[key].evaluate(t_arr)
            est[i] = cif_at
            var[i] = var_at
        return est, var
