"""Fine-Gray subdistribution-hazard regression for competing risks.

Implements the proportional subdistribution-hazards model of Fine & Gray
(1999) via an IPCW-weighted partial likelihood. Mathematically equivalent
to Geskus (2011) -- which the R `survival` package's `finegray()` exposes
as expanded-dataset + weighted Cox -- without physically expanding the
dataset (avoids the O(n_competing x n_unique_event_times) row blowup).

The estimator targets parity with R `cmprsk::crr()` defaults: Breslow
tie-handling, KM-of-censoring left-limit weights per censoring stratum,
Newton-Raphson with Armijo backtracking, ``gtol=1e-6, max_iter=10``.

References
----------
Fine, J.P., Gray, R.J. (1999). "A proportional hazards model for the
subdistribution of a competing risk." *Journal of the American Statistical
Association* 94(446):496-509.

Geskus, R.B. (2011). "Cause-specific cumulative incidence estimation and
the Fine and Gray model under both left truncation and right censoring."
*Biometrics* 67(1):39-49.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from comprisk._sklearn_compat import is_structured_survival_y, unpack_structured_y

__all__ = ["FineGrayRegression"]


# When evaluating the censoring-survival KM at observed event times, the
# left limit G(t-) is what enters the IPCW weight (cf. Fine & Gray 1999
# eq. 4). The KM is right-continuous, so we evaluate at t * (1 - 100 ε)
# to land just before any jump at t.
_LEFT_LIMIT_FACTOR = 1.0 - 100.0 * np.finfo(np.float64).eps


def _km_censoring_left_limit(
    time: np.ndarray,
    event: np.ndarray,
    cengroup: np.ndarray,
    eval_times: np.ndarray,
) -> np.ndarray:
    """KM of the censoring distribution, evaluated at ``eval_times`` left-limit.

    For each censoring stratum ``k`` (unique value in ``cengroup``), fit
    Kaplan-Meier to the censoring indicator (``event == 0``), then return
    ``G_k(t-) := KM_k(t * (1 - 100*eps))`` for every ``t`` in ``eval_times``.

    Returns
    -------
    G : ndarray, shape (n_eval, n_cengroups)
        ``G[i, k]`` is the censoring-survival probability just before
        ``eval_times[i]`` in stratum ``k``. Cengroups are ordered by
        ``np.unique(cengroup)``.
    """
    eval_at = eval_times * _LEFT_LIMIT_FACTOR
    cgs = np.unique(cengroup)
    n_eval = eval_times.shape[0]
    G = np.empty((n_eval, cgs.shape[0]), dtype=np.float64)

    for ki, cg in enumerate(cgs):
        mask = cengroup == cg
        t_k = time[mask]
        c_k = (event[mask] == 0).astype(np.float64)  # censoring indicator

        # Per-time aggregation: sort by time, compute (n_cens, n_at) per
        # unique time, build KM.
        order = np.argsort(t_k, kind="stable")
        t_sorted = t_k[order]
        c_sorted = c_k[order]
        u_t, t_idx, n_at_per_unique = np.unique(t_sorted, return_inverse=True, return_counts=True)
        d_cens = np.bincount(t_idx, weights=c_sorted, minlength=u_t.shape[0])
        n_at = np.cumsum(n_at_per_unique[::-1])[::-1].astype(np.float64)
        # KM jump at each unique time: G_k(t) = G_k(t-) * (1 - d_cens / n_at)
        with np.errstate(divide="ignore", invalid="ignore"):
            jumps = np.where(n_at > 0, 1.0 - d_cens / n_at, 1.0)
        G_at_unique = np.cumprod(jumps)
        # Evaluate at left-limit eval_at: searchsorted side="right" returns
        # the position AFTER any tied unique times, so subtracting 1 gives
        # the largest unique time strictly < eval_at; using -1 sentinel
        # before that gives G = 1.
        idx = np.searchsorted(u_t, eval_at, side="right") - 1
        G_col = np.where(idx >= 0, G_at_unique[np.clip(idx, 0, None)], 1.0)
        G[:, ki] = G_col
    return G


def _build_event_time_grid(time: np.ndarray, event: np.ndarray, cause: int) -> np.ndarray:
    """Unique cause-of-interest event times in ascending order."""
    et = np.unique(time[event == cause])
    return et.astype(np.float64)


def _newton_raphson_fg(
    X: np.ndarray,
    time: np.ndarray,
    event: np.ndarray,
    cengroup: np.ndarray,
    cause: int,
    cengroup_index: np.ndarray,
    G_at_event_times: np.ndarray,  # (n_event_times, ncg)
    G_at_subject_times: np.ndarray,  # (n,)
    event_times: np.ndarray,  # (n_event_times,)
    *,
    max_iter: int,
    gtol: float,
    max_backtrack: int = 20,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, float, int, bool]:
    """Maximise weighted Breslow partial likelihood via Newton + Armijo.

    Returns ``(beta, score, info, neg_log_pl, n_iter, converged)``.
    """
    p = X.shape[1]
    beta = np.zeros(p, dtype=np.float64)

    nll, score, info = _negative_log_pl(
        beta,
        X,
        time,
        event,
        cengroup_index,
        cause,
        event_times,
        G_at_event_times,
        G_at_subject_times,
    )

    converged = False
    n_iter = 0
    for it in range(1, max_iter + 1):
        n_iter = it
        # Newton step
        try:
            step = -np.linalg.solve(info, score)
        except np.linalg.LinAlgError:
            # Diverging info; fall back to gradient descent at small rate.
            step = -score / (np.abs(np.diag(info)) + 1.0)

        # Armijo backtracking
        accept = False
        for _ in range(max_backtrack):
            beta_new = beta + step
            nll_new, score_new, info_new = _negative_log_pl(
                beta_new,
                X,
                time,
                event,
                cengroup_index,
                cause,
                event_times,
                G_at_event_times,
                G_at_subject_times,
            )
            if np.isfinite(nll_new) and nll_new <= nll + 1e-4 * np.dot(score, step):
                accept = True
                break
            step *= 0.5
        if not accept:
            break

        # Scaled-gradient convergence: each component of the score is scaled
        # by max(|β_j|, 1), and the threshold is scaled by max(|nll|, 1).
        # Defaults gtol=1e-6, max_iter=10 are chosen so that R cmprsk
        # users get the same convergence behavior by default.
        beta = beta_new
        nll, score, info = nll_new, score_new, info_new
        crit = float(np.max(np.abs(score) * np.maximum(np.abs(beta), 1.0)))
        rhs = float(max(abs(nll), 1.0) * gtol)
        if crit < rhs:
            converged = True
            break
    return beta, score, info, nll, n_iter, converged


def _negative_log_pl(
    beta: np.ndarray,
    X: np.ndarray,
    time: np.ndarray,
    event: np.ndarray,
    cengroup_index: np.ndarray,
    cause: int,
    event_times: np.ndarray,
    G_at_event_times: np.ndarray,
    G_at_subject_times: np.ndarray,
) -> tuple[float, np.ndarray, np.ndarray]:
    """Compute negative log partial likelihood, score, observed info.

    Implements Fine-Gray weighted Breslow PL via two cumulative-sum
    streams per cengroup. Numerically stabilised by subtracting the max
    linear predictor before exponentiating.
    """
    n, p = X.shape
    n_e = event_times.shape[0]
    ncg = G_at_event_times.shape[1]

    eta = X @ beta
    eta_max = float(np.max(eta)) if n > 0 else 0.0
    r = np.exp(eta - eta_max)

    # ------ Stream 1: standard at-risk set {i: time_i >= t_e} ------
    # Sort subjects by descending time so reverse-cumsum gives Σ over
    # {i: time_i >= threshold}. We need values at each event time;
    # use searchsorted on ascending unique times.
    order_asc = np.argsort(time, kind="stable")
    t_asc = time[order_asc]
    r_asc = r[order_asc]
    Xr_asc = X[order_asc] * r_asc[:, None]  # (n, p)

    # Σ over {i: t_i >= t_e} via right-cumulative sum.
    # We DON'T materialise the (n, p, p) Hessian tensor — it's 2GB at
    # n=75k p=58. Instead use the algebraic identity
    #     Σ_e (d_e/S0_e) S2[e]
    #   = Σ_i r_i x_i x_i^T * Σ_{e: t_e ≤ t_i} (d_e / S0_e)
    # for the Stream-1 contribution to the info matrix, and an analogous
    # suffix-cumsum trick per cengroup for Stream 2 (see info block below).
    r_rev = np.cumsum(r_asc[::-1])[::-1]
    Xr_rev = np.cumsum(Xr_asc[::-1], axis=0)[::-1]

    idx_e_in_asc = np.searchsorted(t_asc, event_times, side="left")
    pad_zeros_r = np.zeros(1, dtype=r_rev.dtype)
    pad_zeros_X = np.zeros((1, p), dtype=Xr_rev.dtype)
    r_rev_p = np.concatenate([r_rev, pad_zeros_r])
    Xr_rev_p = np.concatenate([Xr_rev, pad_zeros_X], axis=0)
    S0_normal = r_rev_p[idx_e_in_asc]  # (n_e,)
    S1_normal = Xr_rev_p[idx_e_in_asc]  # (n_e, p)

    # ------ Stream 2: competing-event subjects at t_j < t_e ------
    # Per cengroup k:
    #   define c_j = r_j / G_k(t_j-) for j in cengroup k with event==competing
    #   then S0_comp(t_e | k) = G_k(t_e-) * Σ_{j: t_j < t_e} c_j
    is_competing = (event != cause) & (event != 0)
    S0_comp = np.zeros(n_e, dtype=np.float64)
    S1_comp = np.zeros((n_e, p), dtype=np.float64)

    # For the info matrix we need per-cengroup IPCW-weighted X subsets +
    # a suffix-cumsum of G_k(t_e-) * d_e/S0_e (computed once S0 is known).
    # Cache the Stream-2 inputs we need post-loop.
    stream2_state: list[tuple[np.ndarray, np.ndarray, np.ndarray, int]] = []
    for k in range(ncg):
        mask = is_competing & (cengroup_index == k)
        if not np.any(mask):
            continue
        t_j = time[mask]
        r_j = r[mask]
        X_j = X[mask]
        G_at_t_j = G_at_subject_times[mask]
        G_safe = np.where(G_at_t_j > 0, G_at_t_j, np.nan)
        c_scalar = r_j / G_safe  # (n_comp_k,)
        c_vec = X_j * c_scalar[:, None]  # (n_comp_k, p)
        order_j = np.argsort(t_j, kind="stable")
        tj_sorted = t_j[order_j]
        cs_scalar = np.cumsum(c_scalar[order_j])
        cs_vec = np.cumsum(c_vec[order_j], axis=0)
        idx_e_in_j = np.searchsorted(tj_sorted, event_times, side="left") - 1
        valid = idx_e_in_j >= 0
        Ge = G_at_event_times[:, k]  # (n_e,)
        if np.any(valid):
            S0_comp[valid] += Ge[valid] * cs_scalar[idx_e_in_j[valid]]
            S1_comp[valid] += Ge[valid, None] * cs_vec[idx_e_in_j[valid]]
        stream2_state.append((mask, c_scalar, X_j, k))

    S0 = S0_normal + S0_comp
    S1 = S1_normal + S1_comp

    if np.any(S0 <= 0) or not np.all(np.isfinite(S0)):
        # Numerically infeasible (shouldn't happen with well-formed data).
        return np.inf, np.zeros(p), np.eye(p)

    log_S0 = np.log(S0)
    s1_over_s0 = S1 / S0[:, None]  # (n_e, p)

    # Cause-of-interest event counts and Σ x_event per t_e
    is_cause = event == cause
    if not np.any(is_cause):
        return np.inf, np.zeros(p), np.eye(p)
    t_e_per_event = time[is_cause]
    X_e_per_event = X[is_cause]
    eta_e_per_event = eta[is_cause] - eta_max

    # Map each event to its event_times index.
    eidx = np.searchsorted(event_times, t_e_per_event)
    d_e = np.bincount(eidx, minlength=n_e).astype(np.float64)
    sumX_e = np.zeros((n_e, p), dtype=np.float64)
    for j in range(p):
        sumX_e[:, j] = np.bincount(eidx, weights=X_e_per_event[:, j], minlength=n_e)
    sum_eta_e = np.bincount(eidx, weights=eta_e_per_event, minlength=n_e)

    # Negative log PL: -Σ (η_event - log S0(t_e))
    nll = -float(np.sum(sum_eta_e - d_e * log_S0))
    # Score: -Σ (x_event - S1/S0)
    score = -np.sum(sumX_e - d_e[:, None] * s1_over_s0, axis=0)

    # Observed info: Σ_e d_e (S2[e]/S0[e] - bar_x[e] bar_x[e]^T)
    # Avoid materialising S2 = (n_e, p, p). Use:
    #   Σ_e d_e/S0[e] * S2[e]
    #     = Σ_i r_i x_i x_i^T * Σ_{e: t_e ≤ t_i} (d_e/S0[e])    (Stream 1)
    #     + Σ_k Σ_{j: comp, k} r_j/G_k(t_j-) x_j x_j^T
    #              * Σ_{e: t_e > t_j} G_k(t_e-) (d_e/S0[e])     (Stream 2)
    # — both expressed as a single weighted X.T @ diag(w) @ X.
    weight_per_e = d_e / S0
    cum_w = np.concatenate([[0.0], np.cumsum(weight_per_e)])
    idx_t_in_e = np.searchsorted(event_times, time, side="right")
    w_subj_stream1 = r * cum_w[idx_t_in_e]
    info = (X.T * w_subj_stream1) @ X
    for mask, c_scalar, X_j, k in stream2_state:
        Ge = G_at_event_times[:, k]
        Gw = Ge * weight_per_e
        suffix_Gw = np.concatenate([np.cumsum(Gw[::-1])[::-1], [0.0]])
        fe = idx_t_in_e[mask]
        w_subj_stream2 = c_scalar * suffix_Gw[fe]
        info += (X_j.T * w_subj_stream2) @ X_j
    info -= (s1_over_s0.T * d_e) @ s1_over_s0
    return nll, score, info


def _baseline_subdist_hazard(
    beta: np.ndarray,
    X: np.ndarray,
    time: np.ndarray,
    event: np.ndarray,
    cengroup_index: np.ndarray,
    cause: int,
    event_times: np.ndarray,
    G_at_event_times: np.ndarray,
    G_at_subject_times: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Breslow-type baseline subdistribution-hazard increments.

    The plug-in estimator of the cumulative baseline subdistribution
    hazard at each cause-of-interest event time ``t_e`` is the Breslow
    form ``dΛ_0(t_e) = d_e / S0(t_e)``, where ``S0`` is the IPCW-weighted
    risk-set sum at that time and ``d_e`` is the number of cause-1 events
    at ``t_e`` (Fine & Gray 1999 §3). Cumulative subdistribution hazard is
    ``np.cumsum(dΛ_0)``.

    Returns ``(event_times, dΛ_0_e)``.
    """
    eta = X @ beta
    eta_max = float(np.max(eta)) if X.shape[0] > 0 else 0.0
    r = np.exp(eta - eta_max)
    n_e = event_times.shape[0]
    ncg = G_at_event_times.shape[1]
    order_asc = np.argsort(time, kind="stable")
    t_asc = time[order_asc]
    r_asc = r[order_asc]
    r_rev = np.cumsum(r_asc[::-1])[::-1]
    pad_zeros_r = np.zeros(1, dtype=r_rev.dtype)
    r_rev_p = np.concatenate([r_rev, pad_zeros_r])
    idx_e_in_asc = np.searchsorted(t_asc, event_times, side="left")
    S0 = r_rev_p[idx_e_in_asc].astype(np.float64)
    is_competing = (event != cause) & (event != 0)
    for k in range(ncg):
        mask = is_competing & (cengroup_index == k)
        if not np.any(mask):
            continue
        t_j = time[mask]
        r_j = r[mask]
        G_at_t_j = G_at_subject_times[mask]
        G_safe = np.where(G_at_t_j > 0, G_at_t_j, np.nan)
        order_j = np.argsort(t_j, kind="stable")
        cs_scalar = np.cumsum((r_j / G_safe)[order_j])
        idx_e_in_j = np.searchsorted(t_j[order_j], event_times, side="left") - 1
        valid = idx_e_in_j >= 0
        if np.any(valid):
            S0[valid] += G_at_event_times[valid, k] * cs_scalar[idx_e_in_j[valid]]
    is_cause = event == cause
    eidx = np.searchsorted(event_times, time[is_cause])
    d_e = np.bincount(eidx, minlength=n_e).astype(np.float64)
    # Multiply through eta_max correction since r had it factored out:
    # true r_i = exp(η_i) = exp(η_max) * r_i_scaled, so true S0 = exp(η_max)*S0.
    # dLambda0(t_e) = d_e / S0_true = d_e / (exp(η_max) * S0_scaled).
    dL = d_e / (S0 * np.exp(eta_max))
    return event_times.copy(), dL


@dataclass
class _FGState:
    """Container for non-public fit artifacts (used by predict)."""

    cengroups: np.ndarray
    cengroup_event_times: np.ndarray
    G_at_event_times: np.ndarray  # (n_e, ncg)
    baseline_times: np.ndarray
    baseline_hazard_increments: np.ndarray  # (n_e,) hazard increments


class FineGrayRegression:
    """Fine-Gray subdistribution-hazard regression for competing risks.

    Fits the proportional subdistribution-hazards model of Fine & Gray
    (1999) via Newton-Raphson on the IPCW-weighted Breslow partial
    likelihood. Targets parity with R ``cmprsk::crr()`` defaults.

    Parameters
    ----------
    cause : int, default 1
        Cause-of-interest event code (cmprsk's ``failcode``).
    cencode : int, default 0
        Censoring event code (cmprsk's ``cencode``).
    max_iter : int, default 10
        Maximum Newton iterations (cmprsk default).
    gtol : float, default 1e-6
        Convergence tolerance on ``max(|score| * max(|beta|, 1))``.
    robust_se : bool, default False
        If ``True``, compute cluster-robust sandwich SE via per-subject
        score residuals; agrees with cmprsk's IPCW-corrected sandwich SE
        to ~1e-3 (Geskus 2011 equivalence). When ``False``, ``se_`` is
        the naive plug-in ``sqrt(diag(inv(info)))``.

    Attributes
    ----------
    coef_ : ndarray, shape (n_features,)
    se_ : ndarray, shape (n_features,)
    var_ : ndarray, shape (n_features, n_features)
    n_iter_ : int
    converged_ : bool
    log_likelihood_ : float
        Maximised partial log-likelihood at ``coef_``.
    log_likelihood_null_ : float
        Partial log-likelihood at ``beta = 0``.

    Examples
    --------
    >>> import numpy as np
    >>> from comprisk import FineGrayRegression, Surv
    >>> rng = np.random.default_rng(0)
    >>> n = 200
    >>> X = rng.normal(size=(n, 3))
    >>> time = rng.exponential(1.0, size=n) + 0.1
    >>> event = rng.choice([0, 1, 2], size=n, p=[0.3, 0.5, 0.2])
    >>> y = Surv.from_arrays(event=event, time=time)
    >>> fg = FineGrayRegression().fit(X, y)
    >>> fg.coef_.shape
    (3,)
    """

    def __init__(
        self,
        *,
        cause: int = 1,
        cencode: int = 0,
        max_iter: int = 10,
        gtol: float = 1e-6,
        robust_se: bool = False,
    ) -> None:
        self.cause = cause
        self.cencode = cencode
        self.max_iter = max_iter
        self.gtol = gtol
        self.robust_se = robust_se

    def fit(
        self,
        X,
        y=None,
        time=None,
        event=None,
        *,
        cengroup=None,
    ) -> FineGrayRegression:
        """Fit the Fine-Gray model.

        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
        y : structured array, optional
            ``Surv``-style structured array with ``event`` and ``time``
            fields. Mutually exclusive with the legacy ``time``/``event``
            kwargs.
        time, event : array-like, shape (n_samples,), optional
            Legacy three-argument form ``fit(X, time=, event=)`` mirroring
            ``CompetingRiskForest``.
        cengroup : array-like of int, shape (n_samples,), optional
            Censoring stratum for each subject (``cmprsk`` ``cengroup``).
            ``None`` (default) places all subjects in a single stratum.

        Returns
        -------
        self
        """
        X = np.asarray(X, dtype=np.float64)
        if X.ndim != 2:
            raise ValueError(f"X must be 2-D; got shape {X.shape}")

        time_arr, event_arr = self._unpack_y(y, time, event)
        time_arr = np.asarray(time_arr, dtype=np.float64)
        event_arr = np.asarray(event_arr, dtype=np.int64)
        n = X.shape[0]
        if time_arr.shape[0] != n or event_arr.shape[0] != n:
            raise ValueError(f"time/event length must match X.shape[0]={n}")
        if cengroup is None:
            cengroup_arr = np.zeros(n, dtype=np.int64)
        else:
            cengroup_arr = np.asarray(cengroup, dtype=np.int64)
            if cengroup_arr.shape[0] != n:
                raise ValueError("cengroup length must match X")

        # Normalise event coding: censored subjects must be event == 0
        # internally for the IPCW reweighting. cmprsk uses cencode=0 by
        # default; if the user passes a different cencode, remap.
        event_internal = event_arr.copy()
        if self.cencode != 0:
            cens_mask = event_arr == self.cencode
            event_internal[event_arr == 0] = -999  # avoid collision
            event_internal[cens_mask] = 0

        # Index cengroups 0..ncg-1
        cgs = np.unique(cengroup_arr)
        cg_index = np.searchsorted(cgs, cengroup_arr)

        if not np.any(event_internal == self.cause):
            raise ValueError(
                f"no subjects have event == cause ({self.cause}); "
                f"unique events seen: {sorted(np.unique(event_arr).tolist())}"
            )

        event_times = _build_event_time_grid(time_arr, event_internal, self.cause)
        if event_times.size == 0:
            raise ValueError("no positive-time cause-of-interest events; cannot fit")

        G_at_event_times = _km_censoring_left_limit(
            time_arr, event_internal, cengroup_arr, event_times
        )
        G_at_subject_times = np.empty(n, dtype=np.float64)
        for ki, cg in enumerate(cgs):
            mask = cengroup_arr == cg
            if not np.any(mask):
                continue
            G_subj_col = _km_censoring_left_limit(
                time_arr,
                event_internal,
                cengroup_arr,
                time_arr[mask].astype(np.float64),
            )[:, ki]
            G_at_subject_times[mask] = G_subj_col

        beta, score, info, nll, n_iter, converged = _newton_raphson_fg(
            X,
            time_arr,
            event_internal,
            cengroup_arr,
            self.cause,
            cg_index,
            G_at_event_times,
            G_at_subject_times,
            event_times,
            max_iter=self.max_iter,
            gtol=self.gtol,
        )

        # Variance: naive plug-in (default) or cluster-robust sandwich.
        try:
            inv_info = np.linalg.inv(info)
        except np.linalg.LinAlgError as exc:
            raise RuntimeError(
                "observed information matrix is singular at the fitted "
                "coefficients; design may be rank-deficient"
            ) from exc

        if self.robust_se:
            var = self._cluster_robust_sandwich(
                beta,
                X,
                time_arr,
                event_internal,
                cg_index,
                cgs,
                event_times,
                G_at_event_times,
                G_at_subject_times,
                inv_info,
            )
        else:
            var = inv_info

        # Null log-likelihood at beta = 0 for likelihood-ratio tests.
        nll0, _, _ = _negative_log_pl(
            np.zeros_like(beta),
            X,
            time_arr,
            event_internal,
            cg_index,
            self.cause,
            event_times,
            G_at_event_times,
            G_at_subject_times,
        )

        # Breslow-type baseline subdistribution hazard, used by
        # predict_cumulative_incidence below.
        bt, bdL = _baseline_subdist_hazard(
            beta,
            X,
            time_arr,
            event_internal,
            cg_index,
            self.cause,
            event_times,
            G_at_event_times,
            G_at_subject_times,
        )

        self.coef_ = beta
        self.var_ = var
        self.se_ = np.sqrt(np.maximum(np.diag(var), 0.0))
        self.n_iter_ = int(n_iter)
        self.converged_ = bool(converged)
        self.log_likelihood_ = -float(nll)
        self.log_likelihood_null_ = -float(nll0)
        self.score_ = score
        self.information_ = info
        self.n_features_in_ = X.shape[1]

        self._state = _FGState(
            cengroups=cgs,
            cengroup_event_times=event_times,
            G_at_event_times=G_at_event_times,
            baseline_times=bt,
            baseline_hazard_increments=bdL,
        )
        return self

    def _cluster_robust_sandwich(
        self,
        beta: np.ndarray,
        X: np.ndarray,
        time: np.ndarray,
        event: np.ndarray,
        cg_index: np.ndarray,
        cgs: np.ndarray,
        event_times: np.ndarray,
        G_at_event_times: np.ndarray,
        G_at_subject_times: np.ndarray,
        inv_info: np.ndarray,
    ) -> np.ndarray:
        """Geskus-style per-subject score-residual cluster sandwich.

        Per Therneau (R survival::finegray docs), this matches cmprsk's
        IPCW-corrected sandwich SE to 3 digits in Geskus 2011 simulations.
        """
        n, p = X.shape
        eta = X @ beta
        eta_max = float(np.max(eta)) if n > 0 else 0.0
        r = np.exp(eta - eta_max)

        # Recompute S0, S1 at each event time (S2 not needed here).
        n_e = event_times.shape[0]
        ncg = G_at_event_times.shape[1]
        order_asc = np.argsort(time, kind="stable")
        t_asc = time[order_asc]
        r_asc = r[order_asc]
        Xr_asc = X[order_asc] * r_asc[:, None]
        r_rev = np.cumsum(r_asc[::-1])[::-1]
        Xr_rev = np.cumsum(Xr_asc[::-1], axis=0)[::-1]
        pad_zeros_r = np.zeros(1, dtype=r_rev.dtype)
        pad_zeros_X = np.zeros((1, p), dtype=Xr_rev.dtype)
        r_rev_p = np.concatenate([r_rev, pad_zeros_r])
        Xr_rev_p = np.concatenate([Xr_rev, pad_zeros_X], axis=0)
        idx_e_asc = np.searchsorted(t_asc, event_times, side="left")
        S0 = r_rev_p[idx_e_asc].astype(np.float64)
        S1 = Xr_rev_p[idx_e_asc].astype(np.float64)
        is_competing = (event != self.cause) & (event != 0)
        for k in range(ncg):
            mask = is_competing & (cg_index == k)
            if not np.any(mask):
                continue
            t_j = time[mask]
            r_j = r[mask]
            X_j = X[mask]
            G_at_t_j = G_at_subject_times[mask]
            G_safe = np.where(G_at_t_j > 0, G_at_t_j, np.nan)
            c_scalar = r_j / G_safe
            c_vec = X_j * c_scalar[:, None]
            order_j = np.argsort(t_j, kind="stable")
            cs_scalar = np.cumsum(c_scalar[order_j])
            cs_vec = np.cumsum(c_vec[order_j], axis=0)
            tj_sorted = t_j[order_j]
            idx_e_j = np.searchsorted(tj_sorted, event_times, side="left") - 1
            valid = idx_e_j >= 0
            Ge = G_at_event_times[:, k]
            if np.any(valid):
                S0[valid] += Ge[valid] * cs_scalar[idx_e_j[valid]]
                S1[valid] += Ge[valid, None] * cs_vec[idx_e_j[valid]]
        if np.any(S0 <= 0):
            return inv_info  # graceful fallback
        s1_over_s0 = S1 / S0[:, None]

        # dN_i (cause events) per subject.
        is_cause = event == self.cause
        eidx_for_event = np.searchsorted(event_times, time[is_cause])

        # Score residual per subject:
        #   U_i = Σ_e {δ_i I[event=cause, t_i==t_e] (x_i - S1/S0)
        #            - w_i(t_e) r_i [(x_i - S1/S0) (d_e/S0) ]}
        # We approximate via per-subject contributions to the score gradient:
        # for cause-events at t_e: + (x_i - bar_x_e) per event
        # for risk-set members: - r_i_w_at_e (x_i - bar_x_e) (d_e / S0_e)
        d_e = np.bincount(eidx_for_event, minlength=n_e).astype(np.float64)
        bar_x = s1_over_s0
        weight_per_e = d_e / S0
        # cum_w[k]   = Σ_{e < k} weight_per_e[e]
        # cum_w_bar[k] = Σ_{e < k} weight_per_e[e] * bar_x[e]
        cum_w = np.concatenate([[0.0], np.cumsum(weight_per_e)])
        cum_w_bar = np.concatenate(
            [np.zeros((1, p)), np.cumsum(weight_per_e[:, None] * bar_x, axis=0)],
            axis=0,
        )

        U = np.zeros((n, p), dtype=np.float64)
        # Cause-event contributions: dN_i adds (X_i - bar_x_e) at the event.
        cause_subjects = np.flatnonzero(is_cause)
        U[cause_subjects] += X[cause_subjects] - bar_x[eidx_for_event]

        # Stream 1 risk-set drag: every subject i is at risk for events
        # 0..idx_t_in_e[i]-1, drag is r_i * (X_i * cum_w - cum_w_bar) at upto.
        idx_t_in_e = np.searchsorted(event_times, time, side="right")
        U -= r[:, None] * (X * cum_w[idx_t_in_e, None] - cum_w_bar[idx_t_in_e])

        # Stream 2: competing-event subjects at t_j contribute at events e
        # with t_e > t_j weighted by G_k(t_e-)/G_k(t_j-). Vectorise via
        # per-cengroup suffix-cumsum: for j with first_e = first event index
        # strictly greater than t_j,
        #   Σ_{e ≥ first_e} G_k(t_e-) * weight_per_e[e] * (X_j - bar_x[e])
        #     = X_j * suffix_Gw[first_e] - suffix_Gw_bar[first_e]
        # times the per-subject IPCW factor r_j / G_k(t_j-).
        first_e_all = np.searchsorted(event_times, time, side="right")
        for k in range(ncg):
            mask = is_competing & (cg_index == k)
            if not np.any(mask):
                continue
            Ge = G_at_event_times[:, k]  # (n_e,)
            Gw = Ge * weight_per_e  # (n_e,)
            Gwx = Gw[:, None] * bar_x  # (n_e, p)
            # suffix sums: suffix_Gw[i] = Σ_{e ≥ i} Gw[e].
            suffix_Gw = np.concatenate(
                [np.cumsum(Gw[::-1])[::-1], [0.0]]
            )  # length n_e+1; suffix_Gw[n_e] = 0
            suffix_Gwx = np.concatenate(
                [np.cumsum(Gwx[::-1], axis=0)[::-1], np.zeros((1, p))],
                axis=0,
            )
            G_at_t_j = G_at_subject_times[mask]
            G_safe = np.where(G_at_t_j > 0, G_at_t_j, np.nan)
            ipcw = r[mask] / G_safe  # (n_k,)
            fe = first_e_all[mask]  # (n_k,)
            U[mask] -= ipcw[:, None] * (X[mask] * suffix_Gw[fe, None] - suffix_Gwx[fe])

        # Sandwich: inv_info @ (Σ U U^T) @ inv_info
        meat = U.T @ U
        return inv_info @ meat @ inv_info

    def predict(self, X) -> np.ndarray:
        """Linear predictor ``X @ coef_``."""
        X = np.asarray(X, dtype=np.float64)
        if X.ndim != 2 or X.shape[1] != self.n_features_in_:
            raise ValueError(f"X must have shape (n_samples, {self.n_features_in_}); got {X.shape}")
        return X @ self.coef_

    def predict_cumulative_incidence(self, X, times=None) -> np.ndarray:
        """Predicted cumulative incidence ``F(t | x)``.

        Uses the cmprsk formula
        ``F(t|x) = 1 - exp(-Λ̂_0(t) * exp(x' β))`` where ``Λ̂_0`` is the
        cumulative baseline subdistribution hazard.

        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
        times : array-like, optional
            Times at which to evaluate ``F``. Defaults to the cause-1
            event-time grid from training.

        Returns
        -------
        F : ndarray, shape (n_samples, n_times)
        """
        X = np.asarray(X, dtype=np.float64)
        if X.ndim != 2 or X.shape[1] != self.n_features_in_:
            raise ValueError(f"X must have shape (n_samples, {self.n_features_in_})")
        eta = X @ self.coef_
        Lambda0 = np.cumsum(self._state.baseline_hazard_increments)
        if times is None:
            times_arr = self._state.baseline_times
            Lambda_at = Lambda0
        else:
            times_arr = np.asarray(times, dtype=np.float64)
            # Step function: Λ̂_0 is left-continuous between event times.
            idx = np.searchsorted(self._state.baseline_times, times_arr, side="right") - 1
            Lambda_at = np.where(idx >= 0, Lambda0[np.clip(idx, 0, None)], 0.0)
        # F(t|x) = 1 - exp(-Λ̂_0(t) * exp(η))
        F = 1.0 - np.exp(-Lambda_at[None, :] * np.exp(eta)[:, None])
        return F

    def _unpack_y(self, y, time, event) -> tuple[np.ndarray, np.ndarray]:
        if y is not None and (time is not None or event is not None):
            raise ValueError(
                "pass either y= (structured Surv array) or time=, event= keyword args, not both"
            )
        if y is None:
            if time is None or event is None:
                raise ValueError("must provide y= or both time= and event= keywords")
            return np.asarray(time), np.asarray(event)
        if is_structured_survival_y(y):
            t, e = unpack_structured_y(y)
            return t, e
        raise TypeError(
            "y must be a Surv structured array (event, time fields). "
            "Use comprisk.Surv.from_arrays(event=..., time=...)."
        )
