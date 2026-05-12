"""Penalized Fine-Gray subdistribution-hazard regression for competing risks.

Variable selection (LASSO / ridge / elastic-net / SCAD / MCP) for the
proportional subdistribution-hazards model of Fine & Gray (1999), fit by
cyclic coordinate descent on the IPCW-weighted partial (pseudo-)likelihood.

The optimization target is

    minimize   -(1/n) * pl(beta)  +  sum_j p_lambda(|beta_j|)

where ``pl`` is the IPCW-weighted Breslow partial log-likelihood (the
Geskus (2011) expanded-dataset weighted Cox reformulation, evaluated
without materializing the expansion -- see :mod:`comprisk.fine_gray`) and
``p_lambda`` is one of the separable penalties

* LASSO / elastic-net : ``lambda * (alpha |t| + (1 - alpha) t^2 / 2)``
* ridge               : ``lambda * t^2 / 2``  (elastic-net with alpha = 0)
* MCP (Zhang 2010)    : ``lambda |t| - t^2 / (2 gamma)`` for ``|t| <= gamma lambda``
* SCAD (Fan & Li 2001): the smoothly-clipped quadratic of Fan & Li

with an optional ridge admixture (``alpha`` blend) for MCP/SCAD.

Each coordinate is updated by the soft-/firm-thresholding rule applied to
a one-term Taylor expansion of the partial likelihood (Simon et al. 2011's
quadratic approximation with the diagonal of the partial-likelihood
information w.r.t. the linear predictor), warm-started along a
data-driven ``lambda`` path from ``lambda_max`` (the smallest value that
zeroes every penalized coefficient) down to ``lambda_min_ratio * lambda_max``.

References
----------
Fine, J.P., Gray, R.J. (1999). "A proportional hazards model for the
subdistribution of a competing risk." *JASA* 94(446):496-509.

Geskus, R.B. (2011). "Cause-specific cumulative incidence estimation and
the Fine and Gray model under both left truncation and right censoring."
*Biometrics* 67(1):39-49.

Fu, Z., Parikh, C.R., Zhou, B. (2017). "Penalized variable selection in
competing risks regression." *Lifetime Data Analysis* 23:353-376.

Kawaguchi, E.S., Shen, J.I., Suchard, M.A., Li, G. (2021). "Scalable
algorithms for large competing risks data." *JCGS* 30(3):685-693.

Simon, N., Friedman, J., Hastie, T., Tibshirani, R. (2011).
"Regularization paths for Cox's proportional hazards model via coordinate
descent." *Journal of Statistical Software* 39(5):1-13.

Breheny, P., Huang, J. (2011). "Coordinate descent algorithms for
nonconvex penalized regression, with applications to biological feature
selection." *Annals of Applied Statistics* 5(1):232-253.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from sklearn.base import BaseEstimator

from comprisk._sklearn_compat import is_structured_survival_y, unpack_structured_y
from comprisk.fine_gray import (
    _baseline_subdist_hazard,
    _build_event_time_grid,
    _km_censoring_left_limit,
)

__all__ = ["PenalizedFineGrayRegression"]

_VALID_PENALTIES = ("lasso", "ridge", "elasticnet", "mcp", "scad")
# Drop-near-constant covariate columns at this standardized-scale floor
# (matches the convention used by penalized-regression toolkits).
_SCALE_FLOOR = 1e-6
# Coordinate updates landing this close to zero on the standardized scale
# are snapped to exactly zero -- a soft-threshold step at |z| just above
# the L1 boundary returns a value of size O(machine eps), which would
# otherwise spuriously inflate the active set and blow up the |beta|^-1
# penalty-curvature term in the sandwich SE.
_NUMERICAL_ZERO = 1e-10


# ---------------------------------------------------------------------------
# Proximal / threshold operators
#
# Each returns the minimizer over ``b`` of
#     (v / 2) (b - z / v)^2  +  p_lambda(|b|)
# where ``v`` is the (scaled) curvature, ``l1 = lambda * alpha`` is the
# L1 weight and ``l2 = lambda * (1 - alpha)`` the ridge weight. The MCP and
# SCAD forms are the firm-thresholding rules of Breheny & Huang (2011)
# (the regime boundaries follow the standardized-design convention common
# to the penalized-regression literature, in which the linear-regression
# curvature is 1).
# ---------------------------------------------------------------------------


def _soft_threshold(z: float, t: float) -> float:
    """``sign(z) * max(|z| - t, 0)``."""
    if z > t:
        return z - t
    if z < -t:
        return z + t
    return 0.0


def _prox_lasso(z: float, l1: float, l2: float, v: float) -> float:
    if abs(z) <= l1:
        return 0.0
    return _soft_threshold(z, l1) / (v + l2)


def _prox_mcp(z: float, l1: float, l2: float, gamma: float, v: float) -> float:
    if abs(z) <= l1:
        return 0.0
    if abs(z) <= gamma * l1 * (1.0 + l2):
        return _soft_threshold(z, l1) / (v * (1.0 + l2 - 1.0 / gamma))
    return z / (v * (1.0 + l2))


def _prox_scad(z: float, l1: float, l2: float, gamma: float, v: float) -> float:
    if abs(z) <= l1:
        return 0.0
    if abs(z) <= l1 * (2.0 + l2):
        return _soft_threshold(z, l1) / (v * (1.0 + l2))
    if abs(z) <= gamma * l1 * (1.0 + l2):
        return _soft_threshold(z, gamma * l1 / (gamma - 1.0)) / (
            v * (1.0 - 1.0 / (gamma - 1.0) + l2)
        )
    return z / (v * (1.0 + l2))


def _penalty_derivative(abs_beta: np.ndarray, lam: float, penalty: str, gamma: float) -> np.ndarray:
    """``p'_lambda(|beta_j|)`` evaluated elementwise (0 where ``beta_j == 0``).

    Used to build the ridge-style curvature correction in the sandwich
    standard error (Fan & Li 2001 eq. 3.3; Fu et al. 2017).
    """
    out = np.zeros_like(abs_beta)
    nz = abs_beta > 0.0
    b = abs_beta[nz]
    if penalty in ("lasso", "ridge", "elasticnet"):
        out[nz] = lam
    elif penalty == "mcp":
        out[nz] = np.maximum(lam - b / gamma, 0.0)
    elif penalty == "scad":
        out[nz] = np.where(b <= lam, lam, np.maximum(gamma * lam - b, 0.0) / (gamma - 1.0))
    return out


# ---------------------------------------------------------------------------
# IPCW-weighted partial-likelihood working quantities
# ---------------------------------------------------------------------------


def _psh_working(
    eta: np.ndarray,
    time: np.ndarray,
    event: np.ndarray,
    cg_index: np.ndarray,
    ncg: int,
    cause: int,
    event_times: np.ndarray,
    g_at_event_times: np.ndarray,  # (n_e, ncg)
    g_at_subject_times: np.ndarray,  # (n,)
):
    """Per-subject score residuals, curvatures and the weighted log PL.

    Returns ``(st, w, loglik, log_s0)`` where, with linear predictor
    ``eta``:

    * ``st[j] = dl/d(eta_j)``  -- the subdistribution score residual of
      subject ``j``; ``sum_j st[j] x_jk`` is the gradient of ``pl`` in
      coefficient ``k``.
    * ``w[j] = -d^2 l / d(eta_j)^2`` (the diagonal of the Breslow
      information w.r.t. ``eta``, dropping the off-diagonal mean-square
      term, after Simon et al. 2011) -- a non-negative working weight.
    * ``loglik`` -- the IPCW-weighted Breslow partial log-likelihood.
    * ``log_s0`` -- ``log S0(t_e)`` at every unique cause-of-interest
      event time, on the natural (un-shifted) scale.

    Returns ``None`` if the weighted risk set degenerates.

    The subdistribution risk set of a cause-of-interest event at time
    ``t`` is the standard at-risk set ``{i : time_i >= t}`` plus the
    competing-event subjects with ``time_i < t``, the latter down-weighted
    by ``G_k(t-) / G_k(time_i-)`` where ``G_k`` is the Kaplan-Meier of the
    censoring distribution in subject ``i``'s censoring stratum ``k``.
    Everything is assembled by reverse cumulative sums over the unique
    event-time grid -- no ``O(n^2)`` risk-set scan and no materialized
    Geskus expansion.
    """
    n = time.shape[0]
    n_e = event_times.shape[0]
    eta_max = float(np.max(eta)) if n else 0.0
    r = np.exp(eta - eta_max)

    # ---- S0(t_e): standard at-risk part ({i : time_i >= t_e}) ----
    order = np.argsort(time, kind="stable")
    t_sorted = time[order]
    r_rev = np.cumsum(r[order][::-1])[::-1]
    r_rev_p = np.append(r_rev, 0.0)
    idx_e = np.searchsorted(t_sorted, event_times, side="left")
    s0 = r_rev_p[idx_e].astype(np.float64)

    # ---- S0(t_e): competing-event part, per censoring stratum ----
    is_competing = (event != cause) & (event != 0)
    for k in range(ncg):
        m = is_competing & (cg_index == k)
        if not np.any(m):
            continue
        tj = time[m]
        gj = g_at_subject_times[m]
        gj_safe = np.where(gj > 0.0, gj, np.nan)
        oj = np.argsort(tj, kind="stable")
        cum_c = np.cumsum((r[m] / gj_safe)[oj])
        ie = np.searchsorted(tj[oj], event_times, side="left") - 1
        valid = ie >= 0
        if np.any(valid):
            s0[valid] += g_at_event_times[valid, k] * cum_c[ie[valid]]

    if not np.all(np.isfinite(s0)) or np.any(s0 <= 0.0):
        return None

    # ---- cause-of-interest event counts d_e and log S0 (natural scale) ----
    is_cause = event == cause
    eidx_cause = np.searchsorted(event_times, time[is_cause])
    d_e = np.bincount(eidx_cause, minlength=n_e).astype(np.float64)
    log_s0 = np.log(s0) + eta_max
    loglik = float(np.sum(eta[is_cause])) - float(np.sum(d_e * log_s0))

    # ---- per-subject cumulative drag over the event grid ----
    w1 = d_e / s0  # sum_e d_e / S0_e  building block
    w2 = d_e / (s0 * s0)
    cum_a = np.append(0.0, np.cumsum(w1))
    cum_a2 = np.append(0.0, np.cumsum(w2))
    n_le = np.searchsorted(event_times, time, side="right")  # # event times <= time_j
    a_j = cum_a[n_le]
    a2_j = cum_a2[n_le]

    st = np.where(is_cause, 1.0, 0.0)
    st -= r * a_j  # standard at-risk membership (all subjects)
    w = r * a_j - (r * r) * a2_j

    for k in range(ncg):
        m = is_competing & (cg_index == k)
        if not np.any(m):
            continue
        ge = g_at_event_times[:, k]
        gw1 = ge * w1
        gw2 = (ge * ge) * w2
        suf1 = np.append(np.cumsum(gw1[::-1])[::-1], 0.0)
        suf2 = np.append(np.cumsum(gw2[::-1])[::-1], 0.0)
        fe = n_le[m]  # first event index with t_e > time_j
        gj = g_at_subject_times[m]
        gj_safe = np.where(gj > 0.0, gj, np.nan)
        cfac = r[m] / gj_safe
        st[m] -= cfac * suf1[fe]
        w[m] += cfac * suf1[fe] - (cfac * cfac) * suf2[fe]

    w = np.where(np.isfinite(w) & (w > 0.0), w, 0.0)
    if not np.all(np.isfinite(st)):
        return None
    return st, w, loglik, log_s0


def _converged(beta: np.ndarray, beta_old: np.ndarray, tol: float) -> bool:
    """Relative-change convergence over the last coordinate sweep.

    Converged unless ``|beta_j - beta_old_j| / |beta_old_j| > tol`` for
    some ``j`` (a coefficient leaving or entering the active set forces
    another sweep; a ``0 -> 0`` coordinate is treated as converged).
    """
    with np.errstate(divide="ignore", invalid="ignore"):
        rel = np.abs(beta - beta_old) / np.abs(beta_old)
    rel = np.where(np.isnan(rel), 0.0, rel)
    return bool(np.all(rel <= tol))


def _fit_path(
    x_std: np.ndarray,
    time: np.ndarray,
    event: np.ndarray,
    cg_index: np.ndarray,
    ncg: int,
    cause: int,
    event_times: np.ndarray,
    g_at_event_times: np.ndarray,
    g_at_subject_times: np.ndarray,
    lambdas: np.ndarray,
    prox_kind: str,
    alpha: float,
    gamma: float,
    max_iter: int,
    tol: float,
):
    """Warm-started cyclic coordinate descent over the ``lambda`` path.

    ``x_std`` is the standardized design; coefficients returned are on the
    standardized scale.
    """
    n, p = x_std.shape
    n_lambda = lambdas.shape[0]
    coef_path = np.zeros((p, n_lambda))
    n_iter = np.zeros(n_lambda, dtype=int)
    deviance = np.full(n_lambda, np.nan)
    converged_flags = np.zeros(n_lambda, dtype=bool)
    last_w = np.zeros((n, n_lambda))

    res0 = _psh_working(
        np.zeros(n),
        time,
        event,
        cg_index,
        ncg,
        cause,
        event_times,
        g_at_event_times,
        g_at_subject_times,
    )
    if res0 is None:
        raise RuntimeError("weighted Fine-Gray risk set is degenerate at beta = 0")
    null_dev = -2.0 * res0[2]

    if prox_kind == "mcp":
        prox = lambda z, l1, l2, v: _prox_mcp(z, l1, l2, gamma, v)  # noqa: E731
    elif prox_kind == "scad":
        prox = lambda z, l1, l2, v: _prox_scad(z, l1, l2, gamma, v)  # noqa: E731
    else:
        prox = _prox_lasso

    beta = np.zeros(p)  # warm-started across lambdas
    eta = np.zeros(n)
    for li in range(n_lambda):
        lam = float(lambdas[li])
        l1 = lam * alpha
        l2 = lam * (1.0 - alpha)
        cur_dev = 0.0
        w_at_sol = res0[1]
        for _ in range(max_iter):
            if cur_dev - null_dev > 0.99 * null_dev:
                break
            n_iter[li] += 1
            res = _psh_working(
                eta,
                time,
                event,
                cg_index,
                ncg,
                cause,
                event_times,
                g_at_event_times,
                g_at_subject_times,
            )
            if res is None:
                break
            st, w, _, _ = res
            with np.errstate(divide="ignore", invalid="ignore"):
                resid = np.where(w > 0.0, st / w, 0.0)  # working residual z - eta
            beta_old = beta.copy()
            for j in range(p):
                xj = x_std[:, j]
                wx = w * xj
                xwx = float(np.dot(wx, xj))
                if xwx <= 0.0:
                    if beta[j] != 0.0:
                        shift = -beta[j]
                        beta[j] = 0.0
                        eta += shift * xj
                        resid -= shift * xj
                    continue
                v = xwx / n
                u = float(np.dot(wx, resid)) / n + v * beta[j]
                bj_new = prox(u, l1, l2, v)
                if abs(bj_new) < _NUMERICAL_ZERO:
                    bj_new = 0.0
                shift = bj_new - beta[j]
                if shift != 0.0:
                    beta[j] = bj_new
                    eta += shift * xj
                    resid -= shift * xj
            res_post = _psh_working(
                eta,
                time,
                event,
                cg_index,
                ncg,
                cause,
                event_times,
                g_at_event_times,
                g_at_subject_times,
            )
            if res_post is None:
                break
            w_at_sol = res_post[1]
            cur_dev = -2.0 * res_post[2]
            if _converged(beta, beta_old, tol):
                converged_flags[li] = True
                break
        coef_path[:, li] = beta
        deviance[li] = cur_dev
        last_w[:, li] = w_at_sol
    return coef_path, n_iter, deviance, null_dev, converged_flags, last_w


def _standardize(x: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Center and scale columns to unit (population) variance.

    Returns ``(x_std, center, scale, keep)`` where ``keep`` masks columns
    with non-negligible variance; near-constant columns are dropped from
    ``x_std`` (their fitted coefficient is forced to zero).
    """
    center = x.mean(axis=0)
    scale = x.std(axis=0, ddof=0)
    keep = scale > _SCALE_FLOOR
    safe_scale = np.where(keep, scale, 1.0)
    x_std = (x - center) / safe_scale
    return x_std[:, keep], center, scale, keep


def _lambda_grid(
    x_std: np.ndarray,
    time: np.ndarray,
    event: np.ndarray,
    cg_index: np.ndarray,
    ncg: int,
    cause: int,
    event_times: np.ndarray,
    g_at_event_times: np.ndarray,
    g_at_subject_times: np.ndarray,
    alpha: float,
    n_lambda: int,
    lambda_min_ratio: float,
) -> np.ndarray:
    """Descending ``lambda`` path: ``lambda_max`` down to ``ratio * lambda_max``.

    ``lambda_max`` is the smallest ``lambda`` zeroing every penalized
    coefficient -- the KKT bound ``max_j |grad_j pl(0)| / (n * alpha)`` --
    using a floor on ``alpha`` so a usable path is produced for ridge.
    """
    n = x_std.shape[0]
    res0 = _psh_working(
        np.zeros(n),
        time,
        event,
        cg_index,
        ncg,
        cause,
        event_times,
        g_at_event_times,
        g_at_subject_times,
    )
    st0 = res0[0]
    grad0 = x_std.T @ st0
    lambda_max = float(np.max(np.abs(grad0))) / (n * max(alpha, 1e-3))
    if not np.isfinite(lambda_max) or lambda_max <= 0.0:
        lambda_max = 1.0
    log_grid = np.linspace(np.log(lambda_max), np.log(lambda_max * lambda_min_ratio), n_lambda)
    return np.exp(log_grid)


def _sandwich_se_path(
    x_std: np.ndarray,
    coef_path_std: np.ndarray,
    last_w: np.ndarray,
    lambdas: np.ndarray,
    scale_keep: np.ndarray,
    penalty: str,
    gamma: float,
) -> np.ndarray:
    """Per-lambda sandwich SE on the standardized scale, then unscaled.

    ``se_j(lambda) = sqrt(diag( M Z'WZ M )) / scale_j`` with
    ``M = pinv(Z'WZ + n P)``, ``Z'WZ = x_std' diag(w) x_std`` evaluated at
    the fitted curvatures, and ``P`` the diagonal ridge-style penalty
    correction ``p'_lambda(|beta_j|) / |beta_j|`` (Fan & Li 2001; Fu et al.
    2017). Coefficients that are exactly zero contribute ``P_jj = 0``.
    """
    n, p_kept = x_std.shape
    n_lambda = lambdas.shape[0]
    se_kept = np.zeros((p_kept, n_lambda))
    for li in range(n_lambda):
        b = coef_path_std[:, li]
        w = last_w[:, li]
        zwz = x_std.T @ (w[:, None] * x_std)
        pdiag = np.zeros(p_kept)
        nz = np.abs(b) > 0.0
        if np.any(nz):
            pdiag[nz] = _penalty_derivative(
                np.abs(b[nz]), float(lambdas[li]), penalty, gamma
            ) / np.abs(b[nz])
        m = np.linalg.pinv(zwz + n * np.diag(pdiag))
        var = m @ zwz @ m
        se_kept[:, li] = np.sqrt(np.maximum(np.diag(var), 0.0))
    se_kept = se_kept / scale_keep[:, None]
    return se_kept


# ---------------------------------------------------------------------------
# Fit artifacts container
# ---------------------------------------------------------------------------


@dataclass
class _PFGState:
    """Non-public artifacts needed by ``predict`` / cross-validation."""

    x_original: np.ndarray
    time: np.ndarray
    event_internal: np.ndarray
    cg_index: np.ndarray
    cengroups: np.ndarray
    cause: int
    event_times: np.ndarray
    g_at_event_times: np.ndarray
    g_at_subject_times: np.ndarray
    baseline_times: np.ndarray
    baseline_hazard_increments: np.ndarray


class PenalizedFineGrayRegression(BaseEstimator):
    """Penalized Fine-Gray subdistribution-hazard regression.

    Fits the proportional subdistribution-hazards model with a sparsity- or
    shrinkage-inducing penalty by cyclic coordinate descent on the
    IPCW-weighted partial likelihood, warm-started along a ``lambda`` path.
    Mirrors the algorithm of Fu et al. (2017) / Kawaguchi et al. (2021).

    Parameters
    ----------
    penalty : {"lasso", "ridge", "elasticnet", "mcp", "scad"}, default "lasso"
        Penalty family. ``"ridge"`` forces ``l1_ratio = 0``;
        ``"elasticnet"`` blends L1 and L2 by ``l1_ratio``; ``"mcp"`` /
        ``"scad"`` are the nonconvex penalties of Zhang (2010) / Fan & Li
        (2001), optionally admixed with a ridge term by ``l1_ratio``.
    l1_ratio : float, default 1.0
        Elastic-net mixing ``alpha``: penalty is ``lambda (alpha * <penalty>
        + (1 - alpha) * t^2 / 2)``. Ignored when ``penalty == "ridge"``
        (treated as ``0``). Must be in ``(0, 1]`` otherwise.
    gamma : float or None, default None
        Concavity parameter for MCP (> 1; default 2.7) and SCAD (> 2;
        default 3.7). Ignored for the convex penalties.
    n_lambda : int, default 100
        Number of ``lambda`` values on the auto-generated path.
    lambda_min_ratio : float, default 0.001
        Smallest ``lambda`` as a fraction of ``lambda_max``.
    lambdas : array-like or None, default None
        Explicit ``lambda`` grid; overrides ``n_lambda`` /
        ``lambda_min_ratio``. Sorted to descending order internally.
    standardize : bool, default True
        Center and scale covariates to unit variance before fitting; the
        penalty then acts on the standardized coefficients (coefficients
        are reported on the original scale).
    cause : int, default 1
        Cause-of-interest event code.
    cencode : int, default 0
        Censoring event code.
    cv : int or None, default None
        If a positive integer ``K``, run ``K``-fold cross-validation on the
        weighted cross-validated partial-likelihood deviance and select
        ``lambda``; otherwise ``lambda`` is chosen by BIC over the path.
    cv_random_state : int or None, default None
        Seed for the cross-validation fold split.
    max_iter : int, default 1000
        Maximum coordinate-descent sweeps per ``lambda``.
    tol : float, default 1e-4
        Relative-change convergence tolerance.

    Attributes
    ----------
    coef_ : ndarray, shape (n_features,)
        Coefficients at the selected ``lambda``.
    se_ : ndarray, shape (n_features,)
        Sandwich standard errors at the selected ``lambda``.
    lambda_ : float
        Selected penalty value.
    lambda_index_ : int
        Index of the selected ``lambda`` in ``lambdas_``.
    coef_path_ : ndarray, shape (n_features, n_lambda)
        Coefficients along the full path (original scale).
    se_path_ : ndarray, shape (n_features, n_lambda)
        Sandwich SEs along the path.
    lambdas_ : ndarray, shape (n_lambda,)
        The ``lambda`` grid (descending).
    deviance_path_ : ndarray, shape (n_lambda,)
        ``-2 * partial-loglik`` along the path.
    null_deviance_ : float
        Deviance at ``beta = 0``.
    bic_path_ : ndarray, shape (n_lambda,)
        ``deviance + df * log(n)`` along the path (``df`` = #nonzero).
    n_iter_path_ : ndarray of int, shape (n_lambda,)
    converged_path_ : ndarray of bool, shape (n_lambda,)
    lambda_min_ : float or None
        CV-deviance-minimizing ``lambda`` (only when ``cv`` is set).
    lambda_1se_ : float or None
        Largest ``lambda`` within one SE of the CV minimum.
    cv_deviance_ : ndarray or None
        Mean CV deviance per ``lambda``.
    cv_deviance_se_ : ndarray or None
        Standard error of the CV deviance per ``lambda``.

    Examples
    --------
    >>> import numpy as np
    >>> from comprisk import PenalizedFineGrayRegression, Surv
    >>> rng = np.random.default_rng(0)
    >>> n = 300
    >>> X = rng.normal(size=(n, 8))
    >>> eta = 0.7 * X[:, 0] - 0.5 * X[:, 1]
    >>> time = rng.exponential(np.exp(-eta)) + 0.05
    >>> event = rng.choice([0, 1, 2], size=n, p=[0.3, 0.5, 0.2])
    >>> y = Surv.from_arrays(event=event, time=time)
    >>> fit = PenalizedFineGrayRegression(penalty="lasso", cv=5,
    ...                                   cv_random_state=0).fit(X, y)
    >>> fit.coef_.shape
    (8,)
    """

    def __init__(
        self,
        *,
        penalty: str = "lasso",
        l1_ratio: float = 1.0,
        gamma: float | None = None,
        n_lambda: int = 100,
        lambda_min_ratio: float = 1e-3,
        lambdas=None,
        standardize: bool = True,
        cause: int = 1,
        cencode: int = 0,
        cv: int | None = None,
        cv_random_state: int | None = None,
        max_iter: int = 1000,
        tol: float = 1e-4,
    ) -> None:
        self.penalty = penalty
        self.l1_ratio = l1_ratio
        self.gamma = gamma
        self.n_lambda = n_lambda
        self.lambda_min_ratio = lambda_min_ratio
        self.lambdas = lambdas
        self.standardize = standardize
        self.cause = cause
        self.cencode = cencode
        self.cv = cv
        self.cv_random_state = cv_random_state
        self.max_iter = max_iter
        self.tol = tol

    # -- penalty resolution -------------------------------------------------

    def _resolve_penalty(self) -> tuple[str, float, float]:
        """Return ``(prox_kind, alpha, gamma)`` after validation."""
        if self.penalty not in _VALID_PENALTIES:
            raise ValueError(f"penalty must be one of {_VALID_PENALTIES}; got {self.penalty!r}")
        if self.penalty == "ridge":
            alpha = 0.0
        elif self.penalty == "lasso":
            alpha = 1.0
        else:
            alpha = float(self.l1_ratio)
            if not (0.0 < alpha <= 1.0):
                raise ValueError(f"l1_ratio must be in (0, 1]; got {self.l1_ratio}")
        if self.penalty == "mcp":
            prox_kind = "mcp"
            gamma = 2.7 if self.gamma is None else float(self.gamma)
            if gamma <= 1.0:
                raise ValueError(f"gamma must be > 1 for MCP; got {gamma}")
        elif self.penalty == "scad":
            prox_kind = "scad"
            gamma = 3.7 if self.gamma is None else float(self.gamma)
            if gamma <= 2.0:
                raise ValueError(f"gamma must be > 2 for SCAD; got {gamma}")
        else:
            prox_kind = "lasso"
            gamma = float("nan")
        return prox_kind, alpha, gamma

    # -- input handling -----------------------------------------------------

    @staticmethod
    def _unpack_y(y, time, event) -> tuple[np.ndarray, np.ndarray]:
        if y is not None and (time is not None or event is not None):
            raise ValueError(
                "pass either y= (structured Surv array) or time=, event= keywords, not both"
            )
        if y is None:
            if time is None or event is None:
                raise ValueError("must provide y= or both time= and event= keywords")
            return np.asarray(time), np.asarray(event)
        if is_structured_survival_y(y):
            return unpack_structured_y(y)
        raise TypeError(
            "y must be a Surv structured array (event, time fields). "
            "Use comprisk.Surv.from_arrays(event=..., time=...)."
        )

    def _prepare(self, x, time_arr, event_arr, cengroup):
        """Validate / normalize inputs; build the IPCW machinery once.

        Returns ``(x, time, event_internal, cg_index, cengroups,
        event_times, g_at_event_times, g_at_subject_times)``.
        """
        x = np.asarray(x, dtype=np.float64)
        if x.ndim != 2:
            raise ValueError(f"X must be 2-D; got shape {x.shape}")
        n = x.shape[0]
        time_arr = np.asarray(time_arr, dtype=np.float64)
        event_arr = np.asarray(event_arr, dtype=np.int64)
        if time_arr.shape[0] != n or event_arr.shape[0] != n:
            raise ValueError(f"time/event length must match X.shape[0]={n}")
        if cengroup is None:
            cengroup_arr = np.zeros(n, dtype=np.int64)
        else:
            cengroup_arr = np.asarray(cengroup, dtype=np.int64)
            if cengroup_arr.shape[0] != n:
                raise ValueError("cengroup length must match X")

        event_internal = event_arr.copy()
        if self.cencode != 0:
            cens_mask = event_arr == self.cencode
            event_internal[event_arr == 0] = -999
            event_internal[cens_mask] = 0

        cengroups = np.unique(cengroup_arr)
        cg_index = np.searchsorted(cengroups, cengroup_arr)

        if not np.any(event_internal == self.cause):
            raise ValueError(
                f"no subjects have event == cause ({self.cause}); "
                f"unique events seen: {sorted(np.unique(event_arr).tolist())}"
            )
        event_times = _build_event_time_grid(time_arr, event_internal, self.cause)
        if event_times.size == 0:
            raise ValueError("no positive-time cause-of-interest events; cannot fit")

        g_at_event_times = _km_censoring_left_limit(
            time_arr, event_internal, cengroup_arr, event_times
        )
        g_at_subject_times = np.empty(n, dtype=np.float64)
        for ki, cg in enumerate(cengroups):
            mask = cengroup_arr == cg
            if not np.any(mask):
                continue
            g_at_subject_times[mask] = _km_censoring_left_limit(
                time_arr, event_internal, cengroup_arr, time_arr[mask].astype(np.float64)
            )[:, ki]
        return (
            x,
            time_arr,
            event_internal,
            cg_index,
            cengroups,
            event_times,
            g_at_event_times,
            g_at_subject_times,
        )

    # -- fit ----------------------------------------------------------------

    def fit(
        self, X, y=None, *, time=None, event=None, cengroup=None
    ) -> PenalizedFineGrayRegression:
        """Fit the penalized Fine-Gray model.

        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
        y : structured array, optional
            ``Surv``-style structured array with ``event`` and ``time``
            fields. Mutually exclusive with ``time`` / ``event``.
        time, event : array-like, shape (n_samples,), optional
            Legacy three-argument form ``fit(X, time=, event=)``.
        cengroup : array-like of int, shape (n_samples,), optional
            Censoring stratum for each subject. ``None`` -> single stratum.

        Returns
        -------
        self
        """
        time_arr, event_arr = self._unpack_y(y, time, event)
        prox_kind, alpha, gamma = self._resolve_penalty()
        (
            x,
            time_arr,
            event_internal,
            cg_index,
            cengroups,
            event_times,
            g_at_event_times,
            g_at_subject_times,
        ) = self._prepare(X, time_arr, event_arr, cengroup)
        n, p = x.shape
        ncg = cengroups.shape[0]

        if self.standardize:
            x_kept, _center, scale, keep = _standardize(x)
        else:
            keep = np.ones(p, dtype=bool)
            scale = np.ones(p)
            x_kept = x
        scale_keep = scale[keep]

        if self.lambdas is not None:
            lambdas = np.sort(np.asarray(self.lambdas, dtype=np.float64))[::-1].copy()
            if lambdas.size < 1:
                raise ValueError("lambdas must be non-empty")
        else:
            if self.n_lambda < 2:
                raise ValueError("n_lambda must be at least 2")
            lambdas = _lambda_grid(
                x_kept,
                time_arr,
                event_internal,
                cg_index,
                ncg,
                self.cause,
                event_times,
                g_at_event_times,
                g_at_subject_times,
                alpha,
                self.n_lambda,
                self.lambda_min_ratio,
            )

        coef_path_std, n_iter_path, deviance_path, null_dev, conv_flags, last_w = _fit_path(
            x_kept,
            time_arr,
            event_internal,
            cg_index,
            ncg,
            self.cause,
            event_times,
            g_at_event_times,
            g_at_subject_times,
            lambdas,
            prox_kind,
            alpha,
            gamma,
            self.max_iter,
            self.tol,
        )

        # Unstandardize.
        coef_path = np.zeros((p, lambdas.shape[0]))
        coef_path[keep] = coef_path_std / scale_keep[:, None]
        se_path = np.zeros((p, lambdas.shape[0]))
        se_path[keep] = _sandwich_se_path(
            x_kept, coef_path_std, last_w, lambdas, scale_keep, prox_kind, gamma
        )

        df = (np.abs(coef_path) > 0.0).sum(axis=0)
        bic_path = deviance_path + df * np.log(n)

        self.coef_path_ = coef_path
        self.se_path_ = se_path
        self.lambdas_ = lambdas
        self.deviance_path_ = deviance_path
        self.loglik_path_ = -0.5 * deviance_path
        self.null_deviance_ = null_dev
        self.bic_path_ = bic_path
        self.n_iter_path_ = n_iter_path
        self.converged_path_ = conv_flags
        self.n_features_in_ = p

        self._state = _PFGState(
            x_original=x,
            time=time_arr,
            event_internal=event_internal,
            cg_index=cg_index,
            cengroups=cengroups,
            cause=self.cause,
            event_times=event_times,
            g_at_event_times=g_at_event_times,
            g_at_subject_times=g_at_subject_times,
            baseline_times=np.empty(0),
            baseline_hazard_increments=np.empty(0),
        )

        # lambda selection
        self.lambda_min_ = None
        self.lambda_1se_ = None
        self.cv_deviance_ = None
        self.cv_deviance_se_ = None
        if self.cv is not None:
            if not (isinstance(self.cv, (int, np.integer)) and self.cv >= 2):
                raise ValueError("cv must be None or an integer >= 2")
            cvm, cvse = self._cross_validate(
                X, time_arr, event_arr, cengroup, lambdas, alpha, gamma, prox_kind
            )
            self.cv_deviance_ = cvm
            self.cv_deviance_se_ = cvse
            i_min = int(np.nanargmin(cvm))
            self.lambda_min_ = float(lambdas[i_min])
            thresh = cvm[i_min] + cvse[i_min]
            within = np.where(cvm <= thresh)[0]
            i_1se = int(within.min()) if within.size else i_min
            self.lambda_1se_ = float(lambdas[i_1se])
            self.lambda_index_ = i_min
        else:
            self.lambda_index_ = int(np.nanargmin(bic_path))

        sel = self.lambda_index_
        self.coef_ = coef_path[:, sel].copy()
        self.se_ = se_path[:, sel].copy()
        self.lambda_ = float(lambdas[sel])
        self.n_iter_ = int(n_iter_path[sel])
        self.converged_ = bool(conv_flags[sel])
        self.log_likelihood_ = float(self.loglik_path_[sel])

        bt, bdl = _baseline_subdist_hazard(
            self.coef_,
            x,
            time_arr,
            event_internal,
            cg_index,
            self.cause,
            event_times,
            g_at_event_times,
            g_at_subject_times,
        )
        self._state.baseline_times = bt
        self._state.baseline_hazard_increments = bdl
        return self

    # -- cross-validation ---------------------------------------------------

    def _cross_validate(self, X, time_arr, event_arr, cengroup, lambdas, alpha, gamma, prox_kind):
        """K-fold cross-validated weighted partial-likelihood deviance.

        Uses the Verweij & van Houwelingen (1993) cross-validated partial
        likelihood: for each held-out fold, the contribution of its
        cause-of-interest events to the *full-data* weighted partial
        likelihood is evaluated at the coefficients fitted on the other
        folds. Returns ``(mean_deviance, deviance_se)`` per ``lambda``.
        """
        x = np.asarray(X, dtype=np.float64)
        time_arr = np.asarray(time_arr, dtype=np.float64)
        event_arr = np.asarray(event_arr, dtype=np.int64)
        n = x.shape[0]
        k = int(self.cv)
        rng = np.random.default_rng(self.cv_random_state)
        fold_of = np.tile(np.arange(k), n // k + 1)[:n]
        rng.shuffle(fold_of)

        # Full-data IPCW machinery (shared across folds).
        (
            _x,
            time_full,
            event_full,
            cg_full,
            cengroups_full,
            event_times_full,
            g_e_full,
            g_s_full,
        ) = self._prepare(x, time_arr, event_arr, cengroup)
        ncg_full = cengroups_full.shape[0]

        n_lambda = lambdas.shape[0]
        dev_per_fold = np.full((k, n_lambda), np.nan)
        base_kwargs = dict(
            penalty=self.penalty,
            l1_ratio=self.l1_ratio,
            gamma=self.gamma,
            lambdas=lambdas,
            standardize=self.standardize,
            cause=self.cause,
            cencode=self.cencode,
            cv=None,
            max_iter=self.max_iter,
            tol=self.tol,
        )
        for fi in range(k):
            train = fold_of != fi
            test = ~train
            if not np.any(event_arr[train] == self.cause) or not np.any(
                event_arr[test] == self.cause
            ):
                continue
            cg_train = None if cengroup is None else np.asarray(cengroup)[train]
            sub = PenalizedFineGrayRegression(**base_kwargs)
            sub.fit(x[train], time=time_arr[train], event=event_arr[train], cengroup=cg_train)
            test_cause = test & (event_full == self.cause)
            if not np.any(test_cause):
                continue
            eidx_test = np.searchsorted(event_times_full, time_full[test_cause])
            for li in range(n_lambda):
                beta = sub.coef_path_[:, li]
                eta_full = x @ beta
                res = _psh_working(
                    eta_full,
                    time_full,
                    event_full,
                    cg_full,
                    ncg_full,
                    self.cause,
                    event_times_full,
                    g_e_full,
                    g_s_full,
                )
                if res is None:
                    continue
                log_s0 = res[3]
                contrib = float(np.sum(eta_full[test_cause] - log_s0[eidx_test]))
                dev_per_fold[fi, li] = -2.0 * contrib

        cvm = np.nansum(dev_per_fold, axis=0)
        n_valid = np.sum(~np.isnan(dev_per_fold), axis=0)
        with np.errstate(invalid="ignore"):
            spread = np.nanstd(dev_per_fold, axis=0, ddof=1)
        cvse = np.where(n_valid > 1, np.sqrt(np.maximum(n_valid, 1.0)) * spread, np.inf)
        return cvm, cvse

    # -- predict ------------------------------------------------------------

    def _check_is_fitted(self) -> None:
        if not hasattr(self, "coef_"):
            raise RuntimeError("estimator is not fitted; call fit() first")

    def predict(self, X) -> np.ndarray:
        """Linear predictor ``X @ coef_`` at the selected ``lambda``."""
        self._check_is_fitted()
        X = np.asarray(X, dtype=np.float64)
        if X.ndim != 2 or X.shape[1] != self.n_features_in_:
            raise ValueError(f"X must have shape (n_samples, {self.n_features_in_}); got {X.shape}")
        return X @ self.coef_

    def predict_cumulative_incidence(self, X, times=None) -> np.ndarray:
        """Predicted cumulative incidence ``F(t | x)`` at the selected ``lambda``.

        ``F(t|x) = 1 - exp(-Lambda_0(t) exp(x' beta))`` with ``Lambda_0``
        the Breslow cumulative baseline subdistribution hazard.

        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
        times : array-like, optional
            Times at which to evaluate; defaults to the training
            cause-of-interest event-time grid.

        Returns
        -------
        F : ndarray, shape (n_samples, n_times)
        """
        self._check_is_fitted()
        X = np.asarray(X, dtype=np.float64)
        if X.ndim != 2 or X.shape[1] != self.n_features_in_:
            raise ValueError(f"X must have shape (n_samples, {self.n_features_in_})")
        eta = X @ self.coef_
        cum_haz = np.cumsum(self._state.baseline_hazard_increments)
        if times is None:
            lam_at = cum_haz
        else:
            times_arr = np.asarray(times, dtype=np.float64)
            idx = np.searchsorted(self._state.baseline_times, times_arr, side="right") - 1
            lam_at = np.where(idx >= 0, cum_haz[np.clip(idx, 0, None)], 0.0)
        return 1.0 - np.exp(-lam_at[None, :] * np.exp(eta)[:, None])

    def coef_at(
        self, lambda_index: int | None = None, *, lambda_value: float | None = None
    ) -> np.ndarray:
        """Coefficients at a specific path index or (nearest) ``lambda`` value."""
        self._check_is_fitted()
        if (lambda_index is None) == (lambda_value is None):
            raise ValueError("specify exactly one of lambda_index= or lambda_value=")
        if lambda_value is not None:
            lambda_index = int(np.argmin(np.abs(self.lambdas_ - lambda_value)))
        return self.coef_path_[:, lambda_index].copy()
