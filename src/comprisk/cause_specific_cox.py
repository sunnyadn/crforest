"""Cause-specific Cox proportional-hazards regression for competing risks.

Fits a Cox PH model under the cause-specific framing: subjects who
experience a competing event are treated as censored at their event time.
Targets parity with ``survival::coxph(..., method="breslow")`` on
``Surv(time, event == cause)``.

References
----------
Cox, D.R. (1972). "Regression models and life-tables." *Journal of the
Royal Statistical Society B* 34(2):187-220.

Prentice, R.L., Kalbfleisch, J.D., Peterson, A.V., Flournoy, N., Farewell,
V.T., Breslow, N.E. (1978). "The analysis of failure times in the presence
of competing risks." *Biometrics* 34(4):541-554.
"""

from __future__ import annotations

import numpy as np

from comprisk._sklearn_compat import is_structured_survival_y, unpack_structured_y

__all__ = ["CauseSpecificCox"]


def _cox_breslow_pl(
    beta: np.ndarray,
    X: np.ndarray,
    time: np.ndarray,
    event: np.ndarray,  # 0/1 indicator (cause-specific failure)
) -> tuple[float, np.ndarray, np.ndarray]:
    """Negative log Breslow partial likelihood, score, observed info.

    Standard Cox PH formulation with Breslow tie handling. Numerically
    stabilised by max-subtraction; vectorised via reverse cumulative sums
    over time-sorted samples.
    """
    n, p = X.shape
    eta = X @ beta
    eta_max = float(np.max(eta)) if n > 0 else 0.0
    r = np.exp(eta - eta_max)

    order = np.argsort(time, kind="stable")
    t_asc = time[order]
    r_asc = r[order]
    Xr_asc = X[order] * r_asc[:, None]

    # Reverse cumulative sums: at threshold position k, sum over i >= k.
    # We don't materialise (n, p, p) — see Hessian rewrite below.
    r_rev = np.cumsum(r_asc[::-1])[::-1]
    Xr_rev = np.cumsum(Xr_asc[::-1], axis=0)[::-1]

    # Unique event times (cause-specific failures only).
    is_event = event == 1
    if not np.any(is_event):
        return np.inf, np.zeros(p), np.eye(p)
    event_times = np.unique(time[is_event])
    n_e = event_times.shape[0]
    idx_e = np.searchsorted(t_asc, event_times, side="left")
    pad_r = np.zeros(1, dtype=r_rev.dtype)
    pad_X = np.zeros((1, p), dtype=Xr_rev.dtype)
    r_rev_p = np.concatenate([r_rev, pad_r])
    Xr_rev_p = np.concatenate([Xr_rev, pad_X], axis=0)
    S0 = r_rev_p[idx_e].astype(np.float64)
    S1 = Xr_rev_p[idx_e].astype(np.float64)

    if np.any(S0 <= 0):
        return np.inf, np.zeros(p), np.eye(p)

    log_S0 = np.log(S0)
    s1_over_s0 = S1 / S0[:, None]

    eidx = np.searchsorted(event_times, time[is_event])
    d_e = np.bincount(eidx, minlength=n_e).astype(np.float64)
    sumX_e = np.zeros((n_e, p), dtype=np.float64)
    X_events = X[is_event]
    for j in range(p):
        sumX_e[:, j] = np.bincount(eidx, weights=X_events[:, j], minlength=n_e)
    sum_eta_e = np.bincount(eidx, weights=eta[is_event] - eta_max, minlength=n_e)

    nll = -float(np.sum(sum_eta_e - d_e * log_S0))
    score = -np.sum(sumX_e - d_e[:, None] * s1_over_s0, axis=0)

    # Hessian via algebraic identity:
    #   Σ_e d_e (S2[e]/S0[e] - bar_x[e] bar_x[e]^T)
    #   = Σ_i r_i x_i x_i^T * Σ_{e: t_e ≤ t_i}(d_e/S0[e]) - Σ_e d_e bar_x[e] bar_x[e]^T
    weight_per_e = d_e / S0
    cum_w = np.concatenate([[0.0], np.cumsum(weight_per_e)])
    idx_t_in_e = np.searchsorted(event_times, time, side="right")
    w_subj = r * cum_w[idx_t_in_e]
    info = (X.T * w_subj) @ X
    info -= (s1_over_s0.T * d_e) @ s1_over_s0
    return nll, score, info


class CauseSpecificCox:
    """Cause-specific Cox proportional-hazards regression.

    Fits a Cox PH model with the cause-specific censoring rule: subjects
    experiencing a competing event are censored at that event time. Parity
    target: ``survival::coxph(Surv(time, event == cause) ~ X, method="breslow")``.

    Parameters
    ----------
    cause : int, default 1
        Cause-of-interest event code. All other positive event codes are
        competing events; ``0`` denotes administrative censoring. Both
        non-cause categories receive identical (censored) treatment.
    max_iter : int, default 25
    gtol : float, default 1e-9

    Attributes
    ----------
    coef_ : ndarray, shape (n_features,)
    se_ : ndarray, shape (n_features,)
    var_ : ndarray, shape (n_features, n_features)
    n_iter_ : int
    converged_ : bool
    log_likelihood_ : float
    log_likelihood_null_ : float

    Examples
    --------
    >>> import numpy as np
    >>> from comprisk import CauseSpecificCox, Surv
    >>> rng = np.random.default_rng(0)
    >>> n = 300
    >>> X = rng.normal(size=(n, 3))
    >>> time = rng.exponential(1.0, size=n)
    >>> event = rng.choice([0, 1, 2], size=n, p=[0.3, 0.5, 0.2])
    >>> y = Surv.from_arrays(event=event, time=time)
    >>> cs = CauseSpecificCox(cause=1).fit(X, y)
    >>> cs.coef_.shape
    (3,)
    """

    def __init__(
        self,
        *,
        cause: int = 1,
        max_iter: int = 25,
        gtol: float = 1e-9,
    ) -> None:
        self.cause = cause
        self.max_iter = max_iter
        self.gtol = gtol

    def fit(self, X, y=None, time=None, event=None) -> CauseSpecificCox:
        """Fit cause-specific Cox PH.

        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
        y : structured array, optional
            ``Surv`` array carrying ``event`` and ``time`` fields.
        time, event : array-like, optional
            Legacy three-argument form.
        """
        X = np.asarray(X, dtype=np.float64)
        if X.ndim != 2:
            raise ValueError(f"X must be 2-D; got shape {X.shape}")
        time_arr, event_arr = self._unpack_y(y, time, event)
        time_arr = np.asarray(time_arr, dtype=np.float64)
        event_arr = np.asarray(event_arr, dtype=np.int64)
        n = X.shape[0]
        if time_arr.shape[0] != n or event_arr.shape[0] != n:
            raise ValueError("time/event length must match X")

        # Cause-specific indicator.
        is_event = (event_arr == self.cause).astype(np.int64)
        if not np.any(is_event):
            raise ValueError(f"no cause-{self.cause} events; cannot fit")

        beta = np.zeros(X.shape[1], dtype=np.float64)
        nll, score, info = _cox_breslow_pl(beta, X, time_arr, is_event)
        converged = False
        n_iter = 0
        for it in range(1, self.max_iter + 1):
            n_iter = it
            try:
                step = -np.linalg.solve(info, score)
            except np.linalg.LinAlgError:
                step = -score / (np.abs(np.diag(info)) + 1.0)
            accept = False
            for _ in range(20):
                beta_new = beta + step
                nll_new, score_new, info_new = _cox_breslow_pl(beta_new, X, time_arr, is_event)
                if np.isfinite(nll_new) and nll_new <= nll + 1e-4 * np.dot(score, step):
                    accept = True
                    break
                step *= 0.5
            if not accept:
                break
            beta = beta_new
            nll, score, info = nll_new, score_new, info_new
            crit = float(np.max(np.abs(score) * np.maximum(np.abs(beta), 1.0)))
            rhs = float(max(abs(nll), 1.0) * self.gtol)
            if crit < rhs:
                converged = True
                break

        try:
            inv_info = np.linalg.inv(info)
        except np.linalg.LinAlgError as exc:
            raise RuntimeError(
                "observed information matrix is singular at the fitted "
                "coefficients; design may be rank-deficient"
            ) from exc

        nll0, _, _ = _cox_breslow_pl(np.zeros_like(beta), X, time_arr, is_event)

        self.coef_ = beta
        self.var_ = inv_info
        self.se_ = np.sqrt(np.maximum(np.diag(inv_info), 0.0))
        self.n_iter_ = int(n_iter)
        self.converged_ = bool(converged)
        self.log_likelihood_ = -float(nll)
        self.log_likelihood_null_ = -float(nll0)
        self.n_features_in_ = X.shape[1]
        return self

    def predict(self, X) -> np.ndarray:
        """Linear predictor ``X @ coef_``."""
        X = np.asarray(X, dtype=np.float64)
        if X.ndim != 2 or X.shape[1] != self.n_features_in_:
            raise ValueError(f"X must have shape (n_samples, {self.n_features_in_})")
        return X @ self.coef_

    def _unpack_y(self, y, time, event) -> tuple[np.ndarray, np.ndarray]:
        if y is not None and (time is not None or event is not None):
            raise ValueError("pass either y= or time=/event= keywords, not both")
        if y is None:
            if time is None or event is None:
                raise ValueError("must provide y= or time=, event= keywords")
            return np.asarray(time), np.asarray(event)
        if is_structured_survival_y(y):
            return unpack_structured_y(y)
        raise TypeError(
            "y must be a Surv structured array. Use comprisk.Surv.from_arrays(event=..., time=...)."
        )
