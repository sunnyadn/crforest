"""Split statistics and best-split search for reference-mode trees.

Composite log-rank (``splitrule="logrankCR"`` in the randomForestSRC
vocabulary) is implemented as a single pooled standardized statistic:

    L = (sum_k num_k)^2 / sum_k var_k

where num_k and var_k are the cause-specific log-rank numerator and
variance contributions.  This matches rfSRC 3.6.1 `logRankCR`
(SURV_CR_LAU) for the cause-combination rule — signed numerators pool
across causes before squaring, so cancellation across causes reshapes the
argmax.  See `docs/superpowers/specs/2026-04-18-p2.5-logrankcr-derivation.md`.

The per-cause numerator and variance use the Lau-inclusive at-risk
convention: for cause k at time m, the risk set adds back all subjects
who had a competing-cause (r != k) event strictly before m.  This is
the subdistribution-style risk set studied by Gray (1988), Fine-Gray
(1999), and Lau et al. (2009).  The variance-accumulation guard
remains on the standard parent at-risk count ``nP(m) >= 2``.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from crforest._estimators import reverse_cumsum


@dataclass
class _TimeBinning:
    """Precomputed time-axis data reused across every candidate split in a node."""

    order: np.ndarray  # argsort(time) — indices into original arrays
    event_sorted: np.ndarray  # event codes sorted by time
    inverse: np.ndarray  # per-sample bin index in sorted order
    n_times: int  # number of unique times
    n_total: np.ndarray  # samples per bin, shape (n_times,)
    d_any: np.ndarray  # any-event count per bin, shape (n_times,)


def bin_times(time: np.ndarray, event: np.ndarray) -> _TimeBinning:
    """Build the per-node time binning used by every split-statistic call."""
    order = np.argsort(time, kind="stable")
    event_sorted = event[order]
    _, inverse = np.unique(time[order], return_inverse=True)
    n_times = int(inverse.max()) + 1 if len(inverse) else 0
    n_total = np.bincount(inverse, minlength=n_times).astype(np.float64)
    d_any = np.bincount(
        inverse,
        weights=(event_sorted > 0).astype(np.float64),
        minlength=n_times,
    )
    return _TimeBinning(order, event_sorted, inverse, n_times, n_total, d_any)


def _logrank_components(tb: _TimeBinning, left_mask: np.ndarray, cause: int) -> tuple[float, float]:
    """Cause-specific log-rank numerator and variance, pre-pooling.

    Uses the Lau-inclusive at-risk set: for cause k at time m,
    subjects who had a competing-cause event strictly before m are
    added back to the risk set.  Matches rfSRC's
    ``nodeParentInclusiveAtRisk`` / ``nodeLeftInclusiveAtRisk`` under
    splitrule="logrankCR" (SURV_CR_LAU).

    Returns ``(numerator, variance_sum)``.  Callers that want the
    per-cause chi-squared compute ``numerator ** 2 / variance_sum``;
    callers that want the pooled statistic sum these components across
    causes before forming the ratio.
    """
    if left_mask.all() or (~left_mask).all():
        return 0.0, 0.0

    left_sorted = left_mask[tb.order].astype(np.float64)
    is_cause = (tb.event_sorted == cause).astype(np.float64)
    is_any_event = (tb.event_sorted > 0).astype(np.float64)

    d = np.bincount(tb.inverse, weights=is_cause, minlength=tb.n_times)
    d_left = np.bincount(tb.inverse, weights=is_cause * left_sorted, minlength=tb.n_times)
    d_any_left = np.bincount(tb.inverse, weights=is_any_event * left_sorted, minlength=tb.n_times)
    n_left = np.bincount(tb.inverse, weights=left_sorted, minlength=tb.n_times)

    at_risk = reverse_cumsum(tb.n_total)
    at_risk_left = reverse_cumsum(n_left)

    # Lau-inclusive at-risk: add back competing-cause events that happened
    # strictly before time m.  cumsum(x) - x gives the strict-prefix sum.
    other = tb.d_any - d
    other_left = d_any_left - d_left
    at_risk_inc = at_risk + (np.cumsum(other) - other)
    at_risk_left_inc = at_risk_left + (np.cumsum(other_left) - other_left)

    # Numerator: no variance-style guard needed here; at-risk >= 1 is
    # guaranteed by construction of the time bins (every bin has at least
    # one observed subject), so the division is always defined.
    expected = d * at_risk_left_inc / at_risk_inc
    numerator = float((d_left - expected).sum())

    # Variance: guarded by standard parent at-risk >= 2.
    # With the Lau-inclusive values, nPinc >= nP >= 2 so (nPinc - 1) >= 1.
    valid = at_risk >= 2
    d_v = d[valid]
    arinc_v = at_risk_inc[valid]
    arlinc_v = at_risk_left_inc[valid]

    variance = (
        d_v * arlinc_v * (arinc_v - arlinc_v) * (arinc_v - d_v) / (arinc_v**2 * (arinc_v - 1))
    )
    variance_sum = float(variance.sum())

    return numerator, variance_sum


def _logrank_components_std(
    tb: _TimeBinning, left_mask: np.ndarray, cause: int
) -> tuple[float, float]:
    """Cause-specific log-rank numerator and variance using the STANDARD at-risk set.

    Unlike ``_logrank_components`` (Lau-inclusive), subjects with a
    competing-cause event at time *t* are **removed** from the risk set
    at times > t.  This is the convention used by rfSRC's
    ``splitrule="logrank"`` setting.

    Returns ``(numerator, variance_sum)``.
    """
    if left_mask.all() or (~left_mask).all():
        return 0.0, 0.0

    left_sorted = left_mask[tb.order].astype(np.float64)
    is_cause = (tb.event_sorted == cause).astype(np.float64)

    d = np.bincount(tb.inverse, weights=is_cause, minlength=tb.n_times)
    d_left = np.bincount(tb.inverse, weights=is_cause * left_sorted, minlength=tb.n_times)
    n_left = np.bincount(tb.inverse, weights=left_sorted, minlength=tb.n_times)

    at_risk = reverse_cumsum(tb.n_total)
    at_risk_left = reverse_cumsum(n_left)

    expected = d * at_risk_left / at_risk  # at_risk >= 1 by construction of time bins
    numerator = float((d_left - expected).sum())

    valid = at_risk >= 2
    d_v = d[valid]
    ar_v = at_risk[valid]
    arl_v = at_risk_left[valid]

    variance = d_v * arl_v * (ar_v - arl_v) * (ar_v - d_v) / (ar_v**2 * (ar_v - 1))
    variance_sum = float(variance.sum())

    return numerator, variance_sum


def cause_specific_log_rank_statistic(
    tb: _TimeBinning,
    left_mask: np.ndarray,
    cause: int = 1,
    weights: np.ndarray | None = None,
    n_causes: int | None = None,
) -> float:
    """Cause-specific log-rank statistic with optional cause weights.

    Matches rfSRC 3.6.1 ``splitrule="logrank"`` (SURV_LR):
    - standard at-risk (competing events remove subjects from the cause-k risk set);
    - single cause: ``num_k² / var_k`` for the given ``cause``;
    - weighted: ``(Σ_k w_k · num_k)² / (Σ_k w_k² · var_k)``.

    Parameters
    ----------
    tb : _TimeBinning
        Precomputed bin times.
    left_mask : np.ndarray
        Boolean left-child indicator.
    cause : int, default=1
        Used only when ``weights is None``.
    weights : np.ndarray or None, default=None
        Cause weight vector of length ``n_causes``. If given, ``cause`` is ignored.
    n_causes : int or None, default=None
        Required when ``weights`` is given. Total number of causes.
    """
    if left_mask.all() or (~left_mask).all():
        return 0.0
    if weights is None:
        num, var = _logrank_components_std(tb, left_mask, cause=cause)
        if var < 1e-12:
            return 0.0
        return float(num * num / var)
    if n_causes is None:
        raise ValueError("n_causes is required when weights is given")
    if len(weights) != n_causes:
        raise ValueError(f"len(weights)={len(weights)} must equal n_causes={n_causes}")
    num_sum = 0.0
    var_sum = 0.0
    for k in range(1, n_causes + 1):
        num_k, var_k = _logrank_components_std(tb, left_mask, cause=k)
        w_k = float(weights[k - 1])
        num_sum += w_k * num_k
        var_sum += w_k * w_k * var_k
    if var_sum < 1e-12:
        return 0.0
    return float(num_sum * num_sum / var_sum)


def log_rank_statistic_relabeled(tb: _TimeBinning, left_mask: np.ndarray, cause: int) -> float:
    """Cause-specific log-rank statistic for the given binary split.

    Events of the given ``cause`` are treated as events; all other codes
    (including competing causes and censoring) are treated as censored.
    """
    if left_mask.all() or (~left_mask).all():
        return 0.0
    numerator, variance_sum = _logrank_components(tb, left_mask, cause)
    if variance_sum < 1e-12:
        return 0.0
    return float(numerator**2 / variance_sum)


def composite_log_rank_statistic(tb: _TimeBinning, left_mask: np.ndarray, n_causes: int) -> float:
    """Composite log-rank statistic (pooled standardized across causes).

    ``L = (sum_k num_k)^2 / sum_k var_k`` where num_k and var_k are the
    cause-specific log-rank numerator and variance for cause k.  Matches
    rfSRC 3.6.1 `logRankCR` (SURV_CR_LAU) for the cause-combination rule;
    the signed numerators pool across causes before squaring, so
    cancellation across causes can reshape the argmax.
    """
    if left_mask.all() or (~left_mask).all():
        return 0.0
    num_sum = 0.0
    var_sum = 0.0
    for k in range(1, n_causes + 1):
        num, var = _logrank_components(tb, left_mask, cause=k)
        num_sum += num
        var_sum += var
    if var_sum < 1e-12:
        return 0.0
    return float(num_sum * num_sum / var_sum)


def find_best_split(
    X: np.ndarray,
    time: np.ndarray,
    event: np.ndarray,
    n_causes: int,
    min_samples_leaf: int,
    *,
    splitrule: str = "logrankCR",
    cause: int = 1,
    cause_weights: np.ndarray | None = None,
    nsplit: int = 0,
    rng: np.random.RandomState | None = None,
) -> tuple[int, float, float]:
    """Search for the best split on X.

    ``splitrule="logrankCR"`` uses the pooled composite log-rank
    (Lau-inclusive at-risk). ``splitrule="logrank"`` uses the
    cause-specific log-rank (standard at-risk), with optional weight
    vector across causes via ``cause_weights``.

    ``nsplit=0`` (default) exhaustively evaluates every midpoint between
    consecutive sorted unique values of each feature. ``nsplit > 0``
    draws ``nsplit`` candidate split points uniformly without replacement
    from the midpoints between consecutive sorted unique values of each
    feature (rfSRC-style sampling-without-replacement at the node, up to
    a midpoint-vs-observed-value convention). Requires ``rng`` when
    ``nsplit > 0``.

    Returns
    -------
    (feature, threshold, statistic) — feature is -1 if no valid split exists;
    threshold is the midpoint between consecutive unique values; statistic
    is the split criterion value at the chosen split.

    Tie-breaking: lower feature index wins, then lower threshold.
    """
    if splitrule not in ("logrankCR", "logrank"):
        raise ValueError(f"splitrule must be 'logrankCR' or 'logrank'; got {splitrule!r}")
    if nsplit < 0:
        raise ValueError(f"nsplit must be >= 0; got {nsplit}")
    if nsplit > 0 and rng is None:
        raise ValueError("nsplit > 0 requires an rng")

    n_samples, n_features = X.shape
    best_stat = 0.0
    best_feature = -1
    best_threshold = -1.0

    tb = bin_times(time, event)

    def score(left_mask: np.ndarray) -> float:
        if splitrule == "logrankCR":
            return composite_log_rank_statistic(tb, left_mask, n_causes)
        return cause_specific_log_rank_statistic(
            tb, left_mask, cause=cause, weights=cause_weights, n_causes=n_causes
        )

    for feat in range(n_features):
        col = X[:, feat]
        unique_vals = np.unique(col)
        if len(unique_vals) < 2:
            continue
        midpoints_all = (unique_vals[:-1] + unique_vals[1:]) / 2.0
        if nsplit > 0:
            # rfSRC semantics: sample `nsplit` distinct observed unique values
            # without replacement, excluding the max. We sample from midpoints
            # between consecutive sorted uniques — there are (n_unique - 1) of
            # them, same cardinality as rfSRC's pool (observed values excluding
            # max), and the split boundaries are semantically equivalent.
            k = min(nsplit, len(midpoints_all))
            candidates = rng.choice(midpoints_all, size=k, replace=False)
        else:
            candidates = midpoints_all

        for t in candidates:
            left_mask = col <= t
            n_left = int(left_mask.sum())
            n_right = n_samples - n_left
            if n_left < min_samples_leaf or n_right < min_samples_leaf:
                continue
            stat = score(left_mask)
            if stat > best_stat:
                best_stat = stat
                best_feature = feat
                best_threshold = float(t)

    return best_feature, best_threshold, best_stat
