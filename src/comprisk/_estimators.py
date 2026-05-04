"""Survival estimators used by reference-mode trees."""

from __future__ import annotations

import numpy as np


def reverse_cumsum(arr: np.ndarray) -> np.ndarray:
    """Right-to-left cumulative sum: ``arr[i] = sum(arr[i:])``.

    Used to compute at-risk counts from per-time-bucket sample counts.
    """
    return np.cumsum(arr[::-1])[::-1].copy()


def _hazard(counts: np.ndarray, at_risk: np.ndarray) -> np.ndarray:
    """Discrete hazard ``counts / at_risk``, zero where ``at_risk == 0``."""
    return np.where(at_risk > 0, counts / np.where(at_risk > 0, at_risk, 1.0), 0.0)


def kaplan_meier_survival(at_risk: np.ndarray, d_any: np.ndarray) -> np.ndarray:
    """Kaplan-Meier survival curve evaluated just before each time bucket.

    Uses the left-continuous convention: ``surv[i]`` is the survival
    probability just before the i-th unique time, so ``surv[0] = 1``.
    """
    n_times = len(at_risk)
    surv = np.ones(n_times)
    if n_times > 1:
        surv[1:] = np.cumprod((1.0 - _hazard(d_any, at_risk))[:-1])
    return surv


def _time_bin_counts(
    time: np.ndarray, event: np.ndarray, unique_times: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return (at_risk, d_any, t_indices) on the ``unique_times`` grid."""
    t_indices = np.searchsorted(unique_times, time)
    n_times = len(unique_times)
    n_at = np.bincount(t_indices, minlength=n_times).astype(np.float64)
    d_any = np.bincount(
        t_indices,
        weights=(event > 0).astype(np.float64),
        minlength=n_times,
    )
    return reverse_cumsum(n_at), d_any, t_indices


def nelson_aalen(time: np.ndarray, event: np.ndarray, unique_times: np.ndarray) -> np.ndarray:
    """Nelson-Aalen cumulative hazard function on ``unique_times``.

    Pre-condition: every value in ``time`` must appear in ``unique_times``.
    The ``np.searchsorted(..., side="left")`` call assumes exact membership;
    calling with a disjoint grid produces silent index errors.
    """
    at_risk, d_any, _ = _time_bin_counts(time, event, unique_times)
    return np.cumsum(_hazard(d_any, at_risk))


def aalen_johansen_from_counts(
    event_counts: np.ndarray,
    at_risk: np.ndarray,
    n_causes: int,
) -> np.ndarray:
    """Aalen-Johansen CIF from pre-aggregated counts on a shared time grid.

    Parameters
    ----------
    event_counts : ndarray, shape (n_causes, n_times)
        ``event_counts[k, t]`` = count of samples with ``event == k+1`` at
        time index ``t``. Any numeric dtype; cast to float64 internally.
    at_risk : ndarray, shape (n_times,)
        Number of samples at risk at each time index. Expected to be the
        reverse-cumulative-sum of per-bucket sample counts.
    n_causes : int

    Returns
    -------
    cif : ndarray, shape (n_causes, n_times), float64
    """
    ar = np.asarray(at_risk, dtype=np.float64)
    ec = np.asarray(event_counts, dtype=np.float64)
    d_any = ec.sum(axis=0)
    surv = kaplan_meier_survival(ar, d_any)
    n_times = ar.shape[0]
    cif = np.zeros((n_causes, n_times), dtype=np.float64)
    for k in range(n_causes):
        cif[k] = np.cumsum(surv * _hazard(ec[k], ar))
    return cif


def aalen_johansen_from_counts_batched(
    event_counts: np.ndarray,
    at_risk: np.ndarray,
    n_causes: int,
) -> np.ndarray:
    """Vectorized Aalen-Johansen CIF over a leaf-axis batch.

    Bulk port of ``aalen_johansen_from_counts`` over a leading leaf axis.
    Bit-identical to a per-leaf loop modulo float64 op order: hazard = d/ar
    (zero where ar==0); KM survival is left-continuous (surv[..., 0]=1);
    per-cause CIF is cumsum_t(surv * d_k / ar).

    Replaces a Python-level loop over leaves; the per-call allocator and
    small-array cumsum overhead grew super-linearly with ntree under loky
    worker contention (profile_fit cliff diagnosis 2026-04-29).

    Parameters
    ----------
    event_counts : ndarray, shape (n_leaves, n_causes, n_times)
    at_risk      : ndarray, shape (n_leaves, n_times)
    n_causes     : int (kept for signature parity; inferred from shape)

    Returns
    -------
    cif : ndarray, shape (n_leaves, n_causes, n_times), float64
    """
    ar = np.asarray(at_risk, dtype=np.float64)  # (L, T)
    ec = np.asarray(event_counts, dtype=np.float64)  # (L, K, T)
    d_any = ec.sum(axis=1)  # (L, T)
    safe_ar = np.where(ar > 0, ar, 1.0)  # (L, T)
    h_any = np.where(ar > 0, d_any / safe_ar, 0.0)  # (L, T)

    n_times = ar.shape[1]
    surv = np.ones_like(ar)
    if n_times > 1:
        surv[:, 1:] = np.cumprod(1.0 - h_any[:, :-1], axis=1)

    h_k = np.where(ar[:, None, :] > 0, ec / safe_ar[:, None, :], 0.0)  # (L, K, T)
    cif = np.cumsum(surv[:, None, :] * h_k, axis=2)  # (L, K, T)
    return cif


def nelson_aalen_from_counts(
    event_counts: np.ndarray,
    at_risk: np.ndarray,
    n_causes: int,
) -> np.ndarray:
    """Nelson-Aalen cause-specific CHF from pre-aggregated counts.

    Parameters
    ----------
    event_counts : ndarray, shape (n_causes, n_times)
        ``event_counts[k, t]`` = count of cause-``k+1`` events at time
        index ``t``. Any numeric dtype; cast to float64 internally.
    at_risk : ndarray, shape (n_times,)
        Number of samples at risk at each time index (reverse cumulative
        sum of per-bucket sample counts).
    n_causes : int

    Returns
    -------
    chf : ndarray, shape (n_causes, n_times), float64
        Cause-specific cumulative hazard. Right-continuous step function:
        ``chf[k, t]`` includes the hazard contribution from events at
        time index ``t``.
    """
    ar = np.asarray(at_risk, dtype=np.float64)
    ec = np.asarray(event_counts, dtype=np.float64)
    chf = np.zeros((n_causes, ar.shape[0]), dtype=np.float64)
    for k in range(n_causes):
        chf[k] = np.cumsum(_hazard(ec[k], ar))
    return chf


def leaf_counts_from_time_event(
    time: np.ndarray,
    event: np.ndarray,
    unique_times: np.ndarray,
    n_causes: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Per-cause event counts and at-risk counts on ``unique_times``.

    Pre-condition: every value in ``time`` must appear in ``unique_times``.

    Returns
    -------
    event_counts : ndarray, shape (n_causes, n_times), float64
    at_risk      : ndarray, shape (n_times,), float64
    """
    at_risk, _, t_indices = _time_bin_counts(time, event, unique_times)
    n_times = len(unique_times)
    event_counts = np.zeros((n_causes, n_times), dtype=np.float64)
    for k in range(n_causes):
        event_counts[k] = np.bincount(
            t_indices,
            weights=(event == (k + 1)).astype(np.float64),
            minlength=n_times,
        )
    return event_counts, at_risk


def nelson_aalen_cs(
    time: np.ndarray,
    event: np.ndarray,
    unique_times: np.ndarray,
    n_causes: int,
) -> np.ndarray:
    """Cause-specific Nelson-Aalen CHF for ``n_causes`` competing risks.

    Mirrors ``aalen_johansen`` in signature and pre-conditions — every
    value in ``time`` must appear in ``unique_times``.

    Returns
    -------
    chf : ndarray, shape (n_causes, len(unique_times)), float64
    """
    event_counts, at_risk = leaf_counts_from_time_event(time, event, unique_times, n_causes)
    return nelson_aalen_from_counts(event_counts, at_risk, n_causes)


def aalen_johansen(
    time: np.ndarray,
    event: np.ndarray,
    unique_times: np.ndarray,
    n_causes: int,
) -> np.ndarray:
    """Aalen-Johansen cumulative incidence function for ``n_causes`` risks.

    Pre-condition: every value in ``time`` must appear in ``unique_times``.

    Returns
    -------
    cif : ndarray, shape (n_causes, len(unique_times))
    """
    event_counts, at_risk = leaf_counts_from_time_event(time, event, unique_times, n_causes)
    return aalen_johansen_from_counts(event_counts, at_risk, n_causes)
