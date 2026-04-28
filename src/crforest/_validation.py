"""Input validation for CompetingRiskForest.fit."""

from __future__ import annotations

import numpy as np


def check_inputs(X, time, event):
    """Validate and canonicalize (X, time, event) for a CR forest fit.

    Parameters
    ----------
    X : array-like, shape (n_samples, n_features)
        Feature matrix. Must be 2-D and numeric with no NaN/inf.
    time : array-like, shape (n_samples,)
        Observed times. Must be 1-D, finite, non-negative.
    event : array-like, shape (n_samples,)
        Event codes. Must be 1-D integer in {0, 1, ..., n_causes}
        where 0 = censored. Causes must form the contiguous set
        {1, ..., n_causes}.

    Returns
    -------
    X : ndarray, float64, shape (n_samples, n_features)
    time : ndarray, float64, shape (n_samples,)
    event : ndarray, int64, shape (n_samples,)
    n_causes : int
    """
    X = np.asarray(X, dtype=np.float64)
    time = np.asarray(time, dtype=np.float64)
    event_raw = np.asarray(event)

    if X.ndim != 2:
        raise ValueError(f"X must be 2-D; got ndim={X.ndim}")
    if time.ndim != 1:
        raise ValueError(f"time must be 1-D; got ndim={time.ndim}")
    if event_raw.ndim != 1:
        raise ValueError(f"event must be 1-D; got ndim={event_raw.ndim}")

    n = X.shape[0]
    if n == 0:
        raise ValueError("X must have at least one row")
    if time.shape[0] != n or event_raw.shape[0] != n:
        raise ValueError(
            f"length mismatch: X has {n} rows, "
            f"time has {time.shape[0]}, event has {event_raw.shape[0]}"
        )

    if not np.all(np.isfinite(X)):
        raise ValueError("X contains non-finite values (NaN or inf)")
    if not np.all(np.isfinite(time)):
        raise ValueError("time contains non-finite values (NaN or inf)")
    if np.any(time < 0):
        raise ValueError("time values must be non-negative")

    if np.issubdtype(event_raw.dtype, np.floating):
        if not np.all(event_raw == np.floor(event_raw)):
            raise ValueError("event values must be integer-valued")
    elif not np.issubdtype(event_raw.dtype, np.integer):
        raise ValueError(f"event must be integer-typed; got dtype={event_raw.dtype}")
    event = event_raw.astype(np.int64)
    if np.any(event < 0):
        raise ValueError("event codes must be non-negative integers (0=censored)")

    causes_present = {int(c) for c in event[event > 0]}
    if not causes_present:
        raise ValueError("event must contain at least one event (> 0)")
    n_causes = max(causes_present)
    expected = set(range(1, n_causes + 1))
    if causes_present != expected:
        raise ValueError(
            "event codes must be contiguous from 1 to n_causes; "
            f"got causes {sorted(causes_present)}, expected {sorted(expected)}"
        )

    return X, time, event, n_causes
