"""sklearn drop-in compatibility helpers.

scikit-learn's ``fit(X, y)`` and ``score(X, y)`` signatures expect a
single ``y`` array, but competing-risks survival data has two outcomes
per subject (``time`` and ``event``). The ecosystem convention,
established by scikit-survival, is to pack both into a structured numpy
array.

This module exposes :class:`Surv` mirroring scikit-survival's API so
existing sksurv users can pipe their ``y`` straight into comprisk, plus
:func:`is_structured_survival_y` and :func:`unpack_structured_y` used by
``CompetingRiskForest.fit`` / ``score`` to dispatch between the
three-positional-argument legacy form and the sklearn-friendly
``fit(X, y)`` form.
"""

from __future__ import annotations

import numpy as np


class Surv:
    """Build the structured survival ``y`` array sklearn workflows want.

    Mirrors :class:`sksurv.util.Surv`. The returned array has named
    fields ``event`` and ``time`` and shape ``(n,)`` — sliceable by
    sklearn cross-validation utilities, picklable, copy-equivalent to
    a pair of 1-D arrays.

    Examples
    --------
    >>> import numpy as np
    >>> from comprisk import Surv
    >>> y = Surv.from_arrays(event=[0, 1, 2, 0], time=[1.0, 2.0, 3.0, 0.5])
    >>> y.dtype.names
    ('event', 'time')
    """

    @staticmethod
    def from_arrays(
        event,
        time,
        *,
        name_event: str = "event",
        name_time: str = "time",
    ) -> np.ndarray:
        """Pack ``event`` and ``time`` 1-D arrays into a structured array.

        Parameters
        ----------
        event : array-like, shape (n,)
            Event indicator. ``0`` for censored, ``1..K`` for cause-of-event
            in competing-risks data.
        time : array-like, shape (n,)
            Observed time-to-event or censoring.
        name_event, name_time : str
            Field names in the structured array. Defaults match sksurv.

        Returns
        -------
        y : structured ndarray, shape (n,)
            Fields ``(name_event, name_time)``, dtypes ``int64`` and
            ``float64``.
        """
        event_arr = np.asarray(event)
        time_arr = np.asarray(time, dtype=np.float64)
        if event_arr.ndim != 1 or time_arr.ndim != 1:
            raise ValueError("event and time must be 1-D arrays")
        if len(event_arr) != len(time_arr):
            raise ValueError(
                f"event and time must be the same length; got {len(event_arr)} and {len(time_arr)}"
            )
        if not (
            np.issubdtype(event_arr.dtype, np.integer) or np.issubdtype(event_arr.dtype, np.bool_)
        ):
            raise TypeError(f"event must be integer-typed (or bool); got dtype {event_arr.dtype}")
        out = np.zeros(
            len(time_arr),
            dtype=[(name_event, np.int64), (name_time, np.float64)],
        )
        out[name_event] = event_arr  # numpy auto-casts bool/int to int64
        out[name_time] = time_arr
        return out


def is_structured_survival_y(y) -> bool:
    """True if ``y`` is a structured array carrying ``time`` and ``event`` fields."""
    if not hasattr(y, "dtype") or y.dtype.names is None:
        return False
    names = y.dtype.names
    return "time" in names and "event" in names


def unpack_structured_y(y) -> tuple[np.ndarray, np.ndarray]:
    """Extract ``(time, event)`` from a structured survival ``y`` array.

    Field order in the structured dtype is irrelevant; we look up by name.
    """
    if not is_structured_survival_y(y):
        raise TypeError(
            "y must be a structured array with 'time' and 'event' fields. "
            "Build it via comprisk.Surv.from_arrays(event=..., time=...) or "
            "use the legacy three-argument form fit(X, time, event)."
        )
    return np.asarray(y["time"]), np.asarray(y["event"])
