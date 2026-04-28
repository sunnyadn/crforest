"""sklearn drop-in compatibility surface tests.

Verifies CompetingRiskForest is a true sklearn-compatible estimator:

* ``Surv.from_arrays(event, time)`` builds the structured y the same way
  scikit-survival does, so users can swap libraries without rewiring data.
* ``fit(X, y)`` and ``score(X, y)`` accept the structured y form, equivalent
  to the legacy three-argument ``fit(X, time, event)`` / ``score(X, time, event)``.
* ``predict(X)`` is a sklearn alias for ``predict_risk(X, cause=1)``.
* ``cross_val_score`` and ``clone`` work without a wrapper.
"""

from __future__ import annotations

import numpy as np
import pytest
from sklearn.base import clone
from sklearn.exceptions import NotFittedError
from sklearn.model_selection import KFold, cross_val_score

from crforest import CompetingRiskForest, Surv


def _toy_cr(n: int = 200, p: int = 5, seed: int = 0):
    rng = np.random.default_rng(seed)
    X = rng.standard_normal((n, p))
    time = rng.exponential(1.0, n) + 0.1
    event = rng.choice([0, 1, 2], size=n, p=[0.4, 0.3, 0.3]).astype(np.int8)
    return X, time, event


# ---------------------------------------------------------------------------
# Surv.from_arrays
# ---------------------------------------------------------------------------


def test_surv_from_arrays_returns_structured():
    _, time, event = _toy_cr()
    y = Surv.from_arrays(event=event, time=time)
    assert y.dtype.names == ("event", "time")
    assert y.shape == (len(time),)
    np.testing.assert_array_equal(y["time"], time)
    np.testing.assert_array_equal(y["event"], event)


def test_surv_from_arrays_accepts_lists():
    y = Surv.from_arrays(event=[0, 1, 2, 0], time=[1.0, 2.0, 3.0, 0.5])
    np.testing.assert_array_equal(y["event"], [0, 1, 2, 0])
    np.testing.assert_array_equal(y["time"], [1.0, 2.0, 3.0, 0.5])


def test_surv_from_arrays_length_mismatch():
    with pytest.raises(ValueError, match="same length"):
        Surv.from_arrays(event=[0, 1], time=[1.0, 2.0, 3.0])


# ---------------------------------------------------------------------------
# fit / score with structured y
# ---------------------------------------------------------------------------


def test_fit_with_structured_y_equivalent_to_three_arg():
    X, time, event = _toy_cr()
    y = Surv.from_arrays(event=event, time=time)
    f1 = CompetingRiskForest(n_estimators=5, random_state=0).fit(X, time, event)
    f2 = CompetingRiskForest(n_estimators=5, random_state=0).fit(X, y)
    np.testing.assert_array_equal(f1.predict_cif(X[:3]), f2.predict_cif(X[:3]))


def test_fit_accepts_reverse_field_order():
    """Match sksurv's (event, time) order AND any user-built (time, event)."""
    X, time, event = _toy_cr()
    # Build structured y with time-first field order (codebase internal convention).
    y_time_first = np.zeros(len(time), dtype=[("time", np.float64), ("event", np.int8)])
    y_time_first["time"] = time
    y_time_first["event"] = event
    f = CompetingRiskForest(n_estimators=5, random_state=0).fit(X, y_time_first)
    f_ref = CompetingRiskForest(n_estimators=5, random_state=0).fit(X, time, event)
    np.testing.assert_array_equal(f.predict_cif(X[:3]), f_ref.predict_cif(X[:3]))


def test_fit_rejects_non_structured_y_when_event_omitted():
    X, time, _ = _toy_cr()
    f = CompetingRiskForest(n_estimators=3)
    with pytest.raises(TypeError, match="structured array"):
        f.fit(X, time)  # bare time array, no event -> ambiguous


def test_score_with_structured_y_equivalent():
    X, time, event = _toy_cr()
    y = Surv.from_arrays(event=event, time=time)
    f = CompetingRiskForest(n_estimators=5, random_state=0).fit(X, time, event)
    assert f.score(X, time, event, cause=1) == f.score(X, y, cause=1)


# ---------------------------------------------------------------------------
# predict() alias
# ---------------------------------------------------------------------------


def test_predict_alias_matches_predict_risk_cause1():
    X, time, event = _toy_cr()
    f = CompetingRiskForest(n_estimators=5, random_state=0).fit(X, time, event)
    np.testing.assert_array_equal(f.predict(X[:5]), f.predict_risk(X[:5], cause=1))


# ---------------------------------------------------------------------------
# clone + cross_val_score
# ---------------------------------------------------------------------------


def test_clone_preserves_constructor_params():
    f = CompetingRiskForest(n_estimators=42, max_depth=5, random_state=7)
    g = clone(f)
    assert g.n_estimators == 42
    assert g.max_depth == 5
    assert g.random_state == 7
    with pytest.raises(NotFittedError):
        g.predict_cif(np.zeros((3, 5)))


def test_cross_val_score_with_kfold():
    X, time, event = _toy_cr(n=120, p=4)
    y = Surv.from_arrays(event=event, time=time)
    f = CompetingRiskForest(n_estimators=8, random_state=0, n_jobs=1)
    cv = KFold(n_splits=3, shuffle=True, random_state=42)
    scores = cross_val_score(f, X, y, cv=cv, n_jobs=1)
    assert scores.shape == (3,)
    assert all(0.0 <= s <= 1.0 for s in scores)
