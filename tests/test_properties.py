"""Property-based tests over the full fit/predict pipeline.

CIF and CHF invariants are parametrized over both modes: they must hold regardless of
histogram binning, compact leaf storage, or whether the CHF leaf table is built eagerly
(reference mode) or materialised lazily from persisted raw counts (default flat-tree).
"""

import numpy as np
import pytest
from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st

from crforest.forest import CompetingRiskForest


@st.composite
def cr_dataset(draw, min_samples=30, max_samples=80, max_features=6, max_causes=3):
    n = draw(st.integers(min_samples, max_samples))
    p = draw(st.integers(2, max_features))
    n_causes = draw(st.integers(2, max_causes))
    seed = draw(st.integers(0, 10_000))

    rng = np.random.default_rng(seed)
    X = rng.uniform(size=(n, p))
    time = rng.uniform(0.1, 10.0, n)
    event = rng.integers(0, n_causes + 1, n)
    for k in range(1, n_causes + 1):
        if not np.any(event == k):
            event[k - 1] = k
    return X, time, event, n_causes


@pytest.mark.parametrize("mode", ["default", "reference"])
@pytest.mark.parametrize("nsplit", [0, 10])
@settings(max_examples=15, deadline=None, suppress_health_check=[HealthCheck.too_slow])
@given(data=cr_dataset())
def test_cif_monotone_nondecreasing(data, mode, nsplit):
    X, time, event, _ = data
    f = CompetingRiskForest(n_estimators=5, mode=mode, nsplit=nsplit, random_state=0).fit(
        X, time, event
    )
    cif = f.predict_cif(X)
    assert np.all(np.diff(cif, axis=2) >= -1e-9)


@pytest.mark.parametrize("mode", ["default", "reference"])
@pytest.mark.parametrize("nsplit", [0, 10])
@settings(max_examples=15, deadline=None, suppress_health_check=[HealthCheck.too_slow])
@given(data=cr_dataset())
def test_cif_in_unit_interval(data, mode, nsplit):
    X, time, event, _ = data
    f = CompetingRiskForest(n_estimators=5, mode=mode, nsplit=nsplit, random_state=0).fit(
        X, time, event
    )
    cif = f.predict_cif(X)
    assert np.all(cif >= 0.0)
    assert np.all(cif <= 1.0 + 1e-9)


@pytest.mark.parametrize("mode", ["default", "reference"])
@pytest.mark.parametrize("nsplit", [0, 10])
@settings(max_examples=15, deadline=None, suppress_health_check=[HealthCheck.too_slow])
@given(data=cr_dataset())
def test_cif_sum_across_causes_bounded_by_one(data, mode, nsplit):
    X, time, event, _ = data
    f = CompetingRiskForest(n_estimators=5, mode=mode, nsplit=nsplit, random_state=0).fit(
        X, time, event
    )
    cif = f.predict_cif(X)
    total_final = cif.sum(axis=1)[:, -1]
    assert np.all(total_final <= 1.0 + 1e-9)
    assert np.all(total_final >= -1e-9)


@pytest.mark.parametrize("mode", ["default", "reference"])
@pytest.mark.parametrize("nsplit", [0, 10])
@settings(max_examples=15, deadline=None, suppress_health_check=[HealthCheck.too_slow])
@given(data=cr_dataset())
def test_chf_monotone_nondecreasing(data, mode, nsplit):
    X, time, event, _ = data
    f = CompetingRiskForest(n_estimators=5, mode=mode, nsplit=nsplit, random_state=0).fit(
        X, time, event
    )
    chf = f.predict_chf(X)
    assert np.all(np.diff(chf, axis=2) >= -1e-9)


@pytest.mark.parametrize("mode", ["default", "reference"])
@pytest.mark.parametrize("nsplit", [0, 10])
@settings(max_examples=15, deadline=None, suppress_health_check=[HealthCheck.too_slow])
@given(data=cr_dataset())
def test_chf_non_negative(data, mode, nsplit):
    X, time, event, _ = data
    f = CompetingRiskForest(n_estimators=5, mode=mode, nsplit=nsplit, random_state=0).fit(
        X, time, event
    )
    chf = f.predict_chf(X)
    assert np.all(chf >= -1e-12)
