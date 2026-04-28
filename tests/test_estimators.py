"""Tests for survival estimators (KM, Nelson-Aalen, Aalen-Johansen)."""

import numpy as np

from crforest._estimators import (
    aalen_johansen,
    kaplan_meier_survival,
    nelson_aalen,
    reverse_cumsum,
)


def test_reverse_cumsum_simple():
    result = reverse_cumsum(np.array([1.0, 2.0, 3.0, 4.0]))
    assert np.allclose(result, [10.0, 9.0, 7.0, 4.0])


def test_reverse_cumsum_empty():
    result = reverse_cumsum(np.array([], dtype=np.float64))
    assert result.shape == (0,)


def test_kaplan_meier_all_survive():
    # No events, all censored: survival stays at 1.0
    at_risk = np.array([5.0, 4.0, 3.0])
    d_any = np.array([0.0, 0.0, 0.0])
    surv = kaplan_meier_survival(at_risk, d_any)
    assert np.allclose(surv, [1.0, 1.0, 1.0])


def test_kaplan_meier_textbook():
    # 5 subjects, event at time 1, censor at 2, event at 3, event at 4, censor at 5
    at_risk = np.array([5.0, 4.0, 3.0, 2.0, 1.0])
    d_any = np.array([1.0, 0.0, 1.0, 1.0, 0.0])
    surv = kaplan_meier_survival(at_risk, d_any)
    expected = np.array([1.0, 0.8, 0.8, 8 / 15, 4 / 15])
    assert np.allclose(surv, expected, atol=1e-12)


def test_nelson_aalen_toy():
    time = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    event = np.array([1, 0, 1, 1, 0])
    unique_times = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    chf = nelson_aalen(time, event, unique_times)
    # hazards: [1/5, 0, 1/3, 1/2, 0]; cumsum
    expected = np.cumsum([0.2, 0.0, 1 / 3, 0.5, 0.0])
    assert np.allclose(chf, expected, atol=1e-12)


def test_aalen_johansen_hand_computed():
    # Competing risks toy dataset from plan header
    time = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    event = np.array([1, 0, 2, 1, 0])
    unique_times = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    cif = aalen_johansen(time, event, unique_times, n_causes=2)

    expected_cause1 = np.array([0.2, 0.2, 0.2, 7 / 15, 7 / 15])
    expected_cause2 = np.array([0.0, 0.0, 4 / 15, 4 / 15, 4 / 15])

    assert cif.shape == (2, 5)
    assert np.allclose(cif[0], expected_cause1, atol=1e-12)
    assert np.allclose(cif[1], expected_cause2, atol=1e-12)


def test_aalen_johansen_monotone_nondecreasing():
    rng = np.random.default_rng(0)
    n = 50
    time = rng.uniform(0.1, 10.0, n)
    event = rng.integers(0, 3, n)  # {0, 1, 2}
    if not np.any(event > 0):
        event[0] = 1
    unique_times = np.sort(np.unique(time))
    cif = aalen_johansen(time, event, unique_times, n_causes=2)
    assert cif.shape == (2, len(unique_times))
    assert np.all(np.diff(cif, axis=1) >= -1e-12)


def test_aalen_johansen_sum_bounded_by_one():
    rng = np.random.default_rng(1)
    n = 100
    time = rng.uniform(0.1, 10.0, n)
    event = rng.integers(0, 4, n)  # {0, 1, 2, 3}
    if not np.any(event > 0):
        event[0] = 1
    unique_times = np.sort(np.unique(time))
    cif = aalen_johansen(time, event, unique_times, n_causes=3)
    total = cif.sum(axis=0)
    assert np.all(total <= 1.0 + 1e-9)
    assert np.all(total >= -1e-12)


def test_aalen_johansen_no_events_for_cause_returns_zero():
    time = np.array([1.0, 2.0, 3.0])
    event = np.array([1, 1, 0])  # no cause-2
    unique_times = np.sort(np.unique(time))
    cif = aalen_johansen(time, event, unique_times, n_causes=2)
    assert np.allclose(cif[1], 0.0)


def test_nelson_aalen_from_counts_hand_computed():
    from crforest._estimators import nelson_aalen_from_counts

    # 5 subjects, times [0, 1, 1, 2, 2], events [1, 2, 1, 0, 1]
    # at_risk (reverse-cumsum of n_at per time): t0: 5, t1: 4, t2: 2
    event_counts = np.array(
        [
            [1, 1, 1],  # cause 1
            [0, 1, 0],  # cause 2
        ],
        dtype=np.uint32,
    )
    at_risk = np.array([5, 4, 2], dtype=np.uint32)
    chf = nelson_aalen_from_counts(event_counts, at_risk, n_causes=2)

    # cause 1 hazards: 1/5, 1/4, 1/2 -> cumsum
    expected_cause1 = np.cumsum([1 / 5, 1 / 4, 1 / 2])
    # cause 2 hazards: 0/5, 1/4, 0/2 -> cumsum
    expected_cause2 = np.cumsum([0.0, 1 / 4, 0.0])

    assert chf.shape == (2, 3)
    assert chf.dtype == np.float64
    assert np.allclose(chf[0], expected_cause1, atol=1e-12)
    assert np.allclose(chf[1], expected_cause2, atol=1e-12)


def test_nelson_aalen_from_counts_zero_at_risk_gives_zero_hazard():
    from crforest._estimators import nelson_aalen_from_counts

    event_counts = np.array([[0, 1, 0]], dtype=np.uint32)
    at_risk = np.array([2, 1, 0], dtype=np.uint32)
    chf = nelson_aalen_from_counts(event_counts, at_risk, n_causes=1)
    # Last bin has at_risk=0; hazard is 0 there (no divide-by-zero)
    assert np.isfinite(chf).all()
    assert chf[0, 2] == chf[0, 1]


def test_nelson_aalen_cs_matches_sum_of_all_cause():
    """Sum of cause-specific CHFs equals all-cause NA (standard at-risk)."""
    from crforest._estimators import nelson_aalen, nelson_aalen_cs

    rng = np.random.default_rng(7)
    n = 80
    time = rng.uniform(0.1, 10.0, n)
    event = rng.integers(0, 3, n)  # {0, 1, 2}
    if not np.any(event > 0):
        event[0] = 1
    unique_times = np.sort(np.unique(time))

    chf_cs = nelson_aalen_cs(time, event, unique_times, n_causes=2)
    chf_all = nelson_aalen(time, event, unique_times)

    assert chf_cs.shape == (2, len(unique_times))
    assert np.allclose(chf_cs.sum(axis=0), chf_all, atol=1e-12)


def test_nelson_aalen_cs_monotone_nondecreasing():
    from crforest._estimators import nelson_aalen_cs

    rng = np.random.default_rng(11)
    n = 60
    time = rng.uniform(0.1, 10.0, n)
    event = rng.integers(0, 3, n)
    if not np.any(event > 0):
        event[0] = 1
    unique_times = np.sort(np.unique(time))
    chf = nelson_aalen_cs(time, event, unique_times, n_causes=2)
    assert np.all(np.diff(chf, axis=1) >= -1e-12)
