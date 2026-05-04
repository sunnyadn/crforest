"""Tests for survival estimators (KM, Nelson-Aalen, Aalen-Johansen)."""

import numpy as np

from comprisk._estimators import (
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
    from comprisk._estimators import nelson_aalen_from_counts

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
    from comprisk._estimators import nelson_aalen_from_counts

    event_counts = np.array([[0, 1, 0]], dtype=np.uint32)
    at_risk = np.array([2, 1, 0], dtype=np.uint32)
    chf = nelson_aalen_from_counts(event_counts, at_risk, n_causes=1)
    # Last bin has at_risk=0; hazard is 0 there (no divide-by-zero)
    assert np.isfinite(chf).all()
    assert chf[0, 2] == chf[0, 1]


def test_nelson_aalen_cs_matches_sum_of_all_cause():
    """Sum of cause-specific CHFs equals all-cause NA (standard at-risk)."""
    from comprisk._estimators import nelson_aalen, nelson_aalen_cs

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


def test_aalen_johansen_from_counts_batched_matches_per_leaf():
    """Vectorized batched AJ matches per-leaf helper bit-identical (1e-12)
    on a multi-leaf fixture with mixed at-risk profiles, parallel to the
    GPU equivalence test in test_gpu_smoke.py."""
    from comprisk._estimators import (
        aalen_johansen_from_counts,
        aalen_johansen_from_counts_batched,
    )

    rng = np.random.default_rng(11)
    n_leaves, n_causes, n_time = 20, 3, 32

    raw_counts = rng.integers(0, 5, size=(n_leaves, n_time), dtype=np.uint32)
    at_risk = np.zeros((n_leaves, n_time), dtype=np.uint32)
    for i in range(n_leaves):
        running = np.uint32(0)
        for t in range(n_time - 1, -1, -1):
            running += raw_counts[i, t]
            at_risk[i, t] = running
    event_counts = rng.integers(0, 4, size=(n_leaves, n_causes, n_time), dtype=np.uint32)
    for i in range(n_leaves):
        for t in range(n_time):
            if int(event_counts[i, :, t].sum()) > at_risk[i, t]:
                event_counts[i, :, t] = 0

    expected = np.zeros((n_leaves, n_causes, n_time), dtype=np.float64)
    for i in range(n_leaves):
        expected[i] = aalen_johansen_from_counts(event_counts[i], at_risk[i], n_causes)
    got = aalen_johansen_from_counts_batched(event_counts, at_risk, n_causes)
    np.testing.assert_allclose(got, expected, rtol=1e-12, atol=1e-12)


def test_aalen_johansen_from_counts_batched_zero_at_risk_tail():
    """Batched AJ handles zero-at-risk tail bins without divide-by-zero
    or NaN, matching the per-leaf helper exactly."""
    from comprisk._estimators import (
        aalen_johansen_from_counts,
        aalen_johansen_from_counts_batched,
    )

    rng = np.random.default_rng(13)
    n_leaves, n_causes, n_time = 8, 2, 16
    event_counts = rng.integers(0, 3, size=(n_leaves, n_causes, n_time), dtype=np.uint32)
    at_risk = np.full((n_leaves, n_time), 5, dtype=np.uint32)
    at_risk[:, -3:] = 0  # zero tail

    expected = np.zeros((n_leaves, n_causes, n_time), dtype=np.float64)
    for i in range(n_leaves):
        expected[i] = aalen_johansen_from_counts(event_counts[i], at_risk[i], n_causes)
    got = aalen_johansen_from_counts_batched(event_counts, at_risk, n_causes)
    assert np.isfinite(got).all()
    np.testing.assert_allclose(got, expected, rtol=1e-12, atol=1e-12)


def test_nelson_aalen_cs_monotone_nondecreasing():
    from comprisk._estimators import nelson_aalen_cs

    rng = np.random.default_rng(11)
    n = 60
    time = rng.uniform(0.1, 10.0, n)
    event = rng.integers(0, 3, n)
    if not np.any(event > 0):
        event[0] = 1
    unique_times = np.sort(np.unique(time))
    chf = nelson_aalen_cs(time, event, unique_times, n_causes=2)
    assert np.all(np.diff(chf, axis=1) >= -1e-12)
