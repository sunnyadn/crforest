"""Tests for split statistics and best-split search."""

import numpy as np

from comprisk._splits import (
    _logrank_components,
    bin_times,
    composite_log_rank_statistic,
    find_best_split,
    log_rank_statistic_relabeled,
)


def test_bin_times_basic():
    time = np.array([3.0, 1.0, 2.0, 1.0])
    event = np.array([1, 1, 0, 1])
    tb = bin_times(time, event)
    # unique times {1, 2, 3} so n_times = 3
    assert tb.n_times == 3
    assert np.array_equal(tb.n_total, [2.0, 1.0, 1.0])
    # any-event count per bin: at t=1 both events, at t=2 none, at t=3 one
    assert np.array_equal(tb.d_any, [2.0, 0.0, 1.0])


def test_log_rank_degenerate_all_left_returns_zero():
    time = np.array([1.0, 2.0, 3.0])
    event = np.array([1, 1, 1])
    tb = bin_times(time, event)
    left = np.array([True, True, True])
    assert log_rank_statistic_relabeled(tb, left, cause=1) == 0.0


def test_log_rank_degenerate_all_right_returns_zero():
    time = np.array([1.0, 2.0, 3.0])
    event = np.array([1, 1, 1])
    tb = bin_times(time, event)
    left = np.array([False, False, False])
    assert log_rank_statistic_relabeled(tb, left, cause=1) == 0.0


def test_log_rank_statistic_positive_for_separating_split():
    # Group A has all early events; Group B has all later events
    time = np.array([1.0, 2.0, 8.0, 9.0])
    event = np.array([1, 1, 1, 1])
    tb = bin_times(time, event)
    left = np.array([True, True, False, False])
    stat = log_rank_statistic_relabeled(tb, left, cause=1)
    assert stat > 0.0


def test_log_rank_ignores_non_cause_events():
    # Same structure but cause-2 events (treated as censored when splitting on cause 1)
    time = np.array([1.0, 2.0, 8.0, 9.0])
    event = np.array([2, 2, 2, 2])
    tb = bin_times(time, event)
    left = np.array([True, True, False, False])
    # No cause-1 events → statistic is zero
    assert log_rank_statistic_relabeled(tb, left, cause=1) == 0.0


def test_composite_equals_single_cause_when_one_cause():
    # Only cause 1 present → composite (sum over k=1..1) == cause-1 log-rank
    time = np.array([1.0, 2.0, 4.0, 5.0, 6.0])
    event = np.array([1, 0, 1, 0, 1])
    tb = bin_times(time, event)
    left = np.array([True, True, False, False, False])
    single = log_rank_statistic_relabeled(tb, left, cause=1)
    composite = composite_log_rank_statistic(tb, left, n_causes=1)
    assert np.isclose(single, composite)


def test_composite_pools_numerators_and_variances_across_causes():
    """Composite matches (Σ_k num_k)^2 / Σ_k var_k (rfSRC logrankCR)."""
    time = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
    event = np.array([1, 2, 1, 2, 1, 2])
    tb = bin_times(time, event)
    left = np.array([True, True, True, False, False, False])
    num1, var1 = _logrank_components(tb, left, cause=1)
    num2, var2 = _logrank_components(tb, left, cause=2)
    pooled = (num1 + num2) ** 2 / (var1 + var2)
    composite = composite_log_rank_statistic(tb, left, n_causes=2)
    assert np.isclose(composite, pooled)


def test_composite_differs_from_sum_of_per_cause_stats():
    """Regression guard against the pre-P2.5 additive formulation."""
    time = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
    event = np.array([1, 2, 1, 2, 1, 2])
    tb = bin_times(time, event)
    left = np.array([True, True, True, False, False, False])
    c1 = log_rank_statistic_relabeled(tb, left, cause=1)
    c2 = log_rank_statistic_relabeled(tb, left, cause=2)
    composite = composite_log_rank_statistic(tb, left, n_causes=2)
    # Under the new pooled formula these are not equal; signed numerators
    # cancel across causes (cause 1 favors left, cause 2 favors right here).
    assert not np.isclose(composite, c1 + c2)


def test_find_best_split_picks_separating_feature():
    # Feature 0: perfectly separates early/late cause-1 events
    # Feature 1: no information
    rng = np.random.default_rng(0)
    n = 40
    X = np.zeros((n, 2))
    X[:, 0] = np.concatenate([np.zeros(20), np.ones(20)])
    X[:, 1] = rng.uniform(size=n)
    time = np.concatenate([rng.uniform(0.5, 1.5, 20), rng.uniform(5.0, 6.0, 20)])
    event = np.ones(n, dtype=int)
    feat, thresh, stat = find_best_split(X, time, event, n_causes=1, min_samples_leaf=2)
    assert feat == 0
    assert 0.0 < thresh < 1.0
    assert stat > 0.0


def test_find_best_split_respects_min_samples_leaf():
    # Only 4 samples; min_samples_leaf=3 means no valid split
    X = np.array([[0.0], [1.0], [2.0], [3.0]])
    time = np.array([1.0, 2.0, 3.0, 4.0])
    event = np.array([1, 1, 1, 1])
    feat, _thresh, stat = find_best_split(X, time, event, n_causes=1, min_samples_leaf=3)
    assert feat == -1
    assert stat == 0.0


def test_find_best_split_returns_no_split_when_all_same_feature():
    X = np.ones((10, 2))
    time = np.arange(1.0, 11.0)
    event = np.ones(10, dtype=int)
    feat, _thresh, stat = find_best_split(X, time, event, n_causes=1, min_samples_leaf=2)
    assert feat == -1
    assert stat == 0.0


def test_logrank_components_std_uses_standard_at_risk():
    """Standard (non-Lau) at-risk: competing events remove subjects from risk set."""
    # Setup: at t=1, cause-2 event; at t=2, cause-1 event in the left child.
    # With standard at-risk, the cause-2 subject is gone at t=2 — so n_P(2) = 3,
    # not 4 (Lau-inclusive would have 4). This gives a different numerator.
    from comprisk._splits import _logrank_components, _logrank_components_std, bin_times

    time = np.array([1.0, 2.0, 3.0, 4.0])
    event = np.array([2, 1, 1, 1])
    tb = bin_times(time, event)
    left = np.array([True, True, False, False])

    num_lau, var_lau = _logrank_components(tb, left, cause=1)
    num_std, var_std = _logrank_components_std(tb, left, cause=1)

    # The numerators must differ because the at-risk sets differ at t=2
    # (Lau adds back the cause-2 subject at t=1; standard does not).
    assert not np.isclose(num_lau, num_std)
    assert var_lau != var_std


def test_logrank_components_std_equals_lau_when_no_competing_events():
    """When all events are cause 1, Lau add-back is zero; std must equal Lau exactly."""
    from comprisk._splits import _logrank_components, _logrank_components_std, bin_times

    time = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
    event = np.array([1, 0, 1, 0, 1, 1])  # only cause 1 events + some censoring
    tb = bin_times(time, event)
    left = np.array([True, True, True, False, False, False])

    num_lau, var_lau = _logrank_components(tb, left, cause=1)
    num_std, var_std = _logrank_components_std(tb, left, cause=1)

    assert np.isclose(num_lau, num_std)
    assert np.isclose(var_lau, var_std)


def test_logrank_components_std_numerator_hand_computed():
    """Pin exact numerator for a small hand-traced example."""
    from comprisk._splits import _logrank_components_std, bin_times

    time = np.array([1.0, 2.0, 3.0, 4.0])
    event = np.array([2, 1, 1, 1])
    tb = bin_times(time, event)
    left = np.array([True, True, False, False])

    num, _var = _logrank_components_std(tb, left, cause=1)
    # See P3a Task 2 review notes: expected numerator = 2/3 (only t=2 contributes).
    assert np.isclose(num, 2.0 / 3.0)


def test_cause_specific_log_rank_single_cause_matches_std_components():
    from comprisk._splits import (
        _logrank_components_std,
        bin_times,
        cause_specific_log_rank_statistic,
    )

    time = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    event = np.array([1, 2, 1, 2, 1])
    tb = bin_times(time, event)
    left = np.array([True, True, True, False, False])

    num, var = _logrank_components_std(tb, left, cause=1)
    expected = num**2 / var if var >= 1e-12 else 0.0
    stat = cause_specific_log_rank_statistic(tb, left, cause=1)
    assert np.isclose(stat, expected)


def test_cause_specific_log_rank_degenerate_splits_return_zero():
    from comprisk._splits import bin_times, cause_specific_log_rank_statistic

    time = np.array([1.0, 2.0, 3.0])
    event = np.array([1, 1, 1])
    tb = bin_times(time, event)
    assert cause_specific_log_rank_statistic(tb, np.array([True, True, True]), cause=1) == 0.0
    assert cause_specific_log_rank_statistic(tb, np.array([False, False, False]), cause=1) == 0.0


def test_cause_specific_log_rank_weighted_pools_weighted_components():
    """Weighted form: (Σ_k w_k num_k)² / (Σ_k w_k² var_k)."""
    from comprisk._splits import (
        _logrank_components_std,
        bin_times,
        cause_specific_log_rank_statistic,
    )

    time = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
    event = np.array([1, 2, 1, 2, 1, 2])
    tb = bin_times(time, event)
    left = np.array([True, True, True, False, False, False])

    num1, var1 = _logrank_components_std(tb, left, cause=1)
    num2, var2 = _logrank_components_std(tb, left, cause=2)
    w = np.array([0.75, 0.25])
    expected = (w[0] * num1 + w[1] * num2) ** 2 / (w[0] ** 2 * var1 + w[1] ** 2 * var2)

    got = cause_specific_log_rank_statistic(tb, left, weights=w, n_causes=2)
    assert np.isclose(got, expected)


def test_find_best_split_logrank_single_cause_respects_cause_param():
    """With cause=1, the only event type that matters is cause-1; cause-2 events
    are treated as censored."""
    from comprisk._splits import find_best_split

    rng = np.random.default_rng(0)
    n = 40
    X = np.zeros((n, 2))
    # Feature 0 separates cause-1 events but NOT cause-2 events
    X[:, 0] = np.concatenate([np.zeros(20), np.ones(20)])
    X[:, 1] = rng.uniform(size=n)
    time = np.concatenate([rng.uniform(0.5, 1.5, 20), rng.uniform(5.0, 6.0, 20)])
    # Left half: cause-1; right half: cause-2
    event = np.concatenate([np.ones(20, dtype=int), 2 * np.ones(20, dtype=int)])
    feat, _thresh, stat = find_best_split(
        X,
        time,
        event,
        n_causes=2,
        min_samples_leaf=2,
        splitrule="logrank",
        cause=1,
    )
    assert feat == 0
    assert stat > 0.0


def test_find_best_split_nsplit_deterministic_given_rng():
    """nsplit=k with a fixed rng reproduces the same split."""
    rng_data = np.random.default_rng(3)
    n, p = 40, 3
    X = rng_data.uniform(0, 10, size=(n, p))
    time = rng_data.uniform(1.0, 10.0, n)
    event = rng_data.integers(0, 3, n)
    event[0] = 1
    event[1] = 2

    rng_a = np.random.RandomState(7)
    out_a = find_best_split(
        X,
        time,
        event,
        n_causes=2,
        min_samples_leaf=1,
        splitrule="logrankCR",
        nsplit=5,
        rng=rng_a,
    )
    rng_b = np.random.RandomState(7)
    out_b = find_best_split(
        X,
        time,
        event,
        n_causes=2,
        min_samples_leaf=1,
        splitrule="logrankCR",
        nsplit=5,
        rng=rng_b,
    )
    assert out_a == out_b


def test_find_best_split_nsplit_zero_matches_exhaustive():
    """nsplit=0 reproduces pre-P3a.5 exhaustive output."""
    rng_data = np.random.default_rng(4)
    n, p = 40, 3
    X = rng_data.uniform(0, 10, size=(n, p))
    time = rng_data.uniform(1.0, 10.0, n)
    event = rng_data.integers(0, 3, n)
    event[0] = 1
    event[1] = 2

    out_ex = find_best_split(
        X,
        time,
        event,
        n_causes=2,
        min_samples_leaf=1,
        splitrule="logrankCR",
    )
    out_ns0 = find_best_split(
        X,
        time,
        event,
        n_causes=2,
        min_samples_leaf=1,
        splitrule="logrankCR",
        nsplit=0,
        rng=None,
    )
    assert out_ex == out_ns0


def test_find_best_split_nsplit_logrank_branch():
    """nsplit > 0 also works with splitrule='logrank'."""
    rng_data = np.random.default_rng(5)
    n, p = 40, 3
    X = rng_data.uniform(0, 10, size=(n, p))
    time = rng_data.uniform(1.0, 10.0, n)
    event = rng_data.integers(0, 3, n)
    event[0] = 1
    event[1] = 2

    rng_a = np.random.RandomState(42)
    out_a = find_best_split(
        X,
        time,
        event,
        n_causes=2,
        min_samples_leaf=1,
        splitrule="logrank",
        cause=1,
        nsplit=5,
        rng=rng_a,
    )
    rng_b = np.random.RandomState(42)
    out_b = find_best_split(
        X,
        time,
        event,
        n_causes=2,
        min_samples_leaf=1,
        splitrule="logrank",
        cause=1,
        nsplit=5,
        rng=rng_b,
    )
    assert out_a == out_b
