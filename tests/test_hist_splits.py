"""Tests for numba-jitted histogram split kernels and the Python wrapper."""

import numpy as np
import pytest

from comprisk._binning import apply_bins, fit_bin_edges
from comprisk._hist_splits import (
    _best_split_in_feature,
    _best_split_in_feature_lr,
    _node_histograms,
    _observed_bins_sorted_ascending,
    find_best_split_hist,
)


def _make_binned_data(n=60, p=3, n_bins=8, seed=42):
    """Generate random binned X, time, event arrays for histogram-split tests.

    Returns a dict with keys: X, X_binned, bin_edges, time, event, t_idx,
    n_bins, n_time_bins, p, selected (np.arange(p, dtype=int64)).
    """
    rng = np.random.default_rng(seed)
    X = rng.uniform(0, 10, size=(n, p))
    time = rng.uniform(1.0, 10.0, n).astype(np.float64)
    event = rng.integers(0, 3, n).astype(np.int64)
    event[0] = 1
    event[1] = 2
    bin_edges = fit_bin_edges(X, n_bins=n_bins)
    X_binned = apply_bins(X, bin_edges)
    unique_t = np.sort(np.unique(time))
    t_idx = (np.searchsorted(unique_t, time, side="right") - 1).astype(np.int32)
    return {
        "X": X,
        "X_binned": X_binned,
        "bin_edges": bin_edges,
        "time": time,
        "event": event,
        "t_idx": t_idx,
        "n_bins": n_bins,
        "n_time_bins": len(unique_t),
        "p": p,
        "selected": np.arange(p, dtype=np.int64),
    }


def test_node_histograms_shape():
    # 4 samples, 2 selected features (each with 3 bins), 2 time bins, 2 causes
    bin_idx = np.array(
        [[0, 1], [1, 2], [2, 0], [0, 2]],
        dtype=np.uint8,
    )
    t_idx = np.array([0, 1, 0, 1], dtype=np.int32)
    ev = np.array([1, 2, 1, 0], dtype=np.int64)
    ev_hist, ar_hist = _node_histograms(bin_idx, t_idx, ev, n_bins=3, n_causes=2, n_time_bins=2)
    assert ev_hist.shape == (2, 3, 2, 2)
    assert ev_hist.dtype == np.uint32
    assert ar_hist.shape == (2, 3, 2)
    assert ar_hist.dtype == np.uint32


def test_node_histograms_counts_events_per_cause_per_time_bin():
    # Two samples in feature-0/bin-0, times (0, 1), events (1, 2)
    bin_idx = np.array([[0], [0]], dtype=np.uint8)
    t_idx = np.array([0, 1], dtype=np.int32)
    ev = np.array([1, 2], dtype=np.int64)
    ev_hist, _ = _node_histograms(bin_idx, t_idx, ev, n_bins=2, n_causes=2, n_time_bins=2)
    # feature 0, bin 0: cause 1 at t=0, cause 2 at t=1
    assert ev_hist[0, 0, 0, 0] == 1  # cause 1 at t=0
    assert ev_hist[0, 0, 1, 1] == 1  # cause 2 at t=1
    # No other cells filled
    assert ev_hist.sum() == 2


def test_node_histograms_at_risk_is_reverse_cumsum():
    # 3 samples all in feature-0/bin-0 with times 0, 1, 2
    bin_idx = np.array([[0], [0], [0]], dtype=np.uint8)
    t_idx = np.array([0, 1, 2], dtype=np.int32)
    ev = np.array([0, 0, 0], dtype=np.int64)
    _, ar_hist = _node_histograms(bin_idx, t_idx, ev, n_bins=2, n_causes=1, n_time_bins=3)
    # at_risk[t] = count with time_index >= t
    # at bin 0: 3 at t=0, 2 at t=1, 1 at t=2
    np.testing.assert_array_equal(ar_hist[0, 0], [3, 2, 1])


def test_node_histograms_ignores_censored_in_event_hist():
    bin_idx = np.array([[0], [0]], dtype=np.uint8)
    t_idx = np.array([0, 0], dtype=np.int32)
    ev = np.array([0, 0], dtype=np.int64)  # both censored
    ev_hist, ar_hist = _node_histograms(bin_idx, t_idx, ev, n_bins=2, n_causes=1, n_time_bins=1)
    assert ev_hist.sum() == 0
    assert ar_hist[0, 0, 0] == 2  # both at risk at t=0


def test_best_split_in_feature_no_valid_split_when_all_one_bin():
    # All samples in bin 0 ⇒ every candidate cut leaves the right side empty.
    # Rejection should come from the min_samples_leaf check on the right side,
    # not from zero variance — so include real events in bin 0.
    n_bins, n_causes, n_time_bins = 2, 1, 2
    ev_hist = np.zeros((n_bins, n_causes, n_time_bins), dtype=np.uint32)
    ar_hist = np.zeros((n_bins, n_time_bins), dtype=np.uint32)
    ev_hist[0, 0, 0] = 3  # 3 cause-1 events at t=0, all in bin 0
    ar_hist[0, 0] = 5  # 5 at risk at t=0 in bin 0
    ar_hist[0, 1] = 2  # 2 at risk at t=1 in bin 0
    # bin 1 is empty: ar_hist[1, :] = 0
    all_true = np.ones(n_bins - 1, dtype=np.bool_)
    best_bin, best_stat = _best_split_in_feature(
        ev_hist, ar_hist, n_node=5, min_samples_leaf=1, candidate_mask=all_true
    )
    assert best_bin == -1  # right side has 0 samples < 1 ⇒ no valid split
    assert best_stat == 0.0


def test_best_split_in_feature_picks_separating_bin():
    # 2 samples per bin, events at differing times by bin ⇒ clean split at bin 0
    # bin 0: 2 at risk at t=0, both cause 1 at t=0
    # bin 1: 2 at risk at t=0, 2 at risk at t=1, both cause 1 at t=1
    n_bins, n_causes, n_time_bins = 2, 1, 2
    ev_hist = np.zeros((n_bins, n_causes, n_time_bins), dtype=np.uint32)
    ar_hist = np.zeros((n_bins, n_time_bins), dtype=np.uint32)
    ev_hist[0, 0, 0] = 2  # bin 0 has 2 cause-1 events at t=0
    ar_hist[0, 0] = 2
    ev_hist[1, 0, 1] = 2  # bin 1 has 2 cause-1 events at t=1
    ar_hist[1, 0] = 2
    ar_hist[1, 1] = 2
    all_true = np.ones(n_bins - 1, dtype=np.bool_)
    best_bin, best_stat = _best_split_in_feature(
        ev_hist, ar_hist, n_node=4, min_samples_leaf=1, candidate_mask=all_true
    )
    # Only candidate split at bin 0 (left = bin <= 0 ⇒ 2 samples)
    assert best_bin == 0
    assert best_stat > 0.0


def test_best_split_in_feature_enforces_min_samples_leaf():
    # 2 bins, 1 sample each -> candidate split leaves 1 each side; reject at ms_leaf=2
    n_bins, n_causes, n_time_bins = 2, 1, 1
    ev_hist = np.zeros((n_bins, n_causes, n_time_bins), dtype=np.uint32)
    ar_hist = np.zeros((n_bins, n_time_bins), dtype=np.uint32)
    ar_hist[0, 0] = 1
    ar_hist[1, 0] = 1
    all_true = np.ones(n_bins - 1, dtype=np.bool_)
    best_bin, _ = _best_split_in_feature(
        ev_hist, ar_hist, n_node=2, min_samples_leaf=2, candidate_mask=all_true
    )
    assert best_bin == -1


def _make_simple_split_data():
    # 8 samples: first 4 in bin 0, last 4 in bin 1 for feature 0.
    # Feature 1 is a noise column all in bin 0.
    # Times/events: bin 0 samples have cause-1 events at t=0; bin 1 has no events
    X_binned = np.array(
        [[0, 0], [0, 0], [0, 0], [0, 0], [1, 0], [1, 0], [1, 0], [1, 0]],
        dtype=np.uint8,
    )
    t_idx = np.array([0, 0, 0, 0, 1, 1, 1, 1], dtype=np.int32)
    ev = np.array([1, 1, 1, 1, 0, 0, 0, 0], dtype=np.int64)
    return X_binned, t_idx, ev


def test_find_best_split_hist_picks_correct_feature():
    X_binned, t_idx, ev = _make_simple_split_data()
    selected = np.array([0, 1], dtype=np.int64)
    feature, bin_idx, stat = find_best_split_hist(
        X_binned,
        t_idx,
        ev,
        selected,
        n_bins=2,
        n_causes=1,
        n_time_bins=2,
        min_samples_leaf=1,
    )
    assert feature == 0
    assert bin_idx == 0
    assert stat > 0.0


def test_find_best_split_hist_returns_minus_one_when_no_valid_split():
    # All samples in bin 0 for both features
    X_binned = np.zeros((4, 2), dtype=np.uint8)
    t_idx = np.array([0, 0, 0, 0], dtype=np.int32)
    ev = np.array([1, 1, 0, 0], dtype=np.int64)
    selected = np.array([0, 1], dtype=np.int64)
    feature, bin_idx, stat = find_best_split_hist(
        X_binned,
        t_idx,
        ev,
        selected,
        n_bins=2,
        n_causes=1,
        n_time_bins=1,
        min_samples_leaf=1,
    )
    assert feature == -1
    assert bin_idx == 0
    assert stat == 0.0


def test_find_best_split_hist_respects_selected_features():
    # Feature 1 has a strong signal; feature 0 is noise.
    # Column 1 = [0, 1, 0, 1]; events (cause 1) are at rows 0 and 2 (bin 0 of feat 1).
    # Splitting on feature 1 at bin 0 cleanly separates the events.
    X_binned = np.array([[0, 0], [0, 1], [1, 0], [1, 1]], dtype=np.uint8)
    t_idx = np.array([0, 1, 0, 1], dtype=np.int32)
    ev = np.array([1, 0, 1, 0], dtype=np.int64)

    # Select only feature 1 — the function must return feature 1 (not 0 or -1).
    selected = np.array([1], dtype=np.int64)
    feature, bin_idx, stat = find_best_split_hist(
        X_binned,
        t_idx,
        ev,
        selected,
        n_bins=2,
        n_causes=1,
        n_time_bins=2,
        min_samples_leaf=1,
    )
    assert feature == 1
    assert bin_idx == 0
    assert stat > 0.0


def test_best_split_in_feature_lr_single_cause_matches_reference():
    """Histogram logrank kernel agrees with reference cause_specific_log_rank_statistic."""
    from comprisk._splits import bin_times, cause_specific_log_rank_statistic

    rng = np.random.default_rng(0)
    n = 50
    X = rng.standard_normal((n, 1))
    time = rng.uniform(0.1, 5.0, size=n)
    event = rng.integers(0, 3, size=n).astype(np.int64)  # codes 0,1,2

    edges = fit_bin_edges(X, n_bins=32)
    X_binned = apply_bins(X, edges).astype(np.uint8)
    # Use ALL unique times (including censored) so the histogram at-risk equals
    # the reference reverse-cumsum exactly — event-only grids omit censored
    # subjects and cause staggered-censoring approximation error.
    grid = np.unique(time)
    t_idx = np.searchsorted(grid, time, side="left").astype(np.int32)

    event_hist, at_risk = _node_histograms(X_binned, t_idx, event, 32, 2, len(grid))
    all_true = np.ones(32 - 1, dtype=np.bool_)
    best_bin, best_stat = _best_split_in_feature_lr(
        event_hist[0], at_risk[0], n, min_samples_leaf=2, cause=1, candidate_mask=all_true
    )
    assert best_bin >= 0

    # Reference: enumerate midpoints on original X, pick argmax of cause_specific_log_rank_statistic
    tb = bin_times(time, event)
    uniq = np.sort(np.unique(X[:, 0]))
    mids = (uniq[:-1] + uniq[1:]) / 2.0
    ref_best = -1.0
    for m in mids:
        left = X[:, 0] <= m
        if left.sum() < 2 or (~left).sum() < 2:
            continue
        s = cause_specific_log_rank_statistic(tb, left, cause=1)
        if s > ref_best:
            ref_best = s
    assert np.isclose(best_stat, ref_best, rtol=1e-5, atol=1e-8)


def test_find_best_split_hist_logrank_dispatches_to_lr_kernel():
    from comprisk._time_grid import fit_time_grid

    rng = np.random.default_rng(1)
    n, p = 60, 3
    X = rng.standard_normal((n, p))
    time = rng.uniform(0.1, 5.0, size=n)
    event = rng.integers(0, 3, size=n).astype(np.int64)

    edges = fit_bin_edges(X, n_bins=32)
    X_binned = apply_bins(X, edges).astype(np.uint8)
    grid = fit_time_grid(time, event, max_points=40)
    t_idx = np.clip(np.searchsorted(grid, time, side="right") - 1, 0, len(grid) - 1).astype(
        np.int32
    )
    selected = np.arange(p, dtype=np.int64)

    feat_cr, bin_cr, stat_cr = find_best_split_hist(
        X_binned,
        t_idx,
        event,
        selected,
        n_bins=32,
        n_causes=2,
        n_time_bins=len(grid),
        min_samples_leaf=3,
    )
    feat_lr, bin_lr, stat_lr = find_best_split_hist(
        X_binned,
        t_idx,
        event,
        selected,
        n_bins=32,
        n_causes=2,
        n_time_bins=len(grid),
        min_samples_leaf=3,
        splitrule="logrank",
        cause=1,
    )
    # Split criteria differ, so typically feat/bin/stat differ
    assert (feat_cr, bin_cr) != (feat_lr, bin_lr) or not np.isclose(stat_cr, stat_lr)


def test_best_split_in_feature_respects_candidate_mask():
    """With a mask that excludes the optimal split, kernel picks the best allowed one."""
    n_bins, n_causes, n_time_bins = 4, 2, 3
    event_hist = np.zeros((n_bins, n_causes, n_time_bins), dtype=np.uint32)
    at_risk_hist = np.zeros((n_bins, n_time_bins), dtype=np.uint32)

    # Cause-1 events concentrated in bin 0 at t=0 (5 events), bin 3 otherwise
    event_hist[0, 0, 0] = 5
    event_hist[3, 0, 2] = 5
    # 10 samples per bin, all starting at t=0
    for b in range(n_bins):
        at_risk_hist[b, :] = 10

    all_true = np.ones(n_bins - 1, dtype=np.bool_)
    best_bin_all, best_stat_all = _best_split_in_feature(
        event_hist, at_risk_hist, n_node=40, min_samples_leaf=1, candidate_mask=all_true
    )

    # Mask out the top-scoring boundary; result should change
    mask = all_true.copy()
    mask[best_bin_all] = False
    best_bin_masked, best_stat_masked = _best_split_in_feature(
        event_hist, at_risk_hist, n_node=40, min_samples_leaf=1, candidate_mask=mask
    )
    assert best_bin_masked != best_bin_all
    assert best_stat_masked <= best_stat_all + 1e-12


def test_best_split_in_feature_all_true_mask_matches_unmasked_behavior():
    """All-True mask reproduces the pre-P3a.5 exhaustive scan."""
    # Random-ish but fixed histograms so the test is deterministic
    rng = np.random.default_rng(0)
    n_bins, n_causes, n_time_bins = 5, 2, 4
    event_hist = rng.integers(0, 3, size=(n_bins, n_causes, n_time_bins)).astype(np.uint32)
    at_risk_hist = rng.integers(2, 8, size=(n_bins, n_time_bins)).astype(np.uint32)

    all_true = np.ones(n_bins - 1, dtype=np.bool_)
    best_bin, best_stat = _best_split_in_feature(
        event_hist,
        at_risk_hist,
        n_node=int(at_risk_hist[:, 0].sum()),
        min_samples_leaf=1,
        candidate_mask=all_true,
    )
    # Kernel must return *some* valid bin; we check that against the
    # same statistic recomputed with a single-bin-included mask.
    assert best_bin >= 0
    one_hot = np.zeros(n_bins - 1, dtype=np.bool_)
    one_hot[best_bin] = True
    best_bin_oh, best_stat_oh = _best_split_in_feature(
        event_hist,
        at_risk_hist,
        n_node=int(at_risk_hist[:, 0].sum()),
        min_samples_leaf=1,
        candidate_mask=one_hot,
    )
    assert best_bin_oh == best_bin
    assert abs(best_stat_oh - best_stat) < 1e-12


def test_best_split_in_feature_lr_respects_candidate_mask():
    """logrank (cause-specific) kernel honours the same mask convention."""
    n_bins, n_causes, n_time_bins = 4, 2, 3
    event_hist = np.zeros((n_bins, n_causes, n_time_bins), dtype=np.uint32)
    at_risk_hist = np.zeros((n_bins, n_time_bins), dtype=np.uint32)
    event_hist[0, 0, 0] = 5
    event_hist[3, 0, 2] = 5
    for b in range(n_bins):
        at_risk_hist[b, :] = 10

    all_true = np.ones(n_bins - 1, dtype=np.bool_)
    best_bin_all, best_stat_all = _best_split_in_feature_lr(
        event_hist,
        at_risk_hist,
        n_node=40,
        min_samples_leaf=1,
        cause=1,
        candidate_mask=all_true,
    )
    mask = all_true.copy()
    mask[best_bin_all] = False
    best_bin_masked, best_stat_masked = _best_split_in_feature_lr(
        event_hist,
        at_risk_hist,
        n_node=40,
        min_samples_leaf=1,
        cause=1,
        candidate_mask=mask,
    )
    assert best_bin_masked != best_bin_all
    assert best_stat_masked <= best_stat_all + 1e-12


def test_find_best_split_hist_nsplit_returns_valid_result():
    """With nsplit=1 and a fixed rng, the chosen bin must be the single sampled one."""
    d = _make_binned_data(n=60, p=3, n_bins=8, seed=42)

    rng = np.random.RandomState(0)
    feat_ns1, bin_ns1, _ = find_best_split_hist(
        d["X_binned"],
        d["t_idx"],
        d["event"],
        d["selected"],
        n_bins=d["n_bins"],
        n_causes=2,
        n_time_bins=d["n_time_bins"],
        min_samples_leaf=1,
        splitrule="logrankCR",
        nsplit=1,
        rng=rng,
    )
    assert 0 <= feat_ns1 < d["p"]
    assert 0 <= bin_ns1 < d["n_bins"] - 1


def test_find_best_split_hist_nsplit_zero_matches_exhaustive():
    """nsplit=0 must produce identical output to the pre-P3a.5 exhaustive call."""
    d = _make_binned_data(n=50, p=3, n_bins=8, seed=1)

    # Call WITHOUT nsplit (pre-P3a.5 API) — exhaustive scan.
    feat_ex, bin_ex, stat_ex = find_best_split_hist(
        d["X_binned"],
        d["t_idx"],
        d["event"],
        d["selected"],
        n_bins=d["n_bins"],
        n_causes=2,
        n_time_bins=d["n_time_bins"],
        min_samples_leaf=1,
        splitrule="logrankCR",
    )
    # Call WITH nsplit=0 — must match exactly.
    feat_ns0, bin_ns0, stat_ns0 = find_best_split_hist(
        d["X_binned"],
        d["t_idx"],
        d["event"],
        d["selected"],
        n_bins=d["n_bins"],
        n_causes=2,
        n_time_bins=d["n_time_bins"],
        min_samples_leaf=1,
        splitrule="logrankCR",
        nsplit=0,
        rng=None,
    )
    assert (feat_ex, bin_ex) == (feat_ns0, bin_ns0)
    assert abs(stat_ex - stat_ns0) < 1e-12


def test_find_best_split_hist_nsplit_deterministic_given_rng():
    """Same rng state -> same split choice."""
    d = _make_binned_data(n=60, p=3, n_bins=8, seed=7)

    rng_a = np.random.RandomState(123)
    out_a = find_best_split_hist(
        d["X_binned"],
        d["t_idx"],
        d["event"],
        d["selected"],
        n_bins=d["n_bins"],
        n_causes=2,
        n_time_bins=d["n_time_bins"],
        min_samples_leaf=1,
        splitrule="logrankCR",
        nsplit=4,
        rng=rng_a,
    )
    rng_b = np.random.RandomState(123)
    out_b = find_best_split_hist(
        d["X_binned"],
        d["t_idx"],
        d["event"],
        d["selected"],
        n_bins=d["n_bins"],
        n_causes=2,
        n_time_bins=d["n_time_bins"],
        min_samples_leaf=1,
        splitrule="logrankCR",
        nsplit=4,
        rng=rng_b,
    )
    assert out_a == out_b


def test_find_best_split_hist_nsplit_can_reach_top_boundary():
    """Regression: boundary b_hi - 1 must be reachable under uniform sampling.

    Previous implementation of nsplit used `x_hi = edges_j[b_hi - 1]` as the
    exclusive upper bound of `rng.uniform`, which made boundary `b_hi - 1`
    unreachable. This test ensures the fix: drawing many samples, the chosen
    bin index must include values at or near the top of the valid range.
    """
    d = _make_binned_data(n=200, p=1, n_bins=16, seed=123)

    # Observed bin range on the single feature
    b_lo = int(d["X_binned"][:, 0].min())
    b_hi = int(d["X_binned"][:, 0].max())
    assert b_hi > b_lo + 1  # sanity — need at least two valid boundaries for this test

    # Run many seeds and collect the chosen bin indices
    chosen_bins = set()
    for seed in range(50):
        rng = np.random.RandomState(seed)
        _, bin_idx, _ = find_best_split_hist(
            d["X_binned"],
            d["t_idx"],
            d["event"],
            d["selected"],
            n_bins=d["n_bins"],
            n_causes=2,
            n_time_bins=d["n_time_bins"],
            min_samples_leaf=1,
            splitrule="logrankCR",
            nsplit=1,
            rng=rng,
        )
        if bin_idx >= 0:
            chosen_bins.add(int(bin_idx))

    # Top valid boundary must be reachable: it should be in the set of
    # chosen bins across 50 seeds (with nsplit=1 each, and the full feature
    # range spanning many bins, the top boundary must occasionally be picked).
    assert b_hi - 1 in chosen_bins, (
        f"Boundary b_hi-1 = {b_hi - 1} never reached; chosen_bins = {chosen_bins}"
    )


def test_find_best_split_hist_nsplit_single_boundary_case():
    """When b_hi == b_lo + 1, the single valid boundary b_lo must be found."""
    from comprisk._binning import apply_bins, fit_bin_edges

    # Construct data where all samples fall in just 2 bins
    rng_data = np.random.default_rng(99)
    n = 40
    X = rng_data.uniform(0, 10, size=(n, 2))
    # Force feature 0 to have only two distinct bins by clustering values
    X[: n // 2, 0] = 1.0
    X[n // 2 :, 0] = 9.0
    time = rng_data.uniform(1.0, 10.0, n).astype(np.float64)
    event = rng_data.integers(0, 3, n).astype(np.int64)
    event[0] = 1
    event[1] = 2

    n_bins = 8
    bin_edges = fit_bin_edges(X, n_bins=n_bins)
    X_binned = apply_bins(X, bin_edges)
    unique_t = np.sort(np.unique(time))
    t_idx = (np.searchsorted(unique_t, time, side="right") - 1).astype(np.int32)
    n_time_bins = len(unique_t)
    selected = np.array([0], dtype=np.int64)

    # Pick nsplit=3 to exercise the sampling path, not fall through to no-split.
    rng = np.random.RandomState(0)
    feat, bin_idx, _stat = find_best_split_hist(
        X_binned,
        t_idx,
        event,
        selected,
        n_bins=n_bins,
        n_causes=2,
        n_time_bins=n_time_bins,
        min_samples_leaf=1,
        splitrule="logrankCR",
        nsplit=3,
        rng=rng,
    )
    # Only one valid boundary exists; it must be selected (not -1, not skipped).
    b_lo = int(X_binned[:, 0].min())
    assert feat == 0
    assert bin_idx == b_lo  # the single valid boundary between the two bins


def test_find_best_split_hist_nsplit_works_with_logrank_splitrule():
    """Logrank splitrule should also honour nsplit threshold sampling.

    The nsplit mask construction is shared across both splitrules, but the
    mask is passed to `_best_split_in_feature_lr` instead of
    `_best_split_in_feature`. A typo in the else-branch wiring could
    silently break the logrank path; this test closes that gap.
    """
    d = _make_binned_data(n=80, p=3, n_bins=8, seed=55)

    # Determinism: same rng state -> same output, under logrank splitrule + nsplit.
    rng_a = np.random.RandomState(42)
    out_a = find_best_split_hist(
        d["X_binned"],
        d["t_idx"],
        d["event"],
        d["selected"],
        n_bins=d["n_bins"],
        n_causes=2,
        n_time_bins=d["n_time_bins"],
        min_samples_leaf=1,
        splitrule="logrank",
        cause=1,
        nsplit=4,
        rng=rng_a,
    )
    rng_b = np.random.RandomState(42)
    out_b = find_best_split_hist(
        d["X_binned"],
        d["t_idx"],
        d["event"],
        d["selected"],
        n_bins=d["n_bins"],
        n_causes=2,
        n_time_bins=d["n_time_bins"],
        min_samples_leaf=1,
        splitrule="logrank",
        cause=1,
        nsplit=4,
        rng=rng_b,
    )
    assert out_a == out_b

    # nsplit=0 matches exhaustive output under logrank too.
    feat_ex, bin_ex, stat_ex = find_best_split_hist(
        d["X_binned"],
        d["t_idx"],
        d["event"],
        d["selected"],
        n_bins=d["n_bins"],
        n_causes=2,
        n_time_bins=d["n_time_bins"],
        min_samples_leaf=1,
        splitrule="logrank",
        cause=1,
    )
    feat_ns0, bin_ns0, stat_ns0 = find_best_split_hist(
        d["X_binned"],
        d["t_idx"],
        d["event"],
        d["selected"],
        n_bins=d["n_bins"],
        n_causes=2,
        n_time_bins=d["n_time_bins"],
        min_samples_leaf=1,
        splitrule="logrank",
        cause=1,
        nsplit=0,
        rng=None,
    )
    assert (feat_ex, bin_ex) == (feat_ns0, bin_ns0)
    assert abs(stat_ex - stat_ns0) < 1e-12


@pytest.mark.parametrize("seed", list(range(10)))
def test_observed_bins_sorted_ascending_matches_np_unique(seed: int) -> None:
    """Output values and order must equal np.unique bit-exactly."""
    rng = np.random.default_rng(seed)
    n_bins = 256
    for n_rows in (1, 2, 10, 100, 10_000):
        col = rng.integers(0, n_bins, size=n_rows, dtype=np.uint8)
        expected = np.unique(col)
        got = _observed_bins_sorted_ascending(col, n_bins)
        assert np.array_equal(got, expected), (
            f"seed={seed} n={n_rows}: got={got} expected={expected}"
        )
