"""Tests for comprisk._importance and forest.compute_importance."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from sklearn.inspection import permutation_importance as sk_permutation_importance

from comprisk._importance import (
    _assemble_df,
    _compute_importance_impl,
    _make_cause_scorer,
    _weighted_mean,
)
from comprisk.forest import CompetingRiskForest


def _make_y(time, event):
    return np.rec.fromarrays([time, event], names=["time", "event"])


def _fit_small_forest(n_causes=2, n=200, p=5, seed=0, n_jobs=1):
    rng = np.random.RandomState(seed)
    X = rng.normal(size=(n, p))
    time = rng.exponential(1.0, size=n) + 0.1
    event = rng.choice(np.arange(n_causes + 1), size=n, p=[0.4] + [0.6 / n_causes] * n_causes)
    forest = CompetingRiskForest(
        n_estimators=30, max_depth=5, random_state=seed, n_jobs=n_jobs
    ).fit(X, time, event)
    return forest, X, time, event


def test_make_cause_scorer_returns_cindex():
    forest, X, time, event = _fit_small_forest()
    y = _make_y(time, event)
    scorer = _make_cause_scorer(cause=1)
    c = scorer(forest, X, y)
    assert 0.0 <= c <= 1.0
    # Sanity: match forest.score for the same cause
    assert np.isclose(c, forest.score(X, time, event, cause=1))


def test_weighted_mean_uniform_reduces_to_arithmetic_mean():
    per_cause = {1: np.array([0.1, 0.2, 0.3]), 2: np.array([0.5, 0.4, 0.3])}
    out = _weighted_mean(per_cause, weights=None)
    expected = np.array([0.3, 0.3, 0.3])
    np.testing.assert_allclose(out, expected)


def test_weighted_mean_respects_cause_weights():
    per_cause = {1: np.array([0.1, 0.2]), 2: np.array([0.5, 0.4])}
    # weights normalize: [3, 1] → [0.75, 0.25]; so col-0 = 0.75*0.1 + 0.25*0.5 = 0.2
    out = _weighted_mean(per_cause, weights=np.array([3.0, 1.0]))
    np.testing.assert_allclose(out, np.array([0.2, 0.25]))


def test_weighted_mean_stacks_in_sorted_cause_order():
    # Insertion order differs from numeric order; result must use numeric order.
    per_cause = {2: np.array([1.0, 1.0]), 1: np.array([0.0, 0.0])}
    out = _weighted_mean(per_cause, weights=np.array([3.0, 1.0]))
    # sorted causes -> [1, 2]; weights normalize to [0.75, 0.25]
    # col -> 0.75*0 + 0.25*1 = 0.25
    np.testing.assert_allclose(out, np.array([0.25, 0.25]))


def test_assemble_df_columns_and_order():
    feature_names = ["age", "bmi", "bp"]
    per_cause = {1: np.array([0.1, 0.2, 0.3]), 2: np.array([0.4, 0.5, 0.6])}
    composite = np.array([0.25, 0.35, 0.45])
    df = _assemble_df(feature_names, per_cause, composite)
    assert list(df.columns) == ["feature", "cause_1_vimp", "cause_2_vimp", "composite_vimp"]
    assert list(df["feature"]) == ["age", "bmi", "bp"]
    np.testing.assert_allclose(df["cause_1_vimp"].to_numpy(), [0.1, 0.2, 0.3])
    np.testing.assert_allclose(df["composite_vimp"].to_numpy(), [0.25, 0.35, 0.45])


def test_assemble_df_sorted_cause_columns():
    feature_names = ["a", "b"]
    # Insertion order (2, 1): output must still be cause_1 before cause_2.
    per_cause = {2: np.array([9.0, 9.0]), 1: np.array([1.0, 1.0])}
    composite = np.array([5.0, 5.0])
    df = _assemble_df(feature_names, per_cause, composite)
    assert list(df.columns) == ["feature", "cause_1_vimp", "cause_2_vimp", "composite_vimp"]
    np.testing.assert_allclose(df["cause_1_vimp"].to_numpy(), [1.0, 1.0])
    np.testing.assert_allclose(df["cause_2_vimp"].to_numpy(), [9.0, 9.0])


def test_compute_importance_impl_runs_end_to_end():
    forest, X, time, event = _fit_small_forest(n_causes=2)
    y = _make_y(time, event)
    feature_names = [f"feature_{i}" for i in range(X.shape[1])]
    df = _compute_importance_impl(
        forest,
        X,
        y,
        feature_names=feature_names,
        causes=[1, 2],
        cause_weights=None,
        n_repeats=2,
        random_state=0,
    )
    assert list(df.columns) == ["feature", "cause_1_vimp", "cause_2_vimp", "composite_vimp"]
    assert len(df) == X.shape[1]
    # composite should match unweighted mean of per-cause columns (weights=None)
    expected_composite = 0.5 * (df["cause_1_vimp"].to_numpy() + df["cause_2_vimp"].to_numpy())
    np.testing.assert_allclose(df["composite_vimp"].to_numpy(), expected_composite)


def test_feature_importances_raises_before_compute_call():
    forest, _, _, _ = _fit_small_forest()
    with pytest.raises(AttributeError, match="compute_importance"):
        _ = forest.feature_importances_


def test_compute_importance_no_args_routes_to_oob():
    """OOB mode landed; calling with no args runs OOB Breiman permutation VIMP."""
    forest, _, _, _ = _fit_small_forest()
    df = forest.compute_importance()
    assert "cause_1_vimp" in df.columns
    assert "composite_vimp" in df.columns


def test_compute_importance_rejects_plain_ndarray_y():
    # Guard against cryptic failures when user passes a plain 2-col array
    # instead of a structured array.
    forest, X, time, event = _fit_small_forest()
    y_bad = np.column_stack([time, event])  # shape (n, 2), NOT structured
    with pytest.raises(TypeError, match="structured array"):
        forest.compute_importance(X, y_bad)


def test_compute_importance_returns_dataframe_and_caches():
    forest, X, time, event = _fit_small_forest()
    y = _make_y(time, event)
    df = forest.compute_importance(X, y, n_repeats=2, random_state=0)
    assert list(df.columns) == ["feature", "cause_1_vimp", "cause_2_vimp", "composite_vimp"]
    assert len(df) == X.shape[1]
    cached = forest.feature_importances_
    pd.testing.assert_frame_equal(df, cached)


def test_compute_importance_second_call_overwrites_cache():
    forest, X, time, event = _fit_small_forest()
    y = _make_y(time, event)
    df1 = forest.compute_importance(X, y, n_repeats=2, random_state=0)
    df2 = forest.compute_importance(X, y, n_repeats=2, random_state=1)  # different seed
    pd.testing.assert_frame_equal(forest.feature_importances_, df2)
    # The two results should NOT be identical (different random_state).
    assert not df1["cause_1_vimp"].equals(df2["cause_1_vimp"])


def test_compute_importance_causes_arg_restricts_columns():
    forest, X, time, event = _fit_small_forest(n_causes=2)
    y = _make_y(time, event)
    df = forest.compute_importance(X, y, causes=[1], n_repeats=2, random_state=0)
    assert list(df.columns) == ["feature", "cause_1_vimp", "composite_vimp"]


def test_compute_importance_default_feature_names_are_positional():
    forest, X, time, event = _fit_small_forest()
    y = _make_y(time, event)
    df = forest.compute_importance(X, y, n_repeats=2, random_state=0)
    assert list(df["feature"]) == [f"feature_{i}" for i in range(X.shape[1])]


def test_compute_importance_bit_equivalent_repeated_calls():
    forest, X, time, event = _fit_small_forest()
    y = _make_y(time, event)
    df_a = forest.compute_importance(X, y, n_repeats=3, random_state=42)
    df_b = forest.compute_importance(X, y, n_repeats=3, random_state=42)
    pd.testing.assert_frame_equal(df_a, df_b, check_exact=True)


def test_compute_importance_bit_equivalent_across_forest_n_jobs():
    # Fit two forests with same seed, different n_jobs; predict must be
    # deterministic (already guaranteed by P2b), so VIMP must match too.
    f1, X, time, event = _fit_small_forest(n_jobs=1)
    f4, _, _, _ = _fit_small_forest(n_jobs=4)
    y = _make_y(time, event)
    df1 = f1.compute_importance(X, y, n_repeats=3, random_state=7)
    df4 = f4.compute_importance(X, y, n_repeats=3, random_state=7)
    pd.testing.assert_frame_equal(df1, df4, check_exact=True)


def _make_cr_data_with_signal(n=1000, n_informative=3, n_noise=3, seed=0):
    """CR data: cause-1 rate depends on X[:, :n_informative]; noise cols don't."""
    rng = np.random.RandomState(seed)
    p = n_informative + n_noise
    X = rng.normal(size=(n, p))
    # Linear predictor for cause-1 hazard; informative columns contribute.
    lp = X[:, :n_informative].sum(axis=1)
    # Cause-1 time from Exp with rate exp(lp); cause-2 time independent baseline.
    t1 = rng.exponential(np.exp(-lp), size=n)  # smaller time when lp large
    t2 = rng.exponential(1.0, size=n)
    tc = rng.exponential(2.0, size=n)  # censoring
    time = np.minimum(np.minimum(t1, t2), tc)
    event = np.where(time == tc, 0, np.where(time == t1, 1, 2)).astype(np.int64)
    return X, time, event, np.arange(n_informative), np.arange(n_informative, p)


def test_null_features_have_small_vimp():
    X, time, event, _info_idx, noise_idx = _make_cr_data_with_signal(n=1000, seed=0)
    forest = CompetingRiskForest(n_estimators=100, max_depth=8, random_state=0, n_jobs=1).fit(
        X, time, event
    )
    y = _make_y(time, event)
    df = forest.compute_importance(X, y, n_repeats=10, random_state=0)
    vimp_noise = df["composite_vimp"].to_numpy()[noise_idx]
    # Noise-feature VIMP should be small in absolute value.
    assert np.max(np.abs(vimp_noise)) < 0.06, f"noise VIMP too large: {vimp_noise}"


def test_informative_features_rank_above_null():
    X, time, event, info_idx, noise_idx = _make_cr_data_with_signal(n=1000, seed=1)
    forest = CompetingRiskForest(n_estimators=100, max_depth=8, random_state=1, n_jobs=1).fit(
        X, time, event
    )
    y = _make_y(time, event)
    df = forest.compute_importance(X, y, n_repeats=10, random_state=1)
    vimp = df["composite_vimp"].to_numpy()
    for i in info_idx:
        for j in noise_idx:
            assert vimp[i] > vimp[j], (
                f"informative feature {i} (vimp={vimp[i]:.4f}) did not rank above "
                f"noise feature {j} (vimp={vimp[j]:.4f})"
            )


def _make_per_cause_signal_data(n=1200, seed=0):
    """X[:, 0] drives cause-1; X[:, 1] drives cause-2; rest are noise."""
    rng = np.random.RandomState(seed)
    p = 5
    X = rng.normal(size=(n, p))
    t1 = rng.exponential(np.exp(-X[:, 0]), size=n)
    t2 = rng.exponential(np.exp(-X[:, 1]), size=n)
    tc = rng.exponential(3.0, size=n)
    time = np.minimum(np.minimum(t1, t2), tc)
    event = np.where(time == tc, 0, np.where(time == t1, 1, 2)).astype(np.int64)
    return X, time, event


def test_per_cause_vimp_specificity():
    X, time, event = _make_per_cause_signal_data(n=1500, seed=2)
    forest = CompetingRiskForest(n_estimators=150, max_depth=8, random_state=2, n_jobs=1).fit(
        X, time, event
    )
    y = _make_y(time, event)
    df = forest.compute_importance(X, y, n_repeats=10, random_state=2)
    c1 = df["cause_1_vimp"].to_numpy()
    c2 = df["cause_2_vimp"].to_numpy()
    # Feature 0 is cause-1-specific; feature 1 is cause-2-specific.
    assert c1[0] > c1[1], f"cause-1 VIMP for X0 should dominate X1: {c1[0]} vs {c1[1]}"
    assert c2[1] > c2[0], f"cause-2 VIMP for X1 should dominate X0: {c2[1]} vs {c2[0]}"


def test_sklearn_direct_call_matches_our_adapter():
    forest, X, time, event = _fit_small_forest(n_causes=2)
    y = _make_y(time, event)
    r = sk_permutation_importance(
        forest,
        X,
        y,
        scoring=_make_cause_scorer(cause=1),
        n_repeats=3,
        random_state=11,
        n_jobs=1,
    )
    df = forest.compute_importance(X, y, n_repeats=3, random_state=11)
    np.testing.assert_array_equal(
        df["cause_1_vimp"].to_numpy(),
        r.importances_mean,
    )
