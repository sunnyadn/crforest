"""Tests for validation.alignment.compare_splits."""

import numpy as np
import pytest
from validation.alignment import _rpy2_available
from validation.alignment.compare_splits import (
    crforest_candidate_stats,
    rfsrc_per_feature_best_split,
    toy_input,
)

from crforest._splits import find_best_split


def test_toy_input_shapes():
    data = toy_input(seed=0, n=30, n_features=3, n_causes=2)
    assert data["X"].shape == (30, 3)
    assert data["time"].shape == (30,)
    assert data["event"].shape == (30,)
    assert data["n_causes"] == 2


def test_toy_input_event_codes_in_range():
    data = toy_input(seed=0, n=30, n_features=3, n_causes=2)
    assert data["event"].min() >= 0
    assert data["event"].max() <= 2


def test_toy_input_is_deterministic_for_same_seed():
    a = toy_input(seed=42, n=30, n_features=3, n_causes=2)
    b = toy_input(seed=42, n=30, n_features=3, n_causes=2)
    assert np.array_equal(a["X"], b["X"])
    assert np.array_equal(a["time"], b["time"])
    assert np.array_equal(a["event"], b["event"])


def test_toy_input_seeds_produce_different_data():
    a = toy_input(seed=0, n=30, n_features=3, n_causes=2)
    b = toy_input(seed=1, n=30, n_features=3, n_causes=2)
    assert not np.array_equal(a["X"], b["X"])


def test_toy_input_has_both_causes_and_censoring():
    # Regression guard: a pathological seed must not produce all-one-cause data.
    data = toy_input(seed=0, n=30, n_features=3, n_causes=2)
    assert (data["event"] == 0).any()
    assert (data["event"] == 1).any()
    assert (data["event"] == 2).any()


def test_crforest_candidate_stats_schema():
    data = toy_input(seed=0, n=30, n_features=3, n_causes=2)
    df = crforest_candidate_stats(data["X"], data["time"], data["event"], data["n_causes"])
    assert set(df.columns) == {"feature", "threshold", "stat"}
    assert len(df) > 0
    assert df["feature"].dtype.kind in "iu"
    assert df["threshold"].dtype.kind == "f"
    assert df["stat"].dtype.kind == "f"


def test_crforest_candidate_stats_covers_all_features():
    data = toy_input(seed=0, n=30, n_features=3, n_causes=2)
    df = crforest_candidate_stats(data["X"], data["time"], data["event"], data["n_causes"])
    assert set(df["feature"].unique()) == {0, 1, 2}


def test_crforest_candidate_stats_best_matches_find_best_split():
    data = toy_input(seed=0, n=30, n_features=3, n_causes=2)
    df = crforest_candidate_stats(data["X"], data["time"], data["event"], data["n_causes"])
    best_row = df.sort_values("stat", ascending=False).iloc[0]
    _feat, _thresh, stat = find_best_split(
        data["X"], data["time"], data["event"], data["n_causes"], min_samples_leaf=1
    )
    # Ties break on lower-feature, lower-threshold in find_best_split; allow equality in stat.
    assert np.isclose(float(best_row["stat"]), stat, atol=1e-12)


@pytest.mark.skipif(not _rpy2_available(), reason="rpy2 not installed")
def test_rfsrc_per_feature_best_split_schema():
    data = toy_input(seed=0, n=30, n_features=3, n_causes=2)
    df = rfsrc_per_feature_best_split(data["X"], data["time"], data["event"])
    assert set(df.columns) == {"feature", "best_threshold"}
    assert set(df["feature"].unique()) == {0, 1, 2}
