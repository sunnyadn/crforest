"""Tests for ``comprisk.gray_test`` (cmprsk::cuminc()$Tests parity)."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from comprisk import gray_test

FIXTURES_DIR = Path(__file__).parent / "fixtures"


@pytest.mark.parametrize("cause", [1, 2])
def test_synth_grouped_matches_cmprsk(cause):
    df = pd.read_csv(FIXTURES_DIR / "cuminc_synth_data.csv")
    expected = pd.read_csv(FIXTURES_DIR / "gray_synth_fit.csv")
    R = expected[expected.cause == cause].iloc[0]
    res = gray_test(
        df["time"].to_numpy(),
        df["event"].to_numpy(),
        df["group"].to_numpy(),
        cause=cause,
    )
    assert abs(res.stat - R.stat) < 1e-10
    assert abs(res.pvalue - R.pv) < 1e-10
    assert res.df == int(R.df)


@pytest.mark.parametrize("cause", [1, 2])
def test_follic_clinstg_groups_match_cmprsk(cause):
    df = pd.read_csv(FIXTURES_DIR / "cmprsk_follic_data.csv")
    expected = pd.read_csv(FIXTURES_DIR / "gray_follic_fit.csv")
    R = expected[expected.cause == cause].iloc[0]
    res = gray_test(
        df["time"].to_numpy(),
        df["status"].to_numpy(),
        df["clinstg"].to_numpy(),
        cause=cause,
    )
    assert abs(res.stat - R.stat) < 1e-9
    assert abs(res.pvalue - R.pv) < 1e-9
    assert res.df == int(R.df)


def test_at_least_two_groups_required():
    rng = np.random.default_rng(0)
    n = 50
    time = rng.exponential(1.0, n)
    event = rng.choice([0, 1, 2], n)
    g = np.zeros(n, dtype=int)
    with pytest.raises(ValueError, match="at least 2 groups"):
        gray_test(time, event, g)


def test_three_group_df_is_two():
    rng = np.random.default_rng(7)
    n = 600
    time = rng.exponential(1.0, n)
    event = rng.choice([0, 1, 2], n, p=[0.3, 0.5, 0.2])
    g = rng.choice(["A", "B", "C"], n)
    res = gray_test(time, event, g, cause=1)
    assert res.df == 2
    assert res.score.shape == (2,)
    assert res.var.shape == (2, 2)


def test_string_groups_supported():
    rng = np.random.default_rng(0)
    n = 200
    time = rng.exponential(1.0, n)
    event = rng.choice([0, 1, 2], n)
    g = rng.choice(["control", "treatment"], n)
    res = gray_test(time, event, g, cause=1)
    assert res.n_groups == 2 and res.df == 1


def test_gray_test_no_events_at_unique_time_does_not_explode():
    """Censoring-only tied times must drain rs without crashing."""
    # Two groups, several censoring-only ties between event times.
    time = np.asarray([1.0, 2.0, 2.0, 2.0, 3.0, 4.0, 4.0, 5.0])
    event = np.asarray([1, 0, 0, 0, 1, 0, 1, 2])  # event=2 at the last
    g = np.asarray([0, 0, 1, 1, 1, 0, 0, 1])
    res = gray_test(time, event, g, cause=1)
    # Just check it produces something finite (no fixture for this one).
    assert np.isfinite(res.stat) and 0 <= res.pvalue <= 1
