"""Tests for ``comprisk.FineGrayRegression``.

The cmprsk-parity tests load reference fits from ``tests/fixtures/`` that
were produced by ``tests/cross_check_cmprsk.R``. Re-generate via::

    Rscript tests/cross_check_cmprsk.R
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from comprisk import FineGrayRegression, Surv

FIXTURES_DIR = Path(__file__).parent / "fixtures"


def _load_fixture(name: str, cov_cols: list[str], event_col: str = "event"):
    df = pd.read_csv(FIXTURES_DIR / f"cmprsk_{name}_data.csv")
    fit_df = pd.read_csv(FIXTURES_DIR / f"cmprsk_{name}_fit.csv")
    coef_df = fit_df[~fit_df.feature.str.startswith("__")].set_index("feature")
    time = df["time"].to_numpy(dtype=float)
    event = df[event_col].to_numpy(dtype=int)
    X = df[cov_cols].to_numpy(dtype=float)
    cmprsk_ll = float(fit_df.loc[fit_df.feature == "__loglik__", "coef"].iloc[0])
    return X, time, event, coef_df, cmprsk_ll


@pytest.mark.parametrize(
    ("name", "cov_cols", "event_col"),
    [
        ("synth", ["x1", "x2", "x3"], "event"),
        ("pbc", ["age", "edema", "bili", "albumin", "protime", "stage"], "event"),
        ("follic", ["age", "hgb", "clinstg", "ch"], "status"),
    ],
)
def test_coef_matches_cmprsk_to_1e_6(name, cov_cols, event_col):
    """β̂ must match cmprsk to 1e-3 (SUN-45 acceptance); we observe 1e-6."""
    X, time, event, coef_df, cmprsk_ll = _load_fixture(name, cov_cols, event_col)
    fg = FineGrayRegression().fit(X, time=time, event=event)
    cmprsk_coef = np.asarray([coef_df.loc[c, "coef"] for c in cov_cols])
    assert np.allclose(fg.coef_, cmprsk_coef, atol=1e-6), (
        f"{name}: cmprsk={cmprsk_coef}, comprisk={fg.coef_}"
    )
    # Log-likelihood agreement is also bit-identical in practice.
    assert abs(fg.log_likelihood_ - cmprsk_ll) < 1e-6


@pytest.mark.parametrize(
    ("name", "cov_cols", "event_col"),
    [
        ("synth", ["x1", "x2", "x3"], "event"),
        ("pbc", ["age", "edema", "bili", "albumin", "protime", "stage"], "event"),
        ("follic", ["age", "hgb", "clinstg", "ch"], "status"),
    ],
)
def test_robust_se_matches_cmprsk_to_3_digits(name, cov_cols, event_col):
    """Cluster-robust sandwich SE matches cmprsk's IPCW-sandwich to 1e-3.

    Per Therneau-survival::finegray docs citing Geskus 2011 simulations.
    """
    X, time, event, coef_df, _ = _load_fixture(name, cov_cols, event_col)
    fg = FineGrayRegression(robust_se=True).fit(X, time=time, event=event)
    cmprsk_se = np.asarray([coef_df.loc[c, "se"] for c in cov_cols])
    assert np.allclose(fg.se_, cmprsk_se, atol=1e-3), (
        f"{name}: cmprsk_se={cmprsk_se}, comprisk_se={fg.se_}"
    )


def test_surv_y_input_equivalent_to_legacy():
    rng = np.random.default_rng(7)
    n = 300
    X = rng.normal(size=(n, 4))
    time = rng.exponential(1.0, size=n) + 0.1
    event = rng.choice([0, 1, 2], size=n, p=[0.3, 0.5, 0.2])

    fg_legacy = FineGrayRegression().fit(X, time=time, event=event)
    y = Surv.from_arrays(event=event, time=time)
    fg_y = FineGrayRegression().fit(X, y)
    np.testing.assert_allclose(fg_legacy.coef_, fg_y.coef_, atol=1e-12)


def test_predict_returns_linear_predictor():
    rng = np.random.default_rng(0)
    n = 100
    X = rng.normal(size=(n, 3))
    time = rng.exponential(1.0, size=n) + 0.1
    event = rng.choice([0, 1, 2], size=n, p=[0.4, 0.4, 0.2])
    fg = FineGrayRegression().fit(X, time=time, event=event)
    eta = fg.predict(X)
    np.testing.assert_allclose(eta, X @ fg.coef_, atol=1e-15)


def test_predict_cumulative_incidence_shape_and_bounds():
    rng = np.random.default_rng(0)
    n = 200
    X = rng.normal(size=(n, 3))
    time = rng.exponential(1.0, size=n) + 0.1
    event = rng.choice([0, 1, 2], size=n, p=[0.3, 0.5, 0.2])
    fg = FineGrayRegression().fit(X, time=time, event=event)

    F_default = fg.predict_cumulative_incidence(X[:5])
    assert F_default.shape == (5, fg._state.baseline_times.shape[0])
    assert np.all((F_default >= 0) & (F_default <= 1))
    assert np.all(np.diff(F_default, axis=1) >= -1e-12), "F(t|x) should be non-decreasing"

    times = np.linspace(0.1, time.max(), 20)
    F_grid = fg.predict_cumulative_incidence(X[:5], times=times)
    assert F_grid.shape == (5, 20)
    assert np.all((F_grid >= 0) & (F_grid <= 1))


def test_no_cause_events_raises():
    rng = np.random.default_rng(0)
    n = 100
    X = rng.normal(size=(n, 2))
    time = rng.exponential(1.0, size=n) + 0.1
    # All censored or competing-cause -- no cause-1 events.
    event = rng.choice([0, 2], size=n, p=[0.6, 0.4])
    with pytest.raises(ValueError, match="cause"):
        FineGrayRegression(cause=1).fit(X, time=time, event=event)


def test_mismatched_lengths_raises():
    X = np.zeros((10, 3))
    with pytest.raises(ValueError, match="length"):
        FineGrayRegression().fit(X, time=np.zeros(11), event=np.zeros(10))


def test_cause_kwarg_selects_other_event():
    """`cause=2` should fit Fine-Gray on the competing event as cause-of-interest."""
    rng = np.random.default_rng(0)
    n = 600
    X = rng.normal(size=(n, 2))
    time = rng.exponential(1.0, size=n) + 0.1
    event = rng.choice([0, 1, 2], size=n, p=[0.3, 0.4, 0.3])
    fg1 = FineGrayRegression(cause=1).fit(X, time=time, event=event)
    fg2 = FineGrayRegression(cause=2).fit(X, time=time, event=event)
    # Distinct fits when both causes have events.
    assert not np.allclose(fg1.coef_, fg2.coef_)


def test_n_iter_within_default_cap():
    rng = np.random.default_rng(0)
    n = 400
    X = rng.normal(size=(n, 3))
    time = rng.exponential(1.0, size=n) + 0.1
    event = rng.choice([0, 1, 2], size=n, p=[0.3, 0.5, 0.2])
    fg = FineGrayRegression(max_iter=10).fit(X, time=time, event=event)
    assert fg.n_iter_ <= 10
    assert fg.converged_


def test_cengroup_path_runs():
    rng = np.random.default_rng(0)
    n = 400
    X = rng.normal(size=(n, 3))
    time = rng.exponential(1.0, size=n) + 0.1
    event = rng.choice([0, 1, 2], size=n, p=[0.3, 0.5, 0.2])
    cg = rng.choice([0, 1], size=n)
    fg = FineGrayRegression().fit(X, time=time, event=event, cengroup=cg)
    fg_no_cg = FineGrayRegression().fit(X, time=time, event=event)
    # Different cengroup yields different fit (KM-of-censoring per stratum).
    assert not np.allclose(fg.coef_, fg_no_cg.coef_, atol=1e-8)
