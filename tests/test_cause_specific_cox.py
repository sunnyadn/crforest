"""Tests for ``comprisk.CauseSpecificCox`` (R survival::coxph parity)."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from comprisk import CauseSpecificCox

FIXTURES_DIR = Path(__file__).parent / "fixtures"


def _strip_prefix(s: str) -> str:
    return s.replace("X_mat", "")


@pytest.mark.parametrize(
    ("name", "cov_cols", "event_col"),
    [
        ("pbc", ["age", "edema", "bili", "albumin", "protime", "stage"], "event"),
        ("follic", ["age", "hgb", "clinstg", "ch"], "status"),
    ],
)
def test_csc_coef_matches_coxph_breslow(name, cov_cols, event_col):
    df = pd.read_csv(FIXTURES_DIR / f"cmprsk_{name}_data.csv")
    fit_df = pd.read_csv(FIXTURES_DIR / f"csc_{name}_fit.csv")
    coefs = fit_df[~fit_df.feature.str.startswith("__")].copy()
    coefs["feature"] = coefs["feature"].map(_strip_prefix)
    coefs = coefs.set_index("feature")

    X = df[cov_cols].to_numpy(dtype=float)
    time = df["time"].to_numpy(dtype=float)
    event = df[event_col].to_numpy(dtype=int)

    cs = CauseSpecificCox(cause=1).fit(X, time=time, event=event)
    expected = np.asarray([coefs.loc[c, "coef"] for c in cov_cols])
    assert np.allclose(cs.coef_, expected, atol=1e-6)
    expected_se = np.asarray([coefs.loc[c, "se"] for c in cov_cols])
    assert np.allclose(cs.se_, expected_se, atol=1e-6)
    cmprsk_ll = float(fit_df.loc[fit_df.feature == "__loglik__", "coef"].iloc[0])
    assert abs(cs.log_likelihood_ - cmprsk_ll) < 1e-6


def test_no_cause_events_raises():
    rng = np.random.default_rng(0)
    n = 100
    X = rng.normal(size=(n, 2))
    time = rng.exponential(1.0, size=n)
    event = rng.choice([0, 2], size=n, p=[0.6, 0.4])  # no cause-1
    with pytest.raises(ValueError, match="cause-1"):
        CauseSpecificCox(cause=1).fit(X, time=time, event=event)


def test_competing_event_treated_as_censored():
    """Subjects with competing events should be treated as censored at t_j.

    Verify by running CauseSpecificCox(cause=1) and confirming it agrees
    with a manual ``event2 := (event == 1)`` Cox PH fit.
    """
    rng = np.random.default_rng(42)
    n = 400
    X = rng.normal(size=(n, 3))
    time = rng.exponential(1.0, size=n) + 0.1
    event = rng.choice([0, 1, 2], size=n, p=[0.3, 0.4, 0.3])

    cs1 = CauseSpecificCox(cause=1).fit(X, time=time, event=event)
    # Manually censor competing events: relabel as 0.
    event_manual = np.where(event == 1, 1, 0)
    cs_manual = CauseSpecificCox(cause=1).fit(X, time=time, event=event_manual)
    np.testing.assert_allclose(cs1.coef_, cs_manual.coef_, atol=1e-12)
