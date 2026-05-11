"""Tests for ``comprisk.CumulativeIncidence`` (cmprsk::cuminc parity)."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from comprisk import CumulativeIncidence

FIXTURES_DIR = Path(__file__).parent / "fixtures"


def _load_cuminc_fixture(name: str) -> pd.DataFrame:
    return pd.read_csv(FIXTURES_DIR / f"cuminc_{name}_fit.csv")


def test_cuminc_synth_matches_cmprsk_grouped():
    """Two-group synthetic dataset; CIF + variance bit-identical to cmprsk."""
    df = pd.read_csv(FIXTURES_DIR / "cuminc_synth_data.csv")
    expected = _load_cuminc_fixture("synth")

    ci = CumulativeIncidence().fit(
        time=df["time"].to_numpy(),
        event=df["event"].to_numpy(),
        group=df["group"].to_numpy(),
    )
    est, var = ci.timepoints([0.5, 1.0, 2.0])
    keys = ci._timepoints_keys_

    for i, (g, c) in enumerate(keys):
        ek = f"{g} {c}"
        rows = expected[expected.curve == ek].sort_values("t")
        np.testing.assert_allclose(est[i], rows["est"].to_numpy(), atol=1e-9)
        np.testing.assert_allclose(var[i], rows["var"].to_numpy(), atol=1e-9)


def test_cuminc_follic_matches_cmprsk_no_groups():
    df = pd.read_csv(FIXTURES_DIR / "cmprsk_follic_data.csv")
    expected = _load_cuminc_fixture("follic")

    ci = CumulativeIncidence().fit(
        time=df["time"].to_numpy(),
        event=df["status"].to_numpy(),
    )
    est, var = ci.timepoints([2.0, 5.0, 10.0])
    keys = ci._timepoints_keys_

    for i, (_g, c) in enumerate(keys):
        # cmprsk labels single-group output as "1 {cause}"
        ek = f"1 {c}"
        rows = expected[expected.curve == ek].sort_values("t")
        np.testing.assert_allclose(est[i], rows["est"].to_numpy(), atol=1e-9)
        np.testing.assert_allclose(var[i], rows["var"].to_numpy(), atol=1e-9)


def test_cif_monotone_nondecreasing():
    rng = np.random.default_rng(0)
    n = 200
    time = rng.exponential(1.0, size=n)
    event = rng.choice([0, 1, 2], size=n, p=[0.4, 0.4, 0.2])
    ci = CumulativeIncidence().fit(time=time, event=event)
    for curve in ci.curves_.values():
        assert np.all(np.diff(curve.cif) >= -1e-12)
        assert np.all(curve.cif >= 0) and np.all(curve.cif <= 1)


def test_evaluate_outside_range():
    rng = np.random.default_rng(0)
    time = rng.exponential(1.0, size=100)
    event = rng.choice([0, 1], size=100)
    ci = CumulativeIncidence().fit(time=time, event=event)
    curve = next(iter(ci.curves_.values()))
    cif, var = curve.evaluate(np.asarray([-1.0, 1e9]))
    assert cif[0] == 0 and var[0] == 0  # below first event
    assert cif[1] == curve.cif[-1]  # above last event


def test_restricted_cause_codes_matches_full_fit_repro():
    """SUN-71: restricting ``cause_codes`` must not drop competing events
    from the at-risk dynamics. The cause-1 CIF here is exactly 2/13."""
    time = np.array(
        [533, 708, 877, 1212, 1328, 1434, 1639, 1783, 2332, 2403, 3468, 3707, 4079],
        dtype=float,
    )
    event = np.array([2, 1, 2, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1])

    restricted = CumulativeIncidence(cause_codes=[1]).fit(time=time, event=event)
    full = CumulativeIncidence().fit(time=time, event=event)

    t = np.array([1825.0])
    r_cif, r_var = restricted.curves_[(None, 1)].evaluate(t)
    f_cif, f_var = full.curves_[(None, 1)].evaluate(t)
    np.testing.assert_allclose(r_cif, f_cif, atol=1e-12)
    np.testing.assert_allclose(r_var, f_var, atol=1e-12)
    np.testing.assert_allclose(r_cif, [2.0 / 13.0], atol=1e-12)


def test_restricted_cause_codes_matches_full_fit_random():
    """SUN-71: point estimate *and* variance under ``cause_codes=[k]`` agree
    with reading cause ``k`` off an all-cause fit, on noisier data."""
    rng = np.random.default_rng(7)
    n = 300
    time = rng.exponential(2.0, size=n)
    event = rng.choice([0, 1, 2, 3], size=n, p=[0.3, 0.3, 0.25, 0.15])
    grid = np.array([0.5, 1.0, 2.0, 4.0])

    full = CumulativeIncidence().fit(time=time, event=event)
    for k in (1, 2, 3):
        restricted = CumulativeIncidence(cause_codes=[k]).fit(time=time, event=event)
        r_cif, r_var = restricted.curves_[(None, k)].evaluate(grid)
        f_cif, f_var = full.curves_[(None, k)].evaluate(grid)
        np.testing.assert_allclose(r_cif, f_cif, atol=1e-12)
        np.testing.assert_allclose(r_var, f_var, atol=1e-12)


def test_restricted_cause_not_present_is_empty():
    """A requested cause with no events in the data yields an empty curve."""
    time = np.array([1.0, 2.0, 3.0, 4.0], dtype=float)
    event = np.array([1, 0, 2, 1])
    ci = CumulativeIncidence(cause_codes=[3]).fit(time=time, event=event)
    curve = ci.curves_[(None, 3)]
    assert curve.times.size == 0 and curve.cif.size == 0


def test_restricted_cause_codes_grouped_matches_full():
    """SUN-71 with stratification: per-(group, cause) agreement holds."""
    rng = np.random.default_rng(11)
    n = 240
    time = rng.exponential(1.5, size=n)
    event = rng.choice([0, 1, 2], size=n, p=[0.4, 0.35, 0.25])
    group = rng.choice(["a", "b"], size=n)
    grid = np.array([0.5, 1.0, 2.0])

    full = CumulativeIncidence().fit(time=time, event=event, group=group)
    restricted = CumulativeIncidence(cause_codes=[1]).fit(time=time, event=event, group=group)
    for g in ("a", "b"):
        r_cif, r_var = restricted.curves_[(g, 1)].evaluate(grid)
        f_cif, f_var = full.curves_[(g, 1)].evaluate(grid)
        np.testing.assert_allclose(r_cif, f_cif, atol=1e-12)
        np.testing.assert_allclose(r_var, f_var, atol=1e-12)
