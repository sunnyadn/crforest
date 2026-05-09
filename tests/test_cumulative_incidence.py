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
