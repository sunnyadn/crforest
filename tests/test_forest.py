"""Tests for the CompetingRiskForest ensemble class."""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from comprisk.forest import CompetingRiskForest


def _make_synthetic_cr(seed=0, n=60):
    rng = np.random.default_rng(seed)
    X = rng.uniform(size=(n, 3))
    # Time roughly anti-correlated with feature 0 (higher feat 0 -> shorter time)
    time = 10.0 - 5.0 * X[:, 0] + rng.normal(scale=0.5, size=n)
    time = np.clip(time, 0.1, None)
    event = rng.integers(0, 3, size=n)  # {0, 1, 2}
    if not np.any(event == 1):
        event[0] = 1
    if not np.any(event == 2):
        event[1] = 2
    return X, time, event


_PBC_PATH = Path(__file__).resolve().parents[1] / "validation" / "data" / "pbc.parquet"


def test_fit_then_predict_cif_shape():
    X, time, event = _make_synthetic_cr()
    f = CompetingRiskForest(n_estimators=5, random_state=0).fit(X, time, event)
    cif = f.predict_cif(X)
    assert cif.shape == (len(X), 2, len(f.unique_times_))


def test_cif_values_in_unit_interval():
    X, time, event = _make_synthetic_cr()
    f = CompetingRiskForest(n_estimators=5, random_state=0).fit(X, time, event)
    cif = f.predict_cif(X)
    assert np.all(cif >= 0.0)
    assert np.all(cif <= 1.0 + 1e-9)


def test_cif_monotone_nondecreasing_in_time():
    X, time, event = _make_synthetic_cr()
    f = CompetingRiskForest(n_estimators=5, random_state=0).fit(X, time, event)
    cif = f.predict_cif(X)
    assert np.all(np.diff(cif, axis=2) >= -1e-9)


def test_deterministic_under_same_seed():
    X, time, event = _make_synthetic_cr()
    c1 = CompetingRiskForest(n_estimators=8, random_state=42).fit(X, time, event)
    c2 = CompetingRiskForest(n_estimators=8, random_state=42).fit(X, time, event)
    assert np.allclose(c1.predict_cif(X), c2.predict_cif(X))


def test_different_seeds_give_different_forests():
    X, time, event = _make_synthetic_cr()
    c1 = CompetingRiskForest(n_estimators=8, random_state=1).fit(X, time, event)
    c2 = CompetingRiskForest(n_estimators=8, random_state=2).fit(X, time, event)
    assert not np.allclose(c1.predict_cif(X), c2.predict_cif(X))


def test_oob_indices_complement_bootstrap_indices():
    X, time, event = _make_synthetic_cr(n=40)
    n = len(X)
    f = CompetingRiskForest(n_estimators=4, random_state=0).fit(X, time, event)

    # Each OOB set is a valid subset of [0, n) with no duplicates
    for oob in f.oob_indices_:
        assert oob.ndim == 1
        assert np.all((oob >= 0) & (oob < n))
        assert len(np.unique(oob)) == len(oob)

    # OOB must be the exact complement of bootstrap indices. Reconstruct
    # the bootstrap draws with the same controller seed to verify.
    rng = np.random.RandomState(0)
    for i in range(len(f.oob_indices_)):
        expected_bootstrap = rng.choice(n, size=n, replace=True)
        rng.randint(0, 2**31)  # consume the per-tree seed, as fit() does
        expected_oob = np.setdiff1d(np.arange(n), expected_bootstrap)
        assert np.array_equal(f.oob_indices_[i], expected_oob)


def test_predict_risk_default_kind_is_integrated_chf():
    X, time, event = _make_synthetic_cr()
    f = CompetingRiskForest(n_estimators=5, random_state=0).fit(X, time, event)
    chf = f.predict_chf(X)
    risk = f.predict_risk(X, cause=1)
    assert np.allclose(risk, chf[:, 0, :].sum(axis=-1))


def test_predict_risk_kind_cif_last_returns_cif_at_last_time():
    X, time, event = _make_synthetic_cr()
    f = CompetingRiskForest(n_estimators=5, random_state=0).fit(X, time, event)
    cif = f.predict_cif(X)
    risk = f.predict_risk(X, cause=1, kind="cif_last")
    assert np.allclose(risk, cif[:, 0, -1])


def test_predict_risk_two_kinds_produce_different_rankings():
    """CIF[last] vs integrated_chf rank subjects differently — that's the
    whole point of exposing the option (κ.exp7 finding)."""
    X, time, event = _make_synthetic_cr()
    f = CompetingRiskForest(n_estimators=10, random_state=0).fit(X, time, event)
    r_cif = f.predict_risk(X, cause=1, kind="cif_last")
    r_chf = f.predict_risk(X, cause=1, kind="integrated_chf")
    # Spearman != 1 → different rankings (we expect strong but imperfect correlation)
    rank_cif = np.argsort(np.argsort(r_cif))
    rank_chf = np.argsort(np.argsort(r_chf))
    assert not np.array_equal(rank_cif, rank_chf)


def test_predict_risk_invalid_kind_raises():
    X, time, event = _make_synthetic_cr()
    f = CompetingRiskForest(n_estimators=5, random_state=0).fit(X, time, event)
    with pytest.raises(ValueError, match="kind must be"):
        f.predict_risk(X, cause=1, kind="bogus")


def test_score_forwards_kind_parameter():
    X, time, event = _make_synthetic_cr()
    f = CompetingRiskForest(n_estimators=5, random_state=0).fit(X, time, event)
    c_default = f.score(X, time, event, cause=1)
    c_cif_last = f.score(X, time, event, cause=1, kind="cif_last")
    assert 0.0 <= c_default <= 1.0
    assert 0.0 <= c_cif_last <= 1.0


def test_score_returns_float_in_unit_interval():
    X, time, event = _make_synthetic_cr()
    f = CompetingRiskForest(n_estimators=5, random_state=0).fit(X, time, event)
    c = f.score(X, time, event)
    assert isinstance(c, float)
    assert 0.0 <= c <= 1.0


def test_score_accepts_cause_parameter():
    X, time, event = _make_synthetic_cr()
    f = CompetingRiskForest(n_estimators=5, random_state=0).fit(X, time, event)
    c1 = f.score(X, time, event, cause=1)
    c2 = f.score(X, time, event, cause=2)
    assert 0.0 <= c1 <= 1.0
    assert 0.0 <= c2 <= 1.0
    # Default still uses cause=1
    assert f.score(X, time, event) == c1


def test_predict_with_times_interpolates_step_function():
    X, time, event = _make_synthetic_cr()
    f = CompetingRiskForest(n_estimators=5, random_state=0).fit(X, time, event)
    custom = np.array([0.1, 5.0, 100.0])
    cif = f.predict_cif(X, times=custom)
    assert cif.shape == (len(X), 2, 3)
    # CIF at t=0.1 (before any event) should be 0 or near-0
    assert np.all(cif[:, :, 0] <= cif[:, :, 1])
    # CIF at t=100 should equal CIF at last training time (step function)
    cif_full = f.predict_cif(X)
    assert np.allclose(cif[:, :, -1], cif_full[:, :, -1])


def _expected_step_interp(full, ut, times):
    idx = np.searchsorted(ut, times, side="right") - 1
    out = full[:, :, np.clip(idx, 0, None)].copy()
    before = idx < 0
    if before.any():
        out[:, :, before] = 0.0
    return out


def test_predict_cif_with_times_bit_equal_to_full_then_slice():
    X, time, event = _make_synthetic_cr()
    f = CompetingRiskForest(n_estimators=8, random_state=0).fit(X, time, event)
    full = f.predict_cif(X)
    ut = f.unique_times_
    times = np.array([0.0, ut[0], ut[len(ut) // 2], ut[-1], ut[-1] + 5.0])
    sliced = f.predict_cif(X, times=times)
    np.testing.assert_array_equal(sliced, _expected_step_interp(full, ut, times))


def test_predict_chf_with_times_bit_equal_to_full_then_slice():
    X, time, event = _make_synthetic_cr()
    f = CompetingRiskForest(n_estimators=8, random_state=0).fit(X, time, event)
    full = f.predict_chf(X)
    ut = f.unique_times_
    times = np.array([0.0, ut[0], ut[len(ut) // 2], ut[-1], ut[-1] + 5.0])
    sliced = f.predict_chf(X, times=times)
    np.testing.assert_array_equal(sliced, _expected_step_interp(full, ut, times))


def test_predict_rejects_wrong_n_features():
    X, time, event = _make_synthetic_cr()
    f = CompetingRiskForest(n_estimators=3, random_state=0).fit(X, time, event)
    with pytest.raises(ValueError, match="n_features"):
        f.predict_cif(X[:, :2])


def test_max_features_sqrt_default():
    X, time, event = _make_synthetic_cr()
    f = CompetingRiskForest(n_estimators=3, random_state=0, max_features="sqrt").fit(X, time, event)
    # Just check it doesn't error and still produces valid output
    assert f.predict_cif(X).shape[1] == 2


def test_no_bootstrap_uses_full_sample():
    X, time, event = _make_synthetic_cr()
    f = CompetingRiskForest(n_estimators=3, random_state=0, bootstrap=False).fit(X, time, event)
    for oob in f.oob_indices_:
        assert len(oob) == 0


def test_sklearn_get_set_params_round_trip():
    f = CompetingRiskForest(n_estimators=7, max_depth=4)
    p = f.get_params()
    assert p["n_estimators"] == 7
    assert p["max_depth"] == 4
    f.set_params(n_estimators=10)
    assert f.n_estimators == 10


def test_default_mode_fit_sets_bin_edges_and_time_grid():
    X, time, event = _make_synthetic_cr()
    f = CompetingRiskForest(n_estimators=3, mode="default", random_state=0).fit(X, time, event)
    assert hasattr(f, "bin_edges_")
    assert len(f.bin_edges_) == X.shape[1]
    assert hasattr(f, "time_grid_")
    assert f.time_grid_.ndim == 1
    np.testing.assert_array_equal(f.unique_times_, f.time_grid_)


def test_default_mode_uses_flat_trees():
    from comprisk._tree_flat import FlatTree

    X, time, event = _make_synthetic_cr()
    f = CompetingRiskForest(n_estimators=3, mode="default", random_state=0).fit(X, time, event)
    assert all(isinstance(t, FlatTree) for t in f.trees_)


def test_reference_mode_still_works():
    from comprisk._tree import RefTreeNode

    X, time, event = _make_synthetic_cr()
    f = CompetingRiskForest(n_estimators=3, mode="reference", random_state=0).fit(X, time, event)
    assert all(isinstance(t, RefTreeNode) for t in f.trees_)


def test_invalid_mode_raises():
    X, time, event = _make_synthetic_cr()
    with pytest.raises(ValueError, match="mode"):
        CompetingRiskForest(mode="invalid").fit(X, time, event)


def test_forest_splitrule_logrank_fit_and_predict_default_mode():
    import numpy as np

    from comprisk import CompetingRiskForest

    rng = np.random.default_rng(7)
    n = 200
    X = rng.standard_normal((n, 4))
    time = rng.uniform(0.1, 5.0, size=n)
    event = rng.integers(0, 3, size=n).astype(np.int64)

    forest = CompetingRiskForest(
        n_estimators=5,
        max_depth=4,
        random_state=0,
        splitrule="logrank",
        cause=1,
        n_jobs=1,
    )
    forest.fit(X, time, event)
    cif = forest.predict_cif(X)
    assert cif.shape == (n, 2, len(forest.unique_times_))


def test_forest_splitrule_unknown_raises():
    import numpy as np
    import pytest

    from comprisk import CompetingRiskForest

    with pytest.raises(ValueError, match="splitrule"):
        CompetingRiskForest(splitrule="bogus").fit(
            np.zeros((10, 2)), np.arange(10.0), np.zeros(10, dtype=np.int64)
        )


def test_forest_logrank_cause_weights_validates_length():
    import numpy as np
    import pytest

    from comprisk import CompetingRiskForest

    forest = CompetingRiskForest(splitrule="logrank", cause_weights=[1.0], mode="reference")
    with pytest.raises(ValueError, match="cause_weights"):
        forest.fit(
            np.zeros((20, 2)),
            np.arange(1.0, 21.0),
            np.array([1, 2] * 10, dtype=np.int64),  # n_causes=2
        )


def test_forest_logrank_cause_weights_rejected_in_default_mode():
    import numpy as np
    import pytest

    from comprisk import CompetingRiskForest

    forest = CompetingRiskForest(splitrule="logrank", cause_weights=[0.6, 0.4], mode="default")
    with pytest.raises(NotImplementedError, match="cause_weights"):
        forest.fit(
            np.zeros((20, 2)),
            np.arange(1.0, 21.0),
            np.array([1, 2] * 10, dtype=np.int64),
        )


def test_forest_logrank_cause_weights_accepted_in_reference_mode():
    import numpy as np

    from comprisk import CompetingRiskForest

    rng = np.random.default_rng(8)
    n = 80
    X = rng.standard_normal((n, 3))
    time = rng.uniform(0.1, 5.0, size=n)
    event = rng.integers(0, 3, size=n).astype(np.int64)

    forest = CompetingRiskForest(
        n_estimators=3,
        max_depth=3,
        random_state=0,
        splitrule="logrank",
        cause_weights=[0.6, 0.4],
        mode="reference",
        n_jobs=1,
    )
    forest.fit(X, time, event)  # should not raise
    cif = forest.predict_cif(X)
    assert cif.shape == (n, 2, len(forest.unique_times_))


@pytest.mark.skipif(not _PBC_PATH.exists(), reason="PBC fixture not present")
def test_fits_pbc_end_to_end():
    df = pd.read_parquet(_PBC_PATH)
    time = df["time"].to_numpy(dtype=np.float64)
    event = df["event"].to_numpy(dtype=np.int64)
    X = df.drop(columns=["time", "event"]).to_numpy(dtype=np.float64)

    f = CompetingRiskForest(n_estimators=20, random_state=0, max_depth=10).fit(X, time, event)
    cif = f.predict_cif(X)
    assert cif.shape == (len(X), f.n_causes_, len(f.unique_times_))
    assert np.all(cif >= 0.0)
    assert np.all(cif <= 1.0 + 1e-9)
    assert np.all(np.diff(cif, axis=2) >= -1e-9)

    # Cause-1 C-index on training data should beat random by a clear margin
    c = f.score(X, time, event)
    assert c > 0.6


def test_split_ntime_default_and_none() -> None:
    """CompetingRiskForest exposes split_ntime; default is 10; None and ints are accepted."""
    f = CompetingRiskForest()
    assert f.split_ntime == 10
    f2 = CompetingRiskForest(split_ntime=None)
    assert f2.split_ntime is None
    f3 = CompetingRiskForest(split_ntime=30)
    assert f3.split_ntime == 30
