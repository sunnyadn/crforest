"""Unit tests for the ``equivalence="rfsrc"`` constructor preset.

The preset bundles two flag changes (``rng_mode="rfsrc_aligned"`` and
``split_ntime=None``) and exposes ``forest.inbag_`` so users can pair
the fit with rfSRC's ``bootstrap="by.user"`` for cross-lib parity. These
tests pin the resolution + validation behavior; the empirical
equivalence quality (~0.01 cross_p95_cif on hd) is validated by
``validation/alignment/mode_vs_perf_aligned.py``.
"""

from __future__ import annotations

import numpy as np
import pytest

from comprisk import CompetingRiskForest


def _toy_data(n=120, p=4, seed=0):
    rng = np.random.RandomState(seed)
    X = rng.randn(n, p)
    time = rng.uniform(0.1, 10.0, n)
    event = rng.randint(0, 2, n).astype(np.int64)
    return X, time, event


def test_default_resolves_to_numpy_and_no_inbag():
    X, t, e = _toy_data()
    f = CompetingRiskForest(n_estimators=5, random_state=42).fit(X, t, e)
    assert f._rng_mode_eff_ == "numpy"
    assert f._split_ntime_eff_ == 10
    assert f.inbag_ is None
    # Public attrs untouched (sklearn convention)
    assert f.rng_mode == "numpy"
    assert f.split_ntime == 10
    assert f.equivalence is None


def test_rfsrc_preset_resolves_flags_and_exposes_inbag():
    X, t, e = _toy_data()
    n = X.shape[0]
    f = CompetingRiskForest(
        n_estimators=5,
        random_state=42,
        equivalence="rfsrc",
    ).fit(X, t, e)
    assert f._rng_mode_eff_ == "rfsrc_aligned"
    assert f._split_ntime_eff_ is None
    assert f.inbag_ is not None
    assert f.inbag_.shape == (n, 5)
    assert f.inbag_.dtype == np.int32
    # Per-tree counts sum to n (with-replacement bootstrap)
    assert np.all(f.inbag_.sum(axis=0) == n)


def test_rfsrc_preset_inbag_matches_legacy_helper():
    """Bootstrap stream is unchanged by the preset — inbag_ must match the
    pre-existing standalone helper used by the alignment validation harness."""
    from validation.alignment.bootstrap_aligned_spike import _comprisk_inbag_counts

    X, t, e = _toy_data()
    n, ntree, seed = X.shape[0], 8, 42
    f = CompetingRiskForest(
        n_estimators=ntree,
        random_state=seed,
        equivalence="rfsrc",
    ).fit(X, t, e)
    expected = _comprisk_inbag_counts(n, ntree, seed)
    np.testing.assert_array_equal(f.inbag_, expected)


def test_rfsrc_preset_no_inbag_when_bootstrap_false():
    X, t, e = _toy_data()
    f = CompetingRiskForest(
        n_estimators=3,
        random_state=42,
        equivalence="rfsrc",
        bootstrap=False,
    ).fit(X, t, e)
    assert f.inbag_ is None


def test_rfsrc_preset_does_not_mutate_public_attrs():
    """Constructor invariants must survive .fit() so a second .fit() with the
    same hyperparameters reproduces the same result."""
    X, t, e = _toy_data()
    f = CompetingRiskForest(
        n_estimators=3,
        random_state=42,
        equivalence="rfsrc",
    ).fit(X, t, e)
    assert f.rng_mode == "numpy"
    assert f.split_ntime == 10
    assert f.equivalence == "rfsrc"


def test_unknown_equivalence_value_raises():
    X, t, e = _toy_data()
    with pytest.raises(ValueError, match="equivalence"):
        CompetingRiskForest(
            n_estimators=3,
            random_state=42,
            equivalence="randomforest",
        ).fit(X, t, e)


def test_rfsrc_preset_conflicting_split_ntime_raises():
    """Any explicit split_ntime != default and != None must raise — pick a
    value guaranteed never to be the default."""
    X, t, e = _toy_data()
    from comprisk.forest import DEFAULT_SPLIT_NTIME

    bogus = DEFAULT_SPLIT_NTIME + 7
    with pytest.raises(ValueError, match="split_ntime"):
        CompetingRiskForest(
            n_estimators=3,
            random_state=42,
            equivalence="rfsrc",
            split_ntime=bogus,
        ).fit(X, t, e)


def test_rfsrc_preset_explicit_rfsrc_aligned_rng_is_compatible():
    """User who already passed rng_mode='rfsrc_aligned' explicitly should not
    be punished — the preset agrees with that choice."""
    X, t, e = _toy_data()
    f = CompetingRiskForest(
        n_estimators=3,
        random_state=42,
        equivalence="rfsrc",
        rng_mode="rfsrc_aligned",
    ).fit(X, t, e)
    assert f._rng_mode_eff_ == "rfsrc_aligned"


def test_rfsrc_preset_requires_random_state():
    """rfsrc_aligned RNG path requires explicit random_state; preset inherits this."""
    X, t, e = _toy_data()
    with pytest.raises(ValueError, match="random_state"):
        CompetingRiskForest(n_estimators=3, equivalence="rfsrc").fit(X, t, e)
