"""Tests for the `nsplit` parameter on CompetingRiskForest."""

import numpy as np
import pytest
from validation.datasets import load as load_dataset

from crforest.forest import CompetingRiskForest
from tests._tree_walkers import walk_tree


def test_nsplit_none_resolves_to_10_in_default_mode():
    """mode='default', nsplit=None -> resolved_nsplit == 10 after fit."""
    X, time, event = load_dataset("pbc")
    f = CompetingRiskForest(n_estimators=2, mode="default", random_state=0).fit(X, time, event)
    assert f._resolved_nsplit_ == 10


def test_nsplit_none_resolves_to_0_in_reference_mode():
    """mode='reference', nsplit=None -> resolved_nsplit == 0 after fit."""
    X, time, event = load_dataset("pbc")
    f = CompetingRiskForest(n_estimators=2, mode="reference", random_state=0).fit(X, time, event)
    assert f._resolved_nsplit_ == 0


def test_nsplit_explicit_override_in_reference_mode():
    """mode='reference', nsplit=5 honors the override."""
    X, time, event = load_dataset("pbc")
    f = CompetingRiskForest(
        n_estimators=2,
        mode="reference",
        nsplit=5,
        random_state=0,
    ).fit(X, time, event)
    assert f._resolved_nsplit_ == 5


def test_nsplit_zero_in_default_mode_honored():
    """mode='default', nsplit=0 honors the explicit opt-out."""
    X, time, event = load_dataset("pbc")
    f = CompetingRiskForest(
        n_estimators=2,
        mode="default",
        nsplit=0,
        random_state=0,
    ).fit(X, time, event)
    assert f._resolved_nsplit_ == 0


def test_nsplit_negative_raises():
    """Negative nsplit is rejected at fit()."""
    X, time, event = load_dataset("pbc")
    with pytest.raises(ValueError, match="nsplit must be >= 0"):
        CompetingRiskForest(n_estimators=2, nsplit=-1, random_state=0).fit(X, time, event)


def test_nsplit_zero_produces_stable_tree_structure():
    """nsplit=0 in default mode is deterministic given random_state."""
    X, time, event = load_dataset("pbc")
    f1 = CompetingRiskForest(
        n_estimators=3,
        mode="default",
        nsplit=0,
        random_state=42,
    ).fit(X, time, event)
    f2 = CompetingRiskForest(
        n_estimators=3,
        mode="default",
        nsplit=0,
        random_state=42,
    ).fit(X, time, event)

    for t1, t2 in zip(f1.trees_, f2.trees_, strict=True):
        assert walk_tree(t1) == walk_tree(t2)


def test_nsplit_10_smoke_default_mode():
    """nsplit=10 in default mode fits successfully and produces valid CIF."""
    X, time, event = load_dataset("pbc")
    f = CompetingRiskForest(
        n_estimators=5,
        mode="default",
        nsplit=10,
        random_state=0,
    ).fit(X, time, event)
    cif = f.predict_cif(X)
    assert cif.shape[0] == X.shape[0]
    assert cif.shape[1] == f.n_causes_
    assert np.all(cif >= 0.0)
    assert np.all(cif <= 1.0 + 1e-9)
    assert np.all(np.diff(cif, axis=2) >= -1e-9)


def test_nsplit_default_resolves_to_10_changes_tree_structure():
    """Default behaviour under mode='default' has shifted from nsplit=0 to nsplit=10.

    Sanity: a forest with nsplit=None (resolved to 10) should produce
    different trees than an explicit nsplit=0, given the same random_state.
    """
    X, time, event = load_dataset("pbc")
    f_default = CompetingRiskForest(
        n_estimators=3,
        mode="default",
        random_state=0,
    ).fit(X, time, event)
    f_ns0 = CompetingRiskForest(
        n_estimators=3,
        mode="default",
        nsplit=0,
        random_state=0,
    ).fit(X, time, event)

    trees_differ = any(
        walk_tree(a) != walk_tree(b) for a, b in zip(f_default.trees_, f_ns0.trees_, strict=True)
    )
    assert trees_differ, "nsplit=None (resolved to 10) should differ from nsplit=0 structure"
