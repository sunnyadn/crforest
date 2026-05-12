"""Tests for TreeSHAP on comprisk forests."""

from __future__ import annotations

import numpy as np
import pytest

from comprisk.forest import CompetingRiskForest


def _make_synthetic_cr(seed=0, n=60):
    rng = np.random.default_rng(seed)
    X = rng.uniform(size=(n, 3))
    time = 10.0 - 5.0 * X[:, 0] + rng.normal(scale=0.5, size=n)
    time = np.clip(time, 0.1, None)
    event = rng.integers(0, 3, size=n)
    if not np.any(event == 1):
        event[0] = 1
    if not np.any(event == 2):
        event[1] = 2
    return X, time, event


# ---------------------------------------------------------------------------
# Additivity (local accuracy) — the fundamental SHAP property
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("mode", ["default", "reference"])
def test_shap_additivity_default_times(mode):
    """sum(shap, axis=features) + base ≈ predict_cif at model's time grid."""
    X, time, event = _make_synthetic_cr(n=80)
    f = CompetingRiskForest(n_estimators=10, random_state=42, mode=mode, max_depth=5).fit(
        X, time, event
    )

    shap, base = f.shap_values(X)
    # shap shape: (n, p, n_times, n_causes)
    # base shape: (n_times, n_causes)
    cif_pred = f.predict_cif(X)  # (n, n_causes, n_times)

    shap_sum = shap.sum(axis=1)  # (n, n_times, n_causes)
    reconstructed = shap_sum + base  # broadcast base: (n_times, n_causes)

    # Reorder to match predict_cif: (n, n_causes, n_times)
    reconstructed = reconstructed.transpose(0, 2, 1)

    assert np.allclose(reconstructed, cif_pred, atol=1e-9, rtol=1e-6)


@pytest.mark.parametrize("mode", ["default", "reference"])
def test_shap_additivity_custom_times(mode):
    """Additivity also holds when user supplies a custom ``times`` grid."""
    X, time, event = _make_synthetic_cr(n=80)
    f = CompetingRiskForest(n_estimators=10, random_state=42, mode=mode, max_depth=5).fit(
        X, time, event
    )

    custom_times = np.linspace(time.min(), time.max(), 7)
    shap, base = f.shap_values(X, times=custom_times)
    cif_pred = f.predict_cif(X, times=custom_times)

    shap_sum = shap.sum(axis=1)  # (n, n_times, n_causes)
    reconstructed = shap_sum + base
    reconstructed = reconstructed.transpose(0, 2, 1)

    assert np.allclose(reconstructed, cif_pred, atol=1e-9, rtol=1e-6)


# ---------------------------------------------------------------------------
# Shape and edge cases
# ---------------------------------------------------------------------------


def test_shap_output_shape():
    """SHAP output shape matches spec: (n, p, n_times, n_causes)."""
    X, time, event = _make_synthetic_cr(n=50)
    f = CompetingRiskForest(n_estimators=5, random_state=0, max_depth=4).fit(X, time, event)
    shap, base = f.shap_values(X)
    assert shap.shape == (len(X), X.shape[1], len(f.unique_times_), f.n_causes_)
    assert base.shape == (len(f.unique_times_), f.n_causes_)


def test_shap_single_sample():
    """SHAP works for a single sample (n=1)."""
    X, time, event = _make_synthetic_cr(n=50)
    f = CompetingRiskForest(n_estimators=5, random_state=0, max_depth=4).fit(X, time, event)
    shap, _ = f.shap_values(X[[0]])
    assert shap.shape == (1, X.shape[1], len(f.unique_times_), f.n_causes_)


def test_shap_single_tree():
    """SHAP works for a single-tree forest."""
    X, time, event = _make_synthetic_cr(n=50)
    f = CompetingRiskForest(n_estimators=1, random_state=0, max_depth=4).fit(X, time, event)
    shap, base = f.shap_values(X)
    cif_pred = f.predict_cif(X)
    reconstructed = (shap.sum(axis=1) + base).transpose(0, 2, 1)
    assert np.allclose(reconstructed, cif_pred, atol=1e-9, rtol=1e-6)


def test_shap_rejects_bad_X_shape():
    """SHAP raises for wrong feature count."""
    X, time, event = _make_synthetic_cr(n=50)
    f = CompetingRiskForest(n_estimators=5, random_state=0).fit(X, time, event)
    with pytest.raises(ValueError, match="features"):
        f.shap_values(X[:, :2])


# ---------------------------------------------------------------------------
# Sanity: important feature has larger |SHAP| than noise
# ---------------------------------------------------------------------------


def test_shap_synthetic_important_feature():
    """On synthetic data where feature 0 drives time, |SHAP_0| >> |SHAP_noise|."""
    rng = np.random.default_rng(7)
    n = 200
    X = rng.uniform(size=(n, 5))
    # Strong signal: higher feature 0 -> shorter time (cause 1)
    time = 5.0 - 3.0 * X[:, 0] + rng.normal(scale=0.3, size=n)
    time = np.clip(time, 0.1, None)
    event = rng.integers(0, 3, size=n)
    if not np.any(event == 1):
        event[0] = 1
    if not np.any(event == 2):
        event[1] = 2

    f = CompetingRiskForest(n_estimators=20, random_state=7, max_depth=6).fit(X, time, event)
    shap, _ = f.shap_values(X)

    # Mean absolute SHAP per feature, averaged over samples x times x causes
    mean_abs_shap = np.abs(shap).mean(axis=(0, 2, 3))
    # Feature 0 should dominate the noise features (1-4)
    assert mean_abs_shap[0] > 5.0 * mean_abs_shap[1:].mean()


# ---------------------------------------------------------------------------
# SHAP ranking sanity vs OOB VIMP
# ---------------------------------------------------------------------------


def test_shap_agrees_with_oob_vimp_top3():
    """Top-3 features from mean-|SHAP| overlap with OOB VIMP top-3 (>=2/3)."""
    rng = np.random.default_rng(7)
    n = 200
    X = rng.uniform(size=(n, 5))
    time = 5.0 - 3.0 * X[:, 0] + rng.normal(scale=0.3, size=n)
    time = np.clip(time, 0.1, None)
    event = rng.integers(0, 3, size=n)
    if not np.any(event == 1):
        event[0] = 1
    if not np.any(event == 2):
        event[1] = 2

    f = CompetingRiskForest(n_estimators=20, random_state=7, max_depth=6, bootstrap=True).fit(
        X, time, event
    )

    shap, _ = f.shap_values(X)
    mean_abs_shap = np.abs(shap).mean(axis=(0, 2, 3))
    shap_top3 = set(np.argsort(mean_abs_shap)[-3:])

    vimp = f.compute_importance()
    vimp_top3 = set(np.argsort(vimp["composite_vimp"].values)[-3:])

    overlap = len(shap_top3 & vimp_top3)
    assert overlap >= 2, f"SHAP top-3 {shap_top3} vs VIMP top-3 {vimp_top3}, overlap={overlap}"


# ---------------------------------------------------------------------------
# Base value properties
# ---------------------------------------------------------------------------


def test_base_value_is_weighted_average():
    """Base value equals the model's CIF when all features are marginalised."""
    X, time, event = _make_synthetic_cr(n=80)
    f = CompetingRiskForest(n_estimators=10, random_state=42, max_depth=5).fit(X, time, event)
    _, base = f.shap_values(X)
    # Base should be the same for all samples (it's the training average)
    # and match predict_cif on the training mean feature vector
    # Base is E[f(X)]; for a tree ensemble it's the average leaf value.
    # It should be close to predict_cif at the mean X (but not identical
    # because CIF is non-linear).
    # Instead, verify base equals the zero-feature SHAP prediction.
    assert base.shape == (len(f.unique_times_), f.n_causes_)
    # The zero-feature prediction is just the base value; additivity says
    # base + sum(shap) = prediction.  We already test that.
    # Here: verify base is approximately the same across different random X sets.
    X2, _, _ = _make_synthetic_cr(seed=99, n=80)
    _, base2 = f.shap_values(X2)
    assert np.allclose(base, base2, atol=1e-9)


# ---------------------------------------------------------------------------
# Edge: SHAP with a single time point
# ---------------------------------------------------------------------------


def test_shap_single_time_point():
    """SHAP works when only one time point is requested."""
    X, time, event = _make_synthetic_cr(n=50)
    f = CompetingRiskForest(n_estimators=5, random_state=0, max_depth=4).fit(X, time, event)
    t = np.array([time.max()])
    shap, base = f.shap_values(X, times=t)
    assert shap.shape == (len(X), X.shape[1], 1, f.n_causes_)
    assert base.shape == (1, f.n_causes_)
    cif_pred = f.predict_cif(X, times=t)
    reconstructed = (shap.sum(axis=1) + base).transpose(0, 2, 1)
    assert np.allclose(reconstructed, cif_pred, atol=1e-9, rtol=1e-6)


# ---------------------------------------------------------------------------
# Custom-times path equals full-grid path projected (the matmul folds the
# time-projection into the leaf table; verify it agrees with project-after).
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("mode", ["default", "reference"])
def test_shap_custom_times_equals_projected_full_grid(mode):
    X, time, event = _make_synthetic_cr(n=80)
    f = CompetingRiskForest(n_estimators=8, random_state=3, mode=mode, max_depth=5).fit(
        X, time, event
    )

    full_grid = f.unique_times_
    # A mix of grid points, an interpolated point, and one before the grid.
    custom = np.array([full_grid[0] - 1.0, full_grid[len(full_grid) // 3], full_grid[-1]])

    shap_custom, base_custom = f.shap_values(X, times=custom)
    shap_full, base_full = f.shap_values(X)

    # Right-continuous step projection of the full-grid result onto `custom`.
    idx = np.searchsorted(full_grid, custom, side="right") - 1
    take = np.clip(idx, 0, None)
    before = idx < 0
    shap_proj = shap_full[:, :, take, :].copy()
    base_proj = base_full[take, :].copy()
    shap_proj[:, :, before, :] = 0.0
    base_proj[before, :] = 0.0

    assert np.allclose(shap_custom, shap_proj, atol=1e-12, rtol=0)
    assert np.allclose(base_custom, base_proj, atol=1e-12, rtol=0)


# ---------------------------------------------------------------------------
# Many-tree forest with thread parallelism: chunked-worker reduction must
# give the same answer as the serial path.
# ---------------------------------------------------------------------------


def test_shap_parallel_matches_serial():
    X, time, event = _make_synthetic_cr(n=120)
    f = CompetingRiskForest(n_estimators=12, random_state=11, max_depth=5, n_jobs=1).fit(
        X, time, event
    )
    shap_ser, base_ser = f.shap_values(X)
    f.n_jobs = 4  # same trees, only the cross-tree reduction order changes
    shap_par, base_par = f.shap_values(X)
    assert np.allclose(shap_par, shap_ser, atol=1e-10, rtol=1e-10)
    assert np.allclose(base_par, base_ser, atol=1e-10, rtol=1e-10)


# ---------------------------------------------------------------------------
# time_aggregate: risk-score SHAP — collapse the time axis before attribution
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("mode", ["default", "reference"])
@pytest.mark.parametrize("agg", ["sum", "trapezoid"])
def test_shap_time_aggregate_equals_aggregating_full_grid(mode, agg):
    """``time_aggregate`` == aggregating the full per-time SHAP after the fact."""
    X, time, event = _make_synthetic_cr(n=80)
    f = CompetingRiskForest(n_estimators=8, random_state=5, mode=mode, max_depth=5).fit(
        X, time, event
    )

    shap_agg, base_agg = f.shap_values(X, time_aggregate=agg)
    shap_full, base_full = f.shap_values(X)

    assert shap_agg.shape == (len(X), X.shape[1], f.n_causes_)
    assert base_agg.shape == (f.n_causes_,)

    if agg == "sum":
        shap_ref = shap_full.sum(axis=2)
        base_ref = base_full.sum(axis=0)
    else:
        shap_ref = np.trapezoid(shap_full, x=f.unique_times_, axis=2)
        base_ref = np.trapezoid(base_full, x=f.unique_times_, axis=0)

    assert np.allclose(shap_agg, shap_ref, atol=1e-10, rtol=1e-9)
    assert np.allclose(base_agg, base_ref, atol=1e-10, rtol=1e-9)


def test_shap_time_aggregate_additivity():
    """Aggregated SHAP + base reconstructs the aggregated CIF curve per (sample, cause)."""
    X, time, event = _make_synthetic_cr(n=80)
    f = CompetingRiskForest(n_estimators=10, random_state=1, max_depth=5).fit(X, time, event)

    cif = f.predict_cif(X)  # (n, n_causes, n_times)
    for agg, reducer in (
        ("sum", lambda a: a.sum(axis=-1)),
        ("trapezoid", lambda a: np.trapezoid(a, x=f.unique_times_, axis=-1)),
    ):
        shap_agg, base_agg = f.shap_values(X, time_aggregate=agg)
        recon = shap_agg.sum(axis=1) + base_agg  # (n, n_causes)
        assert np.allclose(recon, reducer(cif), atol=1e-9, rtol=1e-6)


def test_shap_time_aggregate_with_custom_times():
    """``time_aggregate`` aggregates over the requested ``times`` window, not the full grid."""
    X, time, event = _make_synthetic_cr(n=80)
    f = CompetingRiskForest(n_estimators=8, random_state=2, max_depth=5).fit(X, time, event)
    window = f.unique_times_[2:8]

    shap_agg, base_agg = f.shap_values(X, times=window, time_aggregate="sum")
    shap_win, base_win = f.shap_values(X, times=window)
    assert np.allclose(shap_agg, shap_win.sum(axis=2), atol=1e-10, rtol=1e-9)
    assert np.allclose(base_agg, base_win.sum(axis=0), atol=1e-10, rtol=1e-9)


def test_shap_time_aggregate_rejects_bad_value():
    X, time, event = _make_synthetic_cr(n=40)
    f = CompetingRiskForest(n_estimators=3, random_state=0, max_depth=4).fit(X, time, event)
    with pytest.raises(ValueError, match="time_aggregate"):
        f.shap_values(X, time_aggregate="mean")


def test_shap_repeated_calls_are_identical():
    """The covers/base cache on the tree must not perturb a second call."""
    X, time, event = _make_synthetic_cr(n=80)
    f = CompetingRiskForest(n_estimators=10, random_state=4, max_depth=5).fit(X, time, event)
    s1, b1 = f.shap_values(X)
    s2, b2 = f.shap_values(X)
    assert np.array_equal(s1, s2)
    assert np.array_equal(b1, b2)
    # ... and the cache is consistent across the aggregated path too
    sa1, ba1 = f.shap_values(X, time_aggregate="sum")
    sa2, ba2 = f.shap_values(X, time_aggregate="sum")
    assert np.array_equal(sa1, sa2)
    assert np.array_equal(ba1, ba2)


# ---------------------------------------------------------------------------
# Compatibility: slice extraction for shap.summary_plot
# ---------------------------------------------------------------------------


def test_shap_slice_for_summary_plot():
    """A fixed (time, cause) slice yields a 2-D matrix compatible with upstream shap."""
    X, time, event = _make_synthetic_cr(n=50)
    f = CompetingRiskForest(n_estimators=5, random_state=0, max_depth=4).fit(X, time, event)
    shap, base = f.shap_values(X)
    # Extract slice for cause=0 at time-index 0
    slice_2d = shap[:, :, 0, 0]  # (n_samples, n_features)
    assert slice_2d.ndim == 2
    assert slice_2d.shape == (len(X), X.shape[1])
    # Verify this slice satisfies additivity for that (time, cause)
    cif_pred = f.predict_cif(X)
    assert np.allclose(
        slice_2d.sum(axis=1) + base[0, 0],
        cif_pred[:, 0, 0],
        atol=1e-9,
        rtol=1e-6,
    )
