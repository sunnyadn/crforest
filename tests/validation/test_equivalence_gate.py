"""Unit tests for validation.alignment.equivalence_gate (pure helpers)."""

from __future__ import annotations

import numpy as np
import pytest


def test_build_reference_grid_sorted_unique_event_times():
    from validation.alignment.equivalence_gate import build_reference_grid

    time = np.array([1.0, 2.0, 2.0, 3.0, 4.0, 5.0])
    event = np.array([1, 0, 1, 2, 0, 1])  # event==0 censored
    grid = build_reference_grid(time, event)

    # Only rows with event > 0 contribute: times 1.0, 2.0, 3.0, 5.0
    np.testing.assert_array_equal(grid, np.array([1.0, 2.0, 3.0, 5.0]))


def test_build_reference_grid_empty_events_raises():
    from validation.alignment.equivalence_gate import build_reference_grid

    time = np.array([1.0, 2.0])
    event = np.array([0, 0])  # all censored
    with pytest.raises(ValueError, match="no event times"):
        build_reference_grid(time, event)


def test_eval_on_ref_grid_step_function_semantics():
    from validation.alignment.equivalence_gate import eval_on_ref_grid

    # Lib native grid at event times 2.0 and 4.0 with CIF jumping 0 -> 0.3 -> 0.7.
    # Single test sample: shape (1, n_native) = (1, 2).
    native_grid = np.array([2.0, 4.0])
    cif_native = np.array([[0.3, 0.7]])  # step function: cif(t) = value at grid[idx_right - 1]

    # Reference grid queries: before, at, between, at, beyond.
    ref_grid = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    out = eval_on_ref_grid(cif_native, native_grid, ref_grid)

    # Step-function semantics (searchsorted side="right" - 1):
    #   t=1.0 (before first event): clamped to index 0 -> 0.3 [convention: CIF stays 0.3 pre-t0?]
    #   Our convention matches _pointwise_cif_gap: clip to [0, len-1], so t < grid[0] -> grid[0]'s value.
    #   t=2.0 -> index 0 -> 0.3
    #   t=3.0 -> index 0 -> 0.3
    #   t=4.0 -> index 1 -> 0.7
    #   t=5.0 (beyond): clamped to index 1 -> 0.7
    np.testing.assert_allclose(out, np.array([[0.3, 0.3, 0.3, 0.7, 0.7]]))
    assert out.shape == (1, 5)


def test_eval_on_ref_grid_two_samples():
    from validation.alignment.equivalence_gate import eval_on_ref_grid

    native_grid = np.array([1.0, 3.0])
    cif_native = np.array([[0.1, 0.5], [0.2, 0.6]])  # (n_samples, n_native)
    ref_grid = np.array([2.0, 3.0])

    out = eval_on_ref_grid(cif_native, native_grid, ref_grid)
    # t=2.0 -> index 0, t=3.0 -> index 1.
    np.testing.assert_allclose(out, np.array([[0.1, 0.5], [0.2, 0.6]]))


def _fixture_cells_perfect_match(n_seeds: int = 4) -> list[dict]:
    """4 seeds, comprisk == rfSRC exactly, small seed-to-seed variation within each lib.

    Returns a list of per-cell dicts with keys: seed, cif_cr, cif_rf, risk_cr, risk_rf.
    """
    cells = []
    rng = np.random.default_rng(0)
    for s in range(n_seeds):
        # Within-lib variation: 0.01 rms between seeds.
        cif_base = 0.3 + 0.001 * s + rng.normal(0, 0.005, size=(5, 3))
        risk_base = cif_base[:, -1]
        cells.append(
            {
                "seed": s,
                "cif_cr": cif_base.copy(),
                "cif_rf": cif_base.copy(),  # perfect cross-lib match
                "risk_cr": risk_base.copy(),
                "risk_rf": risk_base.copy(),
            }
        )
    return cells


def test_aggregate_dataset_keys_and_shapes():
    from validation.alignment.equivalence_gate import aggregate_dataset

    cells = _fixture_cells_perfect_match(n_seeds=4)
    agg = aggregate_dataset(cells)

    expected_keys = {
        "within_cr_p95_risk",
        "within_rf_p95_risk",
        "within_cr_p95_cif",
        "within_rf_p95_cif",
        "cross_p95_risk",
        "cross_p95_cif",
        "cross_max_risk",
        "cross_max_cif",
        "cross_p95_max_over_seeds_risk",
        "cross_p95_max_over_seeds_cif",
        "quantiles",
        "n_seeds",
    }
    assert set(agg.keys()) == expected_keys
    assert agg["n_seeds"] == 4


def test_aggregate_dataset_perfect_match_cross_zero():
    from validation.alignment.equivalence_gate import aggregate_dataset

    cells = _fixture_cells_perfect_match(n_seeds=4)
    agg = aggregate_dataset(cells)

    # Perfect cross-lib match → cross gaps are 0.
    assert agg["cross_p95_risk"] == 0.0
    assert agg["cross_p95_cif"] == 0.0
    assert agg["cross_max_risk"] == 0.0
    assert agg["cross_max_cif"] == 0.0
    # Within-lib still positive (seed-to-seed variation).
    assert agg["within_cr_p95_risk"] > 0.0
    assert agg["within_cr_p95_cif"] > 0.0


def test_aggregate_dataset_quantiles_block_shape_and_cross_zero():
    from validation.alignment.equivalence_gate import QUANTILE_GRID, aggregate_dataset

    cells = _fixture_cells_perfect_match(n_seeds=4)
    agg = aggregate_dataset(cells)
    q = agg["quantiles"]

    # Six sub-blocks (cross + within-cr + within-rf, for risk + cif).
    assert set(q.keys()) == {
        "cross_risk",
        "cross_cif",
        "within_cr_risk",
        "within_rf_risk",
        "within_cr_cif",
        "within_rf_cif",
    }
    # Each sub-block has one entry per quantile, keys match QUANTILE_GRID.
    for sub in q.values():
        assert set(sub.keys()) == set(QUANTILE_GRID)

    # Perfect cross-lib match → every cross quantile is 0; every within
    # quantile is >= 0 and ascending in q (|Δ| is non-decreasing in quantile).
    for quant in QUANTILE_GRID:
        assert q["cross_risk"][quant] == 0.0
        assert q["cross_cif"][quant] == 0.0
    within_cr_sorted = [q["within_cr_risk"][quant] for quant in QUANTILE_GRID]
    assert within_cr_sorted == sorted(within_cr_sorted)

    # p95 key in quantiles must agree with the legacy cross_p95_* key.
    assert q["cross_risk"][0.95] == agg["cross_p95_risk"]
    assert q["cross_cif"][0.95] == agg["cross_p95_cif"]


def test_aggregate_dataset_requires_even_seed_count():
    from validation.alignment.equivalence_gate import aggregate_dataset

    cells = _fixture_cells_perfect_match(n_seeds=3)  # odd → can't pair
    with pytest.raises(ValueError, match="even number of seeds"):
        aggregate_dataset(cells)


HARD_CAP = 0.05


def _agg(within_cr, within_rf, cross, metric_suffix):
    """Minimal aggregate dict for a single metric (risk or cif)."""
    return {
        f"within_cr_p95_{metric_suffix}": within_cr,
        f"within_rf_p95_{metric_suffix}": within_rf,
        f"cross_p95_{metric_suffix}": cross,
    }


def _full_agg(wcr_r, wrf_r, x_r, wcr_c, wrf_c, x_c):
    d = {}
    d.update(_agg(wcr_r, wrf_r, x_r, "risk"))
    d.update(_agg(wcr_c, wrf_c, x_c, "cif"))
    return d


def test_apply_tolerance_pass_both():
    from validation.alignment.equivalence_gate import apply_tolerance

    # cross below within-lib floor and below hard-cap for both metrics.
    agg = _full_agg(0.02, 0.02, 0.01, 0.03, 0.03, 0.02)
    out = apply_tolerance(agg, hard_cap=HARD_CAP)
    assert out == {
        "noise_floor_pass_risk": True,
        "hard_cap_pass_risk": True,
        "noise_floor_pass_cif": True,
        "hard_cap_pass_cif": True,
        "overall_pass": True,
        "hard_cap_pass_overall": True,
    }


def test_apply_tolerance_fail_noise_floor_only():
    from validation.alignment.equivalence_gate import apply_tolerance

    # cross > within-lib floor (fail) but < hard-cap (pass) for risk.
    # Under the noise-floor-only gate contract, overall FAILS.
    agg = _full_agg(0.01, 0.01, 0.03, 0.03, 0.03, 0.02)
    out = apply_tolerance(agg, hard_cap=HARD_CAP)
    assert out["noise_floor_pass_risk"] is False
    assert out["hard_cap_pass_risk"] is True
    assert out["overall_pass"] is False
    assert out["hard_cap_pass_overall"] is True


def test_apply_tolerance_noise_floor_pass_hard_cap_fail():
    from validation.alignment.equivalence_gate import apply_tolerance

    # cross < within-lib floor (pass) but > hard-cap for cif.
    # Under the noise-floor-only contract, overall PASSES; hard_cap is advisory.
    # This matches the production-config behavior on hd/follic post-2026-04-24:
    # cross-lib gap is within within-lib seed variance but exceeds the heuristic
    # 0.05 absolute cap (see docs/equivalence-vs-rfsrc.md).
    agg = _full_agg(0.02, 0.02, 0.01, 0.10, 0.10, 0.08)
    out = apply_tolerance(agg, hard_cap=HARD_CAP)
    assert out["noise_floor_pass_cif"] is True
    assert out["hard_cap_pass_cif"] is False
    assert out["overall_pass"] is True
    assert out["hard_cap_pass_overall"] is False


def test_apply_tolerance_fail_both():
    from validation.alignment.equivalence_gate import apply_tolerance

    # cross > within-lib floor and > hard-cap for both.
    agg = _full_agg(0.01, 0.01, 0.20, 0.01, 0.01, 0.20)
    out = apply_tolerance(agg, hard_cap=HARD_CAP)
    assert out["noise_floor_pass_risk"] is False
    assert out["hard_cap_pass_risk"] is False
    assert out["noise_floor_pass_cif"] is False
    assert out["hard_cap_pass_cif"] is False
    assert out["overall_pass"] is False
    assert out["hard_cap_pass_overall"] is False


def test_apply_tolerance_uses_max_of_within_libs():
    from validation.alignment.equivalence_gate import apply_tolerance

    # within_cr_p95_risk = 0.01, within_rf_p95_risk = 0.05 -> floor is 0.05.
    # cross_p95_risk = 0.03 -> below 0.05 floor, so noise-floor passes.
    agg = _full_agg(0.01, 0.05, 0.03, 0.02, 0.02, 0.01)
    out = apply_tolerance(agg, hard_cap=HARD_CAP)
    assert out["noise_floor_pass_risk"] is True


def test_persist_and_load_cell_roundtrip(tmp_path):
    from validation.alignment.equivalence_gate import load_cell, persist_cell

    cif_cr = np.array([[0.1, 0.3, 0.5], [0.2, 0.4, 0.6]])
    cif_rf = np.array([[0.15, 0.3, 0.52], [0.18, 0.41, 0.58]])
    ref_grid = np.array([1.0, 2.0, 3.0])
    cr_native = np.array([0.5, 1.5, 2.5])
    rf_native = np.array([0.8, 1.8, 2.8])

    path = tmp_path / "hd_s0.parquet"
    persist_cell(
        path=path,
        dataset="hd",
        seed=0,
        cif_cr=cif_cr,
        cif_rf=cif_rf,
        ref_grid=ref_grid,
        cr_native_grid=cr_native,
        rf_native_grid=rf_native,
        n_train=100,
        n_test=50,
        commit_sha="abc1234",
    )

    loaded = load_cell(path)
    np.testing.assert_allclose(loaded["cif_cr"], cif_cr)
    np.testing.assert_allclose(loaded["cif_rf"], cif_rf)
    np.testing.assert_allclose(loaded["risk_cr"], cif_cr[:, -1])
    np.testing.assert_allclose(loaded["risk_rf"], cif_rf[:, -1])
    np.testing.assert_allclose(loaded["ref_grid"], ref_grid)
    assert loaded["seed"] == 0
    assert loaded["dataset"] == "hd"
    assert loaded["n_train"] == 100
    assert loaded["n_test"] == 50
    assert loaded["commit_sha"] == "abc1234"
