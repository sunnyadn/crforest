"""Unit tests for equivalence-gate markdown report writer."""

from __future__ import annotations


def _fixture_input():
    datasets_agg = {
        "follic": {
            "within_cr_p95_risk": 0.02,
            "within_rf_p95_risk": 0.02,
            "within_cr_p95_cif": 0.03,
            "within_rf_p95_cif": 0.03,
            "cross_p95_risk": 0.01,
            "cross_p95_cif": 0.02,
            "cross_max_risk": 0.05,
            "cross_max_cif": 0.07,
            "cross_p95_max_over_seeds_risk": 0.02,
            "cross_p95_max_over_seeds_cif": 0.03,
            "n_seeds": 20,
        },
        "pbc": {
            "within_cr_p95_risk": 0.01,
            "within_rf_p95_risk": 0.01,
            "within_cr_p95_cif": 0.02,
            "within_rf_p95_cif": 0.02,
            "cross_p95_risk": 0.005,
            "cross_p95_cif": 0.07,  # cif cross > 0.05 hard cap
            "cross_max_risk": 0.02,
            "cross_max_cif": 0.12,
            "cross_p95_max_over_seeds_risk": 0.01,
            "cross_p95_max_over_seeds_cif": 0.09,
            "n_seeds": 20,
        },
    }
    datasets_pass = {
        "follic": {
            "noise_floor_pass_risk": True,
            "hard_cap_pass_risk": True,
            "noise_floor_pass_cif": True,
            "hard_cap_pass_cif": True,
            "overall_pass": True,
        },
        "pbc": {
            "noise_floor_pass_risk": True,
            "hard_cap_pass_risk": True,
            "noise_floor_pass_cif": False,
            "hard_cap_pass_cif": False,
            "overall_pass": False,
        },
    }
    tree_stats = {
        "follic": {
            "crforest": {"mean_leaves": 42.1, "mean_depth": 6.2, "leaf_p5": 1.0, "leaf_p95": 20.0},
            "rfSRC": {"mean_leaves": 40.5, "mean_depth": None, "leaf_p5": 1.0, "leaf_p95": 19.0},
        },
        "pbc": {
            "crforest": {"mean_leaves": 38.0, "mean_depth": 5.8, "leaf_p5": 1.0, "leaf_p95": 18.0},
            "rfSRC": {"mean_leaves": 36.2, "mean_depth": None, "leaf_p5": 1.0, "leaf_p95": 17.0},
        },
    }
    cause2_symmetry = {"follic": 0.015, "pbc": 0.04}
    header = {
        "timestamp": "2026-04-24T12:00:00",
        "commit_sha": "abc1234",
        "rfsrc_version": "3.2.3",
        "python_version": "3.12.0",
        "r_version": "4.4.0",
        "machine": "Darwin arm64 M2 Pro 32GB",
        "command": "python -m validation.alignment.equivalence_gate --datasets follic pbc",
        "hard_cap": 0.05,
    }
    return datasets_agg, datasets_pass, tree_stats, cause2_symmetry, header


def test_write_report_contains_gate_table_and_verdict(tmp_path):
    from validation.alignment.report_equivalence import write_report

    agg, passes, tree, cause2, header = _fixture_input()
    out = tmp_path / "equivalence_2026-04-24.md"
    write_report(
        datasets_agg=agg,
        datasets_pass=passes,
        tree_stats=tree,
        cause2_symmetry=cause2,
        header=header,
        path=out,
    )
    text = out.read_text()

    # Header fields.
    assert "abc1234" in text
    assert "Darwin arm64 M2 Pro 32GB" in text

    # Gate table rows (one per dataset x metric).
    assert "| follic  | risk" in text or "| follic | risk" in text
    assert "| pbc" in text and "cif" in text

    # Per-dataset pass rows.
    assert "PASS" in text  # follic overall pass
    assert "FAIL" in text  # pbc overall fail

    # Tree-structure diagnostic.
    assert "mean_leaves" in text

    # Cause-2 symmetry line.
    assert "cause2" in text.lower() or "cause-2" in text.lower()


def test_write_report_all_pass_verdict(tmp_path):
    from validation.alignment.report_equivalence import write_report

    agg, passes, tree, cause2, header = _fixture_input()
    # Force pbc to pass too.
    passes["pbc"] = {
        "noise_floor_pass_risk": True,
        "hard_cap_pass_risk": True,
        "noise_floor_pass_cif": True,
        "hard_cap_pass_cif": True,
        "overall_pass": True,
    }
    out = tmp_path / "eq.md"
    write_report(
        datasets_agg=agg,
        datasets_pass=passes,
        tree_stats=tree,
        cause2_symmetry=cause2,
        header=header,
        path=out,
    )
    text = out.read_text()
    assert "EQUIVALENCE AUDIT: PASS (2/2)" in text


def test_write_report_fail_verdict_includes_reason(tmp_path):
    from validation.alignment.report_equivalence import write_report

    agg, passes, tree, cause2, header = _fixture_input()
    out = tmp_path / "eq.md"
    write_report(
        datasets_agg=agg,
        datasets_pass=passes,
        tree_stats=tree,
        cause2_symmetry=cause2,
        header=header,
        path=out,
    )
    text = out.read_text()
    assert "EQUIVALENCE AUDIT: FAIL" in text
    assert "pbc" in text  # the failing dataset is named


def test_write_report_binning_residual_annotation(tmp_path):
    from validation.alignment.report_equivalence import write_report

    agg, passes, tree, cause2, header = _fixture_input()
    # pbc fixture: noise_floor_pass_cif=False, hard_cap_pass_cif=False → not a residual.
    # Flip to noise_floor_pass_cif=True, hard_cap_pass_cif=False to trigger annotation.
    passes["pbc"]["noise_floor_pass_cif"] = True
    out = tmp_path / "eq.md"
    write_report(
        datasets_agg=agg,
        datasets_pass=passes,
        tree_stats=tree,
        cause2_symmetry=cause2,
        header=header,
        path=out,
    )
    text = out.read_text()
    assert "binning residual candidate" in text
    # And the explanatory note must be present. The footer was updated on
    # 2026-04-24 after the nsplit convergence sweep showed binning is a
    # contributor but not the sole mechanism — phrasing now names the split
    # policy divergence (binned vs exhaustive-over-observed-thresholds) and
    # points at the companion IBS section.
    assert "histogram-binned splits" in text
    assert "nsplit convergence" in text
    assert "not the sole mechanism" in text


def test_write_report_quantile_block_rendered_when_present(tmp_path):
    from validation.alignment.report_equivalence import write_report

    agg, passes, tree, cause2, header = _fixture_input()
    # Attach a minimal quantiles block to one dataset — the block is optional,
    # so writer should render only for datasets that include it.
    agg["follic"]["quantiles"] = {
        "cross_risk": {0.5: 0.001, 0.75: 0.003, 0.9: 0.005, 0.95: 0.01, 0.99: 0.02},
        "cross_cif": {0.5: 0.002, 0.75: 0.004, 0.9: 0.006, 0.95: 0.02, 0.99: 0.03},
        "within_cr_risk": {0.5: 0.01, 0.75: 0.02, 0.9: 0.03, 0.95: 0.04, 0.99: 0.05},
        "within_rf_risk": {0.5: 0.01, 0.75: 0.02, 0.9: 0.03, 0.95: 0.04, 0.99: 0.05},
        "within_cr_cif": {0.5: 0.01, 0.75: 0.02, 0.9: 0.03, 0.95: 0.04, 0.99: 0.05},
        "within_rf_cif": {0.5: 0.01, 0.75: 0.02, 0.9: 0.03, 0.95: 0.04, 0.99: 0.05},
    }
    out = tmp_path / "eq.md"
    write_report(
        datasets_agg=agg,
        datasets_pass=passes,
        tree_stats=tree,
        cause2_symmetry=cause2,
        header=header,
        path=out,
    )
    text = out.read_text()
    assert "Quantile-dominance view" in text
    assert "q0.5" in text and "q0.95" in text and "q0.99" in text
    assert "cross-lib (median over seeds)" in text


def test_write_report_no_quantile_block_when_absent(tmp_path):
    """Backward compat: old callers that don't pass a `quantiles` sub-dict still work."""
    from validation.alignment.report_equivalence import write_report

    agg, passes, tree, cause2, header = _fixture_input()
    out = tmp_path / "eq.md"
    write_report(
        datasets_agg=agg,
        datasets_pass=passes,
        tree_stats=tree,
        cause2_symmetry=cause2,
        header=header,
        path=out,
    )
    text = out.read_text()
    assert "Quantile-dominance view" not in text
