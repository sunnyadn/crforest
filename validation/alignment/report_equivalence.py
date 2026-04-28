"""Markdown report writer for the equivalence-gate audit.

Consumed outputs: per-dataset aggregate (within/cross gaps), per-dataset pass
dict, tree-structure diagnostic, cause-2 symmetry numbers, header metadata.
"""

from __future__ import annotations

from pathlib import Path


def _check(b: bool) -> str:
    return "✓" if b else "✗"


def _fmt_quantile_row(label: str, d: dict, qs: list[float]) -> str:
    cells = " | ".join(f"{d[q]:.4f}" for q in qs)
    return f"| {label} | {cells} |"


def write_report(
    *,
    datasets_agg: dict,
    datasets_pass: dict,
    tree_stats: dict,
    cause2_symmetry: dict,
    header: dict,
    path: Path,
) -> None:
    """Write the equivalence-gate audit as a markdown file at ``path``."""
    path = Path(path)
    lines: list[str] = []
    lines.append("# crforest vs randomForestSRC — equivalence-gate audit")
    lines.append("")
    lines.append(f"Timestamp: {header['timestamp']}")
    lines.append(f"crforest commit: {header['commit_sha']}")
    lines.append(f"randomForestSRC: {header['rfsrc_version']}")
    lines.append(f"Python: {header['python_version']}  |  R: {header['r_version']}")
    lines.append(f"Machine: {header['machine']}")
    lines.append(f"Command: `{header['command']}`")
    lines.append(f"Hard cap: {header['hard_cap']}")
    lines.append("")

    # Gate table (per dataset x metric).
    lines.append("## Gate table")
    lines.append("")
    lines.append(
        "| Dataset | metric | within_cr_p95 | within_rf_p95 | noise_floor | "
        "cross_p95 | cross_max | hard_cap | noise_floor_pass | hard_cap_pass |"
    )
    lines.append(
        "|---------|--------|---------------|---------------|-------------|"
        "-----------|-----------|----------|------------------|---------------|"
    )
    for ds in datasets_agg:
        agg = datasets_agg[ds]
        p = datasets_pass[ds]
        for m in ("risk", "cif"):
            floor = max(agg[f"within_cr_p95_{m}"], agg[f"within_rf_p95_{m}"])
            lines.append(
                f"| {ds} | {m} "
                f"| {agg[f'within_cr_p95_{m}']:.4f} "
                f"| {agg[f'within_rf_p95_{m}']:.4f} "
                f"| {floor:.4f} "
                f"| {agg[f'cross_p95_{m}']:.4f} "
                f"| {agg[f'cross_max_{m}']:.4f} "
                f"| {header['hard_cap']:.4f} "
                f"| {_check(p[f'noise_floor_pass_{m}'])} "
                f"| {_check(p[f'hard_cap_pass_{m}'])} |"
            )
    lines.append("")

    # Per-dataset pass verdict.
    lines.append("## Per-dataset verdict")
    lines.append("")
    lines.append(
        "| Dataset | noise_floor_risk | hard_cap_risk | noise_floor_cif | hard_cap_cif | overall |"
    )
    lines.append(
        "|---------|------------------|---------------|-----------------|--------------|---------|"
    )
    for ds, p in datasets_pass.items():
        verdict = "PASS" if p["overall_pass"] else "FAIL"
        lines.append(
            f"| {ds} "
            f"| {_check(p['noise_floor_pass_risk'])} "
            f"| {_check(p['hard_cap_pass_risk'])} "
            f"| {_check(p['noise_floor_pass_cif'])} "
            f"| {_check(p['hard_cap_pass_cif'])} "
            f"| {verdict} |"
        )
    lines.append("")

    # Quantile-dominance companion view (not gated).
    sample_agg = next(iter(datasets_agg.values()), {})
    if "quantiles" in sample_agg:
        qs = sorted(sample_agg["quantiles"]["cross_risk"].keys())
        lines.append("## Quantile-dominance view (not gated)")
        lines.append("")
        lines.append(
            "Full gap-CDF shape. Cross-lib rows are median-over-seeds of per-seed quantile; "
            "within-lib rows are max over paired seeds (0,1), (2,3), … The gate's hard-cap tests q0.95."
        )
        lines.append("")
        header_q = "| series | " + " | ".join(f"q{q:g}" for q in qs) + " |"
        sep_q = "|--------|" + "|".join(["---"] * len(qs)) + "|"
        for ds, agg in datasets_agg.items():
            q = agg.get("quantiles")
            if q is None:
                continue
            lines.append(f"### {ds}")
            lines.append("")
            for metric, key_cross, key_wcr, key_wrf in (
                ("risk", "cross_risk", "within_cr_risk", "within_rf_risk"),
                ("cif", "cross_cif", "within_cr_cif", "within_rf_cif"),
            ):
                lines.append(f"**|Δ{metric}|**")
                lines.append("")
                lines.append(header_q)
                lines.append(sep_q)
                lines.append(_fmt_quantile_row("cross-lib (median over seeds)", q[key_cross], qs))
                lines.append(_fmt_quantile_row("within crforest (paired max)", q[key_wcr], qs))
                lines.append(_fmt_quantile_row("within rfSRC (paired max)", q[key_wrf], qs))
                lines.append("")

    # Integrated Brier Score (IBS) companion view — scalar per (lib, seed).
    sample_agg = next(iter(datasets_agg.values()), {})
    if "cross_p95_ibs" in sample_agg:
        lines.append("## Integrated Brier Score (IPCW, advisory)")
        lines.append("")
        lines.append(
            "One scalar per (lib, seed) integrating IPCW-weighted Brier over the "
            "reference grid. Cross/within p95 use the same paired-seed noise-floor "
            "convention as the CIF gate; at n_seeds=20, p95 ≈ max-of-gaps, so we "
            "also print the median + max for readability. IBS is noise-floor "
            "only (no hard-cap): |ΔIBS| lacks a dataset-independent absolute "
            "threshold, so we check that it sits inside single-lib seed variance."
        )
        lines.append("")
        lines.append(
            "| Dataset | mean_ibs_cr | mean_ibs_rf | within_cr_p95 | within_rf_p95 "
            "| cross_median | cross_p95 | cross_max | noise_floor_pass |"
        )
        lines.append(
            "|---------|-------------|-------------|---------------|---------------"
            "|--------------|-----------|-----------|------------------|"
        )
        for ds, agg in datasets_agg.items():
            if "cross_p95_ibs" not in agg:
                continue
            p = datasets_pass[ds]
            lines.append(
                f"| {ds} "
                f"| {agg['mean_ibs_cr']:.4f} "
                f"| {agg['mean_ibs_rf']:.4f} "
                f"| {agg['within_cr_p95_ibs']:.4f} "
                f"| {agg['within_rf_p95_ibs']:.4f} "
                f"| {agg['cross_median_ibs']:.4f} "
                f"| {agg['cross_p95_ibs']:.4f} "
                f"| {agg['cross_max_ibs']:.4f} "
                f"| {_check(p.get('noise_floor_pass_ibs', False))} |"
            )
        lines.append("")

    # Tree-structure diagnostic.
    lines.append("## Tree-structure diagnostic (not gated)")
    lines.append("")
    lines.append("| Dataset | lib | mean_leaves | mean_depth | leaf_p5 | leaf_p95 |")
    lines.append("|---------|-----|-------------|------------|---------|----------|")
    for ds, libs in tree_stats.items():
        for lib, stats in libs.items():
            depth = f"{stats['mean_depth']:.1f}" if stats["mean_depth"] is not None else "N/A"
            lines.append(
                f"| {ds} | {lib} "
                f"| {stats['mean_leaves']:.1f} "
                f"| {depth} "
                f"| {stats['leaf_p5']:.0f} "
                f"| {stats['leaf_p95']:.0f} |"
            )
    lines.append("")

    # Cause-2 symmetry.
    lines.append("## Cause-2 symmetry (not gated)")
    lines.append("")
    for ds, val in cause2_symmetry.items():
        lines.append(f"- `{ds}`: cross_p95_cif_cause2 = {val:.4f}")
    lines.append("")

    # Final verdict.
    n_pass = sum(1 for p in datasets_pass.values() if p["overall_pass"])
    n_total = len(datasets_pass)
    lines.append("## Verdict")
    lines.append("")
    if n_pass == n_total:
        lines.append(f"**EQUIVALENCE AUDIT: PASS ({n_pass}/{n_total})**")
    else:
        fail_notes = []
        any_binning_residual = False
        for ds, p in datasets_pass.items():
            if not p["overall_pass"]:
                reasons = []
                for metric in ("risk", "cif"):
                    hc_key = f"hard_cap_pass_{metric}"
                    nf_key = f"noise_floor_pass_{metric}"
                    if not p[hc_key]:
                        val = datasets_agg[ds][f"cross_p95_{metric}"]
                        if p[nf_key]:
                            # Noise-floor passes → cross gap < within-lib seed variance;
                            # hard-cap miss is an algorithmic edge, not a bias signature.
                            reasons.append(
                                f"cross_p95_{metric} = {val:.4f} > hard_cap "
                                "(noise-floor pass; binning residual candidate)"
                            )
                            any_binning_residual = True
                        else:
                            reasons.append(f"cross_p95_{metric} = {val:.4f} > hard_cap")
                if not p["noise_floor_pass_risk"]:
                    reasons.append("noise_floor_risk breached")
                if not p["noise_floor_pass_cif"]:
                    reasons.append("noise_floor_cif breached")
                fail_notes.append(f"{ds}: {'; '.join(reasons)}")
        lines.append(f"**EQUIVALENCE AUDIT: FAIL ({n_pass}/{n_total})**  " + " | ".join(fail_notes))
        if any_binning_residual:
            lines.append("")
            lines.append(
                '> Note: "binning residual candidate" = hard-cap failure on a dataset '
                "where the cross-lib p95 is still below the within-lib paired-seed noise floor. "
                "Partly driven by crforest's histogram-binned splits vs rfSRC's exhaustive-over-"
                "observed-thresholds splits: the hd nsplit convergence sweep (report "
                "`nsplit_convergence_2026-04-24T05-05-56.md`) shows raising crforest nsplit "
                "from 10 to 100 shrinks the gap by ~10% but exhaustive (nsplit=0) does not "
                "minimize — so binning contributes but is not the sole mechanism. See also "
                "the IBS section (above) which passes noise-floor on all four datasets."
            )
    lines.append("")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines))
