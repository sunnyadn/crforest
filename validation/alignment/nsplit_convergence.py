"""nsplit convergence sweep — falsification of the binning-residual hypothesis.

Spec: docs/superpowers/specs/2026-04-24-nsplit-convergence-design.md
Follows from: project_equivalence_hd_tail_diagnostic.

Fixes rfSRC at nsplit=10 and sweeps crforest nsplit over {10, 32, 100, 0}
on one dataset, reading the cross_p95 trend against the gate's noise floor.
Monotone shrink to within-noise-floor at nsplit=0 would confirm the
histogram-boundary explanation for the hd/follic hard-cap failures.

Run:
    uv run --extra maintainer python -m validation.alignment.nsplit_convergence \
        --dataset hd --seeds 10
"""

from __future__ import annotations

import argparse
import datetime as _dt
from pathlib import Path

import numpy as np

from validation.alignment.equivalence_gate import (
    HARD_CAP_DEFAULT,
    QUANTILE_GRID,
    _git_sha,
    _machine_fingerprint,
    aggregate_dataset,
    apply_tolerance,
    fit_and_capture,
    load_cell,
)

NSPLIT_GRID: tuple[int, ...] = (10, 32, 100, 0)


def _cache_path(cache_dir: Path, dataset: str, seed: int, nsplit: int) -> Path:
    return cache_dir / f"{dataset}_s{seed}_nsplit{nsplit}.parquet"


def _ck(b: bool) -> str:
    return "✓" if b else "✗"


def _verdict(per_nsplit: dict[int, dict], hard_cap: float) -> str:
    """CONFIRMED if cross_p95_cif monotone non-increasing across NSPLIT_GRID
    AND nsplit=0 brings cross_p95_cif below hard_cap. REFUTED otherwise."""
    cif = [per_nsplit[n]["cross_p95_cif"] for n in NSPLIT_GRID]
    risk = [per_nsplit[n]["cross_p95_risk"] for n in NSPLIT_GRID]
    monotone = all(cif[i] >= cif[i + 1] for i in range(len(cif) - 1))
    exhaustive_passes = per_nsplit[0]["cross_p95_cif"] <= hard_cap
    monotone_risk = all(risk[i] >= risk[i + 1] for i in range(len(risk) - 1))

    bullets = [
        f"- cross_p95_cif across nsplit={list(NSPLIT_GRID)}: "
        f"{[f'{c:.4f}' for c in cif]} → " + ("monotone ↓" if monotone else "NOT monotone"),
        f"- cross_p95_risk: {[f'{r:.4f}' for r in risk]} → "
        + ("monotone ↓" if monotone_risk else "NOT monotone"),
        f"- nsplit=0 cross_p95_cif = {per_nsplit[0]['cross_p95_cif']:.4f} "
        f"({'<= hard_cap' if exhaustive_passes else '> hard_cap'} {hard_cap})",
    ]

    if monotone and exhaustive_passes:
        head = "**Binning-residual hypothesis: CONFIRMED.**"
    elif monotone:
        head = (
            "**Binning-residual hypothesis: PARTIAL.** Trend is monotone, but "
            "exhaustive splits still exceed hard_cap — mechanism is binning "
            "but additional residual remains at nsplit=0."
        )
    else:
        head = "**Binning-residual hypothesis: REFUTED.** Trend is not monotone."
    return head + "\n\n" + "\n".join(bullets)


def _write_report(
    *,
    dataset: str,
    per_nsplit: dict[int, dict],
    per_nsplit_pass: dict[int, dict],
    header: dict,
    path: Path,
) -> None:
    lines = [
        "# nsplit convergence — crforest vs rfSRC",
        "",
        f"Timestamp: {header['timestamp']}",
        f"Dataset: `{dataset}`  |  seeds: {header['n_seeds']}  |  rfSRC fixed at nsplit=10",
        f"crforest commit: {header['commit_sha']}  |  machine: {header['machine']}",
        f"Hard cap: {header['hard_cap']}",
        "",
        "## Trend table",
        "",
        "| crforest nsplit | cross_p95_risk | cross_p95_cif | within_cr_p95_risk "
        "| within_rf_p95_risk | within_cr_p95_cif | within_rf_p95_cif | "
        "noise_floor_risk | hard_cap_risk | noise_floor_cif | hard_cap_cif |",
        "|---|---|---|---|---|---|---|---|---|---|---|",
    ]

    def _label(n: int) -> str:
        return "0 (exhaustive)" if n == 0 else str(n)

    def _qrow(name: str, d: dict) -> str:
        return "| " + name + " | " + " | ".join(f"{d[qq]:.4f}" for qq in QUANTILE_GRID) + " |"

    for n in NSPLIT_GRID:
        a = per_nsplit[n]
        p = per_nsplit_pass[n]
        lines.append(
            f"| {_label(n)} | {a['cross_p95_risk']:.4f} | {a['cross_p95_cif']:.4f} "
            f"| {a['within_cr_p95_risk']:.4f} | {a['within_rf_p95_risk']:.4f} "
            f"| {a['within_cr_p95_cif']:.4f} | {a['within_rf_p95_cif']:.4f} "
            f"| {_ck(p['noise_floor_pass_risk'])} | {_ck(p['hard_cap_pass_risk'])} "
            f"| {_ck(p['noise_floor_pass_cif'])} | {_ck(p['hard_cap_pass_cif'])} |"
        )

    lines += ["", "## Quantile-dominance per nsplit"]
    q_header = "| series | " + " | ".join(f"q{qq:g}" for qq in QUANTILE_GRID) + " |"
    q_sep = "|--------|" + "---|" * len(QUANTILE_GRID)
    for n in NSPLIT_GRID:
        q = per_nsplit[n]["quantiles"]
        lines += ["", f"### nsplit = {_label(n)}"]
        for metric in ("risk", "cif"):
            lines += [
                "",
                f"**|Δ{metric}|**",
                "",
                q_header,
                q_sep,
                _qrow("cross-lib", q[f"cross_{metric}"]),
                _qrow("within crforest", q[f"within_cr_{metric}"]),
                _qrow("within rfSRC", q[f"within_rf_{metric}"]),
            ]

    lines += ["", "## Verdict", "", _verdict(per_nsplit, header["hard_cap"]), ""]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", default="hd")
    parser.add_argument("--seeds", type=int, default=10)
    parser.add_argument("--cache-dir", default="validation/alignment/_cache")
    parser.add_argument("--out", default="validation/reports")
    parser.add_argument("--hard-cap", type=float, default=HARD_CAP_DEFAULT)
    parser.add_argument("--force-refit", action="store_true")
    args = parser.parse_args(argv)

    if args.seeds % 2 != 0:
        parser.error("--seeds must be even (within-lib pairing)")

    cache_dir = Path(args.cache_dir)
    commit_sha = _git_sha()

    per_nsplit: dict[int, dict] = {}
    per_nsplit_pass: dict[int, dict] = {}

    for n in NSPLIT_GRID:
        cells = []
        for s in range(args.seeds):
            path = _cache_path(cache_dir, args.dataset, s, n)
            # Reuse rfSRC from the first-in-sweep nsplit (always computed before
            # any other nsplit for this seed because the outer loop is nsplit-first).
            # fit_and_capture() silently falls back to a fresh rfSRC fit if the
            # reuse-source path does not exist (e.g. when n == NSPLIT_GRID[0]).
            rf_reuse = (
                None
                if n == NSPLIT_GRID[0]
                else _cache_path(cache_dir, args.dataset, s, NSPLIT_GRID[0])
            )
            fit_and_capture(
                dataset=args.dataset,
                seed=s,
                cache_dir=cache_dir,
                commit_sha=commit_sha,
                force_refit=args.force_refit,
                cr_nsplit=n,
                rf_reuse_cache=rf_reuse,
            )
            cells.append(load_cell(path))
        agg = aggregate_dataset(cells)
        per_nsplit[n] = agg
        per_nsplit_pass[n] = apply_tolerance(agg, hard_cap=args.hard_cap)

    timestamp = _dt.datetime.now().isoformat(timespec="seconds")
    out_path = Path(args.out) / f"nsplit_convergence_{timestamp.replace(':', '-')}.md"

    import sys

    header = {
        "timestamp": timestamp,
        "commit_sha": commit_sha,
        "python_version": sys.version.split()[0],
        "machine": _machine_fingerprint(),
        "hard_cap": args.hard_cap,
        "n_seeds": args.seeds,
    }
    _write_report(
        dataset=args.dataset,
        per_nsplit=per_nsplit,
        per_nsplit_pass=per_nsplit_pass,
        header=header,
        path=out_path,
    )
    print(f"wrote {out_path}", flush=True)

    np.set_printoptions(precision=4, suppress=True)
    print(_verdict(per_nsplit, args.hard_cap), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
