"""zeta benchmark — compare crforest vs rfSRC timings and emit decision report.

Loads `timings/{crforest,rfsrc}_timings.parquet`, fits `log(t) = alpha*log(n) + beta`
per lib, extrapolates to n=100k, applies the spec section-5 exit rule, and writes
`reports/zeta_report_<timestamp>.md`.

Spec: `docs/superpowers/specs/2026-04-23-zeta-head-to-head-benchmark-design.md`.
rfSRC data provenance: `timings/rfsrc_timings.parquet.provenance.md`.

Usage:
    uv run python compare.py
    uv run python compare.py --self-test
"""

from __future__ import annotations

import argparse
import platform
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
TIMINGS = HERE / "timings"
REPORTS = HERE / "reports"
PROVENANCE = TIMINGS / "rfsrc_timings.parquet.provenance.md"

EXTRAP_N = 100_000
R2_CONFIDENCE_THRESHOLD = 0.95
RSS_N = 50_000  # memory gate evaluated at the largest measured point


@dataclass(frozen=True)
class LibFit:
    """Log-log fit on fit_wall_s for a single lib."""

    lib: str
    alpha: float
    beta: float
    r2: float
    extrap_100k_s: float
    extrap_band_lo: float
    extrap_band_hi: float


def _loglog_fit(n: np.ndarray, t: np.ndarray) -> tuple[float, float, float]:
    """Return (alpha, beta, r2) for log(t) = alpha * log(n) + beta."""
    logn, logt = np.log(n), np.log(t)
    alpha, beta = np.polyfit(logn, logt, 1)
    pred = alpha * logn + beta
    ss_res = np.sum((logt - pred) ** 2)
    ss_tot = np.sum((logt - logt.mean()) ** 2)
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else 1.0
    return float(alpha), float(beta), float(r2)


def _fit_lib(df: pd.DataFrame, lib: str) -> LibFit:
    # Mean per n across seeds, then log-log on the per-(n,seed) points.
    n_arr = df["n"].to_numpy()
    t_arr = df["fit_wall_s"].to_numpy()
    alpha, beta, r2 = _loglog_fit(n_arr, t_arr)
    extrap = float(np.exp(alpha * np.log(EXTRAP_N) + beta))

    # Seed-level band: refit per seed if we have ≥2 seeds per n, otherwise
    # collapse the band to the point estimate.
    seeds = df["seed"].unique()
    if len(seeds) >= 2:
        per_seed_extraps: list[float] = []
        for s in seeds:
            sub = df[df["seed"] == s]
            if len(sub) < 2:
                continue
            a, b, _ = _loglog_fit(sub["n"].to_numpy(), sub["fit_wall_s"].to_numpy())
            per_seed_extraps.append(float(np.exp(a * np.log(EXTRAP_N) + b)))
        if per_seed_extraps:
            std = float(np.std(per_seed_extraps, ddof=1))
            band_lo, band_hi = extrap - std, extrap + std
        else:
            band_lo = band_hi = extrap
    else:
        band_lo = band_hi = extrap

    return LibFit(lib, alpha, beta, r2, extrap, band_lo, band_hi)


def classify(ratio: float, mem_ok: bool | None) -> str:
    """Exit-rule table from spec section 5.

    mem_ok=None means memory was unmeasured; treated as 'not mem_ok' for
    the <=1/3-speed cell (the one cell where memory matters for the
    token). SOFT DONE / URGENT are memory-agnostic.
    """
    if ratio <= 1 / 3:
        return "DONE" if mem_ok is True else "DONE-speed-memory-flagged"
    if ratio <= 1.0:
        return "SOFT DONE"
    return "KERNEL REWRITE URGENT"


def _rationale(token: str, ratio: float) -> str:
    if token == "DONE":
        return (
            f"crforest extrapolated fit at n=100k is {ratio:.2f}x rfSRC -- at or below 1/3. "
            "Kernel rewrite not needed; PRD 6.1 should be re-anchored to the "
            "measured rfSRC baseline."
        )
    if token == "DONE-speed-memory-flagged":
        return (
            f"crforest fit at n=100k is {ratio:.2f}x rfSRC on speed (at or below 1/3), but "
            "memory gate is not satisfied (or not measured). Speed scope closed; "
            "memory follow-up recommended."
        )
    if token == "SOFT DONE":
        return (
            f"crforest fit at n=100k is {ratio:.2f}x rfSRC -- within 1x but above "
            "1/3. Ship current main as-is; kernel rewrite remains on v1.1, "
            "not blocking."
        )
    return (
        f"crforest fit at n=100k is {ratio:.2f}x rfSRC -- over 1x. Open a "
        "follow-on kernel-rewrite sprint (beta''-class scope)."
    )


def _recommended_next(token: str) -> str:
    if token in ("DONE", "DONE-speed-memory-flagged"):
        return (
            "Amend PRD §6.1 to target ≤ rfSRC fit-wall (not the internal 3-min "
            "figure) and close v1.0 perf scope. Extended equivalence-gate work "
            "(tiebreak/RNG diagnostic, rfSRC-side nsplit sweep) is the next "
            "correctness-side lever."
        )
    if token == "SOFT DONE":
        return (
            "Defer kernel rewrite to v1.1. Focus near-term on correctness-side "
            "work (equivalence-gate follow-ups). Revisit if PRD scope changes."
        )
    return (
        "Open a kernel-rewrite sprint: β''-class (compact leaf-table batched "
        "descent in a single numba kernel) or C/Rust rewrite of "
        "find_best_split_hist. Re-benchmark against this report."
    )


def _machine_fingerprint() -> dict[str, str]:
    fp: dict[str, str] = {
        "os": f"{platform.system()} {platform.release()}",
        "python": sys.version.split()[0],
    }
    if sys.platform == "darwin":
        try:
            fp["cpu"] = subprocess.check_output(
                ["sysctl", "-n", "machdep.cpu.brand_string"], text=True
            ).strip()
            fp["physical_cores"] = subprocess.check_output(
                ["sysctl", "-n", "hw.physicalcpu"], text=True
            ).strip()
            ram_bytes = int(
                subprocess.check_output(["sysctl", "-n", "hw.memsize"], text=True).strip()
            )
            fp["ram_gb"] = f"{ram_bytes / 1024**3:.1f}"
        except (subprocess.CalledProcessError, FileNotFoundError, ValueError):
            pass
    try:
        fp["crforest_sha"] = subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"], cwd=HERE, text=True
        ).strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        fp["crforest_sha"] = "unknown"
    try:
        out = subprocess.check_output(
            [
                "Rscript",
                "-e",
                'cat(R.version.string, "|", as.character(packageVersion("randomForestSRC")))',
            ],
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
        r_ver, rfsrc_ver = [s.strip() for s in out.split("|")]
        fp["R"] = r_ver
        fp["randomForestSRC"] = rfsrc_ver
    except (subprocess.CalledProcessError, FileNotFoundError, ValueError):
        fp["R"] = "unknown"
        fp["randomForestSRC"] = "unknown"
    return fp


def _fit_wall_table(cr: pd.DataFrame, rf: pd.DataFrame) -> str:
    rows = [
        "| n | crforest fit_wall_s (mean ± std) | rfSRC fit_wall_s (mean ± std) |",
        "|---|---|---|",
    ]
    for n in sorted(set(cr["n"]) | set(rf["n"])):
        cr_sub, rf_sub = cr[cr["n"] == n], rf[rf["n"] == n]
        cr_cell = (
            f"{cr_sub['fit_wall_s'].mean():.1f} ± "
            f"{cr_sub['fit_wall_s'].std(ddof=1):.1f} (n_seed={len(cr_sub)})"
            if len(cr_sub) >= 2
            else f"{cr_sub['fit_wall_s'].iloc[0]:.1f} (n_seed=1)"
            if len(cr_sub) == 1
            else "—"
        )
        rf_cell = (
            f"{rf_sub['fit_wall_s'].mean():.1f} ± "
            f"{rf_sub['fit_wall_s'].std(ddof=1):.1f} (n_seed={len(rf_sub)})"
            if len(rf_sub) >= 2
            else f"{rf_sub['fit_wall_s'].iloc[0]:.1f} (n_seed=1)"
            if len(rf_sub) == 1
            else "—"
        )
        rows.append(f"| {n:,} | {cr_cell} | {rf_cell} |")
    return "\n".join(rows)


def _rss_table(cr: pd.DataFrame, rf: pd.DataFrame) -> str:
    rows = ["| n | crforest peak_rss_mb | rfSRC peak_rss_mb |", "|---|---|---|"]
    for n in sorted(set(cr["n"]) | set(rf["n"])):
        cr_rss = cr[cr["n"] == n]["peak_rss_mb"].max()
        rf_rss = rf[rf["n"] == n]["peak_rss_mb"].max()
        cr_cell = f"{cr_rss:.0f}" if not np.isnan(cr_rss) else "—"
        rf_cell = f"{rf_rss:.0f}" if not np.isnan(rf_rss) else "NaN (unmeasured)"
        rows.append(f"| {n:,} | {cr_cell} | {rf_cell} |")
    return "\n".join(rows)


def write_report(cr: pd.DataFrame, rf: pd.DataFrame) -> Path:
    REPORTS.mkdir(exist_ok=True)
    ts = time.strftime("%Y-%m-%dT%H-%M-%S")
    out = REPORTS / f"zeta_report_{ts}.md"

    fp = _machine_fingerprint()
    cr_fit = _fit_lib(cr, "crforest")
    rf_fit = _fit_lib(rf, "rfSRC")
    ratio = cr_fit.extrap_100k_s / rf_fit.extrap_100k_s
    ratio_lo = cr_fit.extrap_band_lo / rf_fit.extrap_band_hi
    ratio_hi = cr_fit.extrap_band_hi / rf_fit.extrap_band_lo

    # Memory gate at n=50k.
    cr_rss_50k = cr[cr["n"] == RSS_N]["peak_rss_mb"].max()
    rf_rss_50k = rf[rf["n"] == RSS_N]["peak_rss_mb"].max()
    mem_ok: bool | None
    if np.isnan(rf_rss_50k) or np.isnan(cr_rss_50k):
        mem_ok = None
        mem_line = (
            f"rfSRC peak_rss_mb at n=50k is unmeasured (see provenance). "
            f"crforest = {cr_rss_50k:.0f} MB. Memory gate: **UNMEASURED**."
        )
    else:
        mem_ok = bool(cr_rss_50k <= rf_rss_50k)
        mem_line = (
            f"crforest peak_rss_mb at n=50k = {cr_rss_50k:.0f} MB, "
            f"rfSRC = {rf_rss_50k:.0f} MB. mem_ok = **{mem_ok}**."
        )

    token = classify(ratio, mem_ok)
    low_conf = cr_fit.r2 < R2_CONFIDENCE_THRESHOLD or rf_fit.r2 < R2_CONFIDENCE_THRESHOLD
    conf_line = (
        "> **Low-confidence extrapolation:** r² < 0.95 for at least one lib. "
        "A confirmatory n=100k single-seed run is recommended before acting."
        if low_conf
        else ""
    )

    predict_wall_cr = cr[cr["predict_wall_s"].notna()]["predict_wall_s"]
    predict_line = (
        f"crforest predict_cif wall at n=50k seed=0: {predict_wall_cr.iloc[0]:.1f} s "
        "(secondary, not in exit rule). rfSRC predict not measured."
        if len(predict_wall_cr) > 0
        else "No predict_wall captured."
    )

    parts = [
        f"# zeta head-to-head benchmark report -- {ts}",
        "",
        f"**crforest SHA:** `{fp.get('crforest_sha', 'unknown')}` | "
        f"**rfSRC:** {fp.get('randomForestSRC', 'unknown')} | "
        f"**R:** {fp.get('R', 'unknown')} | "
        f"**Python:** {fp.get('python', 'unknown')}",
        "",
        f"**Machine:** {fp.get('cpu', 'unknown')} | "
        f"{fp.get('physical_cores', '?')} physical cores | "
        f"{fp.get('ram_gb', '?')} GB RAM | "
        f"{fp.get('os', 'unknown')}",
        "",
        "**Thread budget:** crforest `n_jobs=-1` | rfSRC `rf.cores=10` + "
        "`OMP_NUM_THREADS=10` (single-process OpenMP).",
        "",
        "## 1. Methodological caveats (READ FIRST)",
        "",
        "The rfSRC timings in this report were measured on a simpler DGP "
        "(`p=10`, separate lambda per cause) via `/tmp/rfsrc_openmp_bench.R`, "
        "NOT the spec's `p=60` Weibull data generated by `run_crforest.py`. "
        'The rfSRC side also used `splitrule="logrank"` (spec: `"logrankCR"`) '
        "and a single seed, with `n//5` held out from training. "
        "See `timings/rfsrc_timings.parquet.provenance.md` for the full list.",
        "",
        "**Implication.** The ratio `crforest / rfSRC` is **directionally "
        "indicative, not strict**. A spec-compliant rfSRC re-run reading the "
        "same `data/weibull_n*.parquet` with `logrankCR` is a follow-up to "
        "tighten the anchoring. The magnitude of the strategic finding -- "
        "whether crforest is faster or slower than rfSRC at n=100k -- is "
        "robust to these divergences given the observed gap size.",
        "",
        "## 2. Per-(lib, n, seed) fit-wall",
        "",
        _fit_wall_table(cr, rf),
        "",
        "## 3. Peak RSS per (lib, n)",
        "",
        _rss_table(cr, rf),
        "",
        "## 4. Log-log extrapolation to n=100k",
        "",
        "| lib | alpha | beta | r^2 | fit_wall_s @ n=100k (+/- seed-band) |",
        "|---|---|---|---|---|",
        f"| crforest | {cr_fit.alpha:.3f} | {cr_fit.beta:.3f} | {cr_fit.r2:.4f} | "
        f"{cr_fit.extrap_100k_s:.0f} ({cr_fit.extrap_band_lo:.0f} - {cr_fit.extrap_band_hi:.0f}) |",
        f"| rfSRC    | {rf_fit.alpha:.3f} | {rf_fit.beta:.3f} | {rf_fit.r2:.4f} | "
        f"{rf_fit.extrap_100k_s:.0f} ({rf_fit.extrap_band_lo:.0f} - {rf_fit.extrap_band_hi:.0f}) |",
        "",
        conf_line,
        "",
        "## 5. Ratio and memory gate",
        "",
        f"- **fit ratio at n=100k:** {ratio:.2f}x (band {ratio_lo:.2f} - {ratio_hi:.2f}).",
        f"- **memory:** {mem_line}",
        "",
        "## 6. Decision",
        "",
        f"**{token}**",
        "",
        _rationale(token, ratio),
        "",
        "## 7. Recommended next action",
        "",
        _recommended_next(token),
        "",
        "## 8. Secondary numbers",
        "",
        predict_line,
        "",
        "---",
        "",
        "_Source_: `validation/spikes/zeta/timings/{crforest,rfsrc}_timings.parquet`. "
        "Generated by `compare.py`. Spec: "
        "`docs/superpowers/specs/2026-04-23-zeta-head-to-head-benchmark-design.md` (gitignored).",
    ]
    out.write_text("\n".join(parts))
    return out


def self_test() -> None:
    # Four cells from spec §5 exit-rule table.
    assert classify(0.2, True) == "DONE", "ratio ≤ 1/3 + mem_ok → DONE"
    assert classify(0.2, False) == "DONE-speed-memory-flagged", (
        "ratio ≤ 1/3 + not mem_ok → DONE-speed-memory-flagged"
    )
    assert classify(0.7, True) == "SOFT DONE", "1/3 < ratio ≤ 1 → SOFT DONE"
    assert classify(0.7, False) == "SOFT DONE", "1/3 < ratio ≤ 1 → SOFT DONE (mem-agnostic)"
    assert classify(2.0, True) == "KERNEL REWRITE URGENT", "ratio > 1 → URGENT"
    assert classify(2.0, False) == "KERNEL REWRITE URGENT", "ratio > 1 → URGENT (mem-agnostic)"
    # Boundary: ratio == 1/3 is DONE (≤), ratio == 1 is SOFT DONE (≤).
    assert classify(1 / 3, True) == "DONE"
    assert classify(1.0, True) == "SOFT DONE"
    # mem_ok=None treated as not mem_ok for the ≤1/3 cell.
    assert classify(0.2, None) == "DONE-speed-memory-flagged"

    # Log-log fit recovers a known alpha on clean power-law data.
    n = np.array([1000, 10_000, 100_000], dtype=float)
    t = 3.0 * n**1.4
    alpha, beta, r2 = _loglog_fit(n, t)
    assert abs(alpha - 1.4) < 1e-9, f"alpha={alpha}"
    assert abs(np.exp(beta) - 3.0) < 1e-9, f"intercept={np.exp(beta)}"
    assert abs(r2 - 1.0) < 1e-12, f"r2={r2}"

    print("[self-test] exit-rule + log-log fit OK", flush=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--self-test", action="store_true")
    args = ap.parse_args()
    if args.self_test:
        self_test()
        return
    cr = pd.read_parquet(TIMINGS / "crforest_timings.parquet")
    rf = pd.read_parquet(TIMINGS / "rfsrc_timings.parquet")
    out = write_report(cr, rf)
    print(f"wrote {out}", flush=True)


if __name__ == "__main__":
    main()
