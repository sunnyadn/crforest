"""Tiebreak / RNG alignment diagnostic.

Decomposes the ~5% residual cross-lib p95 CIF gap (seen on hd/follic even
at nsplit=100 or 0) by removing randomness sources one at a time from
both libs. Tracks how much of the residual collapses at each step; what
remains at the fully-deterministic config is pure algorithmic divergence
(binning granularity + tiebreak ordering).

Configs swept on hd (both libs matched at each cell):
    A_default       bootstrap=T  mtry=sqrt(p)  nsplit=10  ntree=500
    B_no_bootstrap  bootstrap=F  mtry=sqrt(p)  nsplit=10  ntree=500
    C_full_mtry     bootstrap=F  mtry=p        nsplit=10  ntree=500
    D_exhaustive    bootstrap=F  mtry=p        nsplit=0   ntree=500
    E_single_tree   bootstrap=F  mtry=p        nsplit=0   ntree=1
    F_no_time_coarsen  E + comprisk split_ntime=None (logrank on full time grid).
    G_strict_alignment F + rfSRC ntime=0 (full event times on rfSRC output grid,
                       so both libs' CIF grids have no coarsening bias).

Reads residual collapse A->B->C->D->E. If residual stays near baseline
through D but collapses at E, the ensemble smooths pure noise. If it
stays near baseline at E, the mechanism is algorithmic divergence on
the shared deterministic tree.

Run:
    uv run --extra maintainer python -m validation.alignment.tiebreak_diagnostic \\
        --dataset hd --seeds 10
"""

from __future__ import annotations

import argparse
import datetime as _dt
import sys
import time as _time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from comprisk import CompetingRiskForest
from validation.alignment.equivalence_gate import (
    QUANTILE_GRID,
    _git_sha,
    _machine_fingerprint,
    aggregate_dataset,
    build_reference_grid,
    eval_on_ref_grid,
)
from validation.datasets import load as load_dataset
from validation.splits import _SPLITS_DIR


@dataclass(frozen=True)
class Config:
    name: str
    bootstrap: bool
    mtry: str  # "sqrt" or "full"
    nsplit: int
    ntree: int
    notes: str
    split_ntime: int | None = 50  # comprisk default; None = full time grid
    rf_ntime: int = 150  # rfSRC default output-grid size; 0 = all event times


CONFIGS: tuple[Config, ...] = (
    Config(
        "A_default", bootstrap=True, mtry="sqrt", nsplit=10, ntree=500, notes="production defaults"
    ),
    Config(
        "B_no_bootstrap", bootstrap=False, mtry="sqrt", nsplit=10, ntree=500, notes="bootstrap off"
    ),
    Config(
        "C_full_mtry",
        bootstrap=False,
        mtry="full",
        nsplit=10,
        ntree=500,
        notes="+ no mtry subsampling",
    ),
    Config(
        "D_exhaustive", bootstrap=False, mtry="full", nsplit=0, ntree=500, notes="+ both exhaustive"
    ),
    Config(
        "E_single_tree",
        bootstrap=False,
        mtry="full",
        nsplit=0,
        ntree=1,
        notes="+ single deterministic tree",
    ),
    Config(
        "F_no_time_coarsen",
        bootstrap=False,
        mtry="full",
        nsplit=0,
        ntree=1,
        notes="+ comprisk split_ntime=None",
        split_ntime=None,
    ),
    Config(
        "G_strict_alignment",
        bootstrap=False,
        mtry="full",
        nsplit=0,
        ntree=1,
        notes="+ rfSRC ntime=0 (full event times on output grid)",
        split_ntime=None,
        rf_ntime=0,
    ),
)


def _fit_comprisk(
    X_tr: np.ndarray,
    t_tr: np.ndarray,
    e_tr: np.ndarray,
    X_te: np.ndarray,
    *,
    seed: int,
    cfg: Config,
    p: int,
) -> dict:
    max_features: int | str = p if cfg.mtry == "full" else "sqrt"
    forest = CompetingRiskForest(
        n_estimators=cfg.ntree,
        min_samples_leaf=1,
        min_samples_split=30,
        max_features=max_features,
        bootstrap=cfg.bootstrap,
        random_state=seed,
        mode="default",
        time_grid=200,
        nsplit=cfg.nsplit,
        split_ntime=cfg.split_ntime,
    ).fit(X_tr, t_tr, e_tr)
    cif = forest.predict_cif(X_te)  # (n_test, n_cause, n_time)
    cif = np.transpose(cif, (0, 2, 1))  # -> (n_test, n_time, n_cause)
    return {"time_grid": np.asarray(forest.time_grid_, dtype=np.float64), "cif": cif}


def _fit_rfsrc(
    X_tr: np.ndarray,
    t_tr: np.ndarray,
    e_tr: np.ndarray,
    X_te: np.ndarray,
    *,
    seed: int,
    cfg: Config,
    p: int,
) -> dict:
    import rpy2.robjects as ro
    from rpy2.robjects.conversion import localconverter
    from rpy2.robjects.packages import importr

    from validation.alignment import _rpy2_converter

    importr("randomForestSRC")
    converter = _rpy2_converter()

    feat_cols = [f"x{j}" for j in range(p)]
    train_df = pd.DataFrame(X_tr, columns=feat_cols)
    train_df["time"] = t_tr
    train_df["event"] = e_tr.astype(np.int32)
    test_df = pd.DataFrame(X_te, columns=feat_cols)

    with localconverter(converter):
        ro.globalenv["train_df"] = train_df
        ro.globalenv["test_df"] = test_df

    bootstrap_arg = '"by.root"' if cfg.bootstrap else '"none"'
    mtry_expr = f"{p}" if cfg.mtry == "full" else f"ceiling(sqrt({p}))"
    ro.r(
        f"""
        fit__ <- rfsrc(
            Surv(time, event) ~ .,
            data       = train_df,
            ntree      = {cfg.ntree},
            nodesize   = 15,
            mtry       = {mtry_expr},
            splitrule  = "logrankCR",
            bootstrap  = {bootstrap_arg},
            samptype   = "swr",
            nsplit     = {cfg.nsplit},
            ntime      = {cfg.rf_ntime},
            seed       = -{int(seed)}
        )
        pred__ <- predict(fit__, newdata = test_df)
        """
    )
    with localconverter(converter):
        time_interest = np.asarray(ro.r("fit__$time.interest"), dtype=np.float64)
        cif_flat = np.asarray(ro.r("pred__$cif"), dtype=np.float64)
        ro.r("rm(fit__); rm(pred__)")

    n_test = X_te.shape[0]
    n_time = len(time_interest)
    if cif_flat.size != n_test * n_time * 2:
        raise RuntimeError(
            f"rfSRC pred$cif size {cif_flat.size} != {n_test}*{n_time}*2 (n_causes assumed 2)"
        )
    cif_rf = cif_flat.reshape(n_test, n_time, 2)
    return {"time_grid": time_interest, "cif": cif_rf}


def _run_cell(dataset: str, seed: int, cfg: Config) -> dict:
    X, time, event = load_dataset(dataset)
    ref_grid = build_reference_grid(time, event)

    splits_df = pd.read_parquet(_SPLITS_DIR / f"{dataset}.parquet")
    row = splits_df[splits_df["seed"] == seed]
    if row.empty:
        raise RuntimeError(f"seed {seed} not in splits for {dataset}")
    train_idx = np.sort(row.loc[row["fold"] == "train", "sample_id"].to_numpy(np.int64))
    test_idx = np.sort(row.loc[row["fold"] == "test", "sample_id"].to_numpy(np.int64))
    p = X.shape[1]

    t0 = _time.perf_counter()
    print(
        f"[{cfg.name} ds={dataset} seed={seed}] cr_fit_start n_train={len(train_idx)} p={p}",
        flush=True,
    )
    cr = _fit_comprisk(
        X[train_idx], time[train_idx], event[train_idx], X[test_idx], seed=seed, cfg=cfg, p=p
    )
    print(f"[{cfg.name} seed={seed}] cr_fit_done wall={_time.perf_counter() - t0:.1f}s", flush=True)

    t0 = _time.perf_counter()
    print(f"[{cfg.name} seed={seed}] rf_fit_start", flush=True)
    rf = _fit_rfsrc(
        X[train_idx], time[train_idx], event[train_idx], X[test_idx], seed=seed, cfg=cfg, p=p
    )
    print(f"[{cfg.name} seed={seed}] rf_fit_done wall={_time.perf_counter() - t0:.1f}s", flush=True)

    cause_idx = 0  # cause 1
    cr_cif_native = cr["cif"][:, :, cause_idx]
    rf_cif_native = rf["cif"][:, :, cause_idx]
    cif_cr = eval_on_ref_grid(cr_cif_native, cr["time_grid"], ref_grid)
    cif_rf = eval_on_ref_grid(rf_cif_native, rf["time_grid"], ref_grid)
    return {
        "seed": seed,
        "cif_cr": cif_cr,
        "cif_rf": cif_rf,
        "risk_cr": cif_cr[:, -1].copy(),
        "risk_rf": cif_rf[:, -1].copy(),
    }


def _format_table(rows: list[dict]) -> str:
    out = [
        "| cell | bootstrap | mtry | nsplit | ntree | cross_p95_cif | cross_p95_risk "
        "| within_cr_p95_cif | within_rf_p95_cif |",
        "|---|---|---|---|---|---|---|---|---|",
    ]
    for r in rows:
        out.append(
            f"| {r['name']} | {r['bootstrap']} | {r['mtry']} | {r['nsplit']} | {r['ntree']} "
            f"| {r['cross_p95_cif']:.4f} | {r['cross_p95_risk']:.4f} "
            f"| {r['within_cr_p95_cif']:.4f} | {r['within_rf_p95_cif']:.4f} |"
        )
    return "\n".join(out)


def _format_quantiles(rows: list[dict]) -> str:
    out = ["| cell | q0.50 | q0.75 | q0.90 | q0.95 | q0.99 |", "|---|---|---|---|---|---|"]
    for r in rows:
        q = r["quantiles"]["cross_cif"]
        out.append(f"| {r['name']} | " + " | ".join(f"{q[qq]:.4f}" for qq in QUANTILE_GRID) + " |")
    return "\n".join(out)


def _attribution(rows: list[dict]) -> str:
    """Cross_p95_cif delta attributed to each config step A->B->C->D->E.

    Positive delta means residual shrank when moving to the tighter config.
    """
    lines = ["| step | delta cross_p95_cif | interpretation |", "|---|---|---|"]
    meaning = {
        ("A_default", "B_no_bootstrap"): "bootstrap sampling contribution",
        ("B_no_bootstrap", "C_full_mtry"): "mtry feature-subsampling contribution",
        ("C_full_mtry", "D_exhaustive"): "nsplit randomness contribution",
        ("D_exhaustive", "E_single_tree"): "ensemble aggregation contribution",
        ("E_single_tree", "F_no_time_coarsen"): "comprisk split_ntime coarsening contribution",
        (
            "F_no_time_coarsen",
            "G_strict_alignment",
        ): "rfSRC ntime output-grid coarsening contribution",
    }
    by_name = {r["name"]: r for r in rows}
    for (a, b), label in meaning.items():
        if a in by_name and b in by_name:
            delta = by_name[a]["cross_p95_cif"] - by_name[b]["cross_p95_cif"]
            lines.append(f"| {a} -> {b} | {delta:+.4f} | {label} |")
    if "E_single_tree" in by_name:
        residual = by_name["E_single_tree"]["cross_p95_cif"]
        lines.append(f"| (remainder at E) | {residual:.4f} | algorithmic divergence floor |")
    return "\n".join(lines)


def _verdict(rows: list[dict]) -> str:
    by_name = {r["name"]: r for r in rows}
    a = by_name.get("A_default", {}).get("cross_p95_cif")
    e = by_name.get("E_single_tree", {}).get("cross_p95_cif")
    if a is None or e is None:
        return "(verdict unavailable - A or E missing)"

    shrink = (a - e) / a if a > 0 else 0.0
    if e < 0.01:
        return (
            f"**Verdict.** At the fully-deterministic config (E), cross_p95_cif collapses "
            f"to {e:.4f} (from {a:.4f} at default, {shrink:.0%} reduction). The hd-tail "
            "residual is dominated by ensemble sampling noise (bootstrap + mtry + nsplit); "
            "single-tree determinism closes the gap. Algorithmic divergence is below 0.01."
        )
    if e < a * 0.5:
        return (
            f"**Verdict.** At the fully-deterministic config (E), cross_p95_cif shrinks to "
            f"{e:.4f} (from {a:.4f}, {shrink:.0%} reduction). Sampling noise explains the "
            f"majority, but an algorithmic-divergence floor of {e:.4f} remains even on a "
            "single deterministic tree."
        )
    return (
        f"**Verdict.** At the fully-deterministic config (E), cross_p95_cif is still "
        f"{e:.4f} (vs {a:.4f} at default, {shrink:.0%} reduction). Removing sampling "
        "sources does NOT close the gap. The residual is genuinely algorithmic "
        "(binning granularity on the comprisk side, tiebreak ordering, or both). "
        "Further diagnostic should walk the single tree's split choices."
    )


def _write_report(
    *,
    dataset: str,
    rows: list[dict],
    header: dict,
    path: Path,
) -> None:
    lines = [
        "# Tiebreak / RNG alignment diagnostic",
        "",
        f"Timestamp: {header['timestamp']}",
        f"Dataset: `{dataset}`  |  seeds: {header['n_seeds']}  |  commit: {header['commit_sha']}",
        f"Machine: {header['machine']}",
        "",
        "## Attribution of the cross-lib residual",
        "",
        "Each row of the main table removes one more randomness source from both libs.",
        "The final cell E is fit as a single deterministic tree on both sides; any "
        "residual there is pure algorithmic divergence.",
        "",
        _format_table(rows),
        "",
        "## Delta attribution",
        "",
        _attribution(rows),
        "",
        "## Quantile-dominance (cross-lib |Delta CIF|) per config",
        "",
        _format_quantiles(rows),
        "",
        "## Verdict",
        "",
        _verdict(rows),
        "",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", default="hd")
    parser.add_argument("--seeds", type=int, default=10)
    parser.add_argument("--out", default="validation/reports")
    parser.add_argument(
        "--configs",
        nargs="*",
        default=None,
        help="Subset of cell names to run (default: all 5).",
    )
    args = parser.parse_args(argv)

    if args.seeds % 2 != 0:
        parser.error("--seeds must be even (within-lib pairing)")

    active_cfgs = CONFIGS
    if args.configs:
        by_name = {c.name: c for c in CONFIGS}
        missing = [n for n in args.configs if n not in by_name]
        if missing:
            parser.error(f"unknown configs: {missing}; choose from {list(by_name)}")
        active_cfgs = tuple(by_name[n] for n in args.configs)

    rows: list[dict] = []
    for cfg in active_cfgs:
        cells: list[dict] = []
        for s in range(args.seeds):
            cells.append(_run_cell(args.dataset, s, cfg))
        agg = aggregate_dataset(cells)
        rows.append(
            {
                "name": cfg.name,
                "bootstrap": str(cfg.bootstrap),
                "mtry": cfg.mtry,
                "nsplit": cfg.nsplit,
                "ntree": cfg.ntree,
                **agg,
            }
        )
        print(
            f"[{cfg.name}] cross_p95_cif={agg['cross_p95_cif']:.4f} "
            f"cross_p95_risk={agg['cross_p95_risk']:.4f}",
            flush=True,
        )

    ts = _dt.datetime.now().isoformat(timespec="seconds")
    header = {
        "timestamp": ts,
        "commit_sha": _git_sha(),
        "python_version": sys.version.split()[0],
        "machine": _machine_fingerprint(),
        "n_seeds": args.seeds,
    }
    out_path = Path(args.out) / f"tiebreak_diagnostic_{ts.replace(':', '-')}.md"
    _write_report(dataset=args.dataset, rows=rows, header=header, path=out_path)
    print(f"wrote {out_path}", flush=True)
    print(_verdict(rows), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
