"""Equivalence-gate rerun at strict-alignment time-grid settings.

The tiebreak diagnostic (tiebreak_diagnostic.py) found cell F on hd
bit-identical between crforest and rfSRC -- but only at a fully-
deterministic single-tree config (bootstrap=F, mtry=p, nsplit=0,
ntree=1, split_ntime=None). The conjecture this driver tested was
that the same `split_ntime=None` fix would also close the gate at
production ensemble config.

This driver re-runs the equivalence-gate metrics (cross_p95_cif /
risk / IBS, within-lib noise floor, hard-cap) at the production
ensemble config but with both time-grid knobs removed:

    crforest: split_ntime=None  (logrank on full time grid)
    rfSRC:    ntime=0           (CIF output on all event times)

Everything else remains production default (bootstrap=T, mtry=sqrt(p),
nsplit=10, ntree=500, nodesize=15/min_samples_leaf=1).

**Empirical result (2026-04-24, 10 seeds, all 4 datasets): hypothesis
REFUTED for hd and follic.** cross_p95_cif at strict alignment is
0.0570 on hd (vs 0.0573 at default) -- the split_ntime=None fix
collapses under 500-tree stochastic ensemble averaging. Hard-cap
still fails on hd (cif 0.057) and follic (risk 0.0505); pbc and
synthetic pass. Noise-floor passes 4/4. The production-config
residual is driven by mtry + bootstrap ensemble sampling variance,
not split_ntime. See `project_tiebreak_diagnostic_completion` memory
for the full corrected reading.

Run:
    uv run --extra maintainer python -m validation.alignment.strict_alignment \\
        --seeds 10
"""

from __future__ import annotations

import argparse
import datetime as _dt
import sys
import time as _time
from pathlib import Path

import numpy as np
import pandas as pd

from crforest import CompetingRiskForest
from validation.alignment.equivalence_gate import (
    HARD_CAP_DEFAULT,
    QUANTILE_GRID,
    _aggregate_ibs,
    _git_sha,
    _machine_fingerprint,
    aggregate_dataset,
    apply_tolerance,
    build_reference_grid,
    eval_on_ref_grid,
)
from validation.alignment.ibs import compute_ibs
from validation.datasets import load as load_dataset
from validation.splits import _SPLITS_DIR

DATASETS: tuple[str, ...] = ("pbc", "follic", "hd", "synthetic")


def _fit_crforest_strict(
    X_tr: np.ndarray,
    t_tr: np.ndarray,
    e_tr: np.ndarray,
    X_te: np.ndarray,
    *,
    seed: int,
) -> dict:
    forest = CompetingRiskForest(
        n_estimators=500,
        min_samples_leaf=1,
        min_samples_split=30,
        max_features="sqrt",
        bootstrap=True,
        random_state=seed,
        mode="default",
        time_grid=200,
        nsplit=None,  # uses resolved default (10 in default mode)
        split_ntime=None,  # <-- strict: no time-grid coarsening during split search
    ).fit(X_tr, t_tr, e_tr)
    cif = forest.predict_cif(X_te)  # (n_test, n_cause, n_time)
    cif = np.transpose(cif, (0, 2, 1))  # -> (n_test, n_time, n_cause)
    return {"time_grid": np.asarray(forest.time_grid_, dtype=np.float64), "cif": cif}


def _fit_rfsrc_strict(
    X_tr: np.ndarray,
    t_tr: np.ndarray,
    e_tr: np.ndarray,
    X_te: np.ndarray,
    *,
    seed: int,
) -> dict:
    import rpy2.robjects as ro
    from rpy2.robjects.conversion import localconverter
    from rpy2.robjects.packages import importr

    from validation.alignment import _rpy2_converter

    importr("randomForestSRC")
    converter = _rpy2_converter()

    p = X_tr.shape[1]
    feat_cols = [f"x{j}" for j in range(p)]
    train_df = pd.DataFrame(X_tr, columns=feat_cols)
    train_df["time"] = t_tr
    train_df["event"] = e_tr.astype(np.int32)
    test_df = pd.DataFrame(X_te, columns=feat_cols)

    with localconverter(converter):
        ro.globalenv["train_df"] = train_df
        ro.globalenv["test_df"] = test_df

    ro.r(
        f"""
        fit__ <- rfsrc(
            Surv(time, event) ~ .,
            data       = train_df,
            ntree      = 500,
            nodesize   = 15,
            mtry       = ceiling(sqrt({p})),
            splitrule  = "logrankCR",
            samptype   = "swr",
            nsplit     = 10,
            ntime      = 0,
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


def _run_cell(dataset: str, seed: int) -> dict:
    X, time, event = load_dataset(dataset)
    ref_grid = build_reference_grid(time, event)

    splits_df = pd.read_parquet(_SPLITS_DIR / f"{dataset}.parquet")
    row = splits_df[splits_df["seed"] == seed]
    if row.empty:
        raise RuntimeError(f"seed {seed} not in splits for {dataset}")
    train_idx = np.sort(row.loc[row["fold"] == "train", "sample_id"].to_numpy(np.int64))
    test_idx = np.sort(row.loc[row["fold"] == "test", "sample_id"].to_numpy(np.int64))

    t0 = _time.perf_counter()
    print(
        f"[strict ds={dataset} seed={seed}] cr_fit_start n_train={len(train_idx)} p={X.shape[1]}",
        flush=True,
    )
    cr = _fit_crforest_strict(
        X[train_idx], time[train_idx], event[train_idx], X[test_idx], seed=seed
    )
    print(
        f"[strict ds={dataset} seed={seed}] cr_fit_done wall={_time.perf_counter() - t0:.1f}s",
        flush=True,
    )

    t0 = _time.perf_counter()
    print(f"[strict ds={dataset} seed={seed}] rf_fit_start", flush=True)
    rf = _fit_rfsrc_strict(X[train_idx], time[train_idx], event[train_idx], X[test_idx], seed=seed)
    print(
        f"[strict ds={dataset} seed={seed}] rf_fit_done wall={_time.perf_counter() - t0:.1f}s",
        flush=True,
    )

    cause_idx = 0  # cause 1
    cr_cif_native = cr["cif"][:, :, cause_idx]
    rf_cif_native = rf["cif"][:, :, cause_idx]
    cif_cr = eval_on_ref_grid(cr_cif_native, cr["time_grid"], ref_grid)
    cif_rf = eval_on_ref_grid(rf_cif_native, rf["time_grid"], ref_grid)

    ibs_cr = compute_ibs(
        cif_cr, ref_grid, time[test_idx], event[test_idx], time[train_idx], event[train_idx]
    )
    ibs_rf = compute_ibs(
        cif_rf, ref_grid, time[test_idx], event[test_idx], time[train_idx], event[train_idx]
    )
    return {
        "seed": seed,
        "cif_cr": cif_cr,
        "cif_rf": cif_rf,
        "risk_cr": cif_cr[:, -1].copy(),
        "risk_rf": cif_rf[:, -1].copy(),
        "ibs_cr": ibs_cr,
        "ibs_rf": ibs_rf,
    }


def _ck(b: bool) -> str:
    return "PASS" if b else "FAIL"


def _write_report(
    *,
    per_dataset: dict[str, dict],
    header: dict,
    path: Path,
) -> None:
    lines = [
        "# Equivalence gate at strict-alignment config",
        "",
        f"Timestamp: {header['timestamp']}",
        f"Seeds: {header['n_seeds']}  |  commit: {header['commit_sha']}",
        f"Machine: {header['machine']}",
        "",
        "Config: production defaults (bootstrap=T, mtry=sqrt(p), nsplit=10, "
        "ntree=500, min_samples_leaf=1) with two alignment overrides:",
        "  - crforest `split_ntime=None` (full time grid for logrank eval)",
        "  - rfSRC    `ntime=0`          (full event times on CIF output grid)",
        "",
        f"Hard cap (default): {header['hard_cap']}",
        "",
        "## Gate summary",
        "",
        "| dataset | cross_p95_cif | within_cr_p95_cif | within_rf_p95_cif | "
        "nf cif | hc cif | cross_p95_risk | nf risk | hc risk | cross_p95_ibs | nf ibs | overall |",
        "|---|---|---|---|---|---|---|---|---|---|---|---|",
    ]
    for ds in per_dataset:
        d = per_dataset[ds]
        a, tol = d["agg"], d["tol"]
        ibs = d["ibs"]
        nf_ibs = ibs["cross_p95_ibs"] <= max(ibs["within_cr_p95_ibs"], ibs["within_rf_p95_ibs"])
        lines.append(
            f"| {ds} | {a['cross_p95_cif']:.4f} | {a['within_cr_p95_cif']:.4f} | "
            f"{a['within_rf_p95_cif']:.4f} | {_ck(tol['noise_floor_pass_cif'])} | "
            f"{_ck(tol['hard_cap_pass_cif'])} | {a['cross_p95_risk']:.4f} | "
            f"{_ck(tol['noise_floor_pass_risk'])} | {_ck(tol['hard_cap_pass_risk'])} | "
            f"{ibs['cross_p95_ibs']:.4f} | {_ck(nf_ibs)} | {_ck(tol['overall_pass'])} |"
        )

    lines += ["", "## Quantile-dominance (cross-lib |Delta CIF|) per dataset", ""]
    q_header = "| dataset | " + " | ".join(f"q{qq:g}" for qq in QUANTILE_GRID) + " |"
    q_sep = "|---|" + "---|" * len(QUANTILE_GRID)
    lines += [q_header, q_sep]
    for ds in per_dataset:
        q = per_dataset[ds]["agg"]["quantiles"]["cross_cif"]
        lines.append(f"| {ds} | " + " | ".join(f"{q[qq]:.4f}" for qq in QUANTILE_GRID) + " |")

    overall = all(per_dataset[ds]["tol"]["overall_pass"] for ds in per_dataset)
    lines += [
        "",
        "## Verdict",
        "",
        f"**Overall: {_ck(overall)}** across {len(per_dataset)} datasets.",
        "",
    ]
    if overall:
        lines.append(
            "All four gate datasets pass noise-floor and hard-cap under strict "
            "alignment. The prior hd/follic hard-cap failures are confirmed as "
            "artifacts of the post-epsilon `split_ntime=50` coarsening, not "
            "algorithmic divergence between crforest and rfSRC."
        )
    else:
        failing = [ds for ds in per_dataset if not per_dataset[ds]["tol"]["overall_pass"]]
        lines.append(
            f"Hard-cap or noise-floor fails on: {', '.join(failing)}. Strict "
            "alignment does NOT close the gap on these datasets; further "
            "diagnostic is warranted."
        )

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seeds", type=int, default=10)
    parser.add_argument("--out", default="validation/reports")
    parser.add_argument("--hard-cap", type=float, default=HARD_CAP_DEFAULT)
    parser.add_argument(
        "--datasets",
        nargs="*",
        default=None,
        help=f"Subset (default: {list(DATASETS)}).",
    )
    args = parser.parse_args(argv)
    if args.seeds % 2 != 0:
        parser.error("--seeds must be even (within-lib pairing)")

    active = tuple(args.datasets) if args.datasets else DATASETS
    unknown = [d for d in active if d not in DATASETS]
    if unknown:
        parser.error(f"unknown datasets: {unknown}; choose from {list(DATASETS)}")

    per_dataset: dict[str, dict] = {}
    for ds in active:
        cells = [_run_cell(ds, s) for s in range(args.seeds)]
        agg = aggregate_dataset(cells)
        ibs = _aggregate_ibs(cells)
        tol = apply_tolerance(agg, hard_cap=args.hard_cap)
        per_dataset[ds] = {"agg": agg, "ibs": ibs, "tol": tol}
        print(
            f"[strict {ds}] cross_p95_cif={agg['cross_p95_cif']:.4f} "
            f"cross_p95_risk={agg['cross_p95_risk']:.4f} "
            f"cross_p95_ibs={ibs['cross_p95_ibs']:.4f} "
            f"overall={'PASS' if tol['overall_pass'] else 'FAIL'}",
            flush=True,
        )

    ts = _dt.datetime.now().isoformat(timespec="seconds")
    header = {
        "timestamp": ts,
        "commit_sha": _git_sha(),
        "python_version": sys.version.split()[0],
        "machine": _machine_fingerprint(),
        "hard_cap": args.hard_cap,
        "n_seeds": args.seeds,
    }
    out_path = Path(args.out) / f"strict_alignment_{ts.replace(':', '-')}.md"
    _write_report(per_dataset={ds: per_dataset[ds] for ds in active}, header=header, path=out_path)
    print(f"wrote {out_path}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
