"""Aligned-mode CIF gap + fit-time sweep across all 4 datasets.

Same shape as mode_vs_perf.py, but with the full Phase 1c alignment recipe:

- comprisk: ``equivalence="rfsrc"`` preset (flips rng_mode + split_ntime;
  exposes ``forest.inbag_``), mode in {default, reference}
- rfSRC: ``bootstrap="by.user"`` with ``forest.inbag_``, ``nsplit=10``,
  ``ntime=0``, ``nodesize=15``, same mtry

Reports cross-lib p95 CIF gap on the reference event-time grid plus wall-clock
fit time for each of (comprisk default+aligned, comprisk reference+aligned,
rfSRC bootstrap=by.user).

Run:
    uv run --extra maintainer python -m validation.alignment.mode_vs_perf_aligned \\
        --seeds 3 --n-estimators 100
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd

from comprisk import CompetingRiskForest
from validation.alignment import _rpy2_converter
from validation.alignment.equivalence_gate import (
    build_reference_grid,
    eval_on_ref_grid,
)
from validation.datasets import load as load_dataset
from validation.splits import _SPLITS_DIR

DATASETS = ("pbc", "hd", "follic", "synthetic")


def _fit_rfsrc_aligned(
    X_train,
    time_train,
    event_train,
    X_test,
    seed,
    n_estimators,
    inbag,
):
    import rpy2.robjects as ro
    from rpy2.robjects.conversion import localconverter
    from rpy2.robjects.packages import importr

    importr("randomForestSRC")
    converter = _rpy2_converter()

    p = X_train.shape[1]
    n_tr = X_train.shape[0]
    feat_cols = [f"x{j}" for j in range(p)]
    train_df = pd.DataFrame(X_train, columns=feat_cols)
    train_df["time"] = time_train
    train_df["event"] = event_train.astype(np.int32)
    test_df = pd.DataFrame(X_test, columns=feat_cols)

    with localconverter(converter):
        ro.globalenv["train_df"] = train_df
        ro.globalenv["test_df"] = test_df
        ro.globalenv["samp_matrix"] = ro.r.matrix(
            ro.FloatVector(inbag.T.reshape(-1).astype(np.float64)),
            nrow=n_tr,
            ncol=n_estimators,
        )

    t0 = time.perf_counter()
    ro.r(
        f"""
        fit__ <- rfsrc(Surv(time, event) ~ ., data=train_df,
            ntree={n_estimators}, nodesize=15, mtry=ceiling(sqrt({p})),
            splitrule="logrankCR", bootstrap="by.user", samp=samp_matrix,
            nsplit=10, ntime=0, seed=-{int(seed)})
        """
    )
    fit_wall = time.perf_counter() - t0
    ro.r("pred__ <- predict(fit__, newdata=test_df)")

    with localconverter(converter):
        ti = np.asarray(ro.r("fit__$time.interest"), dtype=np.float64)
        cif_flat = np.asarray(ro.r("pred__$cif"), dtype=np.float64)
        ro.r("rm(fit__); rm(pred__); rm(samp_matrix)")

    n_te = X_test.shape[0]
    cif = cif_flat.reshape(n_te, len(ti), 2)
    return ti, cif, fit_wall


def _fit_comprisk_aligned(
    X_train,
    time_train,
    event_train,
    X_test,
    seed,
    n_estimators,
    mode,
):
    kwargs = dict(
        n_estimators=n_estimators,
        min_samples_leaf=1,
        min_samples_split=30,
        max_features="sqrt",
        bootstrap=True,
        random_state=seed,
        mode=mode,
        equivalence="rfsrc",
    )
    if mode == "default":
        kwargs["time_grid"] = 200
    t0 = time.perf_counter()
    forest = CompetingRiskForest(**kwargs).fit(X_train, time_train, event_train)
    fit_wall = time.perf_counter() - t0
    cif = np.transpose(forest.predict_cif(X_test), (0, 2, 1))
    grid = np.asarray(
        forest.time_grid_ if mode == "default" else forest.unique_times_,
        dtype=np.float64,
    )
    return grid, cif, fit_wall, forest


def run_cell(dataset: str, seed: int, n_estimators: int, skip_reference: bool) -> dict:
    X, time_all, event_all = load_dataset(dataset)
    splits_df = pd.read_parquet(_SPLITS_DIR / f"{dataset}.parquet")
    row = splits_df[splits_df["seed"] == seed]
    train_idx = np.sort(row.loc[row["fold"] == "train", "sample_id"].to_numpy(np.int64))
    test_idx = np.sort(row.loc[row["fold"] == "test", "sample_id"].to_numpy(np.int64))
    ref_grid = build_reference_grid(time_all, event_all)

    X_tr, t_tr, e_tr = X[train_idx], time_all[train_idx], event_all[train_idx]
    X_te = X[test_idx]
    n_tr = len(train_idx)

    grid_def, cif_def, t_def, forest_def = _fit_comprisk_aligned(
        X_tr, t_tr, e_tr, X_te, seed, n_estimators, "default"
    )
    if skip_reference:
        grid_ref = np.array([0.0])
        cif_ref = np.zeros((X_te.shape[0], 1, 2))
        t_ref = float("nan")
    else:
        grid_ref, cif_ref, t_ref, _ = _fit_comprisk_aligned(
            X_tr, t_tr, e_tr, X_te, seed, n_estimators, "reference"
        )
    # Use the default-mode forest's inbag_ to feed rfSRC. Reference and default
    # share the same bootstrap stream (same random_state), so reusing one
    # inbag matrix keeps the rfSRC fit consistent with both comprisk fits.
    grid_rf, cif_rf, t_rf = _fit_rfsrc_aligned(
        X_tr,
        t_tr,
        e_tr,
        X_te,
        seed,
        n_estimators,
        forest_def.inbag_,
    )

    cif_def_g = eval_on_ref_grid(cif_def[:, :, 0], grid_def, ref_grid)
    cif_rf_g = eval_on_ref_grid(cif_rf[:, :, 0], grid_rf, ref_grid)
    p95_def = float(np.percentile(np.abs(cif_def_g - cif_rf_g), 95))
    if skip_reference:
        p95_ref = float("nan")
    else:
        cif_ref_g = eval_on_ref_grid(cif_ref[:, :, 0], grid_ref, ref_grid)
        p95_ref = float(np.percentile(np.abs(cif_ref_g - cif_rf_g), 95))

    return {
        "dataset": dataset,
        "seed": seed,
        "n": n_tr,
        "p": X.shape[1],
        "p95_default_aligned": p95_def,
        "p95_reference_aligned": p95_ref,
        "fit_default_s": t_def,
        "fit_reference_s": t_ref,
        "fit_rfsrc_s": t_rf,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--datasets", nargs="+", default=list(DATASETS))
    parser.add_argument("--seeds", type=int, default=3)
    parser.add_argument("--n-estimators", type=int, default=100)
    parser.add_argument(
        "--skip-reference-on",
        nargs="*",
        default=["synthetic"],
        help="datasets where reference-mode is too slow to run (default: synthetic)",
    )
    parser.add_argument("--out", type=Path, default=Path("/tmp/mode_vs_perf_aligned.json"))
    args = parser.parse_args()

    rows = []
    for ds in args.datasets:
        skip_ref = ds in args.skip_reference_on
        for s in range(1, args.seeds + 1):
            r = run_cell(ds, s, args.n_estimators, skip_ref)
            rows.append(r)
            ref_str = "skipped" if skip_ref else f"{r['p95_reference_aligned']:.4f}"
            t_ref_str = "—" if skip_ref else f"{r['fit_reference_s']:.1f}"
            print(
                f"[{ds} seed={s}] "
                f"p95(def+aligned)={r['p95_default_aligned']:.4f} "
                f"p95(ref+aligned)={ref_str} | "
                f"fit(s): def={r['fit_default_s']:.1f} ref={t_ref_str} "
                f"rf(by.user)={r['fit_rfsrc_s']:.1f} "
                f"| def/rf={r['fit_default_s'] / r['fit_rfsrc_s']:.2f}",
                flush=True,
            )

    df = pd.DataFrame(rows)
    print("\n=== per-dataset medians (aligned RNG + bootstrap) ===")
    for ds in df["dataset"].unique():
        sub = df[df["dataset"] == ds]
        ref_med = sub["p95_reference_aligned"].median()
        ref_str = f"{ref_med:.4f}" if not np.isnan(ref_med) else "skipped"
        t_ref_med = sub["fit_reference_s"].median()
        t_ref_str = f"{t_ref_med:.2f}" if not np.isnan(t_ref_med) else "—"
        print(
            f"  {ds:>10s}: p95(def)={sub['p95_default_aligned'].median():.4f} "
            f"p95(ref)={ref_str} | "
            f"fit median: def={sub['fit_default_s'].median():.2f} "
            f"ref={t_ref_str} "
            f"rf={sub['fit_rfsrc_s'].median():.2f} | "
            f"def/rf={sub['fit_default_s'].median() / sub['fit_rfsrc_s'].median():.2f}x"
        )

    args.out.write_text(json.dumps(rows, indent=2))
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
