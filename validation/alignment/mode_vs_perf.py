"""Sweep: default vs reference crforest modes, CIF gap + fit-time vs rfSRC.

Runs production-like config (bootstrap=True, mtry=sqrt(p), nsplit=0) on all
four paired-seed datasets across a small number of seeds, recording both:

- cross-lib p95 CIF gap on the reference event-time grid
- wall-clock fit time for each of (crforest default, crforest reference, rfSRC)

Purpose: the equivalence-vs-performance tradeoff requires both numbers in
the same table to make a ship decision.

Run:
    uv run --extra maintainer python -m validation.alignment.mode_vs_perf \\
        --seeds 3 --n-estimators 100
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd

from crforest import CompetingRiskForest
from validation.alignment import _rpy2_converter
from validation.alignment.equivalence_gate import (
    build_reference_grid,
    eval_on_ref_grid,
)
from validation.datasets import load as load_dataset
from validation.splits import _SPLITS_DIR

DATASETS = ("pbc", "hd", "follic", "synthetic")


def _fit_rfsrc(
    X_train: np.ndarray,
    time_train: np.ndarray,
    event_train: np.ndarray,
    X_test: np.ndarray,
    seed: int,
    n_estimators: int,
) -> tuple[np.ndarray, np.ndarray, float]:
    import rpy2.robjects as ro
    from rpy2.robjects.conversion import localconverter
    from rpy2.robjects.packages import importr

    importr("randomForestSRC")
    converter = _rpy2_converter()

    p = X_train.shape[1]
    feat_cols = [f"x{j}" for j in range(p)]
    train_df = pd.DataFrame(X_train, columns=feat_cols)
    train_df["time"] = time_train
    train_df["event"] = event_train.astype(np.int32)
    test_df = pd.DataFrame(X_test, columns=feat_cols)

    with localconverter(converter):
        ro.globalenv["train_df"] = train_df
        ro.globalenv["test_df"] = test_df

    t0 = time.perf_counter()
    ro.r(
        f"""
        fit__ <- rfsrc(Surv(time, event) ~ ., data=train_df,
            ntree={n_estimators}, nodesize=15, mtry=ceiling(sqrt({p})),
            splitrule="logrankCR", samptype="swr",
            nsplit=0, seed=-{int(seed)})
        """
    )
    fit_wall = time.perf_counter() - t0
    ro.r("pred__ <- predict(fit__, newdata=test_df)")

    with localconverter(converter):
        ti = np.asarray(ro.r("fit__$time.interest"), dtype=np.float64)
        cif_flat = np.asarray(ro.r("pred__$cif"), dtype=np.float64)
        ro.r("rm(fit__); rm(pred__)")

    n_te = X_test.shape[0]
    cif = cif_flat.reshape(n_te, len(ti), 2)
    return ti, cif, fit_wall


def _fit_crforest(X_train, time_train, event_train, X_test, seed, n_estimators, mode):
    kwargs = dict(
        n_estimators=n_estimators,
        min_samples_leaf=1,
        min_samples_split=30,
        max_features="sqrt",
        bootstrap=True,
        random_state=seed,
        mode=mode,
        nsplit=0,
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
    return grid, cif, fit_wall


def run_cell(dataset: str, seed: int, n_estimators: int) -> dict:
    X, time_all, event_all = load_dataset(dataset)
    splits_df = pd.read_parquet(_SPLITS_DIR / f"{dataset}.parquet")
    row = splits_df[splits_df["seed"] == seed]
    train_idx = np.sort(row.loc[row["fold"] == "train", "sample_id"].to_numpy(np.int64))
    test_idx = np.sort(row.loc[row["fold"] == "test", "sample_id"].to_numpy(np.int64))
    ref_grid = build_reference_grid(time_all, event_all)

    X_tr, t_tr, e_tr = X[train_idx], time_all[train_idx], event_all[train_idx]
    X_te = X[test_idx]

    grid_def, cif_def, t_def = _fit_crforest(X_tr, t_tr, e_tr, X_te, seed, n_estimators, "default")
    grid_ref, cif_ref, t_ref = _fit_crforest(
        X_tr, t_tr, e_tr, X_te, seed, n_estimators, "reference"
    )
    grid_rf, cif_rf, t_rf = _fit_rfsrc(X_tr, t_tr, e_tr, X_te, seed, n_estimators)

    cif_def_g = eval_on_ref_grid(cif_def[:, :, 0], grid_def, ref_grid)
    cif_ref_g = eval_on_ref_grid(cif_ref[:, :, 0], grid_ref, ref_grid)
    cif_rf_g = eval_on_ref_grid(cif_rf[:, :, 0], grid_rf, ref_grid)

    return {
        "dataset": dataset,
        "seed": seed,
        "n": len(train_idx),
        "p": X.shape[1],
        "p95_default": float(np.percentile(np.abs(cif_def_g - cif_rf_g), 95)),
        "p95_reference": float(np.percentile(np.abs(cif_ref_g - cif_rf_g), 95)),
        "max_default": float(np.abs(cif_def_g - cif_rf_g).max()),
        "max_reference": float(np.abs(cif_ref_g - cif_rf_g).max()),
        "fit_default_s": t_def,
        "fit_reference_s": t_ref,
        "fit_rfsrc_s": t_rf,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--datasets", nargs="+", default=list(DATASETS))
    parser.add_argument("--seeds", type=int, default=3)
    parser.add_argument("--n-estimators", type=int, default=100)
    parser.add_argument("--out", type=Path, default=Path("/tmp/mode_vs_perf.json"))
    args = parser.parse_args()

    rows = []
    for ds in args.datasets:
        for s in range(1, args.seeds + 1):
            r = run_cell(ds, s, args.n_estimators)
            rows.append(r)
            print(
                f"[{ds} seed={s}] "
                f"p95(def)={r['p95_default']:.4f} p95(ref)={r['p95_reference']:.4f} | "
                f"fit(s): def={r['fit_default_s']:.1f} ref={r['fit_reference_s']:.1f} "
                f"rf={r['fit_rfsrc_s']:.1f} "
                f"| ratio(def/rf)={r['fit_default_s'] / r['fit_rfsrc_s']:.2f} "
                f"ratio(ref/rf)={r['fit_reference_s'] / r['fit_rfsrc_s']:.2f}",
                flush=True,
            )

    df = pd.DataFrame(rows)
    print("\n=== per-dataset medians ===")
    for ds in df["dataset"].unique():
        sub = df[df["dataset"] == ds]
        print(
            f"  {ds:>10s}: p95(def)={sub['p95_default'].median():.4f} "
            f"p95(ref)={sub['p95_reference'].median():.4f} | "
            f"fit median: def={sub['fit_default_s'].median():.2f} "
            f"ref={sub['fit_reference_s'].median():.2f} "
            f"rf={sub['fit_rfsrc_s'].median():.2f} | "
            f"ref/def={sub['fit_reference_s'].median() / sub['fit_default_s'].median():.2f}x "
            f"ref/rf={sub['fit_reference_s'].median() / sub['fit_rfsrc_s'].median():.2f}x"
        )

    args.out.write_text(json.dumps(rows, indent=2))
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
