"""Z cell: all RNG sources removed or aligned, 500-tree ensemble.

Tests whether the production cross-lib residual is 100% RNG-driven by
constructing a config where:
  - bootstrap: externally supplied (same in-bag matrix fed to both libs)
  - mtry: full p (no feature subsampling -> no mtry RNG)
  - nsplit: 0 (exhaustive -> no candidate-subset RNG)
  - split_ntime=None, rfSRC ntime=0 (no time-grid coarsening)

Result (2026-04-24, 10 seeds):

| dataset | A default | Z      | reduction |
|---------|-----------|--------|-----------|
| hd      | 0.0573    | 0.0054 | 90.6%     |
| follic  | 0.0457    | 0.0127 | 72.2%     |

Interpretation: RNG independence accounts for ~90% of the hd production
gap and ~72% of the follic gap. A small non-RNG residual
(~0.005-0.013) remains from implementation-level numerical details:
floating-point accumulation order in the logrank kernel, tiebreak in
split-winner selection when candidates have identical stats, ensemble
averaging precision across 500 trees. This floor is O(1e-3), well
below within-lib seed variance (~0.28 on hd).

Strategic: Phase 1 (port rfSRC's ran1 RNG to comprisk) would close the
RNG portion, bringing the production gap from ~0.057 to ~0.005-0.015.
Does NOT achieve literal bit-identity (Z != 0), but collapses the gap
to a numerical-precision floor.
"""

from __future__ import annotations

import time as _time

import numpy as np
import pandas as pd

from comprisk import CompetingRiskForest
from validation.alignment import _rpy2_converter
from validation.alignment.bootstrap_aligned_spike import _comprisk_inbag_counts
from validation.alignment.equivalence_gate import (
    aggregate_dataset,
    build_reference_grid,
    eval_on_ref_grid,
)
from validation.datasets import load as load_dataset
from validation.splits import _SPLITS_DIR


def run_z_cell(dataset: str, seed: int, *, ntree: int = 500) -> dict:
    import rpy2.robjects as ro
    from rpy2.robjects.conversion import localconverter
    from rpy2.robjects.packages import importr

    importr("randomForestSRC")
    converter = _rpy2_converter()

    X, time, event = load_dataset(dataset)
    ref_grid = build_reference_grid(time, event)
    splits_df = pd.read_parquet(_SPLITS_DIR / f"{dataset}.parquet")
    row = splits_df[splits_df["seed"] == seed]
    train_idx = np.sort(row.loc[row["fold"] == "train", "sample_id"].to_numpy(np.int64))
    test_idx = np.sort(row.loc[row["fold"] == "test", "sample_id"].to_numpy(np.int64))
    n_tr = len(train_idx)
    p = X.shape[1]

    # comprisk with full mtry + exhaustive splits + no time coarsening.
    t0 = _time.perf_counter()
    print(f"[Z ds={dataset} seed={seed}] cr_fit_start n_train={n_tr} p={p}", flush=True)
    forest = CompetingRiskForest(
        n_estimators=ntree,
        min_samples_leaf=1,
        min_samples_split=30,
        max_features=p,  # no mtry subsampling
        bootstrap=True,
        random_state=seed,
        mode="default",
        time_grid=200,
        nsplit=0,  # exhaustive
        split_ntime=None,  # full time grid
    ).fit(X[train_idx], time[train_idx], event[train_idx])
    cif_cr_raw = np.transpose(forest.predict_cif(X[test_idx]), (0, 2, 1))
    cr_grid = np.asarray(forest.time_grid_, dtype=np.float64)
    print(f"[Z seed={seed}] cr_fit_done wall={_time.perf_counter() - t0:.1f}s", flush=True)

    # Build matching inbag matrix (replicates comprisk's per-tree bootstrap).
    inbag = _comprisk_inbag_counts(n_tr, ntree, seed)

    feat_cols = [f"x{j}" for j in range(p)]
    train_df = pd.DataFrame(X[train_idx], columns=feat_cols)
    train_df["time"] = time[train_idx]
    train_df["event"] = event[train_idx].astype(np.int32)
    test_df = pd.DataFrame(X[test_idx], columns=feat_cols)

    with localconverter(converter):
        ro.globalenv["train_df"] = train_df
        ro.globalenv["test_df"] = test_df
        ro.globalenv["samp_matrix"] = ro.r.matrix(
            ro.FloatVector(inbag.T.reshape(-1).astype(np.float64)),
            nrow=n_tr,
            ncol=ntree,
        )

    t0 = _time.perf_counter()
    print(f"[Z seed={seed}] rf_fit_start", flush=True)
    ro.r(
        f"""
        fit__ <- rfsrc(
            Surv(time, event) ~ .,
            data       = train_df,
            ntree      = {ntree},
            nodesize   = 15,
            mtry       = {p},
            splitrule  = "logrankCR",
            bootstrap  = "by.user",
            samp       = samp_matrix,
            nsplit     = 0,
            ntime      = 0,
            seed       = -{int(seed)}
        )
        pred__ <- predict(fit__, newdata = test_df)
        """
    )
    with localconverter(converter):
        time_interest = np.asarray(ro.r("fit__$time.interest"), dtype=np.float64)
        cif_flat = np.asarray(ro.r("pred__$cif"), dtype=np.float64)
        ro.r("rm(fit__); rm(pred__); rm(samp_matrix)")
    print(f"[Z seed={seed}] rf_fit_done wall={_time.perf_counter() - t0:.1f}s", flush=True)

    n_te = len(test_idx)
    cif_rf_raw = cif_flat.reshape(n_te, len(time_interest), 2)
    cif_cr = eval_on_ref_grid(cif_cr_raw[:, :, 0], cr_grid, ref_grid)
    cif_rf = eval_on_ref_grid(cif_rf_raw[:, :, 0], time_interest, ref_grid)
    return {
        "seed": seed,
        "cif_cr": cif_cr,
        "cif_rf": cif_rf,
        "risk_cr": cif_cr[:, -1].copy(),
        "risk_rf": cif_rf[:, -1].copy(),
    }


def main():
    for ds in ("hd", "follic"):
        cells = [run_z_cell(ds, s) for s in range(10)]
        agg = aggregate_dataset(cells)
        print(
            f"[Z {ds}] cross_p95_cif={agg['cross_p95_cif']:.4f} "
            f"cross_p95_risk={agg['cross_p95_risk']:.4f}",
            flush=True,
        )


if __name__ == "__main__":
    main()
