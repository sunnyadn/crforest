"""Phase 1b + bootstrap-aligned combined test.

Tests the full-RNG-alignment hypothesis at production config by
simultaneously:
  - crforest ``rng_mode="rfsrc_aligned"`` (aligns stream B for mtry/nsplit)
  - rfSRC ``bootstrap="by.user"`` with crforest-generated inbag matrix
    (aligns stream A for bootstrap)
  - split_ntime=None + rfSRC ntime=0 (removes time-grid coarsening bias)

Everything else at production defaults (mtry=sqrt, nsplit=10, ntree=500).

If this closes the cross-lib gap to the Z-cell floor (~0.005 on hd,
~0.012 on follic), we have empirical confirmation that the ~90% of
the production residual attributed to RNG in the Z cell analysis is
specifically "bootstrap + mtry/nsplit RNG stream independence", and
that crforest's rng_mode="rfsrc_aligned" successfully closes it.

Seeds 1..10 to avoid rfSRC's R-wrapper `get.seed` randomizing seed=0.

**Result (2026-04-24, seeds 1..10, Phase 1c)**:

| dataset | A default | Phase 1c (this spike) | Z floor (reference) |
|---------|-----------|-----------------------|---------------------|
| hd      | 0.0573    | **0.0047**            | 0.0054              |
| follic  | 0.0457    | **0.0117**            | 0.0127              |

Full RNG alignment (aligned bootstrap + rng_mode=rfsrc_aligned +
per-node permissible tracking + split_ntime=None + rfSRC ntime=0)
collapses the production cross-lib cross_p95_cif to the Z-cell
numerical floor (~10x reduction on hd). **RNG hypothesis empirically
validated**: ~90% of the production residual IS stream-A + stream-B
independence + per-node permissible-mask inheritance; the residual
~0.005 floor is non-RNG (float accumulation order + tiebreak). This
result localizes and confirms the mechanism identified in Phase 0
(project_rfsrc_rng_phase0) and Phase 1a/1b (ran1 port +
interleaved call order).
"""

from __future__ import annotations

import time as _time

import numpy as np
import pandas as pd

from crforest import CompetingRiskForest
from validation.alignment import _rpy2_converter
from validation.alignment.bootstrap_aligned_spike import _crforest_inbag_counts
from validation.alignment.equivalence_gate import (
    aggregate_dataset,
    build_reference_grid,
    eval_on_ref_grid,
)
from validation.datasets import load as load_dataset
from validation.splits import _SPLITS_DIR


def run_cell(
    dataset: str,
    seed: int,
    *,
    ntree: int = 500,
    min_samples_split: int = 30,
    min_samples_leaf: int = 1,
    rf_nodesize: int = 15,
) -> dict:
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

    # crforest: rng_mode=rfsrc_aligned, production defaults
    t0 = _time.perf_counter()
    print(f"[full_aligned ds={dataset} seed={seed}] cr_fit_start", flush=True)
    forest = CompetingRiskForest(
        n_estimators=ntree,
        min_samples_leaf=min_samples_leaf,
        min_samples_split=min_samples_split,
        max_features="sqrt",
        bootstrap=True,
        random_state=seed,
        mode="default",
        time_grid=200,
        split_ntime=None,
        rng_mode="rfsrc_aligned",
    ).fit(X[train_idx], time[train_idx], event[train_idx])
    cif_cr_raw = np.transpose(forest.predict_cif(X[test_idx]), (0, 2, 1))
    cr_grid = np.asarray(forest.time_grid_, dtype=np.float64)
    print(
        f"[full_aligned seed={seed}] cr_fit_done wall={_time.perf_counter() - t0:.1f}s", flush=True
    )

    # Build matching inbag: replicate crforest's per-tree bootstrap procedure.
    inbag = _crforest_inbag_counts(n_tr, ntree, seed)

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
    print(f"[full_aligned seed={seed}] rf_fit_start", flush=True)
    ro.r(
        f"""
        fit__ <- rfsrc(
            Surv(time, event) ~ .,
            data       = train_df,
            ntree      = {ntree},
            nodesize   = {rf_nodesize},
            mtry       = ceiling(sqrt({p})),
            splitrule  = "logrankCR",
            bootstrap  = "by.user",
            samp       = samp_matrix,
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
        ro.r("rm(fit__); rm(pred__); rm(samp_matrix)")
    print(
        f"[full_aligned seed={seed}] rf_fit_done wall={_time.perf_counter() - t0:.1f}s", flush=True
    )

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


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--datasets", nargs="*", default=["hd", "follic", "pbc", "synthetic"])
    parser.add_argument("--seeds", type=int, default=10)
    parser.add_argument(
        "--match-stopping",
        action="store_true",
        help="Align crforest min_samples_split to rfSRC nodesize (both 15, leaf=1) "
        "to eliminate the stopping-rule semantic gap identified in "
        "synthetic ntree=1 diagnostics.",
    )
    args = parser.parse_args()

    if args.match_stopping:
        kw = dict(min_samples_split=15, min_samples_leaf=1, rf_nodesize=15)
        label = "full_aligned+match_stopping"
    else:
        kw = dict(min_samples_split=30, min_samples_leaf=1, rf_nodesize=15)
        label = "full_aligned"

    for ds in args.datasets:
        cells = [run_cell(ds, s, **kw) for s in range(1, args.seeds + 1)]
        agg = aggregate_dataset(cells)
        q = agg["quantiles"]["cross_cif"]
        print(
            f"[{label} {ds}] cross_p95_cif={agg['cross_p95_cif']:.4f} "
            f"cross_p95_risk={agg['cross_p95_risk']:.4f} "
            f"q50={q[0.50]:.4f} q75={q[0.75]:.4f} q90={q[0.90]:.4f} "
            f"q95={q[0.95]:.4f} q99={q[0.99]:.4f}",
            flush=True,
        )


if __name__ == "__main__":
    main()
