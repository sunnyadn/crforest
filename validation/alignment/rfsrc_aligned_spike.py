"""Phase 1 validation: production config with rng_mode='rfsrc_aligned'.

Tests whether aligning comprisk's mtry + nsplit RNG draws with rfSRC's
ran1 stream B closes the production cross-lib gap.

Previous evidence:
- Default production cross_p95_cif on hd = 0.0573, follic = 0.0457.
- Z cell (all RNG removed/aligned, mtry=p, nsplit=0) = 0.0054 / 0.0127.
- Bootstrap-alignment alone = ~1.5% reduction (rules out stream A).

Phase 1 hypothesis: comprisk(`rng_mode='rfsrc_aligned'`) at production
defaults should close cross-lib gap to ~0.005 range, matching the Z
floor.

Result (2026-04-24, 10 seeds, rng_mode='rfsrc_aligned'):

| dataset | A default | rfsrc_aligned | delta    |
|---------|-----------|---------------|----------|
| hd      | 0.0573    | 0.0565        | ~1.4%    |
| follic  | 0.0457    | 0.0465        | ~noise   |

**Hypothesis not yet confirmed or refuted.** A ntree=1 diagnostic on
hd seed=0 shows comprisk(rfsrc_aligned) picks a different root split
(feature=2, threshold=1.5) from rfSRC (feature=0, threshold=54.74) --
the RNG algorithm + per-tree seeding are correctly aligned (ran1
port verified against C to float32 precision), but comprisk's per-
node **call order** differs from rfSRC's:

  - comprisk: draw all mtry features upfront, then for each feature
    draw nsplit candidates.
  - rfSRC: interleave -- draw ONE feature, eval split (with nsplit
    draws), draw NEXT feature, eval, ...

Same total stream-B draws per node but different interleaving ->
different pool state at each mtry draw -> different features picked.

Closing this gap requires refactoring ``_hist_tree._build_node_hist``
(Phase 1b) to match rfSRC's interleaved flow in ``splitSurv.c:97`` +
``selectRandomCovariatesGeneric``. Not yet attempted.
"""

from __future__ import annotations

import time as _time

import numpy as np
import pandas as pd

from comprisk import CompetingRiskForest
from validation.alignment.equivalence_gate import (
    aggregate_dataset,
    build_reference_grid,
    eval_on_ref_grid,
)
from validation.datasets import load as load_dataset
from validation.splits import _SPLITS_DIR


def _fit_comprisk_aligned(X_tr, t_tr, e_tr, X_te, *, seed: int) -> dict:
    forest = CompetingRiskForest(
        n_estimators=500,
        min_samples_leaf=1,
        min_samples_split=30,
        max_features="sqrt",
        bootstrap=True,
        random_state=seed,
        mode="default",
        time_grid=200,
        rng_mode="rfsrc_aligned",  # <-- Phase 1 under test
    ).fit(X_tr, t_tr, e_tr)
    cif = np.transpose(forest.predict_cif(X_te), (0, 2, 1))
    return {"time_grid": np.asarray(forest.time_grid_, dtype=np.float64), "cif": cif}


def _fit_rfsrc(X_tr, t_tr, e_tr, X_te, *, seed: int, p: int) -> dict:
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
    cif_rf = cif_flat.reshape(n_test, len(time_interest), 2)
    return {"time_grid": time_interest, "cif": cif_rf}


def run_cell(dataset: str, seed: int) -> dict:
    X, time, event = load_dataset(dataset)
    ref_grid = build_reference_grid(time, event)
    splits_df = pd.read_parquet(_SPLITS_DIR / f"{dataset}.parquet")
    row = splits_df[splits_df["seed"] == seed]
    train_idx = np.sort(row.loc[row["fold"] == "train", "sample_id"].to_numpy(np.int64))
    test_idx = np.sort(row.loc[row["fold"] == "test", "sample_id"].to_numpy(np.int64))
    p = X.shape[1]

    t0 = _time.perf_counter()
    print(
        f"[rfsrc_aligned ds={dataset} seed={seed}] cr_fit_start n_train={len(train_idx)} p={p}",
        flush=True,
    )
    cr = _fit_comprisk_aligned(
        X[train_idx], time[train_idx], event[train_idx], X[test_idx], seed=seed
    )
    print(
        f"[rfsrc_aligned seed={seed}] cr_fit_done wall={_time.perf_counter() - t0:.1f}s",
        flush=True,
    )
    t0 = _time.perf_counter()
    print(f"[rfsrc_aligned seed={seed}] rf_fit_start", flush=True)
    rf = _fit_rfsrc(X[train_idx], time[train_idx], event[train_idx], X[test_idx], seed=seed, p=p)
    print(
        f"[rfsrc_aligned seed={seed}] rf_fit_done wall={_time.perf_counter() - t0:.1f}s",
        flush=True,
    )

    cif_cr = eval_on_ref_grid(cr["cif"][:, :, 0], cr["time_grid"], ref_grid)
    cif_rf = eval_on_ref_grid(rf["cif"][:, :, 0], rf["time_grid"], ref_grid)
    return {
        "seed": seed,
        "cif_cr": cif_cr,
        "cif_rf": cif_rf,
        "risk_cr": cif_cr[:, -1].copy(),
        "risk_rf": cif_rf[:, -1].copy(),
    }


def main() -> None:
    # rfSRC's R-wrapper `get.seed` replaces seed <= 0 with a runif-random
    # value, so we use seeds 1..10 (not 0..9) to keep both libs on
    # matched deterministic master seeds.
    for ds in ("hd", "follic"):
        cells = [run_cell(ds, s) for s in range(1, 11)]
        agg = aggregate_dataset(cells)
        print(
            f"[rfsrc_aligned {ds}] cross_p95_cif={agg['cross_p95_cif']:.4f} "
            f"cross_p95_risk={agg['cross_p95_risk']:.4f}",
            flush=True,
        )


if __name__ == "__main__":
    main()
