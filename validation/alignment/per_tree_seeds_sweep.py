"""Verify per-tree mortality cross-lib match holds across multiple seeds,
not just seed=1. Closes the gap in the elimination argument that "trees
are functionally identical between libs". If Spearman = 1.0 at all
sampled seeds, the elimination is robust; if not, there's seed-dependent
fit-level divergence that contributes to the cross-lib VIMP gap.

For each (dataset, seed): run per-tree unpermuted mortality cross-lib and
report median Spearman across trees.
"""

from __future__ import annotations

import sys

import numpy as np
import pandas as pd
import rpy2.robjects as ro
from rpy2.robjects.conversion import localconverter
from rpy2.robjects.packages import importr
from scipy.stats import spearmanr

from crforest import CompetingRiskForest
from crforest._importance import _predict_tree_mortality
from validation.alignment import _rpy2_converter
from validation.datasets import load as load_dataset
from validation.splits import _SPLITS_DIR


def _print(msg: str) -> None:
    print(msg, flush=True)


def measure(dataset: str, seed: int, ntree: int, converter) -> dict:
    X, T, E = load_dataset(dataset)
    splits_df = pd.read_parquet(_SPLITS_DIR / f"{dataset}.parquet")
    row = splits_df[splits_df["seed"] == seed]
    tr = np.sort(row.loc[row["fold"] == "train", "sample_id"].to_numpy(np.int64))
    p = X.shape[1]
    n_tr = len(tr)
    feat_names = [f"x{j}" for j in range(p)]

    forest = CompetingRiskForest(
        n_estimators=ntree,
        min_samples_leaf=1,
        min_samples_split=30,
        max_features="sqrt",
        bootstrap=True,
        random_state=seed,
        equivalence="rfsrc",
    ).fit(X[tr], T[tr], E[tr])
    bin_edges = forest.bin_edges_
    time_grid = np.asarray(forest.time_grid_, dtype=np.float64)

    train_df = pd.DataFrame(X[tr], columns=feat_names)
    train_df["time"] = T[tr]
    train_df["event"] = E[tr].astype(np.int32)
    with localconverter(converter):
        ro.globalenv["train_df"] = train_df
        ro.globalenv["samp_matrix"] = ro.r.matrix(
            ro.FloatVector(forest.inbag_.T.reshape(-1).astype(np.float64)),
            nrow=n_tr,
            ncol=ntree,
        )
    ro.r(
        f"""
        fit__ <- rfsrc(Surv(time, event) ~ ., data=train_df,
            ntree={ntree}, nodesize=15, mtry=ceiling(sqrt({p})),
            splitrule="logrankCR", bootstrap="by.user", samp=samp_matrix,
            nsplit=10, ntime=0, importance="none",
            use.uno=FALSE, forest=TRUE, seed=-{int(seed)})
        """
    )

    rhos_c1, rhos_c2 = [], []
    for t in range(ntree):
        oob = np.asarray(forest.oob_indices_[t], dtype=np.int64)
        if len(oob) < 5:
            continue
        cr_mort = np.zeros((2, len(oob)), dtype=np.float64)
        for ci, c in enumerate([1, 2]):
            cr_mort[ci] = _predict_tree_mortality(
                forest.trees_[t],
                X[tr][oob],
                cause=c,
                mode=forest.mode,
                bin_edges=bin_edges,
                time_grid=time_grid,
            )
        ro.r(f"pred__ <- predict(fit__, get.tree=c({t + 1}))")
        with localconverter(converter):
            rf_dim = list(np.asarray(ro.r("dim(pred__$predicted.oob)"), dtype=np.int64))
            rf_flat = np.asarray(ro.r("as.vector(pred__$predicted.oob)"), dtype=np.float64)
        rf_full = rf_flat.reshape(rf_dim, order="F")
        rf_oob = rf_full[oob]
        valid = ~np.isnan(rf_oob).any(axis=1)
        if valid.sum() < 5:
            continue
        for ci, store in enumerate([rhos_c1, rhos_c2]):
            cr_v = cr_mort[ci, valid]
            rf_v = rf_oob[valid, ci]
            if len(np.unique(cr_v)) < 2 or len(np.unique(rf_v)) < 2:
                continue
            r, _ = spearmanr(cr_v, rf_v)
            store.append(float(r))
    return {
        "dataset": dataset,
        "seed": seed,
        "trees_c1": len(rhos_c1),
        "trees_c2": len(rhos_c2),
        "median_c1": float(np.median(rhos_c1)) if rhos_c1 else float("nan"),
        "median_c2": float(np.median(rhos_c2)) if rhos_c2 else float("nan"),
        "min_c1": float(np.min(rhos_c1)) if rhos_c1 else float("nan"),
        "min_c2": float(np.min(rhos_c2)) if rhos_c2 else float("nan"),
        "frac_c1_eq_one": float(np.mean(np.array(rhos_c1) > 0.999)) if rhos_c1 else float("nan"),
        "frac_c2_eq_one": float(np.mean(np.array(rhos_c2) > 0.999)) if rhos_c2 else float("nan"),
    }


def main() -> int:
    importr("randomForestSRC")
    converter = _rpy2_converter()
    seeds = (1, 2, 3, 5, 7, 10)
    rows = []
    for s in seeds:
        _print(f"=== hd seed={s} ===")
        stats = measure("hd", s, 100, converter)
        _print(
            f"  c1: median={stats['median_c1']:+.4f} min={stats['min_c1']:+.3f} "
            f"frac>0.999={stats['frac_c1_eq_one']:.2%}"
        )
        _print(
            f"  c2: median={stats['median_c2']:+.4f} min={stats['min_c2']:+.3f} "
            f"frac>0.999={stats['frac_c2_eq_one']:.2%}"
        )
        rows.append(stats)
    df = pd.DataFrame(rows)
    _print("\n## Summary across seeds")
    _print(df.to_string(index=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
