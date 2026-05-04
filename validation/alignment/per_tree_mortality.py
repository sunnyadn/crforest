"""Per-tree OOB mortality cross-lib comparison on hd seed=1.

For each tree, extracts comprisk's per-tree mortality (via
``_predict_tree_mortality``) on its OOB samples and rfSRC's per-tree
mortality (via ``predict.rfsrc(fit, get.tree=t+1)$predicted.oob``), then
reports per-tree Spearman + p99|Δ|/range across the cells.

Result: median Spearman = 1.000 across all 100 trees (both causes) — trees
are functionally identical between libs at the per-tree-OOB level. Rules
out ``per-tree mortality differs from ensemble`` as a mechanism for cross-
lib VIMP divergence.
"""

from __future__ import annotations

import sys

import numpy as np
import pandas as pd
import rpy2.robjects as ro
from rpy2.robjects.conversion import localconverter
from rpy2.robjects.packages import importr
from scipy.stats import spearmanr

from comprisk import CompetingRiskForest
from comprisk._importance import _predict_tree_mortality
from validation.alignment import _rpy2_converter
from validation.datasets import load as load_dataset
from validation.splits import _SPLITS_DIR


def _print(msg: str) -> None:
    print(msg, flush=True)


def main() -> int:
    importr("randomForestSRC")
    converter = _rpy2_converter()
    dataset = "hd"
    seed = 1
    ntree = 100

    X, time_all, event_all = load_dataset(dataset)
    splits_df = pd.read_parquet(_SPLITS_DIR / f"{dataset}.parquet")
    row = splits_df[splits_df["seed"] == seed]
    tr = np.sort(row.loc[row["fold"] == "train", "sample_id"].to_numpy(np.int64))
    p = X.shape[1]
    n_tr = len(tr)
    feat_names = [f"x{j}" for j in range(p)]

    _print(f"# per_tree_mortality: dataset={dataset} seed={seed} ntree={ntree} n_tr={n_tr} p={p}")

    forest = CompetingRiskForest(
        n_estimators=ntree,
        min_samples_leaf=1,
        min_samples_split=30,
        max_features="sqrt",
        bootstrap=True,
        random_state=seed,
        equivalence="rfsrc",
    ).fit(X[tr], time_all[tr], event_all[tr])
    bin_edges = forest.bin_edges_
    time_grid = np.asarray(forest.time_grid_, dtype=np.float64)

    train_df = pd.DataFrame(X[tr], columns=feat_names)
    train_df["time"] = time_all[tr]
    train_df["event"] = event_all[tr].astype(np.int32)
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
            use.uno=FALSE, seed=-{int(seed)})
        """
    )
    _print("rfSRC fit complete; extracting per-tree mortality via predict.rfsrc(get.tree=t)")

    rows = []
    for t in range(ntree):
        oob_idx = np.asarray(forest.oob_indices_[t], dtype=np.int64)
        if len(oob_idx) == 0:
            continue

        # comprisk per-tree mortality on OOB
        cr_mort = np.zeros((2, len(oob_idx)), dtype=np.float64)
        for ci, c in enumerate([1, 2]):
            cr_mort[ci] = _predict_tree_mortality(
                forest.trees_[t],
                X[tr][oob_idx],
                cause=c,
                mode=forest.mode,
                bin_edges=bin_edges,
                time_grid=time_grid,
            )

        # rfSRC per-tree mortality: predict(fit, get.tree=c(t+1)) with no newdata
        # (re-evaluates the original training data through tree t alone).
        ro.r(f"pred__ <- predict(fit__, get.tree=c({t + 1}))")
        with localconverter(converter):
            rf_oob_dim = list(np.asarray(ro.r("dim(pred__$predicted.oob)"), dtype=np.int64))
            rf_oob_flat = np.asarray(ro.r("as.vector(pred__$predicted.oob)"), dtype=np.float64)
        rf_full = rf_oob_flat.reshape(rf_oob_dim, order="F")  # (n_tr, n_causes)
        rf_mort_oob = rf_full[oob_idx]  # (n_oob, n_causes)
        valid = ~np.isnan(rf_mort_oob).any(axis=1)
        if valid.sum() < 5:
            continue
        for ci, c in enumerate([1, 2]):
            cr_v = cr_mort[ci, valid]
            rf_v = rf_mort_oob[valid, ci]
            if len(np.unique(rf_v)) < 2 or len(np.unique(cr_v)) < 2:
                continue
            rho, _ = spearmanr(cr_v, rf_v)
            diff = np.abs(cr_v - rf_v)
            cr_range = cr_v.max() - cr_v.min()
            rf_range = rf_v.max() - rf_v.min()
            rng = max(cr_range, rf_range)
            rows.append(
                {
                    "tree": t,
                    "cause": c,
                    "n_oob": int(valid.sum()),
                    "spearman": float(rho),
                    "p99_over_range": float(np.quantile(diff, 0.99) / rng)
                    if rng > 0
                    else float("nan"),
                    "max_over_range": float(diff.max() / rng) if rng > 0 else float("nan"),
                }
            )
        if (t + 1) % 10 == 0:
            recent = pd.DataFrame(rows[-20:])
            _print(
                f"  t={t + 1}/{ntree} median spearman so far: "
                f"c1={recent[recent.cause == 1]['spearman'].median():+.3f}  "
                f"c2={recent[recent.cause == 2]['spearman'].median():+.3f}"
            )

    df = pd.DataFrame(rows)
    _print("\n## Per-tree OOB mortality match (across 100 trees, hd seed=1)")
    for c in (1, 2):
        sub = df[df.cause == c]
        _print(
            f"  cause={c}: trees={len(sub)}  "
            f"spearman median={sub['spearman'].median():+.3f}  "
            f"q25={sub['spearman'].quantile(0.25):+.3f}  "
            f"q75={sub['spearman'].quantile(0.75):+.3f}  "
            f"min={sub['spearman'].min():+.3f}  "
            f"max={sub['spearman'].max():+.3f}"
        )
        _print(
            f"           p99|Δ|/range median={sub['p99_over_range'].median():.3%}  "
            f"max={sub['max_over_range'].max():.3%}"
        )

    df.to_parquet("/tmp/per_tree_mortality_hd_seed1.parquet")
    _print("\nResults written to /tmp/per_tree_mortality_hd_seed1.parquet")
    return 0


if __name__ == "__main__":
    sys.exit(main())
