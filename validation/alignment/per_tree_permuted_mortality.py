"""Apples-to-apples per-tree PERMUTED mortality cross-lib comparison.

Hypothesis: when we apply the SAME permutation of feature f to both libs
on the SAME tree, do we get the same per-tree mortality? If yes, trees
are functionally identical (no boundary/tie quirks); divergence comes
purely from the libs choosing DIFFERENT random permutations. If no,
trees differ at decision boundaries (eg. tie-handling in feature
comparisons), which would show up under permuted X but not unpermuted.

For each tree t and feature f (test: f=0):
  1. Get OOB indices for tree t.
  2. Generate ONE permutation π via numpy (canonical, lib-agnostic).
  3. Build X_perm: X with column f shuffled within OOB rows by π.
  4. crforest: _predict_tree_mortality(forest.trees_[t], X_perm[oob], ...).
  5. rfSRC: predict.rfsrc(fit, newdata=X_perm, get.tree=c(t+1))$predicted.oob[oob].
  6. Compare cell-by-cell.
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


def main() -> int:
    importr("randomForestSRC")
    converter = _rpy2_converter()
    dataset = "hd"
    seed = 1
    ntree = 100
    feat_to_perm = 0  # any feature works for the test

    X, T, E = load_dataset(dataset)
    splits_df = pd.read_parquet(_SPLITS_DIR / f"{dataset}.parquet")
    row = splits_df[splits_df["seed"] == seed]
    tr = np.sort(row.loc[row["fold"] == "train", "sample_id"].to_numpy(np.int64))
    p = X.shape[1]
    n_tr = len(tr)
    feat_names = [f"x{j}" for j in range(p)]
    Xtr = X[tr].copy()
    Ttr = T[tr]
    Etr = E[tr]

    _print(
        f"# per_tree_permuted_mortality: {dataset} seed={seed} ntree={ntree} "
        f"feat_to_perm={feat_to_perm}"
    )

    forest = CompetingRiskForest(
        n_estimators=ntree,
        min_samples_leaf=1,
        min_samples_split=30,
        max_features="sqrt",
        bootstrap=True,
        random_state=seed,
        equivalence="rfsrc",
    ).fit(Xtr, Ttr, Etr)
    bin_edges = forest.bin_edges_
    time_grid = np.asarray(forest.time_grid_, dtype=np.float64)

    train_df = pd.DataFrame(Xtr, columns=feat_names)
    train_df["time"] = Ttr
    train_df["event"] = Etr.astype(np.int32)
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

    rng = np.random.default_rng(98765)
    rows = []
    for t in range(ntree):
        oob_idx = np.asarray(forest.oob_indices_[t], dtype=np.int64)
        if len(oob_idx) < 5:
            continue

        # canonical permutation (numpy) — identical input fed to both libs
        perm = rng.permutation(len(oob_idx))
        X_perm = Xtr.copy()
        X_perm[oob_idx, feat_to_perm] = Xtr[oob_idx[perm], feat_to_perm]

        # crforest per-tree on permuted X (full input, but only oob_idx is changed)
        cr_mort = np.zeros((2, len(oob_idx)), dtype=np.float64)
        for ci, c in enumerate([1, 2]):
            cr_mort[ci] = _predict_tree_mortality(
                forest.trees_[t],
                X_perm[oob_idx],
                cause=c,
                mode=forest.mode,
                bin_edges=bin_edges,
                time_grid=time_grid,
            )

        # rfSRC per-tree on permuted X
        df_perm = train_df.copy()
        df_perm.iloc[:, feat_to_perm] = X_perm[:, feat_to_perm]
        with localconverter(converter):
            ro.globalenv["nd__"] = df_perm
        ro.r(f"pred__ <- predict(fit__, newdata=nd__, get.tree=c({t + 1}))")
        with localconverter(converter):
            rf_dim = list(np.asarray(ro.r("dim(pred__$predicted)"), dtype=np.int64))
            rf_flat = np.asarray(ro.r("as.vector(pred__$predicted)"), dtype=np.float64)
        rf_mort_full = rf_flat.reshape(rf_dim, order="F")  # (n_tr, n_causes)
        rf_mort_oob = rf_mort_full[oob_idx]
        valid = ~np.isnan(rf_mort_oob).any(axis=1)
        if valid.sum() < 5:
            continue

        for ci, c in enumerate([1, 2]):
            cr_v = cr_mort[ci, valid]
            rf_v = rf_mort_oob[valid, ci]
            if len(np.unique(cr_v)) < 2 or len(np.unique(rf_v)) < 2:
                continue
            rho, _ = spearmanr(cr_v, rf_v)
            diff = np.abs(cr_v - rf_v)
            cr_range = cr_v.max() - cr_v.min()
            rf_range = rf_v.max() - rf_v.min()
            rng_max = max(cr_range, rf_range)
            rows.append(
                {
                    "tree": t,
                    "cause": c,
                    "n_oob": int(valid.sum()),
                    "spearman": float(rho),
                    "p99_over_range": float(np.quantile(diff, 0.99) / rng_max)
                    if rng_max > 0
                    else float("nan"),
                }
            )
        if (t + 1) % 10 == 0:
            recent = pd.DataFrame(rows[-20:])
            _print(
                f"  t={t + 1}/{ntree} median spearman: "
                f"c1={recent[recent.cause == 1]['spearman'].median():+.3f}  "
                f"c2={recent[recent.cause == 2]['spearman'].median():+.3f}"
            )

    df = pd.DataFrame(rows)
    _print("\n## Per-tree PERMUTED mortality match (same canonical permutation, hd seed=1)")
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
            f"max={sub['p99_over_range'].max():.3%}"
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
