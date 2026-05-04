"""Cell-by-cell comparison of ensemble OOB mortality between comprisk and rfSRC
on the same paired-bootstrap forest.

Sweeps over (dataset, ntree, seed). Reports per-sample Spearman/Pearson +
p99|Δ|/range to characterize fit-level mortality match. hd matches near-bit
at ntree=100 (Spearman 0.998); synthetic converges with ntree (0.84 → 0.97
as ntree goes 100 → 500), confirming the synthetic mortality gap is per-tree
noise that averages out, not a structural fit-layer divergence.
"""

from __future__ import annotations

import sys

import numpy as np
import pandas as pd
import rpy2.robjects as ro
from rpy2.robjects.conversion import localconverter
from rpy2.robjects.packages import importr
from scipy.stats import pearsonr, spearmanr

from comprisk import CompetingRiskForest
from comprisk._importance import _ensemble_oob_predictions
from validation.alignment import _rpy2_converter
from validation.datasets import load as load_dataset
from validation.splits import _SPLITS_DIR


def _print(msg: str) -> None:
    print(msg, flush=True)


def measure(dataset: str, seed: int, ntree: int = 100) -> list[dict]:
    """Return one row per cause with cellwise mortality comparison stats."""
    importr("randomForestSRC")
    converter = _rpy2_converter()

    X, time_all, event_all = load_dataset(dataset)
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
    ).fit(X[tr], time_all[tr], event_all[tr])

    bin_edges = forest.bin_edges_
    time_grid = np.asarray(forest.time_grid_, dtype=np.float64)
    cr_pred, count = _ensemble_oob_predictions(
        forest, forest._X_train_oob_, [1, 2], bin_edges, time_grid
    )
    cr_mask = count > 0
    cr_mort = np.full_like(cr_pred, np.nan)
    cr_mort[:, cr_mask] = cr_pred[:, cr_mask] / count[cr_mask]

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
            nsplit=10, ntime=0, importance="none", seed=-{int(seed)})
        """
    )
    with localconverter(converter):
        oob_dim = list(np.asarray(ro.r("dim(fit__$predicted.oob)"), dtype=np.int64))
        oob_flat = np.asarray(ro.r("as.vector(fit__$predicted.oob)"), dtype=np.float64)
        rf_oob_mort = oob_flat.reshape(oob_dim, order="F")  # (n_tr, n_causes)

    rf_mask = ~np.isnan(rf_oob_mort).any(axis=1)
    both_mask = cr_mask & rf_mask

    rows = []
    for ci, c in enumerate([1, 2]):
        cr_v = cr_mort[ci, both_mask]
        rf_v = rf_oob_mort[both_mask, ci]
        diff = cr_v - rf_v
        abs_diff = np.abs(diff)
        cr_range = cr_v.max() - cr_v.min() if cr_v.size else np.nan
        rf_range = rf_v.max() - rf_v.min() if rf_v.size else np.nan
        rho, _ = spearmanr(cr_v, rf_v)
        r, _ = pearsonr(cr_v, rf_v)
        rows.append(
            {
                "dataset": dataset,
                "seed": seed,
                "cause": c,
                "n_oob": int(both_mask.sum()),
                "spearman": float(rho),
                "pearson": float(r),
                "mean_abs_diff": float(abs_diff.mean()),
                "p99_abs_diff": float(np.quantile(abs_diff, 0.99)),
                "max_abs_diff": float(abs_diff.max()),
                "cr_range": float(cr_range),
                "rf_range": float(rf_range),
                "p99_diff_over_range": float(np.quantile(abs_diff, 0.99) / max(cr_range, rf_range))
                if max(cr_range, rf_range) > 0
                else np.nan,
            }
        )
    return rows


def main() -> int:
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--datasets", nargs="+", default=["synthetic"])
    parser.add_argument("--ntrees", nargs="+", type=int, default=[100, 300, 500])
    parser.add_argument("--seeds", nargs="+", type=int, default=[1, 2, 3])
    args = parser.parse_args()

    datasets = tuple(args.datasets)
    seeds = tuple(args.seeds)
    ntrees = tuple(args.ntrees)

    _print(f"# mortality_cellwise: ntrees={ntrees}, seeds={seeds}")
    _print(f"# datasets={datasets}\n")
    all_rows = []
    for ds in datasets:
        for nt in ntrees:
            for s in seeds:
                _print(f"=== {ds} ntree={nt} seed={s} ===")
                rows = measure(ds, s, ntree=nt)
                for r in rows:
                    r["ntree"] = nt
                    _print(
                        f"  c={r['cause']}: n={r['n_oob']:4d}  "
                        f"spearman={r['spearman']:+.4f}  "
                        f"pearson={r['pearson']:+.4f}  "
                        f"mean|Δ|={r['mean_abs_diff']:.4f}  "
                        f"p99|Δ|={r['p99_abs_diff']:.4f}  "
                        f"p99|Δ|/range={r['p99_diff_over_range']:.3%}"
                    )
                    all_rows.append(r)

    df = pd.DataFrame(all_rows)
    _print("\n## Summary (median across seeds, per dataset x ntree x cause)")
    summary = df.groupby(["dataset", "ntree", "cause"], as_index=False).agg(
        spearman=("spearman", "median"),
        pearson=("pearson", "median"),
        mean_abs_diff=("mean_abs_diff", "median"),
        p99_abs_diff=("p99_abs_diff", "median"),
        p99_over_range=("p99_diff_over_range", "median"),
    )
    _print(summary.to_string(index=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
