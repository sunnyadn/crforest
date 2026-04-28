"""Quick check: does rfSRC's ntime=0 time grid match crforest's unique_times_
on hd? If yes, ntime grid hypothesis is eliminated."""

import numpy as np
import pandas as pd
import rpy2.robjects as ro
from rpy2.robjects.conversion import localconverter
from rpy2.robjects.packages import importr

from crforest import CompetingRiskForest
from validation.alignment import _rpy2_converter
from validation.datasets import load as load_dataset
from validation.splits import _SPLITS_DIR


def main():
    importr("randomForestSRC")
    converter = _rpy2_converter()

    X, time_all, event_all = load_dataset("hd")
    splits_df = pd.read_parquet(_SPLITS_DIR / "hd.parquet")
    row = splits_df[splits_df["seed"] == 1]
    tr = np.sort(row.loc[row["fold"] == "train", "sample_id"].to_numpy(np.int64))
    p = X.shape[1]
    n_tr = len(tr)

    forest = CompetingRiskForest(
        n_estimators=100,
        min_samples_leaf=1,
        min_samples_split=30,
        max_features="sqrt",
        bootstrap=True,
        random_state=1,
        equivalence="rfsrc",
    ).fit(X[tr], time_all[tr], event_all[tr])
    cr_grid = np.asarray(forest.unique_times_, dtype=np.float64)

    feat_names = [f"x{j}" for j in range(p)]
    train_df = pd.DataFrame(X[tr], columns=feat_names)
    train_df["time"] = time_all[tr]
    train_df["event"] = event_all[tr].astype(np.int32)
    with localconverter(converter):
        ro.globalenv["train_df"] = train_df
        ro.globalenv["samp_matrix"] = ro.r.matrix(
            ro.FloatVector(forest.inbag_.T.reshape(-1).astype(np.float64)),
            nrow=n_tr,
            ncol=100,
        )
    ro.r(
        f"""
        fit__ <- rfsrc(Surv(time, event) ~ ., data=train_df,
            ntree=100, nodesize=15, mtry=ceiling(sqrt({p})),
            splitrule="logrankCR", bootstrap="by.user", samp=samp_matrix,
            nsplit=10, ntime=0, importance="none",
            use.uno=FALSE, seed=-1)
        """
    )
    with localconverter(converter):
        rf_grid = np.asarray(ro.r("fit__$time.interest"), dtype=np.float64)
        rf_oob_dim = list(np.asarray(ro.r("dim(fit__$predicted.oob)"), dtype=np.int64))

    print(
        f"crforest unique_times_  shape={cr_grid.shape}  range=[{cr_grid.min():.3f}, {cr_grid.max():.3f}]"
    )
    print(
        f"rfSRC time.interest      shape={rf_grid.shape}  range=[{rf_grid.min():.3f}, {rf_grid.max():.3f}]"
    )
    print(f"rfSRC predicted.oob shape={rf_oob_dim}")

    if cr_grid.shape == rf_grid.shape and np.allclose(cr_grid, rf_grid):
        print("\n>>> grids are IDENTICAL — ntime hypothesis FALSIFIED")
    else:
        print("\n>>> grids DIFFER")
        # Show first 10 elements of each
        print(f"  crforest first 10: {cr_grid[:10]}")
        print(f"  rfSRC    first 10: {rf_grid[:10]}")
        # Show set diffs
        cr_set = set(np.round(cr_grid, 6))
        rf_set = set(np.round(rf_grid, 6))
        only_cr = sorted(cr_set - rf_set)
        only_rf = sorted(rf_set - cr_set)
        print(f"  only in cr: {len(only_cr)} elements; first 5: {only_cr[:5]}")
        print(f"  only in rf: {len(only_rf)} elements; first 5: {only_rf[:5]}")


if __name__ == "__main__":
    main()
