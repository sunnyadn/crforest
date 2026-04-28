"""VIMP alignment: crforest.importance() vs rfSRC vimp() at paired seeds.

Both libraries provide permutation-based variable importance, but the
algorithms differ in detail:

- crforest: ``sklearn.inspection.permutation_importance`` scored by
  ``concordance_index_cr`` on ``predict_risk``. Per-cause vectors,
  full-forest prediction with the permuted column, ``n_repeats``
  averaging.
- rfSRC: per-tree OOB Breiman permutation; ``vimp(fit, importance="permute")``
  returns ``vimp$importance`` (per-cause matrix). Internally averages
  per-tree error change.

Both report "performance drop after permutation" (higher = more important).
This script reports per-cause Spearman rho (ranking agreement) and Pearson r
(absolute agreement) on the intersection of feature names, plus the
4-dataset summary.

Run:
    uv run --extra maintainer python -m validation.alignment.vimp_alignment \\
        --seeds 3 --n-estimators 100
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr

from crforest import CompetingRiskForest
from validation.alignment import _rpy2_converter
from validation.datasets import load as load_dataset
from validation.splits import _SPLITS_DIR

DATASETS = ("pbc", "hd", "follic", "synthetic")


def run_cell(dataset: str, seed: int, n_estimators: int, n_repeats: int, mode: str) -> dict:
    import rpy2.robjects as ro
    from rpy2.robjects.conversion import localconverter
    from rpy2.robjects.packages import importr

    importr("randomForestSRC")
    converter = _rpy2_converter()

    X, time_all, event_all = load_dataset(dataset)
    splits_df = pd.read_parquet(_SPLITS_DIR / f"{dataset}.parquet")
    row = splits_df[splits_df["seed"] == seed]
    tr = np.sort(row.loc[row["fold"] == "train", "sample_id"].to_numpy(np.int64))
    p = X.shape[1]
    n_tr = len(tr)
    feat_names = [f"x{j}" for j in range(p)]

    # crforest fit + VIMP
    forest = CompetingRiskForest(
        n_estimators=n_estimators,
        min_samples_leaf=1,
        min_samples_split=30,
        max_features="sqrt",
        bootstrap=True,
        random_state=seed,
        equivalence="rfsrc",
    ).fit(X[tr], time_all[tr], event_all[tr])
    if mode == "oob":
        cr_vimp = forest.compute_importance(random_state=seed)
    else:
        # held-out path; uses training set as eval (in-bag leak — see commit d78582b)
        y_eval = np.zeros(n_tr, dtype=[("time", np.float64), ("event", np.int64)])
        y_eval["time"] = time_all[tr]
        y_eval["event"] = event_all[tr]
        cr_vimp = forest.compute_importance(
            X[tr],
            y_eval,
            n_repeats=n_repeats,
            random_state=seed,
        )

    # rfSRC fit (paired bootstrap) + vimp
    train_df = pd.DataFrame(X[tr], columns=feat_names)
    train_df["time"] = time_all[tr]
    train_df["event"] = event_all[tr].astype(np.int32)
    with localconverter(converter):
        ro.globalenv["train_df"] = train_df
        ro.globalenv["samp_matrix"] = ro.r.matrix(
            ro.FloatVector(forest.inbag_.T.reshape(-1).astype(np.float64)),
            nrow=n_tr,
            ncol=n_estimators,
        )
    # use.uno=TRUE (default since 2026-04-25): rfSRC reports Uno IPCW C-index
    # for VIMP, matching crforest's metric (compute_uno_weights +
    # concordance_index_uno_cr). Override with RFSRC_USE_UNO=FALSE for the
    # historical Harrell-subset baseline.
    use_uno = os.environ.get("RFSRC_USE_UNO", "TRUE").upper()
    if use_uno not in {"TRUE", "FALSE"}:
        raise ValueError(f"RFSRC_USE_UNO must be TRUE or FALSE; got {use_uno!r}")
    block_size = int(os.environ.get("RFSRC_BLOCK_SIZE", "0")) or n_estimators
    ro.r(
        f"""
        fit__ <- rfsrc(Surv(time, event) ~ ., data=train_df,
            ntree={n_estimators}, nodesize=15, mtry=ceiling(sqrt({p})),
            splitrule="logrankCR", bootstrap="by.user", samp=samp_matrix,
            nsplit=10, ntime=0, importance="permute", block.size={block_size},
            use.uno={use_uno}, seed=-{int(seed)})
        """
    )
    with localconverter(converter):
        # rfSRC vimp$importance is (n_features, n_causes+1) where col 1 is overall.
        # Returns a flat vector that we reshape based on dim.
        imp_dim = list(np.asarray(ro.r("dim(fit__$importance)"), dtype=np.int64))
        imp_flat = np.asarray(ro.r("as.vector(fit__$importance)"), dtype=np.float64)
        rf_features = list(ro.r("rownames(fit__$importance)"))
        ro.r("rm(fit__); rm(samp_matrix)")
    rf_imp = imp_flat.reshape(imp_dim, order="F")  # R is column-major
    # rfSRC vimp$importance and sklearn permutation_importance share convention:
    # both report "performance drop after permutation" (higher = more important).
    # Empirical magnitudes line up after this — no sign flip needed.

    # crforest defaults feature names to "feature_{i}"; align by position
    # since crforest preserves column order.
    cr_indexed = cr_vimp.set_index("feature")
    cr_feat_order = list(cr_indexed.index)
    if len(cr_feat_order) != p:
        raise RuntimeError(f"crforest VIMP returned {len(cr_feat_order)} features, expected {p}")

    # Build per-cause comparison frames. rfSRC layout:
    #   col 0 = "all" (overall), col 1 = "event.1", col 2 = "event.2", ...
    rows = []
    for k in (1, 2):  # competing-risk causes
        cr_col = f"cause_{k}_vimp"
        if cr_col not in cr_indexed.columns:
            continue
        rf_col_idx = k
        if rf_col_idx >= rf_imp.shape[1]:
            continue
        df = pd.DataFrame(
            {
                "feature": feat_names,
                "cr": cr_indexed[cr_col].to_numpy(),  # positional alignment
                "rf": [rf_imp[rf_features.index(f), rf_col_idx] for f in feat_names],
            }
        )
        # Drop features with NaN rfSRC vimp (constant features within bootstrap fold)
        df = df.dropna()
        if len(df) < 3:
            continue
        sp_rho, _ = spearmanr(df["cr"], df["rf"])
        pe_r, _ = pearsonr(df["cr"], df["rf"])
        rows.append(
            {
                "dataset": dataset,
                "seed": seed,
                "cause": k,
                "n_features": len(df),
                "spearman_rho": float(sp_rho),
                "pearson_r": float(pe_r),
                "mean_abs_diff": float((df["cr"] - df["rf"]).abs().mean()),
                "cr_range": (float(df["cr"].min()), float(df["cr"].max())),
                "rf_range": (float(df["rf"].min()), float(df["rf"].max())),
            }
        )
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--datasets", nargs="+", default=list(DATASETS))
    parser.add_argument("--seeds", type=int, default=3)
    parser.add_argument("--n-estimators", type=int, default=100)
    parser.add_argument("--n-repeats", type=int, default=5)
    parser.add_argument("--out", type=Path, default=Path("/tmp/vimp_alignment.json"))
    parser.add_argument(
        "--mode",
        choices=("oob", "holdout"),
        default="oob",
        help="crforest VIMP mode for the comparison (default: oob)",
    )
    args = parser.parse_args()

    all_rows: list[dict] = []
    for ds in args.datasets:
        for s in range(1, args.seeds + 1):
            for r in run_cell(ds, s, args.n_estimators, args.n_repeats, args.mode):
                all_rows.append(r)
                print(
                    f"[{ds} seed={s} cause={r['cause']}] "
                    f"spearman={r['spearman_rho']:.3f} pearson={r['pearson_r']:.3f} "
                    f"mean|Δ|={r['mean_abs_diff']:.4f} "
                    f"(cr∈{r['cr_range']}, rf∈{r['rf_range']})",
                    flush=True,
                )

    df = pd.DataFrame(all_rows)
    print("\n=== per-dataset medians ===")
    print(
        f"{'dataset':>10s} | {'cause':>5s} | {'spearman':>9s} | {'pearson':>8s} | {'mean|Δ|':>8s}"
    )
    print("-" * 55)
    for (ds, c), sub in df.groupby(["dataset", "cause"]):
        print(
            f"{ds:>10s} | {c:>5d} | {sub['spearman_rho'].median():9.3f} | "
            f"{sub['pearson_r'].median():8.3f} | {sub['mean_abs_diff'].median():8.4f}"
        )
    print(
        "\nspearman rho = ranking agreement (academic users care most about this)\n"
        "pearson r    = absolute-value agreement\n"
        "mean|Δ|      = mean of |cr_vimp - rf_vimp| across features (C-index units)"
    )

    args.out.write_text(json.dumps(all_rows, indent=2))
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
