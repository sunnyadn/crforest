"""Within-lib stability of crforest OOB VIMP across fit random_state.

For each real dataset, fits the SAME training data with N different
random_state values and computes pairwise Spearman correlation between
VIMP rankings. This characterizes the seed-to-seed noise floor.

Why this matters
----------------
The cross-lib Spearman vs rfSRC default is 0.4-0.7 (vimp_alignment.py).
Without a within-lib baseline, we can't tell whether 0.5 means "real
algorithmic divergence" or "this is the noise floor of the method on
this dataset". If within-lib pairwise Spearman is ≈0.5 too, the gap
to rfSRC is at the noise floor; if within-lib is ≥0.9, the cross-lib
gap is genuine and methodology-specific.

Output
------
Per (dataset, cause) pair:
- Median pairwise within-lib Spearman across all (n_seeds choose 2) pairs.
- 25/75 quantile band.
- Compared side-by-side with the cross-lib Spearman previously measured
  in vimp_alignment.py (hardcoded reference numbers from
  project_vimp_alignment.md).
"""

from __future__ import annotations

import sys
from itertools import combinations

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from crforest import CompetingRiskForest
from validation.datasets import load as load_dataset
from validation.splits import _SPLITS_DIR


def _print(msg: str) -> None:
    print(msg, flush=True)


# Cross-lib Spearman (median across 3 seeds, ntree=100, n_repeats=5)
# from vimp_alignment.py per project_vimp_alignment.md.
# These are the numbers we want a within-lib baseline against.
CROSS_LIB_OOB_SPEARMAN = {
    ("pbc", 1): 0.46,
    ("hd", 1): -0.37,  # post-rfSRC-subset C-index
    ("follic", 1): 0.70,
    ("synthetic", 1): -0.36,
}


def fit_and_get_vimp(
    X: np.ndarray, T: np.ndarray, event: np.ndarray, fit_seed: int, n_estimators: int
) -> pd.DataFrame:
    forest = CompetingRiskForest(
        n_estimators=n_estimators,
        min_samples_leaf=1,
        min_samples_split=30,
        max_features="sqrt",
        bootstrap=True,
        random_state=fit_seed,
        equivalence="rfsrc",
    ).fit(X, T, event)
    return forest.compute_importance()


def measure_dataset(dataset: str, n_seeds: int, n_estimators: int) -> pd.DataFrame:
    X, time_all, event_all = load_dataset(dataset)
    splits_df = pd.read_parquet(_SPLITS_DIR / f"{dataset}.parquet")
    row = splits_df[splits_df["seed"] == 1]
    tr = np.sort(row.loc[row["fold"] == "train", "sample_id"].to_numpy(np.int64))
    Xtr = X[tr]
    Ttr = time_all[tr]
    Etr = event_all[tr]

    vimps_c1 = []
    vimps_c2 = []
    vimps_comp = []
    for s in range(n_seeds):
        vimp_df = fit_and_get_vimp(Xtr, Ttr, Etr, fit_seed=s, n_estimators=n_estimators)
        vimps_c1.append(vimp_df["cause_1_vimp"].to_numpy())
        vimps_c2.append(vimp_df["cause_2_vimp"].to_numpy())
        vimps_comp.append(vimp_df["composite_vimp"].to_numpy())
        _print(f"  {dataset} seed={s}: vimp computed (p={Xtr.shape[1]} features)")

    rows = []
    for cause_label, vimps in [
        ("cause_1", vimps_c1),
        ("cause_2", vimps_c2),
        ("composite", vimps_comp),
    ]:
        rhos = []
        for i, j in combinations(range(n_seeds), 2):
            rho, _ = spearmanr(vimps[i], vimps[j])
            rhos.append(rho)
        rhos = np.asarray(rhos)
        rows.append(
            {
                "dataset": dataset,
                "cause": cause_label,
                "n_pairs": len(rhos),
                "median_spearman": float(np.median(rhos)),
                "q25_spearman": float(np.quantile(rhos, 0.25)),
                "q75_spearman": float(np.quantile(rhos, 0.75)),
                "min_spearman": float(np.min(rhos)),
                "max_spearman": float(np.max(rhos)),
            }
        )
    return pd.DataFrame(rows)


def main() -> int:
    datasets = ("pbc", "hd", "follic", "synthetic")
    n_seeds = 10
    n_estimators = 100

    _print(
        f"# vimp_within_stability: datasets={datasets}, "
        f"n_seeds={n_seeds} ({n_seeds * (n_seeds - 1) // 2} pairs), "
        f"ntree={n_estimators}"
    )
    _print("# Within-lib pairwise Spearman across fit random_state, fixed train split.")
    _print("")

    all_dfs = []
    for ds in datasets:
        _print(f"=== {ds} ===")
        df = measure_dataset(ds, n_seeds=n_seeds, n_estimators=n_estimators)
        all_dfs.append(df)
        _print("")

    summary = pd.concat(all_dfs, ignore_index=True)

    _print("## Within-lib pairwise Spearman summary")
    _print(
        f"{'dataset':10s} {'cause':10s}  "
        f"{'within median':>14s}  {'IQR':>16s}  "
        f"{'min':>6s} {'max':>6s}  cross-lib (rfSRC default, c1)"
    )
    for _, row in summary.iterrows():
        cross_ref = CROSS_LIB_OOB_SPEARMAN.get((row["dataset"], 1), float("nan"))
        cross_str = f"{cross_ref:+.2f}" if row["cause"] == "cause_1" else "—"
        _print(
            f"{row['dataset']:10s} {row['cause']:10s}  "
            f"{row['median_spearman']:>+14.3f}  "
            f"[{row['q25_spearman']:+.3f},{row['q75_spearman']:+.3f}]  "
            f"{row['min_spearman']:>+6.2f} {row['max_spearman']:>+6.2f}  "
            f"{cross_str}"
        )

    summary.to_parquet("/tmp/vimp_within_stability.parquet")
    _print("\nResults written to /tmp/vimp_within_stability.parquet")
    return 0


if __name__ == "__main__":
    sys.exit(main())
