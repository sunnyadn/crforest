"""Isolate permutation-RNG-only noise floor for comprisk OOB VIMP on hd.

Fix the forest fit, vary ONLY compute_importance(random_state=k). Compute
pairwise Spearman between resulting VIMPs. Establishes the per-permutation
noise floor that any cross-lib comparison saturates against (median +0.83
on hd cause_1 at fit_seed=1).
"""

from __future__ import annotations

import sys
from itertools import combinations

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from comprisk import CompetingRiskForest
from validation.datasets import load as load_dataset
from validation.splits import _SPLITS_DIR


def _print(msg: str) -> None:
    print(msg, flush=True)


def main() -> int:
    dataset = "hd"
    fit_seed = 1
    n_perm_seeds = 10
    ntree = 100

    X, T, E = load_dataset(dataset)
    splits_df = pd.read_parquet(_SPLITS_DIR / f"{dataset}.parquet")
    row = splits_df[splits_df["seed"] == fit_seed]
    tr = np.sort(row.loc[row["fold"] == "train", "sample_id"].to_numpy(np.int64))

    _print(
        f"# permutation_only_noise: dataset={dataset} fit_seed={fit_seed} "
        f"ntree={ntree} n_perm_seeds={n_perm_seeds}"
    )

    forest = CompetingRiskForest(
        n_estimators=ntree,
        min_samples_leaf=1,
        min_samples_split=30,
        max_features="sqrt",
        bootstrap=True,
        random_state=fit_seed,
        equivalence="rfsrc",
    ).fit(X[tr], T[tr], E[tr])
    _print(f"forest fit done; n_tr={len(tr)}, p={X.shape[1]}")

    vimps_c1 = []
    vimps_c2 = []
    for k in range(n_perm_seeds):
        v = forest.compute_importance(random_state=10_000 + k)
        vimps_c1.append(v["cause_1_vimp"].to_numpy())
        vimps_c2.append(v["cause_2_vimp"].to_numpy())
        _print(f"  perm_seed={k}: vimp computed")

    rhos_c1 = []
    rhos_c2 = []
    for i, j in combinations(range(n_perm_seeds), 2):
        r1, _ = spearmanr(vimps_c1[i], vimps_c1[j])
        r2, _ = spearmanr(vimps_c2[i], vimps_c2[j])
        rhos_c1.append(r1)
        rhos_c2.append(r2)
    rhos_c1 = np.asarray(rhos_c1)
    rhos_c2 = np.asarray(rhos_c2)

    _print(
        f"\n## Permutation-only pairwise Spearman across {n_perm_seeds} perm-seeds, "
        f"{len(rhos_c1)} pairs"
    )
    for label, rhos in [("cause_1", rhos_c1), ("cause_2", rhos_c2)]:
        _print(
            f"  {label}: median={np.median(rhos):+.3f}  "
            f"q25={np.quantile(rhos, 0.25):+.3f}  "
            f"q75={np.quantile(rhos, 0.75):+.3f}  "
            f"min={rhos.min():+.3f}  max={rhos.max():+.3f}"
        )

    _print("\n## Reference distributions on hd cause_1")
    _print("  cross-lib (10 seeds, paired-bootstrap, use.uno=FALSE):")
    _print("    median=+0.37, max=+0.66, min=-0.14")
    _print("  within-lib (45 pairs across bootstrap+perm seeds, ntree=100):")
    _print("    median=+0.71, q25=+0.54, q75=+0.83, max=+1.00")
    return 0


if __name__ == "__main__":
    sys.exit(main())
