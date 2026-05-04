"""Replay rfSRC's permutation choices through comprisk, verify VIMP match.

Decisive direct-evidence test for cross-lib OOB VIMP equivalence at
use.uno=FALSE, block.size=ntree. Result on hd seed=1: Spearman=1.0,
mean|Δ|<0.001 in C-index units. The two libraries implement the same
algorithm.

Pipeline: instrumented rfSRC at /tmp/rfsrc_patched_lib emits
`vimp_perm tree=T a=feat b=dst_sample c=src_sample` events for every
per-tree per-feature permutation; we parse the trace, replay those
permutations through comprisk's trees, and compare to rfSRC's reported
$importance.

Requires `bash _rfsrc_patches/regen.sh` first.
"""

from __future__ import annotations

import os
import re
import sys
from collections import defaultdict

import numpy as np
import pandas as pd
import rpy2.robjects as ro
from rpy2.robjects.conversion import localconverter
from rpy2.robjects.packages import importr
from scipy.stats import pearsonr, spearmanr

from comprisk import CompetingRiskForest
from comprisk._importance import _predict_tree_mortality
from comprisk.metrics import compute_uno_weights, concordance_index_uno_cr
from validation.alignment import _rpy2_converter
from validation.datasets import load as load_dataset
from validation.splits import _SPLITS_DIR

PATCHED_LIB = "/tmp/rfsrc_patched_lib"
TRACE_PATH = "/tmp/vimp_perm.trace"


def _print(msg: str) -> None:
    print(msg, flush=True)


_RE = re.compile(r"vimp_perm tree=(\d+) a=(-?\d+) b=(-?\d+) c=([-\d.]+)")


def parse_trace(path: str) -> dict[tuple[int, int], dict[int, int]]:
    """Returns perm[(tree_1idx, feat_0idx)][dst_sample_0idx] = src_sample_0idx.

    rfSRC's internal indices are NR-style 1-indexed for trees, features,
    and samples; we convert all-but-tree to 0-indexed here, leaving
    tree to be converted at the call site.
    """
    perm: dict[tuple[int, int], dict[int, int]] = defaultdict(dict)
    with open(path) as f:
        for line in f:
            m = _RE.match(line)
            if not m:
                continue
            tree = int(m.group(1))
            feat = int(m.group(2)) - 1
            dst = int(m.group(3)) - 1
            src = round(float(m.group(4))) - 1
            perm[(tree, feat)][dst] = src
    return perm


def main() -> int:
    os.environ["RFSRC_TRACE"] = TRACE_PATH
    importr("randomForestSRC", lib_loc=PATCHED_LIB)
    converter = _rpy2_converter()

    dataset = "hd"
    seed = 1
    ntree = 100

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

    _print(f"# vimp_perm_replay: dataset={dataset} seed={seed} ntree={ntree} p={p}")
    _print(f"# trace path: {TRACE_PATH}")

    # comprisk paired-bootstrap fit
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

    # rfSRC fit with importance="permute" on patched lib (writes trace)
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
    # Open the trace file (rfsrc_trace_maybe_open re-opens when env changes)
    if os.path.exists(TRACE_PATH):
        os.remove(TRACE_PATH)
    ro.r(
        f"""
        Sys.setenv(RFSRC_TRACE='{TRACE_PATH}')
        fit__ <- rfsrc(Surv(time, event) ~ ., data=train_df,
            ntree={ntree}, nodesize=15, mtry=ceiling(sqrt({p})),
            splitrule="logrankCR", bootstrap="by.user", samp=samp_matrix,
            nsplit=10, ntime=0, importance="permute", block.size={ntree},
            use.uno=FALSE, seed=-{int(seed)})
        Sys.unsetenv("RFSRC_TRACE")
        """
    )
    # rfSRC $importance for CR has shape (p, n_causes) with colnames event.1, event.2.
    with localconverter(converter):
        imp_dim = list(np.asarray(ro.r("dim(fit__$importance)"), dtype=np.int64))
        imp_flat = np.asarray(ro.r("as.vector(fit__$importance)"), dtype=np.float64)
    rf_imp = imp_flat.reshape(imp_dim, order="F")
    _print(f"  rf_imp shape: {rf_imp.shape}\n{rf_imp}")

    perm = parse_trace(TRACE_PATH)
    _print(f"  parsed {len(perm)} (tree, feat) entries from trace")
    if not perm:
        _print("ERROR: no permutation events parsed; aborting")
        return 1

    perm_0idx: dict[tuple[int, int], dict[int, int]] = {
        (t1 - 1, f0): m for (t1, f0), m in perm.items()
    }

    # comprisk unpermuted ensemble OOB mortality (baseline)
    ref_pred = np.zeros((2, n_tr), dtype=np.float64)
    count = np.zeros(n_tr, dtype=np.int64)
    for t in range(ntree):
        oob = np.asarray(forest.oob_indices_[t], dtype=np.int64)
        if len(oob) == 0:
            continue
        for ci, c in enumerate([1, 2]):
            ref_pred[ci, oob] += _predict_tree_mortality(
                forest.trees_[t],
                Xtr[oob],
                cause=c,
                mode=forest.mode,
                bin_edges=bin_edges,
                time_grid=time_grid,
            )
        count[oob] += 1
    mask = count > 0
    ref_pred[:, mask] /= count[mask]

    # Uno IPCW weights computed on full training data (rfSRC RF_unoWeight convention)
    uno_weights = compute_uno_weights(Ttr, Etr)
    weights_oob = uno_weights[mask]

    # Reference C-index (unpermuted)
    ref_C = np.zeros(2, dtype=np.float64)
    for ci, c in enumerate([1, 2]):
        ref_C[ci] = concordance_index_uno_cr(
            Etr[mask],
            Ttr[mask],
            ref_pred[ci, mask],
            cause=c,
            weights=weights_oob,
        )
    _print(f"  comprisk reference C: c1={ref_C[0]:.5f}, c2={ref_C[1]:.5f}")

    # For each feature, build per-feature ensemble using rfSRC's permutations
    cr_with_rf_perms_vimp = np.zeros((p, 2), dtype=np.float64)
    for feat in range(p):
        perm_pred = np.zeros((2, n_tr), dtype=np.float64)
        perm_count = np.zeros(n_tr, dtype=np.int64)
        n_trees_with_perm = 0
        for t in range(ntree):
            oob = np.asarray(forest.oob_indices_[t], dtype=np.int64)
            if len(oob) == 0:
                continue
            tree_perm_map = perm_0idx.get((t, feat))
            if tree_perm_map is None:
                continue
            n_trees_with_perm += 1
            X_perm = Xtr.copy()
            for dst_idx in oob:
                src_idx = tree_perm_map.get(dst_idx)
                if src_idx is not None:
                    X_perm[dst_idx, feat] = Xtr[src_idx, feat]
            for ci, c in enumerate([1, 2]):
                perm_pred[ci, oob] += _predict_tree_mortality(
                    forest.trees_[t],
                    X_perm[oob],
                    cause=c,
                    mode=forest.mode,
                    bin_edges=bin_edges,
                    time_grid=time_grid,
                )
            perm_count[oob] += 1
        local_mask = perm_count > 0
        perm_pred[:, local_mask] /= perm_count[local_mask]
        local_weights = uno_weights[local_mask]
        for ci, c in enumerate([1, 2]):
            perm_C = concordance_index_uno_cr(
                Etr[local_mask],
                Ttr[local_mask],
                perm_pred[ci, local_mask],
                cause=c,
                weights=local_weights,
            )
            cr_with_rf_perms_vimp[feat, ci] = ref_C[ci] - perm_C
        _print(
            f"  feat {feat}: trees_with_perm={n_trees_with_perm}/{ntree}  "
            f"vimp[c1]={cr_with_rf_perms_vimp[feat, 0]:+.5f}  "
            f"vimp[c2]={cr_with_rf_perms_vimp[feat, 1]:+.5f}"
        )

    _print("\n## Compare comprisk-replay-with-rfSRC-perms vs rfSRC-reported vimp")
    for ci in range(2):
        cr_v = cr_with_rf_perms_vimp[:, ci]
        rf_v = rf_imp[:, ci]
        rho, _ = spearmanr(cr_v, rf_v)
        r, _ = pearsonr(cr_v, rf_v)
        diff = np.abs(cr_v - rf_v)
        _print(
            f"  cause {ci + 1}: spearman={rho:+.4f}  pearson={r:+.4f}  "
            f"mean|Δ|={diff.mean():.5f}  max|Δ|={diff.max():.5f}"
        )
        _print(f"    cr_replay: {cr_v}")
        _print(f"    rf_actual: {rf_v}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
