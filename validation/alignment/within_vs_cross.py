"""Within-lib vs cross-lib p95 |ΔCIF| under the equivalence='rfsrc' preset.

For each of 4 datasets, fit comprisk (preset) and rfSRC (bootstrap=by.user
matched to comprisk's inbag_) on 4 seeds, then compute on the reference
event-time grid:

  cross_p95_cif         = median over seeds of  p95(|cif_cr[s] - cif_rf[s]|)
  within_cr_p95_cif     = max over pairs of    p95(|cif_cr[s_a] - cif_cr[s_b]|)
  within_rf_p95_cif     = max over pairs of    p95(|cif_rf[s_a] - cif_rf[s_b]|)

If cross is at or below within, the two libraries are indistinguishable
from same-library seed-to-seed bagging variance — strongest equivalence
claim possible.

Run:
    uv run --extra maintainer python -m validation.alignment.within_vs_cross \\
        --seeds 4 --n-estimators 100
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from comprisk import CompetingRiskForest
from validation.alignment import _rpy2_converter
from validation.alignment.equivalence_gate import (
    build_reference_grid,
    eval_on_ref_grid,
)
from validation.datasets import load as load_dataset
from validation.splits import _SPLITS_DIR

DATASETS = ("pbc", "hd", "follic", "synthetic")


def _fit_pair(dataset: str, seed: int, n_estimators: int):
    import rpy2.robjects as ro
    from rpy2.robjects.conversion import localconverter
    from rpy2.robjects.packages import importr

    importr("randomForestSRC")
    converter = _rpy2_converter()

    X, time_all, event_all = load_dataset(dataset)
    splits_df = pd.read_parquet(_SPLITS_DIR / f"{dataset}.parquet")
    row = splits_df[splits_df["seed"] == seed]
    tr = np.sort(row.loc[row["fold"] == "train", "sample_id"].to_numpy(np.int64))
    te = np.sort(row.loc[row["fold"] == "test", "sample_id"].to_numpy(np.int64))
    ref_grid = build_reference_grid(time_all, event_all)
    p = X.shape[1]
    n_tr = len(tr)

    forest = CompetingRiskForest(
        n_estimators=n_estimators,
        min_samples_leaf=1,
        min_samples_split=30,
        max_features="sqrt",
        bootstrap=True,
        random_state=seed,
        equivalence="rfsrc",
    ).fit(X[tr], time_all[tr], event_all[tr])
    cif_cr_raw = np.transpose(forest.predict_cif(X[te]), (0, 2, 1))
    cif_cr = eval_on_ref_grid(cif_cr_raw[:, :, 0], forest.time_grid_, ref_grid)

    feat_cols = [f"x{j}" for j in range(p)]
    train_df = pd.DataFrame(X[tr], columns=feat_cols)
    train_df["time"] = time_all[tr]
    train_df["event"] = event_all[tr].astype(np.int32)
    test_df = pd.DataFrame(X[te], columns=feat_cols)

    with localconverter(converter):
        ro.globalenv["train_df"] = train_df
        ro.globalenv["test_df"] = test_df
        ro.globalenv["samp_matrix"] = ro.r.matrix(
            ro.FloatVector(forest.inbag_.T.reshape(-1).astype(np.float64)),
            nrow=n_tr,
            ncol=n_estimators,
        )
    ro.r(
        f"""
        fit__ <- rfsrc(Surv(time, event) ~ ., data=train_df,
            ntree={n_estimators}, nodesize=15, mtry=ceiling(sqrt({p})),
            splitrule="logrankCR", bootstrap="by.user", samp=samp_matrix,
            nsplit=10, ntime=0, seed=-{int(seed)})
        pred__ <- predict(fit__, newdata=test_df)
        """
    )
    with localconverter(converter):
        ti = np.asarray(ro.r("fit__$time.interest"), dtype=np.float64)
        cif_flat = np.asarray(ro.r("pred__$cif"), dtype=np.float64)
        ro.r("rm(fit__); rm(pred__); rm(samp_matrix)")
    n_te = len(te)
    cif_rf_raw = cif_flat.reshape(n_te, len(ti), 2)
    cif_rf = eval_on_ref_grid(cif_rf_raw[:, :, 0], ti, ref_grid)

    return cif_cr, cif_rf


def _p95(x: np.ndarray) -> float:
    return float(np.percentile(np.abs(x), 95))


def run_dataset(dataset: str, seeds: list[int], n_estimators: int) -> dict:
    cells = []
    for s in seeds:
        cif_cr, cif_rf = _fit_pair(dataset, s, n_estimators)
        cells.append({"seed": s, "cif_cr": cif_cr, "cif_rf": cif_rf})
        print(
            f"  [{dataset} seed={s}] cross_p95={_p95(cif_cr - cif_rf):.4f}",
            flush=True,
        )

    # Pair seeds (1,2), (3,4), ... for within-lib gaps.
    cells.sort(key=lambda c: c["seed"])
    cross_per_seed = [_p95(c["cif_cr"] - c["cif_rf"]) for c in cells]
    within_cr, within_rf = [], []
    for i in range(0, len(cells) - 1, 2):
        a, b = cells[i], cells[i + 1]
        within_cr.append(_p95(a["cif_cr"] - b["cif_cr"]))
        within_rf.append(_p95(a["cif_rf"] - b["cif_rf"]))
    return {
        "dataset": dataset,
        "cross_p95_cif_median": float(np.median(cross_per_seed)),
        "cross_p95_cif_max": float(np.max(cross_per_seed)),
        "within_cr_p95_cif_max": float(np.max(within_cr)) if within_cr else float("nan"),
        "within_rf_p95_cif_max": float(np.max(within_rf)) if within_rf else float("nan"),
        "cross_per_seed": cross_per_seed,
        "within_cr_per_pair": within_cr,
        "within_rf_per_pair": within_rf,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--datasets", nargs="+", default=list(DATASETS))
    parser.add_argument("--seeds", type=int, default=4, help="number of seeds (must be even)")
    parser.add_argument("--n-estimators", type=int, default=100)
    parser.add_argument("--out", type=Path, default=Path("/tmp/within_vs_cross.json"))
    args = parser.parse_args()

    if args.seeds % 2 != 0:
        raise SystemExit("--seeds must be even (paired into within-lib gaps)")
    seed_list = list(range(1, args.seeds + 1))

    rows = []
    for ds in args.datasets:
        print(f"=== {ds} ===", flush=True)
        rows.append(run_dataset(ds, seed_list, args.n_estimators))

    print("\n=== summary ===")
    print(
        f"{'dataset':>10s} | {'cross_p95':>10s} | {'within_cr':>10s} | "
        f"{'within_rf':>10s} | {'cross/within_max':>17s}"
    )
    print("-" * 72)
    for r in rows:
        denom = max(r["within_cr_p95_cif_max"], r["within_rf_p95_cif_max"])
        ratio = r["cross_p95_cif_median"] / denom if denom > 0 else float("inf")
        print(
            f"{r['dataset']:>10s} | {r['cross_p95_cif_median']:10.4f} | "
            f"{r['within_cr_p95_cif_max']:10.4f} | {r['within_rf_p95_cif_max']:10.4f} | "
            f"{ratio:17.2f}"
        )
    print(
        "\nratio < 1.0  => cross-lib gap below within-lib noise floor (indistinguishable)\n"
        "ratio ~ 1.0  => cross-lib gap at within-lib noise floor (effectively indistinguishable)\n"
        "ratio >> 1.0 => cross-lib gap exceeds within-lib variation (residual systematic gap)"
    )

    args.out.write_text(json.dumps(rows, indent=2))
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
