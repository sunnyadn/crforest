"""Falsification test for the grid_mismatch attribution.

Hypothesis: if the dominant cause of the synthetic ntree=1 CIF gap is
crforest's 256-quantile grid failing to include rfSRC's candidate
partitions, then running crforest in ``mode="reference"`` (which
evaluates splits at every midpoint between sorted unique values --
the SAME candidate set rfSRC uses) should collapse the gap.

Compares three cells on synthetic, ntree=1, bootstrap=F, nsplit=0,
mtry=p, min_samples_split=2, rfSRC ntime=0, seeds 1..10:

  default_mode:  crforest mode="default" (256-quantile bins)
  reference_mode: crforest mode="reference" (observation-level candidates)

Report: per-seed p95 |cr - rf| over the test fold's CIF on cause 1,
aggregated to cross_p95_cif. If reference_mode >> default_mode:
the hypothesis is falsified (grid not the main driver). If
reference_mode << default_mode: the hypothesis is confirmed.

Requires the instrumented rfSRC at /tmp/rfsrc_patched_lib; rebuild via
``bash validation/alignment/_rfsrc_patches/regen.sh``.

Run:
    uv run --extra maintainer python -m validation.alignment.grid_mismatch_falsification \\
        --seeds 10
"""

from __future__ import annotations

import argparse

import numpy as np
import pandas as pd

from crforest import CompetingRiskForest
from validation.alignment import _rpy2_converter
from validation.alignment.equivalence_gate import (
    build_reference_grid,
    eval_on_ref_grid,
)
from validation.datasets import load as load_dataset
from validation.splits import _SPLITS_DIR


def run_one(dataset: str, seed: int, mode: str) -> dict:
    import rpy2.robjects as ro
    from rpy2.robjects.conversion import localconverter
    from rpy2.robjects.packages import importr

    importr("randomForestSRC", lib_loc="/tmp/rfsrc_patched_lib")
    converter = _rpy2_converter()

    X, time, event = load_dataset(dataset)
    ref_grid = build_reference_grid(time, event)
    splits_df = pd.read_parquet(_SPLITS_DIR / f"{dataset}.parquet")
    row = splits_df[splits_df["seed"] == seed]
    train_idx = np.sort(row.loc[row["fold"] == "train", "sample_id"].to_numpy(np.int64))
    test_idx = np.sort(row.loc[row["fold"] == "test", "sample_id"].to_numpy(np.int64))
    p = X.shape[1]

    kwargs = dict(
        n_estimators=1,
        min_samples_leaf=1,
        min_samples_split=30,
        max_features=None,
        bootstrap=False,
        random_state=seed,
        mode=mode,
        nsplit=0,
    )
    if mode == "default":
        kwargs["time_grid"] = 200
        kwargs["split_ntime"] = None
    forest = CompetingRiskForest(**kwargs).fit(X[train_idx], time[train_idx], event[train_idx])
    cif_cr_raw = np.transpose(forest.predict_cif(X[test_idx]), (0, 2, 1))
    cr_grid = np.asarray(forest.unique_times_, dtype=np.float64)

    feat_cols = [f"x{j}" for j in range(p)]
    train_df = pd.DataFrame(X[train_idx], columns=feat_cols)
    train_df["time"] = time[train_idx]
    train_df["event"] = event[train_idx].astype(np.int32)
    test_df = pd.DataFrame(X[test_idx], columns=feat_cols)

    with localconverter(converter):
        ro.globalenv["train_df"] = train_df
        ro.globalenv["test_df"] = test_df
    ro.r(f"""
        fit__ <- rfsrc(Surv(time, event) ~ ., data=train_df,
            ntree=1, nodesize=15, nodedepth=-1, mtry={p},
            splitrule="logrankCR", bootstrap="none",
            nsplit=0, ntime=0, seed=-{int(seed)})
        pred__ <- predict(fit__, newdata=test_df)
    """)
    with localconverter(converter):
        ti = np.asarray(ro.r("fit__$time.interest"), dtype=np.float64)
        cif_flat = np.asarray(ro.r("pred__$cif"), dtype=np.float64)
        ro.r("rm(fit__); rm(pred__)")

    n_te = len(test_idx)
    cif_rf_raw = cif_flat.reshape(n_te, len(ti), 2)
    cif_cr = eval_on_ref_grid(cif_cr_raw[:, :, 0], cr_grid, ref_grid)
    cif_rf = eval_on_ref_grid(cif_rf_raw[:, :, 0], ti, ref_grid)

    diff = np.abs(cif_cr - cif_rf)
    return {
        "seed": seed,
        "mode": mode,
        "p95_cif": float(np.percentile(diff, 95)),
        "max_cif": float(diff.max()),
        "mean_cif": float(diff.mean()),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", default="synthetic")
    parser.add_argument("--seeds", type=int, default=10)
    args = parser.parse_args()

    rows = []
    for mode in ("default", "reference"):
        for s in range(1, args.seeds + 1):
            r = run_one(args.dataset, s, mode)
            rows.append(r)
            print(
                f"[{mode} seed={s}] p95={r['p95_cif']:.4f} max={r['max_cif']:.4f} "
                f"mean={r['mean_cif']:.4f}",
                flush=True,
            )

    df = pd.DataFrame(rows)
    print("\n=== summary ===", flush=True)
    for mode in ("default", "reference"):
        sub = df[df["mode"] == mode]
        p95s = sub["p95_cif"].to_numpy()
        print(
            f"  {mode:>10}: cross_p95_cif (median over seeds) = {float(np.median(p95s)):.4f}"
            f", individual = {[round(x, 4) for x in p95s.tolist()]}",
            flush=True,
        )

    med_default = float(np.median(df[df["mode"] == "default"]["p95_cif"]))
    med_reference = float(np.median(df[df["mode"] == "reference"]["p95_cif"]))
    ratio = med_reference / max(med_default, 1e-12)
    print(f"\n  reference / default = {ratio:.3f}", flush=True)
    if ratio < 0.3:
        print(
            "  ==> grid_mismatch hypothesis STRONGLY CONFIRMED (reference < 30% of default)",
            flush=True,
        )
    elif ratio < 0.7:
        print("  ==> grid_mismatch hypothesis CONFIRMED as major contributor", flush=True)
    elif ratio < 1.2:
        print("  ==> grid_mismatch hypothesis FALSIFIED (reference ~= default)", flush=True)
    else:
        print("  ==> reference worse than default (unexpected)", flush=True)


if __name__ == "__main__":
    main()
