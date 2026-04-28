"""Root-node per-bin + same-partition diagnostic.

Fits a single deterministic tree (ntree=1, bootstrap=F, nsplit=0,
split_ntime=None, rfSRC ntime=0, rng_mode=rfsrc_aligned) on both
libs and produces two pieces of evidence about the root split:

  1. ``per-bin stat profile`` -- for the winning feature at the root,
     dump the full per-bin logrank-z vector from each lib. Reports
     top-K bins in each lib and the near-tie spread.
  2. ``same-partition alignment`` -- map crforest's 256-quantile bin
     boundaries to left-size (count of samples sent left) and match
     against rfSRC's sorted-observation boundaries (where obs_j =
     left-size directly). Bins with equal left-size evaluate the
     IDENTICAL partition, so stats can be compared apples-to-apples.

Structural tree-level divergence across both libs is handled by
``cascade_diagnostic.py``; this script focuses on the per-bin
numerical noise and the discretization-grid alignment at the root.

rfSRC must be loaded from the instrumented library at
/tmp/rfsrc_patched_lib (has feat_stat_CR + bin_stat_CR + node_start
events). Rebuild via ``bash validation/alignment/_rfsrc_patches/regen.sh``.

Run:
    uv run --extra maintainer python -m validation.alignment.rank_flip_diagnostic \\
        --dataset synthetic --seed 1
"""

from __future__ import annotations

import argparse
import math
import os
import re
from pathlib import Path

import numpy as np
import pandas as pd

from crforest import CompetingRiskForest
from crforest._binning import apply_bins
from validation.alignment import _rpy2_converter
from validation.datasets import load as load_dataset
from validation.splits import _SPLITS_DIR

NODE_RE_CR = re.compile(r"^node_start n=(\d+)")
NODE_RE_RF = re.compile(r"^node_start tree=\d+ a=(\d+)")
FEAT_RE_CR = re.compile(r"^feat_stat_CR covariate=(\d+) bin=(\S+) stat=([-\d.eE+]+)")
FEAT_RE_RF = re.compile(r"^feat_stat_CR tree=\d+ a=(\d+) b=(\d+) c=([-\d.eE+]+)")
BIN_STAT_RE_RF = re.compile(r"^bin_stat_CR tree=\d+ a=(\d+) b=(\d+) c=([-\d.eE+]+)")


def _crforest_per_bin_stats(
    X_binned_col: np.ndarray,
    t_idx: np.ndarray,
    event: np.ndarray,
    n_bins: int,
    n_causes: int,
    n_time_bins: int,
    min_samples_leaf: int,
) -> np.ndarray:
    """Pure-numpy shadow of _best_split_in_feature (logrankCR), returning
    the full per-bin stat vector in z-scale (|num|/sqrt(var)) so values
    are directly comparable to rfSRC's bin_stat_CR delta."""
    n_node = X_binned_col.shape[0]
    event_hist = np.zeros((n_bins, n_causes, n_time_bins), dtype=np.float64)
    at_risk_hist = np.zeros((n_bins, n_time_bins), dtype=np.float64)
    for i in range(n_node):
        b = int(X_binned_col[i])
        t = int(t_idx[i])
        e = int(event[i])
        at_risk_hist[b, t] += 1.0
        if e > 0:
            event_hist[b, e - 1, t] += 1.0
    at_risk_hist = np.flip(np.cumsum(np.flip(at_risk_hist, axis=1), axis=1), axis=1)

    at_risk_total = at_risk_hist.sum(axis=0)
    d_k_total = event_hist.sum(axis=0)
    d_any_total = d_k_total.sum(axis=0)

    at_risk_total_inc = np.empty((n_causes, n_time_bins), dtype=np.float64)
    for q in range(n_causes):
        cumsum = 0.0
        for t in range(n_time_bins):
            at_risk_total_inc[q, t] = at_risk_total[t] + cumsum
            cumsum += d_any_total[t] - d_k_total[q, t]

    at_risk_left = np.zeros(n_time_bins, dtype=np.float64)
    d_k_left = np.zeros((n_causes, n_time_bins), dtype=np.float64)
    at_risk_left_inc = np.empty((n_causes, n_time_bins), dtype=np.float64)
    samples_per_bin = at_risk_hist[:, 0]

    out = np.full(n_bins - 1, np.nan, dtype=np.float64)
    n_left_running = 0
    for b in range(n_bins - 1):
        n_left_running += int(samples_per_bin[b])
        at_risk_left += at_risk_hist[b]
        d_k_left += event_hist[b]
        d_any_left = d_k_left.sum(axis=0)

        n_right_running = n_node - n_left_running
        if n_left_running < min_samples_leaf or n_right_running < min_samples_leaf:
            continue

        for q in range(n_causes):
            cumsum = 0.0
            for t in range(n_time_bins):
                at_risk_left_inc[q, t] = at_risk_left[t] + cumsum
                cumsum += d_any_left[t] - d_k_left[q, t]

        num_sum = 0.0
        var_sum = 0.0
        for k in range(n_causes):
            for t in range(n_time_bins):
                if at_risk_total[t] == 0.0:
                    continue
                arinc_t = at_risk_total_inc[k, t]
                arlinc_t = at_risk_left_inc[k, t]
                d_t = d_k_total[k, t]
                dl_t = d_k_left[k, t]
                num_sum += dl_t - d_t * arlinc_t / arinc_t
                if at_risk_total[t] >= 2.0:
                    var_sum += (
                        d_t
                        * arlinc_t
                        * (arinc_t - arlinc_t)
                        * (arinc_t - d_t)
                        / (arinc_t * arinc_t * (arinc_t - 1.0))
                    )
        if var_sum < 1e-12:
            continue
        out[b] = abs(num_sum) / math.sqrt(var_sum)
    return out


def _root_feat_stats(path: Path, *, rfsrc_fmt: bool) -> list[tuple[int, int, float]]:
    """Return the root node's feat_stat_CR events as (feat_1b, bin, z_stat).
    crforest's stat is num^2/var; converted to z-scale for parity with rfSRC."""
    node_re = NODE_RE_RF if rfsrc_fmt else NODE_RE_CR
    feat_re = FEAT_RE_RF if rfsrc_fmt else FEAT_RE_CR
    root: list[tuple[int, int, float]] = []
    in_root = False
    for line in path.read_text().splitlines():
        if node_re.match(line):
            if in_root:
                break  # next node_start ends root
            in_root = True
            continue
        if not in_root:
            continue
        m = feat_re.match(line)
        if m:
            if rfsrc_fmt:
                root.append((int(m.group(1)), int(m.group(2)), float(m.group(3))))
            else:
                z_sq = float(m.group(3))
                root.append((int(m.group(1)), int(m.group(2)), math.sqrt(max(z_sq, 0.0))))
    return root


def _root_rf_bin_stats(path: Path) -> dict[int, list[tuple[int, float]]]:
    """Parse rfSRC bin_stat_CR events from the root node only. Returns
    ``{feat_1b: [(obs_j, stat_z), ...]}``."""
    out: dict[int, list[tuple[int, float]]] = {}
    in_root = False
    seen_any = False
    for line in path.read_text().splitlines():
        if line.startswith("node_start"):
            if seen_any:
                break
            in_root = True
            continue
        if not in_root:
            continue
        m = BIN_STAT_RE_RF.match(line)
        if m:
            out.setdefault(int(m.group(1)), []).append((int(m.group(2)), float(m.group(3))))
            seen_any = True
    return out


def run(dataset: str, seed: int) -> None:
    X, time, event = load_dataset(dataset)
    splits_df = pd.read_parquet(_SPLITS_DIR / f"{dataset}.parquet")
    row = splits_df[splits_df["seed"] == seed]
    train_idx = np.sort(row.loc[row["fold"] == "train", "sample_id"].to_numpy(np.int64))
    p = X.shape[1]
    print(f"[rank_flip ds={dataset} seed={seed}] n_tr={len(train_idx)} p={p}", flush=True)

    tmpdir = Path("/tmp/rank_flip_out")
    tmpdir.mkdir(exist_ok=True)
    cr_trace = tmpdir / "crforest.trace"
    rf_trace = tmpdir / "rfsrc.trace"
    cr_trace.unlink(missing_ok=True)
    rf_trace.unlink(missing_ok=True)

    os.environ["CRFOREST_TRACE"] = str(cr_trace)
    try:
        forest = CompetingRiskForest(
            n_estimators=1,
            min_samples_leaf=1,
            min_samples_split=2,
            max_features=None,
            bootstrap=False,
            random_state=seed,
            mode="default",
            time_grid=200,
            split_ntime=None,
            nsplit=0,
            rng_mode="rfsrc_aligned",
        ).fit(X[train_idx], time[train_idx], event[train_idx])
    finally:
        os.environ.pop("CRFOREST_TRACE", None)

    import rpy2.robjects as ro
    from rpy2.robjects.conversion import localconverter
    from rpy2.robjects.packages import importr

    importr("randomForestSRC", lib_loc="/tmp/rfsrc_patched_lib")
    converter = _rpy2_converter()
    feat_cols = [f"x{j}" for j in range(p)]
    train_df = pd.DataFrame(X[train_idx], columns=feat_cols)
    train_df["time"] = time[train_idx]
    train_df["event"] = event[train_idx].astype(np.int32)
    os.environ["RFSRC_TRACE"] = str(rf_trace)
    try:
        with localconverter(converter):
            ro.globalenv["train_df"] = train_df
        ro.r(f"""
            fit__ <- rfsrc(Surv(time, event) ~ ., data=train_df,
                ntree=1, nodesize=2, nodedepth=-1, mtry={p},
                splitrule="logrankCR", bootstrap="none",
                nsplit=0, ntime=0, seed=-{int(seed)})
            invisible(NULL)
        """)
    finally:
        os.environ.pop("RFSRC_TRACE", None)

    cr_root = _root_feat_stats(cr_trace, rfsrc_fmt=False)
    rf_root = _root_feat_stats(rf_trace, rfsrc_fmt=True)
    cr_winner = max(cr_root, key=lambda x: x[2])
    rf_winner = max(rf_root, key=lambda x: x[2])
    winner_feat_1b = rf_winner[0]
    winner_feat_0b = winner_feat_1b - 1

    print(
        f"  root winners: crforest feat={cr_winner[0]} z={cr_winner[2]:.6f}  "
        f"rfSRC feat={rf_winner[0]} z={rf_winner[2]:.6f}",
        flush=True,
    )

    # Per-feature z-stat comparison (context).
    cr_by_feat = {f: z for f, _b, z in cr_root}
    rf_by_feat = {f: z for f, _b, z in rf_root}
    feats = sorted(set(cr_by_feat) | set(rf_by_feat))
    print("  per-feature root stats (z-scale):", flush=True)
    print(f"    {'feat':>5} {'rf_z':>12} {'cr_z':>12} {'|dev|/rf':>10}", flush=True)
    for f in sorted(feats, key=lambda f: -rf_by_feat.get(f, 0.0)):
        rf_s, cr_s = rf_by_feat.get(f), cr_by_feat.get(f)
        dev = (
            abs(cr_s - rf_s) / max(rf_s, 1e-12)
            if cr_s is not None and rf_s is not None
            else float("nan")
        )
        print(f"    {f:>5} {rf_s or 0:12.6f} {cr_s or 0:12.6f} {dev:>10.4f}", flush=True)

    # --- Per-bin evidence for winner feature ---
    X_binned_full = apply_bins(X[train_idx], forest.bin_edges_)
    t_idx = np.clip(
        np.searchsorted(forest.time_grid_, time[train_idx], side="right") - 1,
        0,
        len(forest.time_grid_) - 1,
    ).astype(np.int64)
    cr_bins_z = _crforest_per_bin_stats(
        X_binned_full[:, winner_feat_0b].astype(np.int64),
        t_idx,
        event[train_idx].astype(np.int64),
        n_bins=256,
        n_causes=int(forest.n_causes_),
        n_time_bins=len(forest.time_grid_),
        min_samples_leaf=1,
    )
    cr_valid = [(b, s) for b, s in enumerate(cr_bins_z) if not math.isnan(s) and s > 0]
    cr_valid.sort(key=lambda x: -x[1])

    rf_bins_by_feat = _root_rf_bin_stats(rf_trace)
    rf_winner_bins = sorted(rf_bins_by_feat.get(winner_feat_1b, []), key=lambda x: -x[1])

    print(f"\n  PER-BIN EVIDENCE at root for winner feat={winner_feat_1b}:", flush=True)
    cr_top = cr_valid[0][1] if cr_valid else 0.0
    rf_top = rf_winner_bins[0][1] if rf_winner_bins else 0.0
    print(f"    crforest top-5 bins (candidates={len(cr_valid)}):", flush=True)
    for b, s in cr_valid[:5]:
        print(
            f"      bin={b:3d} z={s:.6f} rel_to_top={(cr_top - s) / max(cr_top, 1e-12):.6f}",
            flush=True,
        )
    print(f"    rfSRC    top-5 bins (candidates={len(rf_winner_bins)}):", flush=True)
    for j, s in rf_winner_bins[:5]:
        print(
            f"      obs_j={j:3d} z={s:.6f} rel_to_top={(rf_top - s) / max(rf_top, 1e-12):.6f}",
            flush=True,
        )
    for eps in (0.005, 0.01, 0.05):
        cr_near = sum(1 for _b, s in cr_valid if (cr_top - s) / max(cr_top, 1e-12) < eps)
        rf_near = sum(1 for _j, s in rf_winner_bins if (rf_top - s) / max(rf_top, 1e-12) < eps)
        print(
            f"    within {eps * 100:.1f}% of top: crforest {cr_near}/{len(cr_valid)}, "
            f"rfSRC {rf_near}/{len(rf_winner_bins)}",
            flush=True,
        )

    # --- Same-partition alignment ---
    # Map each crforest quantile bin to left-size = count(X_binned <= b).
    # rfSRC obs_j = left-size directly. Match by equal left-size so both
    # libs evaluate the IDENTICAL partition, isolating pure numerical noise.
    x_col = X_binned_full[:, winner_feat_0b].astype(np.int64)
    cr_bin_to_left = np.array([int((x_col <= b).sum()) for b in range(255)], dtype=np.int64)
    rf_by_left = {j: s for j, s in rf_winner_bins}
    aligned = [
        (b, int(cr_bin_to_left[b]), float(cr_bins_z[b]), float(rf_by_left[int(cr_bin_to_left[b])]))
        for b in range(255)
        if int(cr_bin_to_left[b]) in rf_by_left and not math.isnan(cr_bins_z[b])
    ]
    print(f"\n  SAME-PARTITION ALIGNMENT ({len(aligned)} bins matched by left-size):", flush=True)
    if not aligned:
        return
    devs = np.array([abs(c - r) / max(r, 1e-12) for _b, _ls, c, r in aligned])
    print(
        f"    cross-lib |dev|/rf on matched partitions: "
        f"max={devs.max():.6f} p95={np.percentile(devs, 95):.6f} "
        f"median={np.median(devs):.6f}",
        flush=True,
    )

    cr_pick_left = int(cr_bin_to_left[cr_winner[1]])
    rf_pick_left = int(rf_winner[1])
    print(
        f"    crforest chose: bin={cr_winner[1]}, left-size={cr_pick_left}, "
        f"stat_cr={cr_winner[2]:.6f}, "
        f"stat_rf(@same-partition)={rf_by_left.get(cr_pick_left, float('nan')):.6f}",
        flush=True,
    )
    cr_at_rf_pick = (
        cr_bins_z[np.searchsorted(cr_bin_to_left, rf_pick_left)]
        if rf_pick_left in cr_bin_to_left
        else float("nan")
    )
    print(
        f"    rfSRC    chose: obs_j={rf_winner[1]}, left-size={rf_pick_left}, "
        f"stat_rf={rf_winner[2]:.6f}, stat_cr(@same-partition)={cr_at_rf_pick:.6f}",
        flush=True,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", default="synthetic")
    parser.add_argument("--seed", type=int, default=1)
    args = parser.parse_args()
    run(args.dataset, args.seed)


if __name__ == "__main__":
    main()
