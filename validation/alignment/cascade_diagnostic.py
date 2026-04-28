"""Cascade diagnostic: walk both trees in lockstep from root.

Directly observes (not infers) how tree divergence compounds from
the root downward. Fits a single deterministic tree in both libs
(ntree=1, bootstrap=F, nsplit=0, mtry=p, split_ntime=None, rfSRC
ntime=0, min_samples_split=2, nodesize=2, rng_mode=rfsrc_aligned),
extracts per-node (depth, size, winner_feat, winner_left_size)
from the node_start + feat_stat_CR traces, and walks both
sequences in DFS lockstep.

At each step:
  - "in_sync": same depth and same size → the two libs are looking
    at the same population at the same tree position.
  - partition agreement: compare crforest's chosen left-size with
    rfSRC's chosen left-size. If same, both libs send the same
    samples left; no downstream divergence is introduced here.
  - first real divergence: first in-sync node where the winner
    feature or the chosen left-size differs.

The output is per-seed: total in-sync nodes, first-divergence depth,
and the classification of the divergence (grid-mismatch, argmax-flip,
feature-flip).

Requires instrumented rfSRC at /tmp/rfsrc_patched_lib (node_start
with depth + nodeID, bin_stat_CR + feat_stat_CR in logrankCR).
Rebuild via ``bash validation/alignment/_rfsrc_patches/regen.sh``.

Run:
    uv run --extra maintainer python -m validation.alignment.cascade_diagnostic \\
        --dataset synthetic --seeds 10
"""

from __future__ import annotations

import argparse
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

NODE_RE_CR = re.compile(r"^node_start n=(\d+) depth=(\d+)")
NODE_RE_RF = re.compile(r"^node_start tree=(\d+) a=(\d+) b=(\d+) c=([-\d.eE+]+)")
FEAT_RE_CR = re.compile(r"^feat_stat_CR covariate=(\d+) bin=(\S+) stat=([-\d.eE+]+)")
FEAT_RE_RF = re.compile(r"^feat_stat_CR tree=\d+ a=(\d+) b=(\d+) c=([-\d.eE+]+)")


def _parse_nodes(path: Path, rfsrc_fmt: bool) -> list[dict]:
    """Return one dict per node in DFS order with keys:
    depth, size, stats (list of (feat, bin, z)), winner_feat, winner_bin, winner_z.
    """
    nodes: list[dict] = []
    cur: dict | None = None
    for line in path.read_text().splitlines():
        if rfsrc_fmt:
            m = NODE_RE_RF.match(line)
            if m:
                if cur is not None:
                    nodes.append(cur)
                cur = {
                    "size": int(m.group(2)),
                    "depth": int(m.group(3)),
                    "stats": [],
                }
                continue
            m = FEAT_RE_RF.match(line)
            if m and cur is not None:
                cur["stats"].append((int(m.group(1)), int(m.group(2)), float(m.group(3))))
        else:
            m = NODE_RE_CR.match(line)
            if m:
                if cur is not None:
                    nodes.append(cur)
                cur = {
                    "size": int(m.group(1)),
                    "depth": int(m.group(2)),
                    "stats": [],
                }
                continue
            m = FEAT_RE_CR.match(line)
            if m and cur is not None:
                # crforest stat is num^2/var; convert to z-scale for rf comparison.
                z_sq = float(m.group(3))
                z = float(np.sqrt(max(z_sq, 0.0)))
                cur["stats"].append((int(m.group(1)), int(m.group(2)), z))
    if cur is not None:
        nodes.append(cur)
    for n in nodes:
        if n["stats"]:
            best = max(n["stats"], key=lambda x: x[2])
            n["winner_feat"] = best[0]
            n["winner_bin"] = best[1]
            n["winner_z"] = best[2]
        else:
            n["winner_feat"] = -1
            n["winner_bin"] = -1
            n["winner_z"] = 0.0
    return nodes


def _infer_left_sizes(nodes: list[dict]) -> None:
    """In DFS order, the node immediately following an internal node is its
    left child. Set ``left_size`` on each internal node (leaves: -1)."""
    n = len(nodes)
    for node in nodes:
        node["left_size"] = -1
    for i in range(n - 1):
        # Node i is internal iff its stats list is non-empty (split was found).
        if nodes[i]["stats"] and nodes[i + 1]["depth"] == nodes[i]["depth"] + 1:
            nodes[i]["left_size"] = nodes[i + 1]["size"]


def run_one(dataset: str, seed: int) -> dict:
    X, time, event = load_dataset(dataset)
    splits_df = pd.read_parquet(_SPLITS_DIR / f"{dataset}.parquet")
    row = splits_df[splits_df["seed"] == seed]
    train_idx = np.sort(row.loc[row["fold"] == "train", "sample_id"].to_numpy(np.int64))
    p = X.shape[1]

    tmpdir = Path("/tmp/cascade_out")
    tmpdir.mkdir(exist_ok=True)
    cr_trace = tmpdir / f"cr_s{seed}.trace"
    rf_trace = tmpdir / f"rf_s{seed}.trace"
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
            fit__ <- rfsrc(Surv(time, event) ~ ., data = train_df,
                ntree=1, nodesize=2, nodedepth=-1, mtry={p},
                splitrule="logrankCR", bootstrap="none",
                nsplit=0, ntime=0, seed=-{int(seed)})
            invisible(NULL)
        """)
    finally:
        os.environ.pop("RFSRC_TRACE", None)

    cr_nodes = _parse_nodes(cr_trace, rfsrc_fmt=False)
    rf_nodes = _parse_nodes(rf_trace, rfsrc_fmt=True)
    _infer_left_sizes(cr_nodes)
    _infer_left_sizes(rf_nodes)

    # Compute per-feature unique left-sizes in crforest's 256-quantile grid
    # for the root. This is how we classify "grid_mismatch" at the root.
    X_binned = apply_bins(X[train_idx], forest.bin_edges_).astype(np.int64)
    cr_left_sizes_per_feat: dict[int, set[int]] = {}
    for f in range(p):
        col = X_binned[:, f]
        sizes: set[int] = set()
        for b in range(255):
            sizes.add(int((col <= b).sum()))
        cr_left_sizes_per_feat[f] = sizes

    # Walk in lockstep. A pair is "in sync" if (depth, size) match -- both
    # libs are looking at the same population at the same DFS position.
    # As soon as a winner choice diverges, classify and stop; subsequent
    # nodes have different populations and the comparison is meaningless.
    in_sync = 0
    agree_partition = 0
    first_div: dict | None = None
    for i, (cr_n, rf_n) in enumerate(zip(cr_nodes, rf_nodes, strict=False)):
        if cr_n["depth"] != rf_n["depth"] or cr_n["size"] != rf_n["size"]:
            break
        in_sync += 1
        if cr_n["left_size"] < 0 or rf_n["left_size"] < 0:
            continue  # leaf -- no winner choice to compare
        if cr_n["winner_feat"] != rf_n["winner_feat"]:
            kind = "feature_flip"
        elif cr_n["left_size"] != rf_n["left_size"]:
            grid = cr_left_sizes_per_feat.get(rf_n["winner_feat"] - 1, set())
            kind = "argmax_flip" if rf_n["left_size"] in grid else "grid_mismatch"
        else:
            agree_partition += 1
            continue
        first_div = {
            "idx": i,
            "depth": cr_n["depth"],
            "size": cr_n["size"],
            "kind": kind,
            "cr_feat": cr_n["winner_feat"],
            "rf_feat": rf_n["winner_feat"],
            "cr_left": cr_n["left_size"],
            "rf_left": rf_n["left_size"],
        }
        break

    return {
        "seed": seed,
        "cr_total": len(cr_nodes),
        "rf_total": len(rf_nodes),
        "in_sync": in_sync,
        "agree_partition": agree_partition,
        "first_div": first_div,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", default="synthetic")
    parser.add_argument("--seeds", type=int, default=10)
    args = parser.parse_args()

    print(f"[cascade ds={args.dataset} seeds={args.seeds}]", flush=True)
    print(
        f"{'seed':>4} {'cr_tot':>6} {'rf_tot':>6} {'in_sync':>8} "
        f"{'agree_part':>10} {'div_dep':>7} {'div_size':>8} {'kind':>15} "
        f"{'cr_f/rf_f':>10} {'cr_L/rf_L':>12}",
        flush=True,
    )
    summary: list[dict] = []
    for s in range(1, args.seeds + 1):
        r = run_one(args.dataset, s)
        summary.append(r)
        fd = r["first_div"]
        if fd is None:
            print(
                f"{s:>4} {r['cr_total']:>6} {r['rf_total']:>6} {r['in_sync']:>8} "
                f"{r['agree_partition']:>10} {'--':>7} {'--':>8} {'(no div)':>15} "
                f"{'--':>10} {'--':>12}",
                flush=True,
            )
        else:
            print(
                f"{s:>4} {r['cr_total']:>6} {r['rf_total']:>6} {r['in_sync']:>8} "
                f"{r['agree_partition']:>10} {fd['depth']:>7} {fd['size']:>8} "
                f"{fd['kind']:>15} "
                f"{fd['cr_feat']}/{fd['rf_feat']:<8} "
                f"{fd['cr_left']}/{fd['rf_left']:<10}",
                flush=True,
            )

    print("\n[summary]", flush=True)
    kinds: dict[str, int] = {}
    depths: list[int] = []
    for r in summary:
        fd = r["first_div"]
        if fd is None:
            kinds["(no divergence)"] = kinds.get("(no divergence)", 0) + 1
        else:
            kinds[fd["kind"]] = kinds.get(fd["kind"], 0) + 1
            depths.append(fd["depth"])
    for k, v in sorted(kinds.items()):
        print(f"  {k}: {v}", flush=True)
    if depths:
        print(
            f"  first-divergence depth: min={min(depths)} median={int(np.median(depths))} "
            f"max={max(depths)}",
            flush=True,
        )


if __name__ == "__main__":
    main()
