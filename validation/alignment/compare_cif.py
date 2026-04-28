"""P2.6 step 1 diagnostic: per-sample CIF comparison on follic seed 0.

Fits crforest (default mode) and randomForestSRC on the same training fold,
then compares:

1. Training-side time grids: crforest's ``time_grid_`` vs rfSRC's
   ``fit$time.interest``. Mismatch at the last index is a time-grid (H1)
   signal.
2. Per-test-sample cause-1 CIF curves on a common evaluation grid
   (``min(last_time_cr, last_time_rf)``). Shape of the pointwise gap:
       * near-constant offset      -> aggregation-convention (H3)
       * zero until last time      -> final-index time-grid mismatch (H1)
       * linearly growing gap      -> AJ formula difference (H2)
3. Per-sample risk = CIF[test, cause=1, final_time]. This is the value that
   feeds the C-index; the 20-seed median delta_c lives here.

The script runs against one seed. It is a diagnostic, not a test; its
output is human-readable evidence to localize the P2.6 residual.

Maintainer-only (rpy2). Run from repo root:

    uv run --extra maintainer python -m validation.alignment.compare_cif
"""

from __future__ import annotations

import argparse

import numpy as np

from crforest import CompetingRiskForest, concordance_index_cr
from validation.alignment import _rpy2_available
from validation.datasets import load as load_dataset


def _fit_rfsrc(
    X_train: np.ndarray,
    time_train: np.ndarray,
    event_train: np.ndarray,
    X_test: np.ndarray,
    seed: int,
    n_estimators: int = 500,
    nodesize: int = 15,
    nsplit: int | None = None,
) -> dict:
    """Fit rfSRC on training fold, predict on test fold, return CIF + grid."""
    import rpy2.robjects as ro
    from rpy2.robjects.conversion import localconverter
    from rpy2.robjects.packages import importr

    from validation.alignment import _rpy2_converter

    importr("randomForestSRC")
    converter = _rpy2_converter()

    import pandas as pd

    p = X_train.shape[1]
    feat_cols = [f"x{j}" for j in range(p)]
    train_df = pd.DataFrame(X_train, columns=feat_cols)
    train_df["time"] = time_train
    train_df["event"] = event_train.astype(np.int32)
    test_df = pd.DataFrame(X_test, columns=feat_cols)

    with localconverter(converter):
        ro.globalenv["train_df"] = train_df
        ro.globalenv["test_df"] = test_df

    nsplit_arg = f", nsplit = {nsplit}" if nsplit is not None else ""
    ro.r(
        f"""
        fit__ <- rfsrc(
            Surv(time, event) ~ .,
            data       = train_df,
            ntree      = {n_estimators},
            nodesize   = {nodesize},
            mtry       = ceiling(sqrt({p})),
            splitrule  = "logrankCR",
            samptype   = "swr"{nsplit_arg},
            seed       = -{int(seed)}
        )
        pred__ <- predict(fit__, newdata = test_df)
        cat(sprintf("[rfSRC] nsplit resolved to %d\\n", fit__$nsplit))
        """
    )

    with localconverter(converter):
        time_interest = np.asarray(ro.r("fit__$time.interest"), dtype=np.float64)
        # pred$cif shape: (n_test, n_time, n_cause)
        cif_flat = np.asarray(ro.r("pred__$cif"), dtype=np.float64)

    n_test = X_test.shape[0]
    n_time = len(time_interest)
    if cif_flat.size != n_test * n_time * 2:
        raise RuntimeError(
            f"rfSRC pred$cif size {cif_flat.size} != {n_test}*{n_time}*2 (n_causes assumed 2)"
        )
    cif_rf = cif_flat.reshape(n_test, n_time, 2)

    # NOTE: intentionally keep fit__, pred__ in R's global env so downstream
    # diagnostics (tree-structure, per-tree aggregation) can reuse them.

    return {
        "time_grid": time_interest,
        "cif": cif_rf,  # (n_test, n_time, n_cause)
    }


def _fit_crforest(
    X_train: np.ndarray,
    time_train: np.ndarray,
    event_train: np.ndarray,
    X_test: np.ndarray,
    seed: int,
    n_estimators: int = 500,
    mode: str = "default",
    time_grid: int = 200,
    min_samples_leaf: int = 1,
    min_samples_split: int = 30,
    nsplit: int | None = None,
) -> dict:
    forest = CompetingRiskForest(
        n_estimators=n_estimators,
        min_samples_leaf=min_samples_leaf,
        min_samples_split=min_samples_split,
        max_features="sqrt",
        bootstrap=True,
        random_state=seed,
        mode=mode,
        time_grid=time_grid,
        nsplit=nsplit,
    ).fit(X_train, time_train, event_train)
    cif = forest.predict_cif(X_test)  # (n_test, n_cause, n_time)
    # Transpose to (n_test, n_time, n_cause) to match rfSRC layout.
    cif = np.transpose(cif, (0, 2, 1))
    grid = forest.time_grid_ if mode == "default" else forest.unique_times_
    return {
        "time_grid": np.asarray(grid, dtype=np.float64),
        "cif": cif,
        "forest": forest,
    }


def _summarize_grid(name: str, grid: np.ndarray) -> None:
    print(
        f"  [{name}] len={len(grid)} t0={grid[0]:.6f} t_last={grid[-1]:.6f} "
        f"mean_gap={np.diff(grid).mean():.4f}"
    )


def _pointwise_cif_gap(
    cif_cr: np.ndarray,
    grid_cr: np.ndarray,
    cif_rf: np.ndarray,
    grid_rf: np.ndarray,
    cause: int,
) -> dict:
    """Interpolate both CIFs onto a shared grid (intersection) and report gap.

    Both CIF arrays are step functions (constant between event times in each
    grid). We use searchsorted to evaluate each on the shared grid.
    """
    # Shared grid = union of both, restricted to [max(t0), min(t_last)].
    t_lo = max(grid_cr[0], grid_rf[0])
    t_hi = min(grid_cr[-1], grid_rf[-1])
    shared = np.unique(np.concatenate([grid_cr, grid_rf]))
    shared = shared[(shared >= t_lo) & (shared <= t_hi)]

    def _eval(cif: np.ndarray, grid: np.ndarray, ts: np.ndarray) -> np.ndarray:
        # step-function: cif(t) = cif[idx] where idx = searchsorted(grid, t, "right") - 1
        idx = np.clip(np.searchsorted(grid, ts, side="right") - 1, 0, len(grid) - 1)
        return cif[:, idx, cause - 1]  # (n_test, len(ts))

    f_cr = _eval(cif_cr, grid_cr, shared)
    f_rf = _eval(cif_rf, grid_rf, shared)
    gap = f_cr - f_rf
    return {
        "shared_grid": shared,
        "gap": gap,
        "mean_per_t": gap.mean(axis=0),  # (len(shared),)
        "median_per_t": np.median(gap, axis=0),
        "p95_per_t": np.percentile(np.abs(gap), 95, axis=0),
    }


def _cr_tree_stats(tree) -> tuple[int, int, list[int]]:
    """Walk a crforest tree, return (n_leaves, max_depth, _)."""
    n_leaves = 0
    max_depth = 0

    def walk(node, depth):
        nonlocal n_leaves, max_depth
        if node.is_leaf:
            n_leaves += 1
            max_depth = max(max_depth, depth)
            return
        walk(node.left, depth + 1)
        walk(node.right, depth + 1)

    walk(tree, 0)
    return n_leaves, max_depth, []


def _cr_leaf_sizes(forest, X_train: np.ndarray) -> list[int]:
    """Return leaf-sample-size distribution across all trees of a crforest."""
    from crforest._binning import apply_bins
    from crforest._tree import _flatten_tree
    from crforest._tree_flat import predict_leaf_indices

    if forest.mode == "default":
        X_input = apply_bins(X_train, forest.bin_edges_)

        def flatten(tree):
            from crforest._hist_tree import _flatten_tree_hist

            return _flatten_tree_hist(tree)
    else:
        X_input = X_train

        def flatten(tree):
            return _flatten_tree(tree)

    sizes: list[int] = []
    for tree in forest.trees_:
        leaf_idx = predict_leaf_indices(flatten(tree), X_input)
        _, counts = np.unique(leaf_idx, return_counts=True)
        sizes.extend(counts.tolist())
    return sizes


def compare_tree_structure(
    forest_cr,
    X_train: np.ndarray,
) -> None:
    """Compare crforest and rfSRC tree-structure distributions on the same data.

    Assumes rfSRC fit object is stored in R as ``fit__`` (left by ``_fit_rfsrc``).
    """
    import rpy2.robjects as ro
    from rpy2.robjects.conversion import localconverter

    from validation.alignment import _rpy2_converter

    converter = _rpy2_converter()

    # crforest side
    cr_leaves: list[int] = []
    cr_depths: list[int] = []
    for tree in forest_cr.trees_:
        nl, md, _ = _cr_tree_stats(tree)
        cr_leaves.append(nl)
        cr_depths.append(md)
    cr_leaf_sizes = _cr_leaf_sizes(forest_cr, X_train)

    # rfSRC side — requires fit__ still present; re-fit via passed flag upstream.
    with localconverter(converter):
        na_df = ro.r("as.data.frame(fit__$forest$nativeArray)")
    import pandas as pd

    na_df = pd.DataFrame(na_df)
    # Leaf rows have parmID==0.
    leaves = na_df[na_df["parmID"] == 0]
    rf_leaves = leaves.groupby("treeID").size().to_numpy()
    rf_leaf_sizes = leaves["nodeSZ"].to_numpy()
    # Depth per tree: use nodeID, which rfSRC numbers 1..n_nodes preorder.
    # Depth cannot be derived from nodeID alone without walking, so skip for now.

    def _stats(arr, name):
        a = np.asarray(arr, dtype=np.float64)
        return (
            f"  {name}: n={len(a)} mean={a.mean():.1f} median={np.median(a):.1f} "
            f"std={a.std():.1f} p5={np.percentile(a, 5):.0f} p95={np.percentile(a, 95):.0f} "
            f"min={a.min():.0f} max={a.max():.0f}"
        )

    print("\n=== Tree structure (n_leaves per tree) ===")
    print(_stats(cr_leaves, "crforest"))
    print(_stats(rf_leaves, "rfSRC   "))

    print("\n=== Tree depth per tree (crforest only — rfSRC nativeArray lacks direct depth) ===")
    print(_stats(cr_depths, "crforest"))

    print("\n=== Leaf sample sizes ===")
    print(_stats(cr_leaf_sizes, "crforest"))
    print(_stats(rf_leaf_sizes, "rfSRC   "))


def verify_rfsrc_aggregation(X_test: np.ndarray, k_probe: int = 20) -> None:
    """Verify rfSRC's ensemble CIF == plain mean of per-tree CIFs.

    Uses ``predict.rfsrc`` with ``get.tree`` to fetch per-tree predictions, then
    manually averages and compares against ``predict.rfsrc`` over the same subset.
    Assumes rfSRC fit is at ``fit__`` in R.
    """
    import pandas as pd
    import rpy2.robjects as ro
    from rpy2.robjects.conversion import localconverter

    from validation.alignment import _rpy2_converter

    converter = _rpy2_converter()

    p = X_test.shape[1]
    feat_cols = [f"x{j}" for j in range(p)]
    probe_df = pd.DataFrame(X_test, columns=feat_cols)
    with localconverter(converter):
        ro.globalenv["probe_df"] = probe_df

    # One R call: ensemble over 1..k_probe, plus a flat array of all per-tree CIFs.
    ro.r(
        f"""
        pred_sub__ <- predict(fit__, newdata = probe_df, get.tree = 1:{k_probe})
        per_tree__ <- simplify2array(lapply(
            1:{k_probe},
            function(i) predict(fit__, newdata = probe_df, get.tree = i)$cif
        ))
        """
    )
    with localconverter(converter):
        pred_sub = np.asarray(ro.r("pred_sub__$cif"), dtype=np.float64)
        per_tree_flat = np.asarray(ro.r("per_tree__"), dtype=np.float64)

    n_test = X_test.shape[0]
    n_time = pred_sub.size // (n_test * 2)
    pred_sub = pred_sub.reshape(n_test, n_time, 2)
    # simplify2array stacks on the last axis: (n_test, n_time, 2, k_probe)
    per_tree = np.moveaxis(per_tree_flat.reshape(n_test, n_time, 2, k_probe), -1, 0)

    manual_mean = per_tree.mean(axis=0)
    diff = pred_sub - manual_mean
    print(f"\n=== rfSRC aggregation check (first {k_probe} trees) ===")
    print(f"  max|pred_sub - manual_mean|: {np.max(np.abs(diff)):.6e}")
    print(
        f"  cause=1 at t_last: pred_sub={pred_sub[:, -1, 0].mean():.6f} "
        f"manual_mean={manual_mean[:, -1, 0].mean():.6f}"
    )

    with localconverter(converter):
        ro.r("rm(pred_sub__); rm(per_tree__); rm(probe_df)")


def compare_root_aj(
    X_train: np.ndarray,
    time_train: np.ndarray,
    event_train: np.ndarray,
    n_causes: int = 2,
) -> None:
    """Compute root-level AJ on full training fold in both libs, compare term-by-term.

    This isolates the AJ formula (and at-risk / survival convention) from
    the forest's ensemble aggregation. A single rfSRC tree with nodedepth=0
    has one leaf = the full training set, so pred$cif on training *is* the
    root-level AJ estimate.
    """
    from crforest._estimators import aalen_johansen

    # crforest side: AJ on the full training fold.
    unique_times = np.sort(np.unique(time_train))
    cif_cr = aalen_johansen(time_train, event_train, unique_times, n_causes=n_causes)

    # rfSRC side: nodedepth=0 depth-0 single-leaf tree via rpy2.
    import pandas as pd
    import rpy2.robjects as ro
    from rpy2.robjects.conversion import localconverter
    from rpy2.robjects.packages import importr

    from validation.alignment import _rpy2_converter

    importr("randomForestSRC")
    converter = _rpy2_converter()

    p = X_train.shape[1]
    feat_cols = [f"x{j}" for j in range(p)]
    df = pd.DataFrame(X_train, columns=feat_cols)
    df["time"] = time_train
    df["event"] = event_train.astype(np.int32)

    with localconverter(converter):
        ro.globalenv["df"] = df

    ro.r(
        """
        fit_root__ <- rfsrc(
            Surv(time, event) ~ .,
            data       = df,
            ntree      = 1,
            bootstrap  = "none",
            nodedepth  = 0,
            splitrule  = "logrankCR",
            seed       = -1
        )
        pred_root__ <- predict(fit_root__, newdata = df[1, , drop=FALSE])
        """
    )

    with localconverter(converter):
        time_interest = np.asarray(ro.r("fit_root__$time.interest"), dtype=np.float64)
        cif_flat = np.asarray(ro.r("pred_root__$cif"), dtype=np.float64)
    cif_rf = cif_flat.reshape(1, len(time_interest), n_causes)[0]  # (n_time, n_cause)

    with localconverter(converter):
        ro.r("rm(fit_root__); rm(pred_root__); rm(df)")

    print("\n=== Root-level AJ comparison (no splitting, full training fold) ===")
    print(
        f"  crforest grid: len={len(unique_times)} t_last={unique_times[-1]:.4f}  "
        f"(includes censor times)"
    )
    print(
        f"  rfSRC grid:    len={len(time_interest)} t_last={time_interest[-1]:.4f}  "
        f"(event times only)"
    )

    # Evaluate both on rfSRC's grid (which is a subset of crforest's in general).
    # cif_cr shape: (n_causes, n_unique_times); we want cause-1 at t in time_interest.
    def eval_cr(ts):
        idx = np.clip(np.searchsorted(unique_times, ts, side="right") - 1, 0, len(unique_times) - 1)
        return cif_cr[:, idx]  # (n_causes, len(ts))

    cr_at_rf_grid = eval_cr(time_interest)  # (n_causes, len(time_interest))
    rf_T = cif_rf.T  # (n_cause, n_time)

    for cause in range(n_causes):
        diff = cr_at_rf_grid[cause] - rf_T[cause]
        print(
            f"  cause {cause + 1}: "
            f"cr_at_t_last={cr_at_rf_grid[cause, -1]:.6f}  "
            f"rf_at_t_last={rf_T[cause, -1]:.6f}  "
            f"delta_at_t_last={diff[-1]:+.6e}  "
            f"max|diff|={np.max(np.abs(diff)):.6e}"
        )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="P2.6 diagnostic: compare crforest and randomForestSRC CIFs on a matched fold."
    )
    parser.add_argument("--dataset", default="follic")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--cause", type=int, default=1)
    parser.add_argument("--n-estimators", type=int, default=500)
    parser.add_argument("--nodesize", type=int, default=15)
    parser.add_argument("--mode", default="default", choices=["default", "reference"])
    parser.add_argument(
        "--time-grid",
        type=int,
        default=200,
        help="crforest time_grid max points (ignored in reference mode)",
    )
    parser.add_argument(
        "--root-aj",
        action="store_true",
        help="Also compute root-level AJ on full training fold in both libs",
    )
    parser.add_argument(
        "--tree-structure",
        action="store_true",
        help="Also compare tree-structure distributions (n_leaves, depth, leaf sizes)",
    )
    parser.add_argument(
        "--aggregation-check",
        action="store_true",
        help="Also verify rfSRC ensemble CIF == plain mean of per-tree CIFs (first 20 trees)",
    )
    parser.add_argument("--min-samples-leaf", type=int, default=1)
    parser.add_argument("--min-samples-split", type=int, default=30)
    parser.add_argument(
        "--rfsrc-nsplit",
        type=int,
        default=None,
        help="rfSRC nsplit (None=library default). 0 = exhaustive.",
    )
    args = parser.parse_args()

    if not _rpy2_available():
        raise SystemExit("rpy2 not available; install with `uv sync --extra maintainer`")

    X, time, event = load_dataset(args.dataset)
    import pandas as pd

    from validation.splits import _SPLITS_DIR

    splits_df = pd.read_parquet(_SPLITS_DIR / f"{args.dataset}.parquet")
    row = splits_df[(splits_df["seed"] == args.seed)]
    if row.empty:
        available = sorted(splits_df["seed"].unique().tolist())
        raise SystemExit(f"seed {args.seed} not in splits; available: {available[:10]}...")
    train_idx = np.sort(row.loc[row["fold"] == "train", "sample_id"].to_numpy(np.int64))
    test_idx = np.sort(row.loc[row["fold"] == "test", "sample_id"].to_numpy(np.int64))

    print(f"[{args.dataset} seed={args.seed}] n_train={len(train_idx)} n_test={len(test_idx)}")

    cr = _fit_crforest(
        X[train_idx],
        time[train_idx],
        event[train_idx],
        X[test_idx],
        seed=args.seed,
        n_estimators=args.n_estimators,
        mode=args.mode,
        time_grid=args.time_grid,
        min_samples_leaf=args.min_samples_leaf,
        min_samples_split=args.min_samples_split,
    )
    print(
        f"[crforest] mode={args.mode} time_grid={args.time_grid} "
        f"min_samples_leaf={args.min_samples_leaf} "
        f"min_samples_split={args.min_samples_split}"
    )
    rf = _fit_rfsrc(
        X[train_idx],
        time[train_idx],
        event[train_idx],
        X[test_idx],
        seed=args.seed,
        n_estimators=args.n_estimators,
        nodesize=args.nodesize,
        nsplit=args.rfsrc_nsplit,
    )

    print("\n=== Time grids (training fold) ===")
    _summarize_grid("crforest time_grid_", cr["time_grid"])
    _summarize_grid("rfSRC time.interest", rf["time_grid"])
    n_unique_event_train = len(np.unique(time[train_idx][event[train_idx] > 0]))
    print(f"  unique event times in train fold: {n_unique_event_train}")
    last_match = np.isclose(cr["time_grid"][-1], rf["time_grid"][-1])
    print(f"  last-time match (crforest vs rfSRC): {last_match}")
    len_match = len(cr["time_grid"]) == len(rf["time_grid"])
    print(f"  grid length match: {len_match}")
    if len_match:
        max_gap = np.max(np.abs(cr["time_grid"] - rf["time_grid"]))
        print(f"  max pointwise grid diff: {max_gap:.6e}")

    print("\n=== Per-sample risk (cause=1, final time of each grid) ===")
    risk_cr = cr["cif"][:, -1, args.cause - 1]  # crforest last time
    risk_rf = rf["cif"][:, -1, args.cause - 1]  # rfSRC last time
    diff = risk_cr - risk_rf
    print(f"  risk_cr:  mean={risk_cr.mean():.4f} std={risk_cr.std():.4f}")
    print(f"  risk_rf:  mean={risk_rf.mean():.4f} std={risk_rf.std():.4f}")
    print(
        f"  delta:    mean={diff.mean():+.4f} median={np.median(diff):+.4f} "
        f"std={diff.std():.4f} p5={np.percentile(diff, 5):+.4f} "
        f"p95={np.percentile(diff, 95):+.4f}"
    )

    c_cr = concordance_index_cr(event[test_idx], time[test_idx], risk_cr, cause=args.cause)
    c_rf = concordance_index_cr(event[test_idx], time[test_idx], risk_rf, cause=args.cause)
    print(f"\n=== C-index (cause={args.cause}) ===")
    print(f"  crforest: {c_cr:.4f}")
    print(f"  rfSRC:    {c_rf:.4f}")
    print(f"  delta_c:  {c_cr - c_rf:+.4f}")

    print("\n=== Pointwise CIF gap on shared grid ===")
    pw = _pointwise_cif_gap(cr["cif"], cr["time_grid"], rf["cif"], rf["time_grid"], args.cause)
    shared = pw["shared_grid"]
    mean_t = pw["mean_per_t"]
    print(f"  shared grid: len={len(shared)} [{shared[0]:.3f}, {shared[-1]:.3f}]")
    # Print gap at 5 evenly-spaced probe points:
    probes = np.linspace(0, len(shared) - 1, 5).astype(int)
    print("  probe points (mean delta_CIF across test samples):")
    for i in probes:
        print(
            f"    t={shared[i]:.3f}  mean_gap={mean_t[i]:+.5f}  "
            f"median_gap={pw['median_per_t'][i]:+.5f}  p95|gap|={pw['p95_per_t'][i]:.5f}"
        )
    print(f"  overall mean |gap|: {np.abs(pw['gap']).mean():.5f}")
    print(f"  gap at t_max:       mean={mean_t[-1]:+.5f}  median={pw['median_per_t'][-1]:+.5f}")
    print(f"  gap at t_min:       mean={mean_t[0]:+.5f}   median={pw['median_per_t'][0]:+.5f}")

    if args.root_aj:
        compare_root_aj(X[train_idx], time[train_idx], event[train_idx], n_causes=2)

    if args.tree_structure:
        compare_tree_structure(cr["forest"], X[train_idx])

    if args.aggregation_check:
        verify_rfsrc_aggregation(X[test_idx], k_probe=20)


if __name__ == "__main__":
    main()
