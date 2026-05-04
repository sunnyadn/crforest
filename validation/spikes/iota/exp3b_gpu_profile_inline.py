"""Iota.exp3b — inline-timing + cProfile + transfer accounting on single-tree GPU fit.

Goal: localize where the 2.8s/tree GPU wall comes from. We instrument
`build_flat_tree_gpu` with `time.perf_counter()` boundaries at every stage
inside the level loop, plus H2D/D2H counters wrapping `cp.asarray` and
`cp.asnumpy`. Then we run cProfile on a separate fit to catch any expensive
callees the inline scaffolding missed.

Spike — no commit, no test. Single tree, n=100k, RTX 5070 Ti.
"""

import cProfile
import io
import pstats
import time
from collections import defaultdict

import numpy as np

# ----- transfer counters -----
_H2D_N = 0
_H2D_BYTES = 0
_D2H_N = 0
_D2H_BYTES = 0


def _reset_transfer_counters():
    global _H2D_N, _H2D_BYTES, _D2H_N, _D2H_BYTES
    _H2D_N = 0
    _H2D_BYTES = 0
    _D2H_N = 0
    _D2H_BYTES = 0


def _make_profiled_builder():
    """Return a profiled twin of build_flat_tree_gpu plus the (stage_ms, count) dict.

    Mirrors the production function structure exactly, but records elapsed time
    at each stage boundary. GPU stages are wrapped with deviceSynchronize() so
    the ms numbers reflect actual kernel wall, not just launch dispatch.
    """
    import cupy as cp

    from comprisk._estimators import aalen_johansen_from_counts
    from comprisk._gpu_kernels import (
        _accumulate_leaf_cpu,
        _build_cand_mask,
        _partition_inplace,
        best_split_kernel_per_node,
        histogram_kernel_per_level,
    )
    from comprisk._tree_flat import FlatTree

    profile = defaultdict(float)  # stage -> total ms
    counts = defaultdict(int)

    def _record(stage, t0, sync=False):
        if sync:
            cp.cuda.runtime.deviceSynchronize()
        profile[stage] += (time.perf_counter() - t0) * 1000.0
        counts[stage] += 1

    def _h2d(arr):
        global _H2D_N, _H2D_BYTES
        _H2D_N += 1
        _H2D_BYTES += int(arr.nbytes) if hasattr(arr, "nbytes") else 0
        return cp.asarray(arr)

    def _d2h(arr):
        global _D2H_N, _D2H_BYTES
        _D2H_N += 1
        _D2H_BYTES += int(arr.nbytes) if hasattr(arr, "nbytes") else 0
        return cp.asnumpy(arr)

    def build_flat_tree_gpu_profiled(
        X_binned,
        t_idx_split,
        t_idx_full,
        event,
        *,
        bootstrap_indices,
        n_bins,
        n_causes,
        n_time_bins_split,
        n_time_bins_full,
        min_samples_split=30,
        min_samples_leaf=15,
        max_depth=-1,
        max_features=8,
        nsplit=10,
        splitrule_code=0,
        cause=1,
        seed=0,
    ):
        t_total_0 = time.perf_counter()

        rng = np.random.default_rng(seed)
        p = X_binned.shape[1]
        mtry = max_features

        t0 = time.perf_counter()
        td_d = _h2d(t_idx_split)
        ed_d = _h2d(event)
        _record("setup_h2d_static", t0, sync=True)

        sample_perm = bootstrap_indices.copy()
        n_bag = sample_perm.shape[0]

        N_max_nodes = max(64, 4 * n_bag // max(1, min_samples_leaf))
        out_features = np.zeros(N_max_nodes, dtype=np.int64)
        out_split_values = np.zeros(N_max_nodes, dtype=np.int64)
        out_left = np.zeros(N_max_nodes, dtype=np.int64)
        out_right = np.zeros(N_max_nodes, dtype=np.int64)
        out_is_leaf = np.zeros(N_max_nodes, dtype=np.bool_)
        out_leaf_idx_of_node = np.full(N_max_nodes, -1, dtype=np.int64)
        out_leaf_event_counts = np.zeros((N_max_nodes, n_causes, n_time_bins_full), dtype=np.uint32)
        out_leaf_at_risk = np.zeros((N_max_nodes, n_time_bins_full), dtype=np.uint32)

        n_nodes_used = 1
        n_leaves_used = 0

        active = [(0, 0, n_bag, 0)]
        n_levels = 0

        while active:
            n_levels += 1
            t0 = time.perf_counter()
            splittable, leafify = [], []
            for entry in active:
                node_idx, start, end, depth = entry
                n_node = end - start
                if n_node < min_samples_split or (max_depth >= 0 and depth >= max_depth):
                    leafify.append(entry)
                else:
                    splittable.append(entry)
            _record("splittable_decide", t0)

            t0 = time.perf_counter()
            for node_idx, start, end, _depth in leafify:
                _accumulate_leaf_cpu(
                    sample_perm[start:end],
                    t_idx_full,
                    event,
                    n_causes,
                    n_time_bins_full,
                    out_leaf_event_counts,
                    out_leaf_at_risk,
                    n_leaves_used,
                )
                out_features[node_idx] = -1
                out_is_leaf[node_idx] = True
                out_leaf_idx_of_node[node_idx] = n_leaves_used
                n_leaves_used += 1
            _record("leafify", t0)

            if not splittable:
                active = []
                continue

            t0 = time.perf_counter()
            N_active = len(splittable)
            feat_perm = np.empty((N_active, mtry), dtype=np.int32)
            for i in range(N_active):
                pool = np.arange(p, dtype=np.int32)
                rng.shuffle(pool)
                feat_perm[i] = pool[:mtry]
            _record("swor_mtry", t0)

            t0 = time.perf_counter()
            node_starts_h = np.empty(N_active, dtype=np.int32)
            node_ends_h = np.empty(N_active, dtype=np.int32)
            for i, (_, start, end, _) in enumerate(splittable):
                node_starts_h[i] = start
                node_ends_h[i] = end
            _record("node_bounds_pack", t0)

            t0 = time.perf_counter()
            cand_mask_h = _build_cand_mask(
                X_binned,
                sample_perm,
                node_starts_h,
                node_ends_h,
                feat_perm,
                n_bins,
                nsplit,
                rng,
            )
            _record("build_cand_mask", t0)

            t0 = time.perf_counter()
            sample_perm_d = _h2d(sample_perm.astype(np.int32))
            node_starts_d = _h2d(node_starts_h)
            node_ends_d = _h2d(node_ends_h)
            cand_mask_d = _h2d(cand_mask_h)
            ehist = cp.zeros((N_active, mtry, n_bins, n_causes, n_time_bins_split), dtype=cp.uint32)
            nat = cp.zeros((N_active, mtry, n_bins, n_time_bins_split), dtype=cp.uint32)
            _record("h2d_per_level", t0, sync=True)

            t0 = time.perf_counter()
            n_total = X_binned.shape[0]
            Xb_view_h = np.empty((n_total, mtry), dtype=np.uint8)
            for i, (_, start, end, _) in enumerate(splittable):
                samples = sample_perm[start:end]
                for f in range(mtry):
                    feat = feat_perm[i, f]
                    Xb_view_h[samples, f] = X_binned[samples, feat]
            _record("Xb_view_gather", t0)

            t0 = time.perf_counter()
            Xb_view_d = _h2d(Xb_view_h)
            _record("Xb_view_h2d", t0, sync=True)

            t0 = time.perf_counter()
            histogram_kernel_per_level(
                Xb_view_d,
                td_d,
                ed_d,
                sample_perm_d,
                node_starts_d,
                node_ends_d,
                ehist,
                nat,
                n_bins=n_bins,
                n_causes=n_causes,
                n_time_bins=n_time_bins_split,
                mtry=mtry,
            )
            _record("histogram_kernel", t0, sync=True)

            t0 = time.perf_counter()
            out_feat_d = cp.full((N_active,), -1, dtype=cp.int32)
            out_bin_d = cp.full((N_active,), -1, dtype=cp.int32)
            out_stat_d = cp.full((N_active,), -np.inf, dtype=cp.float64)
            best_split_kernel_per_node(
                ehist,
                nat,
                cand_mask_d,
                out_feat_d,
                out_bin_d,
                out_stat_d,
                n_bins=n_bins,
                n_causes=n_causes,
                n_time_bins=n_time_bins_split,
                mtry=mtry,
                min_samples_leaf=min_samples_leaf,
                splitrule_code=splitrule_code,
                cause=cause,
            )
            cp.cuda.runtime.deviceSynchronize()
            _record("best_split_kernel", t0, sync=False)

            t0 = time.perf_counter()
            out_feat_h = _d2h(out_feat_d)
            out_bin_h = _d2h(out_bin_d)
            _record("d2h_per_level", t0, sync=True)

            t0_apply = time.perf_counter()
            new_active = []
            t_part_total = 0.0
            for i, (node_idx, start, end, depth) in enumerate(splittable):
                f_sel = int(out_feat_h[i])
                bin_idx = int(out_bin_h[i])
                if f_sel < 0:
                    _accumulate_leaf_cpu(
                        sample_perm[start:end],
                        t_idx_full,
                        event,
                        n_causes,
                        n_time_bins_full,
                        out_leaf_event_counts,
                        out_leaf_at_risk,
                        n_leaves_used,
                    )
                    out_features[node_idx] = -1
                    out_is_leaf[node_idx] = True
                    out_leaf_idx_of_node[node_idx] = n_leaves_used
                    n_leaves_used += 1
                    continue
                actual_feat = int(feat_perm[i, f_sel])
                tp0 = time.perf_counter()
                mid = _partition_inplace(sample_perm, start, end, X_binned, actual_feat, bin_idx)
                t_part_total += time.perf_counter() - tp0
                if n_nodes_used + 2 > N_max_nodes:
                    _accumulate_leaf_cpu(
                        sample_perm[start:end],
                        t_idx_full,
                        event,
                        n_causes,
                        n_time_bins_full,
                        out_leaf_event_counts,
                        out_leaf_at_risk,
                        n_leaves_used,
                    )
                    out_features[node_idx] = -1
                    out_is_leaf[node_idx] = True
                    out_leaf_idx_of_node[node_idx] = n_leaves_used
                    n_leaves_used += 1
                    continue
                li = n_nodes_used
                ri = n_nodes_used + 1
                n_nodes_used += 2
                out_features[node_idx] = actual_feat
                out_split_values[node_idx] = bin_idx
                out_left[node_idx] = li
                out_right[node_idx] = ri
                out_is_leaf[node_idx] = False
                new_active.append((li, start, mid, depth + 1))
                new_active.append((ri, mid, end, depth + 1))
            profile["partition"] += t_part_total * 1000.0
            counts["partition"] += 1
            profile["apply_splits"] += (time.perf_counter() - t0_apply) * 1000.0
            counts["apply_splits"] += 1

            active = new_active

        t0 = time.perf_counter()
        leaf_table = np.zeros((n_leaves_used, n_causes, n_time_bins_full), dtype=np.float64)
        for k in range(n_leaves_used):
            leaf_table[k] = aalen_johansen_from_counts(
                out_leaf_event_counts[k],
                out_leaf_at_risk[k],
                n_causes,
            )

        tree = FlatTree.from_arrays(
            features=out_features[:n_nodes_used],
            split_values=out_split_values[:n_nodes_used],
            left_children=out_left[:n_nodes_used],
            right_children=out_right[:n_nodes_used],
            is_leaf_flags=out_is_leaf[:n_nodes_used],
            leaf_table=leaf_table,
            leaf_idx_of_node=out_leaf_idx_of_node[:n_nodes_used],
            leaf_event_counts=out_leaf_event_counts[:n_leaves_used].copy(),
            leaf_at_risk=out_leaf_at_risk[:n_leaves_used].copy(),
        )
        profile["finalize_leaf_table"] += (time.perf_counter() - t0) * 1000.0
        counts["finalize_leaf_table"] += 1

        profile["TOTAL"] += (time.perf_counter() - t_total_0) * 1000.0
        counts["TOTAL"] += 1
        profile["__n_levels__"] = float(n_levels)

        return tree

    return build_flat_tree_gpu_profiled, profile, counts


def _print_breakdown(profile, counts):
    total = profile.get("TOTAL", 0.0)
    n_levels = int(profile.get("__n_levels__", 0))
    print(f"TOTAL: {total:.1f} ms   ({n_levels} levels)", flush=True)

    # Order matches the lifecycle of one fit
    per_level_stages = [
        "splittable_decide",
        "leafify",
        "swor_mtry",
        "node_bounds_pack",
        "build_cand_mask",
        "h2d_per_level",
        "Xb_view_gather",
        "Xb_view_h2d",
        "histogram_kernel",
        "best_split_kernel",
        "d2h_per_level",
        "apply_splits",
    ]
    per_level_total = sum(profile.get(s, 0.0) for s in per_level_stages)

    print(
        f"  setup_h2d_static:    {profile.get('setup_h2d_static', 0.0):8.2f} ms  ({profile.get('setup_h2d_static', 0.0) / total * 100:5.1f}%)",
        flush=True,
    )
    print(
        f"  per_level_loop:      {per_level_total:8.2f} ms  ({per_level_total / total * 100:5.1f}%)",
        flush=True,
    )
    for s in per_level_stages:
        ms = profile.get(s, 0.0)
        n = counts.get(s, 0)
        pct = ms / total * 100 if total > 0 else 0.0
        print(f"    {s:<22} {ms:8.2f} ms  ({pct:5.1f}%)  [n={n}]", flush=True)
    # partition is nested inside apply_splits
    p_ms = profile.get("partition", 0.0)
    print(
        f"      partition (subset of apply_splits): {p_ms:.2f} ms  ({p_ms / total * 100:.1f}%)",
        flush=True,
    )
    f_ms = profile.get("finalize_leaf_table", 0.0)
    print(f"  finalize_leaf_table: {f_ms:8.2f} ms  ({f_ms / total * 100:5.1f}%)", flush=True)


def main():
    import cupy as cp

    rng = np.random.default_rng(0)
    n, p = 100_000, 8
    X = rng.uniform(size=(n, p))
    t = rng.exponential(1.0, n) + 0.1
    e = rng.integers(0, 3, n)

    from comprisk import CompetingRiskForest
    from comprisk import _gpu_kernels as _gk

    # Warm-up: compile kernels, prime memory pool
    print("[warm] compile kernels + prime memory pool ...", flush=True)
    CompetingRiskForest(n_estimators=1, device="cuda", random_state=99).fit(
        X[:5000], t[:5000], e[:5000]
    )
    cp.cuda.runtime.deviceSynchronize()

    # ---- INLINE-TIMING PROFILE ----
    print("\n=== INLINE-TIMING PROFILE (n=100k, single tree) ===", flush=True)
    builder, profile, counts = _make_profiled_builder()

    orig = _gk.build_flat_tree_gpu
    _gk.build_flat_tree_gpu = builder
    # Also rebind in forest.py's import-cached lookup is local-import → safe.
    try:
        _reset_transfer_counters()
        t0 = time.perf_counter()
        CompetingRiskForest(n_estimators=1, device="cuda", random_state=0).fit(X, t, e)
        cp.cuda.runtime.deviceSynchronize()
        wall_ms = (time.perf_counter() - t0) * 1000.0
        print(f"outer wall (incl. binning + forest scaffolding): {wall_ms:.1f} ms", flush=True)
        _print_breakdown(profile, counts)
        print(f"\nH2D transfers: {_H2D_N} totaling {_H2D_BYTES / 1e6:.1f} MB", flush=True)
        print(f"D2H transfers: {_D2H_N} totaling {_D2H_BYTES / 1e6:.1f} MB", flush=True)
    finally:
        _gk.build_flat_tree_gpu = orig

    # ---- cProfile ----
    print("\n=== cPROFILE TOP-20 BY cumtime (n=100k, single tree) ===", flush=True)
    pr = cProfile.Profile()
    pr.enable()
    CompetingRiskForest(n_estimators=1, device="cuda", random_state=1).fit(X, t, e)
    cp.cuda.runtime.deviceSynchronize()
    pr.disable()

    buf = io.StringIO()
    ps = pstats.Stats(pr, stream=buf).sort_stats("cumulative")
    ps.print_stats(20)
    print(buf.getvalue(), flush=True)


if __name__ == "__main__":
    main()
