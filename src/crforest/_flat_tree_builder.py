"""Flat-tree njit builder — productionized from validation/spikes/theta.

Builds a tree end-to-end inside one ``@njit(nogil=True)`` call:
  - explicit-stack iterative recursion (no Python recursion)
  - flat (n_nodes, ...) ndarray output (no ``HistTreeNode``)
  - reuses the existing ``find_best_split_hist_batched`` njit kernel
  - leaf event_counts + at_risk accumulated inline

Replaces the per-node Python orchestration of ``_hist_tree.build_tree_hist``
in default mode. The ``equivalence='rfsrc'`` code path is unchanged and
still uses ``_hist_tree``.
"""

from __future__ import annotations

import numpy as np
from numba import njit

from crforest._hist_splits import find_best_split_hist_batched
from crforest._tree_flat import FlatTree


@njit(cache=True, nogil=True)
def _build_flat_tree_njit(
    X_binned,  # (n, p) uint8
    t_idx_split,  # (n,) int32 — split-search grid (e.g., 50 bins)
    t_idx_full,  # (n,) int32 — leaf-accumulation grid (e.g., 200 bins)
    event,  # (n,) int32
    bootstrap_idx,  # (n_bag,) int32 — sample indices to use
    n_bins,
    n_causes,
    n_time_bins_split,
    n_time_bins_full,
    min_samples_split,
    min_samples_leaf,
    max_depth,  # -1 = unlimited
    max_features,  # mtry
    nsplit,
    splitrule_code,  # 0=logrankCR, 1=logrank
    cause,
    seed,
    # Pre-allocated outputs (caller sizes them)
    out_features,  # (N_max_nodes,) int64
    out_split_values,  # (N_max_nodes,) int64
    out_left_children,  # (N_max_nodes,) int64
    out_right_children,  # (N_max_nodes,) int64
    out_is_leaf_flags,  # (N_max_nodes,) bool
    out_leaf_idx_of_node,  # (N_max_nodes,) int64; -1 for internal
    out_leaf_event_counts,  # (N_max_leaves, n_causes, n_time_bins_full) uint32
    out_leaf_at_risk,  # (N_max_leaves, n_time_bins_full) uint32
):
    """Fill output arrays in place; return (n_nodes_used, n_leaves_used)."""
    np.random.seed(seed)
    n_bag = bootstrap_idx.shape[0]
    p = X_binned.shape[1]
    mtry = max_features

    N_max_nodes = out_features.shape[0]

    # Workspace: a permutation buffer over bootstrap_idx that we partition
    # in place. sample_perm[start:end] holds indices into X_binned for the
    # current node's samples.
    sample_perm = np.empty(n_bag, dtype=np.int32)
    for i in range(n_bag):
        sample_perm[i] = bootstrap_idx[i]

    # Per-node scratch: bin_sub holds the (n_node x mtry) view passed to
    # find_best_split_hist_batched.
    bin_sub = np.empty((n_bag, mtry), dtype=np.uint8)
    t_idx_split_node = np.empty(n_bag, dtype=np.int32)
    t_idx_full_node = np.empty(n_bag, dtype=np.int32)
    event_node = np.empty(n_bag, dtype=np.int32)
    cand_mask = np.empty((mtry, n_bins - 1), dtype=np.bool_)
    counts_b = np.empty(n_bins, dtype=np.int64)
    observed = np.empty(n_bins, dtype=np.int64)
    perm_pool = np.empty(p, dtype=np.int32)

    # Stack of (node_idx, start, end, depth) — a node range refers to
    # sample_perm[start:end].
    stack_node_idx = np.empty(N_max_nodes, dtype=np.int32)
    stack_start = np.empty(N_max_nodes, dtype=np.int32)
    stack_end = np.empty(N_max_nodes, dtype=np.int32)
    stack_depth = np.empty(N_max_nodes, dtype=np.int32)

    n_nodes_used = 1
    n_leaves_used = 0
    stack_node_idx[0] = 0
    stack_start[0] = 0
    stack_end[0] = n_bag
    stack_depth[0] = 0
    stack_size = 1

    # Initialize root output node (will be overwritten below).
    out_features[0] = 0
    out_split_values[0] = 0
    out_left_children[0] = 0
    out_right_children[0] = 0
    out_is_leaf_flags[0] = False
    out_leaf_idx_of_node[0] = -1

    while stack_size > 0:
        stack_size -= 1
        node_idx = stack_node_idx[stack_size]
        start = stack_start[stack_size]
        end = stack_end[stack_size]
        depth = stack_depth[stack_size]
        n_node = end - start

        too_small = n_node < min_samples_split
        too_deep = max_depth >= 0 and depth >= max_depth

        # Materialize per-node t_idx (split + full) and event arrays.
        for i in range(n_node):
            si = sample_perm[start + i]
            t_idx_split_node[i] = t_idx_split[si]
            t_idx_full_node[i] = t_idx_full[si]
            event_node[i] = event[si]

        if too_small or too_deep:
            _accumulate_leaf(
                t_idx_full_node,
                event_node,
                n_node,
                n_causes,
                n_time_bins_full,
                out_leaf_event_counts,
                out_leaf_at_risk,
                n_leaves_used,
            )
            out_features[node_idx] = -1
            out_split_values[node_idx] = 0
            out_left_children[node_idx] = 0
            out_right_children[node_idx] = 0
            out_is_leaf_flags[node_idx] = True
            out_leaf_idx_of_node[node_idx] = n_leaves_used
            n_leaves_used += 1
            continue

        # SWOR mtry features (Fisher-Yates partial)
        for j in range(p):
            perm_pool[j] = j
        for j in range(mtry):
            r = j + (np.random.randint(0, p - j) if (p - j) > 0 else 0)
            tmp = perm_pool[j]
            perm_pool[j] = perm_pool[r]
            perm_pool[r] = tmp

        # Build bin_sub[:n_node, :mtry] from X_binned via current sample_perm slice.
        for f in range(mtry):
            feat = perm_pool[f]
            for i in range(n_node):
                bin_sub[i, f] = X_binned[sample_perm[start + i], feat]

        # Build cand_mask: per-feature observed bins, SWOR draw of nsplit candidates.
        for f in range(mtry):
            counts_b[:] = 0
            for i in range(n_node):
                counts_b[bin_sub[i, f]] += 1
            n_obs = 0
            for b in range(n_bins):
                if counts_b[b] > 0:
                    observed[n_obs] = b
                    n_obs += 1
            for b in range(n_bins - 1):
                cand_mask[f, b] = False
            if n_obs < 2:
                continue
            n_valid = n_obs - 1  # exclude max bin
            k = nsplit if (nsplit > 0 and nsplit < n_valid) else n_valid
            for i in range(k):
                jj = i + (np.random.randint(0, n_valid - i) if (n_valid - i) > 0 else 0)
                tmp = observed[i]
                observed[i] = observed[jj]
                observed[jj] = tmp
                cand_mask[f, observed[i]] = True

        f_sel, bin_idx, _stat = find_best_split_hist_batched(
            bin_sub[:n_node],
            t_idx_split_node[:n_node],
            event_node[:n_node],
            n_bins,
            n_causes,
            n_time_bins_split,
            min_samples_leaf,
            splitrule_code,
            cause,
            cand_mask,
        )

        if f_sel < 0:
            _accumulate_leaf(
                t_idx_full_node,
                event_node,
                n_node,
                n_causes,
                n_time_bins_full,
                out_leaf_event_counts,
                out_leaf_at_risk,
                n_leaves_used,
            )
            out_features[node_idx] = -1
            out_split_values[node_idx] = 0
            out_left_children[node_idx] = 0
            out_right_children[node_idx] = 0
            out_is_leaf_flags[node_idx] = True
            out_leaf_idx_of_node[node_idx] = n_leaves_used
            n_leaves_used += 1
            continue

        actual_feature = perm_pool[f_sel]

        # In-place partition sample_perm[start:end] by X[*, actual_feature] <= bin_idx.
        i = start
        j = end - 1
        while i <= j:
            if X_binned[sample_perm[i], actual_feature] <= bin_idx:
                i += 1
            else:
                tmp = sample_perm[i]
                sample_perm[i] = sample_perm[j]
                sample_perm[j] = tmp
                j -= 1
        mid = i  # first right-side index

        # Stack-overflow guard: if we cannot allocate two children + push
        # them onto the stack, coerce this node into a leaf instead of
        # creating orphan children. This preserves the tree invariant
        # under pathological N_max_nodes sizing.
        if n_nodes_used + 2 > N_max_nodes or stack_size + 2 > N_max_nodes:
            _accumulate_leaf(
                t_idx_full_node,
                event_node,
                n_node,
                n_causes,
                n_time_bins_full,
                out_leaf_event_counts,
                out_leaf_at_risk,
                n_leaves_used,
            )
            out_features[node_idx] = -1
            out_split_values[node_idx] = 0
            out_left_children[node_idx] = 0
            out_right_children[node_idx] = 0
            out_is_leaf_flags[node_idx] = True
            out_leaf_idx_of_node[node_idx] = n_leaves_used
            n_leaves_used += 1
            continue

        left_idx = n_nodes_used
        right_idx = n_nodes_used + 1
        n_nodes_used += 2

        out_features[node_idx] = actual_feature
        out_split_values[node_idx] = bin_idx
        out_left_children[node_idx] = left_idx
        out_right_children[node_idx] = right_idx
        out_is_leaf_flags[node_idx] = False
        out_leaf_idx_of_node[node_idx] = -1

        # Push right then left so left processes first.
        stack_node_idx[stack_size] = right_idx
        stack_start[stack_size] = mid
        stack_end[stack_size] = end
        stack_depth[stack_size] = depth + 1
        stack_size += 1
        stack_node_idx[stack_size] = left_idx
        stack_start[stack_size] = start
        stack_end[stack_size] = mid
        stack_depth[stack_size] = depth + 1
        stack_size += 1

    return n_nodes_used, n_leaves_used


@njit(cache=True, nogil=True)
def _accumulate_leaf(
    t_idx_node,
    event_node,
    n_node,
    n_causes,
    n_time_bins,
    out_leaf_event_counts,
    out_leaf_at_risk,
    leaf_idx,
):
    """Tally event_counts[c, t] and at_risk[t] for one leaf."""
    for c in range(n_causes):
        for t in range(n_time_bins):
            out_leaf_event_counts[leaf_idx, c, t] = 0
    for t in range(n_time_bins):
        out_leaf_at_risk[leaf_idx, t] = 0
    for i in range(n_node):
        ti = t_idx_node[i]
        ev = event_node[i]
        if ev > 0 and ev <= n_causes:
            out_leaf_event_counts[leaf_idx, ev - 1, ti] += 1
        out_leaf_at_risk[leaf_idx, ti] += 1
    # Reverse cumsum for at_risk: at_risk[t] = sum(histogram[t..end])
    running = np.uint32(0)
    for t in range(n_time_bins - 1, -1, -1):
        running += out_leaf_at_risk[leaf_idx, t]
        out_leaf_at_risk[leaf_idx, t] = running


def build_flat_tree(
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
) -> FlatTree:
    """Python entry. Allocates output arrays, calls the njit kernel,
    materializes a ``FlatTree`` with leaf CIFs computed from the leaf
    event_counts and at_risk arrays.

    ``t_idx_split`` (coarse, n_time_bins_split) is used by the split-search
    kernel; ``t_idx_full`` (fine, n_time_bins_full) is used to accumulate
    leaf event_counts so that leaf CIFs align with ``forest.time_grid_``.
    """
    from crforest._estimators import aalen_johansen_from_counts_batched

    n_bag = bootstrap_indices.shape[0]
    N_max_nodes = max(64, 4 * n_bag // max(1, min_samples_leaf))

    out_features = np.zeros(N_max_nodes, dtype=np.int64)
    out_split_values = np.zeros(N_max_nodes, dtype=np.int64)
    out_left_children = np.zeros(N_max_nodes, dtype=np.int64)
    out_right_children = np.zeros(N_max_nodes, dtype=np.int64)
    out_is_leaf_flags = np.zeros(N_max_nodes, dtype=np.bool_)
    out_leaf_idx_of_node = np.full(N_max_nodes, -1, dtype=np.int64)
    out_leaf_event_counts = np.zeros((N_max_nodes, n_causes, n_time_bins_full), dtype=np.uint32)
    out_leaf_at_risk = np.zeros((N_max_nodes, n_time_bins_full), dtype=np.uint32)

    n_nodes, n_leaves = _build_flat_tree_njit(
        X_binned,
        np.ascontiguousarray(t_idx_split, dtype=np.int32),
        np.ascontiguousarray(t_idx_full, dtype=np.int32),
        event,
        np.ascontiguousarray(bootstrap_indices, dtype=np.int32),
        n_bins,
        n_causes,
        n_time_bins_split,
        n_time_bins_full,
        min_samples_split,
        min_samples_leaf,
        max_depth,
        max_features,
        nsplit,
        splitrule_code,
        cause,
        seed,
        out_features,
        out_split_values,
        out_left_children,
        out_right_children,
        out_is_leaf_flags,
        out_leaf_idx_of_node,
        out_leaf_event_counts,
        out_leaf_at_risk,
    )

    # Compute leaf-CIF table in one vectorized pass over the leaf axis.
    leaf_table = aalen_johansen_from_counts_batched(
        out_leaf_event_counts[:n_leaves],
        out_leaf_at_risk[:n_leaves],
        n_causes,
    )

    # Persist raw uint32 counts so predict_chf can lazily materialise the
    # Nelson-Aalen leaf table (see _hist_tree.predict_tree_hist_chf).
    return FlatTree.from_arrays(
        features=out_features[:n_nodes],
        split_values=out_split_values[:n_nodes],
        left_children=out_left_children[:n_nodes],
        right_children=out_right_children[:n_nodes],
        is_leaf_flags=out_is_leaf_flags[:n_nodes],
        leaf_table=leaf_table,
        leaf_idx_of_node=out_leaf_idx_of_node[:n_nodes],
        leaf_event_counts=out_leaf_event_counts[:n_leaves].copy(),
        leaf_at_risk=out_leaf_at_risk[:n_leaves].copy(),
    )
