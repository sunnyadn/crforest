"""θ spike — single-tree njit POC.

Builds an entire CR-forest tree end-to-end in ONE njit call:
  - explicit-stack recursion (numba-compatible; no Python recursion)
  - flat tree representation: int32 nodes array + leaf-event/at-risk arrays
  - calls existing ``find_best_split_hist_batched`` njit kernel per node
  - cand-mask building inlined (Fisher-Yates SWOR, exp6-style)
  - bootstrap omitted (use all samples) — perf measurement only

Output flat schema (matches sklearn-tree convention loosely):

    tree_nodes : (N_nodes_used, 4) int32
        col 0 = feature index, or -1 if leaf
        col 1 = bin_idx threshold
        col 2 = left child node_idx, or leaf_idx if leaf
        col 3 = right child node_idx, or 0 if leaf

    leaf_event_counts : (N_leaves_used, n_causes, n_time_bins) uint32
    leaf_at_risk      : (N_leaves_used, n_time_bins) uint32

This is a MEASUREMENT POC: it does not preserve the rfSRC RNG stream
or the production ``HistTreeNode`` representation. Comparing wall time
to current main's per-tree (3.0s on n=100k) decides whether the C-path
kernel rewrite is viable.
"""

from __future__ import annotations

import numpy as np
from numba import njit

from crforest._hist_splits import find_best_split_hist_batched


@njit(cache=True, nogil=True)
def _build_tree_njit(
    X_binned,  # (n, p) uint8
    t_idx,  # (n,) int32
    event,  # (n,) int32
    n_bins,
    n_causes,
    n_time_bins,
    min_samples_split,
    min_samples_leaf,
    max_depth,  # -1 = unlimited
    mtry,
    nsplit,
    splitrule_code,  # 0=logrankCR, 1=logrank
    cause,
    seed,
    # Pre-allocated outputs (caller sizes them)
    tree_nodes,  # (N_max_nodes, 4) int32
    leaf_event_counts,  # (N_max_leaves, n_causes, n_time_bins) uint32
    leaf_at_risk,  # (N_max_leaves, n_time_bins) uint32
):
    """Build one CR tree in one njit call. Returns (n_nodes, n_leaves)."""
    np.random.seed(seed)
    n_total = X_binned.shape[0]
    p = X_binned.shape[1]

    N_max_nodes = tree_nodes.shape[0]
    N_max_leaves = leaf_event_counts.shape[0]

    # Workspace
    sample_perm = np.empty(n_total, dtype=np.int32)
    for i in range(n_total):
        sample_perm[i] = i

    bin_sub = np.empty((n_total, mtry), dtype=np.uint8)
    t_idx_node = np.empty(n_total, dtype=np.int32)
    event_node = np.empty(n_total, dtype=np.int32)
    cand_mask = np.empty((mtry, n_bins - 1), dtype=np.bool_)
    counts_b = np.empty(n_bins, dtype=np.int64)
    observed = np.empty(n_bins, dtype=np.int64)
    perm_pool = np.empty(p, dtype=np.int32)

    # Stack: (node_idx, start, end, depth)
    stack_node_idx = np.empty(N_max_nodes, dtype=np.int32)
    stack_start = np.empty(N_max_nodes, dtype=np.int32)
    stack_end = np.empty(N_max_nodes, dtype=np.int32)
    stack_depth = np.empty(N_max_nodes, dtype=np.int32)

    # Push root
    n_nodes_used = 1
    stack_node_idx[0] = 0
    stack_start[0] = 0
    stack_end[0] = n_total
    stack_depth[0] = 0
    stack_size = 1
    n_leaves_used = 0

    while stack_size > 0:
        stack_size -= 1
        node_idx = stack_node_idx[stack_size]
        start = stack_start[stack_size]
        end = stack_end[stack_size]
        depth = stack_depth[stack_size]
        n_node = end - start

        too_small = n_node < min_samples_split
        too_deep = max_depth >= 0 and depth >= max_depth

        # Build per-node arrays once (used by leaf path AND split path)
        for i in range(n_node):
            t_idx_node[i] = t_idx[sample_perm[start + i]]
            event_node[i] = event[sample_perm[start + i]]

        if too_small or too_deep:
            # Make leaf
            _make_leaf_inline(
                t_idx_node,
                event_node,
                n_node,
                n_causes,
                n_time_bins,
                leaf_event_counts,
                leaf_at_risk,
                n_leaves_used,
            )
            tree_nodes[node_idx, 0] = -1
            tree_nodes[node_idx, 1] = 0
            tree_nodes[node_idx, 2] = n_leaves_used
            tree_nodes[node_idx, 3] = 0
            n_leaves_used += 1
            continue

        # SWOR sample mtry features (Fisher-Yates partial)
        for j in range(p):
            perm_pool[j] = j
        for j in range(mtry):
            r = j + np.random.randint(0, p - j) if (p - j) > 0 else j
            tmp = perm_pool[j]
            perm_pool[j] = perm_pool[r]
            perm_pool[r] = tmp
        # selected = perm_pool[:mtry]

        # Build bin_sub (n_node × mtry)
        for f in range(mtry):
            feat = perm_pool[f]
            for i in range(n_node):
                bin_sub[i, f] = X_binned[sample_perm[start + i], feat]

        # Build cand_mask (njit version of exp6 logic)
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
            # Fisher-Yates partial SWOR over observed[:n_valid]
            for i in range(k):
                jj = i + (np.random.randint(0, n_valid - i) if (n_valid - i) > 0 else 0)
                tmp = observed[i]
                observed[i] = observed[jj]
                observed[jj] = tmp
                cand_mask[f, observed[i]] = True

        # Call existing njit kernel for split scan
        f_sel, bin_idx, _stat = find_best_split_hist_batched(
            bin_sub[:n_node],
            t_idx_node[:n_node],
            event_node[:n_node],
            n_bins,
            n_causes,
            n_time_bins,
            min_samples_leaf,
            splitrule_code,
            cause,
            cand_mask,
        )

        if f_sel < 0:
            # No valid split — make leaf
            _make_leaf_inline(
                t_idx_node,
                event_node,
                n_node,
                n_causes,
                n_time_bins,
                leaf_event_counts,
                leaf_at_risk,
                n_leaves_used,
            )
            tree_nodes[node_idx, 0] = -1
            tree_nodes[node_idx, 1] = 0
            tree_nodes[node_idx, 2] = n_leaves_used
            tree_nodes[node_idx, 3] = 0
            n_leaves_used += 1
            continue

        actual_feature = perm_pool[f_sel]

        # In-place partition sample_perm[start:end] by X[*, actual_feature] <= bin_idx
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
        # i is the first right-side index; left = [start, i), right = [i, end)
        mid = i

        left_idx = n_nodes_used
        right_idx = n_nodes_used + 1
        n_nodes_used += 2

        tree_nodes[node_idx, 0] = actual_feature
        tree_nodes[node_idx, 1] = bin_idx
        tree_nodes[node_idx, 2] = left_idx
        tree_nodes[node_idx, 3] = right_idx

        # Push right then left so left is processed first (matches recursion order)
        if stack_size + 2 > N_max_nodes:
            # Stack overflow — should not happen if N_max_nodes is correct
            break
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
def _make_leaf_inline(
    t_idx_node,
    event_node,
    n_node,
    n_causes,
    n_time_bins,
    leaf_event_counts,
    leaf_at_risk,
    leaf_idx,
):
    # event_counts[c, t]: number of events of cause c at time t
    # at_risk[t]: number of samples with t_idx >= t
    # Reset target slots
    for c in range(n_causes):
        for t in range(n_time_bins):
            leaf_event_counts[leaf_idx, c, t] = 0
    for t in range(n_time_bins):
        leaf_at_risk[leaf_idx, t] = 0
    # Tally
    for i in range(n_node):
        ti = t_idx_node[i]
        ev = event_node[i]
        if ev > 0 and ev <= n_causes:
            leaf_event_counts[leaf_idx, ev - 1, ti] += 1
        # at_risk: increment all t <= ti (samples observed up to ti are at-risk at all earlier t)
        # That's equivalent to: at_risk[t] = sum(1 for ti' >= t in node)
        # We compute it via cumulative reverse-sum after tallying t_idx histogram.
    # Build at_risk via histogram of t_idx then reverse cumsum
    for t in range(n_time_bins):
        leaf_at_risk[leaf_idx, t] = 0
    for i in range(n_node):
        ti = t_idx_node[i]
        leaf_at_risk[leaf_idx, ti] += 1
    # Reverse cumulative sum: at_risk[t] = sum of histogram[t..end]
    running = 0
    for t in range(n_time_bins - 1, -1, -1):
        running += leaf_at_risk[leaf_idx, t]
        leaf_at_risk[leaf_idx, t] = running


def build_tree_njit(
    X_binned,
    t_idx,
    event,
    *,
    n_bins,
    n_causes,
    n_time_bins,
    min_samples_split=30,
    min_samples_leaf=15,
    max_depth=-1,
    mtry=8,
    nsplit=10,
    splitrule_code=0,
    cause=1,
    seed=0,
):
    """Python entry. Allocates output arrays and calls the njit kernel."""
    n = X_binned.shape[0]
    # Conservative cap: 2 * n / min_samples_leaf, plus headroom
    N_max_nodes = max(64, 4 * n // max(1, min_samples_leaf))
    N_max_leaves = N_max_nodes  # ≤ N_max_nodes always

    tree_nodes = np.zeros((N_max_nodes, 4), dtype=np.int32)
    leaf_event_counts = np.zeros((N_max_leaves, n_causes, n_time_bins), dtype=np.uint32)
    leaf_at_risk = np.zeros((N_max_leaves, n_time_bins), dtype=np.uint32)

    n_nodes, n_leaves = _build_tree_njit(
        X_binned,
        t_idx,
        event,
        n_bins,
        n_causes,
        n_time_bins,
        min_samples_split,
        min_samples_leaf,
        max_depth,
        mtry,
        nsplit,
        splitrule_code,
        cause,
        seed,
        tree_nodes,
        leaf_event_counts,
        leaf_at_risk,
    )
    return tree_nodes[:n_nodes], leaf_event_counts[:n_leaves], leaf_at_risk[:n_leaves]
