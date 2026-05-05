"""TreeSHAP for cause-specific CIF on comprisk tree ensembles.

Implements Lundberg, Erion & Lee (2018), *Consistent Individualized Feature
Attribution for Tree Ensembles* (arXiv:1802.03888), Algorithm 1 (EXPVALUE).

The exact O(L·D²) Algorithm 2 will be added in a follow-up for production
scale.  Algorithm 1 is correct and passes the additivity invariant; its
complexity is O(2^D · L) per tree per sample, which is acceptable for
typical comprisk tree depths (D ~= 8-12 on realistic data due to
min_samples_leaf limiting leaf count).

Leaf values are generalised from scalars to ``(n_causes, n_times)`` tensors.
Because the SHAP algorithm is linear in the leaf value, the vectorisation over
cause x time is direct.
"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor

import numpy as np
from numba import njit
from sklearn.utils.validation import check_is_fitted

from comprisk._binning import apply_bins
from comprisk._shap_alg2 import shap_tree_single_alg2
from comprisk._tree_flat import FlatTree, flatten_tree

# ---------------------------------------------------------------------------
# Tree representation helpers (cover counts, flattening)
# ---------------------------------------------------------------------------


def _extract_leaf_counts_reftree(node) -> list[int]:
    """Recursively extract leaf sample counts from a RefTreeNode."""
    if node.is_leaf:
        if node.at_risk is not None and len(node.at_risk) > 0:
            return [round(node.at_risk[0])]
        return [0]
    counts: list[int] = []
    if node.left is not None:
        counts.extend(_extract_leaf_counts_reftree(node.left))
    if node.right is not None:
        counts.extend(_extract_leaf_counts_reftree(node.right))
    return counts


def _extract_leaf_counts_histtree(node) -> list[int]:
    """Recursively extract leaf sample counts from a HistTreeNode."""
    if node.is_leaf:
        if node.at_risk_sparse is not None and len(node.at_risk_sparse.values) > 0:
            return [int(node.at_risk_sparse.values[0])]
        return [0]
    counts: list[int] = []
    if node.left is not None:
        counts.extend(_extract_leaf_counts_histtree(node.left))
    if node.right is not None:
        counts.extend(_extract_leaf_counts_histtree(node.right))
    return counts


def _get_flat_and_leaf_counts(tree):
    """Return (FlatTree, leaf_counts) for any tree representation."""
    if isinstance(tree, FlatTree):
        flat = tree
        if flat.leaf_at_risk is not None:
            leaf_counts = flat.leaf_at_risk[:, 0].astype(np.int64)
        else:
            leaf_counts = None
    elif hasattr(tree, "threshold"):
        flat = flatten_tree(
            tree,
            get_split_value=lambda n: n.threshold,
            get_leaf_table=lambda n: _ensure_cif_refnode(n),
            split_dtype=np.float64,
        )
        leaf_counts = np.asarray(_extract_leaf_counts_reftree(tree), dtype=np.int64)
    elif hasattr(tree, "bin_idx"):
        from comprisk._hist_tree import _flatten_tree_hist

        flat = _flatten_tree_hist(tree)
        leaf_counts = np.asarray(_extract_leaf_counts_histtree(tree), dtype=np.int64)
    else:
        raise TypeError(f"Unknown tree type: {type(tree)}")
    return flat, leaf_counts


def _ensure_cif_refnode(node):
    """Lazy materialise CIF on a RefTreeNode (mirrors _tree.py logic)."""
    from comprisk._estimators import aalen_johansen_from_counts

    if node._cif is None:
        n_causes = node.event_counts.shape[0]
        node._cif = aalen_johansen_from_counts(node.event_counts, node.at_risk, n_causes)
    return node._cif


def _compute_node_covers(
    is_leaf_flags: np.ndarray,
    left_children: np.ndarray,
    right_children: np.ndarray,
    leaf_idx_of_node: np.ndarray,
    leaf_counts: np.ndarray,
) -> np.ndarray:
    """Bottom-up compute cover (training sample count) for each node."""
    n_nodes = len(is_leaf_flags)
    covers = np.zeros(n_nodes, dtype=np.int64)

    def visit(j: int) -> None:
        if is_leaf_flags[j]:
            li = leaf_idx_of_node[j]
            if li >= 0:
                covers[j] = leaf_counts[li]
        else:
            visit(left_children[j])
            visit(right_children[j])
            covers[j] = covers[left_children[j]] + covers[right_children[j]]

    visit(0)
    return covers


# ---------------------------------------------------------------------------
# Numba-accelerated EXPVALUE (Algorithm 1 core)
# ---------------------------------------------------------------------------


@njit(cache=True, nogil=True)
def _expvalue_njit(
    x,
    S_mask,
    features,
    split_values,
    left_children,
    right_children,
    is_leaf_flags,
    leaf_idx_of_node,
    leaf_table,
    covers,
    parent,
    results,
    node,
):
    """EXPVALUE — iterative with parent-propagation, numba-jitted.

    ``results`` is a pre-allocated ``(n_nodes, n_causes, n_times)`` scratch
    array that is overwritten in-place.  ``parent`` is a pre-computed
    ``(n_nodes,)`` parent-pointer array (-1 for root).
    """
    n_causes = leaf_table.shape[1]
    n_times = leaf_table.shape[2]

    max_stack = 128
    stack_n = np.empty(max_stack, dtype=np.int64)
    stack_state = np.empty(max_stack, dtype=np.int64)
    sp = 0

    stack_n[sp] = node
    stack_state[sp] = 0
    sp += 1

    while sp > 0:
        sp -= 1
        cur = stack_n[sp]
        state = stack_state[sp]

        if state == 1:
            lc = left_children[cur]
            rc = right_children[cur]
            cover_c = covers[cur]
            for c in range(n_causes):
                for t in range(n_times):
                    results[cur, c, t] = (
                        covers[lc] * results[lc, c, t] + covers[rc] * results[rc, c, t]
                    ) / cover_c
            # Propagate up through single-child chain
            p = parent[cur]
            while p >= 0:
                feat = features[p]
                if S_mask & (1 << feat):
                    child = left_children[p] if x[feat] <= split_values[p] else right_children[p]
                    for c in range(n_causes):
                        for t in range(n_times):
                            results[p, c, t] = results[child, c, t]
                    p = parent[p]
                else:
                    break
            continue

        if is_leaf_flags[cur]:
            li = leaf_idx_of_node[cur]
            for c in range(n_causes):
                for t in range(n_times):
                    results[cur, c, t] = leaf_table[li, c, t]
            # Propagate up through single-child chain
            p = parent[cur]
            while p >= 0:
                feat = features[p]
                if S_mask & (1 << feat):
                    child = left_children[p] if x[feat] <= split_values[p] else right_children[p]
                    for c in range(n_causes):
                        for t in range(n_times):
                            results[p, c, t] = results[child, c, t]
                    p = parent[p]
                else:
                    break
            continue

        feat = features[cur]
        if S_mask & (1 << feat):
            child = left_children[cur] if x[feat] <= split_values[cur] else right_children[cur]
            stack_n[sp] = child
            stack_state[sp] = 0
            sp += 1
        else:
            lc = left_children[cur]
            rc = right_children[cur]
            stack_n[sp] = cur
            stack_state[sp] = 1
            sp += 1
            stack_n[sp] = rc
            stack_state[sp] = 0
            sp += 1
            stack_n[sp] = lc
            stack_state[sp] = 0
            sp += 1

    return results[node]


@njit(cache=True, nogil=True)
def _shap_tree_njit(
    x,
    features,
    split_values,
    left_children,
    right_children,
    is_leaf_flags,
    leaf_idx_of_node,
    leaf_table,
    covers,
    n_features,
    phi,
):
    """Algorithm 1 TreeSHAP — numba-jitted subset enumeration + cached EXPVALUE."""
    # Find features on the decision path for x
    path_features = np.empty(features.shape[0], dtype=np.int64)
    path_len = 0
    node = 0
    while not is_leaf_flags[node]:
        feat = features[node]
        path_features[path_len] = feat
        path_len += 1
        node = left_children[node] if x[feat] <= split_values[node] else right_children[node]

    # Deduplicate
    unique_feats = np.empty(path_len, dtype=np.int64)
    n_unique = 0
    for i in range(path_len):
        feat = path_features[i]
        seen = False
        for j in range(n_unique):
            if unique_feats[j] == feat:
                seen = True
                break
        if not seen:
            unique_feats[n_unique] = feat
            n_unique += 1

    D = n_unique
    n_causes = leaf_table.shape[1]
    n_times = leaf_table.shape[2]

    if D == 0:
        return

    # Pre-compute parent pointers and scratch array (re-used across EXPVALUE calls)
    n_nodes = features.shape[0]
    parent = np.full(n_nodes, -1, dtype=np.int64)
    for i in range(n_nodes):
        if not is_leaf_flags[i]:
            parent[left_children[i]] = i
            parent[right_children[i]] = i
    results = np.empty((n_nodes, n_causes, n_times), dtype=np.float64)

    # Cache EXPVALUE for all subsets
    n_masks = 1 << D
    cache = np.empty((n_masks, n_causes, n_times), dtype=np.float64)
    for mask in range(n_masks):
        S_mask = 0
        for j in range(D):
            if mask & (1 << j):
                S_mask |= 1 << unique_feats[j]
        cache[mask] = _expvalue_njit(
            x,
            S_mask,
            features,
            split_values,
            left_children,
            right_children,
            is_leaf_flags,
            leaf_idx_of_node,
            leaf_table,
            covers,
            parent,
            results,
            0,
        )

    # Compute SHAP for each unique feature
    for i in range(D):
        feat = unique_feats[i]
        feat_mask = 1 << i
        # Build list of other features
        others = np.empty(D - 1, dtype=np.int64)
        n_others = 0
        for j in range(D):
            if j != i:
                others[n_others] = j
                n_others += 1

        n_subsets = 1 << n_others
        for mask in range(n_subsets):
            S_mask_idx = 0
            for j in range(n_others):
                if mask & (1 << j):
                    S_mask_idx |= 1 << others[j]
            s = 0
            _tmp = S_mask_idx
            while _tmp:
                s += _tmp & 1
                _tmp >>= 1
            # inline factorial (D <= ~15)
            f_s = 1.0
            for _fi in range(2, s + 1):
                f_s *= _fi
            f_dsm1 = 1.0
            for _fi in range(2, D - s):
                f_dsm1 *= _fi
            f_d = 1.0
            for _fi in range(2, D + 1):
                f_d *= _fi
            weight = f_s * f_dsm1 / f_d
            diff = cache[S_mask_idx | feat_mask] - cache[S_mask_idx]
            for c in range(n_causes):
                for t in range(n_times):
                    phi[feat, c, t] += weight * diff[c, t]


def _shap_tree_single(
    flat: FlatTree,
    x: np.ndarray,
    n_features: int,
    covers: np.ndarray,
) -> np.ndarray:
    """Fast TreeSHAP — dispatches to numba-jitted Algorithm 2 (O(L·D²))."""
    return shap_tree_single_alg2(flat, x, n_features, covers)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def shap_values(forest, X, times=None) -> tuple[np.ndarray, np.ndarray]:
    """Compute TreeSHAP values for cause-specific CIF.

    Parameters
    ----------
    forest : CompetingRiskForest
        A fitted competing-risk forest.
    X : array-like, shape (n_samples, n_features)
        Samples to explain.
    times : array-like of float or None, default=None
        Time points at which to evaluate SHAP.  If ``None``, uses
        the model's ``unique_times_`` grid.

    Returns
    -------
    shap_values : ndarray, shape (n_samples, n_features, n_times_out, n_causes)
        Cause-specific CIF SHAP attributions.
    base_value : ndarray, shape (n_times_out, n_causes)
        Expected CIF for the empty conditioning set.

        Additivity holds point-wise:

        .. math::

            sum_d shap_{s,d,t,c} + base_{t,c}
            approx predict_cif(X_s)_{c,t}
    """
    check_is_fitted(forest, "trees_")
    X = np.asarray(X, dtype=np.float64)
    if X.ndim != 2:
        raise ValueError(f"X must be 2-D; got ndim={X.ndim}")
    if X.shape[1] != forest.n_features_in_:
        raise ValueError(f"X has {X.shape[1]} features; expected {forest.n_features_in_}")

    n_samples, n_features = X.shape

    if times is None:
        times_out = forest.unique_times_
        time_projection = None
    else:
        times_out = np.asarray(times, dtype=np.float64)
        time_projection = _make_time_projection(forest.unique_times_, times_out)

    n_times_out = len(times_out)

    X_input = apply_bins(X, forest.bin_edges_) if forest.mode == "default" else X

    total_shap = np.zeros((n_samples, n_features, n_times_out, forest.n_causes_), dtype=np.float64)
    total_base = np.zeros((n_times_out, forest.n_causes_), dtype=np.float64)

    n_jobs = forest.n_jobs if hasattr(forest, "n_jobs") else 1
    if n_jobs == -1:
        import os

        n_jobs = os.cpu_count() or 1
    elif n_jobs is None:
        n_jobs = 1

    # Sample batching caps per-tree scratch memory.  A batch of 2000 samples
    # with 58 features, 200 time points, 2 causes costs ~370 MB.
    batch_size = 2000 if n_samples > 2000 else None

    # Tree-level thread parallelism: threads share memory (no fork/COW overhead)
    # and the numba kernels release the GIL, so multiple trees are processed
    # concurrently.  At n_jobs=1 fall through to avoid executor overhead.
    if n_jobs > 1:
        with ThreadPoolExecutor(max_workers=n_jobs) as executor:
            futures = [
                executor.submit(
                    _compute_tree_shap,
                    tree,
                    X_input,
                    n_features,
                    time_projection,
                    n_times_out,
                    batch_size,
                )
                for tree in forest.trees_
            ]
            for fut in futures:
                phi_tree, base = fut.result()
                total_shap += phi_tree
                total_base += base
    else:
        for tree in forest.trees_:
            phi_tree, base = _compute_tree_shap(
                tree,
                X_input,
                n_features,
                time_projection,
                n_times_out,
                batch_size,
            )
            total_shap += phi_tree
            total_base += base

    n_trees = len(forest.trees_)
    total_shap /= n_trees
    total_base /= n_trees

    return total_shap, total_base


def _shap_tree_samples(
    flat: FlatTree,
    X_batch: np.ndarray,
    n_features: int,
    covers: np.ndarray,
    time_projection,
    n_times_out: int,
) -> np.ndarray:
    """Compute SHAP for a batch of samples on a single (already-flattened) tree."""
    n_samples = len(X_batch)
    n_causes = flat.leaf_table.shape[1]
    phi_tree = np.zeros((n_samples, n_features, n_times_out, n_causes), dtype=np.float64)
    for si in range(n_samples):
        phi = _shap_tree_single(flat, X_batch[si], n_features, covers)
        if time_projection is not None:
            phi = time_projection(phi)
        phi_tree[si] = phi.transpose(0, 2, 1)
    return phi_tree


def _compute_tree_shap(
    tree, X_input, n_features, time_projection, n_times_out, batch_size: int | None = None
):
    """Compute SHAP for all samples on a single tree."""
    flat, leaf_counts = _get_flat_and_leaf_counts(tree)
    if leaf_counts is None:
        raise RuntimeError("Could not determine leaf sample counts for SHAP cover computation.")

    covers = _compute_node_covers(
        flat.is_leaf_flags,
        flat.left_children,
        flat.right_children,
        flat.leaf_idx_of_node,
        leaf_counts,
    )

    base = _base_value(flat, covers)
    if time_projection is not None:
        base = time_projection(base)

    n_samples = len(X_input)
    if batch_size is None or batch_size >= n_samples:
        phi_tree = _shap_tree_samples(
            flat, X_input, n_features, covers, time_projection, n_times_out
        )
    else:
        phi_tree = np.zeros(
            (n_samples, n_features, n_times_out, flat.leaf_table.shape[1]), dtype=np.float64
        )
        for start in range(0, n_samples, batch_size):
            end = min(start + batch_size, n_samples)
            phi_tree[start:end] = _shap_tree_samples(
                flat, X_input[start:end], n_features, covers, time_projection, n_times_out
            )

    return phi_tree, base.T


def _base_value(flat: FlatTree, covers: np.ndarray) -> np.ndarray:
    """Expected leaf value = sum (cover_leaf / cover_root) * leaf_value."""
    root_cover = covers[0]
    if root_cover == 0:
        return np.zeros(flat.leaf_table.shape[1:], dtype=np.float64)
    base = np.zeros(flat.leaf_table.shape[1:], dtype=np.float64)
    for j in range(len(flat.is_leaf_flags)):
        if flat.is_leaf_flags[j]:
            leaf_idx = flat.leaf_idx_of_node[j]
            base += (covers[j] / root_cover) * flat.leaf_table[leaf_idx]
    return base


def _make_time_projection(unique_times: np.ndarray, target_times: np.ndarray):
    """Project ``(..., n_times_full)`` onto ``target_times`` (right-continuous step)."""
    idx = np.searchsorted(unique_times, target_times, side="right") - 1
    take = np.clip(idx, 0, None)
    before = idx < 0
    before_any = bool(before.any())

    def _project(arr):
        out = arr[..., take]
        if before_any:
            out[..., before] = 0.0
        return out

    return _project
