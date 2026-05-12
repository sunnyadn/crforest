"""TreeSHAP for cause-specific CIF on comprisk tree ensembles.

Implements Lundberg, Erion & Lee (2018), *Consistent Individualized Feature
Attribution for Tree Ensembles* (arXiv:1802.03888), Algorithm 2 (O(L·D²)).
Leaf values are generalised from scalars to ``(n_causes, n_times)`` tensors;
SHAP is linear in the leaf value, so the recursion only needs to produce the
*structural* weights — one scalar per ``(leaf, path-feature)`` — and the
``(n_causes, n_times)`` leaf tensors are multiplied back in with a single
BLAS matmul per tree:

    phi[s] = W[s] @ leaf_table.reshape(n_leaves, n_causes * n_times)

This keeps the ``n_causes * n_times`` factor out of the hot recursion.
"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor

import numpy as np
from sklearn.utils.validation import check_is_fitted

from comprisk._binning import apply_bins
from comprisk._shap_alg2 import shap_tree_weights
from comprisk._tree_flat import FlatTree, flatten_tree

# Per-batch budget for the transient ``(batch, n_features, n_leaves)`` weight
# array (and the same-order matmul output).  Sample batching trades a few extra
# BLAS calls per tree for a bounded scratch footprint under thread parallelism.
_W_BATCH_BYTES = 128 * 1024 * 1024


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
    n_causes = forest.n_causes_

    X_input = apply_bins(X, forest.bin_edges_) if forest.mode == "default" else X

    n_jobs = forest.n_jobs if hasattr(forest, "n_jobs") else 1
    if n_jobs == -1:
        import os

        n_jobs = os.cpu_count() or 1
    elif n_jobs is None:
        n_jobs = 1

    out_shape = (n_samples, n_features, n_times_out, n_causes)
    base_shape = (n_times_out, n_causes)
    trees = list(forest.trees_)
    n_trees = len(trees)

    total_shap = np.zeros(out_shape, dtype=np.float64)
    total_base = np.zeros(base_shape, dtype=np.float64)

    # Tree-level thread parallelism: threads share memory (no fork/COW overhead)
    # and the numba weight kernels release the GIL.  Each worker owns a private
    # accumulator over its slice of trees, so the only cross-tree reduction is
    # over ``n_jobs`` arrays at the end (not over ``n_trees``).  The first tree
    # is always done on the calling thread so the numba kernels are compiled
    # before any worker touches them (concurrent first-compile of a recursive
    # jitted function can crash the interpreter).
    _accumulate_tree_shap(
        trees[0], X_input, n_features, time_projection, n_times_out, total_shap, total_base
    )
    rest = trees[1:]
    if rest and n_jobs > 1:
        n_workers = min(n_jobs, len(rest))
        chunks = [rest[w::n_workers] for w in range(n_workers)]
        with ThreadPoolExecutor(max_workers=n_workers) as executor:
            futures = [
                executor.submit(
                    _chunk_shap,
                    chunk,
                    X_input,
                    n_features,
                    time_projection,
                    n_times_out,
                    out_shape,
                    base_shape,
                )
                for chunk in chunks
            ]
            for fut in futures:
                chunk_shap, chunk_base = fut.result()
                total_shap += chunk_shap
                total_base += chunk_base
    elif rest:
        for tree in rest:
            _accumulate_tree_shap(
                tree, X_input, n_features, time_projection, n_times_out, total_shap, total_base
            )

    total_shap /= n_trees
    total_base /= n_trees
    return total_shap, total_base


def _chunk_shap(
    trees,
    X_input: np.ndarray,
    n_features: int,
    time_projection,
    n_times_out: int,
    out_shape: tuple,
    base_shape: tuple,
) -> tuple[np.ndarray, np.ndarray]:
    """Sum SHAP / base over a subset of trees into a fresh accumulator pair."""
    acc_shap = np.zeros(out_shape, dtype=np.float64)
    acc_base = np.zeros(base_shape, dtype=np.float64)
    for tree in trees:
        _accumulate_tree_shap(
            tree, X_input, n_features, time_projection, n_times_out, acc_shap, acc_base
        )
    return acc_shap, acc_base


def _accumulate_tree_shap(
    tree,
    X_input: np.ndarray,
    n_features: int,
    time_projection,
    n_times_out: int,
    acc_shap: np.ndarray,
    acc_base: np.ndarray,
) -> None:
    """Add one tree's SHAP / base contribution into ``acc_shap`` / ``acc_base``.

    The structural TreeSHAP recursion fills a ``(batch, n_features, n_leaves)``
    weight tensor; the ``(n_causes, n_times)`` leaf values are folded in by a
    single ``W @ leaf_table_2d`` matmul.  When ``times`` is a small subset, the
    leaf table is projected onto those columns *first*, so the matmul's right
    operand is ``n_causes * len(times)`` wide rather than the full grid.
    """
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

    leaf_table = flat.leaf_table  # (n_leaves, n_causes, n_times_full)
    n_leaves, n_causes = leaf_table.shape[0], leaf_table.shape[1]

    base = _base_value(flat, covers)  # (n_causes, n_times_full)
    if time_projection is not None:
        leaf_table = time_projection(leaf_table)
        base = time_projection(base)
    acc_base += base.T

    leaf_table_2d = np.ascontiguousarray(leaf_table.reshape(n_leaves, n_causes * n_times_out))

    n_samples = len(X_input)
    batch_size = max(1, min(n_samples, _W_BATCH_BYTES // max(1, n_features * n_leaves * 8)))
    for start in range(0, n_samples, batch_size):
        end = min(start + batch_size, n_samples)
        b = end - start
        weights = shap_tree_weights(
            flat.features,
            flat.split_values,
            flat.left_children,
            flat.right_children,
            flat.is_leaf_flags,
            flat.leaf_idx_of_node,
            covers,
            X_input[start:end],
            n_features,
            n_leaves,
        )  # (b, n_features, n_leaves)
        phi = (weights.reshape(b * n_features, n_leaves) @ leaf_table_2d).reshape(
            b, n_features, n_causes, n_times_out
        )
        acc_shap[start:end] += phi.transpose(0, 1, 3, 2)


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
