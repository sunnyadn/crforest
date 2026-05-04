"""Flattened tree representation and nogil descent kernel.

Both reference-mode and histogram-mode trees flatten into the same
parallel-array layout (``FlatTree``) for vectorized prediction. The only
per-mode differences are (a) the split-value dtype (``float64`` threshold
vs ``int64`` bin index) and (b) how a leaf's leaf quantity (CIF or CHF)
is obtained from its node — these are injected via callables into the
flattening pass (``flatten_tree``). Descent is one shared numba kernel
(``_descend_flat_nogil``) that numba specializes by dtype; the final
leaf-table gather is a plain NumPy fancy-index op.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
from numba import njit


@dataclass
class FlatTree:
    features: np.ndarray  # (n_nodes,) int64
    split_values: np.ndarray  # (n_nodes,) — threshold or bin index
    left_children: np.ndarray  # (n_nodes,) int64
    right_children: np.ndarray  # (n_nodes,) int64
    is_leaf_flags: np.ndarray  # (n_nodes,) bool
    leaf_table: np.ndarray  # (n_leaves, n_causes, n_time_bins) float64
    leaf_idx_of_node: np.ndarray  # (n_nodes,) int64; -1 for internal nodes
    # Optional raw counts persisted by the default-mode njit builder so that
    # secondary leaf quantities (CHF, etc.) can be materialised lazily without
    # rebuilding the tree. None on legacy flatten paths that source from
    # HistTreeNode (which retains its own counts and uses _flat_chf caching).
    leaf_event_counts: np.ndarray | None = None  # (n_leaves, n_causes, n_time_bins) uint32
    leaf_at_risk: np.ndarray | None = None  # (n_leaves, n_time_bins) uint32

    @classmethod
    def from_arrays(
        cls,
        *,
        features: np.ndarray,
        split_values: np.ndarray,
        left_children: np.ndarray,
        right_children: np.ndarray,
        is_leaf_flags: np.ndarray,
        leaf_table: np.ndarray,
        leaf_idx_of_node: np.ndarray,
        leaf_event_counts: np.ndarray | None = None,
        leaf_at_risk: np.ndarray | None = None,
    ) -> FlatTree:
        """Construct a FlatTree from already-flat arrays.

        Used by the njit flat-tree builder. The existing ``flatten_tree``
        path constructs FlatTree internally without going through here.
        """
        n_nodes = features.shape[0]
        for name, arr in (
            ("split_values", split_values),
            ("left_children", left_children),
            ("right_children", right_children),
            ("is_leaf_flags", is_leaf_flags),
            ("leaf_idx_of_node", leaf_idx_of_node),
        ):
            if arr.shape[0] != n_nodes:
                raise ValueError(
                    f"{name} length {arr.shape[0]} does not match features length {n_nodes}"
                )
        return cls(
            features=features,
            split_values=split_values,
            left_children=left_children,
            right_children=right_children,
            is_leaf_flags=is_leaf_flags,
            leaf_table=leaf_table,
            leaf_idx_of_node=leaf_idx_of_node,
            leaf_event_counts=leaf_event_counts,
            leaf_at_risk=leaf_at_risk,
        )


def flatten_tree(
    tree,
    get_split_value: Callable,
    get_leaf_table: Callable,
    split_dtype,
    cache_attr: str = "_flat",
) -> FlatTree:
    """Pre-order DFS flattening. Caches the result on ``getattr(tree, cache_attr)``.

    ``get_split_value(node)`` returns the split scalar for an internal node
    (threshold for reference, bin index for histogram). ``get_leaf_table(node)``
    returns the ``(n_causes, n_time_bins)`` leaf quantity — CIF or CHF.
    Separate cache slots (``_flat`` and ``_flat_chf``) let a tree carry
    both flat representations independently.
    """
    cached = getattr(tree, cache_attr, None)
    if cached is not None:
        return cached

    features: list[int] = []
    splits: list = []
    lefts: list[int] = []
    rights: list[int] = []
    is_leaf: list[bool] = []
    leaf_nodes: list[tuple[int, np.ndarray]] = []

    def visit(node) -> int:
        idx = len(features)
        features.append(0)
        splits.append(0)
        lefts.append(0)
        rights.append(0)
        is_leaf.append(False)
        if node.is_leaf:
            is_leaf[idx] = True
            leaf_nodes.append((idx, get_leaf_table(node)))
            return idx
        li = visit(node.left)
        ri = visit(node.right)
        features[idx] = node.feature
        splits[idx] = get_split_value(node)
        lefts[idx] = li
        rights[idx] = ri
        return idx

    visit(tree)

    n_nodes = len(features)
    n_leaves = len(leaf_nodes)
    example = leaf_nodes[0][1]
    leaf_table = np.empty((n_leaves, *example.shape), dtype=np.float64)
    leaf_idx_of_node = np.full(n_nodes, -1, dtype=np.int64)
    for k, (node_idx, val) in enumerate(leaf_nodes):
        leaf_table[k] = val
        leaf_idx_of_node[node_idx] = k

    flat = FlatTree(
        features=np.asarray(features, dtype=np.int64),
        split_values=np.asarray(splits, dtype=split_dtype),
        left_children=np.asarray(lefts, dtype=np.int64),
        right_children=np.asarray(rights, dtype=np.int64),
        is_leaf_flags=np.asarray(is_leaf, dtype=bool),
        leaf_table=leaf_table,
        leaf_idx_of_node=leaf_idx_of_node,
    )
    setattr(tree, cache_attr, flat)
    return flat


@njit(cache=True, nogil=True)
def _descend_flat_nogil(
    features: np.ndarray,
    split_values: np.ndarray,
    left_children: np.ndarray,
    right_children: np.ndarray,
    is_leaf_flags: np.ndarray,
    leaf_idx_of_node: np.ndarray,
    X: np.ndarray,
) -> np.ndarray:
    """Per-sample root-to-leaf descent; returns the leaf-space index for each row.

    ``leaf_idx_of_node[node]`` maps the descended node index into the compact
    ``leaf_table`` index space ``[0, n_leaves)``. Numba specializes by dtype
    on first call for each ``(X.dtype, split_values.dtype)`` combination.
    Both reference-mode (float64, float64) and histogram-mode (uint8, int64)
    are exercised by the test suite.
    """
    n_samples = X.shape[0]
    leaf_idx = np.empty(n_samples, dtype=np.int64)
    for i in range(n_samples):
        node = 0
        while not is_leaf_flags[node]:
            feat = features[node]
            node = left_children[node] if X[i, feat] <= split_values[node] else right_children[node]
        leaf_idx[i] = leaf_idx_of_node[node]
    return leaf_idx


def predict_leaf_indices(flat: FlatTree, X: np.ndarray) -> np.ndarray:
    """Return the compact leaf index each row of ``X`` descends into.

    Thin Python-level wrapper that unpacks ``flat`` into the ndarray
    arguments the jitted kernel requires. The ``<=`` comparison inside
    is specialized per dtype by numba, so both reference mode
    (float64 ``X``, float thresholds) and histogram mode (uint8 ``X``,
    bin-index thresholds) are handled by the same source kernel.
    """
    return _descend_flat_nogil(
        flat.features,
        flat.split_values,
        flat.left_children,
        flat.right_children,
        flat.is_leaf_flags,
        flat.leaf_idx_of_node,
        X,
    )


def predict_with_flat(flat: FlatTree, X: np.ndarray) -> np.ndarray:
    """Vectorized predict: descend each row of ``X`` and return the leaf table."""
    return flat.leaf_table[predict_leaf_indices(flat, X)]
