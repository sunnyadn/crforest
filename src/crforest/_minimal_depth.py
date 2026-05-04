"""Ishwaran-style minimal-depth variable selection for competing-risks forests.

A variable's *minimal depth* in a tree is the depth of the highest (closest
to root) split that uses that variable. Variables never split on receive a
sentinel depth of ``D_T + 1`` where ``D_T`` is the tree's maximum node depth.
Smaller mean minimal depth across the forest indicates a more important
variable.

The selection threshold is the per-forest mean of the *expected* minimal
depth under the null hypothesis of no association — derived analytically
from each tree's depth structure assuming uniform random feature selection
at every split (Ishwaran et al. 2010, JASA, eq. 4.1).

References
----------
Ishwaran, H., Kogalur, U.B., Gorodeski, E.Z., Minn, A.J., Lauer, M.S. (2010).
"High-dimensional variable selection for survival data."
*Journal of the American Statistical Association* 105(489): 205-217.
-- minimal depth + analytical threshold.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

__all__ = [
    "WalkResult",
    "_ishwaran_expected_md",
    "_walk_min_depth",
    "compute_minimal_depth",
]


@dataclass
class WalkResult:
    """Per-tree output of :func:`_walk_min_depth`."""

    min_depth_per_feature: np.ndarray  # (n_features,) int32, sentinel = D_T + 1
    internal_nodes_per_depth: np.ndarray  # (D_internal_max + 1,) int64
    max_depth: int  # D_T = max depth of any node in the tree


def _walk_min_depth(tree, n_features: int) -> WalkResult:
    """Per-tree minimal-depth walker. Dispatches on tree type via isinstance.

    Returns a :class:`WalkResult` with three arrays:

    - ``min_depth_per_feature`` : ``int32`` of shape ``(n_features,)``;
      depth of highest split using each feature, or ``D_T + 1`` if unused.
    - ``internal_nodes_per_depth`` : ``int64`` of length ``D_internal + 1``;
      count of internal (non-leaf) nodes at each depth.
    - ``max_depth`` : int; depth of the deepest node (leaf or internal).
    """
    from crforest._hist_tree import HistTreeNode
    from crforest._tree import RefTreeNode
    from crforest._tree_flat import FlatTree

    if isinstance(tree, FlatTree):
        return _walk_flat(tree, n_features)
    if isinstance(tree, (HistTreeNode, RefTreeNode)):
        return _walk_recursive(tree, n_features)
    raise TypeError(f"_walk_min_depth: unsupported tree type {type(tree).__name__}")


def _walk_flat(tree, n_features: int) -> WalkResult:
    SENTINEL = np.iinfo(np.int32).max  # interim, replaced with D_T+1
    min_depth = np.full(n_features, SENTINEL, dtype=np.int32)
    L: list[int] = []  # per-depth internal-node histogram, grown on demand
    max_depth = 0
    # iterative DFS; node 0 is root
    stack = [(0, 0)]
    while stack:
        i, d = stack.pop()
        if d > max_depth:
            max_depth = d
        if not tree.is_leaf_flags[i]:
            f = int(tree.features[i])
            if d < min_depth[f]:
                min_depth[f] = d
            while len(L) <= d:
                L.append(0)
            L[d] += 1
            stack.append((int(tree.right_children[i]), d + 1))
            stack.append((int(tree.left_children[i]), d + 1))
    sentinel = np.int32(max_depth + 1)
    min_depth = np.where(min_depth == SENTINEL, sentinel, min_depth).astype(np.int32)
    return WalkResult(
        min_depth_per_feature=min_depth,
        internal_nodes_per_depth=np.asarray(L, dtype=np.int64),
        max_depth=max_depth,
    )


def _walk_recursive(root, n_features: int) -> WalkResult:
    SENTINEL = np.iinfo(np.int32).max
    min_depth = np.full(n_features, SENTINEL, dtype=np.int32)
    L: list[int] = []
    max_depth = 0
    stack = [(root, 0)]
    while stack:
        node, d = stack.pop()
        if d > max_depth:
            max_depth = d
        if not node.is_leaf:
            f = int(node.feature)
            if d < min_depth[f]:
                min_depth[f] = d
            while len(L) <= d:
                L.append(0)
            L[d] += 1
            stack.append((node.right, d + 1))
            stack.append((node.left, d + 1))
    sentinel = np.int32(max_depth + 1)
    min_depth = np.where(min_depth == SENTINEL, sentinel, min_depth).astype(np.int32)
    return WalkResult(
        min_depth_per_feature=min_depth,
        internal_nodes_per_depth=np.asarray(L, dtype=np.int64),
        max_depth=max_depth,
    )


def _ishwaran_expected_md(
    internal_nodes_per_depth: np.ndarray,
    max_depth_T: int,
    n_features: int,
) -> float:
    """Expected minimal depth of any single variable under the null."""
    raise NotImplementedError


def compute_minimal_depth(
    forest,
    *,
    threshold: str = "md",
    conservative: bool = False,
    return_extra: bool = False,
) -> pd.DataFrame:
    """Compute minimal-depth ranking + threshold-based selection. See API spec."""
    raise NotImplementedError
