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
    """Expected minimal depth of any single variable under the null.

    Under the null hypothesis of no association, every internal node picks
    one of ``p = n_features`` features uniformly. With ``L(d)`` internal
    nodes at depth ``d``, the probability that a specific variable ``V`` is
    *not* picked at any node up to and including depth ``k`` is
    ``(1 - 1/p) ** cumL[k]``. Hence:

        E[md(V)] = sum_{k=0..D_T} P(md > k)
                 = sum_{k=0..D_T} (1 - 1/p) ** cumL[k]

    Cumulative ``cumL`` is constant past the deepest internal node, so
    trailing depths contribute the "never used" mass ``(1 - 1/p)^total``.
    """
    p = int(n_features)
    if p <= 1:
        return 0.0
    L = np.asarray(internal_nodes_per_depth, dtype=np.int64)
    D_T = int(max_depth_T)
    cumL_full = np.zeros(D_T + 1, dtype=np.int64)
    n_L = L.shape[0]
    if n_L > 0:
        cumL_full[:n_L] = np.cumsum(L)
        cumL_full[n_L:] = cumL_full[n_L - 1]
    log1m = np.log1p(-1.0 / p)
    P_greater = np.exp(cumL_full.astype(np.float64) * log1m)
    return float(P_greater.sum())


def compute_minimal_depth(
    forest,
    *,
    threshold: str = "md",
    conservative: bool = False,
    return_extra: bool = False,
) -> pd.DataFrame:
    """Ishwaran-style minimal-depth variable selection.

    Parameters
    ----------
    forest : CompetingRiskForest
        Fitted forest (``trees_`` populated).
    threshold : {"md"}, default "md"
        Selection threshold. Only ``"md"`` (Ishwaran analytical) supported
        in v0.3.0; future releases may add ``"vh"`` (variable hunting).
    conservative : bool, default False
        If True, subtract ``2 * stderr(E[md_T])`` from the threshold for a
        stricter cut (Ishwaran's high-confidence mode).
    return_extra : bool, default False
        If True, append ``min_depth_q25``, ``min_depth_q75``,
        ``frac_trees_used`` columns for diagnostic plots.

    Returns
    -------
    pandas.DataFrame
        Sorted ascending by ``mean_min_depth``. Columns:
        ``feature``, ``mean_min_depth``, ``threshold``, ``selected``.
    """
    if threshold != "md":
        raise ValueError(f"threshold must be 'md' (got {threshold!r}); 'vh' is not yet supported.")
    if not isinstance(conservative, bool):
        raise TypeError(f"conservative must be bool, got {type(conservative).__name__}")

    p = forest.n_features_in_
    feature_names = forest._importance_feature_names()
    trees = forest.trees_
    n_trees = len(trees)
    assert n_trees > 0, "fitted forest must have at least one tree"

    md_matrix = np.empty((n_trees, p), dtype=np.int32)
    expected_md = np.empty(n_trees, dtype=np.float64)
    for i, tree in enumerate(trees):
        res = _walk_min_depth(tree, p)
        md_matrix[i] = res.min_depth_per_feature
        expected_md[i] = _ishwaran_expected_md(res.internal_nodes_per_depth, res.max_depth, p)

    mean_md = md_matrix.mean(axis=0)
    thr = float(expected_md.mean())
    if conservative and n_trees > 1:
        se = float(expected_md.std(ddof=1) / np.sqrt(n_trees))
        thr = thr - 2.0 * se
    selected = mean_md <= thr

    out = (
        pd.DataFrame(
            {
                "feature": feature_names,
                "mean_min_depth": mean_md.astype(np.float64),
                "threshold": np.full(p, thr, dtype=np.float64),
                "selected": selected,
            }
        )
        .sort_values("mean_min_depth", kind="mergesort")
        .reset_index(drop=True)
    )

    if return_extra:
        q25 = np.quantile(md_matrix, 0.25, axis=0)
        q75 = np.quantile(md_matrix, 0.75, axis=0)
        # Per-tree sentinel = D_T + 1 = max value in that tree's row
        per_tree_sentinel = md_matrix.max(axis=1, keepdims=True)
        frac_used = (md_matrix < per_tree_sentinel).mean(axis=0)
        # Reorder extras to match the sorted output
        name_to_pos = {n: i for i, n in enumerate(feature_names)}
        orig_idx = np.array([name_to_pos[n] for n in out["feature"]], dtype=np.int64)
        out["min_depth_q25"] = q25[orig_idx]
        out["min_depth_q75"] = q75[orig_idx]
        out["frac_trees_used"] = frac_used[orig_idx]
    return out
