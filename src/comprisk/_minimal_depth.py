"""Ishwaran-style minimal-depth variable selection for competing-risks forests.

A variable's *minimal depth* in a tree is the depth of the highest (closest
to root) split that uses that variable. Variables never split on receive a
sentinel depth of ``D_T`` where ``D_T`` is the tree's maximum node depth
(Ishwaran et al. 2010, JASA, Eq. (2)).

The selection threshold is the expected minimal depth under the null hypothesis
of no association, computed once from forest-averaged node counts and average
tree depth (Ishwaran et al. 2010, JASA, Theorem 1 / Eq. (1) for the
distribution; Section 3 for the forest-averaged threshold construction).

References
----------
Ishwaran, H., Kogalur, U.B., Gorodeski, E.Z., Minn, A.J., Lauer, M.S. (2010).
"High-dimensional variable selection for survival data."
*Journal of the American Statistical Association* 105(489): 205-217.
-- Theorem 1 / Eq. (1): per-tree null distribution of minimal depth.
-- Eq. (2): sentinel convention D_T for never-split variables.
-- Section 3: forest-averaged threshold via l_bar*_d and D_bar.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

__all__ = [
    "WalkResult",
    "_forest_averaged_threshold",
    "_ishwaran_expected_md",
    "_walk_min_depth",
    "compute_minimal_depth",
]


@dataclass
class WalkResult:
    """Per-tree output of :func:`_walk_min_depth`."""

    min_depth_per_feature: np.ndarray  # (n_features,) int32, sentinel = D_T
    internal_nodes_per_depth: np.ndarray  # (D_internal_max + 1,) int64
    max_depth: int  # D_T = max depth of any node in the tree


def _walk_min_depth(tree, n_features: int) -> WalkResult:
    """Per-tree minimal-depth walker. Dispatches on tree type via isinstance.

    Returns a :class:`WalkResult` with three arrays:

    - ``min_depth_per_feature`` : ``int32`` of shape ``(n_features,)``;
      depth of highest split using each feature, or ``D_T`` (sentinel per
      Ishwaran et al. 2010, JASA, Eq. (2)) if unused.
    - ``internal_nodes_per_depth`` : ``int64`` of length ``D_internal + 1``;
      count of internal (non-leaf) nodes at each depth.
    - ``max_depth`` : int; depth of the deepest node (leaf or internal).
    """
    from comprisk._hist_tree import HistTreeNode
    from comprisk._tree import RefTreeNode
    from comprisk._tree_flat import FlatTree

    if isinstance(tree, FlatTree):
        return _walk_flat(tree, n_features)
    if isinstance(tree, (HistTreeNode, RefTreeNode)):
        return _walk_recursive(tree, n_features)
    raise TypeError(f"_walk_min_depth: unsupported tree type {type(tree).__name__}")


def _walk_flat(tree, n_features: int) -> WalkResult:
    SENTINEL = np.iinfo(np.int32).max  # interim, replaced with D_T
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
    sentinel = np.int32(max_depth)
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
    sentinel = np.int32(max_depth)
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

    Per Ishwaran et al. (2010, JASA, Theorem 1 / Eq. (1)) and the sentinel
    convention from Eq. (2). For a weak variable the per-depth survival
    probability is ``(1 - 1/p)^L(d)``; the cumulative survival to depth k
    is ``(1 - 1/p)^cumL[k]``. By the standard non-negative integer identity:

        E[Dv] = sum_{k=0..D_T-1} P(Dv > k)
              = sum_{k=0..D_T-1} (1 - 1/p)^cumL[k]

    where the sum has D_T terms (NOT D_T + 1) because Dv is bounded by D_T
    (the deepest leaf), and P(Dv > D_T) = 0 by the sentinel convention Eq. (2).

    For pure stumps (D_T = 0): empty sum, returns 0.0 (every variable has
    Dv = 0 = D_T trivially).

    This function accepts float-valued ``internal_nodes_per_depth`` to support
    forest-averaged node counts l_bar*_d (Section 3).
    """
    p = int(n_features)
    if p <= 1:
        return 0.0
    L = np.asarray(internal_nodes_per_depth, dtype=np.float64)
    D_T = int(max_depth_T)
    if D_T == 0:
        return 0.0
    cumL_partial = np.zeros(D_T, dtype=np.float64)
    n_L = L.shape[0]
    if n_L > 0:
        n_take = min(D_T, n_L)
        cumL_partial[:n_take] = np.cumsum(L[:n_take])
        if n_take < D_T:
            cumL_partial[n_take:] = cumL_partial[n_take - 1]
    log1m = np.log1p(-1.0 / p)
    P_greater = np.exp(cumL_partial * log1m)
    return float(P_greater.sum())


def _forest_averaged_threshold(walk_results: list[WalkResult], n_features: int) -> float:
    """Threshold = E[D*v] computed under the paper's recommended forest-averaging.

    Per Ishwaran et al. (2010, JASA, Section 3): "in place of Dv we used D*v,
    a random variable with distribution (6), but with node counts l_d replaced
    by forest-averaged estimates l_bar*_d ... let D_bar be the average tree
    depth of the forest. Then D*v in {0, 1, ..., D_bar}".

    Implementation:
    1. l_bar*_d = mean across trees of L_d (count of internal nodes at depth d).
       Trees with no nodes at depth d contribute 0.
    2. D_bar = mean across trees of D_T, rounded to nearest int.
    3. Plug into _ishwaran_expected_md once.

    Note: ``D_bar`` uses Python's ``round()`` (round-half-to-even) on the
    mean tree depth. Half-integer means like 2.5 round to 2, not 3.
    """
    n_trees = len(walk_results)
    if n_trees == 0:
        return 0.0
    # Pad each tree's L to a common max depth, then average
    max_L_len = max((wr.internal_nodes_per_depth.shape[0] for wr in walk_results), default=0)
    if max_L_len == 0:
        return 0.0
    L_matrix = np.zeros((n_trees, max_L_len), dtype=np.float64)
    for i, wr in enumerate(walk_results):
        n_L = wr.internal_nodes_per_depth.shape[0]
        if n_L > 0:
            L_matrix[i, :n_L] = wr.internal_nodes_per_depth
    L_bar = L_matrix.mean(axis=0)  # shape (max_L_len,)
    D_bar = round(float(np.mean([wr.max_depth for wr in walk_results])))
    return _ishwaran_expected_md(L_bar, D_bar, n_features)


def compute_minimal_depth(
    forest,
    *,
    threshold: str = "md",
    return_extra: bool = False,
) -> pd.DataFrame:
    """Ishwaran-style minimal-depth variable selection.

    Implements the forest-averaged threshold method described in Ishwaran
    et al. (2010, JASA, Section 3): per-variable minimal depth is averaged
    empirically across trees, and the selection threshold is E[D*v] computed
    once from the forest-averaged node-count vector l_bar*_d and average
    tree depth D_bar.

    Parameters
    ----------
    forest : CompetingRiskForest
        Fitted forest (``trees_`` populated).
    threshold : {"md"}, default "md"
        Selection threshold method. Only forest-averaged ``"md"`` (the
        paper's recommendation) is supported in v0.3.0.
    return_extra : bool, default False
        If True, append ``min_depth_q25``, ``min_depth_q75``,
        ``frac_trees_used`` columns for diagnostic plots.

    Returns
    -------
    pandas.DataFrame
        Sorted ascending by ``mean_min_depth``. Columns:
        ``feature``, ``mean_min_depth``, ``threshold``, ``selected``.

    Note: rfSRC's max.subtree defaults to a tree-averaged threshold; this
    function implements the paper's forest-averaging (Section 3), so
    numeric thresholds will differ even with ``equivalence='rfsrc'``.
    Variable rankings tend to agree.
    """
    if threshold != "md":
        raise ValueError(
            f"threshold must be 'md' (got {threshold!r}); other methods are not yet supported."
        )

    p = forest.n_features_in_
    feature_names = forest._importance_feature_names()
    trees = forest.trees_
    n_trees = len(trees)
    if n_trees == 0:
        raise ValueError("forest has no trees; refit with n_estimators >= 1")

    walk_results = [_walk_min_depth(tree, p) for tree in trees]
    md_matrix = np.stack([wr.min_depth_per_feature for wr in walk_results], axis=0)
    mean_md = md_matrix.mean(axis=0)
    thr = _forest_averaged_threshold(walk_results, p)
    selected = mean_md <= thr

    data: dict = {
        "feature": feature_names,
        "mean_min_depth": mean_md.astype(np.float64),
        "threshold": np.full(p, thr, dtype=np.float64),
        "selected": selected,
    }
    if return_extra:
        data["min_depth_q25"] = np.quantile(md_matrix, 0.25, axis=0)
        data["min_depth_q75"] = np.quantile(md_matrix, 0.75, axis=0)
        per_tree_sentinel_arr = np.array([wr.max_depth for wr in walk_results], dtype=np.int64)
        data["frac_trees_used"] = (md_matrix < per_tree_sentinel_arr[:, None]).mean(axis=0)
    out = pd.DataFrame(data).sort_values("mean_min_depth", kind="mergesort").reset_index(drop=True)
    return out
