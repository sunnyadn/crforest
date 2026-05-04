"""Reference-mode tree builder and vectorized predictor.

Leaves store count-based sufficient statistics — per-cause event counts
and at-risk counts on the training ``unique_times`` grid — and the CIF
(Aalen-Johansen) and CHF (Nelson-Aalen) tables are materialized lazily
on the first call to ``predict_tree`` / ``predict_tree_chf``. Same
shape as ``_hist_tree.HistTreeNode``; reference mode differs only in
(a) exact-split search instead of histogram kernels and (b) raw
(unbinned) thresholds at internal nodes.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from comprisk._estimators import (
    aalen_johansen_from_counts,
    leaf_counts_from_time_event,
    nelson_aalen_from_counts,
)
from comprisk._splits import find_best_split
from comprisk._tree_flat import FlatTree, flatten_tree, predict_with_flat


@dataclass
class RefTreeNode:
    """Tree node for reference mode. Stores raw (unbinned) threshold values."""

    is_leaf: bool = False
    feature: int = -1
    threshold: float = -1.0
    left: RefTreeNode | None = None
    right: RefTreeNode | None = None
    event_counts: np.ndarray | None = None  # (n_causes, n_times) float64 on leaves
    at_risk: np.ndarray | None = None  # (n_times,) float64 on leaves
    _cif: np.ndarray | None = field(default=None, repr=False, compare=False)
    _chf: np.ndarray | None = field(default=None, repr=False, compare=False)
    _flat: FlatTree | None = field(default=None, repr=False, compare=False)
    _flat_chf: FlatTree | None = field(default=None, repr=False, compare=False)


@dataclass
class _TreeConfig:
    n_causes: int
    max_depth: int | None
    min_samples_split: int
    min_samples_leaf: int
    unique_times: np.ndarray
    max_features: int | None = None
    rng: np.random.RandomState | None = None
    splitrule: str = "logrankCR"
    cause: int = 1
    cause_weights: np.ndarray | None = None
    nsplit: int = 0


def build_tree(
    X: np.ndarray,
    time: np.ndarray,
    event: np.ndarray,
    n_causes: int,
    max_depth: int | None,
    min_samples_split: int,
    min_samples_leaf: int,
    unique_times: np.ndarray | None = None,
    max_features: int | None = None,
    rng: np.random.RandomState | None = None,
    *,
    splitrule: str = "logrankCR",
    cause: int = 1,
    cause_weights: np.ndarray | None = None,
    nsplit: int = 0,
) -> RefTreeNode:
    """Build a reference-mode (exact-split) CR tree. Returns the root node."""
    if max_features is not None and rng is None:
        raise ValueError("max_features requires an rng; pass both or neither")
    if nsplit > 0 and rng is None:
        raise ValueError("nsplit > 0 requires an rng")
    if unique_times is None:
        unique_times = np.sort(np.unique(time))
    cfg = _TreeConfig(
        n_causes=n_causes,
        max_depth=max_depth,
        min_samples_split=min_samples_split,
        min_samples_leaf=min_samples_leaf,
        unique_times=unique_times,
        max_features=max_features,
        rng=rng,
        splitrule=splitrule,
        cause=cause,
        cause_weights=cause_weights,
        nsplit=nsplit,
    )
    return _build_node(X, time, event, cfg, depth=0)


def _build_node(
    X: np.ndarray,
    time: np.ndarray,
    event: np.ndarray,
    cfg: _TreeConfig,
    depth: int,
) -> RefTreeNode:
    n_samples, n_features = X.shape

    too_small = n_samples < cfg.min_samples_split
    too_deep = cfg.max_depth is not None and depth >= cfg.max_depth
    if too_small or too_deep:
        return _make_leaf(time, event, cfg.unique_times, cfg.n_causes)

    if cfg.max_features is not None and cfg.rng is not None:
        k = min(cfg.max_features, n_features)
        selected = cfg.rng.choice(n_features, size=k, replace=False)
        X_sub = X[:, selected]
    else:
        X_sub = X
        selected = None

    best_feature, best_threshold, _ = find_best_split(
        X_sub,
        time,
        event,
        cfg.n_causes,
        cfg.min_samples_leaf,
        splitrule=cfg.splitrule,
        cause=cfg.cause,
        cause_weights=cfg.cause_weights,
        nsplit=cfg.nsplit,
        rng=cfg.rng,
    )
    if best_feature < 0:
        return _make_leaf(time, event, cfg.unique_times, cfg.n_causes)

    actual_feature = int(selected[best_feature]) if selected is not None else best_feature
    col = X[:, actual_feature]
    left_mask = col <= best_threshold

    node = RefTreeNode(feature=actual_feature, threshold=best_threshold)
    node.left = _build_node(X[left_mask], time[left_mask], event[left_mask], cfg, depth + 1)
    node.right = _build_node(X[~left_mask], time[~left_mask], event[~left_mask], cfg, depth + 1)
    return node


def _make_leaf(
    time: np.ndarray,
    event: np.ndarray,
    unique_times: np.ndarray,
    n_causes: int,
) -> RefTreeNode:
    event_counts, at_risk = leaf_counts_from_time_event(time, event, unique_times, n_causes)
    return RefTreeNode(is_leaf=True, event_counts=event_counts, at_risk=at_risk)


def _flatten_tree(tree: RefTreeNode) -> FlatTree:
    """Flatten a reference tree; leaf CIFs are lazily materialized from counts."""

    def get_leaf_cif(node: RefTreeNode) -> np.ndarray:
        if node._cif is None:
            n_causes = node.event_counts.shape[0]
            node._cif = aalen_johansen_from_counts(node.event_counts, node.at_risk, n_causes)
        return node._cif

    return flatten_tree(
        tree,
        get_split_value=lambda n: n.threshold,
        get_leaf_table=get_leaf_cif,
        split_dtype=np.float64,
    )


def predict_tree(tree: RefTreeNode, X: np.ndarray) -> np.ndarray:
    """Predict leaf CIFs for each row of X. Returns (n_samples, n_causes, n_times)."""
    X = np.asarray(X, dtype=np.float64)
    if X.ndim != 2:
        raise ValueError(f"X must be 2-D; got ndim={X.ndim}")
    return predict_with_flat(_flatten_tree(tree), X)


def _flatten_tree_chf(tree: RefTreeNode) -> FlatTree:
    """Flatten a reference tree; leaf CHFs are lazily materialized from counts."""

    def get_leaf_chf(node: RefTreeNode) -> np.ndarray:
        if node._chf is None:
            n_causes = node.event_counts.shape[0]
            node._chf = nelson_aalen_from_counts(node.event_counts, node.at_risk, n_causes)
        return node._chf

    return flatten_tree(
        tree,
        get_split_value=lambda n: n.threshold,
        get_leaf_table=get_leaf_chf,
        split_dtype=np.float64,
        cache_attr="_flat_chf",
    )


def predict_tree_chf(tree: RefTreeNode, X: np.ndarray) -> np.ndarray:
    """Predict leaf CHFs for each row of X. Returns (n_samples, n_causes, n_times)."""
    X = np.asarray(X, dtype=np.float64)
    if X.ndim != 2:
        raise ValueError(f"X must be 2-D; got ndim={X.ndim}")
    return predict_with_flat(_flatten_tree_chf(tree), X)
