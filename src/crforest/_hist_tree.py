"""Histogram-mode tree builder, predictor, and leaf CIF materialization.

Mirrors the structure of ``_tree.py`` (recursive build, vectorized predict
via a flattened-tree descent) but operates on pre-binned uint8 feature
arrays and stores compact per-cause event/at-risk counts at leaves.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from crforest._estimators import aalen_johansen_from_counts, nelson_aalen_from_counts
from crforest._hist_splits import find_best_split_hist
from crforest._sparse_leaves import (
    SparseAtRisk,
    SparseEventCounts,
    to_dense_at_risk,
    to_dense_event_counts,
    to_sparse_at_risk,
    to_sparse_event_counts,
)
from crforest._tree_flat import FlatTree, flatten_tree, predict_leaf_indices, predict_with_flat

_TRANSIENT_CACHE_ATTRS = (
    "_event_counts_dense",
    "_at_risk_dense",
    "_cif",
    "_chf",
    "_flat",
    "_flat_chf",
)


@dataclass
class HistTreeNode:
    is_leaf: bool = False
    feature: int = -1
    bin_idx: int = 0
    left: HistTreeNode | None = None
    right: HistTreeNode | None = None

    # Internal nodes leave count fields None.
    event_counts_sparse: SparseEventCounts | None = None
    at_risk_sparse: SparseAtRisk | None = None
    # Grid shape needed to materialize dense views without querying the forest.
    _n_causes: int = 0
    _n_time_bins: int = 0

    # Lazy dense caches. Trees are frozen after fit, so never invalidated.
    # Dropped in __getstate__ so pickle only carries the sparse rep.
    _event_counts_dense: np.ndarray | None = field(default=None, repr=False, compare=False)
    _at_risk_dense: np.ndarray | None = field(default=None, repr=False, compare=False)

    _cif: np.ndarray | None = field(default=None, repr=False, compare=False)
    _chf: np.ndarray | None = field(default=None, repr=False, compare=False)
    _flat: FlatTree | None = field(default=None, repr=False, compare=False)
    _flat_chf: FlatTree | None = field(default=None, repr=False, compare=False)

    @property
    def event_counts_dense(self) -> np.ndarray:
        if self._event_counts_dense is None:
            self._event_counts_dense = to_dense_event_counts(
                self.event_counts_sparse, self._n_causes, self._n_time_bins
            )
        return self._event_counts_dense

    @property
    def at_risk_dense(self) -> np.ndarray:
        if self._at_risk_dense is None:
            self._at_risk_dense = to_dense_at_risk(self.at_risk_sparse, self._n_time_bins)
        return self._at_risk_dense

    def __getstate__(self):
        state = self.__dict__.copy()
        for attr in _TRANSIENT_CACHE_ATTRS:
            state[attr] = None
        return state


def _leaf_counts(
    time_indices: np.ndarray,  # (n,) int32
    event: np.ndarray,  # (n,) int64
    n_causes: int,
    n_time_bins: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Return (event_counts, at_risk) for a leaf's samples.

    event_counts[k, t] = count of samples with event==k+1 and time_index==t.
    at_risk[t]         = count of samples with time_index >= t.
    """
    event_counts = np.zeros((n_causes, n_time_bins), dtype=np.uint32)
    for k in range(n_causes):
        mask = event == (k + 1)
        if mask.any():
            event_counts[k] = np.bincount(time_indices[mask], minlength=n_time_bins).astype(
                np.uint32
            )
    n_at = np.bincount(time_indices, minlength=n_time_bins)
    at_risk = np.cumsum(n_at[::-1])[::-1].astype(np.uint32)
    return event_counts, at_risk


@dataclass
class _HistBuildConfig:
    n_causes: int
    n_bins: int
    n_time_bins: int  # split-search time bins (coarse if split_ntime < full)
    n_time_bins_full: int  # full time grid, for leaves
    max_depth: int | None
    min_samples_split: int
    min_samples_leaf: int
    max_features: int | None
    rng: np.random.RandomState | None
    splitrule: str = "logrankCR"
    cause: int = 1
    nsplit: int = 0
    use_batched: bool = False
    rng_mode: str = "numpy"  # "numpy" or "rfsrc_aligned"; governs per-node flow
    rfsrc_permissible_: np.ndarray | None = None  # per-node inherited mask; mutated in-place


def build_tree_hist(
    X_binned: np.ndarray,
    time_indices: np.ndarray,
    event: np.ndarray,
    n_causes: int,
    n_bins: int,
    n_time_bins: int,
    max_depth: int | None,
    min_samples_split: int,
    min_samples_leaf: int,
    max_features: int | None,
    rng: np.random.RandomState | None,
    *,
    splitrule: str = "logrankCR",
    cause: int = 1,
    nsplit: int = 0,
    time_indices_full: np.ndarray | None = None,
    n_time_bins_full: int | None = None,
    use_batched: bool = False,
    rng_mode: str = "numpy",
) -> HistTreeNode:
    """Build a histogram-mode tree. Returns the root node.

    Parameters mirror ``_tree.build_tree`` but the data arrays are
    pre-binned (uint8) and time-indexed (int32).

    ``time_indices`` / ``n_time_bins`` drive split search (coarse grid when
    ``split_ntime`` is active).  ``time_indices_full`` / ``n_time_bins_full``
    drive leaf population (full grid).  When the full-grid kwargs are omitted
    they default to the split-grid values, reproducing the uncoarsened
    behaviour.

    ``rng_mode`` selects the per-node mtry/nsplit call order. ``"numpy"``
    (default) draws all mtry features upfront then evaluates.
    ``"rfsrc_aligned"`` interleaves draw + evaluate per feature to match
    rfSRC's per-feature ordering. The two produce different trees when
    ``rng`` consumes stream B differently.
    """
    if max_features is not None and rng is None:
        raise ValueError("max_features requires an rng; pass both or neither")
    if nsplit > 0 and rng is None:
        raise ValueError("nsplit > 0 requires an rng")
    if rng_mode not in ("numpy", "rfsrc_aligned"):
        raise ValueError(f"rng_mode must be 'numpy' or 'rfsrc_aligned'; got {rng_mode!r}")
    if time_indices_full is None:
        time_indices_full = time_indices
        n_time_bins_full = n_time_bins
    elif n_time_bins_full is None:
        raise ValueError("n_time_bins_full must be given alongside time_indices_full")
    cfg = _HistBuildConfig(
        n_causes=n_causes,
        n_bins=n_bins,
        n_time_bins=n_time_bins,
        n_time_bins_full=n_time_bins_full,
        max_depth=max_depth,
        min_samples_split=min_samples_split,
        min_samples_leaf=min_samples_leaf,
        max_features=max_features,
        rng=rng,
        splitrule=splitrule,
        cause=cause,
        nsplit=nsplit,
        use_batched=use_batched,
        rng_mode=rng_mode,
    )
    return _build_node_hist(X_binned, time_indices, time_indices_full, event, cfg, depth=0)


def _build_node_hist(
    X_binned: np.ndarray,
    time_indices_split: np.ndarray,
    time_indices_full: np.ndarray,
    event: np.ndarray,
    cfg: _HistBuildConfig,
    depth: int,
) -> HistTreeNode:
    n_samples, n_features = X_binned.shape
    too_small = n_samples < cfg.min_samples_split
    too_deep = cfg.max_depth is not None and depth >= cfg.max_depth
    if too_small or too_deep:
        return _make_leaf(time_indices_full, event, cfg.n_causes, cfg.n_time_bins_full)

    if cfg.rng_mode == "rfsrc_aligned":
        feature, bin_idx = _find_best_split_rfsrc_aligned(
            X_binned, time_indices_split, event, n_features, cfg, depth
        )
    else:
        if cfg.max_features is not None:
            k = min(cfg.max_features, n_features)
            selected = cfg.rng.choice(n_features, size=k, replace=False).astype(np.int64)
        else:
            selected = np.arange(n_features, dtype=np.int64)

        feature, bin_idx, _ = find_best_split_hist(
            X_binned,
            time_indices_split,
            event,
            selected,
            cfg.n_bins,
            cfg.n_causes,
            cfg.n_time_bins,
            cfg.min_samples_leaf,
            splitrule=cfg.splitrule,
            cause=cfg.cause,
            nsplit=cfg.nsplit,
            rng=cfg.rng,
            use_batched=cfg.use_batched,
        )
    if feature < 0:
        return _make_leaf(time_indices_full, event, cfg.n_causes, cfg.n_time_bins_full)

    left_mask = X_binned[:, feature] <= bin_idx
    node = HistTreeNode(feature=feature, bin_idx=bin_idx)
    # Snapshot the rfsrc_permissible mask so left and right subtrees see
    # the same inherited state from this node (mutations inside one
    # subtree must not leak into the sibling).
    _parent_permissible = cfg.rfsrc_permissible_
    cfg.rfsrc_permissible_ = _parent_permissible.copy() if _parent_permissible is not None else None
    node.left = _build_node_hist(
        X_binned[left_mask],
        time_indices_split[left_mask],
        time_indices_full[left_mask],
        event[left_mask],
        cfg,
        depth + 1,
    )
    cfg.rfsrc_permissible_ = _parent_permissible.copy() if _parent_permissible is not None else None
    node.right = _build_node_hist(
        X_binned[~left_mask],
        time_indices_split[~left_mask],
        time_indices_full[~left_mask],
        event[~left_mask],
        cfg,
        depth + 1,
    )
    cfg.rfsrc_permissible_ = _parent_permissible
    return node


def _find_best_split_rfsrc_aligned(
    X_binned: np.ndarray,
    time_indices_split: np.ndarray,
    event: np.ndarray,
    n_features: int,
    cfg: _HistBuildConfig,
    depth: int,
) -> tuple[int, int]:
    """Reproduce rfSRC's interleaved per-feature mtry + nsplit draw order.

    Draws mtry features one at a time from stream B via SWOR (ceil +
    swap-with-last), evaluating the split on each drawn feature before
    drawing the next. Within each evaluation, ``nsplit`` candidate
    thresholds are sampled from the same stream B (handled inside
    ``find_best_split_hist``). This interleaving is the only reason this
    helper exists — crforest's default ``numpy`` flow batches all mtry
    draws upfront, which consumes stream B in a different order and
    therefore picks different features even at an aligned seed.
    """
    import os

    from crforest._aligned_rng import AlignedRng

    if not isinstance(cfg.rng, AlignedRng):
        raise TypeError(
            "rng_mode='rfsrc_aligned' requires an AlignedRng instance; "
            f"got {type(cfg.rng).__name__}"
        )
    stream = cfg.rng.stream
    k = min(cfg.max_features, n_features) if cfg.max_features is not None else n_features

    # rfSRC maintains a per-node `permissible` mask, inherited from parent and
    # mutated: after drawing a feature, if Phase 1 finds it constant at the
    # current node (vectorSize < 2), the mask entry flips to FALSE so
    # descendants inherit the exclusion. Pool at any given node = features
    # still marked permissible in that node's inherited mask.
    #
    # We replicate this by threading an in-place ``permissible`` ndarray
    # through the tree-building recursion via ``cfg.rfsrc_permissible_``.
    # Before descending into a child we take a copy so sibling subtrees
    # don't see each other's updates (rfSRC does the same via per-tree-
    # parent ownership of the mask).
    permissible = cfg.rfsrc_permissible_
    if permissible is None:
        permissible = np.ones(n_features, dtype=bool)
        cfg.rfsrc_permissible_ = permissible

    _trace_path = os.environ.get("CRFOREST_TRACE")

    def _log(kind: str, a: int, b: int) -> None:
        if _trace_path:
            with open(_trace_path, "a") as tfp:
                tfp.write(f"{kind} a={a} b={b}\n")

    def _log_ran1(val: float) -> None:
        if _trace_path:
            with open(_trace_path, "a") as tfp:
                tfp.write(f"ran1B val={val:.10f}\n")

    if _trace_path:
        with open(_trace_path, "a") as tfp:
            tfp.write(f"node_start n={len(time_indices_split)} depth={depth}\n")

    pool = np.asarray([f for f in range(n_features) if permissible[f]], dtype=np.int64)
    pool_size = len(pool)

    best_feature = -1
    best_bin = 0
    best_stat = 0.0

    for mtry_i in range(k):
        if pool_size == 0:
            break
        # Draw one feature: slot_1 = ceil(u * pool_size) gives 1-based slot.
        u = stream.next()
        _log_ran1(u)
        slot_1 = int(np.ceil(u * pool_size))
        if slot_1 < 1:
            slot_1 = 1
        elif slot_1 > pool_size:
            slot_1 = pool_size
        slot_0 = slot_1 - 1
        f = int(pool[slot_0])
        pool[slot_0] = pool[pool_size - 1]
        pool_size -= 1
        # rfSRC uses 1-based feature indices in its mtry_pick log; add 1 to match.
        _log("mtry_pick", mtry_i + 1, f + 1)

        # rfSRC Phase 1: if the drawn feature's column has <2 unique values
        # in this node's subset, mark it non-permissible for descendants.
        # min == max is equivalent for "all values identical" but skips the
        # unique-set allocation that np.unique would do (was top hotspot).
        col = X_binned[:, f]
        if col[0] == col[-1] and col.min() == col.max():
            permissible[f] = False
            continue
        single = np.array([f], dtype=np.int64)
        f_sel, bin_idx, stat = find_best_split_hist(
            X_binned,
            time_indices_split,
            event,
            single,
            cfg.n_bins,
            cfg.n_causes,
            cfg.n_time_bins,
            cfg.min_samples_leaf,
            splitrule=cfg.splitrule,
            cause=cfg.cause,
            nsplit=cfg.nsplit,
            rng=cfg.rng,
            use_batched=cfg.use_batched,
            skip_nsplit_rng_when_deterministic=True,
        )
        if f_sel >= 0 and stat > best_stat:
            best_feature = f_sel
            best_bin = bin_idx
            best_stat = stat

    return best_feature, best_bin


def _make_leaf(
    time_indices: np.ndarray,
    event: np.ndarray,
    n_causes: int,
    n_time_bins: int,
) -> HistTreeNode:
    event_counts, at_risk = _leaf_counts(time_indices, event, n_causes, n_time_bins)
    return HistTreeNode(
        is_leaf=True,
        event_counts_sparse=to_sparse_event_counts(event_counts),
        at_risk_sparse=to_sparse_at_risk(at_risk),
        _n_causes=n_causes,
        _n_time_bins=n_time_bins,
    )


def _flatten_tree_hist(tree: HistTreeNode) -> FlatTree:
    """Flatten a histogram tree; leaf CIFs are lazily materialized from counts."""

    def get_leaf_cif(node: HistTreeNode) -> np.ndarray:
        if node._cif is None:
            node._cif = aalen_johansen_from_counts(
                node.event_counts_dense, node.at_risk_dense, node._n_causes
            )
        return node._cif

    return flatten_tree(
        tree,
        get_split_value=lambda n: n.bin_idx,
        get_leaf_table=get_leaf_cif,
        split_dtype=np.int64,
    )


def predict_tree_hist(tree: HistTreeNode | FlatTree, X_binned: np.ndarray) -> np.ndarray:
    """Predict leaf CIFs for each row of binned X.

    Accepts either a ``HistTreeNode`` (legacy path) or a ``FlatTree``
    (new default-mode path; Task 5 dispatch).

    Returns
    -------
    cif : ndarray, shape (n_samples, n_causes, n_time_bins), float64
    """
    if X_binned.dtype != np.uint8:
        raise ValueError(f"X_binned must be uint8; got {X_binned.dtype}")
    if X_binned.ndim != 2:
        raise ValueError(f"X_binned must be 2-D; got ndim={X_binned.ndim}")
    if isinstance(tree, FlatTree):
        return predict_with_flat(tree, X_binned)
    return predict_with_flat(_flatten_tree_hist(tree), X_binned)


def _flatten_tree_hist_chf(tree: HistTreeNode) -> FlatTree:
    """Flatten a histogram tree; leaf CHFs are lazily materialized from counts."""

    def get_leaf_chf(node: HistTreeNode) -> np.ndarray:
        if node._chf is None:
            node._chf = nelson_aalen_from_counts(
                node.event_counts_dense, node.at_risk_dense, node._n_causes
            )
        return node._chf

    return flatten_tree(
        tree,
        get_split_value=lambda n: n.bin_idx,
        get_leaf_table=get_leaf_chf,
        split_dtype=np.int64,
        cache_attr="_flat_chf",
    )


def _flat_tree_chf_leaf_table(flat: FlatTree) -> np.ndarray:
    """Lazily materialise the Nelson-Aalen CHF leaf table on a FlatTree.

    Caches on ``flat._chf_leaf_table`` so repeated predict_chf calls on the
    same tree do not re-evaluate Nelson-Aalen per leaf.
    """
    cached = getattr(flat, "_chf_leaf_table", None)
    if cached is not None:
        return cached
    if flat.leaf_event_counts is None or flat.leaf_at_risk is None:
        raise RuntimeError(
            "FlatTree is missing leaf_event_counts/leaf_at_risk; cannot compute CHF lazily. "
            "This indicates a flatten path that did not persist raw counts."
        )
    n_leaves = flat.leaf_event_counts.shape[0]
    n_causes = flat.leaf_event_counts.shape[1]
    n_time_bins = flat.leaf_event_counts.shape[2]
    chf_table = np.empty((n_leaves, n_causes, n_time_bins), dtype=np.float64)
    for k in range(n_leaves):
        chf_table[k] = nelson_aalen_from_counts(
            flat.leaf_event_counts[k], flat.leaf_at_risk[k], n_causes
        )
    flat._chf_leaf_table = chf_table
    return chf_table


def predict_tree_hist_chf(tree: HistTreeNode | FlatTree, X_binned: np.ndarray) -> np.ndarray:
    """Predict leaf CHFs for each row of binned X.

    Accepts either a ``HistTreeNode`` (legacy/rfsrc-aligned path; uses the
    ``_flat_chf`` sister-tree cache) or a ``FlatTree`` (default-mode path;
    materialises a Nelson-Aalen leaf table lazily from the persisted raw
    counts and caches it on the tree as ``_chf_leaf_table``).

    Returns
    -------
    chf : ndarray, shape (n_samples, n_causes, n_time_bins), float64
    """
    if X_binned.dtype != np.uint8:
        raise ValueError(f"X_binned must be uint8; got {X_binned.dtype}")
    if X_binned.ndim != 2:
        raise ValueError(f"X_binned must be 2-D; got ndim={X_binned.ndim}")
    if isinstance(tree, FlatTree):
        chf_table = _flat_tree_chf_leaf_table(tree)
        return chf_table[predict_leaf_indices(tree, X_binned)]
    return predict_with_flat(_flatten_tree_hist_chf(tree), X_binned)
