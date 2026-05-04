"""Permutation variable-importance (VIMP) for competing-risks forests.

This module implements two flavours of permutation VIMP:

* **Held-out VIMP** (:func:`_compute_importance_impl`) -- the user supplies an
  evaluation set ``(X_eval, y_eval)`` and we delegate per-cause permutation
  scoring to :func:`sklearn.inspection.permutation_importance`. The scorer
  used is the cause-specific (Wolbers) concordance index.

* **Out-of-bag VIMP** (:func:`_compute_importance_oob_impl`) -- a single
  ensemble-level permutation pass per feature, scored with the Uno IPCW
  cause-specific concordance on the cached training data. Out-of-bag rows
  are aggregated across trees, so each feature is permuted exactly once
  per tree (no per-repeat resampling).

References
----------
Breiman, L. (2001). "Random forests." *Machine Learning* 45(1): 5-32. --
defines OOB permutation importance.

Ishwaran, H., Kogalur, U.B., Blackstone, E.H., Lauer, M.S. (2008).
"Random survival forests." *Annals of Applied Statistics* 2(3): 841-860.
-- random-survival-forest extension of Breiman's algorithm.

Ishwaran, H., Gerds, T.A., Kogalur, U.B., Moore, R.D., Gange, S.J.,
Lau, B.M. (2014). "Random survival forests for competing risks."
*Biostatistics* 15(4): 757-773. -- competing-risks integrated-CIF
"mortality" used as the per-tree risk score.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from sklearn.inspection import permutation_importance

from comprisk.metrics import (
    compute_uno_weights,
    concordance_index_cr,
    concordance_index_uno_cr,
)

__all__ = [
    "_assemble_df",
    "_compute_importance_impl",
    "_compute_importance_oob_impl",
    "_derive_perm_seeds",
    "_ensemble_oob_predictions",
    "_make_cause_scorer",
    "_predict_tree_mortality",
    "_weighted_mean",
]


# ---------------------------------------------------------------------------
# Small helpers
# ---------------------------------------------------------------------------


def _assemble_df(
    feature_names: list[str],
    per_cause: dict[int, np.ndarray],
    composite: np.ndarray,
) -> pd.DataFrame:
    """Build the canonical VIMP DataFrame.

    Columns are produced in this exact order::

        ["feature", "cause_{k}_vimp" for k in sorted(per_cause), "composite_vimp"]

    The cause keys are emitted in numeric (sorted) order, regardless of the
    insertion order of ``per_cause``.
    """
    sorted_causes = sorted(per_cause)
    data: dict[str, np.ndarray | list[str]] = {"feature": list(feature_names)}
    for k in sorted_causes:
        data[f"cause_{k}_vimp"] = np.asarray(per_cause[k])
    data["composite_vimp"] = np.asarray(composite)
    return pd.DataFrame(data)


def _weighted_mean(
    per_cause: dict[int, np.ndarray],
    weights: np.ndarray | None,
) -> np.ndarray:
    """Weighted mean across causes, with causes stacked in numeric order.

    Parameters
    ----------
    per_cause
        Dict mapping cause code to a 1-D vector (one entry per feature).
    weights
        Per-cause weights. If ``None``, uniform ``1/K`` weights are used.
        Otherwise must have shape ``(K,)`` and ``sum > 0``; the weights are
        normalised to sum to 1 before averaging.

    Returns
    -------
    np.ndarray
        Length-``n_features`` 1-D array of per-feature weighted means.
    """
    sorted_causes = sorted(per_cause)
    stacked = np.stack([np.asarray(per_cause[k]) for k in sorted_causes], axis=0)
    K = stacked.shape[0]
    if weights is None:
        w = np.full(K, 1.0 / K, dtype=np.float64)
    else:
        w = np.asarray(weights, dtype=np.float64)
        if w.shape != (K,):
            raise ValueError(f"cause_weights has shape {w.shape}; expected ({K},) for K={K} causes")
        total = float(w.sum())
        if total <= 0:
            raise ValueError(f"cause_weights must sum to a positive value; got sum={total}")
        w = w / total
    return (w[:, None] * stacked).sum(axis=0)


def _make_cause_scorer(cause: int):
    """Return an sklearn-compatible scorer for the given cause.

    The closure expects ``y`` to be a structured array with ``time`` and
    ``event`` fields (the standard comprisk survival outcome representation).
    Scoring is done with the Wolbers cause-specific concordance index.
    """

    def score(estimator, X, y) -> float:
        time = y["time"]
        event = y["event"]
        risk = estimator.predict_risk(X, cause=cause)
        return concordance_index_cr(event, time, risk, cause=cause)

    return score


# ---------------------------------------------------------------------------
# Held-out VIMP
# ---------------------------------------------------------------------------


def _compute_importance_impl(
    estimator,
    X_eval,
    y_eval,
    *,
    feature_names: list[str],
    causes: list[int],
    cause_weights: np.ndarray | None,
    n_repeats: int,
    random_state: int | None,
) -> pd.DataFrame:
    """Held-out permutation VIMP, one outer ``permutation_importance`` per cause.

    See module docstring for references. The inner ``n_jobs=1`` is intentional:
    ``estimator.predict_risk`` already parallelises across trees via joblib,
    and nesting sklearn's process pool over that both oversubscribes the
    machine and breaks the bit-equivalence-under-``n_jobs`` invariant the
    test suite relies on.
    """
    vimp_by_cause: dict[int, np.ndarray] = {}
    for cause_k in causes:
        scorer_k = _make_cause_scorer(cause=cause_k)
        imp_result = permutation_importance(
            estimator,
            X_eval,
            y_eval,
            scoring=scorer_k,
            n_repeats=n_repeats,
            random_state=random_state,
            n_jobs=1,
        )
        vimp_by_cause[cause_k] = imp_result.importances_mean
    vimp_composite = _weighted_mean(vimp_by_cause, weights=cause_weights)
    return _assemble_df(feature_names, vimp_by_cause, vimp_composite)


# ---------------------------------------------------------------------------
# OOB VIMP -- per-tree mortality + ensemble OOB aggregation
# ---------------------------------------------------------------------------


def _derive_perm_seeds(random_state: int | None, n_trees: int, n_features: int) -> np.ndarray:
    """Deterministic ``(n_trees, n_features)`` int64 seed matrix.

    Seeds are drawn upfront from a single ``RandomState`` so the OOB VIMP is
    identical regardless of how the per-feature work is parallelised.
    """
    rng = np.random.RandomState(random_state)
    return rng.randint(0, np.iinfo(np.int32).max, size=(n_trees, n_features), dtype=np.int64)


def _predict_tree_mortality(
    tree,
    X,
    *,
    cause: int,
    mode: str,
    bin_edges,
    time_grid: np.ndarray,
) -> np.ndarray:
    """Single-tree integrated CIF "mortality" for one cause.

    Implements the discrete left-Riemann integral of the cause-``cause`` CIF
    over ``time_grid`` (Ishwaran et al., 2014)::

        mortality(s) = sum_{q = 0..T-2} CIF[k, q] * (time_grid[q+1] - time_grid[q])

    For ``mode="default"`` the input is binned with ``bin_edges`` before
    being walked through the histogram tree; for ``mode="reference"`` the
    raw float matrix is walked through the reference tree.
    """
    X = np.asarray(X, dtype=np.float64)
    if mode == "default":
        if bin_edges is None:
            raise ValueError("bin_edges required for mode='default'")
        from comprisk._binning import apply_bins
        from comprisk._hist_tree import predict_tree_hist

        X_binned = apply_bins(X, bin_edges)
        cif = predict_tree_hist(tree, X_binned)  # (n, n_causes, n_time_bins)
    elif mode == "reference":
        from comprisk._tree import predict_tree

        cif = predict_tree(tree, X)
    else:
        raise ValueError(f"unknown mode {mode!r}")
    dt = np.diff(time_grid)
    if len(dt) == 0:
        return cif[:, cause - 1, 0]
    return (cif[:, cause - 1, : len(dt)] * dt).sum(axis=-1)


def _ensemble_oob_predictions(
    forest,
    X_train,
    causes: list[int],
    bin_edges,
    time_grid: np.ndarray,
    *,
    feature: int | None = None,
    feature_seeds: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Sum of per-tree OOB mortality predictions, one cause per row.

    Parameters
    ----------
    forest
        Fitted ``CompetingRiskForest``; provides ``trees_``, ``oob_indices_``,
        and ``mode``.
    X_train
        The training feature matrix the forest was fit on.
    causes
        Causes to score, in output order.
    bin_edges, time_grid
        As produced during ``fit``; ``bin_edges`` is ``None`` for
        ``mode="reference"``.
    feature
        If given, the column index of the feature to permute within each
        tree's OOB block. ``None`` runs the unpermuted reference pass.
    feature_seeds
        Length-``n_trees`` int64 seeds used to seed the per-tree permutation
        RNG. Required when ``feature`` is not ``None``.

    Returns
    -------
    pred : ndarray, shape ``(n_causes, n_train)``
        Sum (not mean) of per-tree mortality for each (cause, train sample).
    count : ndarray, shape ``(n_train,)``
        Number of trees for which each train sample was out-of-bag.
    """
    n_train = X_train.shape[0]
    n_causes = len(causes)
    pred = np.zeros((n_causes, n_train), dtype=np.float64)
    count = np.zeros(n_train, dtype=np.int64)
    n_trees = len(forest.trees_)
    for t in range(n_trees):
        oob = forest.oob_indices_[t]
        if len(oob) == 0:
            continue
        if feature is None:
            X_input = X_train[oob]
        else:
            rng_tf = np.random.RandomState(int(feature_seeds[t]))
            perm = rng_tf.permutation(len(oob))
            X_input = X_train[oob].copy()
            X_input[:, feature] = X_train[oob][perm, feature]
        for ci, c in enumerate(causes):
            risk_t = _predict_tree_mortality(
                forest.trees_[t],
                X_input,
                cause=c,
                mode=forest.mode,
                bin_edges=bin_edges,
                time_grid=time_grid,
            )
            pred[ci, oob] += risk_t
        count[oob] += 1
    return pred, count


def _compute_importance_oob_impl(
    forest,
    *,
    causes: list[int],
    random_state: int | None,
    n_jobs: int | None,
) -> pd.DataFrame:
    """Out-of-bag permutation VIMP scored with Uno IPCW concordance.

    Algorithm (Breiman 2001; Ishwaran 2008; Ishwaran 2014):

    1. Build the unpermuted ensemble OOB mortality for each cause and record
       the Uno IPCW C-index on retained rows. This is the reference score.
    2. For each feature, permute that feature inside every tree's OOB block
       (with a deterministic per-(tree, feature) seed), accumulate the
       ensemble OOB mortality, score with Uno IPCW, and record
       ``vimp = ref_C - perm_C``.
    3. Compose the per-cause vectors with ``forest._cause_weights_arr`` to
       get the composite column.

    The deterministic seed matrix is drawn upfront from ``random_state`` so
    the result is invariant to ``n_jobs``. Per-feature work is parallelised
    with a thread pool because the inner predict layer is itself joblib-
    parallel; nesting two process pools would oversubscribe the machine.
    """
    # Bootstrap-only path: OOB scoring needs an out-of-bag set per tree.
    if not forest.bootstrap:
        raise ValueError("OOB importance scoring needs bootstrap=True at fit time")
    cached_X = getattr(forest, "_X_train_oob_", None)
    if cached_X is None:
        raise RuntimeError("forest has no cached training data; refit with a current build")

    X_train = cached_X
    cached_y = forest._y_train_oob_
    time_train = cached_y["time"]
    event_train = cached_y["event"]
    n_causes = len(causes)
    n_trees = len(forest.trees_)
    n_features = X_train.shape[1]

    bin_edges = getattr(forest, "bin_edges_", None)
    if forest.mode == "default":
        time_grid = np.asarray(forest.time_grid_, dtype=np.float64)
    else:
        time_grid = np.asarray(forest.unique_times_, dtype=np.float64)

    w_ipcw = compute_uno_weights(time_train, event_train)

    # Reference (unpermuted) OOB pass: aggregate per-tree mortality across
    # the trees where each training sample is OOB, then normalise.
    pred_ref, n_ref = _ensemble_oob_predictions(forest, X_train, causes, bin_edges, time_grid)
    has_ref = n_ref > 0
    if not has_ref.any():
        raise RuntimeError("no out-of-bag samples produced; cannot compute OOB VIMP")
    pred_ref[:, has_ref] /= n_ref[has_ref]
    w_ref = w_ipcw[has_ref]
    c_baseline = np.empty(n_causes, dtype=np.float64)
    for ci, cause_k in enumerate(causes):
        c_baseline[ci] = concordance_index_uno_cr(
            event_train[has_ref],
            time_train[has_ref],
            pred_ref[ci, has_ref],
            cause=cause_k,
            weights=w_ref,
        )

    # Per-feature permutation pass. Seeds are materialised once so the
    # outer parallel loop is order-independent.
    tree_perm_seeds = _derive_perm_seeds(random_state, n_trees, n_features)
    effective_n_jobs = forest.n_jobs if n_jobs is None else n_jobs

    def per_feature_delta(f_idx: int) -> np.ndarray:
        pred_perm, n_perm = _ensemble_oob_predictions(
            forest,
            X_train,
            causes,
            bin_edges,
            time_grid,
            feature=f_idx,
            feature_seeds=tree_perm_seeds[:, f_idx],
        )
        has_perm = n_perm > 0
        out_f = np.full(n_causes, np.nan, dtype=np.float64)
        if not has_perm.any():
            return out_f
        pred_perm[:, has_perm] /= n_perm[has_perm]
        w_perm = w_ipcw[has_perm]
        for ci, cause_k in enumerate(causes):
            c_after = concordance_index_uno_cr(
                event_train[has_perm],
                time_train[has_perm],
                pred_perm[ci, has_perm],
                cause=cause_k,
                weights=w_perm,
            )
            out_f[ci] = c_baseline[ci] - c_after
        return out_f

    vimp_per_feature = Parallel(n_jobs=effective_n_jobs, prefer="threads")(
        delayed(per_feature_delta)(f_idx) for f_idx in range(n_features)
    )
    vimp_matrix = np.stack(vimp_per_feature, axis=0)  # shape (n_features, n_causes)

    per_cause_dict = {causes[ci]: vimp_matrix[:, ci] for ci in range(n_causes)}
    composite_vec = _weighted_mean(per_cause_dict, weights=forest._cause_weights_arr)
    feat_names = forest._importance_feature_names()
    return _assemble_df(feat_names, per_cause_dict, composite_vec)
