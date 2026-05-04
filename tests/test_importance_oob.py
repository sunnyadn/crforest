"""Unit tests for OOB-mode permutation VIMP."""

from __future__ import annotations

import numpy as np
import pytest

from comprisk import CompetingRiskForest


def _toy(n=200, p=4, seed=0, n_causes=2):
    rng = np.random.RandomState(seed)
    X = rng.randn(n, p)
    time = rng.uniform(0.1, 10, n)
    event = rng.randint(0, n_causes + 1, n).astype(np.int64)
    return X, time, event


def test_derive_perm_seeds_shape_and_determinism():
    from comprisk._importance import _derive_perm_seeds

    seeds_a = _derive_perm_seeds(random_state=42, n_trees=5, n_features=7)
    seeds_b = _derive_perm_seeds(random_state=42, n_trees=5, n_features=7)
    np.testing.assert_array_equal(seeds_a, seeds_b)
    assert seeds_a.shape == (5, 7)
    assert seeds_a.dtype == np.int64
    seeds_c = _derive_perm_seeds(random_state=43, n_trees=5, n_features=7)
    assert not np.array_equal(seeds_a, seeds_c)


def test_derive_perm_seeds_none_random_state():
    from comprisk._importance import _derive_perm_seeds

    seeds = _derive_perm_seeds(random_state=None, n_trees=3, n_features=4)
    assert seeds.shape == (3, 4)
    assert seeds.dtype == np.int64


def _naive_oob_vimp(forest, *, cause: int, random_state: int) -> np.ndarray:
    """Obvious-by-inspection reference: per-feature OOB Breiman permutation VIMP.

    Implements step-by-step the algorithm described in
    ``randomForestSRC/src/importance.c`` + ``importancePerm.c`` +
    ``survival.c:413-417``:

      1. For each (tree, feature), permute that feature within tree's OOB
         samples using a per-(tree, feature) RNG seed.
      2. For each tree-OOB sample, get terminal-leaf mortality (integrated
         CIF over time grid for cause `cause`).
      3. Per-sample ensemble mortality = mean over trees-where-OOB.
      4. C-index of (time, event, ref_mortality) and (time, event, perm_mortality).
      5. VIMP = ref_C - perm_C.

    No parallelism, no shared computation, no clever indexing — by-the-book.
    Returns shape ``(n_features,)``.
    """
    from comprisk._importance import (
        _derive_perm_seeds,
        _predict_tree_mortality,
    )
    from comprisk.metrics import compute_uno_weights, concordance_index_uno_cr

    X = forest._X_train_oob_
    time = forest._y_train_oob_["time"]
    event = forest._y_train_oob_["event"]
    n_train, n_features = X.shape
    n_trees = len(forest.trees_)
    bin_edges = forest.bin_edges_ if forest.mode == "default" else None
    time_grid = forest.time_grid_ if forest.mode == "default" else forest.unique_times_
    seeds = _derive_perm_seeds(random_state, n_trees, n_features)

    uno_weights = compute_uno_weights(time, event)

    # Reference ensemble mortality
    ref_pred_sum = np.zeros(n_train)
    counts = np.zeros(n_train, dtype=np.int64)
    for t in range(n_trees):
        oob = forest.oob_indices_[t]
        if len(oob) == 0:
            continue
        m = _predict_tree_mortality(
            forest.trees_[t],
            X[oob],
            cause=cause,
            mode=forest.mode,
            bin_edges=bin_edges,
            time_grid=time_grid,
        )
        ref_pred_sum[oob] += m
        counts[oob] += 1
    mask = counts > 0
    ref_pred = np.zeros(n_train)
    ref_pred[mask] = ref_pred_sum[mask] / counts[mask]
    ref_C = concordance_index_uno_cr(
        event[mask],
        time[mask],
        ref_pred[mask],
        cause=cause,
        weights=uno_weights[mask],
    )

    # Per-feature permuted ensemble mortality
    out = np.zeros(n_features)
    for f in range(n_features):
        perm_pred_sum = np.zeros(n_train)
        perm_counts = np.zeros(n_train, dtype=np.int64)
        for t in range(n_trees):
            oob = forest.oob_indices_[t]
            if len(oob) == 0:
                continue
            rng = np.random.RandomState(int(seeds[t, f]))
            perm_idx = rng.permutation(len(oob))
            X_perm = X[oob].copy()
            X_perm[:, f] = X[oob][perm_idx, f]
            m = _predict_tree_mortality(
                forest.trees_[t],
                X_perm,
                cause=cause,
                mode=forest.mode,
                bin_edges=bin_edges,
                time_grid=time_grid,
            )
            perm_pred_sum[oob] += m
            perm_counts[oob] += 1
        local_mask = perm_counts > 0
        perm_pred = np.zeros(n_train)
        perm_pred[local_mask] = perm_pred_sum[local_mask] / perm_counts[local_mask]
        perm_C = concordance_index_uno_cr(
            event[local_mask],
            time[local_mask],
            perm_pred[local_mask],
            cause=cause,
            weights=uno_weights[local_mask],
        )
        out[f] = ref_C - perm_C
    return out


def test_oob_vimp_matches_naive_reference_implementation():
    """Implementation correctness gate: prod impl must match an obvious-by-inspection
    naive reference within float epsilon. This is independent of rfSRC; it proves
    no implementation bug between forest and the algorithm described in the spec."""
    X, t, e = _toy(n=200, p=4, seed=1)
    forest = CompetingRiskForest(n_estimators=15, random_state=42).fit(X, t, e)
    df = forest.compute_importance(random_state=99)
    naive_c1 = _naive_oob_vimp(forest, cause=1, random_state=99)
    naive_c2 = _naive_oob_vimp(forest, cause=2, random_state=99)
    np.testing.assert_allclose(df["cause_1_vimp"].to_numpy(), naive_c1, atol=1e-12, rtol=0)
    np.testing.assert_allclose(df["cause_2_vimp"].to_numpy(), naive_c2, atol=1e-12, rtol=0)


def test_oob_constant_feature_yields_zero_vimp():
    """Permuting a constant column gives identical predictions => vimp = 0."""
    X, t, e = _toy(n=200, p=4)
    X[:, 2] = 1.0  # column 2 is constant
    forest = CompetingRiskForest(n_estimators=10, random_state=42).fit(X, t, e)
    df = forest.compute_importance(random_state=42)
    assert df.iloc[2]["cause_1_vimp"] == 0.0
    assert df.iloc[2]["cause_2_vimp"] == 0.0


def test_compute_importance_oob_impl_returns_dataframe():
    from comprisk._importance import _compute_importance_oob_impl

    X, t, e = _toy(n=200, p=4)
    forest = CompetingRiskForest(n_estimators=10, random_state=42).fit(X, t, e)
    forest._X_train_oob_ = X
    y_struct = np.zeros(len(t), dtype=[("time", np.float64), ("event", np.int64)])
    y_struct["time"] = t
    y_struct["event"] = e
    forest._y_train_oob_ = y_struct
    df = _compute_importance_oob_impl(
        forest,
        causes=[1, 2],
        random_state=42,
        n_jobs=1,
    )
    assert "feature" in df.columns
    assert "cause_1_vimp" in df.columns
    assert "cause_2_vimp" in df.columns
    assert "composite_vimp" in df.columns
    assert len(df) == 4


def test_compute_importance_oob_impl_n_jobs_bit_equivalent():
    from comprisk._importance import _compute_importance_oob_impl

    X, t, e = _toy(n=200, p=4)
    forest = CompetingRiskForest(n_estimators=10, random_state=42).fit(X, t, e)
    forest._X_train_oob_ = X
    y_struct = np.zeros(len(t), dtype=[("time", np.float64), ("event", np.int64)])
    y_struct["time"] = t
    y_struct["event"] = e
    forest._y_train_oob_ = y_struct
    df1 = _compute_importance_oob_impl(forest, causes=[1, 2], random_state=42, n_jobs=1)
    dfn = _compute_importance_oob_impl(forest, causes=[1, 2], random_state=42, n_jobs=-1)
    for col in ("cause_1_vimp", "cause_2_vimp", "composite_vimp"):
        np.testing.assert_array_equal(df1[col].to_numpy(), dfn[col].to_numpy())


def test_compute_importance_no_args_routes_to_oob():
    X, t, e = _toy(n=200, p=4)
    forest = CompetingRiskForest(n_estimators=10, random_state=42).fit(X, t, e)
    df = forest.compute_importance(random_state=42)
    assert len(df) == 4
    assert "cause_1_vimp" in df.columns
    assert "composite_vimp" in df.columns


def test_compute_importance_oob_requires_bootstrap_true():
    X, t, e = _toy(n=200, p=4)
    forest = CompetingRiskForest(n_estimators=10, random_state=42, bootstrap=False).fit(X, t, e)
    with pytest.raises(ValueError, match="bootstrap=True"):
        forest.compute_importance(random_state=42)


def test_compute_importance_partial_args_raise():
    X, t, e = _toy(n=200, p=4)
    forest = CompetingRiskForest(n_estimators=5, random_state=42).fit(X, t, e)
    y_eval = np.zeros(len(t), dtype=[("time", np.float64), ("event", np.int64)])
    y_eval["time"] = t
    y_eval["event"] = e
    with pytest.raises(ValueError, match="X_eval"):
        forest.compute_importance(X_eval=None, y_eval=y_eval)
    with pytest.raises(ValueError, match="y_eval"):
        forest.compute_importance(X_eval=X, y_eval=None)


def test_compute_importance_oob_n_repeats_silently_ignored():
    X, t, e = _toy(n=200, p=4)
    forest = CompetingRiskForest(n_estimators=5, random_state=42).fit(X, t, e)
    df_default = forest.compute_importance(random_state=42)
    df_with_nrep = forest.compute_importance(random_state=42, n_repeats=99)
    np.testing.assert_array_equal(
        df_default["cause_1_vimp"].to_numpy(),
        df_with_nrep["cause_1_vimp"].to_numpy(),
    )


def test_held_out_path_unchanged():
    X, t, e = _toy(n=200, p=4)
    forest = CompetingRiskForest(n_estimators=5, random_state=42).fit(X, t, e)
    y_eval = np.zeros(len(t), dtype=[("time", np.float64), ("event", np.int64)])
    y_eval["time"] = t
    y_eval["event"] = e
    df = forest.compute_importance(X, y_eval, random_state=42, n_repeats=2)
    assert len(df) == 4
    assert "cause_1_vimp" in df.columns
