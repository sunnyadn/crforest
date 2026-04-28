"""Integration tests for build_flat_tree_gpu — shape, depth, leaf count
sanity. Determinism gates are in test_gpu_cpu_equivalence.py."""

import numpy as np
import pytest

pytestmark = pytest.mark.gpu


def _make_inputs(n=500, p=8, n_bins=32, n_causes=2, n_time_bins=64, seed=0):
    rng = np.random.default_rng(seed)
    X = rng.integers(0, n_bins, size=(n, p), dtype=np.uint8)
    t_idx = rng.integers(0, n_time_bins, size=n, dtype=np.int32)
    event = rng.integers(0, n_causes + 1, size=n, dtype=np.int32)
    bootstrap = np.arange(n, dtype=np.int32)
    return X, t_idx, event, bootstrap


def test_build_flat_tree_gpu_returns_flat_tree():
    from crforest._gpu_kernels import build_flat_tree_gpu

    X, t_idx, event, bootstrap = _make_inputs()
    flat = build_flat_tree_gpu(
        X,
        t_idx_split=t_idx,
        t_idx_full=t_idx,
        event=event,
        bootstrap_indices=bootstrap,
        n_bins=32,
        n_causes=2,
        n_time_bins_split=64,
        n_time_bins_full=64,
        max_features=4,
        nsplit=10,
        splitrule_code=0,
        cause=1,
        seed=0,
    )
    # Sanity
    assert flat.is_leaf_flags.sum() >= 1
    assert flat.leaf_table.shape == (flat.is_leaf_flags.sum(), 2, 64)


def test_build_flat_tree_gpu_three_runs_identical():
    from crforest._gpu_kernels import build_flat_tree_gpu

    X, t_idx, event, bootstrap = _make_inputs()

    def _run():
        return build_flat_tree_gpu(
            X,
            t_idx_split=t_idx,
            t_idx_full=t_idx,
            event=event,
            bootstrap_indices=bootstrap,
            n_bins=32,
            n_causes=2,
            n_time_bins_split=64,
            n_time_bins_full=64,
            max_features=4,
            nsplit=10,
            splitrule_code=0,
            cause=1,
            seed=42,
        )

    a, b, c = _run(), _run(), _run()
    np.testing.assert_array_equal(a.features, b.features)
    np.testing.assert_array_equal(a.features, c.features)
    np.testing.assert_array_equal(a.leaf_table, b.leaf_table)
    np.testing.assert_array_equal(a.leaf_table, c.leaf_table)


def test_build_flat_tree_gpu_five_runs_bit_identical():
    from crforest._gpu_kernels import build_flat_tree_gpu

    X, t_idx, event, bootstrap = _make_inputs(seed=2)
    runs = []
    for _ in range(5):
        runs.append(
            build_flat_tree_gpu(
                X,
                t_idx_split=t_idx,
                t_idx_full=t_idx,
                event=event,
                bootstrap_indices=bootstrap,
                n_bins=32,
                n_causes=2,
                n_time_bins_split=64,
                n_time_bins_full=64,
                max_features=4,
                nsplit=10,
                splitrule_code=0,
                cause=1,
                seed=42,
            )
        )
    base = runs[0].leaf_table
    for r in runs[1:]:
        assert r.leaf_table.shape == base.shape
        # bit-identical at equal seeds; if a future kernel introduces float atomics, replace with allclose(atol=1e-6)
        np.testing.assert_array_equal(r.leaf_table, base)
