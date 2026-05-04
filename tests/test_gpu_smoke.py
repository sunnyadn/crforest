"""GPU smoke: a single per-tree-level histogram kernel matches a numpy reference."""

import numpy as np
import pytest

pytestmark = pytest.mark.gpu


def test_histogram_kernel_matches_numpy_reference():
    import cupy as cp

    rng = np.random.default_rng(0)
    n_samples = 200
    mtry = 4
    n_bins = 32
    n_causes = 2
    n_time_bins = 16
    N_nodes = 1
    X_binned = rng.integers(0, n_bins, size=(n_samples, mtry), dtype=np.uint8)
    t_idx = rng.integers(0, n_time_bins, size=n_samples, dtype=np.int32)
    event = rng.integers(0, n_causes + 1, size=n_samples, dtype=np.int32)

    # Reference via numpy.
    expected_event = np.zeros((N_nodes, mtry, n_bins, n_causes, n_time_bins), dtype=np.uint32)
    expected_atrisk = np.zeros((N_nodes, mtry, n_bins, n_time_bins), dtype=np.uint32)
    for i in range(n_samples):
        for f in range(mtry):
            b = X_binned[i, f]
            t = t_idx[i]
            ev = event[i]
            if 1 <= ev <= n_causes:
                expected_event[0, f, b, ev - 1, t] += 1
            expected_atrisk[0, f, b, t] += 1

    # Kernel under test.
    from comprisk._gpu_kernels import histogram_kernel_per_level

    Xb_d = cp.asarray(X_binned)
    t_d = cp.asarray(t_idx)
    e_d = cp.asarray(event)
    node_starts = cp.asarray([0], dtype=np.int32)
    node_ends = cp.asarray([n_samples], dtype=np.int32)
    sample_perm = cp.arange(n_samples, dtype=np.int32)

    event_hist = cp.zeros_like(cp.asarray(expected_event))
    n_at = cp.zeros_like(cp.asarray(expected_atrisk))
    histogram_kernel_per_level(
        Xb_d,
        t_d,
        e_d,
        sample_perm,
        node_starts,
        node_ends,
        event_hist,
        n_at,
        n_bins=n_bins,
        n_causes=n_causes,
        n_time_bins=n_time_bins,
        mtry=mtry,
    )

    np.testing.assert_array_equal(cp.asnumpy(event_hist), expected_event)
    np.testing.assert_array_equal(cp.asnumpy(n_at), expected_atrisk)


@pytest.mark.parametrize("splitrule_code", [0, 1])
def test_best_split_kernel_matches_cpu_reference(splitrule_code):
    """GPU best-split scan picks the same (feat, bin) as the existing CPU njit
    kernel (find_best_split_hist_batched) on a tiny fixture."""
    import cupy as cp

    from comprisk._gpu_kernels import (
        best_split_kernel_per_node,
        histogram_kernel_per_level,
    )
    from comprisk._hist_splits import find_best_split_hist_batched

    rng = np.random.default_rng(1)
    n = 200
    mtry = 4
    n_bins = 16
    n_causes = 2
    n_time_bins = 32
    X = rng.integers(0, n_bins, size=(n, mtry), dtype=np.uint8)
    t_idx = rng.integers(0, n_time_bins, size=n, dtype=np.int32)
    event = rng.integers(0, n_causes + 1, size=n, dtype=np.int64)
    cand_mask = np.ones((mtry, n_bins - 1), dtype=np.bool_)

    cpu_feat, cpu_bin, _stat = find_best_split_hist_batched(
        X,
        t_idx,
        event,
        n_bins,
        n_causes,
        n_time_bins,
        10,
        splitrule_code,
        1,
        cand_mask,
    )

    Xb = cp.asarray(X)
    td = cp.asarray(t_idx)
    ed = cp.asarray(event.astype(np.int32))
    perm = cp.arange(n, dtype=np.int32)
    node_starts = cp.asarray([0], dtype=np.int32)
    node_ends = cp.asarray([n], dtype=np.int32)
    ehist = cp.zeros((1, mtry, n_bins, n_causes, n_time_bins), dtype=cp.uint32)
    nat = cp.zeros((1, mtry, n_bins, n_time_bins), dtype=cp.uint32)
    histogram_kernel_per_level(
        Xb,
        td,
        ed,
        perm,
        node_starts,
        node_ends,
        ehist,
        nat,
        n_bins=n_bins,
        n_causes=n_causes,
        n_time_bins=n_time_bins,
        mtry=mtry,
    )

    cm = cp.asarray(cand_mask[None, ...])
    out_feat = cp.full((1,), -1, dtype=cp.int32)
    out_bin = cp.full((1,), -1, dtype=cp.int32)
    out_stat = cp.full((1,), -np.inf, dtype=cp.float64)
    best_split_kernel_per_node(
        ehist,
        nat,
        cm,
        out_feat,
        out_bin,
        out_stat,
        n_bins=n_bins,
        n_causes=n_causes,
        n_time_bins=n_time_bins,
        mtry=mtry,
        min_samples_leaf=10,
        splitrule_code=splitrule_code,
        cause=1,
    )
    assert int(out_feat[0]) == int(cpu_feat)
    assert int(out_bin[0]) == int(cpu_bin)


def test_leafify_kernel_matches_cpu_helper():
    """leafify_kernel_per_level matches _accumulate_leaf_cpu bit-for-bit
    on a multi-leaf batch with mixed event types and a non-trivial
    sample_perm permutation."""
    import cupy as cp

    from comprisk._gpu_kernels import (
        _accumulate_leaf_cpu,
        leafify_kernel_per_level,
    )

    rng = np.random.default_rng(7)
    n_total = 500
    n_causes = 2
    n_time_bins = 32

    t_idx_full = rng.integers(0, n_time_bins, size=n_total, dtype=np.int32)
    event = rng.integers(0, n_causes + 1, size=n_total, dtype=np.int32)
    sample_perm = rng.permutation(n_total).astype(np.int32)

    leafify_specs = [
        (0, 100, 0),
        (100, 250, 1),
        (250, 380, 2),
        (380, 500, 3),
    ]
    N_leaves = 4

    cpu_ec = np.zeros((N_leaves, n_causes, n_time_bins), dtype=np.uint32)
    cpu_ar = np.zeros((N_leaves, n_time_bins), dtype=np.uint32)
    for s, e, li in leafify_specs:
        _accumulate_leaf_cpu(
            sample_perm[s:e],
            t_idx_full,
            event,
            n_causes,
            n_time_bins,
            cpu_ec,
            cpu_ar,
            li,
        )

    gpu_ec = cp.zeros((N_leaves, n_causes, n_time_bins), dtype=cp.uint32)
    gpu_ar = cp.zeros((N_leaves, n_time_bins), dtype=cp.uint32)
    starts = cp.asarray(np.array([s for s, _, _ in leafify_specs], dtype=np.int32))
    ends = cp.asarray(np.array([e for _, e, _ in leafify_specs], dtype=np.int32))
    out_idx = cp.asarray(np.array([li for _, _, li in leafify_specs], dtype=np.int32))
    leafify_kernel_per_level(
        cp.asarray(sample_perm),
        cp.asarray(t_idx_full),
        cp.asarray(event),
        starts,
        ends,
        out_idx,
        gpu_ec,
        gpu_ar,
        n_causes=n_causes,
        n_time_bins=n_time_bins,
    )

    np.testing.assert_array_equal(cp.asnumpy(gpu_ec), cpu_ec)
    np.testing.assert_array_equal(cp.asnumpy(gpu_ar), cpu_ar)


def test_aalen_johansen_batched_matches_cpu_helper():
    """aalen_johansen_from_counts_batched_gpu matches the per-leaf CPU helper
    (within float64 tolerance) on a multi-leaf fixture with mixed at-risk
    profiles including zero-at-risk tail bins."""
    import cupy as cp

    from comprisk._estimators import aalen_johansen_from_counts
    from comprisk._gpu_kernels import aalen_johansen_from_counts_batched_gpu

    rng = np.random.default_rng(11)
    n_leaves = 20
    n_causes = 3
    n_time = 32

    # Build (leaves, causes, time) event counts and reverse-cumsum'd at_risk
    # in the same shape the build_flat_tree_gpu pipeline produces.
    raw_counts = rng.integers(0, 5, size=(n_leaves, n_time), dtype=np.uint32)
    at_risk = np.zeros((n_leaves, n_time), dtype=np.uint32)
    for i in range(n_leaves):
        running = np.uint32(0)
        for t in range(n_time - 1, -1, -1):
            running += raw_counts[i, t]
            at_risk[i, t] = running
    event_counts = rng.integers(0, 4, size=(n_leaves, n_causes, n_time), dtype=np.uint32)
    # Force at_risk smaller than event sum on some bins to keep numerics happy.
    for i in range(n_leaves):
        for t in range(n_time):
            ec_sum = int(event_counts[i, :, t].sum())
            if ec_sum > at_risk[i, t]:
                event_counts[i, :, t] = 0  # zero out causes if implausible

    expected = np.zeros((n_leaves, n_causes, n_time), dtype=np.float64)
    for i in range(n_leaves):
        expected[i] = aalen_johansen_from_counts(event_counts[i], at_risk[i], n_causes)

    got_d = aalen_johansen_from_counts_batched_gpu(
        cp.asarray(event_counts), cp.asarray(at_risk), n_causes
    )
    got = cp.asnumpy(got_d)
    np.testing.assert_allclose(got, expected, rtol=1e-12, atol=1e-12)


def test_partition_kernel_matches_cpu_helper():
    """partition_kernel_per_level produces a sample_perm slice with the same
    multiset as _partition_inplace, and the same mid (split point), on a
    multi-node batch.

    The atomicAdd-based scatter does not preserve ORDER, so we compare
    multisets — that's the invariant the rest of the build_flat_tree_gpu
    pipeline depends on (downstream histograms, leafify, partitions are all
    order-invariant given equal seeds)."""
    import cupy as cp

    from comprisk._gpu_kernels import (
        _partition_inplace,
        partition_kernel_per_level,
    )

    rng = np.random.default_rng(13)
    n = 800
    p = 6
    n_bins = 32
    X = rng.integers(0, n_bins, size=(n, p), dtype=np.uint8)
    sample_perm = rng.permutation(n).astype(np.int32)

    nodes = [
        (0, 200, 2, 16),
        (200, 450, 1, 8),
        (450, 700, 4, 24),
        (700, 800, 0, 12),
    ]
    starts = np.array([s for s, _, _, _ in nodes], dtype=np.int32)
    ends = np.array([e for _, e, _, _ in nodes], dtype=np.int32)
    feats = np.array([f for _, _, f, _ in nodes], dtype=np.int32)
    bins = np.array([b for _, _, _, b in nodes], dtype=np.int32)

    # CPU reference.
    cpu_perm = sample_perm.copy()
    cpu_mids = np.empty(len(nodes), dtype=np.int32)
    for k, (s, e, f, b) in enumerate(nodes):
        cpu_mids[k] = _partition_inplace(cpu_perm, s, e, X, f, b)

    # GPU under test.
    Xd = cp.asarray(X)
    perm_in = cp.asarray(sample_perm)
    perm_out = cp.empty_like(perm_in)
    cp.copyto(perm_out, perm_in)  # pre-fill so unchanged ranges carry through
    starts_d = cp.asarray(starts)
    ends_d = cp.asarray(ends)
    feats_d = cp.asarray(feats)
    bins_d = cp.asarray(bins)
    mids_d = cp.empty(len(nodes), dtype=cp.int32)

    partition_kernel_per_level(
        Xd,
        perm_in,
        perm_out,
        starts_d,
        ends_d,
        feats_d,
        bins_d,
        mids_d,
        p=p,
    )
    gpu_perm = cp.asnumpy(perm_out)
    gpu_mids = cp.asnumpy(mids_d)

    # Mid points must agree exactly (depends on multiset, not order).
    np.testing.assert_array_equal(gpu_mids, cpu_mids)

    # Within each node's [start, mid) and [mid, end), GPU and CPU must agree
    # on the multiset of indices, and (critically) all left-side samples
    # really satisfy X[s, feat] <= bin while right-side do not.
    for k, (s, e, f, b) in enumerate(nodes):
        m = int(gpu_mids[k])
        left_gpu = sorted(gpu_perm[s:m].tolist())
        left_cpu = sorted(cpu_perm[s:m].tolist())
        right_gpu = sorted(gpu_perm[m:e].tolist())
        right_cpu = sorted(cpu_perm[m:e].tolist())
        assert left_gpu == left_cpu, f"node {k}: left multiset diverges"
        assert right_gpu == right_cpu, f"node {k}: right multiset diverges"
        # Predicate check: every left-side sample <= bin, every right-side > bin.
        for i in range(s, m):
            assert int(X[gpu_perm[i], f]) <= b, f"node {k}: left predicate fails at {i}"
        for i in range(m, e):
            assert int(X[gpu_perm[i], f]) > b, f"node {k}: right predicate fails at {i}"


def test_best_split_kernel_rejects_oversized_inputs():
    """Runtime guards reject n_causes / n_time_bins beyond the device-helper buffer caps."""
    import cupy as cp

    from comprisk._gpu_kernels import (
        MAX_GPU_CAUSES,
        MAX_GPU_TIME_BINS,
        best_split_kernel_per_node,
    )

    N_nodes, mtry, n_bins = 1, 2, 4
    # We don't actually need real histograms — the guard runs before launch.
    ehist = cp.zeros((N_nodes, mtry, n_bins, MAX_GPU_CAUSES + 1, 8), dtype=cp.uint32)
    nat = cp.zeros((N_nodes, mtry, n_bins, 8), dtype=cp.uint32)
    cm = cp.zeros((N_nodes, mtry, n_bins - 1), dtype=cp.bool_)
    out_feat = cp.full((N_nodes,), -1, dtype=cp.int32)
    out_bin = cp.full((N_nodes,), -1, dtype=cp.int32)
    out_stat = cp.full((N_nodes,), -np.inf, dtype=cp.float64)
    with pytest.raises(ValueError, match="MAX_GPU_CAUSES"):
        best_split_kernel_per_node(
            ehist,
            nat,
            cm,
            out_feat,
            out_bin,
            out_stat,
            n_bins=n_bins,
            n_causes=MAX_GPU_CAUSES + 1,
            n_time_bins=8,
            mtry=mtry,
            min_samples_leaf=10,
            splitrule_code=0,
            cause=1,
        )

    ehist = cp.zeros((N_nodes, mtry, n_bins, 2, MAX_GPU_TIME_BINS + 1), dtype=cp.uint32)
    nat = cp.zeros((N_nodes, mtry, n_bins, MAX_GPU_TIME_BINS + 1), dtype=cp.uint32)
    with pytest.raises(ValueError, match="MAX_GPU_TIME_BINS"):
        best_split_kernel_per_node(
            ehist,
            nat,
            cm,
            out_feat,
            out_bin,
            out_stat,
            n_bins=n_bins,
            n_causes=2,
            n_time_bins=MAX_GPU_TIME_BINS + 1,
            mtry=mtry,
            min_samples_leaf=10,
            splitrule_code=0,
            cause=1,
        )
