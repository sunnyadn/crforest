"""λ.exp2 — Phase A1 POC: GPU kernel for splittable classification.

Replaces the host-side Python loop at _gpu_kernels.py:1182-1193 with a GPU
kernel that:
  per node i in [0, N_active):
    f_sel = out_feat[i]
    failed = (f_sel < 0)
    rank   = inclusive prefix sum of (1 - failed) at i  (rank among
             non-failed nodes)
    is_partition = (not failed) and (rank <= slots_left)
    is_leaf      = (failed) or (rank > slots_left)
    if is_partition:
        actual_feat[i] = feat_perm[i, f_sel]

Then stream-compaction picks (partition_tasks_d, leaf_batch_d).

This kills 2 of the 4 per-level .get() calls measured in λ.exp1
(out_feat_h + out_bin_h) — they no longer need to be transferred to host
because the classification + actual-feat lookup happens device-side.

POC scope: validate the kernel against the host loop on synthetic inputs
covering all branch combinations (no failed, all failed, partial,
slots_left exhausted mid-list). NOT integrated into build_flat_tree_gpu
yet — that's Phase A2/A3.

Run: ssh win 'export PATH=$HOME/.local/bin:$PATH && cd ~/comprisk && \\
       PYTHONUNBUFFERED=1 uv run --extra gpu --extra dev \\
       python -u validation/spikes/lambda/exp2_classify_splittable_poc.py'
"""

from __future__ import annotations

import sys

import numpy as np


def reference_host_loop(
    out_feat_h: np.ndarray,
    out_bin_h: np.ndarray,
    feat_perm: np.ndarray,
    slots_left: int,
    n_leaves_used_init: int,
) -> dict:
    """Faithful copy of _gpu_kernels.py:1182-1193 host loop."""
    N_active = len(out_feat_h)
    n_leaves_used = n_leaves_used_init
    partition_tasks = []  # (i, actual_feat, bin_idx)
    leaf_batch = []  # (i, leaf_idx)
    for i in range(N_active):
        f_sel = int(out_feat_h[i])
        bin_idx = int(out_bin_h[i])
        if f_sel < 0 or len(partition_tasks) >= slots_left:
            leaf_batch.append((i, n_leaves_used))
            n_leaves_used += 1
            continue
        actual_feat = int(feat_perm[i, f_sel])
        partition_tasks.append((i, actual_feat, bin_idx))
    return {
        "partition_tasks": partition_tasks,
        "leaf_batch": leaf_batch,
        "n_leaves_used_after": n_leaves_used,
    }


def gpu_classify(
    out_feat_d,
    out_bin_d,
    feat_perm_d,
    slots_left: int,
    n_leaves_used_init: int,
):
    """GPU classify + compact. Returns device arrays + scalars."""
    import cupy as cp

    N_active = int(out_feat_d.size)
    if N_active == 0:
        empty_i32 = cp.empty(0, dtype=cp.int32)
        return {
            "partition_idx_d": empty_i32,
            "partition_actual_feat_d": empty_i32,
            "partition_bin_d": empty_i32,
            "leaf_idx_in_active_d": empty_i32,
            "leaf_assigned_idx_d": empty_i32,
            "n_leaves_used_after": int(n_leaves_used_init),
        }

    # Step 1: per-node failed mask + non-failed rank (inclusive prefix sum).
    failed_d = (out_feat_d < 0).astype(cp.int32)
    nonfailed_d = (1 - failed_d).astype(cp.int32)
    rank_d = cp.cumsum(nonfailed_d, dtype=cp.int32)  # inclusive; first non-failed gets rank 1

    is_partition_d = (failed_d == 0) & (rank_d <= cp.int32(slots_left))
    is_leaf_d = ~is_partition_d

    # Step 2: compact partition rows.
    part_idx_d = cp.where(is_partition_d)[0].astype(cp.int32)
    part_bin_d = out_bin_d[part_idx_d]
    # actual_feat[i] = feat_perm[i, f_sel] — gather along axis 1.
    part_fsel_d = out_feat_d[part_idx_d].astype(cp.int64)
    part_rows_d = part_idx_d.astype(cp.int64)
    part_actual_feat_d = feat_perm_d[part_rows_d, part_fsel_d].astype(cp.int32)

    # Step 3: compact leaf rows + assign sequential leaf indices.
    leaf_idx_d = cp.where(is_leaf_d)[0].astype(cp.int32)
    n_leaves_added = int(leaf_idx_d.size)
    leaf_assigned_d = cp.arange(n_leaves_added, dtype=cp.int32) + cp.int32(n_leaves_used_init)

    return {
        "partition_idx_d": part_idx_d,
        "partition_actual_feat_d": part_actual_feat_d,
        "partition_bin_d": part_bin_d,
        "leaf_idx_in_active_d": leaf_idx_d,
        "leaf_assigned_idx_d": leaf_assigned_d,
        "n_leaves_used_after": n_leaves_used_init + n_leaves_added,
    }


def compare(label: str, ref: dict, gpu: dict) -> bool:
    import cupy as cp

    ok = True

    # 1) partition tasks: same i ordering, same actual_feat, same bin_idx.
    ref_part_i = np.array([p[0] for p in ref["partition_tasks"]], dtype=np.int32)
    ref_part_af = np.array([p[1] for p in ref["partition_tasks"]], dtype=np.int32)
    ref_part_bin = np.array([p[2] for p in ref["partition_tasks"]], dtype=np.int32)

    gpu_part_i = cp.asnumpy(gpu["partition_idx_d"])
    gpu_part_af = cp.asnumpy(gpu["partition_actual_feat_d"])
    gpu_part_bin = cp.asnumpy(gpu["partition_bin_d"])

    if not np.array_equal(ref_part_i, gpu_part_i):
        print(f"  [{label}] FAIL partition_idx: ref={ref_part_i} gpu={gpu_part_i}")
        ok = False
    if not np.array_equal(ref_part_af, gpu_part_af):
        print(f"  [{label}] FAIL partition_actual_feat: ref={ref_part_af} gpu={gpu_part_af}")
        ok = False
    if not np.array_equal(ref_part_bin, gpu_part_bin):
        print(f"  [{label}] FAIL partition_bin: ref={ref_part_bin} gpu={gpu_part_bin}")
        ok = False

    # 2) leaf batch: same i ordering, same assigned leaf_idx.
    ref_leaf_i = np.array([p[0] for p in ref["leaf_batch"]], dtype=np.int32)
    ref_leaf_assign = np.array([p[1] for p in ref["leaf_batch"]], dtype=np.int32)

    gpu_leaf_i = cp.asnumpy(gpu["leaf_idx_in_active_d"])
    gpu_leaf_assign = cp.asnumpy(gpu["leaf_assigned_idx_d"])

    if not np.array_equal(ref_leaf_i, gpu_leaf_i):
        print(f"  [{label}] FAIL leaf_idx: ref={ref_leaf_i} gpu={gpu_leaf_i}")
        ok = False
    if not np.array_equal(ref_leaf_assign, gpu_leaf_assign):
        print(f"  [{label}] FAIL leaf_assigned: ref={ref_leaf_assign} gpu={gpu_leaf_assign}")
        ok = False

    # 3) final n_leaves_used.
    if ref["n_leaves_used_after"] != gpu["n_leaves_used_after"]:
        print(
            f"  [{label}] FAIL n_leaves_used_after: ref={ref['n_leaves_used_after']} "
            f"gpu={gpu['n_leaves_used_after']}"
        )
        ok = False

    if ok:
        print(
            f"  [{label}] OK  ({len(ref['partition_tasks'])} partitions, "
            f"{len(ref['leaf_batch'])} leaves)"
        )
    return ok


def make_case(name: str, n: int, mtry: int, *, n_failed: int, slots_left: int, seed: int):
    rng = np.random.default_rng(seed)
    out_feat = rng.integers(0, mtry, size=n, dtype=np.int32)
    fail_idx = rng.choice(n, size=n_failed, replace=False)
    out_feat[fail_idx] = -1
    out_bin = rng.integers(0, 32, size=n, dtype=np.int32)
    feat_perm = rng.integers(0, 100, size=(n, mtry), dtype=np.int32)
    return {
        "name": name,
        "out_feat": out_feat,
        "out_bin": out_bin,
        "feat_perm": feat_perm,
        "slots_left": slots_left,
        "n_leaves_used_init": rng.integers(0, 50),
    }


def main() -> None:
    try:
        import cupy as cp
    except ImportError:
        print("cupy not available — skip POC", file=sys.stderr)
        sys.exit(2)

    cases = [
        # No failures, plenty of slots — all partition.
        make_case("all-partition", n=8, mtry=8, n_failed=0, slots_left=20, seed=0),
        # All failures — all leaf.
        make_case("all-failed", n=8, mtry=8, n_failed=8, slots_left=20, seed=1),
        # Mixed, slots ample — failed are leaves, rest partition.
        make_case("mixed-ample", n=10, mtry=8, n_failed=3, slots_left=20, seed=2),
        # Mixed, slots scarce — failed + overflow are leaves.
        make_case("slots-exhausted", n=12, mtry=8, n_failed=2, slots_left=4, seed=3),
        # Slots = 0 — all should leaf even with no failures.
        make_case("slots-zero", n=6, mtry=8, n_failed=0, slots_left=0, seed=4),
        # Edge: N_active = 1, single partition.
        make_case("single-partition", n=1, mtry=4, n_failed=0, slots_left=10, seed=5),
        # Edge: N_active = 1, single failed.
        make_case("single-failed", n=1, mtry=4, n_failed=1, slots_left=10, seed=6),
        # Larger realistic mid-tree level.
        make_case("realistic-level", n=64, mtry=8, n_failed=10, slots_left=128, seed=7),
        # Slots barely fit non-failed.
        make_case("exact-fit", n=16, mtry=8, n_failed=4, slots_left=12, seed=8),
    ]

    print(f"Running {len(cases)} cases...\n")
    all_ok = True
    for c in cases:
        ref = reference_host_loop(
            c["out_feat"],
            c["out_bin"],
            c["feat_perm"],
            c["slots_left"],
            c["n_leaves_used_init"],
        )
        out_feat_d = cp.asarray(c["out_feat"])
        out_bin_d = cp.asarray(c["out_bin"])
        feat_perm_d = cp.asarray(c["feat_perm"])
        gpu = gpu_classify(
            out_feat_d,
            out_bin_d,
            feat_perm_d,
            c["slots_left"],
            c["n_leaves_used_init"],
        )
        ok = compare(c["name"], ref, gpu)
        all_ok = all_ok and ok

    print(f"\n=== POC: {'PASS' if all_ok else 'FAIL'} ===", flush=True)
    sys.exit(0 if all_ok else 1)


if __name__ == "__main__":
    main()
