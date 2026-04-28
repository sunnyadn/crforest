"""λ.exp3 — Phase A1 Day 3 POC: device-side FlatTree updates.

Replaces two host-side write loops in _gpu_kernels.py per level:

  (a) Leaf updates for failed/overflow nodes (lines 1185-1191):
      out_features[node_idx]       = -1
      out_is_leaf[node_idx]        = True
      out_leaf_idx_of_node[node_idx] = n_leaves_used
      n_leaves_used += 1

  (b) Partition writes for successful splits (lines 1242-1255):
      li, ri = n_nodes_used, n_nodes_used + 1; n_nodes_used += 2
      out_features[node_idx]       = actual_feat
      out_split_values[node_idx]   = bin_idx
      out_left[node_idx]           = li
      out_right[node_idx]          = ri
      out_is_leaf[node_idx]        = False
      new_active.append((li, start, mid, depth + 1))
      new_active.append((ri, mid, end, depth + 1))

Done with pure cupy advanced indexing — no custom RawKernel needed.

Killing these two host loops + keeping FlatTree on device removes the
remaining per-level .get() (mids_h) AND eliminates the host array
backing store for FlatTree, which means the final tree-completion D→H
of (out_features, out_left, out_right, out_is_leaf, ...) becomes a
single bulk transfer at end of fit instead of per-level scribbling.

POC scope: validate against reference Python loop on synthetic level-data
covering branch combinations. NOT integrated yet.

Run: ssh win 'export PATH=$HOME/.local/bin:$PATH && cd ~/crforest && \\
       PYTHONUNBUFFERED=1 uv run --extra gpu --extra dev \\
       python -u validation/spikes/lambda/exp3_flattree_update_poc.py'
"""

from __future__ import annotations

import sys

import numpy as np


def reference_host_loops(
    *,
    leaf_node_idx: np.ndarray,
    leaf_assigned_idx: np.ndarray,
    part_node_idx: np.ndarray,
    part_actual_feat: np.ndarray,
    part_bin: np.ndarray,
    part_mids: np.ndarray,
    part_starts: np.ndarray,
    part_ends: np.ndarray,
    part_depths: np.ndarray,
    out_features: np.ndarray,
    out_split_values: np.ndarray,
    out_left: np.ndarray,
    out_right: np.ndarray,
    out_is_leaf: np.ndarray,
    out_leaf_idx_of_node: np.ndarray,
    n_nodes_used_init: int,
) -> dict:
    out_features = out_features.copy()
    out_split_values = out_split_values.copy()
    out_left = out_left.copy()
    out_right = out_right.copy()
    out_is_leaf = out_is_leaf.copy()
    out_leaf_idx_of_node = out_leaf_idx_of_node.copy()

    # Phase A: leaf updates (failed/overflow at start-of-level).
    for k in range(len(leaf_node_idx)):
        ni = int(leaf_node_idx[k])
        out_features[ni] = -1
        out_is_leaf[ni] = True
        out_leaf_idx_of_node[ni] = int(leaf_assigned_idx[k])

    # Phase B: partition writes + new_active.
    n_nodes_used = n_nodes_used_init
    new_active = []
    for k in range(len(part_node_idx)):
        ni = int(part_node_idx[k])
        af = int(part_actual_feat[k])
        bi = int(part_bin[k])
        mid = int(part_mids[k])
        s = int(part_starts[k])
        e = int(part_ends[k])
        d = int(part_depths[k])
        li = n_nodes_used
        ri = n_nodes_used + 1
        n_nodes_used += 2
        out_features[ni] = af
        out_split_values[ni] = bi
        out_left[ni] = li
        out_right[ni] = ri
        out_is_leaf[ni] = False
        new_active.append((li, s, mid, d + 1))
        new_active.append((ri, mid, e, d + 1))

    return {
        "out_features": out_features,
        "out_split_values": out_split_values,
        "out_left": out_left,
        "out_right": out_right,
        "out_is_leaf": out_is_leaf,
        "out_leaf_idx_of_node": out_leaf_idx_of_node,
        "new_active": new_active,
        "n_nodes_used_after": n_nodes_used,
    }


def gpu_flattree_update(
    *,
    leaf_node_idx_d,
    leaf_assigned_idx_d,
    part_node_idx_d,
    part_actual_feat_d,
    part_bin_d,
    part_mids_d,
    part_starts_d,
    part_ends_d,
    part_depths_d,
    out_features_d,
    out_split_values_d,
    out_left_d,
    out_right_d,
    out_is_leaf_d,
    out_leaf_idx_of_node_d,
    n_nodes_used_init: int,
) -> dict:
    import cupy as cp

    # Phase A: leaf updates. Pure scatter via fancy indexing.
    if leaf_node_idx_d.size > 0:
        out_features_d[leaf_node_idx_d] = cp.int32(-1)
        out_is_leaf_d[leaf_node_idx_d] = True
        out_leaf_idx_of_node_d[leaf_node_idx_d] = leaf_assigned_idx_d

    # Phase B: partition writes + new_active.
    T = int(part_node_idx_d.size)
    if T == 0:
        return {
            "out_features_d": out_features_d,
            "out_split_values_d": out_split_values_d,
            "out_left_d": out_left_d,
            "out_right_d": out_right_d,
            "out_is_leaf_d": out_is_leaf_d,
            "out_leaf_idx_of_node_d": out_leaf_idx_of_node_d,
            "new_active_node_idx_d": cp.empty(0, dtype=cp.int32),
            "new_active_starts_d": cp.empty(0, dtype=cp.int32),
            "new_active_ends_d": cp.empty(0, dtype=cp.int32),
            "new_active_depths_d": cp.empty(0, dtype=cp.int32),
            "n_nodes_used_after": int(n_nodes_used_init),
        }

    li_d = cp.int32(n_nodes_used_init) + 2 * cp.arange(T, dtype=cp.int32)
    ri_d = li_d + cp.int32(1)

    out_features_d[part_node_idx_d] = part_actual_feat_d
    out_split_values_d[part_node_idx_d] = part_bin_d
    out_left_d[part_node_idx_d] = li_d
    out_right_d[part_node_idx_d] = ri_d
    out_is_leaf_d[part_node_idx_d] = False

    # Build new_active by interleaving (li, start, mid, d+1) and (ri, mid, end, d+1).
    new_node_idx_d = cp.empty(2 * T, dtype=cp.int32)
    new_node_idx_d[0::2] = li_d
    new_node_idx_d[1::2] = ri_d

    new_starts_d = cp.empty(2 * T, dtype=cp.int32)
    new_starts_d[0::2] = part_starts_d
    new_starts_d[1::2] = part_mids_d

    new_ends_d = cp.empty(2 * T, dtype=cp.int32)
    new_ends_d[0::2] = part_mids_d
    new_ends_d[1::2] = part_ends_d

    new_depths_d = cp.repeat(part_depths_d + cp.int32(1), 2)

    return {
        "out_features_d": out_features_d,
        "out_split_values_d": out_split_values_d,
        "out_left_d": out_left_d,
        "out_right_d": out_right_d,
        "out_is_leaf_d": out_is_leaf_d,
        "out_leaf_idx_of_node_d": out_leaf_idx_of_node_d,
        "new_active_node_idx_d": new_node_idx_d,
        "new_active_starts_d": new_starts_d,
        "new_active_ends_d": new_ends_d,
        "new_active_depths_d": new_depths_d,
        "n_nodes_used_after": n_nodes_used_init + 2 * T,
    }


def compare(label: str, ref: dict, gpu: dict) -> bool:
    import cupy as cp

    ok = True

    # FlatTree arrays.
    for key in (
        "out_features",
        "out_split_values",
        "out_left",
        "out_right",
        "out_is_leaf",
        "out_leaf_idx_of_node",
    ):
        ref_arr = ref[key]
        gpu_arr = cp.asnumpy(gpu[f"{key}_d"])
        if not np.array_equal(ref_arr, gpu_arr):
            print(f"  [{label}] FAIL {key}: ref={ref_arr} gpu={gpu_arr}")
            ok = False

    # new_active list-of-tuples vs interleaved arrays.
    ref_na = (
        np.array(ref["new_active"], dtype=np.int32)
        if ref["new_active"]
        else np.empty((0, 4), dtype=np.int32)
    )
    if ref_na.size:
        gpu_na = np.column_stack(
            [
                cp.asnumpy(gpu["new_active_node_idx_d"]),
                cp.asnumpy(gpu["new_active_starts_d"]),
                cp.asnumpy(gpu["new_active_ends_d"]),
                cp.asnumpy(gpu["new_active_depths_d"]),
            ]
        )
    else:
        gpu_na = np.column_stack(
            [
                cp.asnumpy(gpu["new_active_node_idx_d"]),
                cp.asnumpy(gpu["new_active_starts_d"]),
                cp.asnumpy(gpu["new_active_ends_d"]),
                cp.asnumpy(gpu["new_active_depths_d"]),
            ]
        )
    if not np.array_equal(ref_na, gpu_na):
        print(f"  [{label}] FAIL new_active:\n    ref={ref_na}\n    gpu={gpu_na}")
        ok = False

    if ref["n_nodes_used_after"] != gpu["n_nodes_used_after"]:
        print(
            f"  [{label}] FAIL n_nodes_used_after: ref={ref['n_nodes_used_after']} "
            f"gpu={gpu['n_nodes_used_after']}"
        )
        ok = False

    if ok:
        L = len(ref["new_active"]) // 2
        Ll = len(ref_na) // 2
        n_leaves = sum(1 for v in ref["out_is_leaf"] if v)
        print(f"  [{label}] OK  ({L} partitions = {2 * L} new actives, is_leaf marks={n_leaves})")
    return ok


def make_case(
    *,
    name: str,
    n_nodes_total: int,
    leaf_idxs: list[int],
    part_idxs: list[int],
    seed: int,
):
    """Build a synthetic level: pick which existing FlatTree nodes are
    being decided this level. leaf_idxs become leaves; part_idxs become
    partitions with random feat/bin/mid/start/end/depth."""
    rng = np.random.default_rng(seed)

    # Existing FlatTree state (zeroed).
    out_features = np.zeros(n_nodes_total, dtype=np.int32)
    out_split_values = np.zeros(n_nodes_total, dtype=np.int32)
    out_left = np.zeros(n_nodes_total, dtype=np.int32)
    out_right = np.zeros(n_nodes_total, dtype=np.int32)
    out_is_leaf = np.zeros(n_nodes_total, dtype=bool)
    out_leaf_idx_of_node = np.full(n_nodes_total, -1, dtype=np.int32)

    leaf_node_idx = np.array(leaf_idxs, dtype=np.int32)
    leaf_assigned_idx = np.arange(len(leaf_idxs), dtype=np.int32) + 100  # offset to spot bugs
    part_node_idx = np.array(part_idxs, dtype=np.int32)
    T = len(part_idxs)
    part_actual_feat = (
        rng.integers(0, 50, size=T, dtype=np.int32) if T else np.empty(0, dtype=np.int32)
    )
    part_bin = rng.integers(0, 32, size=T, dtype=np.int32) if T else np.empty(0, dtype=np.int32)
    part_mids = rng.integers(10, 100, size=T, dtype=np.int32) if T else np.empty(0, dtype=np.int32)
    part_starts = rng.integers(0, 10, size=T, dtype=np.int32) if T else np.empty(0, dtype=np.int32)
    part_ends = (
        part_starts + rng.integers(20, 100, size=T, dtype=np.int32)
        if T
        else np.empty(0, dtype=np.int32)
    )
    part_depths = rng.integers(0, 10, size=T, dtype=np.int32) if T else np.empty(0, dtype=np.int32)

    return {
        "name": name,
        "n_nodes_used_init": max(part_idxs + leaf_idxs) + 1 if (part_idxs or leaf_idxs) else 0,
        "leaf_node_idx": leaf_node_idx,
        "leaf_assigned_idx": leaf_assigned_idx,
        "part_node_idx": part_node_idx,
        "part_actual_feat": part_actual_feat,
        "part_bin": part_bin,
        "part_mids": part_mids,
        "part_starts": part_starts,
        "part_ends": part_ends,
        "part_depths": part_depths,
        "out_features": out_features,
        "out_split_values": out_split_values,
        "out_left": out_left,
        "out_right": out_right,
        "out_is_leaf": out_is_leaf,
        "out_leaf_idx_of_node": out_leaf_idx_of_node,
    }


def run_case(case: dict) -> bool:
    import cupy as cp

    ref = reference_host_loops(
        leaf_node_idx=case["leaf_node_idx"],
        leaf_assigned_idx=case["leaf_assigned_idx"],
        part_node_idx=case["part_node_idx"],
        part_actual_feat=case["part_actual_feat"],
        part_bin=case["part_bin"],
        part_mids=case["part_mids"],
        part_starts=case["part_starts"],
        part_ends=case["part_ends"],
        part_depths=case["part_depths"],
        out_features=case["out_features"],
        out_split_values=case["out_split_values"],
        out_left=case["out_left"],
        out_right=case["out_right"],
        out_is_leaf=case["out_is_leaf"],
        out_leaf_idx_of_node=case["out_leaf_idx_of_node"],
        n_nodes_used_init=case["n_nodes_used_init"],
    )

    out_features_d = cp.asarray(case["out_features"])
    out_split_values_d = cp.asarray(case["out_split_values"])
    out_left_d = cp.asarray(case["out_left"])
    out_right_d = cp.asarray(case["out_right"])
    out_is_leaf_d = cp.asarray(case["out_is_leaf"])
    out_leaf_idx_of_node_d = cp.asarray(case["out_leaf_idx_of_node"])
    leaf_node_idx_d = cp.asarray(case["leaf_node_idx"])
    leaf_assigned_idx_d = cp.asarray(case["leaf_assigned_idx"])
    part_node_idx_d = cp.asarray(case["part_node_idx"])
    part_actual_feat_d = cp.asarray(case["part_actual_feat"])
    part_bin_d = cp.asarray(case["part_bin"])
    part_mids_d = cp.asarray(case["part_mids"])
    part_starts_d = cp.asarray(case["part_starts"])
    part_ends_d = cp.asarray(case["part_ends"])
    part_depths_d = cp.asarray(case["part_depths"])

    gpu = gpu_flattree_update(
        leaf_node_idx_d=leaf_node_idx_d,
        leaf_assigned_idx_d=leaf_assigned_idx_d,
        part_node_idx_d=part_node_idx_d,
        part_actual_feat_d=part_actual_feat_d,
        part_bin_d=part_bin_d,
        part_mids_d=part_mids_d,
        part_starts_d=part_starts_d,
        part_ends_d=part_ends_d,
        part_depths_d=part_depths_d,
        out_features_d=out_features_d,
        out_split_values_d=out_split_values_d,
        out_left_d=out_left_d,
        out_right_d=out_right_d,
        out_is_leaf_d=out_is_leaf_d,
        out_leaf_idx_of_node_d=out_leaf_idx_of_node_d,
        n_nodes_used_init=case["n_nodes_used_init"],
    )

    return compare(case["name"], ref, gpu)


def main() -> None:
    try:
        import cupy as cp  # noqa: F401
    except ImportError:
        print("cupy not available — skip POC", file=sys.stderr)
        sys.exit(2)

    cases = [
        make_case(
            name="all-partition",
            n_nodes_total=64,
            leaf_idxs=[],
            part_idxs=[1, 2, 3, 4, 5, 6, 7, 8],
            seed=0,
        ),
        make_case(
            name="all-leaf",
            n_nodes_total=64,
            leaf_idxs=[1, 2, 3, 4, 5, 6, 7, 8],
            part_idxs=[],
            seed=1,
        ),
        make_case(
            name="mixed",
            n_nodes_total=64,
            leaf_idxs=[1, 3, 5],
            part_idxs=[2, 4, 6, 7, 8],
            seed=2,
        ),
        make_case(
            name="single-partition",
            n_nodes_total=16,
            leaf_idxs=[],
            part_idxs=[3],
            seed=3,
        ),
        make_case(
            name="single-leaf",
            n_nodes_total=16,
            leaf_idxs=[3],
            part_idxs=[],
            seed=4,
        ),
        make_case(
            name="empty-level",
            n_nodes_total=16,
            leaf_idxs=[],
            part_idxs=[],
            seed=5,
        ),
        make_case(
            name="realistic",
            n_nodes_total=512,
            leaf_idxs=list(range(10, 25)),
            part_idxs=list(range(50, 110)),
            seed=6,
        ),
    ]

    all_ok = True
    print(f"Running {len(cases)} cases...\n")
    for c in cases:
        ok = run_case(c)
        all_ok = all_ok and ok

    print(f"\n=== POC: {'PASS' if all_ok else 'FAIL'} ===", flush=True)
    sys.exit(0 if all_ok else 1)


if __name__ == "__main__":
    main()
