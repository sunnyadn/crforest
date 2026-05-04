"""Iota.exp1 — validate per-tree-level batching beats per-node at typical workload.

Spike runs on win box (RTX 5070 Ti). Compares:
  (A) Per-node calls (one launch per node, theta.5 Probe 5 architecture)
  (B) Per-tree-level calls (one launch per level, this plan's G.1 architecture)

Decision rule: G.1 must be >=3x faster than per-node at n=5k, mtry=8, n_bins=32.
Otherwise pause and re-spec.
"""

from __future__ import annotations

import time

import numpy as np


def main():
    import cupy as cp

    from comprisk._gpu_kernels import histogram_kernel_per_level

    rng = np.random.default_rng(0)
    n = 5000
    mtry = 8
    n_bins = 32
    n_causes = 2
    n_time_bins = 50

    X = rng.integers(0, n_bins, size=(n, mtry), dtype=np.uint8)
    t_idx = rng.integers(0, n_time_bins, size=n, dtype=np.int32)
    event = rng.integers(0, n_causes + 1, size=n, dtype=np.int32)

    Xb = cp.asarray(X)
    td = cp.asarray(t_idx)
    ed = cp.asarray(event)
    perm = cp.arange(n, dtype=np.int32)

    # Mimic a level with N_nodes_at_level={1,2,4,8,16,32,64,128} active nodes,
    # each with n // N_nodes samples. Realistic at depth 0..7.
    levels = [1, 2, 4, 8, 16, 32, 64, 128]

    # (A) per-node launches: emulated by N_nodes single-block calls.
    print("=== Per-node launches ===", flush=True)
    per_node_walls = {}
    for N_nodes in levels:
        per_node_n = n // N_nodes
        node_starts = cp.asarray([i * per_node_n for i in range(N_nodes)], dtype=np.int32)
        node_ends = cp.asarray([(i + 1) * per_node_n for i in range(N_nodes)], dtype=np.int32)
        ehist = cp.zeros((N_nodes, mtry, n_bins, n_causes, n_time_bins), dtype=cp.uint32)
        nat = cp.zeros((N_nodes, mtry, n_bins, n_time_bins), dtype=cp.uint32)

        # Warm
        for k in range(N_nodes):
            histogram_kernel_per_level(
                Xb,
                td,
                ed,
                perm,
                node_starts[k : k + 1],
                node_ends[k : k + 1],
                ehist[k : k + 1],
                nat[k : k + 1],
                n_bins=n_bins,
                n_causes=n_causes,
                n_time_bins=n_time_bins,
                mtry=mtry,
            )
        cp.cuda.runtime.deviceSynchronize()

        ehist[:] = 0
        nat[:] = 0
        cp.cuda.runtime.deviceSynchronize()
        t0 = time.perf_counter()
        for k in range(N_nodes):
            histogram_kernel_per_level(
                Xb,
                td,
                ed,
                perm,
                node_starts[k : k + 1],
                node_ends[k : k + 1],
                ehist[k : k + 1],
                nat[k : k + 1],
                n_bins=n_bins,
                n_causes=n_causes,
                n_time_bins=n_time_bins,
                mtry=mtry,
            )
        cp.cuda.runtime.deviceSynchronize()
        per_node_t = time.perf_counter() - t0
        per_node_walls[N_nodes] = per_node_t
        print(f"  N_nodes={N_nodes:4d}  wall={per_node_t * 1e3:7.2f} ms", flush=True)

    # (B) per-level launches: one launch covering all N_nodes.
    print("=== Per-level launches ===", flush=True)
    per_level_walls = {}
    for N_nodes in levels:
        per_node_n = n // N_nodes
        node_starts = cp.asarray([i * per_node_n for i in range(N_nodes)], dtype=np.int32)
        node_ends = cp.asarray([(i + 1) * per_node_n for i in range(N_nodes)], dtype=np.int32)
        ehist = cp.zeros((N_nodes, mtry, n_bins, n_causes, n_time_bins), dtype=cp.uint32)
        nat = cp.zeros((N_nodes, mtry, n_bins, n_time_bins), dtype=cp.uint32)

        # Warm
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
        cp.cuda.runtime.deviceSynchronize()

        ehist[:] = 0
        nat[:] = 0
        cp.cuda.runtime.deviceSynchronize()
        t0 = time.perf_counter()
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
        cp.cuda.runtime.deviceSynchronize()
        per_level_t = time.perf_counter() - t0
        per_level_walls[N_nodes] = per_level_t
        print(f"  N_nodes={N_nodes:4d}  wall={per_level_t * 1e3:7.2f} ms", flush=True)

    # Summary
    print("=== Speedups (per-node / per-level) ===", flush=True)
    for N_nodes in levels:
        speedup = per_node_walls[N_nodes] / per_level_walls[N_nodes]
        print(f"  N_nodes={N_nodes:4d}  speedup={speedup:5.2f}x", flush=True)


if __name__ == "__main__":
    main()
