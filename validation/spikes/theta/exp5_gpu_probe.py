"""θ.5 — GPU feasibility probe.

Self-contained — does NOT need comprisk imports. Just numpy + cupy.
Goal: answer three questions for GPU offload of the CR-forest hot kernel:

  1. **Per-call latency floor**: how fast is a minimal kernel + a single
     (n_node × mtry) histogram? Establishes lower bound for per-node
     GPU split (the architecture used by sklearn HistGB).
  2. **Throughput at the right granularity**: how fast does cupy build
     the (mtry, n_bins, n_causes, n_time_bins) accumulator that
     ``find_best_split_hist_batched`` builds?
  3. **Launch overhead**: how often can we round-trip a no-op kernel?
     If <50 µs, per-node calls are viable; if >100 µs, must batch
     across nodes/trees.

Compare to the CPU njit baseline (1.9 ms for full root, 8 mtry features,
from η.exp4 on the Mac).

Usage on GPU box:
    pip install cupy-cuda12x   # or 13x depending on CUDA version
    python exp5_gpu_probe.py

Outputs to stdout. Paste back into chat for analysis.
"""

from __future__ import annotations

import sys
import time

import numpy as np

N = 100_000
MTRY = 8
N_BINS = 256
N_CAUSES = 2
N_TIME_BINS = 50
SEED = 0


def main() -> None:
    print(f"[θ.5] target n={N} mtry={MTRY} n_bins={N_BINS} n_time_bins={N_TIME_BINS}")
    print("[θ.5] CPU baseline (η.exp4 on Mac, 8 mtry feat, full root): 1.9 ms")
    print()

    try:
        import cupy as cp
    except ImportError:
        print("[θ.5] cupy NOT installed. Install via:")
        print("  pip install cupy-cuda12x  # or cupy-cuda13x")
        print("Then re-run.")
        sys.exit(1)

    print(f"[θ.5] cupy {cp.__version__}")
    try:
        dev = cp.cuda.Device()
        props = cp.cuda.runtime.getDeviceProperties(dev.id)
        name = props["name"].decode() if isinstance(props["name"], bytes) else props["name"]
        print(f"[θ.5] GPU: {name}")
        print(
            f"[θ.5] CUDA: runtime {cp.cuda.runtime.runtimeGetVersion()}, "
            f"driver {cp.cuda.runtime.driverGetVersion()}"
        )
        print(
            f"[θ.5] SM count: {props.get('multiProcessorCount', '?')}, "
            f"compute capability: {props['major']}.{props['minor']}"
        )
    except Exception as ex:
        print(f"[θ.5] WARN: couldn't query GPU properties: {ex}")

    rng = np.random.default_rng(SEED)
    X_cpu = rng.integers(0, N_BINS, size=(N, MTRY), dtype=np.uint8)
    t_cpu = rng.integers(0, N_TIME_BINS, size=N).astype(np.int32)
    e_cpu = rng.integers(0, N_CAUSES + 1, size=N).astype(np.int32)

    X_gpu = cp.asarray(X_cpu)
    t_gpu = cp.asarray(t_cpu)
    e_gpu = cp.asarray(e_cpu)
    cp.cuda.Stream.null.synchronize()

    # ───────── Probe 1: launch overhead (minimal no-op kernel) ─────────
    print("\n=== Probe 1: kernel launch overhead ===")
    noop = cp.RawKernel(
        r"""
    extern "C" __global__ void noop() {}
    """,
        "noop",
    )
    REPS = 1000
    t0 = time.perf_counter()
    for _ in range(REPS):
        noop((1,), (1,), ())
    cp.cuda.Stream.null.synchronize()
    per_launch_us = (time.perf_counter() - t0) / REPS * 1e6
    print(f"  no-op launch + sync = {per_launch_us:6.1f} µs/call ({REPS} reps)")
    print(f"  → {1e6 / per_launch_us:.0f} launches/sec")

    # ───────── Probe 2: per-feature 1D histogram (bincount) ─────────
    print("\n=== Probe 2: per-feature 1D histogram (cp.bincount) ===")
    REPS = 100
    cp.cuda.Stream.null.synchronize()
    t0 = time.perf_counter()
    for _ in range(REPS):
        for f in range(MTRY):
            _ = cp.bincount(X_gpu[:, f], minlength=N_BINS)
    cp.cuda.Stream.null.synchronize()
    per_iter_ms = (time.perf_counter() - t0) / REPS * 1e3
    per_feat_us = per_iter_ms * 1000 / MTRY
    print(
        f"  bincount loop, mtry={MTRY} features = {per_iter_ms:6.2f} ms/iter "
        f"({per_feat_us:.1f} µs/feature)"
    )

    # ───────── Probe 3: full (mtry, n_bins, n_causes, n_time_bins) histogram ─────────
    print("\n=== Probe 3: 4D event_hist + n_at via scatter_add ===")
    # event_hist[mtry, n_bins, n_causes, n_time_bins]
    # n_at[mtry, n_bins, n_time_bins]
    REPS = 30
    cp.cuda.Stream.null.synchronize()
    t0 = time.perf_counter()
    for _ in range(REPS):
        event_hist = cp.zeros((MTRY, N_BINS, N_CAUSES, N_TIME_BINS), dtype=cp.uint32)
        n_at = cp.zeros((MTRY, N_BINS, N_TIME_BINS), dtype=cp.uint32)
        # For each feature: indices = (b, c, t) per sample where event > 0
        for f in range(MTRY):
            bins = X_gpu[:, f]
            # n_at[bin, t] += 1 for every sample
            flat_idx = bins.astype(cp.int64) * N_TIME_BINS + t_gpu.astype(cp.int64)
            cp.add.at(n_at[f].reshape(-1), flat_idx, 1)
            # event_hist[bin, e-1, t] += 1 for samples with event > 0
            mask = e_gpu > 0
            ev_minus = (e_gpu[mask] - 1).astype(cp.int64)
            flat_evt = (
                bins[mask].astype(cp.int64) * (N_CAUSES * N_TIME_BINS)
                + ev_minus * N_TIME_BINS
                + t_gpu[mask].astype(cp.int64)
            )
            cp.add.at(event_hist[f].reshape(-1), flat_evt, 1)
    cp.cuda.Stream.null.synchronize()
    per_call_ms = (time.perf_counter() - t0) / REPS * 1e3
    print(f"  4D hist via cp.add.at = {per_call_ms:6.2f} ms/call (full root, n={N})")

    # ───────── Probe 4: custom RawKernel for the same 4D histogram ─────────
    # cp.add.at is general but slow due to atomics on dispersed indices.
    # A custom kernel can be 5-10× faster. Sketch one and time it.
    print("\n=== Probe 4: custom RawKernel 4D histogram ===")
    kern_src = r"""
extern "C" __global__
void hist_kernel(
    const unsigned char* X,    // [n, mtry]
    const int* t_idx,          // [n]
    const int* event,          // [n]
    unsigned int* event_hist,  // [mtry, n_bins, n_causes, n_time_bins], flat
    unsigned int* n_at,        // [mtry, n_bins, n_time_bins], flat
    int n,
    int mtry,
    int n_bins,
    int n_causes,
    int n_time_bins
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    int t = t_idx[i];
    int e = event[i];
    for (int f = 0; f < mtry; ++f) {
        int b = (int)X[i * mtry + f];
        // n_at[f, b, t] += 1
        int n_at_idx = f * n_bins * n_time_bins + b * n_time_bins + t;
        atomicAdd(n_at + n_at_idx, 1u);
        // event_hist[f, b, e-1, t] += 1 if e > 0
        if (e > 0) {
            int eh_idx = f * n_bins * n_causes * n_time_bins
                       + b * n_causes * n_time_bins
                       + (e - 1) * n_time_bins + t;
            atomicAdd(event_hist + eh_idx, 1u);
        }
    }
}
"""
    hist_kernel = cp.RawKernel(kern_src, "hist_kernel")
    BLOCK = 256
    grid = (N + BLOCK - 1) // BLOCK
    REPS = 100
    # Warm
    event_hist = cp.zeros((MTRY, N_BINS, N_CAUSES, N_TIME_BINS), dtype=cp.uint32)
    n_at = cp.zeros((MTRY, N_BINS, N_TIME_BINS), dtype=cp.uint32)
    hist_kernel(
        (grid,),
        (BLOCK,),
        (
            X_gpu,
            t_gpu,
            e_gpu,
            event_hist,
            n_at,
            np.int32(N),
            np.int32(MTRY),
            np.int32(N_BINS),
            np.int32(N_CAUSES),
            np.int32(N_TIME_BINS),
        ),
    )
    cp.cuda.Stream.null.synchronize()
    t0 = time.perf_counter()
    for _ in range(REPS):
        event_hist.fill(0)
        n_at.fill(0)
        hist_kernel(
            (grid,),
            (BLOCK,),
            (
                X_gpu,
                t_gpu,
                e_gpu,
                event_hist,
                n_at,
                np.int32(N),
                np.int32(MTRY),
                np.int32(N_BINS),
                np.int32(N_CAUSES),
                np.int32(N_TIME_BINS),
            ),
        )
    cp.cuda.Stream.null.synchronize()
    per_call_us = (time.perf_counter() - t0) / REPS * 1e6
    print(f"  custom kernel 4D hist = {per_call_us:6.1f} µs/call (full root, n={N})")
    print(f"  → vs CPU njit (1900 µs):  {1900 / per_call_us:.1f}x faster on GPU")
    print(f"  → throughput: {N * MTRY / (per_call_us * 1e-6) / 1e9:.1f} G-updates/sec")

    # ───────── Probe 5: amortized over many calls (same shape) ─────────
    # Per-tree we have ~6500 nodes × this kernel call. With launch overhead
    # ~10-50µs, the floor is dominated by launch overhead at small node sizes.
    print("\n=== Probe 5: per-tree projection ===")
    # Real-tree node sizes shrink: n_node ≈ n / 2^depth at depth d.
    levels = int(np.log2(N / 15)) + 1  # tree depth
    print(f"  est tree depth = {levels}")
    print(f"  est per-tree GPU kernel work (rough) = {per_call_us * levels:.0f} µs")
    print(f"  est per-tree wall floor (with launch overhead × {levels} per-level calls):")
    print(f"     pure kernel: {per_call_us * levels:.0f} µs = {per_call_us * levels * 1e-3:.2f} ms")
    print(
        f"     + launch overhead × ~6500 nodes: "
        f"{per_launch_us * 6500:.0f} µs = {per_launch_us * 6500 * 1e-3:.1f} ms "
        f"per tree"
    )
    print(
        f"  100 trees × per-tree wall (single GPU stream) = "
        f"{(per_call_us * levels + per_launch_us * 6500) * 100 * 1e-6:.2f} s"
    )

    print()
    print("[θ.5] Summary:")
    print(
        f"  - Launch overhead = {per_launch_us:.0f} µs ({'GOOD <50µs' if per_launch_us < 50 else 'PROBLEM ≥50µs — need batching across nodes'})"
    )
    print(
        f"  - Custom kernel histogram on full root = {per_call_us:.0f} µs vs CPU 1900 µs = {1900 / per_call_us:.0f}x speedup"
    )
    print(
        f"  - Per-tree projection (per-node GPU calls): {(per_call_us * levels + per_launch_us * 6500) * 1e-3:.0f} ms wall"
    )
    print(
        f"  - 100 trees serial GPU = {(per_call_us * levels + per_launch_us * 6500) * 100 * 1e-6:.1f} s wall"
    )


if __name__ == "__main__":
    main()
