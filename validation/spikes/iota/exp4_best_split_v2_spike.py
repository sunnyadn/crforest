"""Iota.exp4 — best_split_kernel_per_node v2 spike (running prefix-sum scan).

================================================================================
DESIGN MEMO
================================================================================

Problem
-------
Production kernel `best_split_kernel_per_node` (_gpu_kernels.py) has a device
helper `compute_split_stat` that REBUILDS the full reverse-cumsum
(at_risk_total / at_risk_left / d_*_total / d_*_left) for EVERY (feat, b_cut)
candidate. Each rebuild is O(n_bins · n_causes · n_time_bins). The block
processes mtry · (n_bins-1) = 8 · 31 = 248 candidates, so per-node work is
O(B^2 · K · T) instead of the algorithmic O(B · K · T) of the CPU twin
`find_best_split_hist_batched` (_hist_splits.py).

At n=100k tree shape (16 levels, ~5600 leaves), production kernel costs
2779 ms / 78.8% of the single-tree wall (Iota.exp3b profile). M1 review
flagged this as the redo-the-prefix-sum-each-candidate hotspot.

Algorithmic fix
---------------
The CPU twin advances running prefix sums monotonically through `b`:

    at_risk_left = zeros(T)
    d_k_left     = zeros(K, T)
    n_left       = 0
    for b in 0 .. n_bins-2:                # iterate cut points
        at_risk_left += per_bin_atrisk[f, b, :]      # O(T) update
        d_k_left     += per_bin_events[f, b, :, :]   # O(K·T) update
        n_left       += per_bin_total[f, b]
        if not cand_mask[f, b]: continue
        if n_left < min_leaf or n_total - n_left < min_leaf: continue
        compute logrank stat from at_risk_left / d_k_left      # O(K·T)
        update best

Per (node, feat) work drops from O(B · B · K · T) to O(B · K · T). For our
fixture B=32, K=2, T=50, that's a 32× theoretical reduction in stat math.

CUDA mapping (v2 design)
------------------------
**Block shape: one block per (node, feat), grid = (N_nodes, mtry, 1).**
This pins the running prefix sums to a single block of threads — no
cross-block coordination needed for the scan. mtry-level argmax across
features happens on a second tiny kernel (or via atomic-max + replay).

**Threads parallelize over time bins inside the block.**
- Block size THREADS (we'll use 64) covers up to MAX_TIME_BINS=256 time
  bins via a strided loop. Each thread owns a stripe of t-indices.
- Running prefix sums (at_risk_left[T], d_k_left[K, T]) live in
  **shared memory**, not registers — they're shared across threads in
  the block (each thread reads/writes its own t-slice; no shared-memory
  contention because partition is by t).

**The b loop runs SERIALLY in the block, threads parallelize the t-axis.**
Per cut-point b:
  1. All threads cooperatively load per-bin contribution (n_at[node,f,b,:],
     event_hist[node,f,b,:,:]) and update their t-slices of at_risk_left
     and d_k_left.
  2. Need n_left (scalar) and at_risk_left[t] cumulative. We track n_left
     as a block-shared scalar updated by tid==0 (cheap).
  3. Block-wide barrier (__syncthreads).
  4. If cand_mask[node, f, b] and min-samples gate passes, all threads
     compute their slice of the logrank stat (sum over their stripe of
     time bins of the per-t contribution to num_sum / var_sum), reduce
     to block scalar, and tid==0 updates the running best (feat=f, bin=b,
     stat).
  5. The "advance the prefix sums" step is FREE on the next b iteration
     because they're already updated in shared memory.

**at_risk_total / d_k_total prepopulation:**
We need the full-feature totals (sum over all bins) to compute the right
side as `total - left`. Precompute these ONCE at block start by walking
through all n_bins of the histogram and reducing into shared memory.
This is O(B · K · T) per (node, feat), single pass.

**at_risk_total[t] = sum_b sum_{s>=t} n_at[b, s]** (reverse cumsum over time
inside each bin, then sum across bins). Equivalently:
    at_risk_total[t] = sum_{b, s>=t} n_at[b, s]
We can compute it via a second pass: for each t, walk b=0..B-1 and
s=t..T-1. But cleaner: compute per_bin_running_total[b, t] (reverse cumsum
in t per bin) on the fly as we add to at_risk_left. Actually we don't need
per-bin reverse cumsum AT ALL — at_risk_left[t] only needs the total
"samples in left side at risk of failing at time >= t", which equals
`sum_{b<=cut, s>=t} n_at[b, s]`. We can achieve this incrementally:

    at_risk_left[t] is the cumulative left sum advancing through cut points.
    When we move bin b from "right" to "left" (advance cut), we add
    bin-b's reverse-cumsum-over-time vector: per_bin_atrisk_revcum[b, t].

So we need per_bin_atrisk_revcum (and similarly d_k_left increments are
just event_hist[b, k, t] since events are PMFs at exactly time t, not
cumulative).

For at_risk increment we can compute per-bin reverse-cumsum on-the-fly
inside the block: each block does ONE pass at start to compute
per_bin_atrisk_revcum[b, t] in shared memory. Cost is O(B · T) per block.
Or we can compute at_risk_left incrementally without ever materializing
per_bin_atrisk_revcum — use this trick:

    at_risk_left[t] := at_risk_left[t] + sum_{s>=t} n_at[b, s]
                    = at_risk_left[t] + (sum_{s>=0} n_at[b, s])  (if t==0)
                    ...

Cleanest: per b, loop t from T-1 down to 0 with a per-thread running
scalar (across t), each thread owning a t-slice — but reverse cumsum
crosses thread boundaries. Easier: precompute per_bin_atrisk_revcum[b, t]
in shared memory in a setup pass. With B=32, T=50, K=2 the shared memory
budget is:
    per_bin_atrisk_revcum: 32 · 50 · 8 = 12,800 B
    at_risk_left:          50 · 8       =    400 B
    d_k_left:              2 · 50 · 8   =    800 B
    at_risk_total:         50 · 8       =    400 B
    d_k_total:             2 · 50 · 8   =    800 B
    d_any_total:           50 · 8       =    400 B
    d_any_left:            50 · 8       =    400 B
    Lau cumsum scratch:    2 · 50 · 8   =    800 B (at_risk_total_inc / left_inc)
                                          ----- 17 KB
Comfortably fits in 48 KB shmem-per-block budget. (RTX 5070 Ti has
100 KB/SM dynamic shmem; default per-block cap is 48 KB without opt-in
but blocks of 17 KB allow many concurrent blocks.)

We REUSE the pre-allocated full event_hist + n_at buffers from the v1
launch (caller doesn't change). Block reads from `event_hist[node, f, :, :, :]`
and `n_at[node, f, :, :]` directly using `(node * mtry + f) * stride`.

**cand_mask interaction:**
The b loop ADVANCES prefix sums even when cand_mask[f, b] is False.
We just skip the stat-compute + best-update steps. This matches the
CPU twin's semantics: prefix sums are always advanced; mask only gates
which b are evaluated. Critical correctness point.

**Argmax over (f, b) within the block, then across blocks (mtry → 1):**
Approach A: one block per (node, feat) yields per-block (best_f=f,
best_b, best_stat). Reduce over feat with a small follow-up kernel that
takes (N_nodes, mtry) per-block output and writes (N_nodes,) lex-min
tie-broken argmax.

Approach B: keep block-per-node like v1 — but then we lose the per-(node,
feat) prefix-sum cache. Each thread within a node-block handles one
feature serially. With THREADS=mtry, each thread runs the full b-loop
private to itself, including its own private prefix-sum buffers in
shared memory (sliced by feature). This trades a thinner block (8
threads = 1/4 warp) for not needing a second kernel — worse occupancy.

We pick **Approach A** (block-per-(node,feat)) for cleaner shmem layout
and full T-parallelism. Second kernel is an O(N_nodes · mtry) lex-min
reduction — trivial cost (microseconds).

**Splitrule 0 vs 1:**
Both rules read the SAME prefix sums (at_risk_left, d_k_left, n_left,
at_risk_total, d_k_total). They differ only in the per-t logrank
contribution formula. Single kernel branches on `splitrule_code` inside
the inner stat loop. No extra device variants needed.

**Tie-break parity with v1:**
v1 picks first (lowest f, then lowest b) on stat ties. v2 in Approach A
processes one (f, b) per block — ties between blocks are resolved in the
second mtry-reduction kernel, which uses the same lex rule.
Within-block (across b) the inner loop uses `>` strict update, so the
LOWEST b wins on within-block ties (matches CPU and v1).

================================================================================
END DESIGN MEMO
================================================================================

This script:
1. Defines `_BEST_SPLIT_V2_SRC` (CUDA C source) implementing the design.
2. Provides a Python wrapper `best_split_kernel_per_node_v2` matching the
   v1 API.
3. Runs a small-fixture correctness check against v1 for both
   splitrule_code 0 and 1.
4. Runs a perf measurement (n=100k, mtry=8, n_bins=32, n_time_bins=50)
   monkey-patching `_gpu_kernels.best_split_kernel_per_node` to v2 and
   reporting per-level wall + total tree wall.

Spike — no commit, no production replacement.
"""

from __future__ import annotations

import sys
import time
from collections import defaultdict

import numpy as np

# ----------------------------------------------------------------------
# v2 CUDA source
# ----------------------------------------------------------------------
_BEST_SPLIT_V2_SRC = r"""
// best_split_kernel_per_node_v2: running prefix-sum scan
//
// Block layout:
//   grid  = (N_nodes, mtry, 1)
//   block = (THREADS, 1, 1)        // THREADS picked at launch (64 default)
//
// One block per (node, feature). Threads in the block parallelize over
// time bins (n_time_bins). The b loop runs serially in the block; per b
// all threads cooperatively (i) update their t-stripe of running prefix
// sums (at_risk_left, d_k_left), (ii) check cand_mask + min_leaf gate,
// and if open (iii) compute their stripe's contribution to (num_sum,
// var_sum), reduce to block scalar, and update the per-block best.
//
// Output is per-(node, feat) (best_b, best_stat) into temporary buffers,
// reduced down to per-node (best_feat, best_bin, best_stat) by a tiny
// follow-up reduction kernel.

#define MAX_CAUSES 8
#define MAX_TIME_BINS 256
#define MAX_THREADS 256

extern "C" __global__ void best_split_kernel_per_node_v2_scan(
    const unsigned int* __restrict__ event_hist,  // (N_nodes, mtry, n_bins, n_causes, n_time_bins)
    const unsigned int* __restrict__ n_at,        // (N_nodes, mtry, n_bins, n_time_bins)
    const bool* __restrict__ cand_mask,           // (N_nodes, mtry, n_bins-1)
    int* __restrict__ pf_bin,                     // (N_nodes, mtry) per-(node,feat) best bin
    double* __restrict__ pf_stat,                 // (N_nodes, mtry) per-(node,feat) best stat
    int n_bins, int n_causes, int n_time_bins, int mtry,
    int min_samples_leaf, int splitrule_code, int cause)
{
    const int node = blockIdx.x;
    const int feat = blockIdx.y;
    const int tid  = threadIdx.x;
    const int blockDimX = blockDim.x;

    // Shared memory layout (single contiguous chunk):
    //   at_risk_left  [n_time_bins]
    //   at_risk_total [n_time_bins]
    //   d_k_left      [n_causes * n_time_bins]
    //   d_k_total     [n_causes * n_time_bins]
    //   d_any_left    [n_time_bins]
    //   d_any_total   [n_time_bins]
    //   ar_total_inc  [n_causes * n_time_bins]
    //   ar_left_inc   [n_causes * n_time_bins]
    //   block scalars: best_stat (double), best_b (int), n_left (int)
    //   reduction scratch: red_num [THREADS], red_var [THREADS]
    extern __shared__ char smem[];
    double* s_at_risk_left  = (double*)smem;
    double* s_at_risk_total = s_at_risk_left  + n_time_bins;
    double* s_d_k_left      = s_at_risk_total + n_time_bins;
    double* s_d_k_total     = s_d_k_left      + n_causes * n_time_bins;
    double* s_d_any_left    = s_d_k_total     + n_causes * n_time_bins;
    double* s_d_any_total   = s_d_any_left    + n_time_bins;
    double* s_ar_total_inc  = s_d_any_total   + n_time_bins;
    double* s_ar_left_inc   = s_ar_total_inc  + n_causes * n_time_bins;
    double* s_red_num       = s_ar_left_inc   + n_causes * n_time_bins;
    double* s_red_var       = s_red_num       + blockDimX;
    int*    s_scalars       = (int*)(s_red_var + blockDimX);
    // s_scalars[0] = n_left, s_scalars[1] = best_b, s_scalars[2] = best_stat_int_view (unused)
    double* s_best_stat     = (double*)(s_scalars + 4);  // 8-byte aligned

    // Initialize accumulators (each thread inits its t-stripe).
    for (int t = tid; t < n_time_bins; t += blockDimX) {
        s_at_risk_left[t]  = 0.0;
        s_at_risk_total[t] = 0.0;
        s_d_any_left[t]    = 0.0;
        s_d_any_total[t]   = 0.0;
        for (int k = 0; k < n_causes; k++) {
            s_d_k_left[k * n_time_bins + t]  = 0.0;
            s_d_k_total[k * n_time_bins + t] = 0.0;
        }
    }
    if (tid == 0) {
        s_scalars[0] = 0;       // n_left
        s_scalars[1] = 0;       // best_b
        s_best_stat[0] = 0.0;   // best_stat (init = 0, matches CPU's best_s = 0.0)
        s_scalars[2] = 0;       // n_total (set in setup pass)
    }
    __syncthreads();

    // -------- Setup pass: compute at_risk_total, d_k_total, d_any_total --------
    // For each bin b, compute reverse-cumsum-over-t of n_at[b, :] and add to
    // at_risk_total. d_k_total / d_any_total are simple sums over bins of
    // event_hist[b, k, t].
    long long ahb = ((long long)node * mtry + feat) * n_bins * n_time_bins;
    long long ehb = ((long long)node * mtry + feat) * n_bins * n_causes * n_time_bins;

    // Each thread computes the reverse-cumsum for one (or several) bins serially
    // along t, but distributes bins across threads. Then atomically adds to the
    // shared at_risk_total. Plus per-bin n_total contribution.
    int n_total_local = 0;
    for (int b = tid; b < n_bins; b += blockDimX) {
        long long bin_off_n = ahb + (long long)b * n_time_bins;
        double running = 0.0;
        // We'll write the reverse-cumsum into a per-thread scratch, then add
        // to s_at_risk_total via atomic operations. To avoid an extra big
        // shared buffer, we issue atomicAdds directly per-t (low contention
        // because n_bins tiny vs threads stride).
        for (int t = n_time_bins - 1; t >= 0; t--) {
            unsigned int v = n_at[bin_off_n + t];
            running += (double)v;
            atomicAdd(&s_at_risk_total[t], running);
        }
        n_total_local += (int)running;  // total samples in this bin
    }
    // Sum n_total across threads via shared scratch
    s_red_num[tid] = (double)n_total_local;
    __syncthreads();
    for (int s = blockDimX / 2; s > 0; s >>= 1) {
        if (tid < s) s_red_num[tid] += s_red_num[tid + s];
        __syncthreads();
    }
    int n_total;
    if (tid == 0) s_scalars[2] = (int)s_red_num[0];
    __syncthreads();
    n_total = s_scalars[2];

    // d_k_total / d_any_total: each thread handles a t-stripe, sums over all bins.
    for (int t = tid; t < n_time_bins; t += blockDimX) {
        double sum_any = 0.0;
        for (int k = 0; k < n_causes; k++) {
            double sum_k = 0.0;
            for (int b = 0; b < n_bins; b++) {
                long long off = ehb + ((long long)b * n_causes + k) * n_time_bins + t;
                sum_k += (double)event_hist[off];
            }
            s_d_k_total[k * n_time_bins + t] = sum_k;
            sum_any += sum_k;
        }
        s_d_any_total[t] = sum_any;
    }
    __syncthreads();

    // -------- Splitrule 0 setup: at_risk_total_inc[k, t] = at_risk_total[t] + Lau cumsum --------
    if (splitrule_code == 0) {
        // Each thread takes a (k, t) row strip; cumsum over t is intrinsically serial,
        // so we parallelize over k only. With n_causes <= 8 we have at most 8 active
        // threads; rest idle. Fine for setup-once cost.
        for (int k = tid; k < n_causes; k += blockDimX) {
            double cumsum = 0.0;
            for (int t = 0; t < n_time_bins; t++) {
                s_ar_total_inc[k * n_time_bins + t] = s_at_risk_total[t] + cumsum;
                cumsum += s_d_any_total[t] - s_d_k_total[k * n_time_bins + t];
            }
        }
        __syncthreads();
    }

    // -------- Main loop: serial b, parallel t --------
    int B_minus_1 = n_bins - 1;
    int n_left = 0;

    for (int b = 0; b < B_minus_1; b++) {
        // (i) advance prefix sums by bin b
        // at_risk_left += reverse_cumsum_t(n_at[b, :])
        // We need that reverse-cumsum FOR THIS BIN ONLY, applied to at_risk_left.
        // Strategy: tid==0 (or distributed) computes the reverse-cumsum scalar
        // sequentially, but updates each t in shared mem. Distributing this
        // across threads while maintaining the running scalar is awkward. For
        // simplicity AND because B*T cumsum-in-block is cheap, do it serially
        // in tid==0.
        long long bin_off_n = ahb + (long long)b * n_time_bins;
        if (tid == 0) {
            double running = 0.0;
            for (int t = n_time_bins - 1; t >= 0; t--) {
                unsigned int v = n_at[bin_off_n + t];
                running += (double)v;
                s_at_risk_left[t] += running;
            }
            n_left = s_scalars[0] + (int)running;
            s_scalars[0] = n_left;
        }
        // d_k_left += event_hist[b, k, t]; d_any_left = sum_k d_k_left
        long long bin_off_e = ehb + (long long)b * n_causes * n_time_bins;
        for (int t = tid; t < n_time_bins; t += blockDimX) {
            double sum_any = 0.0;
            for (int k = 0; k < n_causes; k++) {
                double dv = (double)event_hist[bin_off_e + (long long)k * n_time_bins + t];
                s_d_k_left[k * n_time_bins + t] += dv;
                sum_any += s_d_k_left[k * n_time_bins + t];
            }
            s_d_any_left[t] = sum_any;
        }
        __syncthreads();

        // Refresh n_left from shared (set by tid==0)
        n_left = s_scalars[0];

        // (ii) check cand_mask + min-samples gate
        bool eligible = cand_mask[((long long)node * mtry + feat) * B_minus_1 + b];
        int n_right = n_total - n_left;
        if (n_left < min_samples_leaf || n_right < min_samples_leaf) eligible = false;
        if (!eligible) continue;

        // (iii) compute logrank stat
        double thread_num = 0.0;
        double thread_var = 0.0;

        if (splitrule_code == 0) {
            // First, build at_risk_left_inc[k, t] (Lau cumsum on left). Same
            // serial-over-t shape as the total setup, parallelized over k.
            for (int k = tid; k < n_causes; k += blockDimX) {
                double cumsum = 0.0;
                for (int t = 0; t < n_time_bins; t++) {
                    s_ar_left_inc[k * n_time_bins + t] = s_at_risk_left[t] + cumsum;
                    cumsum += s_d_any_left[t] - s_d_k_left[k * n_time_bins + t];
                }
            }
            __syncthreads();

            // Now each thread sums num/var over its t-stripe, all k.
            for (int t = tid; t < n_time_bins; t += blockDimX) {
                double ar_total_t = s_at_risk_total[t];
                if (ar_total_t == 0.0) continue;
                for (int k = 0; k < n_causes; k++) {
                    double arinc_t  = s_ar_total_inc[k * n_time_bins + t];
                    double arlinc_t = s_ar_left_inc[k * n_time_bins + t];
                    double d_t  = s_d_k_total[k * n_time_bins + t];
                    double dl_t = s_d_k_left[k * n_time_bins + t];
                    thread_num += dl_t - d_t * arlinc_t / arinc_t;
                    if (ar_total_t >= 2.0) {
                        thread_var += d_t * arlinc_t * (arinc_t - arlinc_t) * (arinc_t - d_t)
                                     / (arinc_t * arinc_t * (arinc_t - 1.0));
                    }
                }
            }
        } else {
            int kc = cause - 1;
            for (int t = tid; t < n_time_bins; t += blockDimX) {
                double ar_t = s_at_risk_total[t];
                if (ar_t < 2.0) continue;
                double d_t  = s_d_k_total[kc * n_time_bins + t];
                double dl_t = s_d_k_left[kc * n_time_bins + t];
                double arl_t = s_at_risk_left[t];
                thread_num += dl_t - d_t * arl_t / ar_t;
                thread_var += d_t * arl_t * (ar_t - arl_t) * (ar_t - d_t)
                             / (ar_t * ar_t * (ar_t - 1.0));
            }
        }

        // Reduce num and var across block.
        s_red_num[tid] = thread_num;
        s_red_var[tid] = thread_var;
        __syncthreads();
        for (int s = blockDimX / 2; s > 0; s >>= 1) {
            if (tid < s) {
                s_red_num[tid] += s_red_num[tid + s];
                s_red_var[tid] += s_red_var[tid + s];
            }
            __syncthreads();
        }

        // tid==0 computes stat and updates best
        if (tid == 0) {
            double num_sum = s_red_num[0];
            double var_sum = s_red_var[0];
            if (var_sum >= 1e-12) {
                double stat = num_sum * num_sum / var_sum;
                if (!isnan(stat) && stat > s_best_stat[0]) {
                    s_best_stat[0] = stat;
                    s_scalars[1] = b;
                }
            }
        }
        __syncthreads();
    }

    // Write per-(node, feat) result.
    if (tid == 0) {
        long long off = (long long)node * mtry + feat;
        if (s_best_stat[0] > 0.0) {
            pf_bin[off]  = s_scalars[1];
            pf_stat[off] = s_best_stat[0];
        } else {
            pf_bin[off]  = -1;
            pf_stat[off] = 0.0;
        }
    }
}


// Tiny per-node lex-argmax over (mtry,) per-(node,feat) results.
// One block per node, mtry threads, lex tie-break: pick lowest f among ties on stat.
extern "C" __global__ void best_split_v2_reduce_features(
    const int* __restrict__ pf_bin,     // (N_nodes, mtry)
    const double* __restrict__ pf_stat, // (N_nodes, mtry)
    int* __restrict__ out_feat,         // (N_nodes,)
    int* __restrict__ out_bin,          // (N_nodes,)
    double* __restrict__ out_stat,      // (N_nodes,)
    int mtry)
{
    const int node = blockIdx.x;
    const int tid  = threadIdx.x;

    extern __shared__ char smem[];
    double* s_stat = (double*)smem;
    int*    s_feat = (int*)(s_stat + blockDim.x);
    int*    s_bin  = s_feat + blockDim.x;

    double init_stat = 0.0;
    int init_feat = -1;
    int init_bin  = 0;

    if (tid < mtry) {
        long long off = (long long)node * mtry + tid;
        double st = pf_stat[off];
        int bn = pf_bin[off];
        if (bn >= 0 && st > 0.0) {
            init_stat = st;
            init_feat = tid;
            init_bin = bn;
        }
    }
    s_stat[tid] = init_stat;
    s_feat[tid] = init_feat;
    s_bin[tid]  = init_bin;
    __syncthreads();

    // Pairwise reduction with lex tie-break (lowest f on stat tie).
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            double oa = s_stat[tid], ob = s_stat[tid + s];
            int fa = s_feat[tid], fb = s_feat[tid + s];
            int ba = s_bin[tid],  bb = s_bin[tid + s];
            bool take_other = (ob > oa) ||
                              (ob == oa && fb >= 0 && (fa < 0 || fb < fa || (fb == fa && bb < ba)));
            if (take_other) {
                s_stat[tid] = ob;
                s_feat[tid] = fb;
                s_bin[tid]  = bb;
            }
        }
        __syncthreads();
    }

    if (tid == 0) {
        if (s_feat[0] < 0) {
            out_feat[node] = -1;
            out_bin[node]  = 0;
            out_stat[node] = 0.0;
        } else {
            out_feat[node] = s_feat[0];
            out_bin[node]  = s_bin[0];
            out_stat[node] = s_stat[0];
        }
    }
}
"""


_compiled_v2_scan = None
_compiled_v2_reduce = None


def _get_v2_kernels():
    global _compiled_v2_scan, _compiled_v2_reduce
    if _compiled_v2_scan is None:
        import cupy as cp

        _compiled_v2_scan = cp.RawKernel(
            _BEST_SPLIT_V2_SRC,
            "best_split_kernel_per_node_v2_scan",
            options=("-std=c++14",),
        )
        _compiled_v2_reduce = cp.RawKernel(
            _BEST_SPLIT_V2_SRC,
            "best_split_v2_reduce_features",
            options=("-std=c++14",),
        )
    return _compiled_v2_scan, _compiled_v2_reduce


# Module caps must match the kernel #defines.
MAX_GPU_CAUSES = 8
MAX_GPU_TIME_BINS = 256


def best_split_kernel_per_node_v2(
    event_hist,
    n_at,
    cand_mask,
    out_feat,
    out_bin,
    out_stat,
    *,
    n_bins: int,
    n_causes: int,
    n_time_bins: int,
    mtry: int,
    min_samples_leaf: int,
    splitrule_code: int,
    cause: int,
):
    """Drop-in replacement for best_split_kernel_per_node using running prefix sums.

    Internally launches two kernels:
      1. Per-(node, feat) scan that produces per-feature best (bin, stat).
      2. Per-node mtry reduction that picks lex-min argmax across features.
    """
    import cupy as cp

    if n_causes > MAX_GPU_CAUSES:
        raise ValueError(f"n_causes={n_causes} > MAX_GPU_CAUSES={MAX_GPU_CAUSES}")
    if n_time_bins > MAX_GPU_TIME_BINS:
        raise ValueError(f"n_time_bins={n_time_bins} > MAX_GPU_TIME_BINS={MAX_GPU_TIME_BINS}")

    N_nodes = event_hist.shape[0]
    scan_kernel, reduce_kernel = _get_v2_kernels()

    pf_bin = cp.full((N_nodes, mtry), -1, dtype=cp.int32)
    pf_stat = cp.zeros((N_nodes, mtry), dtype=cp.float64)

    # --- Scan kernel: grid (N_nodes, mtry, 1), block (THREADS, 1, 1) ---
    threads = 64
    # Shared memory budget (in bytes):
    #   8B doubles:
    #     at_risk_left, at_risk_total, d_any_left, d_any_total: 4 * T
    #     d_k_left, d_k_total, ar_total_inc, ar_left_inc: 4 * K * T
    #     red_num, red_var: 2 * THREADS
    #     best_stat: 1
    #   plus s_scalars: 4 ints = 16 B
    smem_bytes = (
        8 * (4 * n_time_bins + 4 * n_causes * n_time_bins + 2 * threads + 1)
        + 16
        + 16  # alignment slack
    )

    grid = (N_nodes, mtry, 1)
    block = (threads, 1, 1)
    scan_kernel(
        grid,
        block,
        (
            event_hist,
            n_at,
            cand_mask,
            pf_bin,
            pf_stat,
            np.int32(n_bins),
            np.int32(n_causes),
            np.int32(n_time_bins),
            np.int32(mtry),
            np.int32(min_samples_leaf),
            np.int32(splitrule_code),
            np.int32(cause),
        ),
        shared_mem=smem_bytes,
    )

    # --- Reduce kernel: grid (N_nodes,), block (mtry_pow2,) ---
    # Reduction wants a power-of-2 block size >= mtry; use 8 since mtry=8 in
    # production and the reduction handles tid >= mtry as no-op (init).
    mtry_pow2 = 1
    while mtry_pow2 < mtry:
        mtry_pow2 <<= 1
    rd_smem = mtry_pow2 * (8 + 4 + 4)
    reduce_kernel(
        (N_nodes, 1, 1),
        (mtry_pow2, 1, 1),
        (
            pf_bin,
            pf_stat,
            out_feat,
            out_bin,
            out_stat,
            np.int32(mtry),
        ),
        shared_mem=rd_smem,
    )


# ----------------------------------------------------------------------
# Correctness check (small fixture)
# ----------------------------------------------------------------------
def correctness_check():
    import cupy as cp

    from comprisk._gpu_kernels import (
        best_split_kernel_per_node,
        histogram_kernel_per_level,
    )

    rng = np.random.default_rng(42)
    n = 200
    mtry = 4
    n_bins = 16
    n_causes = 2
    n_time_bins = 32

    # Build a small fixture: random X_binned, t_idx, event.
    X_binned_h = rng.integers(0, n_bins, size=(n, mtry), dtype=np.uint8)
    t_idx_h = rng.integers(0, n_time_bins, size=n, dtype=np.int32)
    event_h = rng.integers(0, n_causes + 1, size=n, dtype=np.int32)

    # Two nodes: split sample-perm in two ranges.
    sample_perm_h = np.arange(n, dtype=np.int32)
    node_starts_h = np.array([0, 100], dtype=np.int32)
    node_ends_h = np.array([100, 200], dtype=np.int32)
    N_nodes = 2

    # Random cand_mask, ~80% true.
    cand_mask_h = rng.random(size=(N_nodes, mtry, n_bins - 1)) < 0.8

    X_binned_d = cp.asarray(X_binned_h)
    t_idx_d = cp.asarray(t_idx_h)
    event_d = cp.asarray(event_h)
    sample_perm_d = cp.asarray(sample_perm_h)
    node_starts_d = cp.asarray(node_starts_h)
    node_ends_d = cp.asarray(node_ends_h)
    cand_mask_d = cp.asarray(cand_mask_h)

    ehist = cp.zeros((N_nodes, mtry, n_bins, n_causes, n_time_bins), dtype=cp.uint32)
    nat = cp.zeros((N_nodes, mtry, n_bins, n_time_bins), dtype=cp.uint32)
    histogram_kernel_per_level(
        X_binned_d,
        t_idx_d,
        event_d,
        sample_perm_d,
        node_starts_d,
        node_ends_d,
        ehist,
        nat,
        n_bins=n_bins,
        n_causes=n_causes,
        n_time_bins=n_time_bins,
        mtry=mtry,
    )
    cp.cuda.runtime.deviceSynchronize()

    print(
        "=== Correctness check (n=200, mtry=4, n_bins=16, n_causes=2, n_time_bins=32) ===",
        flush=True,
    )

    all_pass = True
    for splitrule_code in (0, 1):
        for cause in (1, 2) if splitrule_code == 1 else (1,):
            out_feat_v1 = cp.full((N_nodes,), -1, dtype=cp.int32)
            out_bin_v1 = cp.full((N_nodes,), -1, dtype=cp.int32)
            out_stat_v1 = cp.full((N_nodes,), -np.inf, dtype=cp.float64)
            best_split_kernel_per_node(
                ehist,
                nat,
                cand_mask_d,
                out_feat_v1,
                out_bin_v1,
                out_stat_v1,
                n_bins=n_bins,
                n_causes=n_causes,
                n_time_bins=n_time_bins,
                mtry=mtry,
                min_samples_leaf=5,
                splitrule_code=splitrule_code,
                cause=cause,
            )
            cp.cuda.runtime.deviceSynchronize()

            out_feat_v2 = cp.full((N_nodes,), -1, dtype=cp.int32)
            out_bin_v2 = cp.full((N_nodes,), -1, dtype=cp.int32)
            out_stat_v2 = cp.full((N_nodes,), -np.inf, dtype=cp.float64)
            best_split_kernel_per_node_v2(
                ehist,
                nat,
                cand_mask_d,
                out_feat_v2,
                out_bin_v2,
                out_stat_v2,
                n_bins=n_bins,
                n_causes=n_causes,
                n_time_bins=n_time_bins,
                mtry=mtry,
                min_samples_leaf=5,
                splitrule_code=splitrule_code,
                cause=cause,
            )
            cp.cuda.runtime.deviceSynchronize()

            f1 = cp.asnumpy(out_feat_v1)
            b1 = cp.asnumpy(out_bin_v1)
            s1 = cp.asnumpy(out_stat_v1)
            f2 = cp.asnumpy(out_feat_v2)
            b2 = cp.asnumpy(out_bin_v2)
            s2 = cp.asnumpy(out_stat_v2)

            label = f"splitrule={splitrule_code} cause={cause}"
            print(
                f"  v1 {label}: feat={f1.tolist()} bin={b1.tolist()} stat={[f'{x:.6f}' for x in s1.tolist()]}",
                flush=True,
            )
            print(
                f"  v2 {label}: feat={f2.tolist()} bin={b2.tolist()} stat={[f'{x:.6f}' for x in s2.tolist()]}",
                flush=True,
            )

            ok_feat = bool((f1 == f2).all())
            ok_bin = bool((b1 == b2).all())
            stat_match = bool(np.allclose(s1, s2, rtol=1e-9, atol=1e-9, equal_nan=True))
            verdict = "PASS" if (ok_feat and ok_bin and stat_match) else "FAIL"
            print(
                f"    {label}: feat_match={ok_feat} bin_match={ok_bin} stat_match={stat_match} -> {verdict}",
                flush=True,
            )
            if not (ok_feat and ok_bin):
                all_pass = False

    print(f"\nOverall correctness: {'PASS' if all_pass else 'FAIL'}", flush=True)
    return all_pass


# ----------------------------------------------------------------------
# Perf measurement (production fixture, monkey-patched best_split kernel)
# ----------------------------------------------------------------------
def _make_profiled_builder(use_v2: bool):
    """Mirror exp3b but parametrize between v1 and v2 best-split kernel.

    Profiles every per-level stage so we can see where the wall is now spent
    after the v2 best-split scan eliminates that hotspot.
    """
    import cupy as cp

    from comprisk._estimators import aalen_johansen_from_counts
    from comprisk._gpu_kernels import (
        _accumulate_leaf_cpu,
        _build_cand_mask,
        _partition_inplace,
        best_split_kernel_per_node,
        histogram_kernel_per_level,
    )
    from comprisk._tree_flat import FlatTree

    profile = defaultdict(float)
    counts = defaultdict(int)

    def _record(stage, t0, sync=False):
        if sync:
            cp.cuda.runtime.deviceSynchronize()
        profile[stage] += (time.perf_counter() - t0) * 1000.0
        counts[stage] += 1

    bs_kernel = best_split_kernel_per_node_v2 if use_v2 else best_split_kernel_per_node

    def build_flat_tree_gpu_profiled(
        X_binned,
        t_idx_split,
        t_idx_full,
        event,
        *,
        bootstrap_indices,
        n_bins,
        n_causes,
        n_time_bins_split,
        n_time_bins_full,
        min_samples_split=30,
        min_samples_leaf=15,
        max_depth=-1,
        max_features=8,
        nsplit=10,
        splitrule_code=0,
        cause=1,
        seed=0,
    ):
        rng = np.random.default_rng(seed)
        p = X_binned.shape[1]
        mtry = max_features
        td_d = cp.asarray(t_idx_split)
        ed_d = cp.asarray(event)

        sample_perm = bootstrap_indices.copy()
        n_bag = sample_perm.shape[0]
        N_max_nodes = max(64, 4 * n_bag // max(1, min_samples_leaf))
        out_features = np.zeros(N_max_nodes, dtype=np.int64)
        out_split_values = np.zeros(N_max_nodes, dtype=np.int64)
        out_left = np.zeros(N_max_nodes, dtype=np.int64)
        out_right = np.zeros(N_max_nodes, dtype=np.int64)
        out_is_leaf = np.zeros(N_max_nodes, dtype=np.bool_)
        out_leaf_idx_of_node = np.full(N_max_nodes, -1, dtype=np.int64)
        out_leaf_event_counts = np.zeros((N_max_nodes, n_causes, n_time_bins_full), dtype=np.uint32)
        out_leaf_at_risk = np.zeros((N_max_nodes, n_time_bins_full), dtype=np.uint32)

        n_nodes_used = 1
        n_leaves_used = 0
        active = [(0, 0, n_bag, 0)]
        n_levels = 0

        while active:
            n_levels += 1
            splittable, leafify = [], []
            for entry in active:
                node_idx, start, end, depth = entry
                n_node = end - start
                if n_node < min_samples_split or (max_depth >= 0 and depth >= max_depth):
                    leafify.append(entry)
                else:
                    splittable.append(entry)

            for node_idx, start, end, _depth in leafify:
                _accumulate_leaf_cpu(
                    sample_perm[start:end],
                    t_idx_full,
                    event,
                    n_causes,
                    n_time_bins_full,
                    out_leaf_event_counts,
                    out_leaf_at_risk,
                    n_leaves_used,
                )
                out_features[node_idx] = -1
                out_is_leaf[node_idx] = True
                out_leaf_idx_of_node[node_idx] = n_leaves_used
                n_leaves_used += 1

            if not splittable:
                active = []
                continue

            N_active = len(splittable)
            t0 = time.perf_counter()
            feat_perm = np.empty((N_active, mtry), dtype=np.int32)
            for i in range(N_active):
                pool = np.arange(p, dtype=np.int32)
                rng.shuffle(pool)
                feat_perm[i] = pool[:mtry]
            _record("swor_mtry", t0)

            t0 = time.perf_counter()
            node_starts_h = np.empty(N_active, dtype=np.int32)
            node_ends_h = np.empty(N_active, dtype=np.int32)
            for i, (_, start, end, _) in enumerate(splittable):
                node_starts_h[i] = start
                node_ends_h[i] = end

            cand_mask_h = _build_cand_mask(
                X_binned,
                sample_perm,
                node_starts_h,
                node_ends_h,
                feat_perm,
                n_bins,
                nsplit,
                rng,
            )
            _record("build_cand_mask", t0)

            t0 = time.perf_counter()
            sample_perm_d = cp.asarray(sample_perm.astype(np.int32))
            node_starts_d = cp.asarray(node_starts_h)
            node_ends_d = cp.asarray(node_ends_h)
            cand_mask_d = cp.asarray(cand_mask_h)
            ehist = cp.zeros((N_active, mtry, n_bins, n_causes, n_time_bins_split), dtype=cp.uint32)
            nat = cp.zeros((N_active, mtry, n_bins, n_time_bins_split), dtype=cp.uint32)
            _record("h2d_per_level", t0, sync=True)

            t0 = time.perf_counter()
            n_total = X_binned.shape[0]
            Xb_view_h = np.empty((n_total, mtry), dtype=np.uint8)
            for i, (_, start, end, _) in enumerate(splittable):
                samples = sample_perm[start:end]
                for f in range(mtry):
                    feat = feat_perm[i, f]
                    Xb_view_h[samples, f] = X_binned[samples, feat]
            _record("Xb_view_gather", t0)
            t0 = time.perf_counter()
            Xb_view_d = cp.asarray(Xb_view_h)
            _record("Xb_view_h2d", t0, sync=True)

            t0 = time.perf_counter()
            histogram_kernel_per_level(
                Xb_view_d,
                td_d,
                ed_d,
                sample_perm_d,
                node_starts_d,
                node_ends_d,
                ehist,
                nat,
                n_bins=n_bins,
                n_causes=n_causes,
                n_time_bins=n_time_bins_split,
                mtry=mtry,
            )
            _record("histogram_kernel", t0, sync=True)

            t0 = time.perf_counter()
            out_feat_d = cp.full((N_active,), -1, dtype=cp.int32)
            out_bin_d = cp.full((N_active,), -1, dtype=cp.int32)
            out_stat_d = cp.full((N_active,), -np.inf, dtype=cp.float64)
            bs_kernel(
                ehist,
                nat,
                cand_mask_d,
                out_feat_d,
                out_bin_d,
                out_stat_d,
                n_bins=n_bins,
                n_causes=n_causes,
                n_time_bins=n_time_bins_split,
                mtry=mtry,
                min_samples_leaf=min_samples_leaf,
                splitrule_code=splitrule_code,
                cause=cause,
            )
            cp.cuda.runtime.deviceSynchronize()
            _record("best_split_kernel", t0)

            t0 = time.perf_counter()
            out_feat_h = cp.asnumpy(out_feat_d)
            out_bin_h = cp.asnumpy(out_bin_d)
            _record("d2h_per_level", t0, sync=True)

            t0 = time.perf_counter()
            new_active = []
            for i, (node_idx, start, end, depth) in enumerate(splittable):
                f_sel = int(out_feat_h[i])
                bin_idx = int(out_bin_h[i])
                if f_sel < 0:
                    _accumulate_leaf_cpu(
                        sample_perm[start:end],
                        t_idx_full,
                        event,
                        n_causes,
                        n_time_bins_full,
                        out_leaf_event_counts,
                        out_leaf_at_risk,
                        n_leaves_used,
                    )
                    out_features[node_idx] = -1
                    out_is_leaf[node_idx] = True
                    out_leaf_idx_of_node[node_idx] = n_leaves_used
                    n_leaves_used += 1
                    continue
                actual_feat = int(feat_perm[i, f_sel])
                mid = _partition_inplace(sample_perm, start, end, X_binned, actual_feat, bin_idx)
                if n_nodes_used + 2 > N_max_nodes:
                    _accumulate_leaf_cpu(
                        sample_perm[start:end],
                        t_idx_full,
                        event,
                        n_causes,
                        n_time_bins_full,
                        out_leaf_event_counts,
                        out_leaf_at_risk,
                        n_leaves_used,
                    )
                    out_features[node_idx] = -1
                    out_is_leaf[node_idx] = True
                    out_leaf_idx_of_node[node_idx] = n_leaves_used
                    n_leaves_used += 1
                    continue
                li = n_nodes_used
                ri = n_nodes_used + 1
                n_nodes_used += 2
                out_features[node_idx] = actual_feat
                out_split_values[node_idx] = bin_idx
                out_left[node_idx] = li
                out_right[node_idx] = ri
                out_is_leaf[node_idx] = False
                new_active.append((li, start, mid, depth + 1))
                new_active.append((ri, mid, end, depth + 1))
            _record("apply_splits", t0)

            active = new_active

        leaf_table = np.zeros((n_leaves_used, n_causes, n_time_bins_full), dtype=np.float64)
        for k in range(n_leaves_used):
            leaf_table[k] = aalen_johansen_from_counts(
                out_leaf_event_counts[k],
                out_leaf_at_risk[k],
                n_causes,
            )

        tree = FlatTree.from_arrays(
            features=out_features[:n_nodes_used],
            split_values=out_split_values[:n_nodes_used],
            left_children=out_left[:n_nodes_used],
            right_children=out_right[:n_nodes_used],
            is_leaf_flags=out_is_leaf[:n_nodes_used],
            leaf_table=leaf_table,
            leaf_idx_of_node=out_leaf_idx_of_node[:n_nodes_used],
            leaf_event_counts=out_leaf_event_counts[:n_leaves_used].copy(),
            leaf_at_risk=out_leaf_at_risk[:n_leaves_used].copy(),
        )
        profile["__n_levels__"] = float(n_levels)
        return tree

    return build_flat_tree_gpu_profiled, profile, counts


def perf_measure():
    import cupy as cp

    from comprisk import CompetingRiskForest
    from comprisk import _gpu_kernels as _gk

    rng = np.random.default_rng(0)
    n, p = 100_000, 8
    X = rng.uniform(size=(n, p))
    t = rng.exponential(1.0, n) + 0.1
    e = rng.integers(0, 3, n)

    print(
        "\n=== PERF MEASUREMENT (n=100k, single tree, mtry=8, n_bins=32, n_time_bins=50) ===",
        flush=True,
    )

    print("[warm] compile both kernels + prime memory pool ...", flush=True)
    CompetingRiskForest(n_estimators=1, device="cuda", random_state=99).fit(
        X[:5000], t[:5000], e[:5000]
    )
    cp.cuda.runtime.deviceSynchronize()

    # Pre-compile v2 kernels too via a small launch.
    _ = _get_v2_kernels()
    cp.cuda.runtime.deviceSynchronize()

    stage_order = [
        "swor_mtry",
        "build_cand_mask",
        "h2d_per_level",
        "Xb_view_gather",
        "Xb_view_h2d",
        "histogram_kernel",
        "best_split_kernel",
        "d2h_per_level",
        "apply_splits",
    ]

    results = {}
    profiles = {}
    for use_v2 in (False, True):
        label = "v2" if use_v2 else "v1"
        builder, profile, _counts = _make_profiled_builder(use_v2)
        orig = _gk.build_flat_tree_gpu
        _gk.build_flat_tree_gpu = builder
        try:
            t0 = time.perf_counter()
            CompetingRiskForest(n_estimators=1, device="cuda", random_state=0).fit(X, t, e)
            cp.cuda.runtime.deviceSynchronize()
            wall_ms = (time.perf_counter() - t0) * 1000.0
            bs_ms = profile.get("best_split_kernel", 0.0)
            n_levels = int(profile.get("__n_levels__", 0))
            print(f"\n  [{label}] outer wall: {wall_ms:.1f} ms   ({n_levels} levels)", flush=True)
            for s in stage_order:
                ms = profile.get(s, 0.0)
                pct = ms / wall_ms * 100 if wall_ms else 0
                print(f"    {s:<22} {ms:8.2f} ms  ({pct:5.1f}%)", flush=True)
            results[label] = (wall_ms, bs_ms, n_levels)
            profiles[label] = dict(profile)
        finally:
            _gk.build_flat_tree_gpu = orig

    return results, profiles


def decision_gate(results):
    v1_wall, v1_bs, _ = results["v1"]
    v2_wall, v2_bs, _ = results["v2"]
    speedup_total = v1_wall / v2_wall if v2_wall > 0 else float("inf")
    speedup_bs = v1_bs / v2_bs if v2_bs > 0 else float("inf")

    print("\n=== PERF TABLE ===", flush=True)
    print(f"{'metric':<22} {'v1 (ms)':>10} {'v2 (ms)':>10} {'speedup':>10}", flush=True)
    print("-" * 56, flush=True)
    print(
        f"{'best_split_kernel':<22} {v1_bs:>10.1f} {v2_bs:>10.1f} {speedup_bs:>9.2f}x", flush=True
    )
    print(
        f"{'total tree wall':<22} {v1_wall:>10.1f} {v2_wall:>10.1f} {speedup_total:>9.2f}x",
        flush=True,
    )

    print("\n=== DECISION GATE ===", flush=True)
    if v2_wall <= 100:
        verdict = "HARD-PASS (<=100 ms)"
    elif v2_wall <= 200:
        verdict = "SOFT-PASS (100..200 ms)"
    elif v2_wall <= 500:
        verdict = "PARTIAL (200..500 ms; significant speedup, surface to user)"
    else:
        verdict = "FAIL (>500 ms; re-spec needed)"
    print(f"  v2 total = {v2_wall:.1f} ms -> {verdict}", flush=True)
    return verdict


def main():
    print(f"Python: {sys.version.split()[0]}", flush=True)
    import cupy as cp

    print(f"CuPy:   {cp.__version__}", flush=True)
    print(f"GPU:    {cp.cuda.runtime.getDeviceProperties(0)['name'].decode()}", flush=True)
    print("", flush=True)

    ok = correctness_check()
    if not ok:
        print("\nCorrectness check FAILED — perf measurement skipped.", flush=True)
        sys.exit(1)

    results, _profiles = perf_measure()
    decision_gate(results)


if __name__ == "__main__":
    main()
