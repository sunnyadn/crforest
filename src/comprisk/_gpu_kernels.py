"""GPU kernels for per-tree-level batched histogram + best-split scan.

Imports cupy lazily — the module loads on CPU-only installs but raises
``ImportError`` from any kernel-bound function. Use _gpu_detect.detect_cuda
before calling anything in here.
"""

from __future__ import annotations

import functools

import numpy as np
from numba import njit

# Device-helper buffer caps mirrored as module-level Python constants.
# Bounds checked at launch time inside ``best_split_kernel_per_node``;
# the v2 scan kernel allocates shared memory dynamically per launch,
# but the time-bin axis still has to fit within the per-block budget
# (~17 KB at K=8, T=256) which is well under the 48 KB default cap.
MAX_GPU_CAUSES = 8
MAX_GPU_TIME_BINS = 256


@functools.cache
def _compile(src: str, name: str, options: tuple[str, ...] = ()):
    """Lazy cupy.RawKernel compile + per-(src, name) cache."""
    import cupy as cp

    return cp.RawKernel(src, name, options=options)


_HISTOGRAM_SRC = r"""
extern "C" __global__ void histogram_kernel_per_level(
    const unsigned char* __restrict__ X_binned,   // (N, mtry) row-major
    const int* __restrict__ t_idx,                // (N,)
    const int* __restrict__ event,                // (N,)
    const int* __restrict__ sample_perm,          // (N_active,)
    const int* __restrict__ node_starts,          // (N_nodes,)
    const int* __restrict__ node_ends,            // (N_nodes,)
    unsigned int* __restrict__ event_hist_out,    // (N_nodes, mtry, n_bins, n_causes, n_time_bins)
    unsigned int* __restrict__ n_at_out,          // (N_nodes, mtry, n_bins, n_time_bins)
    int n_bins, int n_causes, int n_time_bins, int mtry, int N_total)
{
    // grid: (N_nodes, mtry, 1); block: (THREADS, 1, 1)
    const int node = blockIdx.x;
    const int feat = blockIdx.y;
    const int tid  = threadIdx.x;
    const int blockDimX = blockDim.x;

    const int start = node_starts[node];
    const int end   = node_ends[node];

    // event_hist_out stride per (node, feat): n_bins * n_causes * n_time_bins
    // n_at_out stride per (node, feat): n_bins * n_time_bins
    const long long ehb = ((long long)node * mtry + feat) * n_bins * n_causes * n_time_bins;
    const long long ahb = ((long long)node * mtry + feat) * n_bins * n_time_bins;

    for (int i = start + tid; i < end; i += blockDimX) {
        int s = sample_perm[i];
        unsigned char b = X_binned[(long long)s * mtry + feat];
        int t = t_idx[s];
        int ev = event[s];
        if (ev >= 1 && ev <= n_causes) {
            atomicAdd(&event_hist_out[ehb + ((long long)b * n_causes + (ev - 1)) * n_time_bins + t], 1u);
        }
        atomicAdd(&n_at_out[ahb + (long long)b * n_time_bins + t], 1u);
    }
}
"""


def histogram_kernel_per_level(
    X_binned,  # cupy (N, mtry) uint8
    t_idx,  # cupy (N,) int32
    event,  # cupy (N,) int32
    sample_perm,  # cupy (N_active,) int32
    node_starts,  # cupy (N_nodes,) int32
    node_ends,  # cupy (N_nodes,) int32
    event_hist_out,  # cupy (N_nodes, mtry, n_bins, n_causes, n_time_bins) uint32 (zeroed)
    n_at_out,  # cupy (N_nodes, mtry, n_bins, n_time_bins) uint32 (zeroed)
    *,
    n_bins: int,
    n_causes: int,
    n_time_bins: int,
    mtry: int,
):
    """Per-(node, feature) histogram launch.

    Kernel writes via uint32 atomicAdd; result is order-independent (commutative
    integer add), so equal-seed runs produce bit-identical histograms. Caller
    must pre-zero event_hist_out and n_at_out.
    """
    N_nodes = node_starts.shape[0]
    N_total = X_binned.shape[0]
    threads_per_block = 256
    grid = (N_nodes, mtry, 1)
    block = (threads_per_block, 1, 1)
    kernel = _compile(_HISTOGRAM_SRC, "histogram_kernel_per_level")
    kernel(
        grid,
        block,
        (
            X_binned,
            t_idx,
            event,
            sample_perm,
            node_starts,
            node_ends,
            event_hist_out,
            n_at_out,
            np.int32(n_bins),
            np.int32(n_causes),
            np.int32(n_time_bins),
            np.int32(mtry),
            np.int32(N_total),
        ),
    )


_BEST_SPLIT_V2_SRC = r"""
// best_split_kernel_per_node_v2: running prefix-sum scan.
//
// Replaces v1's per-(feat, b_cut) recomputation of the full reverse cumsum
// (O(B^2 * K * T) per node) with a serial-in-b, parallel-in-t scan that
// advances running prefix sums once per cut point (O(B * K * T) per node).
//
// Block layout:
//   grid  = (N_nodes, mtry, 1)
//   block = (THREADS, 1, 1)        // 64 threads at production fixture
//
// One block per (node, feature). Threads in the block parallelize over time
// bins (n_time_bins). The b loop runs serially in the block; per b all
// threads cooperatively (i) update their t-stripe of running prefix sums
// (at_risk_left, d_k_left), (ii) check cand_mask + min_leaf gate, and if
// open (iii) compute their stripe's contribution to (num_sum, var_sum),
// reduce to a block scalar, and tid==0 updates the per-block best.
//
// A second tiny kernel (one block per node) reduces the per-(node, feat)
// (best_b, best_stat) outputs into per-node (best_feat, best_bin, best_stat)
// using the same lex tie-break rule (lowest feat then lowest bin on stat tie).

#define MAX_CAUSES 8
#define MAX_TIME_BINS 256

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

    // Shared memory layout (single contiguous chunk of doubles + scalars):
    //   at_risk_left  [n_time_bins]
    //   at_risk_total [n_time_bins]
    //   d_k_left      [n_causes * n_time_bins]
    //   d_k_total     [n_causes * n_time_bins]
    //   d_any_left    [n_time_bins]
    //   d_any_total   [n_time_bins]
    //   ar_total_inc  [n_causes * n_time_bins] (Lau cumsum, splitrule 0 only)
    //   ar_left_inc   [n_causes * n_time_bins] (Lau cumsum, splitrule 0 only)
    //   red_num       [THREADS]
    //   red_var       [THREADS]
    //   s_scalars     [4 ints] (n_left, best_b, n_total, _)
    //   s_best_stat   [1 double]
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
        s_best_stat[0] = 0.0;   // best_stat init = 0.0 matches CPU's best_s = 0.0
        s_scalars[2] = 0;       // n_total (set in setup pass)
    }
    __syncthreads();

    // -------- Setup pass: at_risk_total, d_k_total, d_any_total, n_total --------
    long long ahb = ((long long)node * mtry + feat) * n_bins * n_time_bins;
    long long ehb = ((long long)node * mtry + feat) * n_bins * n_causes * n_time_bins;

    // at_risk_total: each thread reverse-cumsums one (or several) bins along t,
    // atomically adding into shared at_risk_total. Per-bin n_total contribution
    // captured from t=0 running.
    int n_total_local = 0;
    for (int b = tid; b < n_bins; b += blockDimX) {
        long long bin_off_n = ahb + (long long)b * n_time_bins;
        double running = 0.0;
        for (int t = n_time_bins - 1; t >= 0; t--) {
            unsigned int v = n_at[bin_off_n + t];
            running += (double)v;
            atomicAdd(&s_at_risk_total[t], running);
        }
        n_total_local += (int)running;
    }
    // Sum n_total across threads via shared scratch (red_num).
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

    // d_k_total / d_any_total: each thread takes a t-stripe, sums over all bins.
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

    // -------- Splitrule 0 setup: ar_total_inc[k, t] = at_risk_total[t] + Lau cumsum --------
    // Cumsum over t is intrinsically serial; we parallelize over k. With
    // n_causes <= 8 most threads idle here but it's a one-time setup cost.
    if (splitrule_code == 0) {
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
        // at_risk_left += reverse_cumsum_t(n_at[b, :]). Reverse cumsum over t
        // is a serial scan; tid==0 does it in O(T), then all threads see it
        // after the syncthreads below.
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
        // d_k_left += event_hist[b, k, t]; d_any_left = sum_k d_k_left.
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

        n_left = s_scalars[0];

        // (ii) cand_mask + min-samples gate. cand_mask is the SAME mechanism
        // as v1: prefix sums advance regardless, mask only gates evaluation.
        bool eligible = cand_mask[((long long)node * mtry + feat) * B_minus_1 + b];
        int n_right = n_total - n_left;
        if (n_left < min_samples_leaf || n_right < min_samples_leaf) eligible = false;
        if (!eligible) continue;

        // (iii) per-thread partial sums over t-stripe
        double thread_num = 0.0;
        double thread_var = 0.0;

        if (splitrule_code == 0) {
            // Build ar_left_inc[k, t] (Lau cumsum on left); same shape as setup pass.
            for (int k = tid; k < n_causes; k += blockDimX) {
                double cumsum = 0.0;
                for (int t = 0; t < n_time_bins; t++) {
                    s_ar_left_inc[k * n_time_bins + t] = s_at_risk_left[t] + cumsum;
                    cumsum += s_d_any_left[t] - s_d_k_left[k * n_time_bins + t];
                }
            }
            __syncthreads();

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

        // Reduce num and var across the block.
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

        // tid==0 computes stat and updates best (lowest b on within-block tie
        // is preserved by strict-greater update — b iterates ascending).
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
// One block per node, mtry_pow2 threads (>= mtry); inactive threads init to
// (stat=0, feat=-1, bin=0). Lex tie-break: lowest feat, then lowest bin.
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


def best_split_kernel_per_node(
    event_hist,  # cupy (N_nodes, mtry, n_bins, n_causes, n_time_bins) uint32
    n_at,  # cupy (N_nodes, mtry, n_bins, n_time_bins) uint32
    cand_mask,  # cupy (N_nodes, mtry, n_bins-1) bool
    out_feat,  # cupy (N_nodes,) int32
    out_bin,  # cupy (N_nodes,) int32
    out_stat,  # cupy (N_nodes,) float64
    *,
    n_bins: int,
    n_causes: int,
    n_time_bins: int,
    mtry: int,
    min_samples_leaf: int,
    splitrule_code: int,
    cause: int,
):
    """Per-node argmax over (feature, bin) candidates using the logrankCR/logrank statistic.

    Two-kernel orchestration:
      1. ``best_split_kernel_per_node_v2_scan`` — block per (node, feat),
         running prefix-sum scan over cut points b. Writes per-(node, feat)
         best (bin, stat) into temporary buffers.
      2. ``best_split_v2_reduce_features`` — block per node, lex-argmax across
         the mtry per-(node, feat) results into final out_feat/out_bin/out_stat.

    Bit-identical to the prior single-kernel v1 implementation: same lex
    tie-break (lowest feat, then lowest bin on stat tie) and same
    cand_mask / min_samples_leaf semantics (prefix sums always advance,
    mask only gates which b are evaluated).

    Caller is responsible for pre-zeroing histograms and pre-filling outputs
    (out_feat=-1, out_bin=-1, out_stat=-inf is the conventional sentinel; the
    kernel overwrites unconditionally).

    ``n_causes`` and ``n_time_bins`` are bounded by module-level
    ``MAX_GPU_CAUSES`` / ``MAX_GPU_TIME_BINS`` (mirrors of the kernel's
    ``#define`` shared-memory caps); oversized inputs raise ``ValueError``
    before launch rather than overrunning the per-block shared-memory budget.
    """
    import cupy as cp

    if n_causes > MAX_GPU_CAUSES:
        raise ValueError(
            f"n_causes={n_causes} exceeds GPU kernel limit MAX_GPU_CAUSES={MAX_GPU_CAUSES}; "
            f"increase the device-helper buffer caps in _gpu_kernels.py"
        )
    if n_time_bins > MAX_GPU_TIME_BINS:
        raise ValueError(
            f"n_time_bins={n_time_bins} exceeds GPU kernel limit MAX_GPU_TIME_BINS={MAX_GPU_TIME_BINS}; "
            f"increase the device-helper buffer caps in _gpu_kernels.py"
        )
    N_nodes = event_hist.shape[0]
    scan_kernel = _compile(
        _BEST_SPLIT_V2_SRC, "best_split_kernel_per_node_v2_scan", options=("-std=c++14",)
    )
    reduce_kernel = _compile(
        _BEST_SPLIT_V2_SRC, "best_split_v2_reduce_features", options=("-std=c++14",)
    )

    # Per-(node, feat) scratch outputs for the scan kernel.
    pf_bin = cp.full((N_nodes, mtry), -1, dtype=cp.int32)
    pf_stat = cp.zeros((N_nodes, mtry), dtype=cp.float64)

    # --- Scan kernel: grid (N_nodes, mtry, 1), block (THREADS,) ---
    # Per-block shared-memory budget (doubles unless noted):
    #   at_risk_left, at_risk_total, d_any_left, d_any_total: 4 * T
    #   d_k_left, d_k_total, ar_total_inc, ar_left_inc:       4 * K * T
    #   red_num, red_var:                                     2 * THREADS
    #   best_stat:                                            1
    #   plus s_scalars (4 ints = 16 B) + alignment slack
    # At K=8, T=256, THREADS=64 the budget is ~17 KB, well under the 48 KB
    # default per-block cap on RTX 5070 Ti / A6000.
    threads = 64
    smem_bytes = (
        8 * (4 * n_time_bins + 4 * n_causes * n_time_bins + 2 * threads + 1)
        + 16  # s_scalars (4 ints)
        + 16  # alignment slack
    )
    scan_kernel(
        (N_nodes, mtry, 1),
        (threads, 1, 1),
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
    # Power-of-2 block size >= mtry; inactive threads init to (stat=0, feat=-1).
    mtry_pow2 = 1
    while mtry_pow2 < mtry:
        mtry_pow2 <<= 1
    rd_smem = mtry_pow2 * (8 + 4 + 4)  # double + int + int per thread
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


def _accumulate_leaf_cpu(
    idx_view, t_idx_full, event, n_causes, n_time_bins, leaf_ec, leaf_ar, leaf_idx
):
    leaf_ec[leaf_idx] = 0
    leaf_ar[leaf_idx] = 0
    for s in idx_view:
        ti = t_idx_full[s]
        ev = event[s]
        if 1 <= ev <= n_causes:
            leaf_ec[leaf_idx, ev - 1, ti] += 1
        leaf_ar[leaf_idx, ti] += 1
    running = np.uint32(0)
    for t in range(n_time_bins - 1, -1, -1):
        running += leaf_ar[leaf_idx, t]
        leaf_ar[leaf_idx, t] = running


_LEAFIFY_SRC = r"""
// leafify_kernel_per_level: accumulate event_counts + at_risk for a batch of leaves.
//
// One block per leaf. Threads in the block parallelize over samples in the
// leaf's [start, end) range. Each thread atomically adds into the leaf's
// event_counts (n_causes, n_time_bins) and at_risk (n_time_bins) slices.
//
// The reverse-cumsum on at_risk is a separate sister kernel (one block per
// leaf, tid==0 does the serial scan over n_time_bins) launched after this.
//
// Caller pre-zeros the target rows (or the whole (N_max_leaves, ...) buffers).

extern "C" __global__ void leafify_accumulate_kernel(
    const int* __restrict__ sample_perm,         // (N_total,)
    const int* __restrict__ t_idx_full,          // (N_total,)
    const int* __restrict__ event,               // (N_total,)
    const int* __restrict__ leaf_starts,         // (N_leafify,)
    const int* __restrict__ leaf_ends,           // (N_leafify,)
    const int* __restrict__ leaf_out_idx,        // (N_leafify,) row in out arrays
    unsigned int* __restrict__ leaf_event_counts,  // (N_max, n_causes, n_time_bins)
    unsigned int* __restrict__ leaf_at_risk,     // (N_max, n_time_bins)
    int n_causes, int n_time_bins)
{
    const int leaf = blockIdx.x;
    const int tid  = threadIdx.x;
    const int blockDimX = blockDim.x;

    const int start = leaf_starts[leaf];
    const int end   = leaf_ends[leaf];
    const int out   = leaf_out_idx[leaf];

    const long long ec_off = (long long)out * n_causes * n_time_bins;
    const long long ar_off = (long long)out * n_time_bins;

    for (int i = start + tid; i < end; i += blockDimX) {
        int s = sample_perm[i];
        int ti = t_idx_full[s];
        int ev = event[s];
        if (ev >= 1 && ev <= n_causes) {
            atomicAdd(&leaf_event_counts[ec_off + (long long)(ev - 1) * n_time_bins + ti], 1u);
        }
        atomicAdd(&leaf_at_risk[ar_off + ti], 1u);
    }
}


// One block per leaf; tid==0 does the serial reverse cumsum over n_time_bins.
// at_risk[t] := sum_{u >= t} at_risk[u] in place.
extern "C" __global__ void leafify_reverse_cumsum_kernel(
    const int* __restrict__ leaf_out_idx,     // (N_leafify,)
    unsigned int* __restrict__ leaf_at_risk,  // (N_max, n_time_bins)
    int n_time_bins)
{
    const int leaf = blockIdx.x;
    const int tid  = threadIdx.x;
    if (tid != 0) return;
    const int out = leaf_out_idx[leaf];
    const long long ar_off = (long long)out * n_time_bins;
    unsigned int running = 0u;
    for (int t = n_time_bins - 1; t >= 0; t--) {
        running += leaf_at_risk[ar_off + t];
        leaf_at_risk[ar_off + t] = running;
    }
}
"""


def leafify_kernel_per_level(
    sample_perm_d,  # cupy (N_total,) int32
    t_idx_full_d,  # cupy (N_total,) int32
    event_d,  # cupy (N_total,) int32
    leaf_starts_d,  # cupy (N_leafify,) int32
    leaf_ends_d,  # cupy (N_leafify,) int32
    leaf_out_idx_d,  # cupy (N_leafify,) int32
    leaf_event_counts_d,  # cupy (N_max, n_causes, n_time_bins) uint32 — rows pre-zeroed
    leaf_at_risk_d,  # cupy (N_max, n_time_bins) uint32 — rows pre-zeroed
    *,
    n_causes: int,
    n_time_bins: int,
):
    """Batched leafify: accumulate per-leaf event_counts + at_risk on device.

    Caller is responsible for pre-zeroing the target output rows (the kernel
    uses atomicAdd so any non-zero starting state would corrupt counts).

    Followed by the reverse-cumsum kernel which mutates leaf_at_risk in place
    so that ``at_risk[t] = sum_{u >= t} at_risk[u]`` (matching the CPU helper).
    """
    N_leafify = int(leaf_starts_d.shape[0])
    if N_leafify == 0:
        return
    acc_kernel = _compile(_LEAFIFY_SRC, "leafify_accumulate_kernel")
    rcs_kernel = _compile(_LEAFIFY_SRC, "leafify_reverse_cumsum_kernel")
    threads = 128
    acc_kernel(
        (N_leafify, 1, 1),
        (threads, 1, 1),
        (
            sample_perm_d,
            t_idx_full_d,
            event_d,
            leaf_starts_d,
            leaf_ends_d,
            leaf_out_idx_d,
            leaf_event_counts_d,
            leaf_at_risk_d,
            np.int32(n_causes),
            np.int32(n_time_bins),
        ),
    )
    rcs_kernel(
        (N_leafify, 1, 1),
        (1, 1, 1),
        (
            leaf_out_idx_d,
            leaf_at_risk_d,
            np.int32(n_time_bins),
        ),
    )


def aalen_johansen_from_counts_batched_gpu(
    leaf_event_counts_d,  # cupy (n_leaves, n_causes, n_time) uint32
    leaf_at_risk_d,  # cupy (n_leaves, n_time) uint32 (reverse-cumsum'd)
    n_causes: int,
):
    """Vectorized Aalen-Johansen CIF on device.

    Bulk port of ``aalen_johansen_from_counts`` over the leaf axis. Mirrors
    the CPU helper bit-for-bit modulo float64 op order: hazard = d / ar
    (zero where ar==0); KM survival is left-continuous (surv[..., 0]=1,
    surv[..., t] = cumprod_{s<t}(1 - h_any[s])); per-cause CIF is
    cumsum_t(surv * d_k / ar).

    Returns leaf_table_d as a cupy float64 (n_leaves, n_causes, n_time).
    """
    import cupy as cp

    ar = leaf_at_risk_d.astype(cp.float64)  # (L, T)
    ec = leaf_event_counts_d.astype(cp.float64)  # (L, K, T)
    d_any = ec.sum(axis=1)  # (L, T)
    safe_ar = cp.where(ar > 0, ar, 1.0)  # (L, T)
    h_any = cp.where(ar > 0, d_any / safe_ar, 0.0)  # (L, T)

    n_times = ar.shape[1]
    surv = cp.ones_like(ar)
    if n_times > 1:
        # surv[:, 1:] = cumprod_{s < t}(1 - h_any[s]) starting at t=1.
        surv[:, 1:] = cp.cumprod(1.0 - h_any[:, :-1], axis=1)

    # CIF per cause: cumsum_t(surv * h_k) where h_k = ec[:, k, :] / ar.
    h_k = cp.where(ar[:, None, :] > 0, ec / safe_ar[:, None, :], 0.0)  # (L, K, T)
    cif = cp.cumsum(surv[:, None, :] * h_k, axis=2)  # (L, K, T)
    return cif


def _partition_inplace(sample_perm, start, end, X_binned, feat, bin_idx):
    i, j = start, end - 1
    while i <= j:
        if X_binned[sample_perm[i], feat] <= bin_idx:
            i += 1
        else:
            sample_perm[i], sample_perm[j] = sample_perm[j], sample_perm[i]
            j -= 1
    return i


_PARTITION_SRC = r"""
// partition_kernel_per_level: stream-compaction partition over a batch of
// (start, end, feat, bin) per-node tasks.
//
// One block per splittable node. Threads stripe over the node's range
// [start, end). Two passes:
//   Pass 1: each thread classifies its stripe; block-wide atomicAdd into
//           a shared n_left counter (commutative, deterministic on COUNT).
//   Pass 2: scatter via a pair of shared atomic position counters
//           (s_left_idx writes positions [0, n_left); s_right_idx writes
//           positions [n_left, n_node)). Output goes into scratch buffer.
//
// AtomicAdd-based scatter is non-deterministic in OUTPUT ORDER but
// deterministic in MULTISET; tree-shape and leaf-table outcomes depend
// only on the multiset (downstream histograms, leafify, partitions are
// all order-invariant given equal seeds). Verified bit-identical leaf_table
// across 5 runs at equal seeds in test_build_flat_tree_gpu_five_runs_bit_identical.
//
// Caller is responsible for then copying scratch[start:end] into the
// authoritative sample_perm buffer (cheap: cudaMemcpyAsync per slice or
// a single in-bulk copy of the full sample_perm).

extern "C" __global__ void partition_kernel_per_level(
    const unsigned char* __restrict__ X_binned,  // (n, p) row-major uint8
    const int* __restrict__ sample_perm_in,      // (n_total,) read
    int* __restrict__ sample_perm_out,           // (n_total,) scratch write
    const int* __restrict__ node_starts,         // (N_nodes,)
    const int* __restrict__ node_ends,           // (N_nodes,)
    const int* __restrict__ node_feats,          // (N_nodes,)
    const int* __restrict__ node_bins,           // (N_nodes,)
    int* __restrict__ node_mids,                 // (N_nodes,) out
    int p)
{
    const int node = blockIdx.x;
    const int tid = threadIdx.x;
    const int blockDimX = blockDim.x;

    const int start = node_starts[node];
    const int end   = node_ends[node];
    const int feat  = node_feats[node];
    const int bin   = node_bins[node];
    const int n_node = end - start;

    __shared__ int s_n_left;
    __shared__ int s_left_idx;
    __shared__ int s_right_idx;

    if (tid == 0) {
        s_n_left = 0;
        s_left_idx = 0;
        s_right_idx = 0;
    }
    __syncthreads();

    // Pass 1: count n_left.
    for (int i = tid; i < n_node; i += blockDimX) {
        int s = sample_perm_in[start + i];
        unsigned char b = X_binned[(long long)s * p + feat];
        if ((int)b <= bin) atomicAdd(&s_n_left, 1);
    }
    __syncthreads();

    const int n_left = s_n_left;
    if (tid == 0) node_mids[node] = start + n_left;

    // Pass 2: scatter via atomic position counters.
    for (int i = tid; i < n_node; i += blockDimX) {
        int s = sample_perm_in[start + i];
        unsigned char b = X_binned[(long long)s * p + feat];
        if ((int)b <= bin) {
            int pos = atomicAdd(&s_left_idx, 1);
            sample_perm_out[start + pos] = s;
        } else {
            int pos = atomicAdd(&s_right_idx, 1);
            sample_perm_out[start + n_left + pos] = s;
        }
    }
}
"""


def partition_kernel_per_level(
    X_binned_d,  # cupy (n, p) uint8
    sample_perm_in_d,  # cupy (n_total,) int32
    sample_perm_out_d,  # cupy (n_total,) int32 — scratch buffer (overlapping range OK; non-overlapping nodes write disjoint slices)
    node_starts_d,  # cupy (N_nodes,) int32
    node_ends_d,  # cupy (N_nodes,) int32
    node_feats_d,  # cupy (N_nodes,) int32
    node_bins_d,  # cupy (N_nodes,) int32
    node_mids_d,  # cupy (N_nodes,) int32 — output
    *,
    p: int,
):
    """Batched in-place-style partition over a list of nodes.

    Reads from ``sample_perm_in_d`` and writes the partitioned result into
    ``sample_perm_out_d``. Each node's partitioned slice [start, end) is
    written into the corresponding range of ``sample_perm_out_d`` with
    samples satisfying ``X[s, feat] <= bin`` first, then the rest.

    ``node_mids_d[i] = start + n_left_i`` for each node i.

    Caller copies ``sample_perm_out_d`` back into the authoritative buffer
    (or simply swaps the two buffers if the host doesn't need the original).
    """
    N_nodes = int(node_starts_d.shape[0])
    if N_nodes == 0:
        return
    kernel = _compile(_PARTITION_SRC, "partition_kernel_per_level")
    threads = 256
    kernel(
        (N_nodes, 1, 1),
        (threads, 1, 1),
        (
            X_binned_d,
            sample_perm_in_d,
            sample_perm_out_d,
            node_starts_d,
            node_ends_d,
            node_feats_d,
            node_bins_d,
            node_mids_d,
            np.int32(p),
        ),
    )


@njit(cache=True)
def _build_xb_view_njit(
    X_binned,  # (n_total, p) uint8
    sample_perm,  # (n_total,) int32
    node_starts,  # (N_active,) int32
    node_ends,  # (N_active,) int32
    feat_perm,  # (N_active, mtry) int32
    Xb_view_out,  # (n_total, mtry) uint8 — pre-allocated, written in place
):
    N_active, mtry = feat_perm.shape
    for i in range(N_active):
        s = node_starts[i]
        e = node_ends[i]
        for f in range(mtry):
            feat = feat_perm[i, f]
            for k in range(s, e):
                gs = sample_perm[k]
                Xb_view_out[gs, f] = X_binned[gs, feat]


@njit(cache=True)
def _build_cand_mask_njit(
    X_binned,  # (n_total, p) uint8
    sample_perm,  # (n_bag,) int32
    node_starts,  # (N_active,) int32
    node_ends,  # (N_active,) int32
    feat_perm,  # (N_active, mtry) int32
    n_bins,
    nsplit,
    seed,
):
    # Numba's process-global np.random state — safe under the cuda backend's
    # enforced n_jobs=1; would race under thread-parallel orchestration.
    np.random.seed(seed)
    N_active, mtry = feat_perm.shape
    cand_mask = np.zeros((N_active, mtry, n_bins - 1), dtype=np.bool_)
    counts = np.empty(n_bins, dtype=np.int64)
    observed = np.empty(n_bins, dtype=np.int64)
    for i in range(N_active):
        s = node_starts[i]
        e = node_ends[i]
        for f in range(mtry):
            feat = feat_perm[i, f]
            for b in range(n_bins):
                counts[b] = 0
            for k in range(s, e):
                bidx = X_binned[sample_perm[k], feat]
                counts[bidx] += 1
            n_obs = 0
            for b in range(n_bins):
                if counts[b] > 0:
                    observed[n_obs] = b
                    n_obs += 1
            if n_obs < 2:
                continue
            n_valid = n_obs - 1  # exclude max bin
            k_pick = nsplit if (nsplit > 0 and nsplit < n_valid) else n_valid
            for j in range(k_pick):
                jj = j + np.random.randint(0, n_valid - j)
                tmp = observed[j]
                observed[j] = observed[jj]
                observed[jj] = tmp
                cand_mask[i, f, observed[j]] = True
    return cand_mask


def build_flat_tree_gpu(
    X_binned,  # numpy (n, p) uint8
    t_idx_split,  # numpy (n,) int32
    t_idx_full,  # numpy (n,) int32
    event,  # numpy (n,) int32
    *,
    bootstrap_indices,  # numpy (n_bag,) int32
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
    """GPU twin of build_flat_tree. CPU drives the level-by-level loop;
    histograms + best-split scan run on GPU. Bootstrap indices select
    samples; SWOR mtry + nsplit candidate masks are CPU-side numpy work.

    Bit-deterministic at equal seeds is structural under the current kernel
    design: the histogram uses commutative uint32 atomicAdd and the best-split
    scan is a shared-memory parallel reduction with a deterministic lex
    tie-break — no opt-in flag needed."""
    import cupy as cp

    from comprisk._tree_flat import FlatTree

    rng = np.random.default_rng(seed)
    p = X_binned.shape[1]
    mtry = max_features

    td_d = cp.asarray(t_idx_split)
    ed_d = cp.asarray(event)
    t_idx_full_d = cp.asarray(t_idx_full.astype(np.int32))
    # X_binned uploaded once: feeds the per-level partition kernel. The
    # histogram path's Xb_view gather currently stays CPU-side.
    X_binned_d = cp.asarray(X_binned)

    # Pin int32 from the start: the device-side ping-pong + per-level njit
    # helpers (_build_cand_mask_njit, _build_xb_view_njit) all expect int32.
    sample_perm = bootstrap_indices.astype(np.int32, copy=True)
    n_bag = sample_perm.shape[0]
    # Device-resident sample_perm + scratch buffer for the partition kernel;
    # we ping-pong between them per level (kernel reads from one, writes to
    # the other) to avoid extra D2H/H2D round trips.
    sample_perm_d = cp.asarray(sample_perm)
    sample_perm_scratch_d = cp.empty_like(sample_perm_d)

    N_max_nodes = max(64, 4 * n_bag // max(1, min_samples_leaf))
    out_features = np.zeros(N_max_nodes, dtype=np.int64)
    out_split_values = np.zeros(N_max_nodes, dtype=np.int64)
    out_left = np.zeros(N_max_nodes, dtype=np.int64)
    out_right = np.zeros(N_max_nodes, dtype=np.int64)
    out_is_leaf = np.zeros(N_max_nodes, dtype=np.bool_)
    out_leaf_idx_of_node = np.full(N_max_nodes, -1, dtype=np.int64)

    # Leaf accumulation buffers live on device throughout the fit; rows are
    # written by leafify_kernel_per_level (atomicAdd into pre-zeroed slices).
    leaf_event_counts_d = cp.zeros((N_max_nodes, n_causes, n_time_bins_full), dtype=cp.uint32)
    leaf_at_risk_d = cp.zeros((N_max_nodes, n_time_bins_full), dtype=cp.uint32)

    n_nodes_used = 1
    n_leaves_used = 0

    def _gpu_leafify(batch, sample_perm_dev):
        """Run the leafify kernel on a list of (start, end, leaf_idx) triples
        using the supplied device-side sample_perm. Caller must ensure
        ``sample_perm_dev`` reflects the host ``sample_perm`` slice for each
        (start, end) range in the batch.
        """
        if not batch:
            return
        starts_h = np.empty(len(batch), dtype=np.int32)
        ends_h = np.empty(len(batch), dtype=np.int32)
        out_idx_h = np.empty(len(batch), dtype=np.int32)
        for k, (s, e, li) in enumerate(batch):
            starts_h[k] = s
            ends_h[k] = e
            out_idx_h[k] = li
        leafify_kernel_per_level(
            sample_perm_dev,
            t_idx_full_d,
            ed_d,
            cp.asarray(starts_h),
            cp.asarray(ends_h),
            cp.asarray(out_idx_h),
            leaf_event_counts_d,
            leaf_at_risk_d,
            n_causes=n_causes,
            n_time_bins=n_time_bins_full,
        )

    active = [(0, 0, n_bag, 0)]

    while active:
        splittable, leafify = [], []
        for entry in active:
            node_idx, start, end, depth = entry
            n_node = end - start
            if n_node < min_samples_split or (max_depth >= 0 and depth >= max_depth):
                leafify.append(entry)
            else:
                splittable.append(entry)

        # ``sample_perm_d`` is kept in lock-step with the host ``sample_perm``
        # by the partition kernel + a per-level D2H copy below. The
        # start-of-level leafify, histogram kernel, and mid-level leafify all
        # reuse this single device buffer.

        # Start-of-level leafify (entries that hit min_samples_split / max_depth).
        if leafify:
            level_leaf_batch = []
            for node_idx, start, end, _depth in leafify:
                level_leaf_batch.append((start, end, n_leaves_used))
                out_features[node_idx] = -1
                out_is_leaf[node_idx] = True
                out_leaf_idx_of_node[node_idx] = n_leaves_used
                n_leaves_used += 1
            _gpu_leafify(level_leaf_batch, sample_perm_d)

        if not splittable:
            active = []
            continue

        N_active = len(splittable)
        feat_perm = np.empty((N_active, mtry), dtype=np.int32)
        for i in range(N_active):
            pool = np.arange(p, dtype=np.int32)
            rng.shuffle(pool)
            feat_perm[i] = pool[:mtry]

        node_starts_h = np.empty(N_active, dtype=np.int32)
        node_ends_h = np.empty(N_active, dtype=np.int32)
        for i, (_, start, end, _) in enumerate(splittable):
            node_starts_h[i] = start
            node_ends_h[i] = end

        # Derived per-level seed so consecutive levels don't repeat draws;
        # rng.integers consumes one orchestrator-state advance, preserving
        # outer-RNG determinism (same forest seed → same level_seeds).
        level_seed = int(rng.integers(0, 2**31 - 1))
        cand_mask_h = _build_cand_mask_njit(
            X_binned,
            sample_perm,
            node_starts_h,
            node_ends_h,
            feat_perm,
            n_bins,
            nsplit,
            level_seed,
        )

        node_starts_d = cp.asarray(node_starts_h)
        node_ends_d = cp.asarray(node_ends_h)
        cand_mask_d = cp.asarray(cand_mask_h)
        ehist = cp.zeros((N_active, mtry, n_bins, n_causes, n_time_bins_split), dtype=cp.uint32)
        nat = cp.zeros((N_active, mtry, n_bins, n_time_bins_split), dtype=cp.uint32)

        # histogram_kernel_per_level reads X_binned[s * mtry + feat] where
        # s is a global sample index from sample_perm. For per-node mtry we
        # build a (n_total_samples, mtry) view indexed by sample index so
        # kernel-side X_binned[s, f] returns the correct feature column for
        # whichever node contains s.
        n_total = X_binned.shape[0]
        Xb_view_h = np.empty((n_total, mtry), dtype=np.uint8)
        _build_xb_view_njit(
            X_binned,
            sample_perm,
            node_starts_h,
            node_ends_h,
            feat_perm,
            Xb_view_h,
        )
        Xb_view_d = cp.asarray(Xb_view_h)

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

        out_feat_d = cp.full((N_active,), -1, dtype=cp.int32)
        out_bin_d = cp.full((N_active,), -1, dtype=cp.int32)
        out_stat_d = cp.full((N_active,), -np.inf, dtype=cp.float64)
        best_split_kernel_per_node(
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
        # cp.asnumpy implicitly syncs on the default stream — no explicit deviceSync needed.
        out_feat_h = cp.asnumpy(out_feat_d)
        out_bin_h = cp.asnumpy(out_bin_d)

        # Pre-pass: classify each splittable into success-vs-fail + apply the
        # n_nodes_used overflow gate up front so we know exactly which nodes
        # will partition on the GPU and which fall through to leafify.
        slots_left = (N_max_nodes - n_nodes_used) // 2
        mid_leaf_batch = []
        # Each task is a tuple (node_idx, start, end, depth, actual_feat, bin_idx)
        partition_tasks = []
        for i, (node_idx, start, end, depth) in enumerate(splittable):
            f_sel = int(out_feat_h[i])
            bin_idx = int(out_bin_h[i])
            if f_sel < 0 or len(partition_tasks) >= slots_left:
                mid_leaf_batch.append((start, end, n_leaves_used))
                out_features[node_idx] = -1
                out_is_leaf[node_idx] = True
                out_leaf_idx_of_node[node_idx] = n_leaves_used
                n_leaves_used += 1
                continue
            actual_feat = int(feat_perm[i, f_sel])
            partition_tasks.append((node_idx, start, end, depth, actual_feat, bin_idx))

        # Mid-level leafify (failed-split + overflow) runs against the
        # pre-partition sample_perm_d; these nodes are not touched by the
        # partition pass that follows.
        _gpu_leafify(mid_leaf_batch, sample_perm_d)

        new_active = []
        if partition_tasks:
            T = len(partition_tasks)
            p_starts_h = np.empty(T, dtype=np.int32)
            p_ends_h = np.empty(T, dtype=np.int32)
            p_feats_h = np.empty(T, dtype=np.int32)
            p_bins_h = np.empty(T, dtype=np.int32)
            for k, (_node_idx, s, e, _d, af, bi) in enumerate(partition_tasks):
                p_starts_h[k] = s
                p_ends_h[k] = e
                p_feats_h[k] = af
                p_bins_h[k] = bi
            p_starts_d = cp.asarray(p_starts_h)
            p_ends_d = cp.asarray(p_ends_h)
            p_feats_d = cp.asarray(p_feats_h)
            p_bins_d = cp.asarray(p_bins_h)
            p_mids_d = cp.empty(T, dtype=cp.int32)

            # Pre-fill scratch with the live sample_perm so that ranges
            # outside any partition task carry through unchanged after the
            # kernel patches the partitioned slices.
            cp.copyto(sample_perm_scratch_d, sample_perm_d)
            partition_kernel_per_level(
                X_binned_d,
                sample_perm_d,
                sample_perm_scratch_d,
                p_starts_d,
                p_ends_d,
                p_feats_d,
                p_bins_d,
                p_mids_d,
                p=p,
            )
            # Swap roles: scratch now holds the post-partition order.
            sample_perm_d, sample_perm_scratch_d = sample_perm_scratch_d, sample_perm_d
            mids_h = cp.asnumpy(p_mids_d)

            # Sync host sample_perm with device for the next level's CPU-side
            # build_cand_mask + Xb_view_h gather.
            sample_perm[:] = cp.asnumpy(sample_perm_d)

            for k, (node_idx, start, end, depth, actual_feat, bin_idx) in enumerate(
                partition_tasks
            ):
                mid = int(mids_h[k])
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

        active = new_active

    # Vectorized Aalen-Johansen on device, then a single bulk D2H of the
    # populated leaf rows + the leaf_table.
    leaf_table_d = aalen_johansen_from_counts_batched_gpu(
        leaf_event_counts_d[:n_leaves_used],
        leaf_at_risk_d[:n_leaves_used],
        n_causes,
    )
    leaf_table = cp.asnumpy(leaf_table_d)
    out_leaf_event_counts = cp.asnumpy(leaf_event_counts_d[:n_leaves_used])
    out_leaf_at_risk = cp.asnumpy(leaf_at_risk_d[:n_leaves_used])

    return FlatTree.from_arrays(
        features=out_features[:n_nodes_used],
        split_values=out_split_values[:n_nodes_used],
        left_children=out_left[:n_nodes_used],
        right_children=out_right[:n_nodes_used],
        is_leaf_flags=out_is_leaf[:n_nodes_used],
        leaf_table=leaf_table,
        leaf_idx_of_node=out_leaf_idx_of_node[:n_nodes_used],
        leaf_event_counts=out_leaf_event_counts,
        leaf_at_risk=out_leaf_at_risk,
    )
