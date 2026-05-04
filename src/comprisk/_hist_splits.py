"""Histogram split search kernels and wrapper for default-mode trees.

The three jitted kernels:

- ``_node_histograms`` accumulates per-feature per-bin per-cause histograms
  of event counts and a stratified at-risk array over the node's samples.
- ``_best_split_in_feature`` scans the candidate bin splits for one feature
  and returns the best composite log-rank and its bin index.
- ``_best_split_in_feature_lr`` is the sibling for cause-specific log-rank
  (standard at-risk).

Composite log-rank matches the reference-mode formula in ``_splits.py``:
pooled standardized across causes (``(Σ_k num_k)^2 / Σ_k var_k``) with
Lau-inclusive at-risk (subjects who had a competing-cause event stay in
the cause-k risk set).  See
``docs/superpowers/specs/2026-04-18-p2.5-logrankcr-derivation.md``.

``find_best_split_hist`` is the Python entry point used by the tree builder.
"""

from __future__ import annotations

import os

import numpy as np
from numba import njit


@njit(cache=True, nogil=True)
def _node_histograms(
    bin_indices_node: np.ndarray,  # (n_node, k) uint8
    time_indices_node: np.ndarray,  # (n_node,) int32
    event_node: np.ndarray,  # (n_node,) int64
    n_bins: int,
    n_causes: int,
    n_time_bins: int,
):
    n_node, k = bin_indices_node.shape
    event_hist = np.zeros((k, n_bins, n_causes, n_time_bins), dtype=np.uint32)
    # n_at[f, b, t] = count of samples in feature f / bin b at exactly time t
    n_at = np.zeros((k, n_bins, n_time_bins), dtype=np.uint32)

    for i in range(n_node):
        t = time_indices_node[i]
        e = event_node[i]
        for f in range(k):
            b = bin_indices_node[i, f]
            n_at[f, b, t] += 1
            if e > 0:
                event_hist[f, b, e - 1, t] += 1

    # at_risk[f, b, t] = sum over s >= t of n_at[f, b, s] (reverse cumsum)
    at_risk = np.zeros((k, n_bins, n_time_bins), dtype=np.uint32)
    for f in range(k):
        for b in range(n_bins):
            running = np.uint32(0)
            for t in range(n_time_bins - 1, -1, -1):
                running += n_at[f, b, t]
                at_risk[f, b, t] = running
    return event_hist, at_risk


@njit(cache=True, nogil=True)
def _best_split_in_feature(
    event_hist: np.ndarray,  # (n_bins, n_causes, n_time_bins)
    at_risk_hist: np.ndarray,  # (n_bins, n_time_bins)
    n_node: int,
    min_samples_leaf: int,
    candidate_mask: np.ndarray,  # (n_bins - 1,) bool
):
    """Composite log-rank over candidate bin splits for one feature.

    For each candidate bin cut b in {0, 1, ..., n_bins - 2}, computes the
    pooled-standardized composite log-rank statistic ``(Σ_k num_k)^2 / Σ_k var_k``
    with Lau-inclusive at-risk per cause, and tracks the maximum.  Matches
    ``composite_log_rank_statistic`` in ``_splits.py``.

    ``candidate_mask`` is a bool array of length ``n_bins - 1``; only
    boundaries with ``candidate_mask[b] == True`` are evaluated. Pass an
    all-True mask to reproduce the exhaustive scan.

    Returns (best_bin, best_stat). best_bin = -1 if no valid split.
    """
    n_bins, n_causes, n_time_bins = event_hist.shape

    at_risk_total = at_risk_hist.sum(axis=0).astype(np.float64)
    d_k_total = event_hist.sum(axis=0).astype(np.float64)
    d_any_total = d_k_total.sum(axis=0)

    # Lau-inclusive parent at-risk per cause (computed once per feature).
    # nPinc_q[t] = at_risk_total[t] + cumsum_{s<t}(d_any_total[s] - d_k_total[q, s])
    at_risk_total_inc = np.empty((n_causes, n_time_bins), dtype=np.float64)
    for q in range(n_causes):
        cumsum = 0.0
        for t in range(n_time_bins):
            at_risk_total_inc[q, t] = at_risk_total[t] + cumsum
            cumsum += d_any_total[t] - d_k_total[q, t]

    # Running prefix sums for "left" (bin <= b)
    at_risk_left = np.zeros(n_time_bins, dtype=np.float64)
    d_k_left = np.zeros((n_causes, n_time_bins), dtype=np.float64)
    d_any_left = np.zeros(n_time_bins, dtype=np.float64)
    at_risk_left_inc = np.empty((n_causes, n_time_bins), dtype=np.float64)

    samples_per_bin = at_risk_hist[:, 0]

    best_bin = -1
    best_stat = 0.0
    n_left_running = 0

    for b in range(n_bins - 1):
        # Running left-side prefix sums must be updated whether or not
        # this boundary is a candidate, since later boundaries depend on
        # the same running totals.  d_any_left is also kept in sync here
        # (derived from the cumulative d_k_left) for the same reason.
        n_left_running += samples_per_bin[b]
        at_risk_left += at_risk_hist[b]
        d_k_left += event_hist[b]
        for t in range(n_time_bins):
            d_any_left[t] = 0.0
            for k in range(n_causes):
                d_any_left[t] += d_k_left[k, t]

        if not candidate_mask[b]:
            continue

        n_right_running = n_node - n_left_running
        if n_left_running < min_samples_leaf or n_right_running < min_samples_leaf:
            continue

        # Lau-inclusive left at-risk per cause
        for q in range(n_causes):
            cumsum = 0.0
            for t in range(n_time_bins):
                at_risk_left_inc[q, t] = at_risk_left[t] + cumsum
                cumsum += d_any_left[t] - d_k_left[q, t]

        # Pool signed numerators and variances across causes.  Unlike the
        # reference-mode _TimeBinning (which only holds node-local times),
        # the histogram kernel iterates over the global time grid, so some
        # bins have at_risk_total[t] = 0 at this node.  Skip those entirely:
        # d_k_total is also 0 there, so both numerator and variance would
        # contribute 0 — we just need to avoid the 0/0 division.
        # Numerator has no nP >= 2 guard; variance is guarded by the
        # standard parent at-risk >= 2 condition.
        num_sum = 0.0
        var_sum = 0.0
        for k in range(n_causes):
            for t in range(n_time_bins):
                if at_risk_total[t] == 0.0:
                    continue
                arinc_t = at_risk_total_inc[k, t]
                arlinc_t = at_risk_left_inc[k, t]
                d_t = d_k_total[k, t]
                dl_t = d_k_left[k, t]
                num_sum += dl_t - d_t * arlinc_t / arinc_t
                if at_risk_total[t] >= 2.0:
                    var_sum += (
                        d_t
                        * arlinc_t
                        * (arinc_t - arlinc_t)
                        * (arinc_t - d_t)
                        / (arinc_t * arinc_t * (arinc_t - 1.0))
                    )

        if var_sum < 1e-12:
            continue
        stat_total = num_sum * num_sum / var_sum

        if stat_total > best_stat:
            best_stat = stat_total
            best_bin = b

    return best_bin, best_stat


@njit(cache=True, nogil=True)
def _best_split_in_feature_lr(
    event_hist: np.ndarray,  # (n_bins, n_causes, n_time_bins)
    at_risk_hist: np.ndarray,  # (n_bins, n_time_bins)
    n_node: int,
    min_samples_leaf: int,
    cause: int,
    candidate_mask: np.ndarray,  # (n_bins - 1,) bool
):
    """Cause-specific log-rank (standard at-risk) over candidate bin splits.

    Matches ``cause_specific_log_rank_statistic(..., cause=cause)`` in
    ``_splits.py``.  Uses the standard parent at-risk (competing events
    remove the subject from the risk set), unlike
    ``_best_split_in_feature`` which uses Lau-inclusive.

    ``candidate_mask`` is a bool array of length ``n_bins - 1``; only
    boundaries with ``candidate_mask[b] == True`` are evaluated.

    Returns (best_bin, best_stat). best_bin = -1 if no valid split.
    """
    n_bins, n_causes, n_time_bins = event_hist.shape
    k_idx = cause - 1

    at_risk_total = at_risk_hist.sum(axis=0).astype(np.float64)
    d_k_total = event_hist.sum(axis=0).astype(np.float64)

    at_risk_left = np.zeros(n_time_bins, dtype=np.float64)
    d_k_left = np.zeros((n_causes, n_time_bins), dtype=np.float64)

    samples_per_bin = at_risk_hist[:, 0]

    best_bin = -1
    best_stat = 0.0
    n_left_running = 0

    for b in range(n_bins - 1):
        # Prefix sums updated unconditionally before mask check — same
        # rationale as _best_split_in_feature: later bins need the running total.
        n_left_running += samples_per_bin[b]
        at_risk_left += at_risk_hist[b]
        d_k_left += event_hist[b]

        if not candidate_mask[b]:
            continue

        n_right_running = n_node - n_left_running
        if n_left_running < min_samples_leaf or n_right_running < min_samples_leaf:
            continue

        num_sum = 0.0
        var_sum = 0.0
        for t in range(n_time_bins):
            ar_t = at_risk_total[t]
            if ar_t == 0.0:
                continue
            d_t = d_k_total[k_idx, t]
            dl_t = d_k_left[k_idx, t]
            arl_t = at_risk_left[t]
            num_sum += dl_t - d_t * arl_t / ar_t
            if ar_t >= 2.0:
                var_sum += (
                    d_t * arl_t * (ar_t - arl_t) * (ar_t - d_t) / (ar_t * ar_t * (ar_t - 1.0))
                )

        if var_sum < 1e-12:
            continue
        stat_total = num_sum * num_sum / var_sum
        if stat_total > best_stat:
            best_stat = stat_total
            best_bin = b

    return best_bin, best_stat


@njit(cache=True, nogil=True, fastmath=True)
def find_best_split_hist_batched(
    X_binned: np.ndarray,  # (n_node, mtry) uint8
    t_idx: np.ndarray,  # (n_node,) int32
    event: np.ndarray,  # (n_node,) int64
    n_bins: int,
    n_causes: int,
    n_time_bins: int,
    min_samples_leaf: int,
    splitrule_code: int,  # 0=logrankCR, 1=logrank
    cause: int,  # 1-indexed (used only for logrank)
    candidate_mask: np.ndarray,  # (mtry, n_bins - 1) bool
):
    """Fused across-feature histogram + split-scan kernel.

    splitrule_code: 0 = logrankCR (pooled composite, Lau-inclusive)
                    1 = logrank   (cause-specific, standard at-risk)

    Returns (best_feature_selected, best_bin, best_stat).
    best_feature_selected = -1 if no valid split.
    """
    n_node, mtry = X_binned.shape

    # Histograms across all mtry features in a single sample pass.
    event_hist = np.zeros((mtry, n_bins, n_causes, n_time_bins), dtype=np.uint32)
    n_at = np.zeros((mtry, n_bins, n_time_bins), dtype=np.uint32)
    for i in range(n_node):
        t = t_idx[i]
        e = event[i]
        for f in range(mtry):
            b = X_binned[i, f]
            n_at[f, b, t] += 1
            if e > 0:
                event_hist[f, b, e - 1, t] += 1

    at_risk = np.zeros((mtry, n_bins, n_time_bins), dtype=np.uint32)
    for f in range(mtry):
        for b in range(n_bins):
            running = np.uint32(0)
            for t in range(n_time_bins - 1, -1, -1):
                running += n_at[f, b, t]
                at_risk[f, b, t] = running

    # Running left-side accumulators, reallocated per feature.
    at_risk_left = np.zeros(n_time_bins, dtype=np.float64)
    d_k_left = np.zeros((n_causes, n_time_bins), dtype=np.float64)
    d_any_left = np.zeros(n_time_bins, dtype=np.float64)
    at_risk_left_inc = np.empty((n_causes, n_time_bins), dtype=np.float64)
    at_risk_total_inc = np.empty((n_causes, n_time_bins), dtype=np.float64)

    best_f = -1
    best_b = 0
    best_s = 0.0
    cause_idx = cause - 1

    for f in range(mtry):
        at_risk_total = at_risk[f].sum(axis=0).astype(np.float64)
        d_k_total = event_hist[f].sum(axis=0).astype(np.float64)
        d_any_total = d_k_total.sum(axis=0)

        if splitrule_code == 0:
            for q in range(n_causes):
                cumsum = 0.0
                for t in range(n_time_bins):
                    at_risk_total_inc[q, t] = at_risk_total[t] + cumsum
                    cumsum += d_any_total[t] - d_k_total[q, t]

        at_risk_left.fill(0.0)
        d_k_left.fill(0.0)
        # d_any_left is overwritten every inner-loop iteration (per-bin, per-t)
        # before use, so a per-feature fill would be dead.
        samples_per_bin = at_risk[f, :, 0]
        n_left_running = 0

        for b in range(n_bins - 1):
            n_left_running += samples_per_bin[b]
            at_risk_left += at_risk[f, b]
            d_k_left += event_hist[f, b]
            for t in range(n_time_bins):
                d_any_left[t] = 0.0
                for k in range(n_causes):
                    d_any_left[t] += d_k_left[k, t]

            if not candidate_mask[f, b]:
                continue

            n_right_running = n_node - n_left_running
            if n_left_running < min_samples_leaf or n_right_running < min_samples_leaf:
                continue

            if splitrule_code == 0:
                for q in range(n_causes):
                    cumsum = 0.0
                    for t in range(n_time_bins):
                        at_risk_left_inc[q, t] = at_risk_left[t] + cumsum
                        cumsum += d_any_left[t] - d_k_left[q, t]

                num_sum = 0.0
                var_sum = 0.0
                for k in range(n_causes):
                    for t in range(n_time_bins):
                        if at_risk_total[t] == 0.0:
                            continue
                        arinc_t = at_risk_total_inc[k, t]
                        arlinc_t = at_risk_left_inc[k, t]
                        d_t = d_k_total[k, t]
                        dl_t = d_k_left[k, t]
                        num_sum += dl_t - d_t * arlinc_t / arinc_t
                        if at_risk_total[t] >= 2.0:
                            var_sum += (
                                d_t
                                * arlinc_t
                                * (arinc_t - arlinc_t)
                                * (arinc_t - d_t)
                                / (arinc_t * arinc_t * (arinc_t - 1.0))
                            )
                if var_sum < 1e-12:
                    continue
                stat_total = num_sum * num_sum / var_sum
            else:
                num_sum = 0.0
                var_sum = 0.0
                for t in range(n_time_bins):
                    if at_risk_total[t] < 2.0:
                        continue
                    d_t = d_k_total[cause_idx, t]
                    dl_t = d_k_left[cause_idx, t]
                    ar_t = at_risk_total[t]
                    arl_t = at_risk_left[t]
                    num_sum += dl_t - d_t * arl_t / ar_t
                    var_sum += (
                        d_t * arl_t * (ar_t - arl_t) * (ar_t - d_t) / (ar_t * ar_t * (ar_t - 1.0))
                    )
                if var_sum < 1e-12:
                    continue
                stat_total = num_sum * num_sum / var_sum

            if stat_total != stat_total:  # NaN guard (no math.isfinite under numba nogil)
                continue

            if stat_total > best_s:
                best_s = stat_total
                best_b = b
                best_f = f

    return best_f, best_b, best_s


def _observed_bins_sorted_ascending(column: np.ndarray, n_bins: int) -> np.ndarray:
    """Return the sorted distinct bin values observed in a uint8 column.

    Bincount over ``n_bins`` slots is ``O(n)`` vs ``O(n log n)`` sort.
    Output is always sorted ascending because ``np.flatnonzero`` walks the
    counts array in index order. Return dtype is ``int64`` (``np.flatnonzero``'s
    native output); the result is per-call scratch and never persisted, so the
    uint8 bin-index convention does not apply here. Keeping int64 avoids a
    measured ~15% slowdown on the downstream ``mask[chosen] = True`` fancy
    index (numpy fancy-index fast path is native integer only).

    Parameters
    ----------
    column : np.ndarray
        1-D array of dtype ``uint8``. All values must lie in ``[0, n_bins)``;
        guaranteed by the caller because ``_binning.py`` caps bin indices at
        ``n_bins - 1``.
    n_bins : int
        Bincount length. Matches the forest's bin-count hyperparameter.
    """
    return np.flatnonzero(np.bincount(column, minlength=n_bins))


def _nsplit_draw_mask(
    valid_candidates: np.ndarray,
    nsplit: int,
    n_bins: int,
    rng,
    skip_rng_when_all_used: bool,
) -> np.ndarray:
    """Build the nsplit candidate mask for one feature.

    When ``skip_rng_when_all_used`` is True AND the feature has at most
    ``nsplit + 1`` unique values, rfSRC takes a deterministic branch
    that evaluates every candidate without consuming any stream-B draws.
    Matching this is required for bit-identity under
    ``rng_mode='rfsrc_aligned'`` -- skipping the RNG here keeps stream
    state aligned for the next mtry draw. Default ``False`` preserves
    the legacy numpy-mode behavior (always call ``rng.choice``).
    """
    mask = np.zeros(n_bins - 1, dtype=np.bool_)
    if skip_rng_when_all_used and nsplit >= len(valid_candidates):
        mask[valid_candidates] = True
        return mask
    k = min(nsplit, len(valid_candidates))
    trace_path = os.environ.get("CRFOREST_TRACE")
    if trace_path and hasattr(rng, "stream"):
        # Replicate rfSRC's SWOR on a 1-based position vector (sworVector)
        # for log-format parity: rfSRC's "b" in nsplit_pick logs the value
        # at the drawn slot in sworVector (the position of the picked
        # threshold within the node's sorted uniques, 1-based).
        with open(trace_path, "a") as tfp:
            tfp.write(f"nsplit_start a={len(valid_candidates) + 1} b={k}\n")
            swor = list(range(1, len(valid_candidates) + 1))  # 1-based positions
            swor_size = len(swor)
            picked_positions: list[int] = []
            for _ in range(k):
                u = rng.stream.next()
                tfp.write(f"ran1B val={u:.10f}\n")
                slot_1 = int(np.ceil(u * swor_size))
                if slot_1 < 1:
                    slot_1 = 1
                elif slot_1 > swor_size:
                    slot_1 = swor_size
                pos = swor[slot_1 - 1]
                picked_positions.append(pos)
                tfp.write(f"nsplit_pick a={slot_1} b={pos}\n")
                swor[slot_1 - 1] = swor[swor_size - 1]
                swor_size -= 1
            # valid_candidates is sorted ascending; position i (1-based) -> valid_candidates[i-1].
            chosen = np.asarray(
                [valid_candidates[p - 1] for p in picked_positions],
                dtype=np.int64,
            )
    else:
        chosen = rng.choice(valid_candidates, size=k, replace=False)
    mask[chosen] = True
    return mask


def find_best_split_hist(
    X_binned_node: np.ndarray,
    time_indices_node: np.ndarray,
    event_node: np.ndarray,
    selected_features: np.ndarray,
    n_bins: int,
    n_causes: int,
    n_time_bins: int,
    min_samples_leaf: int,
    *,
    splitrule: str = "logrankCR",
    cause: int = 1,
    nsplit: int = 0,
    rng: np.random.RandomState | None = None,
    use_batched: bool = False,
    skip_nsplit_rng_when_deterministic: bool = False,
) -> tuple[int, int, float]:
    """Python entry for histogram split search.

    ``splitrule="logrankCR"`` uses the pooled composite (Lau-inclusive)
    kernel; ``splitrule="logrank"`` uses the cause-specific
    (standard at-risk) kernel with the given ``cause``.

    ``nsplit > 0`` activates random split-point subsampling: for each
    candidate feature, ``nsplit`` distinct observed bin indices are drawn
    without replacement from the set of unique bins present at the node,
    excluding the maximum bin (splitting at max leaves an empty right
    child). Matches rfSRC's sampling-without-replacement semantics for
    candidate split values. Requires ``rng``. ``nsplit=0`` (default)
    evaluates every boundary.

    ``use_batched=True`` routes to ``find_best_split_hist_batched`` — the
    fused across-feature kernel that pairs with the ``split_ntime``
    coarsening path. Default ``False`` preserves the per-feature path
    for bit-identity with the uncoarsened behaviour.
    """
    if splitrule not in ("logrankCR", "logrank"):
        raise ValueError(f"splitrule must be 'logrankCR' or 'logrank'; got {splitrule!r}")
    if selected_features.ndim != 1:
        raise ValueError("selected_features must be 1-D")
    if X_binned_node.dtype != np.uint8:
        raise ValueError(f"X_binned_node must be uint8; got {X_binned_node.dtype}")
    if time_indices_node.dtype != np.int32:
        raise ValueError(f"time_indices_node must be int32; got {time_indices_node.dtype}")
    if event_node.dtype != np.int64:
        raise ValueError(f"event_node must be int64; got {event_node.dtype}")
    if nsplit < 0:
        raise ValueError(f"nsplit must be >= 0; got {nsplit}")
    if nsplit > 0 and rng is None:
        raise ValueError("nsplit > 0 requires an rng")

    bin_sub = np.ascontiguousarray(X_binned_node[:, selected_features])

    if use_batched:
        mtry = bin_sub.shape[1]
        if nsplit == 0:
            cand_mask = np.ones((mtry, n_bins - 1), dtype=np.bool_)
        else:
            cand_mask = np.zeros((mtry, n_bins - 1), dtype=np.bool_)
            for f in range(mtry):
                observed_bins = _observed_bins_sorted_ascending(bin_sub[:, f], n_bins)
                if len(observed_bins) < 2:
                    continue
                valid = observed_bins[:-1]
                k = min(nsplit, len(valid))
                chosen = rng.choice(valid, size=k, replace=False)
                cand_mask[f, chosen] = True

        splitrule_code = 0 if splitrule == "logrankCR" else 1
        f_sel, bin_idx, stat = find_best_split_hist_batched(
            bin_sub,
            time_indices_node,
            event_node,
            n_bins,
            n_causes,
            n_time_bins,
            min_samples_leaf,
            splitrule_code,
            cause,
            cand_mask,
        )
        if f_sel == -1:
            return -1, 0, 0.0
        return int(selected_features[f_sel]), int(bin_idx), float(stat)

    # --- legacy path below (unchanged) ---
    event_hist, at_risk_hist = _node_histograms(
        bin_sub, time_indices_node, event_node, n_bins, n_causes, n_time_bins
    )
    n_node = X_binned_node.shape[0]

    # Shared all-True mask for the nsplit=0 path (allocated once, reused across
    # features). In the nsplit>0 path each feature builds its own sparse mask.
    all_true_mask = np.ones(n_bins - 1, dtype=np.bool_)

    best_feature_selected = -1
    best_bin = 0
    best_stat = 0.0
    for f_sel in range(len(selected_features)):
        if nsplit == 0:
            mask = all_true_mask
        else:
            # Sample `nsplit` distinct observed unique values at the node
            # without replacement, excluding the maximum. Each sampled value
            # is interpreted as a bin boundary (split "bin <= b" goes left).
            # Matches rfSRC's split-candidate sampling semantics.
            observed_bins = _observed_bins_sorted_ascending(bin_sub[:, f_sel], n_bins)
            if len(observed_bins) < 2:
                continue  # no valid split
            valid_candidates = observed_bins[:-1]  # exclude max bin
            mask = _nsplit_draw_mask(
                valid_candidates,
                nsplit,
                n_bins,
                rng,
                skip_rng_when_all_used=skip_nsplit_rng_when_deterministic,
            )

        if splitrule == "logrankCR":
            bin_idx, stat = _best_split_in_feature(
                event_hist[f_sel],
                at_risk_hist[f_sel],
                n_node,
                min_samples_leaf,
                mask,
            )
        else:
            bin_idx, stat = _best_split_in_feature_lr(
                event_hist[f_sel],
                at_risk_hist[f_sel],
                n_node,
                min_samples_leaf,
                cause,
                mask,
            )
        _trace_path = os.environ.get("CRFOREST_TRACE")
        if _trace_path:
            with open(_trace_path, "a") as _tfp:
                _tfp.write(
                    f"feat_stat_CR covariate={int(selected_features[f_sel]) + 1} "
                    f"bin={int(bin_idx)} stat={float(stat):.10f}\n"
                )
        if stat > best_stat:
            best_stat = stat
            best_bin = int(bin_idx)
            best_feature_selected = f_sel

    if best_feature_selected == -1:
        return -1, 0, 0.0
    return int(selected_features[best_feature_selected]), best_bin, best_stat
