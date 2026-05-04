"""Parity and edge-case tests for find_best_split_hist_batched (ε Task 2.2)."""

from __future__ import annotations

import numpy as np
import pytest

from comprisk._hist_splits import (
    _best_split_in_feature,
    _best_split_in_feature_lr,
    _node_histograms,
    find_best_split_hist_batched,
)


def _make_inputs(seed: int, n: int = 2000, mtry: int = 6):
    rng = np.random.default_rng(seed)
    n_bins = 32
    n_causes = 2
    n_time_bins = 50
    X_binned = rng.integers(0, n_bins, size=(n, mtry), dtype=np.uint8)
    t_idx = rng.integers(0, n_time_bins, size=n, dtype=np.int32)
    event = rng.integers(0, 3, size=n, dtype=np.int64)
    return X_binned, t_idx, event, n_bins, n_causes, n_time_bins


def _legacy_best(
    X_binned,
    t_idx,
    event,
    n_bins,
    n_causes,
    n_time_bins,
    min_samples_leaf,
    splitrule: str,
    cause: int,
):
    """Reference: legacy per-feature scan — mirrors find_best_split_hist for the exhaustive path."""
    mtry = X_binned.shape[1]
    selected = np.arange(mtry, dtype=np.int64)
    bin_sub = np.ascontiguousarray(X_binned[:, selected])
    event_hist, at_risk = _node_histograms(bin_sub, t_idx, event, n_bins, n_causes, n_time_bins)
    mask = np.ones(n_bins - 1, dtype=np.bool_)
    best_f, best_b, best_s = -1, 0, 0.0
    for f in range(mtry):
        if splitrule == "logrankCR":
            bi, st = _best_split_in_feature(
                event_hist[f], at_risk[f], X_binned.shape[0], min_samples_leaf, mask
            )
        else:
            bi, st = _best_split_in_feature_lr(
                event_hist[f], at_risk[f], X_binned.shape[0], min_samples_leaf, cause, mask
            )
        if st > best_s:
            best_s, best_b, best_f = st, int(bi), f
    return best_f, best_b, best_s


@pytest.mark.parametrize("seed", list(range(20)))
@pytest.mark.parametrize(
    "splitrule_code,splitrule,cause", [(0, "logrankCR", 1), (1, "logrank", 1), (1, "logrank", 2)]
)
def test_batched_matches_legacy_exhaustive(
    seed: int, splitrule_code: int, splitrule: str, cause: int
) -> None:
    X_binned, t_idx, event, n_bins, n_causes, n_time_bins = _make_inputs(seed)
    mtry = X_binned.shape[1]
    mask = np.ones((mtry, n_bins - 1), dtype=np.bool_)

    f_leg, b_leg, s_leg = _legacy_best(
        X_binned,
        t_idx,
        event,
        n_bins,
        n_causes,
        n_time_bins,
        min_samples_leaf=5,
        splitrule=splitrule,
        cause=cause,
    )
    f_new, b_new, s_new = find_best_split_hist_batched(
        X_binned,
        t_idx,
        event,
        n_bins,
        n_causes,
        n_time_bins,
        min_samples_leaf=5,
        splitrule_code=splitrule_code,
        cause=cause,
        candidate_mask=mask,
    )
    assert f_new == f_leg, f"feature mismatch seed={seed}: {f_new} vs {f_leg}"
    assert b_new == b_leg, f"bin mismatch seed={seed}: {b_new} vs {b_leg}"
    # stat may differ by a tiny amount under fastmath; cap at 1e-8 relative.
    if s_leg == 0.0:
        assert s_new == 0.0
    else:
        assert abs(s_new - s_leg) / abs(s_leg) < 1e-8, f"stat drift seed={seed}: {s_new} vs {s_leg}"


def test_batched_degenerate_variance_returns_minus_one() -> None:
    """All events at a single time → no valid split → best_feature_selected = -1."""
    rng = np.random.default_rng(42)
    n, mtry, n_bins, n_causes, n_time_bins = 200, 4, 16, 2, 20
    X_binned = rng.integers(0, n_bins, size=(n, mtry), dtype=np.uint8)
    event = np.full(n, 1, dtype=np.int64)
    t_idx = np.full(n, 5, dtype=np.int32)  # all at one time → variance is 0
    mask = np.ones((mtry, n_bins - 1), dtype=np.bool_)

    f, _b, s = find_best_split_hist_batched(
        X_binned,
        t_idx,
        event,
        n_bins,
        n_causes,
        n_time_bins,
        min_samples_leaf=5,
        splitrule_code=0,
        cause=1,
        candidate_mask=mask,
    )
    assert f == -1, f"expected -1 (no valid split); got {f}"
    assert np.isfinite(s), f"stat must be finite even when no split found; got {s}"


def test_batched_all_false_mask_returns_minus_one() -> None:
    X_binned, t_idx, event, n_bins, n_causes, n_time_bins = _make_inputs(7)
    mtry = X_binned.shape[1]
    mask = np.zeros((mtry, n_bins - 1), dtype=np.bool_)
    f, _, _ = find_best_split_hist_batched(
        X_binned,
        t_idx,
        event,
        n_bins,
        n_causes,
        n_time_bins,
        min_samples_leaf=5,
        splitrule_code=0,
        cause=1,
        candidate_mask=mask,
    )
    assert f == -1


def test_find_best_split_hist_use_batched_matches_legacy() -> None:
    """Full-path dispatcher: use_batched=True should agree with use_batched=False on the same inputs."""
    from comprisk._hist_splits import find_best_split_hist

    rng = np.random.default_rng(3)
    n, p = 1500, 8
    n_bins = 32
    n_causes = 2
    n_time_bins = 40
    X_binned = rng.integers(0, n_bins, size=(n, p), dtype=np.uint8)
    t_idx = rng.integers(0, n_time_bins, size=n, dtype=np.int32)
    event = rng.integers(0, 3, size=n, dtype=np.int64)
    selected = np.arange(p, dtype=np.int64)

    f_leg, b_leg, s_leg = find_best_split_hist(
        X_binned,
        t_idx,
        event,
        selected,
        n_bins,
        n_causes,
        n_time_bins,
        5,
        splitrule="logrankCR",
        cause=1,
        nsplit=0,
        rng=None,
        use_batched=False,
    )
    f_new, b_new, s_new = find_best_split_hist(
        X_binned,
        t_idx,
        event,
        selected,
        n_bins,
        n_causes,
        n_time_bins,
        5,
        splitrule="logrankCR",
        cause=1,
        nsplit=0,
        rng=None,
        use_batched=True,
    )
    assert f_new == f_leg
    assert b_new == b_leg
    assert abs(s_new - s_leg) / max(abs(s_leg), 1e-12) < 1e-8
