"""Tests for cause-specific concordance index."""

import numpy as np
import pytest

from comprisk.metrics import concordance_index_cr


def test_perfect_ordering_returns_one():
    # All cause-1 events; risk is perfectly anti-correlated with time
    event = np.array([1, 1, 1, 1])
    time = np.array([1.0, 2.0, 3.0, 4.0])
    risk = np.array([4.0, 3.0, 2.0, 1.0])  # higher risk = earlier event
    assert concordance_index_cr(event, time, risk, cause=1) == 1.0


def test_perfect_anti_ordering_returns_zero():
    event = np.array([1, 1, 1, 1])
    time = np.array([1.0, 2.0, 3.0, 4.0])
    risk = np.array([1.0, 2.0, 3.0, 4.0])  # higher risk = later event (wrong)
    assert concordance_index_cr(event, time, risk, cause=1) == 0.0


def test_all_tied_returns_half():
    event = np.array([1, 1, 1, 1])
    time = np.array([1.0, 2.0, 3.0, 4.0])
    risk = np.array([0.5, 0.5, 0.5, 0.5])
    assert concordance_index_cr(event, time, risk, cause=1) == 0.5


def test_no_events_of_cause_returns_half():
    event = np.array([2, 2, 0, 0])
    time = np.array([1.0, 2.0, 3.0, 4.0])
    risk = np.array([0.1, 0.2, 0.3, 0.4])
    assert concordance_index_cr(event, time, risk, cause=1) == 0.5


def test_competing_event_before_index_time_is_incomparable():
    # Subject 2 has a cause-2 event at t=1, before subject 0's cause-1 at t=3
    event = np.array([1, 0, 2])
    time = np.array([3.0, 5.0, 1.0])
    risk = np.array([0.5, 0.1, 0.9])  # high risk for subject 2 should not count
    # Only comparable pair is (subject 0 index, subject 1): time 3 < 5 with
    # risk 0.5 > 0.1, so concordant.
    assert concordance_index_cr(event, time, risk, cause=1) == 1.0


def test_random_risk_close_to_half():
    rng = np.random.default_rng(0)
    n = 500
    event = rng.integers(0, 3, n)  # {0, 1, 2}
    time = rng.uniform(0.5, 10.0, n)
    risk = rng.uniform(size=n)
    if not np.any(event == 1):
        event[0] = 1
    c = concordance_index_cr(event, time, risk, cause=1)
    assert 0.4 < c < 0.6


def test_cause_not_in_event_raises():
    event = np.array([1, 0, 1])
    time = np.array([1.0, 2.0, 3.0])
    risk = np.array([0.1, 0.2, 0.3])
    with pytest.raises(ValueError, match="cause=3"):
        concordance_index_cr(event, time, risk, cause=3)


# ---- Uno IPCW weights ----------------------------------------------------


def _naive_km_censor_fit(time, event):
    """Brute-force KM-of-censoring matching rfSRC km_censor_fit
    (utilities_survival.R:419). At each unique time t_k:
        d_one_k   = #{i: time[i] == t_k AND event[i] == 1}  (rfSRC's d_death)
        d_other_k = #{i: time[i] == t_k AND event[i] != 1}  (rfSRC's d_cens)
        n_risk_k  = #{i: time[i] >= t_k}
    Survivor: surv *= (1 - d_other / (n_risk - d_one))  if d_other > 0.
    For CR data with cause >= 2, those events go into d_other (lumped with
    true censoring) — this is rfSRC's convention.
    """
    time = np.asarray(time, dtype=float)
    event = np.asarray(event)
    t_unique = np.sort(np.unique(time))
    G = np.empty(len(t_unique), dtype=float)
    surv = 1.0
    for k, t_k in enumerate(t_unique):
        n_risk = int((time >= t_k).sum())
        d_one = int(((time == t_k) & (event == 1)).sum())
        d_other = int(((time == t_k) & (event != 1)).sum())
        if d_other > 0:
            denom = n_risk - d_one
            surv = 0.0 if denom <= 0 else surv * (1.0 - d_other / denom)
        G[k] = surv
    return t_unique, G


def _naive_ghat_minus(t_km, G_km, query):
    """G(t^-) by direct definition: G evaluated strictly BEFORE t. For
    t <= t_km[0], returns 1.0. For t == t_km[k], returns G_km[k-1] (or 1.0
    if k==0). For t > t_km[-1], returns G_km[-1]."""
    t_km = np.asarray(t_km)
    out = np.empty(len(query), dtype=float)
    for i, t in enumerate(query):
        before = t_km < t
        if not before.any():
            out[i] = 1.0
        else:
            out[i] = G_km[before.sum() - 1]
    return out


def _naive_compute_uno_weights(
    time, event, gmin="auto", ess_frac=0.20, ess_min=20, eps=1e-12, eps_keep=None
):
    if eps_keep is None:
        eps_keep = float(np.finfo(float).eps)
    time = np.asarray(time, dtype=float)
    event = np.asarray(event)
    n = len(time)
    t_km, G_km = _naive_km_censor_fit(time, event)
    G_gate = _naive_ghat_minus(t_km, G_km, time)

    if isinstance(gmin, str):
        if gmin == "none":
            gmin_used = 0.0
        elif gmin == "auto":
            mask_ev = event != 0
            G_ev = G_gate[mask_ev]
            G_ev = G_ev[~np.isnan(G_ev)]
            d = len(G_ev)
            if d <= 1 or G_ev.min() >= 1 - 1e-12:
                gmin_used = 0.0
            else:
                g_sorted = np.sort(G_ev)
                w_desc = 1.0 / np.maximum(g_sorted, eps) ** 2
                ess_target = max(ess_min, int(np.ceil(ess_frac * d)))
                ess_target = min(ess_target, d)
                best_k = None
                for k in range(0, d - ess_target + 1):
                    sum_w = w_desc[k:].sum()
                    sum_w2 = (w_desc[k:] ** 2).sum()
                    ess_k = (sum_w * sum_w) / sum_w2 if sum_w2 > 0 else float("nan")
                    if np.isfinite(ess_k) and ess_k >= ess_target:
                        best_k = k
                        break
                if best_k is None:
                    best_k = d - ess_target
                gmin_used = float(g_sorted[best_k])
        else:
            raise ValueError(f"unknown gmin='{gmin}'")
    else:
        gmin_used = float(gmin)

    w = np.zeros(n, dtype=float)
    ok = ~np.isnan(G_gate)
    keep = ok & (G_gate >= gmin_used)
    drop = ok & ~keep
    if keep.any():
        Gsafe = np.maximum(G_gate[keep], eps)
        w[keep] = 1.0 / (Gsafe * Gsafe)
    if drop.any():
        w[drop] = eps_keep
    return w


def test_compute_uno_weights_all_cause_one_yields_unit_weights():
    from comprisk.metrics import compute_uno_weights

    # rfSRC's km_censor_fit treats cause-1 specially (s==1 → "death", removed
    # first), so all-cause-1 means d_other == 0 everywhere → G stays at 1.
    # Pure cause-2 or mixed CR would NOT yield unit weights under this
    # convention (cause-2 is lumped with censoring).
    time = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    event = np.array([1, 1, 1, 1, 1])
    w = compute_uno_weights(time, event, gmin="none")
    np.testing.assert_allclose(w, np.ones(5), atol=1e-12)


def test_compute_uno_weights_naive_ref_mixed():
    from comprisk.metrics import compute_uno_weights

    rng = np.random.default_rng(0)
    n = 200
    time = rng.uniform(0.1, 10.0, n)
    event = rng.integers(0, 3, n)
    if not (event != 0).any():
        event[0] = 1
    w_prod = compute_uno_weights(time, event)
    w_ref = _naive_compute_uno_weights(time, event)
    np.testing.assert_allclose(w_prod, w_ref, atol=1e-12, rtol=0)


def test_compute_uno_weights_naive_ref_high_censoring():
    from comprisk.metrics import compute_uno_weights

    rng = np.random.default_rng(1)
    n = 100
    time = rng.uniform(0.1, 10.0, n)
    event = (rng.uniform(size=n) > 0.7).astype(int)
    if not (event != 0).any():
        event[0] = 1
    w_prod = compute_uno_weights(time, event)
    w_ref = _naive_compute_uno_weights(time, event)
    np.testing.assert_allclose(w_prod, w_ref, atol=1e-12, rtol=0)


def test_compute_uno_weights_naive_ref_competing_risks():
    from comprisk.metrics import compute_uno_weights

    rng = np.random.default_rng(2)
    n = 150
    time = rng.uniform(0.1, 10.0, n)
    cause = rng.integers(1, 3, n)
    cens = rng.uniform(size=n) < 0.4
    event = np.where(cens, 0, cause)
    if not (event != 0).any():
        event[0] = 1
    w_prod = compute_uno_weights(time, event)
    w_ref = _naive_compute_uno_weights(time, event)
    np.testing.assert_allclose(w_prod, w_ref, atol=1e-12, rtol=0)


def test_compute_uno_weights_gmin_none_disables_gating():
    from comprisk.metrics import compute_uno_weights

    rng = np.random.default_rng(3)
    n = 100
    time = rng.uniform(0.1, 10.0, n)
    event = rng.integers(0, 3, n)
    if not (event != 0).any():
        event[0] = 1
    w_none = compute_uno_weights(time, event, gmin="none")
    w_zero = compute_uno_weights(time, event, gmin=0.0)
    np.testing.assert_array_equal(w_none, w_zero)


def test_compute_uno_weights_eps_keep_for_gated_out():
    from comprisk.metrics import compute_uno_weights

    # Construct: 4 early events, 95 censored interleaved before time 100,
    # then one late event at t=100 with G(t-) = 1/96 ≫ 1 weight.
    # G_event = [1, 1, 1, 1, 1/96]. ess_target = max(2, ceil(0.4*5)) = 2.
    # k=0 ESS ~ 1.0 (single huge weight dominates); k=1 ESS = 4 ≥ 2 → pass.
    # gmin = g_sorted[1] = 1.0 → late event (G=1/96) gated → eps_keep.
    time = np.array([1, 2, 3, 4, *list(range(5, 100)), 100], dtype=float)
    event = np.array([1, 1, 1, 1, *([0] * 95), 1], dtype=int)
    w = compute_uno_weights(time, event, ess_frac=0.4, ess_min=2)
    eps_keep = float(np.finfo(float).eps)
    assert (w == eps_keep).any()


def test_compute_uno_weights_handles_single_observation():
    from comprisk.metrics import compute_uno_weights

    w = compute_uno_weights(np.array([1.0]), np.array([1]))
    assert w.shape == (1,)
    assert w[0] == 1.0


def test_compute_uno_weights_handles_all_censored():
    from comprisk.metrics import compute_uno_weights

    time = np.array([1.0, 2.0, 3.0])
    event = np.array([0, 0, 0])
    w = compute_uno_weights(time, event)
    # G(t1-)=1.0; at t1 d_cens=1, n_risk-d_event=3, surv = 1*(1-1/3)=2/3. G(t2-)=2/3.
    # At t2: d_cens=1, n_risk-d_event=2, surv = 2/3 * (1-1/2) = 1/3. G(t3-)=1/3.
    # weights = [1, (3/2)^2, 9] = [1, 2.25, 9].
    np.testing.assert_allclose(w, np.array([1.0, 9.0 / 4.0, 9.0]), atol=1e-12)


# ---- Uno IPCW C-index ----------------------------------------------------


def _naive_concordance_index_uno_cr(event, time, estimate, *, cause, weights, eps_tie=0.0):
    """Brute-force two-pass O(n²) reference. Mirrors rfSRC
    getCRConcordanceIndexIPCW_Fenwick (survivalE.c:761) but returns
    concordance (num/denom) rather than 1-num/denom.

    Pass 1: case vs censored at same/later time; weight per pair = 2*w_i.
    Pass 2: case vs competing at same/earlier time; weight per pair =
            2*sqrt(w_i)*sqrt(w_j).
    Tied case-time branch: within tied-time case set, contribute
        denom += (d-1)*sumW, num += 0.5*denomTie + 0.5*tieMass.
    """
    event = np.asarray(event)
    time = np.asarray(time, dtype=float)
    estimate = np.asarray(estimate, dtype=float)
    weights = np.asarray(weights, dtype=float)
    keep = (
        (weights != 0)
        & ~np.isnan(time)
        & ~np.isnan(estimate)
        & ~np.isnan(np.asarray(event, dtype=float))
    )
    t = time[keep]
    e = event[keep]
    p = estimate[keep]
    w = weights[keep]
    n = len(t)
    if n < 2:
        return float("nan")
    w1 = np.sqrt(w)
    is_case = e == cause
    is_cens = e == 0
    is_comp = (e > 0) & (e != cause)

    denom = 0.0
    numer = 0.0

    # Pass 1: case vs (t_j > t_i, any status) OR (t_j == t_i, censored). Mirrors
    # rfSRC bitCount accumulation: event-status comparators added at end of
    # each iteration (line 899) survive into later iterations; censored at
    # curTime added at start (line 837) before case processing.
    for i in np.where(is_case)[0]:
        for j in range(n):
            if j == i:
                continue
            comparable = (t[j] > t[i]) or (t[j] == t[i] and is_cens[j])
            if not comparable:
                continue
            denom += 2.0 * w[i]
            if p[j] < p[i]:
                numer += 2.0 * w[i]
            elif abs(p[j] - p[i]) <= eps_tie:
                numer += w[i]

    # Tied case-time branch
    case_idx = np.where(is_case)[0]
    if len(case_idx) >= 2:
        order = np.argsort(t[case_idx], kind="stable")
        sorted_case = case_idx[order]
        sorted_t = t[sorted_case]
        EPS_T = 0.0
        i_grp = 0
        while i_grp < len(sorted_case):
            j_grp = i_grp
            while (
                j_grp + 1 < len(sorted_case) and abs(sorted_t[j_grp + 1] - sorted_t[i_grp]) <= EPS_T
            ):
                j_grp += 1
            d = j_grp - i_grp + 1
            if d >= 2:
                grp = sorted_case[i_grp : j_grp + 1]
                w_grp = w[grp]
                p_grp = p[grp]
                sumW = w_grp.sum()
                p_order = np.argsort(p_grp, kind="stable")
                sorted_p = p_grp[p_order]
                tieMass = 0.0
                k = 0
                while k < d:
                    L = k
                    while d > L + 1 and abs(sorted_p[L + 1] - sorted_p[L]) <= eps_tie:
                        L += 1
                    c = L - k + 1
                    if c >= 2:
                        rank_w_sum = w_grp[p_order[k : L + 1]].sum()
                        tieMass += (c - 1) * rank_w_sum
                    k = L + 1
                denomTie = (d - 1) * sumW
                denom += denomTie
                numer += 0.5 * denomTie + 0.5 * tieMass
            i_grp = j_grp + 1

    # Pass 2: case vs competing (t_comp <= t_case)
    for i in np.where(is_case)[0]:
        for j in np.where(is_comp)[0]:
            if t[j] <= t[i]:
                pair_w = w1[i] * w1[j]
                denom += 2.0 * pair_w
                if p[j] < p[i]:
                    numer += 2.0 * pair_w
                elif abs(p[j] - p[i]) <= eps_tie:
                    numer += pair_w

    if denom <= 0:
        return float("nan")
    return numer / denom


def test_concordance_index_uno_cr_perfect_ranking():
    from comprisk.metrics import compute_uno_weights, concordance_index_uno_cr

    event = np.array([1, 1, 1, 1])
    time = np.array([1.0, 2.0, 3.0, 4.0])
    risk = np.array([4.0, 3.0, 2.0, 1.0])
    w = compute_uno_weights(time, event, gmin="none")
    assert concordance_index_uno_cr(event, time, risk, cause=1, weights=w) == 1.0


def test_concordance_index_uno_cr_anti_ranking():
    from comprisk.metrics import compute_uno_weights, concordance_index_uno_cr

    event = np.array([1, 1, 1, 1])
    time = np.array([1.0, 2.0, 3.0, 4.0])
    risk = np.array([1.0, 2.0, 3.0, 4.0])
    w = compute_uno_weights(time, event, gmin="none")
    assert concordance_index_uno_cr(event, time, risk, cause=1, weights=w) == 0.0


def test_concordance_index_uno_cr_all_tied_estimate():
    from comprisk.metrics import compute_uno_weights, concordance_index_uno_cr

    event = np.array([1, 1, 1, 1])
    time = np.array([1.0, 2.0, 3.0, 4.0])
    risk = np.array([0.5, 0.5, 0.5, 0.5])
    w = compute_uno_weights(time, event, gmin="none")
    assert concordance_index_uno_cr(event, time, risk, cause=1, weights=w) == 0.5


def test_concordance_index_uno_cr_no_events_returns_nan():
    from comprisk.metrics import compute_uno_weights, concordance_index_uno_cr

    event = np.array([2, 2, 0, 0])
    time = np.array([1.0, 2.0, 3.0, 4.0])
    risk = np.array([0.1, 0.2, 0.3, 0.4])
    w = compute_uno_weights(time, event, gmin="none")
    c = concordance_index_uno_cr(event, time, risk, cause=1, weights=w)
    assert np.isnan(c)


def test_concordance_index_uno_cr_naive_ref_no_censoring():
    from comprisk.metrics import compute_uno_weights, concordance_index_uno_cr

    rng = np.random.default_rng(10)
    n = 80
    time = rng.uniform(0.1, 10.0, n)
    cause = rng.integers(1, 3, n)
    event = cause
    risk = rng.uniform(size=n)
    w = compute_uno_weights(time, event, gmin="none")
    c_prod = concordance_index_uno_cr(event, time, risk, cause=1, weights=w)
    c_ref = _naive_concordance_index_uno_cr(event, time, risk, cause=1, weights=w)
    np.testing.assert_allclose(c_prod, c_ref, atol=1e-12, rtol=0)


def test_concordance_index_uno_cr_naive_ref_mixed_censoring():
    from comprisk.metrics import compute_uno_weights, concordance_index_uno_cr

    rng = np.random.default_rng(11)
    n = 100
    time = rng.uniform(0.1, 10.0, n)
    cause = rng.integers(1, 3, n)
    cens = rng.uniform(size=n) < 0.4
    event = np.where(cens, 0, cause)
    if not (event == 1).any():
        event[0] = 1
    risk = rng.uniform(size=n)
    w = compute_uno_weights(time, event)
    c_prod = concordance_index_uno_cr(event, time, risk, cause=1, weights=w)
    c_ref = _naive_concordance_index_uno_cr(event, time, risk, cause=1, weights=w)
    np.testing.assert_allclose(c_prod, c_ref, atol=1e-12, rtol=0)


def test_concordance_index_uno_cr_naive_ref_with_time_ties():
    from comprisk.metrics import compute_uno_weights, concordance_index_uno_cr

    rng = np.random.default_rng(12)
    n = 60
    time = rng.integers(1, 11, n).astype(float)  # ties at 10 distinct values
    cause = rng.integers(1, 3, n)
    cens = rng.uniform(size=n) < 0.3
    event = np.where(cens, 0, cause)
    if not (event == 1).any():
        event[0] = 1
    risk = rng.uniform(size=n)
    w = compute_uno_weights(time, event)
    c_prod = concordance_index_uno_cr(event, time, risk, cause=1, weights=w)
    c_ref = _naive_concordance_index_uno_cr(event, time, risk, cause=1, weights=w)
    np.testing.assert_allclose(c_prod, c_ref, atol=1e-12, rtol=0)


def test_concordance_index_uno_cr_zero_weights_filtered():
    from comprisk.metrics import concordance_index_uno_cr

    event = np.array([1, 1, 0, 0])
    time = np.array([1.0, 2.0, 3.0, 4.0])
    risk = np.array([2.0, 1.0, 0.5, 0.1])
    w = np.array([0.0, 1.0, 1.0, 1.0])
    c = concordance_index_uno_cr(event, time, risk, cause=1, weights=w)
    # After filter: 1 case (i=1, t=2, risk=1), 2 censored at t>=2 with lower risk → C=1.0
    assert c == 1.0


def test_concordance_index_uno_cr_empty_input_returns_nan():
    from comprisk.metrics import concordance_index_uno_cr

    c = concordance_index_uno_cr(
        np.array([1]),
        np.array([1.0]),
        np.array([0.5]),
        cause=1,
        weights=np.array([1.0]),
    )
    assert np.isnan(c)
