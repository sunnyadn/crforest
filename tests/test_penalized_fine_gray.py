"""Tests for ``comprisk.PenalizedFineGrayRegression``.

Correctness is pinned three ways:

* the ``lambda -> 0`` limit must reproduce the unpenalized
  ``FineGrayRegression`` (an independently implemented Newton-Raphson fit);
* coefficients and sandwich SEs along the path must match R ``crrp`` (Fu
  et al. 2017) to 1e-3 on the ``pbc`` / ``follic`` fixtures (regenerate via
  ``Rscript tests/cross_check_crrp.R``);
* the first-order (KKT) optimality conditions must hold at the fitted
  coefficients.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from sklearn.base import clone

from comprisk import FineGrayRegression, PenalizedFineGrayRegression, Surv
from comprisk.fine_gray import _build_event_time_grid, _km_censoring_left_limit
from comprisk.penalized_fine_gray import (
    _prox_lasso,
    _prox_mcp,
    _prox_scad,
    _psh_working,
    _soft_threshold,
)

FIXTURES_DIR = Path(__file__).parent / "fixtures"


def _toy_cr(n: int = 300, p: int = 8, seed: int = 0, signal=(0.9, -0.7, 0.4)):
    """Competing-risks data with sparse Fine-Gray signal in the first columns."""
    rng = np.random.default_rng(seed)
    x = rng.standard_normal((n, p))
    eta = np.zeros(n)
    for j, b in enumerate(signal):
        eta += b * x[:, j]
    t1 = rng.exponential(np.exp(-eta))  # cause 1 (depends on covariates)
    t2 = rng.exponential(2.0, n)  # competing cause 2
    cens = rng.exponential(3.0, n)
    time = np.minimum.reduce([t1, t2, cens]) + 0.01
    raw = time - 0.01
    event = np.where(raw == t1, 1, np.where(raw == t2, 2, 0)).astype(np.int64)
    return x, time, event


# ---------------------------------------------------------------------------
# Proximal / threshold operators
# ---------------------------------------------------------------------------


def test_soft_threshold():
    assert _soft_threshold(5.0, 2.0) == 3.0
    assert _soft_threshold(-5.0, 2.0) == -3.0
    assert _soft_threshold(1.5, 2.0) == 0.0
    assert _soft_threshold(-1.5, 2.0) == 0.0


def test_prox_lasso_basic():
    # Below the L1 threshold -> exactly zero.
    assert _prox_lasso(0.5, 1.0, 0.0, 2.0) == 0.0
    # Above: soft-thresholded then divided by (v + l2).
    assert _prox_lasso(3.0, 1.0, 0.0, 2.0) == pytest.approx((3.0 - 1.0) / 2.0)
    # Pure ridge (l1 = 0): linear shrinkage, never zero for nonzero z.
    assert _prox_lasso(3.0, 0.0, 1.0, 2.0) == pytest.approx(3.0 / 3.0)


def test_prox_mcp_unbiased_for_large_signal():
    # |z| > gamma*l1 -> no shrinkage from the L1 part (the MCP point).
    gamma, l1, v = 3.0, 1.0, 2.0
    assert _prox_mcp(10.0, l1, 0.0, gamma, v) == pytest.approx(10.0 / v)
    assert _prox_mcp(0.5, l1, 0.0, gamma, v) == 0.0


def test_prox_scad_three_regimes():
    gamma, l1, v = 3.7, 1.0, 2.0
    assert _prox_scad(0.5, l1, 0.0, gamma, v) == 0.0  # regime 1: zero
    # regime 3 (|z| > gamma*l1): unbiased
    assert _prox_scad(20.0, l1, 0.0, gamma, v) == pytest.approx(20.0 / v)


# ---------------------------------------------------------------------------
# Path endpoints
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("penalty", ["lasso", "elasticnet", "mcp", "scad"])
def test_lambda_max_zeros_all_coefficients(penalty):
    x, time, event = _toy_cr()
    kw = {"l1_ratio": 0.5} if penalty == "elasticnet" else {}
    m = PenalizedFineGrayRegression(penalty=penalty, n_lambda=50, **kw).fit(
        x, time=time, event=event
    )
    np.testing.assert_allclose(m.coef_path_[:, 0], 0.0, atol=1e-9)


def test_small_lambda_matches_unpenalized_fine_gray():
    x, time, event = _toy_cr()
    y = Surv.from_arrays(event=event, time=time)
    fg = FineGrayRegression().fit(x, y)
    # A negligible penalty: the LASSO path's last value scaled way down.
    m = PenalizedFineGrayRegression(penalty="lasso", n_lambda=60)
    lam_small = m.fit(x, y).lambdas_[-1] * 1e-4
    pen = PenalizedFineGrayRegression(
        penalty="lasso", lambdas=[lam_small], max_iter=5000, tol=1e-8
    ).fit(x, y)
    np.testing.assert_allclose(pen.coef_path_[:, 0], fg.coef_, atol=1e-3)


def test_ridge_shrinks_without_zeroing():
    x, time, event = _toy_cr()
    m = PenalizedFineGrayRegression(penalty="ridge", n_lambda=40).fit(x, time=time, event=event)
    # No exact zeros anywhere on a ridge path (with non-degenerate data).
    assert np.all(np.abs(m.coef_path_) > 0.0)
    # Coefficient norm decreases as lambda grows (lambdas_ is descending).
    norms = np.linalg.norm(m.coef_path_, axis=0)
    assert norms[0] < norms[-1]
    assert np.all(np.diff(norms) >= -1e-9)  # non-decreasing toward small lambda


def test_elasticnet_sparser_than_ridge_denser_than_lasso():
    x, time, event = _toy_cr(seed=3)
    common = dict(n_lambda=50)
    lasso = PenalizedFineGrayRegression(penalty="lasso", **common).fit(x, time=time, event=event)
    enet = PenalizedFineGrayRegression(penalty="elasticnet", l1_ratio=0.4, **common).fit(
        x, time=time, event=event
    )
    ridge = PenalizedFineGrayRegression(penalty="ridge", **common).fit(x, time=time, event=event)
    # Compare active-set sizes at a comparable mid-path lambda.
    idx = 30
    nz_lasso = int((np.abs(lasso.coef_path_[:, idx]) > 1e-8).sum())
    nz_enet = int((np.abs(enet.coef_path_[:, idx]) > 1e-8).sum())
    nz_ridge = int((np.abs(ridge.coef_path_[:, idx]) > 1e-8).sum())
    assert nz_lasso <= nz_enet <= nz_ridge == x.shape[1]


# ---------------------------------------------------------------------------
# First-order optimality (KKT) at the fitted coefficients
# ---------------------------------------------------------------------------


def test_kkt_conditions_at_lasso_solution():
    x, time, event = _toy_cr(n=400, seed=1)
    # standardize=False so KKT is stated directly on the reported coefficients.
    m = PenalizedFineGrayRegression(
        penalty="lasso", n_lambda=60, standardize=False, max_iter=5000, tol=1e-9
    ).fit(x, time=time, event=event)
    # Pick a lambda giving a genuinely partial model.
    nnz = (np.abs(m.coef_path_) > 1e-8).sum(axis=0)
    li = int(np.argmax((nnz > 0) & (nnz < x.shape[1])))
    lam = float(m.lambdas_[li])
    beta = m.coef_path_[:, li]

    n = x.shape[0]
    cengroup = np.zeros(n, dtype=np.int64)
    et = _build_event_time_grid(time, event, 1)
    g_e = _km_censoring_left_limit(time, event, cengroup, et)
    g_s = _km_censoring_left_limit(time, event, cengroup, time)[:, 0]
    st, _w, _ll, _ls = _psh_working(x @ beta, time, event, cengroup, 1, 1, et, g_e, g_s)
    grad = (x.T @ st) / n  # gradient of the (1/n)-scaled partial log-likelihood

    active = np.abs(beta) > 1e-8
    assert active.any() and (~active).any()
    # Active coordinates: |grad_j| == lambda, sign(grad_j) == sign(beta_j).
    np.testing.assert_allclose(np.abs(grad[active]), lam, atol=2e-3)
    np.testing.assert_array_equal(np.sign(grad[active]), np.sign(beta[active]))
    # Inactive coordinates: |grad_j| <= lambda.
    assert np.all(np.abs(grad[~active]) <= lam + 2e-3)


# ---------------------------------------------------------------------------
# R `crrp` regression
# ---------------------------------------------------------------------------

_CRRP_CASES = [
    ("pbc", ["age", "edema", "bili", "albumin", "protime", "stage"], "event"),
    ("follic", ["age", "hgb", "clinstg", "ch"], "status"),
]


@pytest.mark.parametrize("name, cov_cols, event_col", _CRRP_CASES)
@pytest.mark.parametrize("penalty", ["lasso", "mcp", "scad"])
def test_path_matches_crrp_to_1e_3(name, cov_cols, event_col, penalty):
    fixture = FIXTURES_DIR / f"crrp_{name}_{penalty}_fit.csv"
    if not fixture.exists():
        pytest.skip(f"{fixture.name} missing; run Rscript tests/cross_check_crrp.R")
    ref = pd.read_csv(fixture)
    lambdas = ref["lambda"].drop_duplicates().to_numpy()
    n_lambda, p = lambdas.shape[0], len(cov_cols)
    beta_ref = ref["coef"].to_numpy().reshape(n_lambda, p).T
    se_ref = ref["se"].to_numpy().reshape(n_lambda, p).T

    data = pd.read_csv(FIXTURES_DIR / f"cmprsk_{name}_data.csv")
    x = data[cov_cols].to_numpy(dtype=float)
    time = data["time"].to_numpy(dtype=float)
    event = data[event_col].to_numpy(dtype=int)

    m = PenalizedFineGrayRegression(penalty=penalty, lambdas=lambdas, max_iter=5000, tol=1e-7).fit(
        x, time=time, event=event
    )
    order = np.argsort(m.lambdas_)[::-1]  # align to crrp's descending grid
    np.testing.assert_allclose(m.coef_path_[:, order], beta_ref, atol=1e-3)
    np.testing.assert_allclose(m.se_path_[:, order], se_ref, atol=1e-3)


# ---------------------------------------------------------------------------
# Cross-validation
# ---------------------------------------------------------------------------


def test_cv_selects_lambda_and_orders_min_within_1se():
    x, time, event = _toy_cr(n=400, seed=2)
    m = PenalizedFineGrayRegression(penalty="lasso", n_lambda=40, cv=5, cv_random_state=0).fit(
        x, time=time, event=event
    )
    assert m.cv_deviance_.shape == (40,)
    assert m.cv_deviance_se_.shape == (40,)
    assert m.lambda_min_ is not None and m.lambda_1se_ is not None
    assert m.lambda_1se_ >= m.lambda_min_
    # Selected coef_ corresponds to lambda_min_.
    assert m.lambda_ == pytest.approx(m.lambda_min_)
    # The CV-selected model should recover the true signal sign on x0..x2.
    assert m.coef_[0] > 0 and m.coef_[1] < 0


def test_cv_requires_integer_at_least_two():
    x, time, event = _toy_cr(n=120)
    with pytest.raises(ValueError, match="cv must be"):
        PenalizedFineGrayRegression(cv=1).fit(x, time=time, event=event)


# ---------------------------------------------------------------------------
# Standardization invariance
# ---------------------------------------------------------------------------


def test_standardize_makes_path_scale_equivariant():
    x, time, event = _toy_cr(seed=4)
    scales = np.array([1.0, 100.0, 0.01, 5.0, 1.0, 2.0, 1.0, 50.0])
    m1 = PenalizedFineGrayRegression(penalty="lasso", n_lambda=30).fit(x, time=time, event=event)
    m2 = PenalizedFineGrayRegression(penalty="lasso", n_lambda=30).fit(
        x * scales, time=time, event=event
    )
    # On the original scale, beta_j(scaled) = beta_j(orig) / scale_j.
    np.testing.assert_allclose(m2.coef_path_ * scales[:, None], m1.coef_path_, atol=1e-5)


# ---------------------------------------------------------------------------
# sklearn surface
# ---------------------------------------------------------------------------


def test_clone_preserves_constructor_params():
    m = PenalizedFineGrayRegression(penalty="mcp", l1_ratio=0.7, gamma=3.0, n_lambda=33, cv=4)
    c = clone(m)
    assert c.penalty == "mcp"
    assert c.l1_ratio == 0.7
    assert c.gamma == 3.0
    assert c.n_lambda == 33
    assert c.cv == 4
    assert not hasattr(c, "coef_")


def test_get_set_params_roundtrip():
    m = PenalizedFineGrayRegression()
    m.set_params(penalty="scad", n_lambda=20)
    assert m.get_params()["penalty"] == "scad"
    assert m.get_params()["n_lambda"] == 20


def test_surv_y_equivalent_to_legacy_kwargs():
    x, time, event = _toy_cr(seed=5)
    y = Surv.from_arrays(event=event, time=time)
    a = PenalizedFineGrayRegression(penalty="lasso", n_lambda=30).fit(x, y)
    b = PenalizedFineGrayRegression(penalty="lasso", n_lambda=30).fit(x, time=time, event=event)
    np.testing.assert_array_equal(a.coef_path_, b.coef_path_)


# ---------------------------------------------------------------------------
# predict / predict_cumulative_incidence
# ---------------------------------------------------------------------------


def test_predict_returns_linear_predictor():
    x, time, event = _toy_cr()
    m = PenalizedFineGrayRegression(penalty="lasso", n_lambda=20).fit(x, time=time, event=event)
    np.testing.assert_allclose(m.predict(x[:5]), x[:5] @ m.coef_)


def test_predict_cumulative_incidence_shape_and_bounds():
    x, time, event = _toy_cr()
    m = PenalizedFineGrayRegression(penalty="lasso", n_lambda=20, cv=5, cv_random_state=0).fit(
        x, time=time, event=event
    )
    f_default = m.predict_cumulative_incidence(x[:7])
    assert f_default.shape[0] == 7
    assert np.all((f_default >= 0.0) & (f_default <= 1.0))
    # Monotone non-decreasing in time.
    assert np.all(np.diff(f_default, axis=1) >= -1e-12)
    f_times = m.predict_cumulative_incidence(x[:3], times=[0.5, 1.0, 2.0, 5.0])
    assert f_times.shape == (3, 4)
    assert np.all((f_times >= 0.0) & (f_times <= 1.0))


def test_coef_at_index_and_value():
    x, time, event = _toy_cr()
    m = PenalizedFineGrayRegression(penalty="lasso", n_lambda=30).fit(x, time=time, event=event)
    np.testing.assert_array_equal(m.coef_at(lambda_index=0), m.coef_path_[:, 0])
    # Nearest-lambda lookup.
    target = float(m.lambdas_[7])
    np.testing.assert_array_equal(m.coef_at(lambda_value=target), m.coef_path_[:, 7])
    with pytest.raises(ValueError, match="exactly one"):
        m.coef_at()


def test_predict_before_fit_raises():
    m = PenalizedFineGrayRegression()
    with pytest.raises(RuntimeError, match="not fitted"):
        m.predict(np.zeros((3, 4)))


# ---------------------------------------------------------------------------
# BIC selection (default when cv is None)
# ---------------------------------------------------------------------------


def test_bic_selection_yields_sparse_coef():
    x, time, event = _toy_cr(n=400, seed=6)
    m = PenalizedFineGrayRegression(penalty="lasso", n_lambda=60).fit(x, time=time, event=event)
    assert m.lambda_min_ is None  # no CV ran
    assert m.lambda_index_ == int(np.nanargmin(m.bic_path_))
    # The true model has 3 active covariates; BIC should land near that.
    assert 1 <= int((np.abs(m.coef_) > 0).sum()) <= 5
    assert m.coef_[0] > 0 and m.coef_[1] < 0


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------


def test_invalid_penalty_raises():
    x, time, event = _toy_cr(n=100)
    with pytest.raises(ValueError, match="penalty must be one of"):
        PenalizedFineGrayRegression(penalty="bridge").fit(x, time=time, event=event)


def test_invalid_gamma_raises():
    x, time, event = _toy_cr(n=100)
    with pytest.raises(ValueError, match="gamma must be > 1 for MCP"):
        PenalizedFineGrayRegression(penalty="mcp", gamma=0.5).fit(x, time=time, event=event)
    with pytest.raises(ValueError, match="gamma must be > 2 for SCAD"):
        PenalizedFineGrayRegression(penalty="scad", gamma=1.5).fit(x, time=time, event=event)


def test_invalid_l1_ratio_raises():
    x, time, event = _toy_cr(n=100)
    with pytest.raises(ValueError, match="l1_ratio must be in"):
        PenalizedFineGrayRegression(penalty="elasticnet", l1_ratio=0.0).fit(
            x, time=time, event=event
        )


def test_no_cause_events_raises():
    x, time, event = _toy_cr(n=120)
    event = np.where(event == 1, 2, event)  # delete all cause-1 events
    with pytest.raises(ValueError, match="no subjects have event == cause"):
        PenalizedFineGrayRegression().fit(x, time=time, event=event)


def test_mismatched_lengths_raise():
    x, time, event = _toy_cr(n=120)
    with pytest.raises(ValueError, match="time/event length"):
        PenalizedFineGrayRegression().fit(x, time=time[:-1], event=event)


def test_cengroup_path_runs():
    x, time, event = _toy_cr(n=200, seed=7)
    cg = (np.arange(200) % 2).astype(np.int64)
    m = PenalizedFineGrayRegression(penalty="lasso", n_lambda=20).fit(
        x, time=time, event=event, cengroup=cg
    )
    assert m.coef_path_.shape == (8, 20)


def test_cencode_remap():
    x, time, event = _toy_cr(n=200, seed=8)
    # Recode censored 0 -> 9 and ask the estimator to treat 9 as censoring.
    event_remap = np.where(event == 0, 9, event)
    a = PenalizedFineGrayRegression(penalty="lasso", n_lambda=20).fit(x, time=time, event=event)
    b = PenalizedFineGrayRegression(penalty="lasso", n_lambda=20, cencode=9).fit(
        x, time=time, event=event_remap
    )
    np.testing.assert_allclose(a.coef_path_, b.coef_path_, atol=1e-12)
