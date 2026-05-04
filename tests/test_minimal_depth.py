"""Unit tests for minimal-depth variable selection (SUN-42)."""

from __future__ import annotations

import numpy as np
import pytest

from crforest import CompetingRiskForest


def _toy(n=200, p=4, seed=0, n_causes=2):
    rng = np.random.RandomState(seed)
    X = rng.randn(n, p)
    time = rng.uniform(0.1, 10, n)
    event = rng.randint(0, n_causes + 1, n).astype(np.int64)
    return X, time, event


def _fit(seed=0, n_jobs=1, equivalence=None, mode=None, **kw):
    X, time, event = _toy(seed=seed)
    kwargs = dict(
        n_estimators=20, max_depth=4, min_samples_leaf=5, random_state=seed, n_jobs=n_jobs
    )
    if equivalence is not None:
        kwargs["equivalence"] = equivalence
    if mode is not None:
        kwargs["mode"] = mode
    kwargs.update(kw)
    forest = CompetingRiskForest(**kwargs).fit(
        X, np.array(list(zip(event, time, strict=False)), dtype=[("event", "i8"), ("time", "f8")])
    )
    return forest


def test_schema():
    forest = _fit(seed=0)
    df = forest.minimal_depth()
    assert list(df.columns) == ["feature", "mean_min_depth", "threshold", "selected"]
    assert len(df) == forest.n_features_in_
    # sorted ascending by mean_min_depth
    assert (df["mean_min_depth"].values[:-1] <= df["mean_min_depth"].values[1:]).all()


def test_walker_flat_tree_finds_root_split():
    from crforest._minimal_depth import _walk_min_depth
    from crforest._tree_flat import FlatTree

    forest = _fit(seed=0)
    tree = forest.trees_[0]
    # FlatTree path is the default
    assert isinstance(tree, FlatTree)
    res = _walk_min_depth(tree, n_features=forest.n_features_in_)
    assert res.min_depth_per_feature.shape == (forest.n_features_in_,)
    assert res.min_depth_per_feature.dtype == np.int32
    # At least one feature must be the root split (depth 0)
    assert res.min_depth_per_feature.min() == 0
    # Every value is in [0, D_T + 1]
    assert (res.min_depth_per_feature >= 0).all()
    assert (res.min_depth_per_feature <= res.max_depth + 1).all()


def test_ishwaran_threshold_handcomputed():
    """Depth-2 toy tree: 1 internal node at depth 0, 2 internals at depth 1.

    p = 4, so (1 - 1/p) = 3/4. Cumulative internal counts cumL = [1, 3, 3].
    P(md > 0) = (3/4)^1 = 0.75
    P(md > 1) = (3/4)^3 = 27/64 = 0.421875
    P(md > 2) = (3/4)^3 = 27/64 = 0.421875
    E[md] = 0.75 + 0.421875 + 0.421875 = 1.59375

    Sanity: P(md=0)=0.25, P(md=1)=0.328125, P(md=2)=0, P(md=3)=0.421875
            -> 0*0.25 + 1*0.328125 + 2*0 + 3*0.421875 = 1.59375
    """
    from crforest._minimal_depth import _ishwaran_expected_md

    L = np.array([1, 2], dtype=np.int64)
    expected = 1.59375
    got = _ishwaran_expected_md(L, max_depth_T=2, n_features=4)
    assert abs(got - expected) < 1e-12, f"got {got}, expected {expected}"


def test_ishwaran_threshold_pure_stump():
    """Pure stump (D_T = 0, no internals): expected md = 1.0."""
    from crforest._minimal_depth import _ishwaran_expected_md

    L = np.array([], dtype=np.int64)
    got = _ishwaran_expected_md(L, max_depth_T=0, n_features=4)
    # cumL_full = [0]; P(md>0) = (3/4)^0 = 1.0; sum = 1.0
    assert abs(got - 1.0) < 1e-12


def test_determinism_across_n_jobs():
    f1 = _fit(seed=42, n_jobs=1)
    f4 = _fit(seed=42, n_jobs=4)
    df1 = f1.minimal_depth()
    df4 = f4.minimal_depth()
    pd_assert_frame = __import__("pandas").testing.assert_frame_equal
    pd_assert_frame(df1, df4)


def test_planted_signal_ranks_above_noise():
    """3 informative + 7 noise features. Informative land in top-3."""
    rng = np.random.RandomState(7)
    n, p_signal, p_noise = 2000, 3, 7
    X = rng.randn(n, p_signal + p_noise)
    # event hazard depends on first 3 features
    lin = X[:, :p_signal] @ np.array([1.0, -1.0, 0.5])
    time = rng.exponential(scale=np.exp(-lin) + 0.1)
    event = (rng.uniform(size=n) < 0.7).astype(np.int64)  # ~70% event
    event[event == 1] = (rng.randint(1, 3, size=event.sum())).astype(np.int64)
    y = np.array(list(zip(event, time, strict=False)), dtype=[("event", "i8"), ("time", "f8")])
    forest = CompetingRiskForest(
        n_estimators=200, max_depth=8, min_samples_leaf=10, random_state=7, n_jobs=1
    ).fit(X, y)
    df = forest.minimal_depth()
    top3 = set(df["feature"].iloc[:3].tolist())
    informative = {f"feature_{i}" for i in range(p_signal)}
    overlap = len(top3 & informative)
    assert overlap >= 2, f"top-3 = {top3}; informative = {informative}; overlap {overlap}/3"


def test_pure_stump_edge_case():
    """Forest fit with min_samples_leaf > n/2 yields pure stumps (no splits)."""
    rng = np.random.RandomState(0)
    n, p = 50, 4
    X = rng.randn(n, p)
    time = rng.uniform(0.1, 10, n)
    # 2 events only; min_samples_leaf=40 forces every bootstrap sample to be unsplittable
    event = np.zeros(n, dtype=np.int64)
    event[:2] = 1
    y = np.array(list(zip(event, time, strict=False)), dtype=[("event", "i8"), ("time", "f8")])
    forest = CompetingRiskForest(
        n_estimators=10, max_depth=4, min_samples_leaf=40, random_state=0, n_jobs=1
    ).fit(X, y)
    df = forest.minimal_depth()
    # All features should have mean_min_depth == 1.0 (D_T=0, sentinel = 1)
    assert (df["mean_min_depth"] == 1.0).all()
    # Edge-case: thr = 1.0 too, so selected = mean_md <= thr is True everywhere.
    # This is uninformative output, but mathematically consistent.
    assert df["selected"].all()


def test_tree_mode_coverage():
    """Works on default FlatTree, equivalence='rfsrc' (HistTreeNode), mode='reference' (RefTreeNode)."""
    from crforest._hist_tree import HistTreeNode
    from crforest._tree import RefTreeNode
    from crforest._tree_flat import FlatTree

    cols = ["feature", "mean_min_depth", "threshold", "selected"]

    f_default = _fit(seed=1)
    df_default = f_default.minimal_depth()
    assert isinstance(f_default.trees_[0], FlatTree)
    assert list(df_default.columns) == cols
    assert len(df_default) == f_default.n_features_in_

    f_rfsrc = _fit(seed=1, equivalence="rfsrc")
    df_rfsrc = f_rfsrc.minimal_depth()
    assert isinstance(f_rfsrc.trees_[0], HistTreeNode)
    assert list(df_rfsrc.columns) == cols
    assert len(df_rfsrc) == f_rfsrc.n_features_in_

    f_ref = _fit(seed=1, mode="reference")
    df_ref = f_ref.minimal_depth()
    assert isinstance(f_ref.trees_[0], RefTreeNode)
    assert list(df_ref.columns) == cols
    assert len(df_ref) == f_ref.n_features_in_


def test_conservative_selects_subset_of_default():
    forest = _fit(seed=3)
    sel_default = set(forest.minimal_depth(conservative=False).query("selected").feature.tolist())
    sel_strict = set(forest.minimal_depth(conservative=True).query("selected").feature.tolist())
    assert sel_strict.issubset(sel_default)


def test_return_extra_columns():
    forest = _fit(seed=4)
    df = forest.minimal_depth(return_extra=True)
    assert list(df.columns) == [
        "feature",
        "mean_min_depth",
        "threshold",
        "selected",
        "min_depth_q25",
        "min_depth_q75",
        "frac_trees_used",
    ]
    # quartiles bounded by mean_min_depth ordering invariants
    assert (df["min_depth_q25"] <= df["min_depth_q75"]).all()
    assert ((df["frac_trees_used"] >= 0.0) & (df["frac_trees_used"] <= 1.0)).all()


def test_unfitted_raises():
    from sklearn.exceptions import NotFittedError

    forest = CompetingRiskForest(n_estimators=5)
    with pytest.raises(NotFittedError):
        forest.minimal_depth()


def test_invalid_threshold_raises():
    forest = _fit(seed=0)
    with pytest.raises(ValueError, match="threshold must be 'md'"):
        forest.minimal_depth(threshold="vh")


def test_invalid_conservative_type_raises():
    forest = _fit(seed=0)
    with pytest.raises(TypeError, match="conservative must be bool"):
        forest.minimal_depth(conservative="yes")  # type: ignore[arg-type]


def test_rfsrc_var_select_match_follic():
    """Bit-equivalent ranking vs randomForestSRC::var.select(method='md') on follic.

    rfSRC >= 3.x renamed var.select(method='md') to max.subtree(max.order=1).
    The oracle JSON was generated by validation/alignment/gen_var_select_oracle.R.
    """
    import json
    from pathlib import Path

    fixture = Path(__file__).resolve().parent / "fixtures" / "rfsrc_var_select_follic.json"
    if not fixture.exists():
        pytest.skip(f"oracle missing: {fixture}; run validation/alignment/gen_var_select_oracle.R")
    oracle = json.loads(fixture.read_text())

    # Load follic from rpy2 if available, otherwise skip — we mirror rfSRC's bundled data
    pytest.importorskip("rpy2")
    import rpy2.robjects
    from rpy2.robjects import pandas2ri
    from rpy2.robjects import r as R  # noqa: N812

    R("suppressMessages(library(randomForestSRC)); data(follic)")
    follic_r = R("follic")
    with (rpy2.robjects.default_converter + pandas2ri.converter).context():
        follic = rpy2.robjects.conversion.get_conversion().rpy2py(follic_r)

    import pandas as pd

    feature_cols = [c for c in follic.columns if c not in ("time", "status")]
    # Encode categoricals as integer codes to match rfSRC's internal numeric encoding.
    # rfSRC treats factor levels as their 1-based integer codes internally.
    X_raw = {}
    for c in feature_cols:
        col = follic[c]
        if hasattr(col, "cat"):
            # factor → 1-based integer code (R convention: level index starting at 1)
            X_raw[c] = col.cat.codes.to_numpy(dtype=np.float64) + 1.0
        else:
            X_raw[c] = col.to_numpy(dtype=np.float64)
    X = pd.DataFrame(X_raw)
    y = np.array(
        list(
            zip(
                follic["status"].astype(np.int64),
                follic["time"].astype(np.float64),
                strict=False,
            )
        ),
        dtype=[("event", "i8"), ("time", "f8")],
    )

    forest = CompetingRiskForest(
        n_estimators=oracle["ntree"],
        equivalence="rfsrc",
        random_state=oracle["seed"],
        min_samples_leaf=15,
        n_jobs=1,
    ).fit(X, y)
    # Inject feature names so minimal_depth() uses the oracle's column names
    forest.feature_names_in_ = feature_cols  # type: ignore[attr-defined]

    df = forest.minimal_depth()
    got_ranking = df["feature"].tolist()
    assert got_ranking == oracle["ranking"], (
        f"ranking mismatch:\n  got: {got_ranking}\n  exp: {oracle['ranking']}"
    )
