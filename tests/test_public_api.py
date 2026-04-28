"""Sanity check that the public API is importable from the package root."""

import crforest


def test_top_level_exports():
    assert hasattr(crforest, "CompetingRiskForest")
    assert hasattr(crforest, "concordance_index_cr")


def test_private_modules_not_in_all():
    assert "_tree" not in crforest.__all__
    assert "_splits" not in crforest.__all__
    assert "_estimators" not in crforest.__all__
    assert "_validation" not in crforest.__all__


def test_public_symbols_in_all():
    assert "CompetingRiskForest" in crforest.__all__
    assert "concordance_index_cr" in crforest.__all__
    assert "__version__" in crforest.__all__


def test_n_jobs_stored_on_estimator():
    forest = crforest.CompetingRiskForest(n_jobs=2)
    assert forest.n_jobs == 2


def test_n_jobs_defaults_to_minus_one():
    forest = crforest.CompetingRiskForest()
    assert forest.n_jobs == -1


def test_predict_chf_exposed_on_forest():
    """predict_chf is part of the public API and works on both modes."""
    import numpy as np

    rng = np.random.default_rng(0)
    X = rng.uniform(size=(30, 2))
    time = rng.uniform(0.1, 5.0, 30)
    event = rng.integers(0, 3, 30)
    if not np.any(event == 1):
        event[0] = 1
    if not np.any(event == 2):
        event[1] = 2
    f = crforest.CompetingRiskForest(n_estimators=3, mode="reference", random_state=0).fit(
        X, time, event
    )
    chf = f.predict_chf(X)
    assert chf.ndim == 3
    assert chf.shape[0] == 30
