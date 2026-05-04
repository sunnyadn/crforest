"""Parallel-fit equivalence gate.

Guarantees ``CompetingRiskForest.fit`` produces bit-identical trees,
oob indices, and predictions regardless of ``n_jobs``, given a fixed
``random_state``. The invariant is preserved by serially pre-seeding
per-tree RNGs inside ``_build_ensemble`` before joblib dispatch.
"""

import numpy as np
import pytest
from validation.datasets import load as load_dataset

from comprisk.forest import CompetingRiskForest
from tests._tree_walkers import walk_tree


def _assert_forests_equal(f1, f2, mode):
    assert len(f1.trees_) == len(f2.trees_)
    for i, (t1, t2) in enumerate(zip(f1.trees_, f2.trees_, strict=True)):
        assert walk_tree(t1) == walk_tree(t2), (
            f"tree {i} differs between n_jobs=1 and n_jobs=2 (mode={mode})"
        )
    assert len(f1.oob_indices_) == len(f2.oob_indices_)
    for i, (o1, o2) in enumerate(zip(f1.oob_indices_, f2.oob_indices_, strict=True)):
        np.testing.assert_array_equal(o1, o2, err_msg=f"oob indices tree {i}")


@pytest.mark.parametrize("mode", ["default", "reference"])
@pytest.mark.parametrize("splitrule", ["logrankCR", "logrank"])
@pytest.mark.parametrize("nsplit", [0, 10])
def test_parallel_fit_bit_identical(mode, splitrule, nsplit):
    X, time, event = load_dataset("pbc")

    f_serial = CompetingRiskForest(
        n_estimators=10,
        mode=mode,
        random_state=0,
        n_jobs=1,
        splitrule=splitrule,
        cause=1,
        nsplit=nsplit,
    ).fit(X, time, event)
    f_parallel = CompetingRiskForest(
        n_estimators=10,
        mode=mode,
        random_state=0,
        n_jobs=2,
        splitrule=splitrule,
        cause=1,
        nsplit=nsplit,
    ).fit(X, time, event)

    _assert_forests_equal(f_serial, f_parallel, mode)

    cif_serial = f_serial.predict_cif(X)
    cif_parallel = f_parallel.predict_cif(X)
    np.testing.assert_array_equal(cif_serial, cif_parallel)

    chf_serial = f_serial.predict_chf(X)
    chf_parallel = f_parallel.predict_chf(X)
    np.testing.assert_array_equal(chf_serial, chf_parallel)
