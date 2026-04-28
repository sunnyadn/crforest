from pathlib import Path

import numpy as np
import pytest
from validation.datasets import load as load_dataset
from validation.splits import load as load_splits
from validation.splits import make_splits

_SPLITS_DIR = Path(__file__).resolve().parents[2] / "validation" / "splits"


def test_make_splits_deterministic():
    event = np.array([0, 1, 2] * 100)
    train1, test1 = make_splits(len(event), event, seed=42, test_frac=0.2)
    train2, test2 = make_splits(len(event), event, seed=42, test_frac=0.2)
    np.testing.assert_array_equal(train1, train2)
    np.testing.assert_array_equal(test1, test2)


def test_make_splits_complement():
    event = np.array([0, 1, 2] * 100)
    n = len(event)
    train, test = make_splits(n, event, seed=0, test_frac=0.2)
    assert len(np.intersect1d(train, test)) == 0
    assert len(train) + len(test) == n
    assert set(np.concatenate([train, test]).tolist()) == set(range(n))


def test_make_splits_stratification_preserves_causes():
    rng = np.random.default_rng(0)
    event = rng.choice([0, 1, 2], size=400, p=[0.6, 0.25, 0.15])
    for seed in range(10):
        train, test = make_splits(len(event), event, seed=seed, test_frac=0.2)
        for code in (0, 1, 2):
            assert (event[train] == code).sum() > 0, f"seed {seed}, code {code} missing from train"
            assert (event[test] == code).sum() > 0, f"seed {seed}, code {code} missing from test"


def test_make_splits_test_frac_shape():
    event = np.array([0, 1] * 500)
    _train, test = make_splits(len(event), event, seed=0, test_frac=0.2)
    assert abs(len(test) - 200) <= 3


def test_make_splits_rejects_small_stratum():
    event = np.array([0, 0, 0, 1, 1, 1, 2])
    with pytest.raises(ValueError, match="stratum"):
        make_splits(len(event), event, seed=0, test_frac=0.2)


def test_make_splits_rejects_length_mismatch():
    event = np.array([0, 1, 0, 1])
    with pytest.raises(ValueError, match="does not match"):
        make_splits(99, event, seed=0)


def test_splits_parquet_matches_generator():
    """Committed splits parquet must equal what make_splits produces today."""
    for name in ["pbc", "follic", "hd", "synthetic"]:
        pq = _SPLITS_DIR / f"{name}.parquet"
        if not pq.exists():
            pytest.skip(f"{pq} missing; run `uv run python -m validation.gen_splits`")
        _, _, event = load_dataset(name)
        loaded = load_splits(name)
        assert len(loaded) == 100, f"{name}: expected 100 seeds"
        for seed, (train, test) in enumerate(loaded):
            exp_train, exp_test = make_splits(len(event), event, seed=seed, test_frac=0.2)
            np.testing.assert_array_equal(
                np.sort(train), exp_train, err_msg=f"{name} seed {seed} train"
            )
            np.testing.assert_array_equal(
                np.sort(test), exp_test, err_msg=f"{name} seed {seed} test"
            )
