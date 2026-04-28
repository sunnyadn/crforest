import dataclasses

import pytest
from validation.config import DATASETS, DEFAULT, HarnessConfig


def test_harness_config_is_frozen():
    assert dataclasses.is_dataclass(HarnessConfig)
    with pytest.raises(dataclasses.FrozenInstanceError):
        DEFAULT.n_estimators = 999


def test_default_values():
    assert DEFAULT.n_estimators == 500
    assert DEFAULT.min_samples_leaf == 1
    assert DEFAULT.min_samples_split == 30
    assert DEFAULT.max_features == "sqrt"
    assert DEFAULT.max_depth is None
    assert DEFAULT.bootstrap is True
    assert DEFAULT.test_frac == 0.2
    assert DEFAULT.cause == 1
    assert DEFAULT.n_seeds == 20


def test_datasets_registry():
    assert DATASETS == ["pbc", "follic", "hd", "synthetic"]
