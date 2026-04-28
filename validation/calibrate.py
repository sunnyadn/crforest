"""Fit crforest on 1 seed per dataset and return wall-clock fit time."""

from __future__ import annotations

import time

from crforest import CompetingRiskForest
from validation.config import HarnessConfig
from validation.datasets import load as load_dataset
from validation.splits import load as load_splits


def calibrate(datasets: list[str], config: HarnessConfig) -> dict[str, float]:
    """Return seconds per seed for each named dataset."""
    timings: dict[str, float] = {}
    for name in datasets:
        X, time_arr, event = load_dataset(name)
        splits = load_splits(name)
        train_idx, _ = splits[0]
        forest = CompetingRiskForest(
            n_estimators=config.n_estimators,
            min_samples_leaf=config.min_samples_leaf,
            min_samples_split=config.min_samples_split,
            max_features=config.max_features,
            max_depth=config.max_depth,
            bootstrap=config.bootstrap,
            random_state=0,
        )
        start = time.perf_counter()
        forest.fit(X[train_idx], time_arr[train_idx], event[train_idx])
        elapsed = time.perf_counter() - start
        timings[name] = elapsed
    return timings
