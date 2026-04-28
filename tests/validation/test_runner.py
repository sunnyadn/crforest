from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from validation.config import HarnessConfig
from validation.runner import SeedResult, run_dataset


def _make_toy(tmp_path: Path, n: int = 60) -> None:
    rng = np.random.default_rng(0)
    X = rng.uniform(size=(n, 3))
    time = 10.0 - 5.0 * X[:, 0] + rng.normal(scale=0.5, size=n)
    time = np.clip(time, 0.1, None)
    event = rng.integers(0, 3, size=n).astype(np.int64)
    if not (event == 1).any():
        event[0] = 1
    if not (event == 2).any():
        event[1] = 2
    df = pd.DataFrame(X, columns=["x0", "x1", "x2"])
    df["time"] = time
    df["event"] = event
    (tmp_path / "data").mkdir()
    df.to_parquet(tmp_path / "data" / "toy.parquet", index=False)

    from validation.splits import make_splits

    rows = []
    for seed in [0, 1]:
        train, test = make_splits(n, event, seed=seed, test_frac=0.2)
        rows.append(
            pd.DataFrame(
                {
                    "seed": np.full(len(train), seed, dtype=np.int32),
                    "sample_id": train.astype(np.int32),
                    "fold": "train",
                }
            )
        )
        rows.append(
            pd.DataFrame(
                {
                    "seed": np.full(len(test), seed, dtype=np.int32),
                    "sample_id": test.astype(np.int32),
                    "fold": "test",
                }
            )
        )
    splits_df = pd.concat(rows, ignore_index=True)
    (tmp_path / "splits").mkdir()
    splits_df.to_parquet(tmp_path / "splits" / "toy.parquet", index=False)

    baseline_rows = []
    for seed in [0, 1]:
        test_ids = splits_df.loc[
            (splits_df["seed"] == seed) & (splits_df["fold"] == "test"),
            "sample_id",
        ].to_numpy()
        baseline_rows.append(
            pd.DataFrame(
                {
                    "seed": np.full(len(test_ids), seed, dtype=np.int32),
                    "sample_id": test_ids.astype(np.int32),
                    "risk_cause_1": rng.uniform(size=len(test_ids)),
                }
            )
        )
    baselines_df = pd.concat(baseline_rows, ignore_index=True)
    (tmp_path / "baselines").mkdir()
    baselines_df.to_parquet(tmp_path / "baselines" / "toy.parquet", index=False)


def test_run_dataset_returns_seed_results(tmp_path: Path, monkeypatch):
    _make_toy(tmp_path)
    monkeypatch.setattr("validation.datasets._DATA_DIR", tmp_path / "data")
    monkeypatch.setattr("validation.splits._SPLITS_DIR", tmp_path / "splits")
    monkeypatch.setattr("validation.baselines._BASELINES_DIR", tmp_path / "baselines")

    config = HarnessConfig(n_estimators=3, min_samples_leaf=3, min_samples_split=6)
    results = run_dataset("toy", seeds=[0, 1], config=config, n_jobs=1)

    assert len(results) == 2
    assert all(isinstance(r, SeedResult) for r in results)
    seeds = sorted(r.seed for r in results)
    assert seeds == [0, 1]
    for r in results:
        assert r.dataset == "toy"
        assert 0.0 <= r.c_crforest <= 1.0
        assert 0.0 <= r.c_rfsrc <= 1.0
        assert np.isfinite(r.delta_c)
        assert abs(r.delta_c - (r.c_crforest - r.c_rfsrc)) < 1e-12


def test_run_dataset_parallel_matches_serial(tmp_path: Path, monkeypatch):
    _make_toy(tmp_path)
    monkeypatch.setattr("validation.datasets._DATA_DIR", tmp_path / "data")
    monkeypatch.setattr("validation.splits._SPLITS_DIR", tmp_path / "splits")
    monkeypatch.setattr("validation.baselines._BASELINES_DIR", tmp_path / "baselines")

    config = HarnessConfig(n_estimators=3, min_samples_leaf=3, min_samples_split=6)
    serial = run_dataset("toy", seeds=[0, 1], config=config, n_jobs=1)
    # Use threading backend so the monkeypatched module globals propagate
    # (loky workers would re-import modules and lose the monkeypatch).
    import joblib

    with joblib.parallel_backend("threading"):
        parallel = run_dataset("toy", seeds=[0, 1], config=config, n_jobs=2)

    serial_map = {r.seed: r for r in serial}
    parallel_map = {r.seed: r for r in parallel}
    for seed in [0, 1]:
        assert serial_map[seed].c_crforest == pytest.approx(parallel_map[seed].c_crforest)
        assert serial_map[seed].c_rfsrc == pytest.approx(parallel_map[seed].c_rfsrc)
