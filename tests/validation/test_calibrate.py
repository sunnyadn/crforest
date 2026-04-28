from pathlib import Path

import numpy as np
import pandas as pd
from validation.calibrate import calibrate
from validation.config import HarnessConfig


def _make_toy(tmp_path: Path, n: int = 40) -> None:
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
    train, test = make_splits(n, event, seed=0, test_frac=0.2)
    rows.append(
        pd.DataFrame(
            {
                "seed": np.zeros(len(train), dtype=np.int32),
                "sample_id": train.astype(np.int32),
                "fold": "train",
            }
        )
    )
    rows.append(
        pd.DataFrame(
            {
                "seed": np.zeros(len(test), dtype=np.int32),
                "sample_id": test.astype(np.int32),
                "fold": "test",
            }
        )
    )
    splits_df = pd.concat(rows, ignore_index=True)
    (tmp_path / "splits").mkdir()
    splits_df.to_parquet(tmp_path / "splits" / "toy.parquet", index=False)


def test_calibrate_reports_positive_seconds(tmp_path: Path, monkeypatch):
    _make_toy(tmp_path)
    monkeypatch.setattr("validation.datasets._DATA_DIR", tmp_path / "data")
    monkeypatch.setattr("validation.splits._SPLITS_DIR", tmp_path / "splits")

    config = HarnessConfig(n_estimators=3, min_samples_leaf=3, min_samples_split=6)
    timings = calibrate(["toy"], config=config)
    assert set(timings.keys()) == {"toy"}
    assert timings["toy"] > 0.0
