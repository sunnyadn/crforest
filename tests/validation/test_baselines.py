from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from validation.baselines import load_baseline


def test_load_baseline_returns_ordered_risk(tmp_path: Path, monkeypatch):
    df = pd.DataFrame(
        {
            "seed": np.array([0, 0, 0, 1, 1, 1], dtype=np.int32),
            "sample_id": np.array([7, 2, 5, 3, 8, 1], dtype=np.int32),
            "risk_cause_1": np.array([0.1, 0.9, 0.4, 0.2, 0.5, 0.3], dtype=np.float64),
        }
    )
    baselines_dir = tmp_path / "baselines"
    baselines_dir.mkdir()
    df.to_parquet(baselines_dir / "toy.parquet", index=False)
    monkeypatch.setattr("validation.baselines._BASELINES_DIR", baselines_dir)

    risk_seed_0 = load_baseline("toy", seed=0)
    np.testing.assert_array_equal(risk_seed_0, np.array([0.9, 0.4, 0.1]))

    risk_seed_1 = load_baseline("toy", seed=1)
    np.testing.assert_array_equal(risk_seed_1, np.array([0.3, 0.2, 0.5]))


def test_load_baseline_missing_file(tmp_path: Path, monkeypatch):
    monkeypatch.setattr("validation.baselines._BASELINES_DIR", tmp_path)
    with pytest.raises(FileNotFoundError, match="missing"):
        load_baseline("nonexistent", seed=0)
