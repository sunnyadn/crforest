from pathlib import Path

import numpy as np
import pytest
from validation.datasets import load

DATASETS_DIR = Path(__file__).resolve().parents[2] / "validation" / "data"


@pytest.mark.parametrize("name", ["pbc", "follic", "hd", "synthetic"])
def test_dataset_loads_with_shape_contract(name: str):
    pq = DATASETS_DIR / f"{name}.parquet"
    if not pq.exists():
        pytest.skip(
            f"{name} parquet missing; run "
            "`Rscript validation/gen_datasets.R && uv run python -m validation.gen_datasets`"
        )
    X, time, event = load(name)
    assert X.ndim == 2
    assert time.ndim == 1
    assert event.ndim == 1
    assert len(X) == len(time) == len(event)
    assert np.all(time > 0)
    assert np.all(event >= 0)
    assert np.any(event > 0)
    causes = set(np.unique(event[event > 0]).tolist())
    assert causes.issubset({1, 2})
    assert causes == {1, 2}
