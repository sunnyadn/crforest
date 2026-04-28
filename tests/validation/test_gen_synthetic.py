from pathlib import Path

import numpy as np
import pandas as pd
from validation.gen_synthetic import generate_synthetic


def test_generate_synthetic_shape_and_deterministic():
    df1 = generate_synthetic()
    df2 = generate_synthetic()
    assert (df1.values == df2.values).all()
    assert df1.shape == (1000, 12)
    expected_x = [f"x{i}" for i in range(10)]
    assert list(df1.columns) == [*expected_x, "time", "event"]
    events = df1["event"].to_numpy()
    assert set(np.unique(events).tolist()) == {0, 1, 2}
    times = df1["time"].to_numpy()
    assert np.all(times > 0) and np.all(np.isfinite(times))
    censored_frac = (events == 0).mean()
    assert 0.15 < censored_frac < 0.45


def test_vendored_parquet_exists_and_matches():
    pq = Path(__file__).resolve().parents[2] / "validation" / "data" / "synthetic.parquet"
    assert pq.exists(), "run `python -m validation.gen_synthetic` to materialize"
    df = pd.read_parquet(pq)
    expected = generate_synthetic()
    pd.testing.assert_frame_equal(df, expected)
