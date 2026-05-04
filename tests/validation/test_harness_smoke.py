"""End-to-end smoke test; skips when baselines aren't present.

Marked slow so it doesn't run in the default pytest sweep.
Opt in via: ``pytest -m slow tests/validation/test_harness_smoke.py``.
"""

from pathlib import Path

import pytest
from validation.config import HarnessConfig
from validation.runner import run_dataset

VALIDATION_DIR = Path(__file__).resolve().parents[2] / "validation"
PBC_BASELINE = VALIDATION_DIR / "baselines" / "pbc.parquet"


@pytest.mark.slow
def test_pbc_smoke_1seed():
    if not PBC_BASELINE.exists():
        pytest.skip("rfSRC baselines not generated yet; run validation/gen_rfsrc_baselines.R")
    config = HarnessConfig(n_estimators=50)  # keep fast
    results = run_dataset("pbc", seeds=[0], config=config, n_jobs=1)
    assert len(results) == 1
    r = results[0]
    assert 0.0 <= r.c_comprisk <= 1.0
    assert 0.0 <= r.c_rfsrc <= 1.0
    assert abs(r.delta_c) <= 1.0
