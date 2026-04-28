"""Numerical equivalence: crforest vs rfSRC per-feature argmax threshold.

Two splitrules covered: "logrankCR" (SURV_CR_LAU) and "logrank" (SURV_LR).
Each invariant: crforest's composite/cause-specific log-rank statistic
reaches its per-feature maximum at the same threshold rfSRC chooses
under the corresponding splitrule, within atol=1e-6.

- ``test_crforest_argmax_matches_rfsrc_fixture[splitrule]`` — CI-gated,
  reads a pinned fixture (no rpy2 needed).
- ``test_crforest_argmax_matches_live_rfsrc[splitrule-seed]`` — dev-only,
  runs rfSRC live via rpy2. Parametrized over (splitrule, seed).
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest
from validation.alignment import _rpy2_available
from validation.alignment.compare_splits import (
    crforest_candidate_stats,
    rfsrc_per_feature_best_split,
    toy_input,
)

_FIXTURE_DIR = Path(__file__).resolve().parent / "fixtures"
_THRESHOLD_ATOL = 1e-6

# (splitrule, cause, fixture_filename)
_CASES = [
    ("logrankCR", None, "rfsrc_toy_best_thresholds.parquet"),
    ("logrank", 1, "rfsrc_toy_best_thresholds_logrank.parquet"),
]


def _crforest_argmax_thresholds(data: dict, splitrule: str, cause: int | None) -> pd.DataFrame:
    df = crforest_candidate_stats(
        data["X"],
        data["time"],
        data["event"],
        data["n_causes"],
        splitrule=splitrule,
        cause=cause if cause is not None else 1,
    )
    best = (
        df.sort_values(["stat", "threshold"], ascending=[False, True])
        .groupby("feature", as_index=False)
        .first()
        .rename(columns={"threshold": "best_threshold"})
    )
    return best[["feature", "best_threshold"]].reset_index(drop=True)


def _threshold_mismatches(cr: pd.DataFrame, rf: pd.DataFrame, atol: float) -> pd.DataFrame:
    merged = cr.merge(rf, on="feature", suffixes=("_cr", "_rf"))
    merged["abs_diff"] = (merged["best_threshold_cr"] - merged["best_threshold_rf"]).abs()
    return merged[merged["abs_diff"] > atol].reset_index(drop=True)


@pytest.mark.parametrize(
    "splitrule, cause, fixture_name",
    _CASES,
    ids=lambda c: c if isinstance(c, str) else str(c),
)
def test_crforest_argmax_matches_rfsrc_fixture(splitrule, cause, fixture_name):
    fixture = _FIXTURE_DIR / fixture_name
    rfsrc = pd.read_parquet(fixture)
    mismatches = []
    for seed in sorted(rfsrc["seed"].unique()):
        data = toy_input(seed=int(seed), n=30, n_features=3, n_causes=2)
        cr = _crforest_argmax_thresholds(data, splitrule=splitrule, cause=cause)
        rf = rfsrc[rfsrc["seed"] == seed][["feature", "best_threshold"]].reset_index(drop=True)
        diff = _threshold_mismatches(cr, rf, atol=_THRESHOLD_ATOL)
        if len(diff) > 0:
            diff["seed"] = seed
            mismatches.append(diff)
    assert not mismatches, (
        f"splitrule={splitrule}: {len(mismatches)} seed(s) disagree with rfSRC fixture:\n"
        + "\n".join(str(m) for m in mismatches)
    )


@pytest.mark.skipif(not _rpy2_available(), reason="rpy2 not installed")
@pytest.mark.parametrize(
    "splitrule, cause, fixture_name",
    _CASES,
    ids=lambda c: c if isinstance(c, str) else str(c),
)
@pytest.mark.parametrize("seed", range(5))
def test_crforest_argmax_matches_live_rfsrc(splitrule, cause, fixture_name, seed):
    del fixture_name  # unused in live path
    data = toy_input(seed=seed, n=30, n_features=3, n_causes=2)
    cr = _crforest_argmax_thresholds(data, splitrule=splitrule, cause=cause)
    rf = rfsrc_per_feature_best_split(
        data["X"],
        data["time"],
        data["event"],
        splitrule=splitrule,
        cause=cause,
    )[["feature", "best_threshold"]]
    diff = _threshold_mismatches(cr, rf, atol=_THRESHOLD_ATOL)
    assert len(diff) == 0, f"splitrule={splitrule} seed={seed} mismatch:\n{diff}"
