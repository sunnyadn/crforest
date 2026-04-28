"""Generate pinned rfSRC fixtures used by the CI-gating equivalence tests.

Produces two fixture files — one per splitrule:
- ``tests/fixtures/rfsrc_toy_best_thresholds.parquet``         — logrankCR
- ``tests/fixtures/rfsrc_toy_best_thresholds_logrank.parquet`` — logrank, cause=1

Run manually when the fixtures need to be regenerated:
    uv run python -m validation.alignment.gen_fixtures

Pinned versions at the time of last regeneration:
- randomForestSRC: 3.6.1
- R:               4.5.2
- rpy2:            3.6.7

Re-runnable and deterministic (seeds fixed).

Fixture schema: one row per (seed, feature) with rfSRC's chosen split
threshold for a depth-1, bootstrap="none", nsplit=0 tree fit on
toy_input(seed, n=30, n_features=3, n_causes=2).
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from validation.alignment.compare_splits import rfsrc_per_feature_best_split, toy_input

FIXTURE_PATH_CR = (
    Path(__file__).resolve().parents[2] / "tests" / "fixtures" / "rfsrc_toy_best_thresholds.parquet"
)
FIXTURE_PATH_LR = (
    Path(__file__).resolve().parents[2]
    / "tests"
    / "fixtures"
    / "rfsrc_toy_best_thresholds_logrank.parquet"
)


def generate(splitrule: str, cause: int | None) -> pd.DataFrame:
    frames = []
    for seed in range(5):
        data = toy_input(seed=seed, n=30, n_features=3, n_causes=2)
        df = rfsrc_per_feature_best_split(
            data["X"],
            data["time"],
            data["event"],
            splitrule=splitrule,
            cause=cause,
        )
        df["seed"] = seed
        frames.append(df)
    return pd.concat(frames, ignore_index=True)[["seed", "feature", "best_threshold"]]


def main() -> None:
    for target, splitrule, cause in [
        (FIXTURE_PATH_CR, "logrankCR", None),
        (FIXTURE_PATH_LR, "logrank", 1),
    ]:
        target.parent.mkdir(parents=True, exist_ok=True)
        df = generate(splitrule, cause)
        df.to_parquet(target, index=False)
        print(f"Wrote {target} ({len(df)} rows)")


if __name__ == "__main__":
    main()
