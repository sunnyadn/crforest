"""Paired-seed runner: fits comprisk and loads rfSRC baseline, scores both."""

from __future__ import annotations

from dataclasses import dataclass

from joblib import Parallel, delayed

from comprisk import concordance_index_cr
from validation.baselines import fit_reference_baseline, load_baseline, make_forest
from validation.config import HarnessConfig
from validation.datasets import load as load_dataset
from validation.splits import load as load_splits


@dataclass
class SeedResult:
    dataset: str
    seed: int
    c_comprisk: float
    c_rfsrc: float
    delta_c: float


def _run_one(dataset: str, seed: int, config: HarnessConfig, compare: str = "rfsrc") -> SeedResult:
    X, time, event = load_dataset(dataset)
    splits = load_splits(dataset)
    train_idx, test_idx = splits[seed]

    forest = make_forest(config, seed).fit(X[train_idx], time[train_idx], event[train_idx])
    risk_cr = forest.predict_risk(X[test_idx], cause=config.cause)
    c_cr = concordance_index_cr(event[test_idx], time[test_idx], risk_cr, cause=config.cause)

    if compare == "rfsrc":
        risk_base = load_baseline(
            dataset,
            seed,
            splitrule=config.splitrule,
            cause=config.cause,
            nsplit=config.nsplit,
        )
    elif compare == "reference":
        risk_base = fit_reference_baseline(dataset, seed, config)
    else:
        raise ValueError(f"compare must be 'rfsrc' or 'reference'; got {compare!r}")
    c_base = concordance_index_cr(event[test_idx], time[test_idx], risk_base, cause=config.cause)

    return SeedResult(
        dataset=dataset,
        seed=seed,
        c_comprisk=float(c_cr),
        c_rfsrc=float(c_base),  # field name retained for schema compatibility
        delta_c=float(c_cr - c_base),
    )


def run_dataset(
    dataset: str,
    seeds: list[int],
    config: HarnessConfig,
    n_jobs: int = -1,
    compare: str = "rfsrc",
) -> list[SeedResult]:
    """Fit + score comprisk; compare against baseline; return results."""
    results = Parallel(n_jobs=n_jobs)(
        delayed(_run_one)(dataset, seed, config, compare) for seed in seeds
    )
    return list(results)
