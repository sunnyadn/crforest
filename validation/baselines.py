"""Read pre-computed rfSRC cause-1 risk baselines per (dataset, seed)."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from comprisk import CompetingRiskForest
from validation.datasets import load as load_dataset
from validation.splits import load as load_splits

_BASELINES_DIR = Path(__file__).resolve().parent / "baselines"


def load_baseline(
    dataset: str,
    seed: int,
    splitrule: str = "logrankCR",
    cause: int | None = None,
    nsplit: int | None = None,
) -> np.ndarray:
    """Return rfSRC cause-1 risk on the test fold for ``(dataset, seed)``.

    Risks are ordered by ``sample_id`` ascending — the same order
    ``validation.splits.load`` returns for the test index.

    Parameters
    ----------
    dataset:
        Dataset name (e.g. ``"pbc"``).
    seed:
        Integer seed index.
    splitrule:
        Which rfSRC splitrule baseline to load.

        * ``"logrankCR"`` — reads ``{dataset}.parquet`` (production,
          rfSRC default nsplit=10) or ``{dataset}_ns0.parquet`` when
          ``nsplit=0`` (diagnostic, exhaustive).
        * ``"logrank"`` — reads
          ``{dataset}_logrank_cause{cause}.parquet`` (production) or
          ``{dataset}_logrank_cause{cause}_ns0.parquet`` (diagnostic).
    cause:
        Cause index; required when ``splitrule="logrank"``.
    nsplit:
        ``None`` or ``10`` selects the production baseline (rfSRC default
        nsplit=10). ``0`` selects the diagnostic baseline (exhaustive
        search). Other values raise ValueError.
    """
    if nsplit not in (None, 0, 10):
        raise ValueError(f"nsplit must be None, 0, or 10; got {nsplit!r}")
    suffix = "_ns0" if nsplit == 0 else ""
    if splitrule == "logrankCR":
        pq = _BASELINES_DIR / f"{dataset}{suffix}.parquet"
    elif splitrule == "logrank":
        if cause is None:
            raise ValueError("cause must be provided when splitrule='logrank'")
        pq = _BASELINES_DIR / f"{dataset}_logrank_cause{cause}{suffix}.parquet"
    else:
        raise ValueError(f"splitrule must be 'logrankCR' or 'logrank'; got {splitrule!r}")
    if not pq.exists():
        raise FileNotFoundError(f"{pq} missing; run `Rscript validation/gen_rfsrc_baselines.R`")
    df = pd.read_parquet(pq)
    sub = df[df["seed"] == seed].sort_values("sample_id")
    return sub["risk_cause_1"].to_numpy(dtype=np.float64)


def make_forest(config, seed: int, mode: str = "default") -> CompetingRiskForest:
    """Construct a CompetingRiskForest with harness hyperparameters from config."""
    return CompetingRiskForest(
        n_estimators=config.n_estimators,
        min_samples_leaf=config.min_samples_leaf,
        min_samples_split=config.min_samples_split,
        max_features=config.max_features,
        max_depth=config.max_depth,
        bootstrap=config.bootstrap,
        random_state=seed,
        mode=mode,
        splitrule=config.splitrule,
        cause=config.cause,
        cause_weights=list(config.cause_weights) if config.cause_weights is not None else None,
        nsplit=config.nsplit,
        split_ntime=config.split_ntime,
    )


def fit_reference_baseline(dataset: str, seed: int, config) -> np.ndarray:
    """Fit a reference-mode comprisk and return test-fold risk array.

    Shape and dtype match ``load_baseline``'s return. ``config`` is a
    ``HarnessConfig`` carrying hyperparameters (n_estimators, etc).
    """
    X, time, event = load_dataset(dataset)
    train_idx, test_idx = load_splits(dataset)[seed]
    forest = make_forest(config, seed, mode="reference").fit(
        X[train_idx], time[train_idx], event[train_idx]
    )
    return forest.predict_risk(X[test_idx], cause=config.cause)
