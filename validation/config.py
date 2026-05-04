"""Shared configuration for the validation harness."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class HarnessConfig:
    """Locked-down comparison config for paired comprisk/rfSRC runs.

    ``min_samples_leaf=1`` matches rfSRC's absence of a per-child terminal-node
    constraint. rfSRC's ``nodesize=15`` only thresholds pre-split node size
    (approx. ``min_samples_split``); it does NOT require each child to have
    >=15 samples. Setting ``min_samples_leaf=15`` on the comprisk side (as an
    earlier version of this harness did) made comprisk's trees systematically
    shallower than rfSRC's on datasets where rfSRC picks highly unbalanced
    splits, producing a spurious +0.006 C-index residual on follic in P2.5.
    See ``docs/superpowers/specs/2026-04-18-p2.6-cif-localization.md``.
    """

    n_estimators: int = 500
    min_samples_leaf: int = 1
    min_samples_split: int = 30
    max_features: str = "sqrt"
    max_depth: int | None = None
    bootstrap: bool = True
    test_frac: float = 0.2
    cause: int = 1
    n_seeds: int = 20
    splitrule: str = "logrankCR"
    cause_weights: tuple[float, ...] | None = None
    nsplit: int = 10
    split_ntime: int | None = None


DEFAULT = HarnessConfig()
DATASETS = ["pbc", "follic", "hd", "synthetic"]
