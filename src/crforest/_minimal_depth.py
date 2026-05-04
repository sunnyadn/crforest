"""Ishwaran-style minimal-depth variable selection for competing-risks forests.

A variable's *minimal depth* in a tree is the depth of the highest (closest
to root) split that uses that variable. Variables never split on receive a
sentinel depth of ``D_T + 1`` where ``D_T`` is the tree's maximum node depth.
Smaller mean minimal depth across the forest indicates a more important
variable.

The selection threshold is the per-forest mean of the *expected* minimal
depth under the null hypothesis of no association — derived analytically
from each tree's depth structure assuming uniform random feature selection
at every split (Ishwaran et al. 2010, JASA, eq. 4.1).

References
----------
Ishwaran, H., Kogalur, U.B., Gorodeski, E.Z., Minn, A.J., Lauer, M.S. (2010).
"High-dimensional variable selection for survival data."
*Journal of the American Statistical Association* 105(489): 205-217.
-- minimal depth + analytical threshold.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

__all__ = [
    "WalkResult",
    "_ishwaran_expected_md",
    "_walk_min_depth",
    "compute_minimal_depth",
]


@dataclass
class WalkResult:
    """Per-tree output of :func:`_walk_min_depth`."""

    min_depth_per_feature: np.ndarray  # (n_features,) int32, sentinel = D_T + 1
    internal_nodes_per_depth: np.ndarray  # (D_internal_max + 1,) int64
    max_depth: int  # D_T = max depth of any node in the tree


def _walk_min_depth(tree, n_features: int) -> WalkResult:
    """Per-tree minimal-depth walker. Dispatches on tree type."""
    raise NotImplementedError


def _ishwaran_expected_md(
    internal_nodes_per_depth: np.ndarray,
    max_depth_T: int,
    n_features: int,
) -> float:
    """Expected minimal depth of any single variable under the null."""
    raise NotImplementedError


def compute_minimal_depth(
    forest,
    *,
    threshold: str = "md",
    conservative: bool = False,
    return_extra: bool = False,
) -> pd.DataFrame:
    """Compute minimal-depth ranking + threshold-based selection. See API spec."""
    raise NotImplementedError
