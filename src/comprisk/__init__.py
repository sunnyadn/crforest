"""comprisk: Python toolkit for competing risks (forest today; regression in v0.4)."""

from comprisk._sklearn_compat import Surv
from comprisk.evaluation import ScoreResult, score_cr
from comprisk.forest import CompetingRiskForest
from comprisk.metrics import (
    compute_uno_weights,
    concordance_index_cr,
    concordance_index_uno_cr,
)

__version__ = "0.3.1"

__all__ = [
    "CompetingRiskForest",
    "ScoreResult",
    "Surv",
    "__version__",
    "compute_uno_weights",
    "concordance_index_cr",
    "concordance_index_uno_cr",
    "score_cr",
]
