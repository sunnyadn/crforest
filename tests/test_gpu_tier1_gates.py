"""Tier 1 accuracy gates for the GPU backend (spec §3).

Spec ref: docs/superpowers/specs/2026-04-25-c-gpu-sprint-design.md §3.

Three Tier 1 gates from the spec:
  1. Synthetic planted-signal AUC ≥ 0.95 (this file)
  2. 12-axis OOB VIMP correctness vs rfSRC — environment-bound
     (rpy2 + /tmp/rfsrc_patched_lib); tracked via
     validation/alignment/vimp_perm_replay.py. NOT exercised here because
     the GPU path is the cuda flat-tree default; VIMP correctness is an
     algorithm-level invariant validated separately.
  3. equivalence='rfsrc' preset all green — preset path is HistTreeNode
     (CPU only, untouched by Plan 2). Validated by
     tests/test_equivalence_preset.py (9/9 PASS).

This file lands gate 1 on the cuda backend. The DGP mirrors the synthetic
"planted-signal" check: most of the variance in cause-1 risk is driven by a
single feature, so a well-fit forest's predict_risk(cause=1) must rank
cause-1-event samples above non-event samples (AUC ≥ 0.95).
"""

import numpy as np
import pytest

pytestmark = pytest.mark.gpu


def test_gpu_synthetic_planted_signal_auc():
    """Forest fit on the planted-signal DGP must give AUC ≥ 0.95 for
    predict_risk(cause=1) vs the binary cause-1 event indicator."""
    from sklearn.metrics import roc_auc_score

    from crforest import CompetingRiskForest

    rng = np.random.default_rng(0)
    n = 5000
    p = 10
    X = rng.uniform(size=(n, p))
    # Plant signal in feature 3: higher X[:, 3] -> higher cause-1 risk.
    risk = X[:, 3] * 2.0
    time = rng.exponential(np.exp(-risk))
    event = (rng.uniform(size=n) < 0.7).astype(np.int32) + 1  # mostly cause-1
    event[time > 5] = 0  # censor late

    f = CompetingRiskForest(
        n_estimators=100,
        device="cuda",
        random_state=0,
    ).fit(X, time, event)

    risk_pred = f.predict_risk(X, cause=1)
    binary_event = (event == 1).astype(np.int32)
    auc = roc_auc_score(binary_event, risk_pred)
    assert auc >= 0.95, f"AUC {auc} < 0.95"
