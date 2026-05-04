"""Regression guard: ``split_ntime=None`` tree-building output must be stable.

The ``ANCHOR_*`` constants below were last re-anchored when Task 5 of
Plan 1 (C-CPU flat-tree) wired the new flat-tree builder as the
default-mode path. FlatTree stores dense float64 CIF tables (no sparse
rep yet), so pickle size increased substantially vs the HistTreeNode
baseline. History:

  - pre-sprint anchor (Task 0.2): ``b2b65716..`` / 1285307 bytes
  - post-ε plumbing:              ``71e87f19..`` / 1285322 bytes  (+15, split_ntime)
  - post-Phase 1:                 ``4ed4b112..`` / 1285335 bytes  (+13, rng_mode)
  - equivalence preset:           ``33bf973c..`` / 1285400 bytes  (+65, equivalence + inbag_ + _eff_)
  - oob VIMP storage:             ``5e600f25..`` / 1477560 bytes  (+192k, _X_train_oob_ + _y_train_oob_)
  - Plan 1 Task 5 (FlatTree):     ``9190560a..`` / 8766783 bytes  (+7.3 MB, dense CIF tables in FlatTree)
  - Plan 1.5 (predict_chf lazy):  ``cfb10138..`` / 14801489 bytes (+5.7 MB, raw uint32 counts persisted on FlatTree for lazy Nelson-Aalen)
  - Plan 2 Task 6 (device arg):   ``08cb8cd4..`` / 14801532 bytes (+43, ``device`` ctor attr + ``_effective_device_`` post-fit)
  - Plan 2 Task 9d.5 (pin cpu):   ``e7930718..`` / 14801527 bytes (-5, ctor pinned ``device='cpu'`` so the test is hardware-independent — ``device='auto'`` resolves to cuda when cupy is installed and produces a different pickle by design)
  - Surv y_train_oob field order: ``70efbe29..`` / 14801527 bytes (0 byte delta, struct layout swap — ``_y_train_oob_`` field order changed from ``[("time", float64), ("event", int64)]`` to sksurv-canonical ``[("event", int64), ("time", float64)]`` after the ``Surv.from_arrays`` simplification)
  - SUN-42 time-grid fix:         ``ff50915f..`` / 14801551 bytes (+24, ``_time_grid_max_eff_`` int added as fitted attr by ``_resolve_equivalence``; tree-building unchanged on default path)

Future changes that affect ``split_ntime=None`` tree-building behavior
will drift this digest and flag for investigation.
"""

from __future__ import annotations

import hashlib
import pickle

import numpy as np

from crforest import CompetingRiskForest

ANCHOR_SHA256 = "ff50915f2e2dc9e6b80efbe93dc41539596ea36bdd1ccfb98322c8fc5f948d2b"
ANCHOR_PICKLE_BYTES = 14801551


def test_split_ntime_none_matches_anchor_digest() -> None:
    rng = np.random.default_rng(0)
    n, p = 2000, 10
    X = rng.normal(size=(n, p))
    time = rng.exponential(1.0, size=n) + 0.1
    event = rng.integers(0, 3, size=n)
    # Pin cpu: defensive anchor against a future v1.1 auto→cuda flip
    # (GPU produces byte-different pickles by design — DFS vs BFS node order).
    forest = CompetingRiskForest(
        n_estimators=50,
        max_depth=6,
        random_state=0,
        n_jobs=1,
        mode="default",
        split_ntime=None,
        device="cpu",
    ).fit(X, time, event)
    blob = pickle.dumps(forest, protocol=pickle.HIGHEST_PROTOCOL)
    digest = hashlib.sha256(blob).hexdigest()
    assert len(blob) == ANCHOR_PICKLE_BYTES, (
        f"pickle byte count drift: got {len(blob)}, expected {ANCHOR_PICKLE_BYTES}. "
        "Investigate — tree-building behavior changed for split_ntime=None."
    )
    assert digest == ANCHOR_SHA256, (
        f"pickle digest drift: got {digest}, expected {ANCHOR_SHA256}. "
        "split_ntime=None behavior must be stable."
    )
