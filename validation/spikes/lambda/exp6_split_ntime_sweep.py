"""λ.exp6 — split_ntime accuracy / speed tradeoff on real CHF.

Cheapest paper-grade optimization: split_ntime is the time-grid bucket count
used for cause-specific log-rank evaluation in find_best_split. Default = 50;
None means use all unique event times (much slower). Going 50 → 20 → 10
should give ~2-3× per-tree wall, IF C-index doesn't degrade.

Sweep:
  split_ntime ∈ {None, 50, 30, 20, 10, 5}
  × 3 seeds × ntree=200 (mid plateau per κ.exp9)
  × CPU only (CUDA correctness already gated bit-equivalent at split_ntime=50,
    the question here is purely accuracy-vs-speed on CPU)

Output: /tmp/lambda_exp6_split_ntime.parquet — per (split_ntime, seed) the
HF/Death Harrell + Uno + wall.

Decision: if split_ntime=20 keeps Δ HF Harrell within seed-noise (±0.0005),
ship it as the new default. Free 2-3× speedup, no algorithm change.

Run: ssh win 'export PATH=$HOME/.local/bin:$PATH && cd ~/crforest && \\
       PYTHONUNBUFFERED=1 uv run --extra dev \\
       python -u validation/spikes/lambda/exp6_split_ntime_sweep.py \\
       2>&1 | tee /tmp/lambda_exp6.log'
"""

from __future__ import annotations

import sys
import time as _time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parents[1]))

from _lambda_helpers import load_chf, score_four_cindex

from crforest import CompetingRiskForest
from crforest.metrics import compute_uno_weights

OUT = Path("/tmp/lambda_exp6_split_ntime.parquet")
SPLIT_NTIME_VALUES = [None, 50, 30, 20, 10, 5]
SEEDS = [42, 43, 44]
NTREE = 200


def main() -> None:
    X_tr, t_tr, e_tr, X_te, t_te, e_te, p = load_chf(with_test=True)
    print(f"[load] train n={len(X_tr):,} test n={len(X_te):,} p={p}", flush=True)

    w_te = compute_uno_weights(t_te, e_te)
    print(
        f"[uno-weights] kept={int((w_te > np.finfo(float).eps).sum()):,} of {len(w_te):,}",
        flush=True,
    )

    rows = []
    for split_ntime in SPLIT_NTIME_VALUES:
        for seed in SEEDS:
            print(f"\n[fit] split_ntime={split_ntime} seed={seed}", flush=True)
            f = CompetingRiskForest(
                n_estimators=NTREE,
                n_jobs=-1,
                random_state=seed,
                device="cpu",
                split_ntime=split_ntime,
            )
            t0 = _time.perf_counter()
            f.fit(X_tr, t_tr, e_tr)
            wall = _time.perf_counter() - t0
            cindex = score_four_cindex(f, X_te, t_te, e_te, w_te)
            print(
                f"  wall={wall:.1f}s  HF_H={cindex['hf_harrell']:.4f}  "
                f"Death_H={cindex['death_harrell']:.4f}  "
                f"HF_U={cindex['hf_uno']:.4f}  Death_U={cindex['death_uno']:.4f}",
                flush=True,
            )
            rows.append(
                {
                    "split_ntime": -1 if split_ntime is None else int(split_ntime),
                    "seed": seed,
                    "wall": wall,
                    **cindex,
                }
            )

    out = pd.DataFrame(rows)
    out.to_parquet(OUT)
    print(f"\n[dump] {OUT} ({len(out)} rows)", flush=True)

    print("\n=== Mean ± std across 3 seeds, by split_ntime ===\n", flush=True)
    metrics = ["hf_harrell", "death_harrell", "hf_uno", "death_uno", "wall"]
    summary = out.groupby("split_ntime")[metrics].agg(["mean", "std"]).round(4)
    print(summary.to_string(), flush=True)

    print("\n=== Δ vs split_ntime=50 baseline (mean of 3 seeds) ===\n", flush=True)
    means = out.groupby("split_ntime")[metrics[:-1]].mean()
    if 50 in means.index:
        delta = means.subtract(means.loc[50])
        print(delta.round(4).to_string(), flush=True)

    print("\n=== Speedup vs split_ntime=50 ===\n", flush=True)
    walls = out.groupby("split_ntime")["wall"].mean()
    if 50 in walls.index:
        baseline = walls.loc[50]
        speedup = baseline / walls
        print(speedup.round(3).to_string(), flush=True)


if __name__ == "__main__":
    main()
