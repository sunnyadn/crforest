"""κ.exp9 — does C-index plateau on real CHF as ntree grows past 100?

Existing canonical bench (κ.exp4d) only ran ntree=100. This sweeps
ntree ∈ {100, 200, 500, 1000} × 3 seeds on real CHF and scores all four
C-indices on the test split: HF Harrell, Death Harrell, HF Uno, Death Uno.

Wall budget: per seed ≈ (22 + 45 + 113 + 225)s = 405s on win i7+5070Ti.
3 seeds ≈ 20 min total. CPU only (GPU adds 15% wall on this shape and
isn't relevant to the C-index question).

Output: /tmp/chf_2012_ntree_cindex.parquet

Run: ssh win 'export PATH=$HOME/.local/bin:$PATH && cd ~/crforest && \\
       PYTHONUNBUFFERED=1 uv run --extra dev \\
       python -u validation/spikes/kappa/exp9_ntree_cindex_curve.py \\
       2>&1 | tee /tmp/exp9_ntree_cindex_curve.log'
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

OUT = Path("/tmp/chf_2012_ntree_cindex.parquet")
NTREES = [100, 200, 500, 1000]
SEEDS = [42, 43, 44]


def main() -> None:
    X_tr, t_tr, e_tr, X_te, t_te, e_te, p = load_chf(with_test=True)
    print(f"[load] train n={len(X_tr):,} test n={len(X_te):,} p={p}", flush=True)

    w_te = compute_uno_weights(t_te, e_te)
    print(
        f"[uno-weights] kept={int((w_te > np.finfo(float).eps).sum()):,} of {len(w_te):,}",
        flush=True,
    )

    rows = []
    for seed in SEEDS:
        for ntree in NTREES:
            print(f"\n[fit] ntree={ntree} seed={seed}", flush=True)
            f = CompetingRiskForest(
                n_estimators=ntree,
                n_jobs=-1,
                random_state=seed,
                device="cpu",
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
            rows.append({"seed": seed, "ntree": ntree, "wall": wall, **cindex})

    out = pd.DataFrame(rows)
    out.to_parquet(OUT)
    print(f"\n[dump] {OUT} ({len(out)} rows)", flush=True)

    print("\n=== Mean ± std across 3 seeds, by ntree ===", flush=True)
    metrics = ["hf_harrell", "death_harrell", "hf_uno", "death_uno", "wall"]
    summary = (
        out.groupby("ntree")[metrics]
        .agg(["mean", "std"])
        .round({(m, "mean"): 4 for m in metrics} | {(m, "std"): 4 for m in metrics})
    )
    print(summary.to_string(), flush=True)

    print("\n=== Δ vs ntree=100 baseline (mean of 3 seeds) ===", flush=True)
    means = out.groupby("ntree")[metrics[:-1]].mean()
    delta = means.subtract(means.loc[100])
    print(delta.round(4).to_string(), flush=True)


if __name__ == "__main__":
    main()
