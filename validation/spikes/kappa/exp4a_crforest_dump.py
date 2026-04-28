"""κ.exp4a — crforest multi-seed, dump risk per seed.

Loops 5 seeds, fits CompetingRiskForest, dumps test-set risk vectors per seed
to /tmp/chf_2012_crforest_risks_multiseed.parquet (mirrors what exp4_multiseed_rfsrc.R
dumps for rfSRC). exp4b aggregates both later. Run while rfSRC bench is in
flight on win — the two are independent.

Run: uv run python -u validation/spikes/kappa/exp4a_crforest_dump.py
"""

from __future__ import annotations

import time as _time
from pathlib import Path

import numpy as np
import pandas as pd

from crforest import CompetingRiskForest

CLEAN_PARQUET = Path("/tmp/chf_2012_clean.parquet")
TRAIN_IDX = Path("/tmp/chf_2012_train_idx.txt")
TEST_IDX = Path("/tmp/chf_2012_test_idx.txt")
OUT_RISKS = Path("/tmp/chf_2012_crforest_risks_multiseed.parquet")
OUT_WALLS = Path("/tmp/chf_2012_crforest_walls_multiseed.parquet")
SEEDS = [42, 43, 44, 45, 46]


def main() -> None:
    df = pd.read_parquet(CLEAN_PARQUET)
    train_idx = np.loadtxt(TRAIN_IDX, dtype=np.int64)
    test_idx = np.loadtxt(TEST_IDX, dtype=np.int64)
    feature_cols = [c for c in df.columns if c not in ("time", "status")]
    X = df[feature_cols].to_numpy(dtype=np.float64)
    t = df["time"].to_numpy(dtype=np.float64)
    e = df["status"].to_numpy(dtype=np.int64)
    X_tr, t_tr, e_tr = X[train_idx], t[train_idx], e[train_idx]
    X_te = X[test_idx]
    print(
        f"[load] train n={len(train_idx):,}, test n={len(test_idx):,}, p={len(feature_cols)}",
        flush=True,
    )

    rows_risk = []
    rows_wall = []
    for seed in SEEDS:
        print(f"\n[crforest] seed={seed} fitting...", flush=True)
        f = CompetingRiskForest(n_estimators=100, n_jobs=-1, random_state=seed)
        t0 = _time.perf_counter()
        f.fit(X_tr, t_tr, e_tr)
        wall = _time.perf_counter() - t0
        risk1 = f.predict_risk(X_te, cause=1)
        risk2 = f.predict_risk(X_te, cause=2)
        print(f"  wall={wall:.2f}s", flush=True)
        for j, idx in enumerate(test_idx):
            rows_risk.append(
                {
                    "seed": seed,
                    "test_idx": int(idx),
                    "risk1": float(risk1[j]),
                    "risk2": float(risk2[j]),
                }
            )
        rows_wall.append({"seed": seed, "fit_wall": wall})

    pd.DataFrame(rows_risk).to_parquet(OUT_RISKS)
    pd.DataFrame(rows_wall).to_parquet(OUT_WALLS)
    print(f"\n[dump] {OUT_RISKS}  ({len(rows_risk):,} rows)", flush=True)
    print(f"[dump] {OUT_WALLS}", flush=True)
    walls = pd.DataFrame(rows_wall)
    print(
        f"[summary] crforest fit_wall: mean={walls['fit_wall'].mean():.2f}s  "
        f"std={walls['fit_wall'].std(ddof=1):.2f}s  "
        f"range=[{walls['fit_wall'].min():.2f}, {walls['fit_wall'].max():.2f}]",
        flush=True,
    )


if __name__ == "__main__":
    main()
