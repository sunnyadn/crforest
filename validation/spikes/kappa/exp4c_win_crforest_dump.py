"""κ.exp4c — crforest multi-seed × multi-device dump on the win canonical machine.

Loops 5 seeds × {device='cpu', device='cuda'} for canonical paper bench
on the win box (i7-14700K, 28 threads, RTX 5070 Ti, 24 GB WSL). Dumps
risk vectors per (seed, device) combination so exp4b-style aggregation
can score everything with the same metric.

Companion to exp4_multiseed_rfsrc.R (rfSRC same machine, rf.cores=16).
Together these give same-machine wall-time comparison vs cross-machine
(Mac numbers we already have).

Run: ssh win 'cd ~/crforest && uv run --extra gpu --extra dev python -u
                validation/spikes/kappa/exp4c_win_crforest_dump.py'
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
OUT_RISKS = Path("/tmp/chf_2012_crforest_win_risks.parquet")
OUT_WALLS = Path("/tmp/chf_2012_crforest_win_walls.parquet")
SEEDS = [42, 43, 44, 45, 46]
DEVICES = ["cpu", "cuda"]


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
    for device in DEVICES:
        # cuda backend forces n_jobs=1 (single host driver); cpu uses all cores
        n_jobs = 1 if device == "cuda" else -1
        for seed in SEEDS:
            print(f"\n[crforest] device={device} seed={seed} fitting...", flush=True)
            f = CompetingRiskForest(
                n_estimators=100,
                n_jobs=n_jobs,
                random_state=seed,
                device=device,
            )
            t0 = _time.perf_counter()
            f.fit(X_tr, t_tr, e_tr)
            wall = _time.perf_counter() - t0
            risk1 = f.predict_risk(X_te, cause=1)
            risk2 = f.predict_risk(X_te, cause=2)
            risk1_chf = f.predict_risk(X_te, cause=1, kind="integrated_chf")
            risk2_chf = f.predict_risk(X_te, cause=2, kind="integrated_chf")
            print(
                f"  wall={wall:.2f}s effective_device={f._effective_device_}",
                flush=True,
            )
            for j, idx in enumerate(test_idx):
                rows_risk.append(
                    {
                        "device": device,
                        "seed": seed,
                        "test_idx": int(idx),
                        "risk1_cif": float(risk1[j]),
                        "risk2_cif": float(risk2[j]),
                        "risk1_chf": float(risk1_chf[j]),
                        "risk2_chf": float(risk2_chf[j]),
                    }
                )
            rows_wall.append(
                {
                    "device": device,
                    "seed": seed,
                    "fit_wall": wall,
                    "effective_device": f._effective_device_,
                }
            )

    pd.DataFrame(rows_risk).to_parquet(OUT_RISKS)
    pd.DataFrame(rows_wall).to_parquet(OUT_WALLS)
    print(f"\n[dump] {OUT_RISKS} ({len(rows_risk):,} rows)", flush=True)
    print(f"[dump] {OUT_WALLS}", flush=True)
    walls = pd.DataFrame(rows_wall)
    print("\n[summary] crforest win wall-time:")
    for device in DEVICES:
        sub = walls[walls["device"] == device]
        print(
            f"  device={device}: mean={sub['fit_wall'].mean():.2f}s  "
            f"std={sub['fit_wall'].std(ddof=1):.2f}s  "
            f"range=[{sub['fit_wall'].min():.2f}, {sub['fit_wall'].max():.2f}]",
            flush=True,
        )


if __name__ == "__main__":
    main()
