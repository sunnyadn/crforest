"""η spike — Experiment 2: ablation ladder at n=100k, ntree=100.

Trimmed scope after exp1+exp4 made the picture clear:
  - knob ablation: just nsplit + kitchen-sink (cheaper iteration)
  - n_jobs scaling sweep (the parallel-efficiency picture matters most)

Reports wall_s + speedup_vs_baseline. Output to logs/exp2_ablation.log.
"""

from __future__ import annotations

import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from _dgp import load

from comprisk import CompetingRiskForest

LOG_DIR = Path(__file__).parent / "logs"
LOG_DIR.mkdir(exist_ok=True)
LOG_PATH = LOG_DIR / "exp2_ablation.log"

N = 100_000
NTREE = 100
SEED = 0


def _print(msg: str, fp) -> None:
    print(msg, flush=True)
    fp.write(msg + "\n")
    fp.flush()


def _fit(X, t, e, **kwargs) -> float:
    cfg = dict(
        n_estimators=NTREE,
        min_samples_leaf=15,
        max_features=8,
        nsplit=10,
        splitrule="logrankCR",
        split_ntime=50,
        random_state=SEED,
        n_jobs=-1,
    )
    cfg.update(kwargs)
    forest = CompetingRiskForest(**cfg)
    t0 = time.perf_counter()
    forest.fit(X, t, e)
    return time.perf_counter() - t0


def main() -> None:
    fp = open(LOG_PATH, "w")
    _print(f"[exp2] dataset weibull n={N} seed={SEED} p=60 ntree={NTREE}", fp)

    X, t, e = load(N, SEED)
    print("[exp2] warming njit caches…", flush=True)
    CompetingRiskForest(
        n_estimators=4,
        min_samples_leaf=15,
        max_features=8,
        nsplit=10,
        splitrule="logrankCR",
        split_ntime=50,
        random_state=0,
        n_jobs=1,
    ).fit(X[:2000], t[:2000], e[:2000])

    cpu = os.cpu_count() or 1
    _print(f"[exp2] cpu_count = {cpu}", fp)

    # ────────────── A) n_jobs scaling sweep at default knobs ──────────────
    _print("\n=== A) n_jobs scaling sweep (default knobs) ===", fp)
    walls_by_jobs = {}
    for nj in [1, 2, 5, 10]:
        wall = _fit(X, t, e, n_jobs=nj)
        walls_by_jobs[nj] = wall
        speedup_vs_serial = walls_by_jobs[1] / wall if nj > 1 else 1.0
        ideal = walls_by_jobs[1] / nj
        eff = ideal / wall * 100
        _print(
            f"  n_jobs={nj:3d}  wall={wall:6.1f}s  speedup={speedup_vs_serial:4.2f}x  "
            f"parallel_eff={eff:4.0f}%",
            fp,
        )

    base_wall = walls_by_jobs[10]
    _print(f"\n[exp2] baseline at n_jobs=10 = {base_wall:.1f}s\n", fp)

    # ────────────── B) Knob ablation @ n_jobs=10 ──────────────
    _print("=== B) knob ablation (at n_jobs=10) ===", fp)
    runs = [
        ("baseline (nsplit=10)", {}),
        ("nsplit=1", {"nsplit": 1}),
        ("nsplit=0 (all bins)", {"nsplit": 0}),
        ("min_samples_leaf=50", {"min_samples_leaf": 50}),
        (
            "kitchen-sink approx",
            {"nsplit": 1, "split_ntime": 20, "min_samples_leaf": 50, "max_depth": 8},
        ),
    ]
    for label, kwargs in runs:
        wall = _fit(X, t, e, **kwargs)
        ratio = base_wall / wall
        _print(f"  {label:30s} wall={wall:6.1f}s   speedup_vs_baseline={ratio:5.2f}x", fp)

    fp.close()


if __name__ == "__main__":
    main()
