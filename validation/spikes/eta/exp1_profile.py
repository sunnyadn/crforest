"""η spike — Experiment 1: re-profile current main at n=100k, ntree=100.

Goal: identify post-ε top hot lines/functions for the default-mode fit
in the absolute-time regime that matters to users (n=100k).

Outputs to logs/exp1_profile.log:
  - wall time (warm)
  - cProfile top 30 by cumulative + tottime
"""

from __future__ import annotations

import cProfile
import io
import pstats
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from _dgp import load

from comprisk import CompetingRiskForest

LOG_DIR = Path(__file__).parent / "logs"
LOG_DIR.mkdir(exist_ok=True)
LOG_PATH = LOG_DIR / "exp1_profile.log"


# Match zeta's CompetingRiskForest config so numbers extrapolate.
N = 100_000
NTREE = 100  # 1/5 of zeta's 500 to fit within spike budget; α≈1, scales linearly
SEED = 0


def _fit() -> CompetingRiskForest:
    X, t, e = load(N, SEED)
    forest = CompetingRiskForest(
        n_estimators=NTREE,
        min_samples_leaf=15,
        max_features=8,
        nsplit=10,
        splitrule="logrankCR",
        split_ntime=50,
        random_state=SEED,
        n_jobs=-1,
    )
    forest.fit(X, t, e)
    return forest


def _print(msg: str, fp) -> None:
    print(msg, flush=True)
    fp.write(msg + "\n")
    fp.flush()


def main() -> None:
    fp = open(LOG_PATH, "w")
    _print(f"[exp1] dataset weibull n={N} seed={SEED} p=60", fp)
    _print(f"[exp1] forest ntree={NTREE} mtry=8 nsplit=10 split_ntime=50", fp)

    # Warm njit caches with a tiny fit.
    _print("[exp1] warming njit caches (n=2000 ntree=4)…", fp)
    X_w, t_w, e_w = load(N, SEED)
    CompetingRiskForest(
        n_estimators=4,
        min_samples_leaf=15,
        max_features=8,
        nsplit=10,
        splitrule="logrankCR",
        split_ntime=50,
        random_state=0,
        n_jobs=1,
    ).fit(X_w[:2000], t_w[:2000], e_w[:2000])
    del X_w, t_w, e_w

    # Wall time pass (untimed for cProfile overhead).
    _print("[exp1] wall-time pass…", fp)
    t0 = time.perf_counter()
    _fit()
    wall_clean = time.perf_counter() - t0
    _print(f"[exp1] WALL_CLEAN = {wall_clean:.2f}s ({NTREE} trees)", fp)

    # cProfile pass.
    _print("[exp1] cProfile pass…", fp)
    pr = cProfile.Profile()
    pr.enable()
    t0 = time.perf_counter()
    _fit()
    wall_prof = time.perf_counter() - t0
    pr.disable()
    _print(f"[exp1] WALL_PROFILE = {wall_prof:.2f}s (overhead {wall_prof - wall_clean:+.2f}s)", fp)

    s = io.StringIO()
    pstats.Stats(pr, stream=s).strip_dirs().sort_stats("cumulative").print_stats(30)
    _print("\n=== cumulative ===\n" + s.getvalue(), fp)

    s2 = io.StringIO()
    pstats.Stats(pr, stream=s2).strip_dirs().sort_stats("tottime").print_stats(30)
    _print("\n=== tottime ===\n" + s2.getvalue(), fp)

    # Extrapolate to ntree=500 (zeta's headline config).
    _print(
        f"[exp1] EXTRAPOLATED @ ntree=500: {wall_clean * 5:.0f}s (zeta ntree=500 mean was 1845s)",
        fp,
    )
    fp.close()


if __name__ == "__main__":
    main()
