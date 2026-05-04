"""η spike — Experiment 3: lightgbm reference baseline at n=100k.

Not apples-to-apples (binary classification vs competing-risks survival),
but answers "what's a well-engineered tree library's wall on this data
shape?". If lightgbm is e.g. 30s and comprisk is 360s, that bounds our
engineering headroom at ~10×.

Encoding: event=0 (censored) → 0; event∈{1,2} → 1. Loses CR structure
but exercises the same n×p×ntree×depth tree-fit machinery.
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

import numpy as np
from _dgp import load

LOG_DIR = Path(__file__).parent / "logs"
LOG_DIR.mkdir(exist_ok=True)
LOG_PATH = LOG_DIR / "exp3_lightgbm.log"

N = 100_000
NTREE = 100
SEED = 0


def _print(msg: str, fp) -> None:
    print(msg, flush=True)
    fp.write(msg + "\n")
    fp.flush()


def main() -> None:
    fp = open(LOG_PATH, "w")
    _print(f"[exp3] dataset weibull n={N} seed={SEED} p=60 ntree={NTREE}", fp)

    try:
        import lightgbm as lgb
    except ImportError:
        _print(
            "[exp3] lightgbm not installed — install via 'uv pip install lightgbm' to run this.", fp
        )
        fp.close()
        return

    X, _, e = load(N, SEED)
    y_binary = (e > 0).astype(np.int32)

    # Histogram-based, comparable knobs to comprisk's:
    # - num_leaves ≈ 2^max_depth; pick something reasonable for 100k
    # - bagging_fraction ≈ bootstrap fraction (1.0 in our default — we use boots WITH replacement)
    # - feature_fraction ≈ mtry/p = 8/60 ≈ 0.13
    params = {
        "objective": "binary",
        "num_leaves": 63,
        "min_data_in_leaf": 15,
        "feature_fraction": 8 / 60,
        "bagging_fraction": 1.0,
        "bagging_freq": 0,
        "verbose": -1,
        "num_threads": -1,
    }

    # Warmup
    _print("[exp3] warmup…", fp)
    train_w = lgb.Dataset(X[:2000], label=y_binary[:2000])
    lgb.train(params, train_w, num_boost_round=4)

    # Random forest mode (boosting=rf) — closer to comprisk semantics than gbdt
    rf_params = dict(params)
    rf_params["boosting"] = "rf"
    rf_params["bagging_fraction"] = 0.632  # rf-style sampling-w/o-replacement proxy
    rf_params["bagging_freq"] = 1

    _print("[exp3] lightgbm RF mode (boosting=rf)…", fp)
    train = lgb.Dataset(X, label=y_binary)
    t0 = time.perf_counter()
    lgb.train(rf_params, train, num_boost_round=NTREE)
    wall_rf = time.perf_counter() - t0
    _print(f"[exp3] lightgbm RF      wall = {wall_rf:6.2f}s ({NTREE} trees)", fp)

    _print("[exp3] lightgbm GBDT mode (boosting=gbdt, baseline only)…", fp)
    t0 = time.perf_counter()
    lgb.train(params, train, num_boost_round=NTREE)
    wall_gbdt = time.perf_counter() - t0
    _print(f"[exp3] lightgbm GBDT    wall = {wall_gbdt:6.2f}s ({NTREE} trees)", fp)

    _print("\n[exp3] note: this is BINARY survival proxy, not CR. Reference only.", fp)
    fp.close()


if __name__ == "__main__":
    main()
