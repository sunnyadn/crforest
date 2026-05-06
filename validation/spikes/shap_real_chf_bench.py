"""SUN-43 follow-up: real-data SHAP wall-clock bench on the CHF cohort.

Reports per-config wall-time + samples/sec for shap_values() on
CompetingRiskForest, plus additivity max-error as numerical sanity.
The README's TreeSHAP section quotes representative numbers from this
script.

Pre-req: run `validation/spikes/kappa/exp1_chf_smoke.py` first to stage the
preprocessed parquet at `/tmp/chf_2012_*`. Then:

    uv run python -u validation/spikes/shap_real_chf_bench.py
"""

from __future__ import annotations

import os
import sys
import time as _time

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(__file__))
from _lambda_helpers import load_chf

from comprisk import CompetingRiskForest


def bench(X_full, t_full, e_full, n_train, n_explain, n_times, n_jobs, seed=0):
    rng = np.random.default_rng(seed)
    idx = rng.permutation(len(X_full))[: n_train + n_explain]
    X_tr, t_tr, e_tr = X_full[idx[:n_train]], t_full[idx[:n_train]], e_full[idx[:n_train]]
    X_ex = X_full[idx[n_train : n_train + n_explain]]

    t0 = _time.perf_counter()
    f = CompetingRiskForest(n_estimators=100, random_state=42, n_jobs=n_jobs).fit(X_tr, t_tr, e_tr)
    t_fit = _time.perf_counter() - t0
    print(
        f"[fit] n_train={n_train} n_jobs={n_jobs} wall={t_fit:.1f}s "
        f"ntimes_full={len(f.unique_times_)}",
        flush=True,
    )

    times_grid = np.quantile(t_tr[e_tr > 0], np.linspace(0.1, 0.9, n_times))

    print(
        f"[shap] explaining n={n_explain}, times={n_times}, n_jobs={n_jobs} ...",
        flush=True,
    )
    t0 = _time.perf_counter()
    sh, base = f.shap_values(X_ex, times=times_grid)
    t_shap = _time.perf_counter() - t0

    cif_pred = f.predict_cif(X_ex, times=times_grid)
    add_err = float(np.max(np.abs((sh.sum(axis=1) + base).transpose(0, 2, 1) - cif_pred)))

    rate = n_explain / t_shap
    print(
        f"[shap] n_train={n_train} n_explain={n_explain} n_times={n_times} "
        f"n_jobs={n_jobs} -> wall={t_shap:.2f}s rate={rate:.1f} samples/s "
        f"add_err={add_err:.2e}",
        flush=True,
    )
    return {
        "n_train": n_train,
        "n_explain": n_explain,
        "n_times": n_times,
        "n_jobs": n_jobs,
        "fit_s": t_fit,
        "shap_s": t_shap,
        "rate_samples_per_s": rate,
        "add_err": add_err,
    }


def main():
    X, t, e, _ = load_chf()
    print(f"[data] X={X.shape} cpu_count={os.cpu_count()}", flush=True)

    configs = [
        (2000, 200, 10, 4),
        (10000, 200, 10, 4),
        (10000, 200, 10, -1),
        (10000, 1000, 10, -1),
    ]
    results = [bench(X, t, e, *cfg) for cfg in configs]

    print("\n=== summary ===", flush=True)
    print(pd.DataFrame(results).to_string(index=False), flush=True)


if __name__ == "__main__":
    sys.stdout.reconfigure(line_buffering=True)
    main()
