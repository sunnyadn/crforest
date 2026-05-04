"""λ.exp1 — η-style profile of Plan 2 GPU on real CHF p=58.

Plan 3 pre-flight gate. η.exp1 measured single-tree wall 3.0s with kernel-only
floor 0.03s = 99% non-kernel overhead, but on synthetic p=8. At real CHF p=58
each split evaluates ~8 candidate features (mtry≈√58), so cand_mask + xb_view
host loops do ~2.5× more iterations. Question: is host orchestration STILL
dominant at p=58, or has the kernel slice grown?

Two sweeps:
  A. Real CHF (n=75k train, p=58)   single-tree GPU fit → cProfile
  B. Synthetic competing risks p=10 single-tree GPU fit → cProfile (control)

For each: total wall + cProfile top 30 by cumulative + tottime, plus a
cupy-vs-non-cupy aggregate so we can read host/GPU split off the table.

POC gate: if non-cupy tottime fraction ≥ 70% on real CHF → η's 99% finding
extends to wide-p; Plan 3 Phase A (full-device pipeline) can keep its
2-3× ceiling target. If ≤ 50% → revise.

Run: ssh win 'export PATH=$HOME/.local/bin:$PATH && cd ~/comprisk && \\
       PYTHONUNBUFFERED=1 uv run --extra gpu --extra dev \\
       python -u validation/spikes/lambda/exp1_eta_on_real_chf.py \\
       2>&1 | tee /tmp/lambda_exp1.log'
"""

from __future__ import annotations

import cProfile
import io
import pstats
import sys
import time
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).parents[1]))

from _lambda_helpers import load_chf, make_synthetic

from comprisk import CompetingRiskForest

SEED = 0
NTREE_TIMED = 1  # single-tree, n_jobs=1 — clean per-tree wall
NTREE_WARMUP = 4  # JIT + cupy compile warmup
SYNTH_N = 100_000
SYNTH_P = 10


def fit_one(X, t, e, *, n_estimators, sync_after=True):
    f = CompetingRiskForest(
        n_estimators=n_estimators,
        n_jobs=1,
        random_state=SEED,
        device="cuda",
    )
    f.fit(X, t, e)
    if sync_after:
        import cupy as cp

        cp.cuda.runtime.deviceSynchronize()
    return f


def aggregate_cupy_share(profile: cProfile.Profile) -> dict[str, float]:
    """Sum tottime grouped by file-prefix. Picks out cupy vs everything else.

    Uses the un-stripped pstats.Stats so absolute paths reveal module owner
    (cupy / comprisk / numpy / stdlib). The function name field also catches
    built-in C method names like ``{method 'get' of 'cupy._core...' objects}``.
    """
    stats = pstats.Stats(profile)  # NOT strip_dirs — keep full paths
    cupy_tottime = 0.0
    comprisk_tottime = 0.0
    other_tottime = 0.0
    total_tottime = 0.0
    for func, (_cc, _nc, tt, _ct, _) in stats.stats.items():
        path, _line, name = func
        haystack = (str(path) + " " + str(name)).lower()
        total_tottime += tt
        if "cupy" in haystack or "/cuda/" in haystack:
            cupy_tottime += tt
        elif "comprisk" in haystack:
            comprisk_tottime += tt
        else:
            other_tottime += tt
    return {
        "total_tottime": total_tottime,
        "cupy_tottime": cupy_tottime,
        "comprisk_tottime": comprisk_tottime,
        "other_tottime": other_tottime,
    }


def profile_one(label: str, X, t, e):
    print(f"\n========== {label}: n={len(X):,} p={X.shape[1]} ==========", flush=True)

    print("[warmup] cuda compile + 4-tree fit on 5k slice...", flush=True)
    fit_one(X[:5000], t[:5000], e[:5000], n_estimators=NTREE_WARMUP)

    print(f"[wall] timing single-tree fit (n_jobs=1, ntree={NTREE_TIMED})...", flush=True)
    t0 = time.perf_counter()
    fit_one(X, t, e, n_estimators=NTREE_TIMED)
    wall_clean = time.perf_counter() - t0
    print(f"[wall] WALL_CLEAN = {wall_clean:.3f}s", flush=True)

    print("[cprofile] timing single-tree fit under cProfile...", flush=True)
    pr = cProfile.Profile()
    pr.enable()
    t0 = time.perf_counter()
    fit_one(X, t, e, n_estimators=NTREE_TIMED)
    wall_prof = time.perf_counter() - t0
    pr.disable()
    print(
        f"[cprofile] WALL_PROFILE = {wall_prof:.3f}s (overhead {wall_prof - wall_clean:+.3f}s)",
        flush=True,
    )

    s = io.StringIO()
    pstats.Stats(pr, stream=s).strip_dirs().sort_stats("cumulative").print_stats(25)
    print(f"\n--- {label}: top 25 by cumulative ---", flush=True)
    print(s.getvalue(), flush=True)

    s2 = io.StringIO()
    pstats.Stats(pr, stream=s2).strip_dirs().sort_stats("tottime").print_stats(25)
    print(f"--- {label}: top 25 by tottime ---", flush=True)
    print(s2.getvalue(), flush=True)

    agg = aggregate_cupy_share(pr)
    total = agg["total_tottime"] or 1e-9
    print(f"--- {label}: tottime split ---", flush=True)
    print(f"  total_tottime    = {agg['total_tottime']:.3f}s", flush=True)
    print(
        f"  cupy_tottime     = {agg['cupy_tottime']:.3f}s "
        f"({100 * agg['cupy_tottime'] / total:.1f}%)  "
        "[mostly D↔H sync round-trips — Phase A removes most]",
        flush=True,
    )
    print(
        f"  comprisk_tottime = {agg['comprisk_tottime']:.3f}s "
        f"({100 * agg['comprisk_tottime'] / total:.1f}%)  "
        "[Python loop in build_flat_tree_gpu — Phase A removes]",
        flush=True,
    )
    print(
        f"  other_tottime    = {agg['other_tottime']:.3f}s "
        f"({100 * agg['other_tottime'] / total:.1f}%)  "
        "[numpy on host (searchsorted, unique, sort) — Phase A1 moves to GPU]",
        flush=True,
    )

    addressable = agg["cupy_tottime"] + agg["comprisk_tottime"] + agg["other_tottime"]
    print(
        f"  Phase-A addressable = {100 * addressable / wall_clean:.1f}% of WALL "
        f"(host orch + sync round-trips + host numpy; "
        f"residual {100 * (1 - addressable / wall_clean):.1f}% is async GPU compute)",
        flush=True,
    )

    return {
        "label": label,
        "n": len(X),
        "p": X.shape[1],
        "wall_clean": wall_clean,
        "wall_prof": wall_prof,
        "addressable_share_of_wall": addressable / wall_clean,
        **agg,
    }


def main() -> None:
    rows = []

    Xr, tr, er, _p_real = load_chf()
    rows.append(profile_one("real_chf", Xr, tr, er))

    Xs, ts, es = make_synthetic(SYNTH_N, SYNTH_P, seed=20260417)
    rows.append(profile_one("synth_p10", Xs, ts, es))

    print("\n========== Summary ==========", flush=True)
    summary = pd.DataFrame(rows)
    print(summary.to_string(index=False), flush=True)

    print("\n========== POC gate ==========", flush=True)
    real = next(r for r in rows if r["label"] == "real_chf")
    addr = real["addressable_share_of_wall"]
    if addr >= 0.85:
        verdict = "PASS"
        msg = "η's 99%-non-kernel finding extends to GPU at p=58. Phase A 2-3× ceiling holds."
    elif addr >= 0.65:
        verdict = "MARGINAL"
        msg = "Phase A still pays off but ceiling shaves to maybe 1.5-2× on real CHF."
    else:
        verdict = "FAIL"
        msg = "Kernel work dominates; Phase A small-payoff. Pivot to kernel-only or accept ceiling."
    print(
        f"{verdict}: real CHF Phase-A addressable = {addr:.1%}.  {msg}",
        flush=True,
    )


if __name__ == "__main__":
    main()
