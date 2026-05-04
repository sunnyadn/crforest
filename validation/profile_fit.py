"""Profile a representative CompetingRiskForest fit under cProfile.

Sprint-1 #1 of the comprisk perf plan. Generates synthetic 2-cause Weibull
data at a wanqi-cr-shaped size (n=60k by p=30 by default), fits the forest
under cProfile, and dumps both binary pstats and a sorted text report.

Usage
-----
    python -m validation.profile_fit                       # ntree=500 canonical
    python -m validation.profile_fit --ntree 100           # fast iteration pass
    python -m validation.profile_fit --ntree 500 --jobs 1  # serialize for cleaner profile
    python -m validation.profile_fit --device cuda         # GPU path (requires comprisk[gpu])
    python -m validation.profile_fit --n 20000 --p 20      # smaller smoke

Outputs (under validation/data/):
    profile_fit_{tag}.pstats                              # binary, view with snakeviz / pstats
    profile_fit_{tag}.txt                                 # top-30 cumulative + tottime

The tag includes n, p, ntree, jobs, and device so multiple runs coexist.

Top-of-script DGP mirrors gen_synthetic.py (2-cause Weibull cause-specific
hazards, 5 informative per cause, rest noise) but with adjustable N and P.
"""

from __future__ import annotations

import argparse
import cProfile
import io
import pstats
import time
from pathlib import Path

import numpy as np


def make_synthetic_cr(
    n: int, p: int, seed: int = 20260417
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """2-cause Weibull competing-risks DGP. 5 informative features per cause, rest noise."""
    rng = np.random.default_rng(seed)
    X = rng.standard_normal(size=(n, p)).astype(np.float64)
    beta_1 = np.zeros(p)
    beta_2 = np.zeros(p)
    beta_1[:5] = np.array([0.8, 0.4, -0.3, 0.0, 0.0])[:5]
    beta_2[:5] = np.array([0.0, 0.0, 0.0, -0.5, 0.6])[:5]
    if p >= 10:
        beta_2[5:10] = np.array([0.5, -0.4, 0.3, 0.0, -0.6])
        beta_1[5:10] = np.array([0.0, 0.3, -0.5, 0.4, 0.0])
    alpha = (1.2, 0.9)
    intercept_1, intercept_2 = -3.0, -3.5
    censor_rate = 0.06
    lam1 = np.exp(intercept_1 + X @ beta_1)
    lam2 = np.exp(intercept_2 + X @ beta_2)
    u1 = rng.uniform(size=n)
    u2 = rng.uniform(size=n)
    t1 = (-np.log(u1) / lam1) ** (1.0 / alpha[0])
    t2 = (-np.log(u2) / lam2) ** (1.0 / alpha[1])
    c = rng.exponential(scale=1.0 / censor_rate, size=n)
    times = np.minimum.reduce([t1, t2, c])
    event = np.where(times == t1, 1, np.where(times == t2, 2, 0)).astype(np.int64)
    return X, times, event


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--n", type=int, default=60_000, help="rows (default: 60000, wanqi-cr scale)")
    ap.add_argument("--p", type=int, default=30, help="features (default: 30, wanqi-cr scale)")
    ap.add_argument("--ntree", type=int, default=500, help="n_estimators (default: 500)")
    ap.add_argument("--jobs", type=int, default=-1, help="n_jobs (default: -1)")
    ap.add_argument("--seed", type=int, default=20260417)
    ap.add_argument("--n-bins", type=int, default=256, help="n_bins (default: 256)")
    ap.add_argument(
        "--device",
        choices=("auto", "cpu", "cuda"),
        default="auto",
        help="compute backend (default: auto -> cpu in v0.1; cuda requires comprisk[gpu])",
    )
    ap.add_argument("--outdir", type=Path, default=Path(__file__).resolve().parent / "data")
    args = ap.parse_args()

    args.outdir.mkdir(parents=True, exist_ok=True)
    tag_device = "" if args.device == "auto" else f"_dev-{args.device}"
    tag = f"n{args.n}_p{args.p}_ntree{args.ntree}_jobs{args.jobs}{tag_device}"
    pstats_path = args.outdir / f"profile_fit_{tag}.pstats"
    text_path = args.outdir / f"profile_fit_{tag}.txt"

    print(f"[profile_fit] generating synthetic CR data: n={args.n}, p={args.p}, seed={args.seed}")
    t0 = time.perf_counter()
    X, t, e = make_synthetic_cr(args.n, args.p, seed=args.seed)
    censor_pct = (e == 0).mean()
    cause1_pct = (e == 1).mean()
    cause2_pct = (e == 2).mean()
    print(
        f"[profile_fit] data ready in {time.perf_counter() - t0:.2f}s "
        f"(censor={censor_pct:.1%}, c1={cause1_pct:.1%}, c2={cause2_pct:.1%})"
    )

    # Lazy import so cProfile sees the relevant call stack from `fit`, not from
    # numba JIT cache loading at module import.
    from comprisk import CompetingRiskForest

    forest = CompetingRiskForest(
        n_estimators=args.ntree,
        n_jobs=args.jobs,
        n_bins=args.n_bins,
        random_state=args.seed,
        device=args.device,
    )

    # Warm numba JIT with a tiny fit so the profile doesn't double-count
    # one-time compile time as "fit cost".
    print("[profile_fit] warming numba JIT (tiny fit)...")
    warm = CompetingRiskForest(
        n_estimators=2, n_jobs=1, n_bins=args.n_bins, random_state=args.seed, device=args.device
    )
    warm.fit(X[:200], t[:200], e[:200])

    print(
        f"[profile_fit] profiling fit (ntree={args.ntree}, n_jobs={args.jobs}, device={args.device})..."
    )
    profiler = cProfile.Profile()
    t0 = time.perf_counter()
    profiler.enable()
    forest.fit(X, t, e)
    profiler.disable()
    wall = time.perf_counter() - t0
    print(f"[profile_fit] fit complete: wall={wall:.2f}s")

    profiler.dump_stats(str(pstats_path))
    print(f"[profile_fit] pstats -> {pstats_path}")

    buf = io.StringIO()
    buf.write("# comprisk profile_fit\n")
    buf.write(f"# tag: {tag}\n")
    buf.write(f"# wall_seconds: {wall:.4f}\n")
    buf.write(f"# n={args.n} p={args.p} ntree={args.ntree} jobs={args.jobs} n_bins={args.n_bins}\n")
    buf.write(f"# censor={censor_pct:.4f} c1={cause1_pct:.4f} c2={cause2_pct:.4f}\n\n")
    buf.write("## Top 30 by cumulative time\n")
    ps = pstats.Stats(profiler, stream=buf).sort_stats("cumulative")
    ps.print_stats(30)
    buf.write("\n## Top 30 by tottime (self time)\n")
    ps = pstats.Stats(profiler, stream=buf).sort_stats("tottime")
    ps.print_stats(30)
    buf.write("\n## Callers of top-5 tottime\n")
    ps = pstats.Stats(profiler, stream=buf).sort_stats("tottime")
    ps.print_callers(5)
    text_path.write_text(buf.getvalue())
    print(f"[profile_fit] report -> {text_path}")
    print(f"[profile_fit] DONE  wall={wall:.2f}s  ntree={args.ntree}")


if __name__ == "__main__":
    main()
