"""Single-config comprisk bench. Emits one CSV row to bench/results/results.csv.

Usage:
    python -m bench.run_comprisk --n 60000 --p 30 --ntree 100 \\
        --leaf 3 --jobs 10 --label mac-m3pro

All knobs default to v0.2-canonical values; override per run. Run from the
comprisk repo root so the relative imports work.
"""

from __future__ import annotations

import argparse
import csv
import os
import socket
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

# Resolve repo root from this file's location.
BENCH_DIR = Path(__file__).resolve().parent
REPO_ROOT = BENCH_DIR.parent
sys.path.insert(0, str(REPO_ROOT))

from bench.dgp import make_synthetic_cr  # noqa: E402


def get_commit() -> str:
    try:
        return subprocess.check_output(
            ["git", "-C", str(REPO_ROOT), "rev-parse", "--short", "HEAD"],
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except Exception:
        return ""


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=60_000)
    ap.add_argument("--p", type=int, default=30)
    ap.add_argument("--ntree", type=int, default=100)
    ap.add_argument("--leaf", type=int, default=3, help="min_samples_leaf")
    ap.add_argument("--split", type=int, default=None, help="min_samples_split (default 2*leaf)")
    ap.add_argument("--n-bins", type=int, default=256)
    ap.add_argument("--jobs", type=int, default=-1)
    ap.add_argument("--seed", type=int, default=20260417)
    ap.add_argument("--label", default=socket.gethostname())
    ap.add_argument("--splitrule", default="logrankCR")
    args = ap.parse_args()

    if args.split is None:
        args.split = 2 * args.leaf

    print(
        f"comprisk | n={args.n} p={args.p} ntree={args.ntree} "
        f"leaf={args.leaf} split={args.split} jobs={args.jobs} label={args.label}"
    )

    X, t, e = make_synthetic_cr(args.n, args.p, seed=args.seed)
    print(
        f"data ready: censor={(e == 0).mean():.1%} c1={(e == 1).mean():.1%} c2={(e == 2).mean():.1%}"
    )

    from comprisk import CompetingRiskForest
    from comprisk import __version__ as crf_version

    # Warm numba JIT so timing excludes one-shot compile cost
    warm = CompetingRiskForest(
        n_estimators=2,
        n_jobs=1,
        n_bins=args.n_bins,
        random_state=args.seed,
        device="cpu",
        min_samples_leaf=args.leaf,
        min_samples_split=args.split,
    )
    warm.fit(X[:200], t[:200], e[:200])

    forest = CompetingRiskForest(
        n_estimators=args.ntree,
        n_jobs=args.jobs,
        n_bins=args.n_bins,
        random_state=args.seed,
        device="cpu",
        min_samples_leaf=args.leaf,
        min_samples_split=args.split,
        splitrule=args.splitrule,
    )

    t0_wall = time.perf_counter()
    t0_cpu = time.process_time()
    forest.fit(X, t, e)
    wall = time.perf_counter() - t0_wall
    cpu = time.process_time() - t0_cpu  # main process only — undercounts joblib workers

    print(f"DONE wall={wall:.2f}s cpu_main={cpu:.2f}s")

    results_path = BENCH_DIR / "results" / "results.csv"
    results_path.parent.mkdir(parents=True, exist_ok=True)
    header_needed = not results_path.exists()
    with results_path.open("a", newline="") as f:
        w = csv.writer(f, quoting=csv.QUOTE_MINIMAL)
        if header_needed:
            w.writerow(
                [
                    "timestamp",
                    "library",
                    "version",
                    "hardware",
                    "n_cores_used",
                    "n",
                    "p",
                    "ntree",
                    "leaf_or_nodesize",
                    "nsplit",
                    "n_bins",
                    "splitrule",
                    "wall_s",
                    "cpu_s",
                    "parallel_ratio",
                    "commit",
                    "notes",
                ]
            )
        # comprisk's effective n_jobs: -1 → os.cpu_count()
        n_cores_used = os.cpu_count() if args.jobs == -1 else args.jobs
        w.writerow(
            [
                datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S%z"),
                "comprisk",
                crf_version,
                args.label,
                n_cores_used,
                args.n,
                args.p,
                args.ntree,
                args.leaf,
                10,  # comprisk default-mode nsplit
                args.n_bins,
                args.splitrule,
                round(wall, 3),
                round(cpu, 3),
                "",  # main-process cpu only; not directly comparable to R's user+sys
                get_commit(),
                "",
            ]
        )
    print(f"appended -> {results_path}")


if __name__ == "__main__":
    main()
