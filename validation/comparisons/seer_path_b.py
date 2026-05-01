"""Paired rfSRC OMP-on vs crforest matched-pair on SEER breast cancer.

Mirrors the n75k_path_b harness structure for cross-dataset comparison.
Reads the staged SEER cohort at /tmp/seer_breast_*.parquet (run
validation/gen_seer_breast.py once to vendor it).

Each (lib, seed) cell runs in its own subprocess wrapped by `/usr/bin/time -v`
so peak RSS is measured uniformly. Reports mean ± std across seeds for
fit_wall, peak_rss_gb, and Harrell C-index for both competing causes.

Run on a Linux box with rfSRC + crforest installed:
  PYTHONUNBUFFERED=1 python -u validation/comparisons/seer_path_b.py \\
    --seeds 42,43,44 --cells rfsrc_on,crforest

For memory-constrained boxes (rfSRC at full n=238k needs ~55GB), regenerate
the cohort with `gen_seer_breast.py --subsample 75000` first.

Output: /tmp/seer_path_b.parquet + .fingerprint.json.
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import re
import subprocess
import sys
import time
from pathlib import Path

import pandas as pd

THIS = Path(__file__).resolve()
HERE = THIS.parent
RFSRC_R = HERE / "_seer_path_b_rfsrc.R"

DATA_PATH = Path("/tmp/seer_breast_clean.parquet")
TRAIN_IDX = Path("/tmp/seer_breast_train_idx.txt")
TEST_IDX = Path("/tmp/seer_breast_test_idx.txt")

RSS_RE = re.compile(r"Maximum resident set size \(kbytes\):\s+(\d+)")


def parse_peak_rss_gb(stderr_text: str) -> float:
    """GNU /usr/bin/time -v reports max RSS in KB on Linux."""
    m = RSS_RE.search(stderr_text)
    return int(m.group(1)) / (1024**2) if m else float("nan")


def _emit(row: dict) -> None:
    print("RESULT_JSON " + json.dumps(row), flush=True)


def child_crforest(seed: int, n_jobs: int) -> None:
    """Single crforest fit on staged SEER cohort."""
    import numpy as np
    import pandas as pd

    from crforest import CompetingRiskForest, concordance_index_cr

    df = pd.read_parquet(DATA_PATH)
    train_idx = np.loadtxt(TRAIN_IDX, dtype=np.int64)
    test_idx = np.loadtxt(TEST_IDX, dtype=np.int64)
    feat = [c for c in df.columns if c not in ("time", "status")]
    X = df[feat].to_numpy(dtype=np.float64)
    t = df["time"].to_numpy(dtype=np.float64)
    e = df["status"].to_numpy(dtype=np.int64)
    X_tr, t_tr, e_tr = X[train_idx], t[train_idx], e[train_idx]
    X_te, t_te, e_te = X[test_idx], t[test_idx], e[test_idx]

    f = CompetingRiskForest(n_estimators=100, n_jobs=n_jobs, random_state=seed, device="cpu")
    t0 = time.perf_counter()
    f.fit(X_tr, t_tr, e_tr)
    wall = time.perf_counter() - t0
    risk1 = f.predict_risk(X_te, cause=1)
    risk2 = f.predict_risk(X_te, cause=2)
    c1 = concordance_index_cr(e_te, t_te, risk1, cause=1)
    c2 = concordance_index_cr(e_te, t_te, risk2, cause=2)

    _emit(
        {
            "lib": "crforest",
            "seed": seed,
            "rf_cores": -1,
            "fit_wall": wall,
            "harrell_c1": c1,
            "harrell_c2": c2,
        }
    )


def run_cell(*, lib: str, seed: int, rf_cores: int, timeout_s: int) -> dict:
    """Spawn time -v subprocess for one fit. Returns parsed row + RSS."""
    if lib == "rfsrc":
        cmd = ["/usr/bin/time", "-v", "Rscript", str(RFSRC_R), str(seed), str(rf_cores)]
    elif lib == "crforest":
        cmd = [
            "/usr/bin/time",
            "-v",
            sys.executable,
            str(THIS),
            "--child",
            "--seed",
            str(seed),
            "--n-jobs",
            str(rf_cores),
        ]
    else:
        raise ValueError(f"unknown lib: {lib}")

    t0 = time.perf_counter()
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout_s)
        elapsed = time.perf_counter() - t0
    except subprocess.TimeoutExpired:
        return {
            "lib": lib,
            "seed": seed,
            "rf_cores": rf_cores,
            "fit_wall": float("nan"),
            "harrell_c1": float("nan"),
            "harrell_c2": float("nan"),
            "peak_rss_gb": float("nan"),
            "wall_outer": time.perf_counter() - t0,
            "status": "timeout",
            "error": f"timeout after {timeout_s}s",
        }

    row = None
    for line in proc.stdout.splitlines():
        if line.startswith("RESULT_JSON "):
            row = json.loads(line[len("RESULT_JSON ") :])
            break

    rss = parse_peak_rss_gb(proc.stderr)
    if row is None:
        return {
            "lib": lib,
            "seed": seed,
            "rf_cores": rf_cores,
            "fit_wall": float("nan"),
            "harrell_c1": float("nan"),
            "harrell_c2": float("nan"),
            "peak_rss_gb": rss,
            "wall_outer": elapsed,
            "status": "no_result",
            "error": f"rc={proc.returncode}; stderr_tail={proc.stderr[-300:]}",
        }
    row.update(peak_rss_gb=rss, wall_outer=elapsed, status="ok", error="")
    return row


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--child", action="store_true", help="(internal) run crforest cell")
    parser.add_argument("--seed", type=int)
    parser.add_argument("--n-jobs", type=int, default=-1)
    parser.add_argument("--seeds", default="42,43,44", help="comma-sep seeds; default 3")
    parser.add_argument(
        "--cores-on",
        type=int,
        default=os.cpu_count() or 1,
        help="rf.cores for OMP-on cell (default: all)",
    )
    parser.add_argument(
        "--timeout",
        type=int,
        default=14400,
        help="per-cell timeout in s (default 4hr)",
    )
    parser.add_argument("--out", default="/tmp/seer_path_b.parquet")
    parser.add_argument(
        "--cells",
        default="rfsrc_on,crforest",
        help="comma-sep subset of {rfsrc_on, crforest}",
    )
    parser.add_argument("--machine", default=platform.node())
    args = parser.parse_args()

    if args.child:
        child_crforest(args.seed, args.n_jobs)
        return

    if not DATA_PATH.exists():
        sys.exit(f"missing {DATA_PATH}; run validation/gen_seer_breast.py first")

    seeds = [int(s) for s in args.seeds.split(",") if s.strip()]
    cells = [c.strip() for c in args.cells.split(",") if c.strip()]

    sys.path.insert(0, str(HERE))
    from _fingerprint import dump_fingerprint

    fp_path = dump_fingerprint(args.out)
    print(
        f"[harness] machine={args.machine} cells={cells} seeds={seeds} cores_on={args.cores_on}",
        flush=True,
    )
    print(f"[harness] fingerprint -> {fp_path}", flush=True)

    cell_specs = []
    if "rfsrc_on" in cells:
        cell_specs += [("rfsrc", s, args.cores_on) for s in seeds]
    if "crforest" in cells:
        cell_specs += [("crforest", s, -1) for s in seeds]

    rows = []
    for i, (lib, seed, cores) in enumerate(cell_specs, 1):
        print(f"\n[cell {i}/{len(cell_specs)}] lib={lib} seed={seed} cores={cores}", flush=True)
        row = run_cell(lib=lib, seed=seed, rf_cores=cores, timeout_s=args.timeout)
        row["machine"] = args.machine
        tag = (
            f"wall={row['fit_wall']:.2f}s rss={row['peak_rss_gb']:.2f}GB c1={row['harrell_c1']:.4f}"
            if row["status"] == "ok"
            else f"{row['status']}: {row.get('error', '')[:120]}"
        )
        print(f"  {tag}", flush=True)
        rows.append(row)
        pd.DataFrame(rows).to_parquet(args.out)

    df = pd.DataFrame(rows)
    print(f"\n[dump] {args.out} ({len(df)} rows)\n", flush=True)

    print("## Summary (mean ± std across seeds)\n", flush=True)
    ok = df[df["status"] == "ok"].copy()
    if len(ok):
        ok["cell"] = ok.apply(
            lambda r: "rfsrc_on" if r["lib"] == "rfsrc" else "crforest",
            axis=1,
        )
        agg = (
            ok.groupby("cell")[["fit_wall", "peak_rss_gb", "harrell_c1", "harrell_c2"]]
            .agg(["mean", "std"])
            .round(4)
        )
        print(agg.to_string(), flush=True)


if __name__ == "__main__":
    main()
