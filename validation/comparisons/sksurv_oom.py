"""scikit-survival vs crforest paired wall + memory benchmark.

Cited by the README "vs scikit-survival" table. Each (lib, n, seed) cell
runs in its own subprocess so an OOM in one cell does not poison the
harness; RSS-cap via resource.RLIMIT_AS keeps the host responsive on hit.
The two libraries solve different tasks (sksurv RSF: single-event;
crforest: competing-risks); for memory-and-wall comparison, both fit on
the same X + t with a single-event collapse for sksurv (event in {1,2}
-> 1).

Run on a Linux benchmark host:
  PYTHONUNBUFFERED=1 uv run --with scikit-survival --extra dev \\
    python -u validation/comparisons/sksurv_oom.py \\
    --machine $(hostname) --ns 5000,10000,25000,50000

Output: /tmp/sksurv_oom.parquet (one row per cell) +
/tmp/sksurv_oom.parquet.fingerprint.json (git SHA + libs + machine).
"""

from __future__ import annotations

import argparse
import json
import pickle
import platform
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

THIS = Path(__file__).resolve()


def make_synthetic(n: int, p: int, seed: int):
    """Same 2-cause Weibull DGP used by validation/spikes/lambda exp5."""
    rng = np.random.default_rng(seed)
    X = rng.standard_normal(size=(n, p))
    beta1 = np.zeros(p)
    beta1[: min(5, p)] = [0.8, 0.4, -0.3, 0.0, 0.0][: min(5, p)]
    beta2 = np.zeros(p)
    if p >= 5:
        beta2[3:5] = [-0.5, 0.6]
    lam1 = np.exp(-3.0 + X @ beta1)
    lam2 = np.exp(-3.5 + X @ beta2)
    u1 = rng.uniform(size=n)
    u2 = rng.uniform(size=n)
    t1 = (-np.log(u1) / lam1) ** (1.0 / 1.2)
    t2 = (-np.log(u2) / lam2) ** (1.0 / 0.9)
    c = rng.exponential(scale=1.0 / 0.06, size=n)
    times = np.minimum.reduce([t1, t2, c])
    event = np.where(times == t1, 1, np.where(times == t2, 2, 0)).astype(np.int64)
    return X.astype(np.float64), times.astype(np.float64), event


def _emit(row: dict) -> None:
    print("RESULT_JSON " + json.dumps(row), flush=True)


def _peak_rss_gb() -> float:
    import resource

    ru = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    # Linux ru_maxrss is KB, macOS is bytes.
    return ru / (1024**2) if sys.platform.startswith("linux") else ru / (1024**3)


def child_main(
    lib: str,
    n: int,
    p: int,
    seed: int,
    ntree: int,
    mem_cap_gb: float,
    n_jobs: int,
    max_depth_sksurv: int | None,
    low_memory_sksurv: bool,
) -> None:
    """Run inside the cell subprocess; print one JSON line to stdout."""
    import resource

    out: dict = {
        "lib": lib,
        "n": n,
        "p": p,
        "seed": seed,
        "ntree": ntree,
        "mem_cap_gb": mem_cap_gb,
        "n_jobs": n_jobs,
        "max_depth_sksurv": max_depth_sksurv,
        "low_memory_sksurv": low_memory_sksurv,
    }

    # RLIMIT_AS is reliable on Linux; on Darwin the hard limit is often below
    # the size needed for a fresh CPython process, so setting a soft cap fails
    # with "current limit exceeds maximum limit". Skip silently on darwin.
    if mem_cap_gb > 0 and not sys.platform.startswith("darwin"):
        cap = int(mem_cap_gb * (1024**3))
        try:
            resource.setrlimit(resource.RLIMIT_AS, (cap, cap))
        except (ValueError, OSError) as exc:
            out.update(
                status="rlimit_error",
                wall=float("nan"),
                peak_rss_gb=float("nan"),
                pickle_gb=float("nan"),
                error=f"setrlimit({mem_cap_gb}GB): {exc}",
            )
            _emit(out)
            return

    try:
        X, t, e_cr = make_synthetic(n, p, seed)
        e_se = (e_cr > 0).astype(np.int64)  # single-event collapse for sksurv

        if lib == "sksurv":
            from sksurv.ensemble import RandomSurvivalForest

            y = np.array(
                list(zip(e_se.astype(bool), t, strict=True)),
                dtype=[("event", "?"), ("time", "<f8")],
            )
            model = RandomSurvivalForest(
                n_estimators=ntree,
                n_jobs=n_jobs,
                random_state=seed,
                max_depth=max_depth_sksurv,
                low_memory=low_memory_sksurv,
            )
            t0 = time.perf_counter()
            model.fit(X, y)
            wall = time.perf_counter() - t0
            fitted = model

        elif lib == "crforest":
            from crforest import CompetingRiskForest

            fitted = CompetingRiskForest(
                n_estimators=ntree,
                n_jobs=n_jobs,
                random_state=seed,
                device="cpu",  # avoid auto-detect on hosts w/ broken cupy install
            )
            t0 = time.perf_counter()
            fitted.fit(X, t, e_cr)
            wall = time.perf_counter() - t0

        else:
            raise ValueError(f"unknown lib: {lib}")

        try:
            pickle_bytes = len(pickle.dumps(fitted, protocol=pickle.HIGHEST_PROTOCOL))
        except MemoryError:
            pickle_bytes = -1

        out.update(
            status="ok",
            wall=wall,
            peak_rss_gb=_peak_rss_gb(),
            pickle_gb=pickle_bytes / 1e9 if pickle_bytes >= 0 else float("nan"),
            error="",
        )

    except MemoryError as exc:
        out.update(
            status="memory_error",
            wall=float("nan"),
            peak_rss_gb=_peak_rss_gb(),
            pickle_gb=float("nan"),
            error=f"MemoryError: {exc}",
        )

    except Exception as exc:
        out.update(
            status="error",
            wall=float("nan"),
            peak_rss_gb=_peak_rss_gb(),
            pickle_gb=float("nan"),
            error=f"{type(exc).__name__}: {str(exc)[:300]}",
        )

    _emit(out)


def run_cell(
    *,
    lib: str,
    n: int,
    p: int,
    seed: int,
    ntree: int,
    mem_cap_gb: float,
    timeout_s: int,
    n_jobs: int,
    max_depth_sksurv: int | None,
    low_memory_sksurv: bool,
) -> dict:
    """Spawn a child subprocess that runs child_main(...). Returns parsed row."""
    cmd = [
        sys.executable,
        str(THIS),
        "--child",
        "--lib",
        lib,
        "--n",
        str(n),
        "--p",
        str(p),
        "--seed",
        str(seed),
        "--ntree",
        str(ntree),
        "--mem-cap-gb",
        str(mem_cap_gb),
        "--n-jobs",
        str(n_jobs),
    ]
    if max_depth_sksurv is not None:
        cmd += ["--max-depth-sksurv", str(max_depth_sksurv)]
    if low_memory_sksurv:
        cmd += ["--low-memory-sksurv"]

    def _failure(status: str, error: str, elapsed: float) -> dict:
        return {
            "lib": lib,
            "n": n,
            "p": p,
            "seed": seed,
            "ntree": ntree,
            "mem_cap_gb": mem_cap_gb,
            "n_jobs": n_jobs,
            "max_depth_sksurv": max_depth_sksurv,
            "low_memory_sksurv": low_memory_sksurv,
            "status": status,
            "wall": float("nan"),
            "peak_rss_gb": float("nan"),
            "pickle_gb": float("nan"),
            "error": error,
            "wall_outer": elapsed,
        }

    t0 = time.perf_counter()
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout_s)
    except subprocess.TimeoutExpired:
        return _failure("timeout", f"timeout after {timeout_s}s", time.perf_counter() - t0)
    elapsed = time.perf_counter() - t0
    for line in proc.stdout.splitlines():
        if line.startswith("RESULT_JSON "):
            row = json.loads(line[len("RESULT_JSON ") :])
            row["wall_outer"] = elapsed
            return row
    # No JSON: child probably killed (e.g. OS OOM-killer not caught).
    return _failure("no_result", f"rc={proc.returncode}; stderr={proc.stderr[-300:]}", elapsed)


def fit_powerlaw(ns, ys):
    """Fit y = a * n^b on (ns, ys), return (a, b). Drops NaN/non-positive."""
    ns = np.asarray(ns, dtype=float)
    ys = np.asarray(ys, dtype=float)
    mask = np.isfinite(ns) & np.isfinite(ys) & (ns > 0) & (ys > 0)
    if mask.sum() < 2:
        return float("nan"), float("nan")
    log_n = np.log(ns[mask])
    log_y = np.log(ys[mask])
    b, log_a = np.polyfit(log_n, log_y, 1)
    return float(np.exp(log_a)), float(b)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--child", action="store_true", help="internal: run a single cell child")
    parser.add_argument("--lib", choices=["sksurv", "crforest"])
    parser.add_argument("--n", type=int)
    parser.add_argument("--p", type=int, default=58)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--ntree", type=int, default=100)
    parser.add_argument("--mem-cap-gb", type=float, default=0.0)
    parser.add_argument(
        "--n-jobs",
        type=int,
        default=1,
        help="passed to both libs; sksurv default is 1, crforest default is -1",
    )
    parser.add_argument(
        "--max-depth-sksurv",
        type=int,
        default=None,
        help="cap sksurv tree depth (set to 15 for crforest-matched config)",
    )
    parser.add_argument(
        "--low-memory-sksurv",
        action="store_true",
        help="sksurv RSF low_memory=True (each leaf 1 float; "
        "predict_chf/_survival become NotImplemented — predict-only)",
    )
    parser.add_argument("--machine", default=platform.node())
    parser.add_argument(
        "--ns", default="10000,25000,50000,100000", help="comma-sep n values to sweep"
    )
    parser.add_argument("--seeds", default="42")
    parser.add_argument("--libs", default="sksurv,crforest")
    parser.add_argument("--timeout", type=int, default=3600)
    parser.add_argument("--out", default="/tmp/sksurv_oom.parquet")
    args = parser.parse_args()

    if args.child:
        child_main(
            args.lib,
            args.n,
            args.p,
            args.seed,
            args.ntree,
            args.mem_cap_gb,
            args.n_jobs,
            args.max_depth_sksurv,
            args.low_memory_sksurv,
        )
        return

    if sys.platform == "darwin":
        total_gb = int(subprocess.check_output(["sysctl", "-n", "hw.memsize"]).strip()) / (1024**3)
    else:
        total_gb = 0.0
        for line in Path("/proc/meminfo").read_text().splitlines():
            if line.startswith("MemTotal:"):
                total_gb = int(line.split()[1]) / (1024**2)
                break
    cap = max(2.0, total_gb - 2.0)

    ns = [int(x) for x in args.ns.split(",") if x.strip()]
    seeds = [int(x) for x in args.seeds.split(",") if x.strip()]
    libs = [x.strip() for x in args.libs.split(",") if x.strip()]

    print(
        f"[harness] machine={args.machine} total_ram_gb={total_gb:.1f} cap={cap:.1f} GB", flush=True
    )
    print(
        f"[harness] libs={libs} ns={ns} seeds={seeds} ntree={args.ntree} p={args.p} "
        f"n_jobs={args.n_jobs} max_depth_sksurv={args.max_depth_sksurv}",
        flush=True,
    )

    sys.path.insert(0, str(Path(__file__).parent))
    from _fingerprint import dump_fingerprint

    fp_path = dump_fingerprint(args.out)
    print(f"[harness] fingerprint -> {fp_path}", flush=True)

    rows = []
    for n in ns:
        for lib in libs:
            for seed in seeds:
                print(f"\n[cell] lib={lib} n={n} seed={seed}", flush=True)
                row = run_cell(
                    lib=lib,
                    n=n,
                    p=args.p,
                    seed=seed,
                    ntree=args.ntree,
                    mem_cap_gb=cap,
                    timeout_s=args.timeout,
                    n_jobs=args.n_jobs,
                    max_depth_sksurv=args.max_depth_sksurv,
                    low_memory_sksurv=args.low_memory_sksurv,
                )
                row["machine"] = args.machine
                row["host_ram_gb"] = total_gb
                tag = (
                    f"wall={row['wall']:.1f}s rss={row['peak_rss_gb']:.2f}GB "
                    f"pickle={row['pickle_gb']:.2f}GB"
                    if row["status"] == "ok"
                    else f"{row['status']} ({row.get('error', '')[:120]})"
                )
                print(f"  {tag}", flush=True)
                rows.append(row)
                df_partial = pd.DataFrame(rows)
                df_partial.to_parquet(args.out)

    df = pd.DataFrame(rows)
    print(f"\n[dump] {args.out} ({len(df)} rows)", flush=True)

    print("\n## Summary\n", flush=True)
    ok = df[df["status"] == "ok"].copy()
    if len(ok):
        agg = ok.groupby(["lib", "n"])[["wall", "peak_rss_gb", "pickle_gb"]].mean().round(3)
        print(agg.to_string(), flush=True)

    for lib in libs:
        sub = ok[ok["lib"] == lib]
        if len(sub) >= 2:
            a, b = fit_powerlaw(sub["n"].to_numpy(), sub["peak_rss_gb"].to_numpy())
            est_100k = a * (100_000**b) if np.isfinite(a) else float("nan")
            print(
                f"\n[{lib}] peak_rss_gb ≈ {a:.3e} * n^{b:.2f}   "
                f"→ extrap @ n=100k: {est_100k:.2f} GB",
                flush=True,
            )

    err = df[df["status"] != "ok"]
    if len(err):
        print(f"\n## {len(err)} non-ok cells\n", flush=True)
        for _, r in err.iterrows():
            print(
                f"  lib={r['lib']} n={r['n']} seed={r['seed']} → "
                f"{r['status']}: {str(r.get('error', ''))[:180]}",
                flush=True,
            )


if __name__ == "__main__":
    main()
