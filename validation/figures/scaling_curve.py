"""Generate the wall-vs-n scaling figure for docs/benchmarks.md.

Two modes (one script):

  python validation/figures/scaling_curve.py --bench [--out walls.json]
      Re-measures comprisk walls at the n grid {5k, 10k, ..., 1M} and
      writes a JSON file with mean walls per n. Run from a venv with
      comprisk installed; matplotlib is NOT required for this mode.

  python validation/figures/scaling_curve.py --render [--in walls.json]
      Reads walls.json and renders an SVG. Requires matplotlib. If your
      comprisk venv lacks matplotlib, run:
          uvx --from matplotlib --with matplotlib python \\
              validation/figures/scaling_curve.py --render

Synthetic 2-cause Weibull DGP, p=58, ntree=100; mirrors the DGP in
validation/comparisons/sksurv_oom.py exactly so cross-figure comparison
to the sksurv table is apples-to-apples. sksurv walls are baked in
(deterministic + unchanged install + already cited in the README).
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

THIS = Path(__file__).resolve()
DEFAULT_WALLS_JSON = THIS.parent / "scaling_walls.json"
DEFAULT_SVG = THIS.parent.parent.parent / "docs/figures/scaling_curve.svg"

# sksurv low_memory=True walls (cited from README, single-seed deterministic).
SKSURV_WALLS = {5_000: 18.2, 10_000: 85.0, 25_000: 609.7, 50_000: 2935.3}


def make_synthetic(n: int, p: int, seed: int):
    """Same DGP as validation/comparisons/sksurv_oom.py:make_synthetic."""
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


def child(n: int, p: int, seed: int, ntree: int):
    """Run inside a clean subprocess so peak RSS / numba caches don't bleed."""
    from comprisk import CompetingRiskForest

    X, t, e = make_synthetic(n, p, seed)
    f = CompetingRiskForest(n_estimators=ntree, n_jobs=-1, random_state=seed, device="cpu")
    t0 = time.perf_counter()
    f.fit(X, t, e)
    wall = time.perf_counter() - t0
    print("RESULT_JSON " + json.dumps({"n": n, "seed": seed, "wall": wall}), flush=True)


def measure_comprisk(ns, seeds, p, ntree):
    walls = {n: [] for n in ns}
    for n in ns:
        for seed in seeds:
            cmd = [
                sys.executable,
                str(THIS),
                "--child",
                "--n",
                str(n),
                "--seed",
                str(seed),
                "--ntree",
                str(ntree),
                "--p",
                str(p),
            ]
            proc = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
            for line in proc.stdout.splitlines():
                if line.startswith("RESULT_JSON "):
                    row = json.loads(line[len("RESULT_JSON ") :])
                    walls[n].append(row["wall"])
                    print(f"[comprisk n={n} seed={seed}] wall={row['wall']:.2f}s", flush=True)
                    break
    return {n: float(np.mean(walls[n])) for n in ns}


def render_svg(comprisk_walls, out_path, rfsrc_walls=None):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.ticker import FuncFormatter

    _fig, ax = plt.subplots(figsize=(10.0, 6.5), dpi=120)

    sk_ns = sorted(SKSURV_WALLS)
    sk_ts = [SKSURV_WALLS[n] for n in sk_ns]
    cf_ns = sorted(comprisk_walls)
    cf_ts = [comprisk_walls[n] for n in cf_ns]

    ax.plot(
        sk_ns,
        sk_ts,
        "o-",
        color="#888888",
        linewidth=3,
        markersize=11,
        label="scikit-survival (low_memory=True)",
    )
    if rfsrc_walls:
        rf_ns = sorted(rfsrc_walls)
        rf_ts = [rfsrc_walls[n] for n in rf_ns]
        ax.plot(
            rf_ns,
            rf_ts,
            "s-",
            color="#C44E52",
            linewidth=3,
            markersize=11,
            label="randomForestSRC (OMP-on, 28 threads)",
        )
    ax.plot(cf_ns, cf_ts, "o-", color="#2E5C8A", linewidth=3.5, markersize=12, label="comprisk")

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("n (subjects)", fontsize=17)
    ax.set_ylabel("wall time (seconds)", fontsize=17)
    ax.set_title(
        "Wall time vs n — synthetic 2-cause Weibull, p=58, ntree=100\n"
        "i7-14700K (28 threads), n_jobs=-1",
        fontsize=15,
    )

    def _fmt_y(y, _):
        if y < 1:
            return f"{y:g}"
        return f"{y:.0f}"

    def _fmt_x(x, _):
        if x >= 1_000_000:
            return f"{x / 1_000_000:g}M"
        if x >= 1000:
            return f"{x / 1000:g}k"
        return f"{x:g}"

    ax.yaxis.set_major_formatter(FuncFormatter(_fmt_y))
    ax.xaxis.set_major_formatter(FuncFormatter(_fmt_x))
    ax.tick_params(axis="both", which="major", labelsize=14)
    ax.tick_params(axis="both", which="minor", labelsize=11, length=3)
    ax.grid(True, which="major", linestyle="-", alpha=0.3)
    ax.grid(True, which="minor", linestyle=":", alpha=0.2)
    ax.legend(loc="upper left", fontsize=14, framealpha=0.95)

    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, format="svg", bbox_inches="tight")
    print(f"[svg] {out_path} ({out_path.stat().st_size / 1024:.1f} KB)")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--child", action="store_true", help="(internal)")
    parser.add_argument("--n", type=int)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--ntree", type=int, default=100)
    parser.add_argument("--p", type=int, default=58)
    parser.add_argument(
        "--bench", action="store_true", help="re-measure comprisk walls and write JSON"
    )
    parser.add_argument("--render", action="store_true", help="render SVG from existing walls JSON")
    parser.add_argument("--ns", default="5000,10000,25000,50000,100000,250000,500000,1000000")
    parser.add_argument("--seeds", default="42,43,44")
    parser.add_argument("--walls-json", type=Path, default=DEFAULT_WALLS_JSON)
    parser.add_argument("--svg-out", type=Path, default=DEFAULT_SVG)
    args = parser.parse_args()

    if args.child:
        child(args.n, args.p, args.seed, args.ntree)
        return

    if args.bench:
        ns = [int(s) for s in args.ns.split(",")]
        seeds = [int(s) for s in args.seeds.split(",")]
        walls = measure_comprisk(ns, seeds, args.p, args.ntree)
        args.walls_json.parent.mkdir(parents=True, exist_ok=True)
        args.walls_json.write_text(json.dumps({str(k): v for k, v in walls.items()}, indent=2))
        print(f"[walls] {args.walls_json} ({len(walls)} points)")

    if args.render:
        if not args.walls_json.exists():
            sys.exit(f"missing {args.walls_json}; run with --bench first")
        data = json.loads(args.walls_json.read_text())
        if "comprisk" in data and "rfsrc" in data:
            walls = {int(k): float(v) for k, v in data["comprisk"].items()}
            rfsrc = {int(k): float(v) for k, v in data["rfsrc"].items()}
        else:
            walls = {int(k): float(v) for k, v in data.items()}
            rfsrc = None
        render_svg(walls, args.svg_out, rfsrc_walls=rfsrc)


if __name__ == "__main__":
    main()
