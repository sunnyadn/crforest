"""λ.exp5 — paper-grade scaling bench (n axis × p axis).

Two-axis scan that gives a JSS-style tool paper its headline scaling table:

  n axis  (p=58 fixed, match real CHF feature width):
      n ∈ {100k, 250k, 500k, 1M}

  p axis  (n=250k fixed, conservative on GPU memory):
      p ∈ {58, 200, 500, 1000}

Both axes × {cpu, cuda} × 2 seeds. CUDA failures (OOM, kernel cap) skip
gracefully so a partial cell still ships. Synthetic 2-cause Weibull
DGP, ntree=100, mtry=ceil(sqrt(p)), nsplit=10.

rfSRC comparison: skipped here (peak RSS at n=75k 100trees = 14.7 GB
on win; extrapolated n=500k = ~98 GB exceeds WSL 24 GB cap and 64 GB
host budget. rfSRC at scale is a "OOM" data point, not a wall number.)

Output: /tmp/lambda_exp5_walls.parquet — one row per (axis, n, p, device,
seed) with wall + status (ok/oom/error) + GPU pool usage.

Run: ssh win 'export PATH=$HOME/.local/bin:$PATH && cd ~/comprisk && \\
       PYTHONUNBUFFERED=1 uv run --extra gpu --extra dev \\
       python -u validation/spikes/lambda/exp5_paper_scale_bench.py \\
       2>&1 | tee /tmp/lambda_exp5.log'
"""

from __future__ import annotations

import sys
import time as _time
import traceback
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).parents[1]))

from _lambda_helpers import make_synthetic

OUT = Path("/tmp/lambda_exp5_walls.parquet")
SEEDS = [42, 43]
NTREE = 100

N_AXIS_NS = [100_000, 250_000, 500_000, 1_000_000]
N_AXIS_P = 58
P_AXIS_PS = [58, 200, 500, 1000]
P_AXIS_N = 250_000


def gpu_mem_gb() -> tuple[float, float]:
    try:
        import cupy as cp

        pool = cp.get_default_memory_pool()
        return pool.used_bytes() / 1e9, pool.total_bytes() / 1e9
    except Exception:
        return 0.0, 0.0


def fit_one(X, t, e, *, device: str, seed: int) -> dict:
    from comprisk import CompetingRiskForest

    n_jobs = 1 if device == "cuda" else -1
    f = CompetingRiskForest(
        n_estimators=NTREE,
        n_jobs=n_jobs,
        random_state=seed,
        device=device,
    )
    t0 = _time.perf_counter()
    try:
        f.fit(X, t, e)
        if device == "cuda":
            import cupy as cp

            cp.cuda.runtime.deviceSynchronize()
        wall = _time.perf_counter() - t0
        used_gb, total_gb = gpu_mem_gb()
        return {
            "status": "ok",
            "wall": wall,
            "effective_device": f._effective_device_,
            "gpu_used_gb": used_gb,
            "gpu_total_gb": total_gb,
            "error": "",
        }
    except Exception as exc:  # OOM / kernel cap / etc — graceful skip
        wall = _time.perf_counter() - t0
        msg = f"{type(exc).__name__}: {str(exc)[:200]}"
        return {
            "status": "error",
            "wall": wall,
            "effective_device": "?",
            "gpu_used_gb": 0.0,
            "gpu_total_gb": 0.0,
            "error": msg,
        }


def warmup_cuda():
    """Warm cupy compile + memory pool. Skipped silently if cupy missing."""
    try:
        import cupy as cp  # noqa: F401

        Xw, tw, ew = make_synthetic(5000, 58, seed=0)
        print("[warmup] cuda 4-tree fit on n=5k p=58...", flush=True)
        fit_one(Xw, tw, ew, device="cuda", seed=0)
    except Exception as exc:
        print(f"[warmup] skipped: {exc}", flush=True)


def run_cells(label: str, cells: list[dict]) -> list[dict]:
    rows = []
    for cell in cells:
        n, p = cell["n"], cell["p"]
        # Generate once per cell, reuse for both devices + seeds.
        print(f"\n[{label}] generating synthetic n={n:,} p={p}...", flush=True)
        Xs, ts, es = make_synthetic(n, p, seed=20260417)
        for device in ["cpu", "cuda"]:
            for seed in SEEDS:
                print(
                    f"[{label}] cell n={n:,} p={p} device={device} seed={seed}...",
                    flush=True,
                )
                try:
                    r = fit_one(Xs, ts, es, device=device, seed=seed)
                except Exception:
                    traceback.print_exc()
                    r = {
                        "status": "error",
                        "wall": 0.0,
                        "effective_device": "?",
                        "gpu_used_gb": 0.0,
                        "gpu_total_gb": 0.0,
                        "error": "outer-exception",
                    }
                tag = (
                    f"wall={r['wall']:.2f}s"
                    if r["status"] == "ok"
                    else f"ERROR ({r['error'][:80]})"
                )
                pool_tag = (
                    f"  pool used={r['gpu_used_gb']:.1f}GB / {r['gpu_total_gb']:.1f}GB"
                    if device == "cuda" and r["status"] == "ok"
                    else ""
                )
                print(f"  {tag}{pool_tag}", flush=True)
                rows.append(
                    {
                        "axis": label,
                        "n": n,
                        "p": p,
                        "device": device,
                        "seed": seed,
                        **r,
                    }
                )
        # Free the dataset before next cell so memory budget is per-cell.
        del Xs, ts, es
    return rows


def main() -> None:
    warmup_cuda()

    n_cells = [{"n": n, "p": N_AXIS_P} for n in N_AXIS_NS]
    p_cells = [{"n": P_AXIS_N, "p": p} for p in P_AXIS_PS if p != N_AXIS_P]

    rows = []
    rows.extend(run_cells("n_axis", n_cells))
    rows.extend(run_cells("p_axis", p_cells))

    df = pd.DataFrame(rows)
    df.to_parquet(OUT)
    print(f"\n[dump] {OUT} ({len(df)} rows)", flush=True)

    print("\n=== Summary by (axis, n, p, device): mean wall + std + status ===\n", flush=True)
    ok_only = df[df["status"] == "ok"]
    summary = ok_only.groupby(["axis", "n", "p", "device"])["wall"].agg(["mean", "std"]).round(2)
    print(summary.to_string(), flush=True)

    err = df[df["status"] != "ok"]
    if len(err):
        print(f"\n=== {len(err)} failed cells ===", flush=True)
        for _, r in err.iterrows():
            print(
                f"  axis={r['axis']} n={r['n']} p={r['p']} device={r['device']} "
                f"seed={r['seed']} → {r['error'][:120]}",
                flush=True,
            )


if __name__ == "__main__":
    main()
