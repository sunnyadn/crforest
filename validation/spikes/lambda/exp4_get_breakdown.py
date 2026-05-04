"""λ.exp4 — per-call .get() breakdown in build_flat_tree_gpu.

Plan 3 prioritization data: λ.exp1 measured cumulative cupy.get() = 187ms
on real CHF single-tree fit. To pick between Day 4 v2 (active+FlatTree on
device, kills out_feat/out_bin/mids gets) vs Day 5-6 (cand_mask + xb_view
on GPU, kills sample_perm get), we need each .get()'s individual wall.

Two parts:

  Part 1 — cupy .get() micro-benchmark by array size. Establishes the
  per-call constant cost vs the data-dependent transfer cost. Useful as
  an absolute reference: if a single .get() takes 3ms regardless of size,
  the bottleneck is sync barriers, not bandwidth.

  Part 2 — instrumented build_flat_tree_gpu fit. Monkey-patches
  cupy.ndarray.get + cupy.asnumpy to log call site + size + wall. Single
  tree fit on real CHF, summary table per .get() call site.

Run: ssh win 'export PATH=$HOME/.local/bin:$PATH && cd ~/comprisk && \\
       PYTHONUNBUFFERED=1 uv run --extra gpu --extra dev \\
       python -u validation/spikes/lambda/exp4_get_breakdown.py'
"""

from __future__ import annotations

import sys
import time as _time
import traceback
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parents[1]))

from _lambda_helpers import load_chf


def part1_micro_benchmark():
    import cupy as cp

    print("\n=== Part 1: cupy .get() micro-benchmark by size ===\n", flush=True)
    print(f"{'size':>10}  {'bytes':>12}  {'wall_ms':>10}  {'gb_per_s':>10}", flush=True)
    sizes = [1, 16, 256, 4096, 65_536, 1_048_576, 16_777_216]
    rows = []
    for n in sizes:
        a_d = cp.arange(n, dtype=cp.int32)
        cp.cuda.runtime.deviceSynchronize()
        # Warm.
        _ = a_d.get()
        # Timed: median of 5 runs.
        walls = []
        for _ in range(5):
            cp.cuda.runtime.deviceSynchronize()
            t0 = _time.perf_counter()
            _ = a_d.get()
            walls.append(_time.perf_counter() - t0)
        wall_ms = float(np.median(walls)) * 1000.0
        bytes_ = n * 4
        gb_per_s = bytes_ / 1e9 / max(wall_ms / 1000, 1e-9)
        rows.append({"size": n, "bytes": bytes_, "wall_ms": wall_ms, "gb_per_s": gb_per_s})
        print(f"{n:>10}  {bytes_:>12}  {wall_ms:>10.3f}  {gb_per_s:>10.2f}", flush=True)
    return pd.DataFrame(rows)


def part2_instrumented_fit():
    import cupy as cp

    print("\n\n=== Part 2: instrumented build_flat_tree_gpu ===\n", flush=True)

    # Monkey-patch cp.asnumpy + cp.ndarray.get with a logger.
    log = []

    orig_asnumpy = cp.asnumpy
    orig_get = cp.ndarray.get

    def _log_call(api: str, x):
        # Two-frame backtrace to pin call site (skip patch wrapper + caller).
        stack = traceback.extract_stack(limit=3)
        site = stack[-3] if len(stack) >= 3 else stack[-1]
        cp.cuda.runtime.deviceSynchronize()
        t0 = _time.perf_counter()
        out = orig_asnumpy(x) if api == "asnumpy" else orig_get(x)
        cp.cuda.runtime.deviceSynchronize()
        wall_ms = (_time.perf_counter() - t0) * 1000.0
        size = int(x.size) if hasattr(x, "size") else -1
        log.append(
            {
                "api": api,
                "wall_ms": wall_ms,
                "size": size,
                "site": f"{Path(site.filename).name}:{site.lineno}",
                "code": site.line,
            }
        )
        return out

    def _patched_asnumpy(x, *a, **kw):
        return _log_call("asnumpy", x)

    def _patched_get(self, *a, **kw):
        return _log_call("get", self)

    cp.asnumpy = _patched_asnumpy
    cp.ndarray.get = _patched_get

    X_tr, t_tr, e_tr, p = load_chf()
    print(f"[load] real CHF train n={len(X_tr):,} p={p}", flush=True)

    from comprisk import CompetingRiskForest

    print("[warmup] cuda compile + 4-tree fit on 5k slice...", flush=True)
    log.clear()
    CompetingRiskForest(n_estimators=4, n_jobs=1, random_state=0, device="cuda").fit(
        X_tr[:5000], t_tr[:5000], e_tr[:5000]
    )

    print(f"[warmup] {len(log)} .get/asnumpy calls during warm — discarding", flush=True)
    log.clear()

    print("[fit] single-tree on full real CHF (n_jobs=1 ntree=1)...", flush=True)
    t0 = _time.perf_counter()
    CompetingRiskForest(n_estimators=1, n_jobs=1, random_state=0, device="cuda").fit(
        X_tr, t_tr, e_tr
    )
    wall = _time.perf_counter() - t0
    print(f"[fit] WALL = {wall * 1000:.1f}ms", flush=True)
    print(f"[fit] {len(log)} .get/asnumpy calls during fit", flush=True)

    df_log = pd.DataFrame(log)
    if df_log.empty:
        print("no calls captured", flush=True)
        return df_log
    print("\n--- per-call-site summary ---\n", flush=True)
    by_site = (
        df_log.groupby("site")
        .agg(
            n_calls=("wall_ms", "size"),
            sum_ms=("wall_ms", "sum"),
            mean_ms=("wall_ms", "mean"),
            sum_size=("size", "sum"),
        )
        .sort_values("sum_ms", ascending=False)
    )
    print(by_site.to_string(), flush=True)
    print(
        f"\n[totals] sum .get/asnumpy wall = {df_log['wall_ms'].sum():.1f}ms "
        f"({100 * df_log['wall_ms'].sum() / (wall * 1000):.1f}% of single-tree wall)",
        flush=True,
    )

    # Bucket by category for Plan 3 prioritization.
    site_to_category = defaultdict(lambda: "other")

    def _categorize(site: str, code: str | None) -> str:
        c = (code or "").lower()
        if "out_feat" in c or "out_bin" in c:
            return "best_split_outputs"
        if "p_mids_d" in c or "mids_h" in c:
            return "partition_mids"
        if "sample_perm" in c:
            return "sample_perm"
        if "leaf_table" in c or "leaf_event" in c or "leaf_at_risk" in c:
            return "tree_completion_aj"
        if "out_feature" in c or "out_split" in c or "out_left" in c or "out_right" in c:
            return "flattree_final"
        return "other"

    df_log["category"] = df_log.apply(lambda r: _categorize(r["site"], r["code"]), axis=1)
    print("\n--- by Plan 3 category ---\n", flush=True)
    by_cat = (
        df_log.groupby("category")
        .agg(n=("wall_ms", "size"), sum_ms=("wall_ms", "sum"), mean_ms=("wall_ms", "mean"))
        .sort_values("sum_ms", ascending=False)
    )
    by_cat["pct_of_wall"] = 100 * by_cat["sum_ms"] / (wall * 1000)
    print(by_cat.to_string(), flush=True)

    return df_log


def main() -> None:
    part1_micro_benchmark()
    part2_instrumented_fit()


if __name__ == "__main__":
    main()
