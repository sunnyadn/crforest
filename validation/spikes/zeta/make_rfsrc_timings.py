"""Transcribe rfSRC measurements from /tmp/rfsrc_openmp.err into parquet.

One-shot utility. Source numbers are from the `/tmp/rfsrc_openmp_bench.R`
run captured on 2026-04-24 (see provenance file next to the output).
Re-run only if that .err file changes; otherwise the parquet is the
load-bearing artifact for compare.py.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
OUT = HERE / "timings" / "rfsrc_timings.parquet"

MEASUREMENTS = [
    {"n": 5000, "seed": 20260417, "fit_wall_s": 6.4, "peak_rss_mb": np.nan},
    {"n": 20000, "seed": 20260417, "fit_wall_s": 150.8, "peak_rss_mb": np.nan},
    {"n": 50000, "seed": 20260417, "fit_wall_s": 715.3, "peak_rss_mb": np.nan},
]


def main() -> None:
    OUT.parent.mkdir(exist_ok=True)
    pd.DataFrame(MEASUREMENTS).to_parquet(OUT)
    print(f"wrote {OUT} ({len(MEASUREMENTS)} rows)")


if __name__ == "__main__":
    main()
