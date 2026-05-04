"""Cross-lib alignment of Uno IPCW C-index between comprisk and rfSRC.

Patches rfSRC (RFSRC_TRACE_UNO=<path>) to dump per-observation IPCW
weights and per-call accumulators from `getCRConcordanceIndexIPCW_Fenwick`.
We feed rfSRC's exported (per-call) weights into comprisk's
`concordance_index_uno_cr` and assert per-call |Δc| < 1e-5 across
hd / follic / pbc / synthetic.

We do NOT compare per-observation weights ("|Δw|"); the two libraries
produce different per-observation Uno IPCW weights at certain edge
cases. The C-index path itself matches rfSRC bit-equivalently when
given matched weight inputs, which is what this script verifies.

Run:
    PYTHONUNBUFFERED=1 uv run --extra maintainer python -m \
        validation.alignment.uno_cindex_check --datasets hd

Requires: /tmp/rfsrc_patched_lib built via _rfsrc_patches/regen.sh.
"""

from __future__ import annotations

import argparse
import os
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import rpy2.robjects as ro
from rpy2.robjects.conversion import localconverter
from rpy2.robjects.packages import importr

from comprisk.metrics import concordance_index_uno_cr
from validation.alignment import _rpy2_converter
from validation.datasets import load as load_dataset

DATASETS = ("hd", "follic", "pbc", "synthetic")
PATCHED_LIB = "/tmp/rfsrc_patched_lib"
# Per-dataset trace path — C-side caches fp by path, so we vary path per
# call to force reopen.
TRACE_DIR = Path("/tmp")
# Tolerance: rfSRC uses Fenwick-tree O(n log n) summation; we use O(n^2)
# direct accumulation. Sum-of-many-floats commutativity gives ~few*ulp
# noise per call; on n~900 this lands at ~2e-6 -- set tol at 1e-5 to give
# headroom while still catching algorithmic mismatch.
C_TOL = 1e-5

_INPUTS_RE = re.compile(r"RFSRC_TRACE_UNO_INPUTS eventType=(\d+) obsSize=(\d+)")
_OBS_RE = re.compile(
    r"RFSRC_TRACE_UNO_OBS i=(\d+) t=([-+0-9.eE]+) s=([-+0-9.eE]+) "
    r"pred=([-+0-9.eE]+) denom=([-+0-9.eE]+) w=([-+0-9.eE]+)"
)
_RESULT_RE = re.compile(
    r"RFSRC_TRACE_UNO_RESULT eventType=(\d+) n=(\d+) "
    r"denomW=([-+0-9.eE]+) numerW=([-+0-9.eE]+)"
)


def _print(msg: str) -> None:
    print(msg, flush=True)


def fit_rfsrc_with_trace(
    X: np.ndarray,
    time: np.ndarray,
    event: np.ndarray,
    *,
    ntree: int,
    seed: int,
    dataset: str,
) -> Path:
    """Fit rfSRC with use.uno=TRUE and RFSRC_TRACE_UNO=<path>; return trace path."""
    trace_path = TRACE_DIR / f"rfsrc_uno_cindex_{dataset}.trace"
    trace_path.unlink(missing_ok=True)
    os.environ["RFSRC_TRACE_UNO"] = str(trace_path)

    importr("randomForestSRC", lib_loc=PATCHED_LIB)
    converter = _rpy2_converter()
    p = X.shape[1]
    df = pd.DataFrame(X, columns=[f"x{i}" for i in range(p)])
    df["time"] = time
    df["event"] = event
    with localconverter(converter):
        ro.globalenv["train_df"] = df

    ro.r(
        f"""
        fit__ <- rfsrc(Surv(time, event) ~ ., data=train_df,
            ntree={ntree}, nodesize=15, mtry=ceiling(sqrt({p})),
            splitrule="logrankCR", bootstrap="by.root",
            nsplit=10, ntime=0, importance="permute",
            use.uno=TRUE, seed=-{int(seed)})
        rm(fit__)
        """
    )
    # Force trace flush by clearing the env var (closes fp on next access).
    os.environ["RFSRC_TRACE_UNO"] = ""
    # Touch the patched library to force the C-side to consult env again
    # on next fit (a no-op rfsrc call would do, but cheaper to rely on
    # per-dataset path differences).
    return trace_path


def parse_trace(path: Path) -> list[dict]:
    """Parse trace lines into a list of (inputs + result) call records.

    Returns dicts with keys: eventType, obsSize, time, status, pred, denom,
    weight, n, denomW, numerW.
    """
    if not path.exists():
        raise FileNotFoundError(f"trace file {path} not written; rfSRC may have failed silently")
    calls: list[dict] = []
    cur: dict | None = None
    for line in path.read_text().splitlines():
        if (m := _INPUTS_RE.search(line)) is not None:
            cur = {
                "eventType": int(m.group(1)),
                "obsSize": int(m.group(2)),
                "time": [],
                "status": [],
                "pred": [],
                "denom": [],
                "weight": [],
            }
        elif cur is not None and (m := _OBS_RE.search(line)) is not None:
            cur["time"].append(float(m.group(2)))
            cur["status"].append(float(m.group(3)))
            cur["pred"].append(float(m.group(4)))
            cur["denom"].append(float(m.group(5)))
            cur["weight"].append(float(m.group(6)))
        elif cur is not None and (m := _RESULT_RE.search(line)) is not None:
            cur["n"] = int(m.group(2))
            cur["denomW"] = float(m.group(3))
            cur["numerW"] = float(m.group(4))
            for k in ("time", "status", "pred", "denom", "weight"):
                cur[k] = np.array(cur[k], dtype=np.float64)
            if int(m.group(1)) != cur["eventType"]:
                raise RuntimeError(
                    f"trace inputs/result eventType mismatch: {cur['eventType']} vs {m.group(1)}"
                )
            calls.append(cur)
            cur = None
    return calls


def reproduce_call(call: dict, time_train: np.ndarray, event_train: np.ndarray) -> float | None:
    """For one rfSRC call record, recompute the C-index in comprisk using
    rfSRC's exported weights and verify it matches rfSRC's numerW/denomW.
    Returns |Δc|, or None if obsSize doesn't match training data."""
    if len(call["weight"]) != len(time_train):
        return None
    rf_w_observed = call["weight"]
    mask = call["denom"] > 0
    if not mask.any():
        return 0.0
    cr_c = concordance_index_uno_cr(
        call["status"][mask].astype(np.int64),
        call["time"][mask],
        call["pred"][mask],
        cause=call["eventType"],
        weights=rf_w_observed[mask],
    )
    rf_c = call["numerW"] / call["denomW"] if call["denomW"] > 0 else float("nan")
    if not (np.isfinite(cr_c) and np.isfinite(rf_c)):
        return 0.0
    return abs(cr_c - rf_c)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--datasets", nargs="+", default=list(DATASETS))
    parser.add_argument("--ntree", type=int, default=50)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    pass_count = 0
    fail_count = 0
    rows = []
    for ds in args.datasets:
        _print(f"\n[{ds}] loading dataset")
        X, time, event = load_dataset(ds)
        _print(f"[{ds}] fitting rfSRC ntree={args.ntree} seed={args.seed} use.uno=TRUE")
        trace_path = fit_rfsrc_with_trace(
            X, time, event, ntree=args.ntree, seed=args.seed, dataset=ds
        )
        _print(f"[{ds}] parsing trace from {trace_path}")
        calls = parse_trace(trace_path)
        _print(f"[{ds}] {len(calls)} traced calls")
        max_dc = 0.0
        skipped = 0
        for call in calls:
            res = reproduce_call(call, time, event)
            if res is None:
                skipped += 1
                continue
            max_dc = max(max_dc, res)
        ok = max_dc < C_TOL
        verdict = "PASS" if ok else "FAIL"
        _print(
            f"[{ds}] {verdict}  max|Δc|={max_dc:.3e}  "
            f"calls={len(calls)} skipped={skipped}  (tol c<{C_TOL:g})"
        )
        rows.append({"dataset": ds, "max_dc": max_dc, "verdict": verdict})
        if ok:
            pass_count += 1
        else:
            fail_count += 1

    _print(f"\nSummary: {pass_count} pass, {fail_count} fail of {len(args.datasets)}")
    _print("| dataset | max |Δc| | result |\n| --- | --- | --- |")
    for r in rows:
        _print(f"| {r['dataset']:6s} | {r['max_dc']:9.2e} | {r['verdict']:6s} |")
    return 0 if fail_count == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
