"""Compute paired-seed C-index agreement from cached equivalence-gate cells.

The equivalence gate's cells persist (cif_cr, cif_rf) on the reference grid
but not (test_time, test_event). This driver reloads splits, computes
C-index for comprisk and rfSRC risk scores per cell (risk = CIF at last
ref_grid time), and reports cross-lib delta + within-lib paired variance
per dataset. Noise-floor advisory: cross-lib |DeltaC| should be <=
max(within-lib pair |DeltaC|).

Runs against existing ``validation/alignment/_cache/<ds>_s<seed>.parquet``
cells -- no refits. Pairs seeds (0,1), (2,3), ... for within-lib.

Run:
    uv run --extra maintainer python -m validation.alignment.cindex_from_cache \
        --datasets hd follic pbc synthetic --seeds 20
"""

from __future__ import annotations

import argparse
import datetime as _dt
from pathlib import Path

import numpy as np
import pandas as pd

from comprisk.metrics import concordance_index_cr
from validation.alignment.equivalence_gate import load_cell
from validation.datasets import load as load_dataset
from validation.splits import _SPLITS_DIR

CACHE_DIR = Path("validation/alignment/_cache")


def _cell_path(dataset: str, seed: int) -> Path:
    return CACHE_DIR / f"{dataset}_s{seed}.parquet"


def _test_outcomes(dataset: str, seed: int) -> tuple[np.ndarray, np.ndarray]:
    _, time, event = load_dataset(dataset)
    splits_df = pd.read_parquet(_SPLITS_DIR / f"{dataset}.parquet")
    row = splits_df[splits_df["seed"] == seed]
    test_idx = np.sort(row.loc[row["fold"] == "test", "sample_id"].to_numpy(np.int64))
    return time[test_idx], event[test_idx]


def compute_per_seed(dataset: str, seed: int, cause: int = 1) -> dict | None:
    path = _cell_path(dataset, seed)
    if not path.exists():
        return None
    cell = load_cell(path)
    test_time, test_event = _test_outcomes(dataset, seed)
    # Risk = CIF at last ref_grid time (same convention as equivalence_gate's
    # "risk" metric). concordance_index_cr wants cause-specific risk.
    c_cr = concordance_index_cr(test_event, test_time, cell["risk_cr"], cause=cause)
    c_rf = concordance_index_cr(test_event, test_time, cell["risk_rf"], cause=cause)
    return {"seed": seed, "c_cr": float(c_cr), "c_rf": float(c_rf)}


def aggregate(dataset: str, seeds: list[int]) -> dict:
    cells = [compute_per_seed(dataset, s) for s in seeds]
    cells = [c for c in cells if c is not None]
    if len(cells) == 0:
        return {"dataset": dataset, "n_cells": 0}
    cells.sort(key=lambda c: c["seed"])
    c_cr = np.array([c["c_cr"] for c in cells])
    c_rf = np.array([c["c_rf"] for c in cells])
    cross = np.abs(c_cr - c_rf)
    # Within-lib paired-seed variance: |c_cr[0]-c_cr[1]|, |c_cr[2]-c_cr[3]|, ...
    n_pairs = len(cells) // 2
    if n_pairs >= 1:
        within_cr = np.abs(c_cr[0::2][:n_pairs] - c_cr[1::2][:n_pairs])
        within_rf = np.abs(c_rf[0::2][:n_pairs] - c_rf[1::2][:n_pairs])
    else:
        within_cr = within_rf = np.array([0.0])
    return {
        "dataset": dataset,
        "n_cells": len(cells),
        "c_cr_mean": float(c_cr.mean()),
        "c_rf_mean": float(c_rf.mean()),
        "cross_max": float(cross.max()),
        "cross_mean": float(cross.mean()),
        "within_cr_max": float(within_cr.max()),
        "within_rf_max": float(within_rf.max()),
        "noise_floor_pass": bool(cross.max() <= max(within_cr.max(), within_rf.max())),
    }


def _report(per_dataset: dict[str, dict], out_path: Path) -> None:
    lines = [
        "# Paired-seed C-index agreement (comprisk vs rfSRC)",
        "",
        f"Timestamp: {_dt.datetime.now().isoformat(timespec='seconds')}",
        "",
        "Risk score = CIF at last reference-grid time (same convention as the",
        "equivalence_gate risk metric). C-index computed via",
        "``comprisk.metrics.concordance_index_cr`` on (test_event, test_time,",
        "risk, cause=1). Cells loaded from",
        "``validation/alignment/_cache/<ds>_s<seed>.parquet`` -- no refits.",
        "",
        "**Noise-floor advisory**: cross-lib ``|DeltaC|`` should be <= the",
        "max paired-seed within-lib ``|DeltaC|`` for each library. If yes, the",
        "two libraries agree within each one's own seed-to-seed variance.",
        "",
        "| dataset | n_cells | c_cr_mean | c_rf_mean | cross_max | cross_mean "
        "| within_cr_max | within_rf_max | noise_floor |",
        "|---|---|---|---|---|---|---|---|---|",
    ]
    for ds, d in per_dataset.items():
        if d.get("n_cells", 0) == 0:
            lines.append(f"| {ds} | 0 | — | — | — | — | — | — | N/A |")
            continue
        lines.append(
            f"| {ds} | {d['n_cells']} | {d['c_cr_mean']:.4f} | {d['c_rf_mean']:.4f} "
            f"| {d['cross_max']:.4f} | {d['cross_mean']:.4f} "
            f"| {d['within_cr_max']:.4f} | {d['within_rf_max']:.4f} "
            f"| {'PASS' if d['noise_floor_pass'] else 'FAIL'} |"
        )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(lines))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--datasets", nargs="*", default=["pbc", "follic", "hd", "synthetic"])
    parser.add_argument("--seeds", type=int, default=20)
    parser.add_argument("--out", default="validation/reports")
    args = parser.parse_args(argv)

    seeds = list(range(args.seeds))
    per_dataset: dict[str, dict] = {}
    for ds in args.datasets:
        per_dataset[ds] = aggregate(ds, seeds)
        d = per_dataset[ds]
        if d.get("n_cells", 0) == 0:
            print(f"[cindex {ds}] no cached cells found under {CACHE_DIR}", flush=True)
            continue
        print(
            f"[cindex {ds}] n={d['n_cells']} "
            f"c_cr={d['c_cr_mean']:.4f} c_rf={d['c_rf_mean']:.4f} "
            f"cross_max={d['cross_max']:.4f} within_cr_max={d['within_cr_max']:.4f} "
            f"within_rf_max={d['within_rf_max']:.4f} "
            f"noise_floor={'PASS' if d['noise_floor_pass'] else 'FAIL'}",
            flush=True,
        )

    ts = _dt.datetime.now().isoformat(timespec="seconds").replace(":", "-")
    out_path = Path(args.out) / f"cindex_{ts}.md"
    _report(per_dataset, out_path)
    print(f"wrote {out_path}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
