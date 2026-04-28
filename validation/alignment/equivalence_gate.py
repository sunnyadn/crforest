"""Equivalence-gate audit: crforest vs rfSRC predictive equivalence.

Purpose (A) from the 2026-04-24 brainstorm: maintainer-invoked audit, not CI.
Gates predictive equivalence (per-sample risk + pointwise CIF) using noise-floor
tolerance + absolute hard-cap on four paired-seed datasets.

Spec: docs/superpowers/specs/2026-04-24-equivalence-gate-design.md
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from validation.alignment.compare_cif import _fit_crforest, _fit_rfsrc
from validation.alignment.ibs import compute_ibs
from validation.datasets import load as load_dataset
from validation.splits import _SPLITS_DIR


def build_reference_grid(time: np.ndarray, event: np.ndarray) -> np.ndarray:
    """Sorted unique event times from the full dataset (event > 0 rows only).

    Fixed per dataset — every CIF comparison in the audit is evaluated on this
    grid via step-function semantics, so within-lib and cross-lib gaps are
    commensurate regardless of which seed's training fold a fit saw.
    """
    mask = event > 0
    if not mask.any():
        raise ValueError("dataset has no event times (event > 0); cannot build reference grid")
    return np.unique(time[mask]).astype(np.float64)


def eval_on_ref_grid(
    cif_native: np.ndarray,
    native_grid: np.ndarray,
    ref_grid: np.ndarray,
) -> np.ndarray:
    """Evaluate a step-function CIF at reference-grid times.

    Step-function convention (matches ``_pointwise_cif_gap`` in compare_cif.py):
      cif(t) = cif_native[:, idx] where idx = clip(searchsorted(grid, t, "right") - 1, 0, n-1).

    Values of t below ``native_grid[0]`` clamp to ``native_grid[0]``'s CIF;
    values above ``native_grid[-1]`` clamp to ``native_grid[-1]``'s (flat
    extrapolation, standard AJ convention).
    """
    idx = np.clip(np.searchsorted(native_grid, ref_grid, side="right") - 1, 0, len(native_grid) - 1)
    return cif_native[:, idx]


def _p95_abs(x: np.ndarray) -> float:
    """p95 of |x| across all elements."""
    return float(np.percentile(np.abs(x), 95))


# Companion quantile-dominance grid. The gate itself runs on p95 (index 3); the
# other quantiles give the full gap-CDF shape that distinguishes bulk-agreement
# datasets from concentrated-tail datasets (see project_equivalence_hd_tail_diagnostic).
QUANTILE_GRID: tuple[float, ...] = (0.50, 0.75, 0.90, 0.95, 0.99)


def _quantiles_abs(x: np.ndarray, qs: tuple[float, ...] = QUANTILE_GRID) -> dict[float, float]:
    """Multi-quantile version of ``_p95_abs`` — one np.percentile call for all qs."""
    vals = np.percentile(np.abs(x), [q * 100 for q in qs])
    return {q: float(v) for q, v in zip(qs, vals, strict=True)}


def _aggregate_ibs(cells: list[dict]) -> dict:
    """IBS is a scalar per (lib, seed). Within-lib noise floor pairs seeds
    (0,1), (2,3), ... and takes max |IBS_L[a] - IBS_L[b]| across pairs. Cross
    gap is |IBS_cr[seed] - IBS_rf[seed]| per seed.

    At n=seeds ≤ 20, "p95" of the seed-level distribution is essentially
    max (for n=20 p95 corresponds to rank 18-19). We report both p95 and
    max explicitly so readers can judge.
    """
    ordered = sorted(cells, key=lambda c: c["seed"])
    ibs_cr = np.array([c["ibs_cr"] for c in ordered])
    ibs_rf = np.array([c["ibs_rf"] for c in ordered])
    cross = np.abs(ibs_cr - ibs_rf)
    within_cr = np.abs(ibs_cr[0::2] - ibs_cr[1::2])
    within_rf = np.abs(ibs_rf[0::2] - ibs_rf[1::2])

    return {
        "cross_p95_ibs": float(np.percentile(cross, 95)),
        "cross_max_ibs": float(cross.max()),
        "cross_median_ibs": float(np.median(cross)),
        "within_cr_p95_ibs": float(np.percentile(within_cr, 95)),
        "within_cr_max_ibs": float(within_cr.max()),
        "within_rf_p95_ibs": float(np.percentile(within_rf, 95)),
        "within_rf_max_ibs": float(within_rf.max()),
        "mean_ibs_cr": float(ibs_cr.mean()),
        "mean_ibs_rf": float(ibs_rf.mean()),
    }


def aggregate_dataset(cells: list[dict]) -> dict:
    """Per-dataset aggregation of within-lib and cross-lib gaps.

    ``cells`` is a list of per-seed dicts with keys: seed, cif_cr, cif_rf,
    risk_cr, risk_rf. Returns the aggregated numbers that feed the tolerance
    rules in apply_tolerance().

    Within-lib noise floor (10 paired gaps from 20 seeds, max across pairs):
      within_<lib>_p95_risk = max over pairs of p95_samples(|risk_L[s_a] - risk_L[s_b]|)
      within_<lib>_p95_cif  = max over pairs of p95_{samples,time}(|cif_L[s_a] - cif_L[s_b]|)

    Cross-lib gap (median across seeds):
      cross_p95_risk = median over seeds of p95_samples(|risk_cr[s] - risk_rf[s]|)
      cross_p95_cif  = median over seeds of p95_{samples,time}(|cif_cr[s] - cif_rf[s]|)

    Also reported:
      cross_max_<metric>                   = max over seeds of per-seed max|gap|
      cross_p95_max_over_seeds_<metric>    = max over seeds of per-seed p95|gap|
    """
    n = len(cells)
    if n == 0 or n % 2 != 0:
        raise ValueError(f"aggregate_dataset expects an even number of seeds; got {n}")

    # Order by seed so pairing (0,1), (2,3), ... is deterministic regardless of input order.
    ordered = sorted(cells, key=lambda c: c["seed"])

    def within(lib: str, key: str) -> float:
        worst = 0.0
        for i in range(0, n, 2):
            a, b = ordered[i], ordered[i + 1]
            gap = a[f"{key}_{lib}"] - b[f"{key}_{lib}"]
            worst = max(worst, _p95_abs(gap))
        return worst

    per_seed_p95_risk = np.array([_p95_abs(c["risk_cr"] - c["risk_rf"]) for c in ordered])
    per_seed_p95_cif = np.array([_p95_abs(c["cif_cr"] - c["cif_rf"]) for c in ordered])
    per_seed_max_risk = np.array(
        [float(np.max(np.abs(c["risk_cr"] - c["risk_rf"]))) for c in ordered]
    )
    per_seed_max_cif = np.array([float(np.max(np.abs(c["cif_cr"] - c["cif_rf"]))) for c in ordered])

    def cross_q(key_cr: str, key_rf: str) -> dict[float, float]:
        per_seed = [_quantiles_abs(c[key_cr] - c[key_rf]) for c in ordered]
        return {q: float(np.median([d[q] for d in per_seed])) for q in QUANTILE_GRID}

    def within_q(lib: str, key: str) -> dict[float, float]:
        worst = {q: 0.0 for q in QUANTILE_GRID}
        for i in range(0, n, 2):
            a, b = ordered[i], ordered[i + 1]
            pair = _quantiles_abs(a[f"{key}_{lib}"] - b[f"{key}_{lib}"])
            for q in QUANTILE_GRID:
                worst[q] = max(worst[q], pair[q])
        return worst

    return {
        "within_cr_p95_risk": within("cr", "risk"),
        "within_rf_p95_risk": within("rf", "risk"),
        "within_cr_p95_cif": within("cr", "cif"),
        "within_rf_p95_cif": within("rf", "cif"),
        "cross_p95_risk": float(np.median(per_seed_p95_risk)),
        "cross_p95_cif": float(np.median(per_seed_p95_cif)),
        "cross_max_risk": float(np.max(per_seed_max_risk)),
        "cross_max_cif": float(np.max(per_seed_max_cif)),
        "cross_p95_max_over_seeds_risk": float(np.max(per_seed_p95_risk)),
        "cross_p95_max_over_seeds_cif": float(np.max(per_seed_p95_cif)),
        "quantiles": {
            "cross_risk": cross_q("risk_cr", "risk_rf"),
            "cross_cif": cross_q("cif_cr", "cif_rf"),
            "within_cr_risk": within_q("cr", "risk"),
            "within_rf_risk": within_q("rf", "risk"),
            "within_cr_cif": within_q("cr", "cif"),
            "within_rf_cif": within_q("rf", "cif"),
        },
        "n_seeds": n,
    }


HARD_CAP_DEFAULT = 0.05


def apply_tolerance(agg: dict, hard_cap: float = HARD_CAP_DEFAULT) -> dict:
    """Apply the equivalence-gate tolerance rules.

    **Gate contract (overall_pass)**: noise-floor on cross-lib p95 (risk + cif).
    A dataset passes when cross-lib p95 is within the larger of the two within-
    lib seed-pair p95s. This is the scientifically principled criterion: it
    says the two libraries agree to within each library's own seed-to-seed
    variation, which is the natural equivalence scale for a stochastic method.

    **Advisory (hard_cap_pass_*)**: cross_p95 ≤ hard_cap (default 0.05). Kept
    as reported diagnostic only -- not part of overall_pass. The original
    hard-cap semantics (pre-2026-04-24) were heuristic and, as characterized
    by the tiebreak + Z-cell spikes, fail on small-n datasets (hd, follic)
    for reasons that are implementation-level random-choice independence
    between the two libs, NOT algorithmic divergence. See
    ``docs/equivalence-vs-rfsrc.md`` for the decomposition.

    IBS is noise-floor only (no absolute threshold exists): |ΔIBS| has no
    natural cap independent of dataset event rate / scale. IBS noise-floor
    pass is reported but not part of overall_pass (advisory), since the
    primary contract is pointwise CIF + risk equivalence.
    """
    result = {}
    nf_all = True
    hc_all = True
    for metric in ("risk", "cif"):
        floor = max(agg[f"within_cr_p95_{metric}"], agg[f"within_rf_p95_{metric}"])
        cross = agg[f"cross_p95_{metric}"]
        nf_pass = cross <= floor
        hc_pass = cross <= hard_cap
        result[f"noise_floor_pass_{metric}"] = nf_pass
        result[f"hard_cap_pass_{metric}"] = hc_pass
        nf_all = nf_all and nf_pass
        hc_all = hc_all and hc_pass
    if "cross_p95_ibs" in agg:
        ibs_floor = max(agg["within_cr_p95_ibs"], agg["within_rf_p95_ibs"])
        result["noise_floor_pass_ibs"] = agg["cross_p95_ibs"] <= ibs_floor
    result["overall_pass"] = nf_all
    result["hard_cap_pass_overall"] = hc_all
    return result


def persist_cell(
    *,
    path: Path,
    dataset: str,
    seed: int,
    cif_cr: np.ndarray,
    cif_rf: np.ndarray,
    ref_grid: np.ndarray,
    cr_native_grid: np.ndarray,
    rf_native_grid: np.ndarray,
    n_train: int,
    n_test: int,
    commit_sha: str,
) -> None:
    """Persist one (dataset, seed) cell to parquet.

    Long-form columns: sample_id, t_ref, cif_cr, cif_rf (one row per (sample, t)).
    Scalars and native grids are stored in parquet schema metadata.
    """
    n_test_rows, n_ref = cif_cr.shape
    if cif_rf.shape != (n_test_rows, n_ref):
        raise ValueError(f"cif_cr/cif_rf shape mismatch: {cif_cr.shape} vs {cif_rf.shape}")
    if len(ref_grid) != n_ref:
        raise ValueError(f"ref_grid length {len(ref_grid)} != cif width {n_ref}")

    sample_id = np.repeat(np.arange(n_test_rows), n_ref)
    t_ref = np.tile(ref_grid, n_test_rows)
    cif_cr_flat = cif_cr.reshape(-1)
    cif_rf_flat = cif_rf.reshape(-1)

    table = pa.table(
        {
            "sample_id": sample_id.astype(np.int32),
            "t_ref": t_ref.astype(np.float64),
            "cif_cr": cif_cr_flat.astype(np.float64),
            "cif_rf": cif_rf_flat.astype(np.float64),
        }
    )
    metadata = {
        b"dataset": dataset.encode(),
        b"seed": str(seed).encode(),
        b"n_train": str(n_train).encode(),
        b"n_test": str(n_test).encode(),
        b"commit_sha": commit_sha.encode(),
        b"cr_native_grid": np.asarray(cr_native_grid, dtype=np.float64).tobytes(),
        b"rf_native_grid": np.asarray(rf_native_grid, dtype=np.float64).tobytes(),
    }
    table = table.replace_schema_metadata({**(table.schema.metadata or {}), **metadata})
    path.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(table, path)


def load_cell(path: Path) -> dict:
    """Load one (dataset, seed) cell. Returns dict with cif_cr, cif_rf, risk_cr,
    risk_rf, ref_grid, cr_native_grid, rf_native_grid, plus scalar metadata
    (dataset, seed, n_train, n_test, commit_sha)."""
    table = pq.read_table(path)
    meta = table.schema.metadata or {}

    df = table.to_pandas()
    n_test = int(meta[b"n_test"].decode())
    ref_grid_len = int(df["sample_id"].value_counts().iloc[0])
    n_rows = df["sample_id"].nunique()
    cif_cr = df["cif_cr"].to_numpy().reshape(n_rows, ref_grid_len)
    cif_rf = df["cif_rf"].to_numpy().reshape(n_rows, ref_grid_len)
    ref_grid = df[df["sample_id"] == 0]["t_ref"].to_numpy()

    return {
        "cif_cr": cif_cr,
        "cif_rf": cif_rf,
        "risk_cr": cif_cr[:, -1].copy(),
        "risk_rf": cif_rf[:, -1].copy(),
        "ref_grid": ref_grid,
        "cr_native_grid": np.frombuffer(meta[b"cr_native_grid"], dtype=np.float64),
        "rf_native_grid": np.frombuffer(meta[b"rf_native_grid"], dtype=np.float64),
        "dataset": meta[b"dataset"].decode(),
        "seed": int(meta[b"seed"].decode()),
        "n_train": int(meta[b"n_train"].decode()),
        "n_test": n_test,
        "commit_sha": meta[b"commit_sha"].decode(),
    }


def fit_and_capture(
    *,
    dataset: str,
    seed: int,
    cache_dir: Path,
    commit_sha: str,
    force_refit: bool = False,
    cause: int = 1,
    n_estimators: int = 500,
    time_grid: int = 200,
    min_samples_leaf: int = 1,
    min_samples_split: int = 30,
    cr_nsplit: int | None = None,
    rf_reuse_cache: Path | None = None,
) -> Path:
    """Fit crforest + rfSRC on (dataset, seed), eval on reference grid, persist.

    Returns the cache path. Idempotent: if the cache file exists and force_refit
    is False, skips the fit.

    ``cr_nsplit`` threads a crforest nsplit override into the fit. When None
    (default), crforest resolves to its own default (nsplit=10 in default mode)
    and the cache key is the legacy ``<ds>_s<seed>.parquet`` so existing cached
    cells from the main equivalence gate stay hot. Non-None values append
    ``_nsplit<n>`` to the key.

    ``rf_reuse_cache`` points to an existing cell file whose rfSRC CIF +
    native grid should be reused instead of refitting rfSRC. rfSRC is
    deterministic given (data, seed) so this is a pure compute saver for
    the nsplit convergence sweep where rfSRC is fixed but crforest varies.
    """
    suffix = "" if cr_nsplit is None else f"_nsplit{cr_nsplit}"
    path = cache_dir / f"{dataset}_s{seed}{suffix}.parquet"
    if path.exists() and not force_refit:
        print(f"[lib=both ds={dataset} seed={seed}{suffix}] cache_hit path={path}", flush=True)
        return path

    X, time, event = load_dataset(dataset)

    # Reference grid from the full dataset (pre-split).
    ref_grid = build_reference_grid(time, event)

    # Train/test split from the paired-seed splits parquet.
    splits_df = pd.read_parquet(_SPLITS_DIR / f"{dataset}.parquet")
    row = splits_df[splits_df["seed"] == seed]
    if row.empty:
        raise RuntimeError(f"seed {seed} not in splits for dataset {dataset}")
    train_idx = np.sort(row.loc[row["fold"] == "train", "sample_id"].to_numpy(np.int64))
    test_idx = np.sort(row.loc[row["fold"] == "test", "sample_id"].to_numpy(np.int64))

    print(
        f"[lib=crforest ds={dataset} seed={seed}{suffix}] fit_start n_train={len(train_idx)}",
        flush=True,
    )
    cr = _fit_crforest(
        X[train_idx],
        time[train_idx],
        event[train_idx],
        X[test_idx],
        seed=seed,
        n_estimators=n_estimators,
        time_grid=time_grid,
        min_samples_leaf=min_samples_leaf,
        min_samples_split=min_samples_split,
        nsplit=cr_nsplit,
    )
    print(f"[lib=crforest ds={dataset} seed={seed}{suffix}] fit_done", flush=True)

    cause_idx = cause - 1
    cr_cif_native = cr["cif"][:, :, cause_idx]  # (n_test, n_native_cr)
    cif_cr_ref = eval_on_ref_grid(cr_cif_native, cr["time_grid"], ref_grid)

    if rf_reuse_cache is not None and rf_reuse_cache.exists():
        print(
            f"[lib=rfSRC ds={dataset} seed={seed}{suffix}] reuse_from={rf_reuse_cache}",
            flush=True,
        )
        reused = load_cell(rf_reuse_cache)
        cif_rf_ref = reused["cif_rf"]
        rf_native_grid = reused["rf_native_grid"]
    else:
        print(f"[lib=rfSRC ds={dataset} seed={seed}{suffix}] fit_start", flush=True)
        rf = _fit_rfsrc(
            X[train_idx],
            time[train_idx],
            event[train_idx],
            X[test_idx],
            seed=seed,
            n_estimators=n_estimators,
            nodesize=15,
            nsplit=10,
        )
        print(f"[lib=rfSRC ds={dataset} seed={seed}{suffix}] fit_done", flush=True)
        rf_cif_native = rf["cif"][:, :, cause_idx]
        rf_native_grid = rf["time_grid"]
        cif_rf_ref = eval_on_ref_grid(rf_cif_native, rf_native_grid, ref_grid)

    persist_cell(
        path=path,
        dataset=dataset,
        seed=seed,
        cif_cr=cif_cr_ref,
        cif_rf=cif_rf_ref,
        ref_grid=ref_grid,
        cr_native_grid=cr["time_grid"],
        rf_native_grid=rf_native_grid,
        n_train=len(train_idx),
        n_test=len(test_idx),
        commit_sha=commit_sha,
    )
    return path


def capture_tree_stats(forest_cr, rfsrc_native_array_df) -> dict:
    """Summarize tree structure for both libs from one representative fit.

    Parameters
    ----------
    forest_cr :
        A crforest ``CompetingRiskForest`` that has been fit.
    rfsrc_native_array_df :
        The pandas DataFrame form of ``rfSRC fit$forest$nativeArray`` — caller
        passes this from R via rpy2 (or ``None`` if rfSRC info is unavailable).
    """
    from validation.alignment.compare_cif import _cr_leaf_sizes, _cr_tree_stats

    cr_leaves = []
    cr_depths = []
    for tree in forest_cr.trees_:
        nl, md, _ = _cr_tree_stats(tree)
        cr_leaves.append(nl)
        cr_depths.append(md)
    # leaf-size needs the binned X_train cached on the forest; when it is not
    # (older fit-path), leave the descriptive percentiles as nan rather than
    # silently recomputing — gate does not depend on them (spec §5).
    if hasattr(forest_cr, "_X_train_binned_"):
        cr_leaf_sizes_list = _cr_leaf_sizes(forest_cr, forest_cr._X_train_binned_)
    else:
        cr_leaf_sizes_list = []

    out = {
        "crforest": {
            "mean_leaves": float(np.mean(cr_leaves)),
            "mean_depth": float(np.mean(cr_depths)),
            "leaf_p5": float(np.percentile(cr_leaf_sizes_list, 5))
            if cr_leaf_sizes_list
            else float("nan"),
            "leaf_p95": float(np.percentile(cr_leaf_sizes_list, 95))
            if cr_leaf_sizes_list
            else float("nan"),
        }
    }

    if rfsrc_native_array_df is not None:
        leaves = rfsrc_native_array_df[rfsrc_native_array_df["parmID"] == 0]
        rf_leaves = leaves.groupby("treeID").size().to_numpy()
        rf_leaf_sizes = leaves["nodeSZ"].to_numpy()
        out["rfSRC"] = {
            "mean_leaves": float(rf_leaves.mean()),
            "mean_depth": None,  # nativeArray lacks direct depth
            "leaf_p5": float(np.percentile(rf_leaf_sizes, 5)),
            "leaf_p95": float(np.percentile(rf_leaf_sizes, 95)),
        }
    return out


def _self_test() -> int:
    """Inline corner-case check for apply_tolerance (spec §11). Exit 0 on pass, 1 on fail.

    Four cases:
      - both pass
      - noise-floor fail only
      - hard-cap fail only
      - both fail
    """

    def agg(wcr_r, wrf_r, x_r, wcr_c, wrf_c, x_c):
        return {
            "within_cr_p95_risk": wcr_r,
            "within_rf_p95_risk": wrf_r,
            "cross_p95_risk": x_r,
            "within_cr_p95_cif": wcr_c,
            "within_rf_p95_cif": wrf_c,
            "cross_p95_cif": x_c,
        }

    cases = [
        ("both_pass", agg(0.02, 0.02, 0.01, 0.03, 0.03, 0.02), True, True, True, True),
        ("nf_fail_risk", agg(0.01, 0.01, 0.03, 0.03, 0.03, 0.02), False, True, True, True),
        ("hc_fail_cif", agg(0.02, 0.02, 0.01, 0.10, 0.10, 0.08), True, True, True, False),
        ("both_fail", agg(0.01, 0.01, 0.20, 0.01, 0.01, 0.20), False, False, False, False),
    ]
    fails = []
    for name, a, exp_nf_r, exp_hc_r, exp_nf_c, exp_hc_c in cases:
        r = apply_tolerance(a, hard_cap=HARD_CAP_DEFAULT)
        got = (
            r["noise_floor_pass_risk"],
            r["hard_cap_pass_risk"],
            r["noise_floor_pass_cif"],
            r["hard_cap_pass_cif"],
        )
        expected = (exp_nf_r, exp_hc_r, exp_nf_c, exp_hc_c)
        if got != expected:
            fails.append((name, got, expected))
    if fails:
        print("self-test FAIL:", fails, flush=True)
        return 1
    print("self-test OK (4/4 corners)", flush=True)
    return 0


def _git_sha() -> str:
    import subprocess

    try:
        out = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
        )
        return out.stdout.strip()
    except Exception:
        return "unknown"


def _machine_fingerprint() -> str:
    import platform

    return f"{platform.system()} {platform.machine()} {platform.processor() or 'unknown CPU'}"


def main(argv: list[str] | None = None) -> int:
    import argparse
    import datetime as _dt

    from validation.alignment.report_equivalence import write_report

    parser = argparse.ArgumentParser(
        description="crforest vs rfSRC equivalence gate (maintainer-invoked audit).",
    )
    parser.add_argument(
        "--datasets",
        nargs="+",
        default=["follic", "hd", "pbc", "synthetic"],
    )
    parser.add_argument(
        "--seeds", type=int, default=20, help="number of seeds per dataset (must be even)"
    )
    parser.add_argument(
        "--out", default="validation/reports", help="output directory for the markdown report"
    )
    parser.add_argument("--cache-dir", default="validation/alignment/_cache")
    parser.add_argument("--force-refit", action="store_true")
    parser.add_argument("--hard-cap", type=float, default=HARD_CAP_DEFAULT)
    parser.add_argument(
        "--self-test", action="store_true", help="run inline tolerance corner-case check and exit"
    )
    args = parser.parse_args(argv)

    if args.self_test:
        return _self_test()

    if args.seeds % 2 != 0:
        parser.error("--seeds must be even (needed for within-lib pairing)")

    cache_dir = Path(args.cache_dir)
    commit_sha = _git_sha()

    datasets_agg: dict[str, dict] = {}
    datasets_pass: dict[str, dict] = {}
    tree_stats: dict[str, dict] = {}
    cause2_symmetry: dict[str, float] = {}

    for ds in args.datasets:
        _, time_full, event_full = load_dataset(ds)
        splits_df = pd.read_parquet(_SPLITS_DIR / f"{ds}.parquet")

        cells = []
        for s in range(args.seeds):
            fit_and_capture(
                dataset=ds,
                seed=s,
                cache_dir=cache_dir,
                commit_sha=commit_sha,
                force_refit=args.force_refit,
            )
            cell = load_cell(cache_dir / f"{ds}_s{s}.parquet")
            row = splits_df[splits_df["seed"] == s]
            train_idx = np.sort(row.loc[row["fold"] == "train", "sample_id"].to_numpy(np.int64))
            test_idx = np.sort(row.loc[row["fold"] == "test", "sample_id"].to_numpy(np.int64))
            ibs_kwargs = dict(
                ref_grid=cell["ref_grid"],
                test_time=time_full[test_idx],
                test_event=event_full[test_idx],
                train_time=time_full[train_idx],
                train_event=event_full[train_idx],
            )
            cell["ibs_cr"] = compute_ibs(cell["cif_cr"], **ibs_kwargs)
            cell["ibs_rf"] = compute_ibs(cell["cif_rf"], **ibs_kwargs)
            cells.append(cell)
        agg = aggregate_dataset(cells)
        agg.update(_aggregate_ibs(cells))
        passes = apply_tolerance(agg, hard_cap=args.hard_cap)
        datasets_agg[ds] = agg
        datasets_pass[ds] = passes
        tree_stats[ds] = {}  # populated manually post-run for v1; see TODO below.
        cause2_symmetry[ds] = float("nan")  # likewise — placeholder lane for v1.

    timestamp = _dt.datetime.now().isoformat(timespec="seconds")
    out_dir = Path(args.out)
    out_path = out_dir / f"equivalence_{timestamp.replace(':', '-')}.md"

    import sys

    header = {
        "timestamp": timestamp,
        "commit_sha": commit_sha,
        "rfsrc_version": "see --extra maintainer install",
        "python_version": sys.version.split()[0],
        "r_version": "see R --version",
        "machine": _machine_fingerprint(),
        "command": " ".join(
            ["python", "-m", "validation.alignment.equivalence_gate", *(argv or [])]
        ),
        "hard_cap": args.hard_cap,
    }
    write_report(
        datasets_agg=datasets_agg,
        datasets_pass=datasets_pass,
        tree_stats=tree_stats,
        cause2_symmetry=cause2_symmetry,
        header=header,
        path=out_path,
    )
    print(f"wrote {out_path}", flush=True)

    all_pass = all(p["overall_pass"] for p in datasets_pass.values())
    return 0 if all_pass else 2


if __name__ == "__main__":
    raise SystemExit(main())
