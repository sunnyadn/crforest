"""Synthetic-data sanity benchmark for crforest OOB permutation VIMP.

Independent of rfSRC: generates competing-risks data with KNOWN signal features
and measures whether crforest VIMP separates them from noise. This is the
strongest "implementation correctness" check we have, since it doesn't rely on
matching another library — it asserts the algorithm correctly identifies the
features the data-generating process actually used.

DGP
---
- Features X ~ N(0, I_p), n samples x p features.
- Cause-specific hazards: λ_c(t | X) = λ0_c · exp(β_c^T X), constant baseline.
- First k1 features are signal for cause 1 (β_1[0:k1] = β_strength).
- Next k2 features are signal for cause 2 (β_2[k1:k1+k2] = β_strength).
- Remaining p - k1 - k2 features are pure noise (β = 0 in both).
- Event times: T_c ~ Exp(rate_c(X)), T = min(T_1, T_2, C); event = argmin.
- Censoring: independent exponential + admin cap.

Metrics (across n_seeds independent (data_seed, fit_seed) pairs)
---
For each cause c separately:
- **AUC(signal vs noise)** treating "is signal feature" as the label and
  cause_c_vimp as the score. AUC = 1.0 means signal features are always
  ranked above noise features. AUC ≈ 0.5 means random.
- **P(perfect separation)**: fraction of seeds where min(signal_vimp) >
  max(noise_vimp). Strict version of AUC=1.0.
- **Median signal rank** (1 = highest VIMP). For k_c signals, perfect ⇒ ranks
  {1, ..., k_c}; their median should be (k_c+1)/2.

For composite_vimp:
- Same metrics treating union of c1+c2 signals as combined signal set.

Outputs
---
Per-seed and aggregate stats printed to stdout; raw VIMP frame saved to
`/tmp/vimp_sanity_raw.parquet` for inspection.
"""

from __future__ import annotations

import sys

import numpy as np
import pandas as pd
from scipy.stats import rankdata
from sklearn.metrics import roc_auc_score

from crforest import CompetingRiskForest


def _print(msg: str) -> None:
    print(msg, flush=True)


def generate_cr_data(
    n: int,
    p: int,
    k1: int,
    k2: int,
    beta_strength: float,
    seed: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return (X, time, event) for a CR DGP with known signal features."""
    rng = np.random.default_rng(seed)
    X = rng.standard_normal((n, p)).astype(np.float64)

    beta1 = np.zeros(p)
    beta1[:k1] = beta_strength
    beta2 = np.zeros(p)
    beta2[k1 : k1 + k2] = beta_strength

    lp1 = X @ beta1
    lp2 = X @ beta2

    lam0_1 = 0.05
    lam0_2 = 0.05
    rate1 = lam0_1 * np.exp(lp1)
    rate2 = lam0_2 * np.exp(lp2)

    T1 = rng.exponential(1.0 / rate1)
    T2 = rng.exponential(1.0 / rate2)

    cens_rate = 0.02
    C = rng.exponential(1.0 / cens_rate, size=n)
    admin = 50.0
    C = np.minimum(C, admin)

    T = np.minimum(np.minimum(T1, T2), C)
    event = np.where(T == T1, 1, np.where(T == T2, 2, 0)).astype(np.int32)
    return X, T.astype(np.float64), event


def fit_one(
    X: np.ndarray,
    T: np.ndarray,
    event: np.ndarray,
    fit_seed: int,
    n_estimators: int,
) -> pd.DataFrame:
    forest = CompetingRiskForest(
        n_estimators=n_estimators,
        min_samples_leaf=1,
        min_samples_split=20,
        max_features="sqrt",
        bootstrap=True,
        random_state=fit_seed,
        equivalence="rfsrc",
    ).fit(X, T, event)
    return forest.compute_importance()


def evaluate_one_seed(vimp_df: pd.DataFrame, k1: int, k2: int, p: int) -> dict:
    """Per-seed metrics for both causes and composite."""
    signal_c1 = set(range(k1))
    signal_c2 = set(range(k1, k1 + k2))
    signal_union = signal_c1 | signal_c2
    feat_idx = np.arange(p)

    out = {}
    for cause, signals in [(1, signal_c1), (2, signal_c2), ("composite", signal_union)]:
        col = f"cause_{cause}_vimp" if isinstance(cause, int) else "composite_vimp"
        scores = vimp_df[col].to_numpy(dtype=np.float64)
        labels = np.array([f in signals for f in feat_idx], dtype=np.int64)
        # AUC: handle degenerate case where all scores equal
        auc = 0.5 if len(np.unique(scores)) == 1 else roc_auc_score(labels, scores)
        # Ranks: 1 = highest VIMP (argsort descending)
        ranks = rankdata(-scores, method="average")
        signal_ranks = ranks[list(signals)]
        signal_vimp = scores[list(signals)]
        noise_vimp = scores[[f for f in feat_idx if f not in signals]]
        perfect = float(signal_vimp.min() > noise_vimp.max())
        out[f"auc_{cause}"] = auc
        out[f"perfect_{cause}"] = perfect
        out[f"median_rank_{cause}"] = float(np.median(signal_ranks))
        out[f"signal_vimp_mean_{cause}"] = float(signal_vimp.mean())
        out[f"noise_vimp_mean_{cause}"] = float(noise_vimp.mean())
    return out


def run_one_beta(
    beta_strength: float,
    n: int,
    p: int,
    k1: int,
    k2: int,
    n_estimators: int,
    n_seeds: int,
) -> tuple[pd.DataFrame, list[dict]]:
    raw_rows = []
    seed_stats = []
    _print(f"\n=== β={beta_strength} (HR/SD ≈ {np.exp(beta_strength):.2f}) ===")
    for s in range(n_seeds):
        data_seed = 1000 + s
        fit_seed = 7000 + s
        X, T, event = generate_cr_data(n, p, k1, k2, beta_strength, seed=data_seed)
        ev_dist = {c: float(np.mean(event == c)) for c in (0, 1, 2)}
        vimp_df = fit_one(X, T, event, fit_seed=fit_seed, n_estimators=n_estimators)
        stats = evaluate_one_seed(vimp_df, k1, k2, p)
        stats["beta"] = beta_strength
        stats["seed"] = s
        seed_stats.append(stats)
        for _, row in vimp_df.iterrows():
            raw_rows.append({"beta": beta_strength, "seed": s, **row.to_dict()})
        _print(
            f"  seed={s:2d} ev[0/1/2]={ev_dist[0]:.2f}/{ev_dist[1]:.2f}/{ev_dist[2]:.2f}  "
            f"AUC c1={stats['auc_1']:.3f}  c2={stats['auc_2']:.3f}  "
            f"comp={stats['auc_composite']:.3f}  "
            f"perfect c1/c2/comp={int(stats['perfect_1'])}/"
            f"{int(stats['perfect_2'])}/{int(stats['perfect_composite'])}  "
            f"med_rank c1/c2={stats['median_rank_1']:.1f}/{stats['median_rank_2']:.1f}"
        )
    return pd.DataFrame(raw_rows), seed_stats


def main() -> int:
    n = 400
    p = 20
    k1 = 3
    k2 = 3
    n_estimators = 100
    n_seeds = 15
    beta_grid = (0.3, 0.5, 1.0)

    _print(
        f"# vimp_sanity: n={n} p={p} k1={k1} k2={k2} β-grid={beta_grid} "
        f"ntree={n_estimators} n_seeds={n_seeds}"
    )
    _print(
        f"# signal_c1=features {list(range(k1))}, "
        f"signal_c2=features {list(range(k1, k1 + k2))}, "
        f"noise=features {list(range(k1 + k2, p))}"
    )
    _print("# β=0.3→HR/SD≈1.35 (subtle); β=0.5→HR/SD≈1.65 (moderate); β=1.0→HR/SD≈2.72 (strong).")

    all_raw = []
    all_stats = []
    for b in beta_grid:
        raw, stats = run_one_beta(b, n, p, k1, k2, n_estimators, n_seeds)
        all_raw.append(raw)
        all_stats.extend(stats)

    stats_df = pd.DataFrame(all_stats)

    _print("\n## Aggregate by β (median across 15 seeds, p2.5/p97.5 in brackets)")
    metrics = [
        "auc_1",
        "auc_2",
        "auc_composite",
        "perfect_1",
        "perfect_2",
        "perfect_composite",
        "median_rank_1",
        "median_rank_2",
    ]
    header = "  β     " + "  ".join(f"{m:>20s}" for m in metrics)
    _print(header)
    for b in beta_grid:
        sub = stats_df[stats_df["beta"] == b]
        cells = []
        for m in metrics:
            v = sub[m].to_numpy()
            cells.append(
                f"{np.median(v):.3f} [{np.quantile(v, 0.025):.2f},{np.quantile(v, 0.975):.2f}]"
            )
        _print(f"  {b:.2f}  " + "  ".join(f"{c:>20s}" for c in cells))

    _print("\n## Signal vs noise VIMP magnitude (median across seeds, by β)")
    for b in beta_grid:
        sub = stats_df[stats_df["beta"] == b]
        _print(f"  β={b}:")
        for cause in (1, 2, "composite"):
            sm = sub[f"signal_vimp_mean_{cause}"].median()
            nm = sub[f"noise_vimp_mean_{cause}"].median()
            ratio = sm / nm if abs(nm) > 1e-12 else float("inf")
            _print(
                f"    cause={cause}: signal_mean={sm:+.4f}  noise_mean={nm:+.4f}  "
                f"ratio={ratio:+.2f}"
            )

    raw = pd.concat(all_raw, ignore_index=True)
    raw.to_parquet("/tmp/vimp_sanity_raw.parquet")
    _print("\nRaw VIMP frame written to /tmp/vimp_sanity_raw.parquet")
    return 0


if __name__ == "__main__":
    sys.exit(main())
