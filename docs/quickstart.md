# Quickstart

A working tour of crforest: data format, fitting, prediction shapes, scoring,
permutation importance, performance levers, and rfSRC migration.

Every code block is runnable end-to-end.

## 1. Data format

crforest expects three arrays and treats them positionally:

```python
import numpy as np

# n = number of subjects, p = number of features.
X = np.random.default_rng(0).normal(size=(500, 6))   # (n, p) float
time = np.random.default_rng(1).exponential(size=500) + 0.1  # (n,) float, > 0
event = np.random.default_rng(2).choice([0, 1, 2], size=500)  # (n,) int

# Convention:
#   event[i] == 0    → subject i was censored at time[i]
#   event[i] == k≥1  → subject i had cause-k event at time[i]
```

Causes are indexed from 1 — `event` codes 1, 2, … K. crforest infers `K`
from the training data (`forest.n_causes_` after fit; supports up to 255
causes).

Pandas / polars DataFrames are accepted for `X`; `time` and `event` are
fed as 1-D arrays. No structured-array `y` needed at fit time (unlike
scikit-survival).

## 2. Fitting

```python
from crforest import CompetingRiskForest

forest = CompetingRiskForest(
    n_estimators=200,
    max_features="sqrt",   # mtry; can also be int / float / "log2" / None
    max_depth=15,
    min_samples_leaf=3,
    splitrule="logrankCR", # composite CR log-rank (default)
    n_jobs=-1,             # all CPUs; bit-equivalent across n_jobs values
    random_state=42,
).fit(X, time, event)
```

Defaults match common practice: 100 trees, sqrt-mtry, depth 15, leaves of
size 3, joblib over all cores. Output is bit-identical for a fixed
`random_state` regardless of `n_jobs`.

The two splitrules:

| `splitrule` | What it optimises | When to use |
|---|---|---|
| `"logrankCR"` (default) | Composite CR log-rank pooled over all causes (Lau-inclusive at-risk) | When you want one forest that respects every cause |
| `"logrank"` | Cause-specific log-rank for one cause; competing events remove the subject from the risk set | When one cause is your primary outcome — fits faster than `logrankCR` |

For `splitrule="logrank"`, set `cause=k` to choose the cause to optimise,
or supply `cause_weights=[w_1, …, w_K]` (reference mode only) for a
weighted combination.

After `.fit()`:

| attribute | meaning |
|---|---|
| `forest.n_causes_` | number of causes inferred from training event labels |
| `forest.n_features_in_` | number of features at fit time |
| `forest.unique_times_` | union of training event times (used as the prediction time grid) |
| `forest.time_grid_` | the histogram-mode internal grid (≤ `time_grid` points; `unique_times_` is identical to it in default mode) |
| `forest.trees_` | list of fitted trees; each is a flat-tree node array (default mode) or a tree-node object graph (reference mode) |

## 3. Prediction

Three call shapes covering the standard CR-survival outputs:

```python
# (a) Cumulative incidence (Aalen-Johansen). Shape: (n_samples, n_causes, n_times).
cif_full = forest.predict_cif(X)

# (b) CIF interpolated to user-supplied times (right-continuous step function).
cif_at = forest.predict_cif(X, times=[0.5, 1.0, 5.0])  # (n, K, 3)

# (c) Cumulative hazard (Nelson-Aalen). Same shape semantics.
chf_full = forest.predict_chf(X)
chf_at   = forest.predict_chf(X, times=[1.0, 2.0])

# (d) Per-subject risk scalar — convenient for Wolbers C-index pairs.
risk_cause1 = forest.predict_risk(X, cause=1)            # default: integrated CHF over time grid
risk_cif_last = forest.predict_risk(X, cause=1, kind="cif_last")  # CIF at t_max
```

`kind` for `predict_risk`:

- `"integrated_chf"` (default) — the rfSRC `predict$predicted[, cause]`
  convention. Captures curve shape + saturation behaviour.
- `"cif_last"` — single-time-point summary (`CIF[k, t_max]`). Simpler but
  loses signal when CIF saturates near 1.

If you need raw per-tree CIFs, walk `forest.trees_` directly — each tree's
predict function lives in `crforest._tree.predict_tree` (reference mode) or
`crforest._hist_tree.predict_tree_hist` (default mode).

## 4. Scoring

`forest.score(X, time, event, cause=1)` runs Wolbers's cause-specific C-index
on the chosen risk scalar:

```python
c_cause1 = forest.score(X, time, event, cause=1)
c_cause2 = forest.score(X, time, event, cause=2)
```

For Uno IPCW C-index (recommended in heavily-censored data), use the
top-level metric on a precomputed risk vector:

```python
from crforest.metrics import compute_uno_weights, concordance_index_uno_cr

w = compute_uno_weights(time, event)                 # IPCW weights, shape (n,)
risk = forest.predict_risk(X, cause=1)
c_uno = concordance_index_uno_cr(event, time, risk, cause=1, weights=w)
```

`compute_uno_weights` defaults to ESS-truncation gating
(`gmin="auto"`, `ess_frac=0.20`), per Cole & Hernán (2008).

### Cross-validation

Two equivalent paths.

**Manual loop with the 3-arg form** — straightforward when `time` /
`event` already exist as separate arrays:

```python
from sklearn.model_selection import KFold

kf = KFold(n_splits=5, shuffle=True, random_state=42)
scores = []
for train_idx, test_idx in kf.split(X):
    f = CompetingRiskForest(n_estimators=100, random_state=42).fit(
        X[train_idx], time[train_idx], event[train_idx]
    )
    scores.append(f.score(X[test_idx], time[test_idx], event[test_idx], cause=1))
print(f"CV C-index, cause 1: {np.mean(scores):.3f} ± {np.std(scores):.3f}")
```

**sklearn drop-in with `cross_val_score`** — works because
`CompetingRiskForest` is a real `BaseEstimator` subclass that accepts
the scikit-survival-style structured `y`. No wrapper, no custom scorer:

```python
from sklearn.model_selection import KFold, cross_val_score
from crforest import CompetingRiskForest, Surv

y = Surv.from_arrays(event=event, time=time)
forest = CompetingRiskForest(n_estimators=100, random_state=42, n_jobs=-1)

cv = KFold(n_splits=5, shuffle=True, random_state=42)
scores = cross_val_score(forest, X, y, cv=cv, n_jobs=-1)
print(f"5-fold C-index, cause 1: {scores.mean():.3f} ± {scores.std():.3f}")
```

`predict(X)` is an alias for `predict_risk(X, cause=1)`, so the estimator
also slots into `Pipeline` / `cross_val_predict`. For cause-`k` risk or
CIF / CHF curves use the explicit methods.

## 5. Permutation variable importance (VIMP)

Two flavours, both returning a `pandas.DataFrame` with columns
`feature`, `cause_{k}_vimp` per cause, and `composite_vimp`.

### OOB Breiman (default — no eval set needed)

```python
forest = CompetingRiskForest(
    n_estimators=200, bootstrap=True, random_state=42
).fit(X, time, event)

vimp_oob = forest.compute_importance(random_state=42)
print(vimp_oob.sort_values("composite_vimp", ascending=False).head())
```

OOB scoring uses the Uno IPCW C-index over the cached training data, with
each tree's permutation seeded reproducibly so results are bit-equivalent
across `n_jobs`.

### Held-out

When you want a clean train / eval split, supply `(X_eval, y_eval)`. `y_eval`
must be a structured array with `time` and `event` fields (matches sklearn's
permutation_importance contract):

```python
X_train, X_eval = X[:400], X[400:]
t_train,  t_eval = time[:400], time[400:]
e_train,  e_eval = event[:400], event[400:]

y_eval = np.zeros(len(t_eval), dtype=[("time", "f8"), ("event", "i8")])
y_eval["time"], y_eval["event"] = t_eval, e_eval

forest = CompetingRiskForest(n_estimators=100, random_state=42).fit(X_train, t_train, e_train)
vimp_held = forest.compute_importance(X_eval, y_eval, n_repeats=5, random_state=42)
```

## 6. Performance levers

| Lever | Default | Rule of thumb |
|---|---|---|
| `n_jobs` | `-1` (all cores) | The split kernel is numba-jitted and releases the GIL, so threads scale well in default mode. Reference mode is GIL-bound — `n_jobs=1` is fine there. Output is bit-identical across `n_jobs` for a fixed `random_state`. |
| `split_ntime` | `10` | Coarse time bins for split-search log-rank; full grid is kept for CIF/CHF output. λ.exp6 measured 5.5× wall reduction at zero accuracy delta on real CHF. For very small cohorts (n ≲ 500) prefer `50` or `None`. |
| `nsplit` | `10` (default mode), `0` (reference mode) | rfSRC-style random split-point sampling. `0` evaluates every threshold (slower, exact). |
| `time_grid` | `200` | Cap on internal time-grid size. Larger means more memory; smaller means coarser CIF. |
| `mode` | `"default"` | Histogram tree (uint8-binned features). Use `"reference"` only when you need exhaustive split search for equivalence work. |

For repeated fits at the same n / p / ntree, the histogram split kernel
benefits from numba caching after the first call — second-run wall is a
useful "warm" baseline.

## 7. GPU preview (optional)

```bash
pip install "crforest[gpu]"   # cupy-cuda12x + cuda runtime
```

```python
forest = CompetingRiskForest(
    n_estimators=100,
    random_state=42,
    device="cuda",            # opt-in; "auto" resolves to "cpu" in v0.1
).fit(X, time, event)
```

Today the cuda path is faster only at low feature count (p ≲ 20). At
typical clinical workloads (p ≈ 58) it is ~1.15× **slower** than CPU on
the same machine because host orchestration and D ↔ H sync dominate the
single-tree wall (kernels themselves are 4 % of total). The full GPU
rewrite is scheduled for v1.1; until then `device="cuda"` is a preview
flag for benchmarking and for users with low-p problems.

`device="cuda"` is incompatible with `equivalence="rfsrc"` /
`rng_mode="rfsrc_aligned"`.

## 8. Migrating from rfSRC

`equivalence="rfsrc"` ports randomForestSRC's per-tree mtry/nsplit RNG
stream + bootstrap-by-user inbag exposure so paired fits are bit-equivalent
to the Z-cell numerical floor:

```python
forest = CompetingRiskForest(
    n_estimators=100,
    equivalence="rfsrc",       # bundles rng_mode='rfsrc_aligned' + split_ntime=None
    random_state=42,           # explicit random_state required
).fit(X, time, event)

# Pair with R:  rfsrc(..., bootstrap = "by.user", samp = forest.inbag_, seed = -42)
```

Cross-library agreement on the four standard CR datasets (pbc, follic, hd,
synthetic; ntree = 100): cross_p95_cif typically 0.005 — 0.07, dominated
by per-tree-structure variance rather than RNG mismatch (see
[`docs/equivalence-vs-rfsrc.md`](equivalence-vs-rfsrc.md) for the full
methodology and limits).

If your rfSRC fits feel slow on macOS, you're likely running effectively
single-threaded: the CRAN R binary is not built with OpenMP, so rfSRC's
`rf.cores` option silently does nothing. Confirm with `rfsrc(...)$openmp`
or `library(parallel); detectCores()` against actual `top` CPU usage
during a fit. Fixing it requires rebuilding R against a Homebrew
gcc/clang with OpenMP support. crforest gets parallelism out of the box
(`n_jobs=-1` default) without any compile-time setup.

Cost: the aligned RNG runs the per-tree state in pure Python (correctness
over speed), so `equivalence="rfsrc"` is ~2–3× slower than the default
numpy-RNG path. Use it for cross-checks; leave the default on for
production fits.

## What's not here yet

- Calibration plots / time-dependent Brier score helpers (use
  [scikit-survival](https://scikit-survival.readthedocs.io/) for those, or
  roll your own from `predict_cif` curves).
- `predict_proba` / classification-style outputs.
- Distributed (multi-machine) training.
- Confidence intervals on VIMP (sklearn's `permutation_importance` reports
  `importances_std`; we surface only `importances_mean` in the DataFrame).

These are tracked for v1.0 / v1.1 — see [`docs/prd.md`](prd.md).
