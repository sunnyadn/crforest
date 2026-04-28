# crforest

Competing-risks random survival forests for Python. scikit-learn-compatible, CPU-first,
designed to replace the Python → R workflow split that applied researchers
currently endure for competing-risks survival analysis.

> **Status: pre-alpha (v0.1).** API and internals may change before v1.0.

## Highlights

- **The only competing-risks Random Survival Forest in Python.**
  Three-state survival fit and predict, Aalen-Johansen CIF, Nelson-Aalen
  CHF, cause-specific Harrell + Uno IPCW C-indices — out of the box, no
  event collapse, fits cleanly to n = 1 000 000 on commodity CPU.
  scikit-survival has a single-event `RandomSurvivalForest`; lifelines /
  pycox have estimators but no forest; nothing else covers CR + tree
  ensemble + scale.
- **5–7× faster than [randomForestSRC](https://cran.r-project.org/package=randomForestSRC)**,
  the gold-standard R reference, at matched accuracy on real clinical
  workloads (real CHF cohort, n ≈ 75k, p = 58, ntree = 100, same machine;
  HF Harrell C-index tied at 0.864). Sub-linear wall scaling to n = 1M (122 s).
- **Order-of-magnitude faster than [scikit-survival](https://scikit-survival.readthedocs.io/)
  at sksurv's best config**, gap widening with n (5.7× at n = 5 000, 15× at
  n = 10 000, 64× at n = 25 000, 205× at n = 50 000). crforest also keeps
  the CHF + CIF outputs that sksurv has to disable to fit at scale: sksurv
  either OOMs (`low_memory=False`, the only mode that exposes
  `predict_cumulative_hazard_function`) or produces risk scores only
  (`low_memory=True`). Numbers in [Bench](#bench).
- **Bit-identical to randomForestSRC** with `equivalence="rfsrc"` —
  reproduces the per-tree mtry/nsplit RNG stream; pairs with rfSRC
  `bootstrap=by.user` fits to reach the Z-cell numerical floor (~0.01–0.07
  cross_p95_cif on standard CR datasets, ntree = 100). Useful for paper
  reviews, sensitivity checks, and migrations.
- **Drop-in for Python pipelines.** `BaseEstimator` subclass, pickleable;
  `fit` / `predict_cif` / `predict_chf` / `predict_risk` / `score` /
  `compute_importance` follow scikit-learn conventions.
- **Reproducible permutation VIMP.** OOB Breiman or held-out, scored with
  Uno IPCW C-index — independent re-implementation of the textbook formula,
  bit-equivalent across `n_jobs`.
- **Two splitrules.** `logrankCR` (composite competing-risks log-rank,
  default) and `logrank` (cause-specific).
- **GPU preview.** Optional CUDA backend (`device="cuda"` +
  `crforest[gpu]`); faster only at low feature count today, full rewrite
  scheduled for v1.1.

## Install

```bash
pip install crforest          # core (CPU)
pip install "crforest[gpu]"   # + cupy / CUDA 12 preview
```

Requires Python ≥ 3.10. Core dependencies: numpy, scipy, pandas, joblib,
numba, scikit-learn.

## Quickstart

```python
import numpy as np
from crforest import CompetingRiskForest

# Toy competing-risks data: 500 subjects, 6 features, 2 causes (+ censoring).
rng = np.random.default_rng(42)
n = 500
X = rng.normal(size=(n, 6))
time = rng.exponential(2.0, size=n) + 0.1
event = rng.choice([0, 1, 2], size=n, p=[0.4, 0.4, 0.2])  # 0 = censored

# Fit. Defaults: n_estimators=100, max_features="sqrt", logrankCR, n_jobs=-1.
forest = CompetingRiskForest(n_estimators=100, random_state=42).fit(X, time, event)

# Per-subject risk score for cause 1 (suitable for Wolbers C-index).
risk = forest.predict_risk(X[:5], cause=1)

# Aalen-Johansen cumulative incidence over the forest's chosen time grid.
cif = forest.predict_cif(X[:5])                       # (5, n_causes, n_times)
cif_at = forest.predict_cif(X[:5], times=[1.0, 2.0, 5.0])

# Cause-specific Wolbers concordance.
print("C-index, cause 1:", forest.score(X, time, event, cause=1))

# OOB permutation VIMP, scored with Uno IPCW.
vimp = forest.compute_importance(random_state=42)
print(vimp.sort_values("composite_vimp", ascending=False).head())
```

See [docs/quickstart.md](docs/quickstart.md) for the full walkthrough — data
format, prediction shapes, cross-validation, GPU, and migrating from rfSRC.

## Why crforest?

The Python ecosystem currently leaves competing-risks survival users with
no good option:

- **[scikit-survival](https://scikit-survival.readthedocs.io/)** does not
  natively support competing risks (single-event only). Its
  `RandomSurvivalForest` has two storage modes, both losing: `low_memory=False`
  (the only mode that supports `predict_cumulative_hazard_function` /
  `predict_survival_function`) stores per-leaf full CHF arrays and OOMs at
  moderate n; `low_memory=True` fits at scale but only `predict()` works —
  no CHF, no survival function.
- **[randomForestSRC](https://cran.r-project.org/package=randomForestSRC)**
  is correct and feature-complete but slow, requires R, and is awkward
  inside Python pipelines (rpy2 glue, OpenMP rebuild on macOS, GC pauses).
  A 5-fold CV on a moderate cohort takes overnight.

crforest is the first Python package to fit all three jobs at once: native
competing risks, fits and outputs CHF/CIF at scale (n ≥ 10⁶ on commodity
CPU), and validated bit-identical to rfSRC for paper-grade reproducibility.

### Bench

**vs randomForestSRC, paired same machine** — i7-13700K, 24 threads, rfSRC
built with full OpenMP; real CHF cohort, HF / death competing risks;
n = 75 000, p = 58, ntree = 100:

| | crforest | rfSRC | speedup |
|---|---|---|---|
| Wall time | **22.5 s** | 111.7 s | **4.96×** |
| HF Harrell C-index | 0.8642 | 0.8643 | tied |

Apples-to-apples vs rfSRC's best `ntime` config on the same workload:
**6.13×**. Speedup ratio is stable across `ntree ∈ {100, 500, 1000}`.

**vs scikit-survival, paired same machine** — i7-13700K, 24 threads,
synthetic 2-cause Weibull DGP, p = 58, ntree = 100, both libraries at
their best config (`n_jobs=-1`; sksurv `low_memory=True`):

| n | sksurv `low_memory=True` | crforest | speedup |
|---|---|---|---|
| 5 000 | 29.3 s / 0.22 GB peak RSS | **5.1 s / 1.64 GB** | **5.7×** |
| 10 000 | 135.3 s / 0.27 GB | **8.9 s / 2.67 GB** | **15.2×** |
| 25 000 | 992.2 s / 0.42 GB | **15.4 s / 3.81 GB** | **64.4×** |
| 50 000 | 4 925.5 s (82 min) / 0.66 GB | **24.0 s / 5.06 GB** | **205×** |

The wall-time gap **widens** with n (sksurv RSF wall scales ≈ n^2.2 with
default `min_samples_leaf=3`; crforest histogram split kernel ≈ n^0.8). At
every paired point crforest also provides Aalen-Johansen CIF, Nelson-Aalen
CHF, and risk scores; sksurv `low_memory=True` provides only `predict()`
risk scores — `predict_cumulative_hazard_function` and
`predict_survival_function` raise `NotImplementedError`. Sksurv's other
mode (`low_memory=False`) restores those outputs but OOMs: at n = 5k it
already peaks at 16.8 GB RSS; at n = 10k it exceeds a 21.5 GB cap on a
24 GB host. Numbers reproducible via
[`validation/comparisons/sksurv_oom.py`](validation/comparisons/sksurv_oom.py).

**Scaling (one-sided beyond the paired ranges).** crforest exhibits
sub-linear wall growth in n with the histogram split kernel:

| Workload (default config, ntree = 100) | crforest CPU wall | rfSRC | sksurv |
|---|---|---|---|
| n = 75 000 (real CHF, paired) | 22.5 s | 111.7 s | OOM (`low_memory=False`) / extrapolated ~3 hr (`low_memory=True`) |
| n = 1 000 000 (UKB-scale feasibility) | 122 s | not measured (rfSRC peak RSS at n = 75 000 already 14.7 GB; extrapolation puts n = 500 000 past 80 GB) | not feasible at full output capability |

We do not publish a paired number above n ≈ 100 000 because both R and
Python alternatives are impractical there, not because the comparison gets
unfavourable.

> **For R users running rfSRC effectively single-threaded** — common on
> macOS, where rfSRC's OpenMP requires rebuilding R against a Homebrew
> gcc/clang with OpenMP support — the headline speedup understates what
> you'll see. crforest auto-parallelises across all cores by default
> (`n_jobs=-1`); the histogram split kernel is numba-jitted and releases
> the GIL, so it actually scales with thread count. crforest at
> `n_jobs=1` is also still numba-compiled, so even a forced-serial fit
> stays fast.

The compact leaf storage (per-cause integer event/at-risk counts only;
CIF/CHF tables materialise lazily on first predict) keeps pickle size
proportional to the cohort: n = 100k, ntree = 100 pickles to ~3.6 GB.

## API (one-line summary)

| | |
|---|---|
| `CompetingRiskForest(...)` | the estimator; full parameter list in [`forest.py`](src/crforest/forest.py) |
| `.fit(X, time, event)` | `time` is `(n,)` float; `event` is `(n,)` int (`0` = censored, `1..K` = cause) |
| `.predict_cif(X, times=None)` | shape `(n_samples, n_causes, n_times)` |
| `.predict_chf(X, times=None)` | shape `(n_samples, n_causes, n_times)` |
| `.predict_risk(X, cause=1, kind="integrated_chf")` | shape `(n_samples,)` |
| `.score(X, time, event, cause=1)` | Wolbers cause-specific concordance |
| `.compute_importance(...)` | per-cause + composite permutation VIMP DataFrame |
| `concordance_index_cr(event, time, estimate, cause=1)` | top-level metric for any risk score |

After `.fit()`: `forest.n_causes_`, `forest.unique_times_`, `forest.time_grid_`,
`forest.n_features_in_`, `forest.feature_importances_`, `forest.trees_`,
`forest.inbag_` (when `equivalence="rfsrc"`).

## Documentation

- [Quickstart](docs/quickstart.md) — common tasks with runnable code
- [PRD](docs/prd.md) — what crforest aims to be at v1.0
- [Equivalence vs rfSRC](docs/equivalence-vs-rfsrc.md) — cross-library validation methodology
- [References](docs/REFERENCES.md) — algorithmic provenance (Park-Miller, Bays-Durham, Wolbers 2009, Uno 2011, Cole & Hernán 2008, Breiman 2001, Ishwaran 2008/2014, etc.)

## Development

Requires [`uv`](https://docs.astral.sh/uv/).

```bash
uv venv
uv pip install -e ".[dev]"
uv run pre-commit install
uv run pytest
uv run ruff check .
uv run ruff format --check .
```

## License

Apache-2.0. See [LICENSE](LICENSE) and [NOTICE](NOTICE).

## Citation

A paper describing crforest is in preparation. Until it is out, please cite
the project URL and version. Algorithmic references (Park-Miller, Bays-Durham,
Wolbers, Uno, Cole & Hernán, Kaplan-Meier, Breiman, Ishwaran) are listed in
[`docs/REFERENCES.md`](docs/REFERENCES.md).
