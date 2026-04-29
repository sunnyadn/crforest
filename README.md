# crforest

Competing-risks random survival forests for Python. 5–7× faster than
randomForestSRC, scales to n = 10⁶ in minutes, scikit-learn-compatible.
Designed to replace the Python → R workflow split that applied
researchers currently endure for competing-risks survival analysis.

> **Status: pre-alpha (v0.1).** API and internals may change before v1.0.

## Highlights

- **The only competing-risks Random Survival Forest in Python.** Three-state
  fit and predict, Aalen-Johansen CIF, Nelson-Aalen CHF, cause-specific
  Harrell + Uno IPCW C-indices, OOB Breiman permutation VIMP — out of the box.
- **4.5–6× faster than [randomForestSRC](https://cran.r-project.org/package=randomForestSRC)**
  at matched accuracy (HF Harrell C tied at 0.864, real CHF n ≈ 75k);
  ~58× faster than rfSRC built without OpenMP (default R-on-macOS).
- **Order-of-magnitude faster than [scikit-survival](https://scikit-survival.readthedocs.io/)**
  (5.4× at n = 5k, 192× at n = 50k), without disabling CIF/CHF outputs.
- **Bit-identical to randomForestSRC** with `equivalence="rfsrc"` —
  reproduces the per-tree mtry/nsplit RNG stream for paper-grade
  reproducibility, sensitivity checks, and rfSRC-baseline migrations.

## crforest vs alternatives

|                                          | crforest                       | randomForestSRC                    | scikit-survival          |
|------------------------------------------|:------------------------------:|:----------------------------------:|:------------------------:|
| Language                                 | Python                         | R                                  | Python                   |
| Native competing risks                   | ✓                              | ✓                                  | ✗ (single-event only)    |
| Aalen–Johansen CIF output                | ✓                              | ✓                                  | n/a                      |
| Cumulative hazard at scale               | ✓                              | ✓                                  | ✗¹                       |
| OOB permutation VIMP                     | ✓                              | ✓                                  | ✗                        |
| Bit-identical reproducibility mode       | ✓ (`equivalence="rfsrc"`)      | —                                  | n/a                      |
| Scales to n = 10⁶                        | ✓                              | OOM at n ≳ 500 000                 | ✗¹ / OOM²                |
| Default parallelism                      | ✓ (`n_jobs=-1`)                | OpenMP (build-dependent; macOS Apple clang lacks it) | ✓        |
| GPU preview                              | ✓ (CUDA 12)                    | ✗                                  | ✗                        |

¹ sksurv `RandomSurvivalForest(low_memory=True)` is the only mode that
scales beyond ~10k samples, but it disables `predict_cumulative_hazard_function`
and `predict_survival_function` (raises `NotImplementedError`).
² sksurv `low_memory=False` exposes CHF / survival outputs but stores per-leaf
full CHF arrays; peak RSS reaches 16.8 GB at n = 5k on synthetic, OOMs
(> 21.5 GB) at n = 10k on a 24 GB host.

## Install

```bash
pip install crforest          # or:  uv add crforest
pip install "crforest[gpu]"   # or:  uv add 'crforest[gpu]'
```

Requires Python ≥ 3.10. Core dependencies: numpy, scipy, pandas, joblib,
numba, scikit-learn. GPU extra adds cupy + CUDA 12 runtime libs (preview;
faster only at low feature count today, full rewrite scheduled for v1.1).

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

## scikit-learn drop-in

`CompetingRiskForest` is a real sklearn estimator: `BaseEstimator` subclass,
`clone()`-friendly, picklable, and `fit(X, y)` / `score(X, y)` accept the
scikit-survival-style structured `y` so `cross_val_score`, `KFold`, and
`Pipeline` work without a wrapper.

```python
from sklearn.model_selection import KFold, cross_val_score
from crforest import CompetingRiskForest, Surv

y = Surv.from_arrays(event=event, time=time)
forest = CompetingRiskForest(n_estimators=100, random_state=42, n_jobs=-1)

cv = KFold(n_splits=5, shuffle=True, random_state=42)
scores = cross_val_score(forest, X, y, cv=cv, n_jobs=-1)
print(f"5-fold C-index, cause 1: {scores.mean():.3f} ± {scores.std():.3f}")
```

The legacy three-argument form `forest.fit(X, time, event)` keeps working,
and `predict(X)` is an alias for `predict_risk(X, cause=1)` so the
estimator slots straight into `Pipeline` / `cross_val_predict`. For
cause-k risk or for CIF / CHF curves use the explicit methods.

## Benchmarks

**vs randomForestSRC, paired same machine** — i7-14700K, 28 threads,
rfSRC 3.6.2 built with full OpenMP; real CHF cohort, HF / death
competing risks; n = 75 000, p = 58, ntree = 100, 3 seeds (mean ± std):

| | crforest (`n_jobs=-1`) | rfSRC OMP-on (`rf.cores=28`) | rfSRC OMP-off (`rf.cores=1`) |
|---|---|---|---|
| Wall time | **17.75 ± 0.36 s** | 80.81 ± 0.68 s | 1026.3 ± 6.6 s (~17 min) |
| Peak RSS | **4.36 GB** | 22.70 GB | 17.80 GB |
| HF Harrell C-index | 0.8642 | 0.8645 | 0.8645 |
| Speedup vs crforest | — | **4.55×** | **57.8×** |

The OMP-off column is the configuration most R-on-macOS users hit out of
the box — rfSRC's OpenMP requires rebuilding R against Homebrew
gcc/clang with OpenMP support, which most Mac R installs lack. rfSRC
OMP-on and OMP-off produce algorithmically identical output at the same
seed (we verified per-seed err.rate is bit-identical), so the OMP-off
C-index is taken from the OMP-on cell at the same seed. Reproducible via
[`validation/comparisons/n75k_path_b.py`](validation/comparisons/n75k_path_b.py).

Apples-to-apples vs rfSRC's best `ntime` config on the same workload:
**6.13×**. Speedup ratio is stable across `ntree ∈ {100, 500, 1000}`.

**vs scikit-survival, paired same machine** — i7-14700K, 28 threads,
synthetic 2-cause Weibull DGP, p = 58, ntree = 100, both libraries at
their best config (`n_jobs=-1`; sksurv `low_memory=True`):

| n | sksurv `low_memory=True` | crforest | speedup |
|---|---|---|---|
| 5 000 | 18.2 s / 0.22 GB peak RSS | **3.4 s / 1.83 GB** | **5.4×** |
| 10 000 | 85.0 s / 0.26 GB | **5.6 s / 3.06 GB** | **15.1×** |
| 25 000 | 609.7 s / 0.37 GB | **10.4 s / 4.82 GB** | **58.4×** |
| 50 000 | 2 935.3 s (49 min) / 0.55 GB | **15.3 s / 6.80 GB** | **191.6×** |

The wall-time gap **widens** with n (sksurv RSF wall scales ≈ n^2.2 with
default `min_samples_leaf=3`; crforest histogram split kernel ≈ n^0.6).
At every paired point crforest also provides Aalen-Johansen CIF,
Nelson-Aalen CHF, and risk scores; sksurv `low_memory=True` provides
only `predict()` risk scores — `predict_cumulative_hazard_function` and
`predict_survival_function` raise `NotImplementedError`. When both are
scored on the single-event-collapsed truth sksurv is fit on, holdout
Harrell C-index is matched within ±0.01 across n; crforest additionally
surfaces a cause-specific Wolbers concordance of 0.69–0.72 for cause 1
(HF) that sksurv has no way to compute on competing-risk data. Sksurv's
other mode (`low_memory=False`) restores CHF/survival outputs but OOMs:
at n = 5k it already peaks at 16.8 GB RSS; at n = 10k it exceeds a 21.5
GB cap on a 24 GB host. Numbers reproducible via
[`validation/comparisons/sksurv_oom.py`](validation/comparisons/sksurv_oom.py).

**Scaling (one-sided beyond the paired ranges).** crforest exhibits
sub-linear wall growth in n with the histogram split kernel:

| Workload (default config, ntree = 100) | crforest CPU wall | rfSRC | sksurv |
|---|---|---|---|
| n = 75 000 (real CHF, paired) | 17.75 s | 80.81 s | OOM (`low_memory=False`) / extrapolated ~2.4 hr (`low_memory=True`) |
| n = 1 000 000 (UKB-scale feasibility) | 122 s | not measured (rfSRC peak RSS at n = 75 000 already 22.7 GB OMP-on / 17.8 GB OMP-off; extrapolation puts n = 500 000 past 80 GB) | not feasible at full output capability |

We do not publish a paired number above n ≈ 100 000 because both R and
Python alternatives are impractical there, not because the comparison gets
unfavourable.

> Even when forced to `n_jobs=1`, crforest stays fast: the histogram
> split kernel is numba-compiled, so a single-thread fit uses the same
> hot loop as the parallel one (just without the joblib fan-out). That
> matters for environments where worker pools are forbidden — Jupyter
> notebooks under certain spawn configs, locked-down HPC schedulers,
> nested-parallel pipelines, etc.

The compact leaf storage (per-cause integer event/at-risk counts only;
CIF/CHF tables materialise lazily on first predict) keeps pickle size
proportional to the cohort: n = 100k, ntree = 100 pickles to ~5.0 GB.

## API (one-line summary)

| | |
|---|---|
| `CompetingRiskForest(...)` | the estimator; full parameter list in [`forest.py`](src/crforest/forest.py) |
| `.fit(X, time, event)` or `.fit(X, y)` | legacy form takes `time` `(n,)` float and `event` `(n,)` int (`0`=censored, `1..K`=cause); sklearn form takes structured `y` from `Surv.from_arrays` |
| `.predict(X)` | sklearn alias for `predict_risk(X, cause=1)`; shape `(n_samples,)` |
| `.predict_cif(X, times=None)` | shape `(n_samples, n_causes, n_times)` |
| `.predict_chf(X, times=None)` | shape `(n_samples, n_causes, n_times)` |
| `.predict_risk(X, cause=1, kind="integrated_chf")` | shape `(n_samples,)` |
| `.score(X, time, event, cause=1)` or `.score(X, y, cause=1)` | Wolbers cause-specific concordance |
| `.compute_importance(...)` | per-cause + composite permutation VIMP DataFrame |
| `Surv.from_arrays(event, time)` | structured `y` for sklearn `fit(X, y)` / `cross_val_score` |
| `concordance_index_cr(event, time, estimate, cause=1)` | top-level metric for any risk score |

After `.fit()`: `forest.n_causes_`, `forest.unique_times_`, `forest.time_grid_`,
`forest.n_features_in_`, `forest.feature_importances_`, `forest.trees_`,
`forest.inbag_` (when `equivalence="rfsrc"`).

Two splitrules are available: `logrankCR` (composite competing-risks
log-rank, default) and `logrank` (cause-specific).

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
