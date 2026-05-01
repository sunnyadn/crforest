# crforest

[![PyPI version](https://img.shields.io/pypi/v/crforest.svg)](https://pypi.org/project/crforest/)
[![CI](https://github.com/sunnyadn/crforest/actions/workflows/ci.yml/badge.svg)](https://github.com/sunnyadn/crforest/actions/workflows/ci.yml)
[![DOI](https://img.shields.io/badge/DOI-10.5281%2Fzenodo.19876282-blue)](https://doi.org/10.5281/zenodo.19876282)

Competing-risks random survival forests for Python. ~15× faster than
randomForestSRC on real EHR-shaped data, scales to n = 10⁶ on a consumer
desktop in ~2 min, scikit-learn-compatible.
Designed to replace the Python → R workflow split that applied
researchers currently endure for competing-risks survival analysis.

> **Status: pre-alpha (v0.1).** API and internals may change before v1.0.

## Highlights

- **The only competing-risks Random Survival Forest in Python.** Three-state
  fit and predict, Aalen-Johansen CIF, Nelson-Aalen CHF, cause-specific
  Harrell + Uno IPCW C-indices, OOB Breiman permutation VIMP — out of the box.
- **14–22× faster than [randomForestSRC](https://cran.r-project.org/package=randomForestSRC)**
  on real EHR-shaped data (HF Harrell C tied at 0.864, real CHF n ≈ 75k),
  measured matched-pair across consumer desktop / laptop / HPC; ~95× faster
  than rfSRC built without OpenMP (default R-on-macOS install).
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

**vs randomForestSRC, matched-pair across hardware** — real CHF cohort,
HF / death competing risks; n = 75 278 train / p = 58, ntree = 100, leaf
= 3, nsplit = 10, seeds 42–44 (mean wall):

| Hardware | crforest (`n_jobs=-1`) | rfSRC OMP-on | Speedup | RSS ratio |
|---|---|---|---|---|
| Apple M4 (10-core, 16 GB) | 9.42 s | 207.3 s | **22.0×** | — |
| Intel i7-14700K (28-thread, 32 GB) | **5.79 s** | 84.75 s | **14.6×** | 3.7× less |
| HPC Xeon Gold 6148 (32-core, 187 GB) | 5.61 s | 111.05 s | **19.8×** | 3.6× less |

Both libraries report HF C-index ≈ 0.85 at this workload — crforest
0.864 (cause-specific Wolbers concordance), rfSRC 0.847–0.849 (rfSRC's
own native cause-specific C from `err.rate`). These are computed from
different code paths and should not be subtracted directly; both are
well above paper-grade thresholds and confirm the libraries fit
similarly well. The 14–22× speedup band reflects how rfSRC's OpenMP
scales with per-core speed: the i7's high-clock P-cores benefit rfSRC
most, so the gap is smallest there; on slower-per-core HPC silicon the
gap widens. Reproducible via
[`validation/comparisons/n75k_path_b.py`](validation/comparisons/n75k_path_b.py).

R-on-macOS users hit a separate scenario: rfSRC's OpenMP requires
rebuilding R against Homebrew gcc/clang, which most Mac R installs lack.
On a 10-core Apple M4, rfSRC built without OpenMP runs at ~895 s vs
crforest 9.42 s = **~95×** speedup at the default install path.

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

**Scaling (one-sided beyond the paired ranges).** Same consumer desktop
as the paired benches above (i7-14700K, 28 threads). crforest exhibits
sub-linear wall growth in n with the histogram split kernel:

| Workload (default config, ntree = 100) | crforest CPU wall | rfSRC | sksurv |
|---|---|---|---|
| n = 75 000 (real CHF, paired) | 5.79 s | 84.75 s | OOM (`low_memory=False`) / extrapolated ~2.4 hr (`low_memory=True`) |
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

If you use crforest in your research, please cite the version-specific
Zenodo DOI for the release you ran. For v0.1.2:

```bibtex
@software{yang_crforest_2026,
  author    = {Yang, Sunny and Zhao, Wanqi},
  title     = {{crforest: competing-risks random survival forests for Python}},
  year      = {2026},
  publisher = {Zenodo},
  version   = {0.1.2},
  doi       = {10.5281/zenodo.19876283},
  url       = {https://doi.org/10.5281/zenodo.19876282},
}
```

The `doi` field is the version-specific DOI (frozen at v0.1.2); the
`url` resolves through the concept DOI to whatever is the latest
release. GitHub's "Cite this repository" button (top-right of the repo
page) generates the same record from [`CITATION.cff`](CITATION.cff).
A paper describing crforest is in preparation; this section will be
updated when it is out.

Algorithmic references (Park-Miller, Bays-Durham, Wolbers, Uno, Cole &
Hernán, Kaplan-Meier, Breiman, Ishwaran) are listed in
[`docs/REFERENCES.md`](docs/REFERENCES.md).
