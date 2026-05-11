# comprisk

[![PyPI version](https://img.shields.io/pypi/v/comprisk.svg)](https://pypi.org/project/comprisk/)
[![CI](https://github.com/sunnyadn/comprisk/actions/workflows/ci.yml/badge.svg)](https://github.com/sunnyadn/comprisk/actions/workflows/ci.yml)
[![DOI](https://img.shields.io/badge/DOI-10.5281%2Fzenodo.19876282-blue)](https://doi.org/10.5281/zenodo.19876282)

**comprisk** — a Python toolkit for competing risks. Ships a scalable,
scikit-learn-compatible competing-risks random survival forest plus the
three canonical regression / non-parametric methods clinical researchers
actually need: Fine-Gray subdistribution-hazard regression, a stand-alone
Aalen-Johansen cumulative-incidence estimator with cmprsk-parity
variance, and cause-specific Cox PH (see [Roadmap](#roadmap)). Designed
to remove the Python → R workflow split that applied researchers
currently endure for competing-risks survival analysis.

> **Status: alpha.** API and internals may change before v1.0.
> **Renamed from `crforest` in 0.3.1** — `pip install comprisk`,
> `from comprisk import CompetingRiskForest`.

## Highlights

- **The four canonical CR methods, native Python.** `FineGrayRegression`
  matches `R cmprsk::crr()` β̂ to floating-point noise (max |Δβ| = 1.4e-15
  on three reference datasets); `robust_se=True` returns the Geskus
  cluster sandwich agreeing with cmprsk's IPCW-corrected SE to ~3 digits.
  `CumulativeIncidence` reproduces `cmprsk::cuminc()` to 1e-9 across CIF
  and variance. `gray_test` reproduces `cmprsk::cuminc()$Tests` to 1e-14.
  `CauseSpecificCox` matches `survival::coxph(method="breslow")` to 1e-9.
- **Only native-Python competing-risks RSF.** Cause-specific log-rank
  splitting + composite CR log-rank, Aalen-Johansen CIF, Nelson-Aalen CHF,
  Wolbers + Uno IPCW concordance, OOB Breiman VIMP, Ishwaran minimal-depth
  variable selection, exact TreeSHAP.
- **CR-aware model evaluation.** `score_cr` reports IPCW time-dependent
  AUC and Brier score under competing risks, plus integrated AUC / Brier
  (iAUC, IBS) with bootstrap CIs; `calibration_cr` returns tidy quantile-
  decile calibration data with per-bin Wilson intervals — one-call
  replacements for the CR-mode `riskRegression::Score()` / `plotCalibration()`
  blocks, taking a dict of named candidate models.
- **10–22× faster than [randomForestSRC](https://cran.r-project.org/package=randomForestSRC)**
  on real EHR data (CHF 14–22×, SEER 11.6×; full tables in
  [docs/benchmarks.md](docs/benchmarks.md)), with C ≈ 0.85 on both
  libraries. ~95× faster than rfSRC built without OpenMP (default R-on-macOS).
- **Order-of-magnitude faster than [scikit-survival](https://scikit-survival.readthedocs.io/)**
  (16.6× at n = 5k, 544× at n = 50k), without disabling CIF/CHF outputs.
- **Bit-identical to randomForestSRC** with `equivalence="rfsrc"` —
  reproduces the per-tree mtry/nsplit RNG stream for paper-grade
  reproducibility, sensitivity checks, and rfSRC-baseline migrations.

## comprisk vs alternatives

|                                          | comprisk                       | randomForestSRC                    | scikit-survival          |
|------------------------------------------|:------------------------------:|:----------------------------------:|:------------------------:|
| Language                                 | Python                         | R                                  | Python                   |
| Native competing risks                   | ✓                              | ✓                                  | ✗ (single-event only)    |
| Aalen–Johansen CIF output                | ✓                              | ✓                                  | n/a                      |
| Cumulative hazard at scale               | ✓                              | ✓                                  | ✗¹                       |
| OOB permutation VIMP                     | ✓                              | ✓                                  | ✗                        |
| Bit-identical reproducibility mode       | ✓ (`equivalence="rfsrc"`)      | —                                  | n/a                      |
| Scales to n = 10⁶                        | ✓ (63 s on i7)                 | memory-bound past n ≈ 500 000 on consumer hardware | ✗¹ / OOM²                |
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
pip install comprisk          # or:  uv add comprisk
pip install "comprisk[gpu]"   # or:  uv add 'comprisk[gpu]'
```

Requires Python ≥ 3.10. Core dependencies: numpy, scipy, pandas, joblib,
numba, scikit-learn. GPU extra adds cupy + CUDA 12 runtime libs (preview;
faster only at low feature count today, full rewrite scheduled for v1.1).

## Quickstart

```python
import numpy as np
from comprisk import CompetingRiskForest

# Toy competing-risks data: 500 subjects, 6 features, 2 causes (+ censoring).
rng = np.random.default_rng(42)
n = 500
X = rng.normal(size=(n, 6))
time = rng.exponential(2.0, size=n) + 0.1
event = rng.choice([0, 1, 2], size=n, p=[0.4, 0.4, 0.2])  # 0 = censored

# Fit. Defaults: n_estimators=100, max_features="sqrt", logrankCR, n_jobs=-1.
forest = CompetingRiskForest(n_estimators=100, random_state=42).fit(X, time, event)

# Aalen-Johansen cumulative incidence over the forest's chosen time grid.
cif = forest.predict_cif(X[:5])                       # (5, n_causes, n_times)

# Cause-specific Wolbers concordance.
print("C-index, cause 1:", forest.score(X, time, event, cause=1))
```

### Explainability and feature selection

```python
# OOB permutation importance (Uno IPCW-scored).
vimp = forest.compute_importance(random_state=42)

# Ishwaran minimal-depth variable selection.
selected = forest.minimal_depth().query("selected")["feature"].tolist()

# Exact TreeSHAP attributions (Lundberg 2018, Algorithm 2).
shap, base = forest.shap_values(X[:10])               # (n, p, n_times, n_causes)
```

### Fine-Gray, Aalen-Johansen, Gray's test, and cause-specific Cox

```python
from comprisk import (
    FineGrayRegression, CumulativeIncidence, CauseSpecificCox, gray_test,
)

# Fine-Gray subdistribution-hazard regression — matches R cmprsk::crr()
# β̂ to floating-point noise. robust_se=True gives the Geskus cluster
# sandwich (matches cmprsk's IPCW-corrected SE to ~3 digits).
fg = FineGrayRegression(cause=1, robust_se=True).fit(X, time=time, event=event)
print(fg.coef_, fg.se_)
F = fg.predict_cumulative_incidence(X[:5])            # (5, n_event_times)

# Non-parametric Aalen-Johansen CIF (cmprsk::cuminc parity, optional groups).
ci = CumulativeIncidence().fit(time=time, event=event, group=group_var)
est, var = ci.timepoints([1.0, 5.0, 10.0])            # (n_curves, 3)

# Gray's K-sample test for CIFs — matches cmprsk::cuminc()$Tests to 1e-14.
result = gray_test(time, event, group_var, cause=1)
print(result.stat, result.pvalue, result.df)

# Cause-specific Cox PH — competing events censored at t_j.
# Matches survival::coxph(method="breslow") to 1e-9.
cs = CauseSpecificCox(cause=1).fit(X, time=time, event=event)
```

Detailed walkthroughs — additivity checks, global SHAP importance, sklearn-
compatible slicing, performance caveats, rfSRC threshold compatibility — in
[docs/quickstart.md](docs/quickstart.md), which also covers data format,
prediction shapes, cross-validation, GPU, and rfSRC migration.

> **scikit-learn drop-in.** `CompetingRiskForest` is a real sklearn
> estimator (`BaseEstimator`, `clone()`-friendly, picklable).
> `cross_val_score`, `KFold`, `Pipeline` work without a wrapper — pass
> `Surv.from_arrays(event, time)` as the `y` argument, or use the legacy
> 3-arg `fit(X, time, event)` form. Full example in
> [docs/quickstart.md § Cross-validation](docs/quickstart.md#cross-validation).

## Roadmap

comprisk is intentionally CR-focused. For non-CR survival methods
(general Cox PH, AFT, parametric, deep-survival, Kaplan-Meier as a
standalone API), use [lifelines](https://lifelines.readthedocs.io/) or
[scikit-survival](https://scikit-survival.readthedocs.io/).

| Version  | Module                                                | Status               |
|----------|-------------------------------------------------------|----------------------|
| v0.3     | `CompetingRiskForest` (CR-RSF)                        | Shipped              |
| **v0.4** | `FineGrayRegression` (subdistribution hazard)         | Shipped              |
| **v0.4** | `CumulativeIncidence` (stand-alone Aalen-Johansen)    | Shipped              |
| **v0.4** | `gray_test` (Gray's K-sample log-rank)                | Shipped              |
| **v0.4** | `CauseSpecificCox` (CR-aware censoring)               | Shipped              |
| **v0.4** | `score_cr` / `calibration_cr` (CR-aware evaluation)   | Shipped              |
| v1.0     | API freeze + JMLR MLOSS submission                    | Planned              |
| v1.1     | Full GPU rewrite                                      | Planned              |

## Benchmarks

Headline numbers — full tables, methodology, and reproducibility scripts
in [docs/benchmarks.md](docs/benchmarks.md).

**vs randomForestSRC, matched-pair on real EHR data:**

| Cohort | n × p | Hardware | comprisk | rfSRC OMP-on | Speedup |
|---|---|---|---|---|---|
| CHF (cardio) | 75k × 58 | Apple M4 / i7-14700K / HPC | 5.6–9.4 s | 84.8–207.3 s | **14–22×** |
| SEER breast (oncology) | 238k × 17 | HPC Xeon Gold 6148 | 7.0 s | 81.6 s | **11.6×** |

Both libraries fit similarly well at every tested workload (HF /
cancer-specific C ≈ 0.85). The 10–22× cross-dataset band tracks feature
count: rfSRC's per-split exhaustive scan scales with p, so the gap
narrows on lower-p cohorts. ~95× speedup vs rfSRC built without OpenMP
(default R-on-macOS install).

**vs scikit-survival, paired on i7-14700K** — synthetic 2-cause Weibull,
p = 58, both libraries at their best config:

| n | sksurv `low_memory=True` | comprisk | speedup |
|---|---|---|---|
| 5 000 | 18.2 s | 1.10 s | **16.6×** |
| 50 000 | 2935 s (49 min) | 5.40 s | **544×** |

The gap widens super-linearly (sksurv ≈ n^2.2; comprisk ≈ n^0.7).
comprisk also provides Aalen-Johansen CIF + Nelson-Aalen CHF that
sksurv `low_memory=True` raises `NotImplementedError` for.

**Scaling on a consumer desktop:** n = 10⁶ in **63 s** on i7-14700K,
14.5 GB RSS. Reproducible via
[`validation/spikes/lambda/exp5_paper_scale_bench.py`](validation/spikes/lambda/exp5_paper_scale_bench.py).

## API

Full parameter list in [`src/comprisk/forest.py`](src/comprisk/forest.py);
usage by task in [docs/quickstart.md](docs/quickstart.md). Two splitrules
are available: `logrankCR` (composite competing-risks log-rank, default)
and `logrank` (cause-specific).

## Documentation

- [Quickstart](docs/quickstart.md) — common tasks with runnable code
- [PRD](docs/prd.md) — what comprisk aims to be at v1.0
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

```bibtex
@software{yang_comprisk_2026,
  author    = {Yang, Sunny and Zhao, Wanqi},
  title     = {{comprisk: a Python toolkit for competing risks}},
  year      = {2026},
  publisher = {Zenodo},
  version   = {0.3.1},
  doi       = {10.5281/zenodo.19876282},
  url       = {https://doi.org/10.5281/zenodo.19876282},
}
```

DOI is concept-level (always resolves to the latest version). GitHub's
"Cite this repository" button generates a version-specific record from
[`CITATION.cff`](CITATION.cff). Algorithmic references in
[`docs/REFERENCES.md`](docs/REFERENCES.md).
