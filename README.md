# comprisk

[![PyPI version](https://img.shields.io/pypi/v/comprisk.svg)](https://pypi.org/project/comprisk/)
[![CI](https://github.com/sunnyadn/comprisk/actions/workflows/ci.yml/badge.svg)](https://github.com/sunnyadn/comprisk/actions/workflows/ci.yml)
[![DOI](https://img.shields.io/badge/DOI-10.5281%2Fzenodo.19876282-blue)](https://doi.org/10.5281/zenodo.19876282)

**comprisk** — a Python toolkit for competing risks. v0.3 ships a
scalable, scikit-learn-compatible competing-risks random survival forest;
v0.4 adds Fine-Gray subdistribution-hazard regression, a stand-alone
Aalen-Johansen cumulative-incidence estimator, Gray's K-sample test, and
cause-specific Cox PH (see [Roadmap](#roadmap)). Designed to remove the
Python → R workflow split that applied researchers currently endure for
competing-risks survival analysis.

> **Status: alpha.** API and internals may change before v1.0.
> **Renamed from `crforest` in 0.3.1** — `pip install comprisk`,
> `from comprisk import CompetingRiskForest` (see
> [Migrating from crforest](#migrating-from-crforest)).

## Highlights

- **Forest today, regression next.** v0.3 ships the only native Python
  competing-risks RSF (cause-specific log-rank splitting + composite CR
  log-rank, Aalen-Johansen CIF, Nelson-Aalen CHF, Wolbers + Uno IPCW
  concordance, OOB Breiman VIMP, Ishwaran minimal-depth variable
  selection). v0.4 adds Fine-Gray regression and the rest of the canonical
  CR toolbox (see [Roadmap](#roadmap)).
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

### Variable selection

Rank features by Ishwaran's minimal-depth criterion and apply the forest-
averaged null-distribution threshold from Ishwaran et al. (2010, JASA,
Theorem 1 + Section 3):

```python
forest = CompetingRiskForest(n_estimators=200, random_state=0).fit(X, time, event)
vs = forest.minimal_depth()
selected = vs.loc[vs["selected"], "feature"].tolist()
```

Variables with mean minimal depth below the threshold are flagged as
informative. Pass `return_extra=True` to additionally inspect quartiles
and per-feature usage rates across trees.

Note on rfSRC compatibility: this implements the paper's forest-averaged
threshold (Section 3); `randomForestSRC::max.subtree` defaults to a
tree-averaged threshold, so the threshold *scalar* differs. Per-feature
mean minimal depth values are bit-equivalent under matched fit config
(`equivalence='rfsrc', bootstrap=False, min_samples_split=2*nodesize,
min_samples_leaf=1, max_depth=None`).

See [docs/quickstart.md](docs/quickstart.md) for the full walkthrough — data
format, prediction shapes, cross-validation, GPU, and migrating from rfSRC.

> **scikit-learn drop-in.** `CompetingRiskForest` is a real sklearn
> estimator (`BaseEstimator`, `clone()`-friendly, picklable).
> `cross_val_score`, `KFold`, `Pipeline` work without a wrapper — pass
> `Surv.from_arrays(event, time)` as the `y` argument, or use the legacy
> 3-arg `fit(X, time, event)` form. Full example in
> [docs/quickstart.md § Cross-validation](docs/quickstart.md#cross-validation).

## Roadmap

comprisk is positioned as a complete, Python-native CR toolkit. The 12-month
scope is locked to competing-risks methods only — generalist survival
methods (general Cox PH, AFT, parametric, deep-survival, Kaplan-Meier as
a standalone API) are out of scope; use
[lifelines](https://lifelines.readthedocs.io/) or
[scikit-survival](https://scikit-survival.readthedocs.io/) for those.

| Version  | Module                                                | Status               |
|----------|-------------------------------------------------------|----------------------|
| **v0.3** | `CompetingRiskForest` (CR-RSF)                        | Shipped              |
| v0.4     | `FineGrayRegression` (subdistribution hazard)         | Planned (Q3-Q4 2026) |
| v0.4     | `CumulativeIncidence` (stand-alone Aalen-Johansen)    | Planned (Q3-Q4 2026) |
| v0.4     | `gray_test` (Gray's K-sample log-rank)                | Planned (Q3-Q4 2026) |
| v0.4     | `CauseSpecificCox` (CR-aware censoring)               | Planned (Q3-Q4 2026) |
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

## Migrating from crforest

comprisk **0.3.1** is the same codebase as crforest 0.3.0 under a new name
and a slightly broader scope. Update one line:

```python
# before
from crforest import CompetingRiskForest

# after
from comprisk import CompetingRiskForest
```

The package is otherwise identical — same `CompetingRiskForest` API,
same `equivalence="rfsrc"` mode, same minimal-depth feature selection,
same GPU preview, same metrics module. Pin to `comprisk>=0.3.1` (or
`crforest==0.3.0` on the legacy name); the `crforest` PyPI package will
emit a deprecation pointer and stop receiving new releases. The GitHub
URL `github.com/sunnyadn/crforest` auto-redirects to
`github.com/sunnyadn/comprisk`.

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
