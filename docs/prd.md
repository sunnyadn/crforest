# crforest: Product Requirements Document

Version 1.0 — scope for v0.1 → v1.0

## 1. Project Summary

crforest is a Python-native competing risks random forest library, scikit-learn compatible, designed to eliminate the Python-to-R workflow split that Python users currently endure for competing risks survival analysis.

Python users doing survival analysis today face a forced choice: use scikit-survival (which does not meaningfully support competing risks and OOMs on moderate datasets) or switch to R's randomForestSRC (correct and feature-complete but slow, memory-inefficient, and ecosystem-incompatible). crforest fills this gap with a CPU-based implementation that matches randomForestSRC in statistical capability while offering substantially better performance, memory efficiency, and developer experience within the Python ecosystem.

**Project status:** Pre-alpha. A NumPy reference implementation exists and aligns with randomForestSRC on 3/5 validation datasets within 0.01 C-index.

## 2. Problem Statement

### 2.1 The Python / R Split for Competing Risks

Applied researchers whose primary stack is Python — epidemiology, clinical outcomes, health economics, cardiovascular research — routinely hit a forced workflow split when they need competing risks analysis:

- Data cleaning, feature engineering, and EDA happen in Python (pandas, scikit-learn).
- Data is exported to R.
- A competing risks random forest is fit in randomForestSRC.
- Results are brought back to Python for downstream analysis and visualization.

This workflow is painful for three independent reasons:

- **Language split:** Two environments, two sets of code, two mental models. Fragile at boundaries, slow to iterate.
- **Fit time:** Single-fit on a moderate cohort (n ≈ 100k, p ≈ 60) takes several hours in randomForestSRC. 5-fold CV takes overnight. This is enough to push researchers toward guess-and-check hyperparameter tuning and to skip sensitivity analyses they would otherwise do.
- **Memory / output tradeoff:** scikit-survival's `RandomSurvivalForest` forces a choice. Its default `low_memory=False` (the only mode that supports `predict_cumulative_hazard_function` / `predict_survival_function`) stores per-leaf full CHF arrays and OOMs at moderate n (≈ 16.8 GB peak RSS at n = 5 000; >21.5 GB at n = 10 000 on a 24 GB host, n_jobs=1). Switching to `low_memory=True` fits at scale but disables CHF / survival-function predictions — only `predict()` (risk score) remains. Users wanting both at scale have no Python option.

### 2.2 Target Segment

Characteristics of the users this project is built for:

- Applied researchers (epidemiology, clinical outcomes, health economics) whose primary stack is Python.
- Need competing risks analysis (common in cardiovascular, oncology, geriatrics).
- Currently forced to R for this specific capability.
- Data regime: n = 10k–500k, p = 20–200, J = 2–4 competing causes.
- CPU environments (laptop, workstation, or server); GPU uncommon.

Estimated audience: hundreds to low thousands globally. Niche but underserved.

### 2.3 Why Existing Solutions Fall Short

| Tool | Language | CR Support | Performance | Memory | API |
|------|----------|------------|-------------|--------|-----|
| scikit-survival | Python | No real CR | Slow at scale (n^2.2 wall) | Memory/output tradeoff: full CHF outputs OOM at moderate n; `low_memory=True` fits but disables CHF / survival predictions | Good (sklearn) |
| randomForestSRC | R | Full | Slow | Good | R idioms only |
| pycox | Python | Deep learning only | N/A | N/A | Neural net focus |
| pysurvival | Python | Limited, abandoned | Unknown | Unknown | Poor |

No existing Python library combines correct CR implementation, good performance, and modern Python ergonomics. crforest targets exactly this combination.

## 3. Goals and Non-Goals

### 3.1 Goals (v1.0)

**Functional:**

- Correct implementation of competing risks random forest matching randomForestSRC behavior
- Support two split rules: cause-specific log-rank and composite log-rank (modified Gray's test deferred to v1.1; see §5.1 and §11)
- Aalen-Johansen cumulative incidence function (CIF) estimation at leaves
- Cause-specific and composite permutation-based variable importance

**Performance:**

- Single-fit on n=100k, p=60, J=2, 500 trees: ≤ 3 minutes on 16-core CPU
- 5-fold CV on same data: ≤ 20 minutes
- Peak memory on same data: ≤ 5 GB
- No OOM on 100k-row datasets on standard workstations (32 GB RAM)

**Developer Experience:**

- scikit-learn compatible API (fit/predict/set_params)
- pandas DataFrame-first I/O
- Built-in plotting helpers for common outputs
- Friendly warnings for low-event-rate and memory-risk situations
- Progress bars during training and CV

**Validation:**

- Statistical equivalence with randomForestSRC on benchmark datasets (paired-seed C-index)
- Reference NumPy implementation for CI ground truth

### 3.2 Non-Goals (Explicit)

**Not doing in v1.0 (possibly in v2.0):**

- GPU acceleration (users do not need it; adds install friction; out of scope)
- Streaming / out-of-core training
- Distributed training
- Oblique splits
- Time-varying covariates
- Multivariate survival outcomes
- Missing data imputation (document that users must pre-impute)
- Interval-censored data

**Not doing ever:**

- Bit-exact matching with randomForestSRC
- Publication of a methods paper as a precondition for release
- Becoming a general survival analysis framework (scope limited to CR forests)

**Explicit philosophy:**

- When in doubt, prefer ergonomics over features.
- When in doubt, prefer CPU correctness over GPU speed.
- When in doubt, prefer tested code over novel algorithms.

## 4. Target Users and Use Cases

Three personas representing the intended audience:

**Alex, cardiovascular outcomes PhD candidate.**

- Stack: Python (pandas, sklearn, matplotlib, jupyter) for 90% of work.
- Data: 50k–500k patient cohorts from EHR, registry, or biobank extracts.
- Analyses: Cause-specific CIF prediction, risk stratification, variable importance for feature selection, calibration analysis, subgroup analyses.
- Pain: Currently round-trips data to R for CR; wants to stop.
- Success: Delete the R scripts and do the full analysis in a single Jupyter notebook.

**Blair, epidemiology postdoc.** Works on cancer registry data. Needs CR because cancer-specific vs other-cause mortality must be separated. Collaborates with a biostatistician who uses R, but Blair's own code is Python. Uses crforest for prototyping, hands off to R for final modeling if required by journal.

**Chen, health-economics data scientist at a pharmaceutical company.** Builds risk prediction models for clinical trial design. Needs Python for integration with internal tooling. Values reliability and clear error messages over cutting-edge features.

**Non-users (out of scope):**

- Deep survival learning practitioners (use pycox, DeepSurv, DeepHit)
- Pure Cox regression users (use lifelines, statsmodels)
- Huge-scale (n > 10M) genomics survival (use specialized tools)
- Anyone needing interval censoring or recurrent events

## 5. Functional Requirements

### 5.1 Core Algorithm

**Tree construction:**

- Binary recursive partitioning with histogram-based split search
- Quantile-based feature binning, default 256 bins, stored as uint8
- Level-wise construction on CPU with joblib parallelization
- Bootstrap sampling per tree (with replacement, default) with out-of-bag tracking
- Random feature subsampling per node (mtry, default sqrt(p))

**Split rules (v1.0):**

- **Cause-specific log-rank** (`splitrule="logrank"`): Event-specific log-rank statistic for a user-specified cause (with optional weight vector across causes). Equivalent to randomForestSRC's logrank rule with cause weights.
- **Composite log-rank** (`splitrule="logrankCR"`, default): Weighted average of cause-specific log-rank statistics across all causes. Matches randomForestSRC's default for competing risks.

**Deferred to v1.1:**

- **Modified Gray's test** (`splitrule="logrankgray"`): Based on Gray's test for subdistribution hazards with IPCW weights, for CIF-focused splitting. Deferred because rfSRC does not expose an equivalent splitrule (so no paired-seed baseline is available), and applied-user value is marginal given logrankCR is the community default. Landing this post-v1.0 also moves crforest ahead of rfSRC on this axis. See §11.

**Leaf estimation:**

- Aalen-Johansen estimator for cause-specific CIFs
- Nelson-Aalen estimator for cause-specific cumulative hazards (available but CIF is primary output)
- Leaf storage: compact sufficient statistics (per-cause event counts and at-risk counts on a shared time grid of ≤200 points), not full CIF arrays

**Tree depth and stopping:**

- `max_depth` (default 15)
- `min_samples_leaf` (default `max(10, n_samples // 1000)`)
- `min_samples_split` (default `2 × min_samples_leaf`)

### 5.2 API

**Main class:**

```python
class CompetingRiskForest(BaseEstimator):
    def __init__(
        self,
        n_estimators: int = 500,
        splitrule: str = "logrankCR",   # or "logrank" (logrankgray deferred to v1.1)
        cause: int = 1,
        cause_weights: array-like | None = None,
        max_depth: int = 15,
        min_samples_leaf: int | None = None,  # auto if None
        min_samples_split: int | None = None,
        max_features: str | int | float = "sqrt",
        n_bins: int = 256,
        time_grid: int | array-like = 200,  # int = n points, array = custom
        bootstrap: bool = True,
        oob_score: bool = False,
        n_jobs: int = -1,
        random_state: int | None = None,
        verbose: bool = False,
        mode: str = "default",  # "default" or "reference"
    ): ...

    def fit(self, X, y=None, *, time=None, event=None, time_col=None, event_col=None): ...
    def predict_cif(self, X, times=None, causes=None) -> pd.DataFrame: ...
    def predict_risk(self, X, cause: int = 1) -> np.ndarray: ...
    def predict_chf(self, X, times=None, causes=None) -> pd.DataFrame: ...
    def score(self, X, y) -> float: ...  # cause-specific C-index

    @property
    def feature_importances_(self) -> pd.DataFrame: ...
    @property
    def oob_score_(self) -> float: ...
    @property
    def unique_times_(self) -> np.ndarray: ...
```

**Input flexibility (fit):**

```python
# Three supported input styles
forest.fit(X, y)  # y = structured array with fields ('event', 'time') or similar
forest.fit(X, time=t, event=e)  # explicit arrays
forest.fit(df, time_col="followup_months", event_col="event_type")  # DataFrame
```

X may be: pandas DataFrame, numpy array, polars DataFrame (converted internally).

**Output format (predict_cif):**

Returns a pandas DataFrame with a MultiIndex `(sample_id, time)` and columns for each requested cause:

```
              cause_1     cause_2
sample time
0      12     0.034       0.021
       24     0.067       0.045
       36     0.094       0.072
1      12     0.058       0.033
       ...
```

Also supports `format="wide"` for a 3D array output if the user prefers.

### 5.3 Variable Importance

Permutation-based VIMP computed on OOB samples (or user-provided validation set):

```python
vimp = forest.feature_importances_
# Returns pd.DataFrame:
#   feature  cause_1_vimp  cause_2_vimp  composite_vimp
#   0   age          0.034         0.021         0.028
#   1   bmi          0.012         0.005         0.009
#   ...
```

- **Cause-specific VIMP:** drop in cause-specific C-index after feature permutation
- **Composite VIMP:** drop in composite score

Computed lazily via `compute_importance(X_eval, y_eval)`, whose result is cached and returned by `feature_importances_`. Must not be computed during fit by default (expensive, not always needed). Auto-trigger on first `feature_importances_` access lands with the v1.1 OOB path, which can use stored training data; v1.0 requires the explicit call since held-out VIMP needs caller-supplied evaluation data.

### 5.4 Evaluation Metrics

Expose standard metrics as functions:

```python
from crforest.metrics import (
    concordance_index,         # cause-specific C-index
    brier_score,               # time-dependent Brier score
    integrated_brier_score,    # IBS over time interval
)
```

### 5.5 Visualization Helpers

```python
from crforest.plotting import (
    plot_cif,           # CIF curves for one or more samples
    plot_calibration,   # calibration plot at given time horizon
    plot_vimp,          # horizontal bar chart of VIMP, per cause
)
```

All return matplotlib Axes for further customization. Sensible defaults that produce publication-ready plots without tweaking.

### 5.6 Reference Mode

`mode="reference"` activates a pure-NumPy, exact-splitting implementation:

- No histogram binning (evaluates all unique feature values)
- Deterministic tie-breaking (lowest feature index, lowest threshold)
- Single-threaded
- 10-100× slower than default mode
- Used for: CI ground truth, debugging, and publication of results where exactness is requested

Reference mode must pass an equivalence test against default mode: paired C-index within 0.005 on same seed.

### 5.7 Error Handling and Warnings

**Issue UserWarning for:**

- Cause with < 100 events or < 1% of samples ("low event rate")
- Single cause present (not really competing risks; suggests single-event RSF)
- Estimated memory > 50% of system memory (pre-fit check)
- n_samples < 100 (too small for forest)
- All-censored subset in a tree's bootstrap sample

**Raise ValueError for:**

- Negative times
- Event codes outside declared range
- Missing values in X (future: support, for now reject with clear message)
- Inconsistent array lengths

Error messages must name the offending column and row index where applicable.

## 6. Non-Functional Requirements

### 6.1 Performance Targets

**Benchmark machine:** 16-core CPU, 64 GB RAM, n=100k, p=60, J=2, 500 trees, default parameters.

| Metric | Target | Stretch |
|--------|--------|---------|
| Single fit | ≤ 3 min | ≤ 1.5 min |
| 5-fold CV | ≤ 20 min | ≤ 10 min |
| Predict CIF on 10k × 200 times | ≤ 10 s | ≤ 3 s |
| Permutation VIMP | ≤ 5 min | ≤ 2 min |
| Peak RAM during fit | ≤ 5 GB | ≤ 3 GB |
| Peak RAM during CV | ≤ 8 GB | ≤ 5 GB |

**Comparison targets (same workload):**

| vs | Speed | Memory |
|----|-------|--------|
| scikit-survival (`low_memory=True`, single-event collapse) | 5×+ faster, gap widens with n (5.7× at n=5k → 64× at n=25k → projected 100×+ at n=100k); sksurv `low_memory=False` cannot fit at n ≥ 10k | sksurv `low_memory=False` OOMs; sksurv `low_memory=True` is more compact but lacks CHF / survival predictions (architectural tradeoff that crforest avoids) |
| randomForestSRC | 5–7× faster on real CHF n=75k (4.96× default; 6.13× best ntime config) | 2–5× less RSS (rfSRC peak n=75k = 14.7 GB; crforest n=100k peak = 7.4 GB) |

### 6.2 Correctness Validation

**Continuous (every commit):**

- Reference mode agrees with default mode: paired C-index ΔC < 0.005 on toy and PBC data, 10 seeds
- Reference mode matches hand-computed split decisions on 3 toy datasets
- CIF output is monotone non-decreasing, bounded in [0, 1], and CIFs across causes sum to ≤ 1

**Pre-release (v0.1, v1.0):**

- Paired 100-seed comparison with randomForestSRC on 5 benchmark datasets (PBC, follic, hd, veteran, synthetic CR)
- Median |ΔC-index| < 0.01 on all datasets
- Distribution of ΔC-index no wider than randomForestSRC's own seed-to-seed variance

**Pre-release, manual:**

- With all randomness disabled (no bootstrap, mtry=p, exact splitting), split decisions at root and first 3 levels match randomForestSRC on PBC

### 6.3 Compatibility

- **Python:** 3.10, 3.11, 3.12 (drop 3.9 and earlier; too old by release date)
- **OS:** Linux, macOS (Intel and Apple Silicon), Windows
- **Dependencies (hard):** numpy>=1.24, scipy>=1.10, pandas>=2.0, joblib>=1.3, numba>=0.58 (for histogram kernels)
- **Dependencies (soft, optional):** polars, matplotlib, tqdm, scikit-survival (for benchmark comparisons)

### 6.4 Installation

```bash
pip install crforest
```

No compilation required by end user (ship wheels for major platforms). No CUDA, no C++ compiler needed.

### 6.5 Packaging

- `pyproject.toml` with PEP 621 metadata
- Build backend: hatchling or setuptools
- CI builds wheels for Linux (x86_64), macOS (x86_64 + arm64), Windows (x86_64) via cibuildwheel
- Published to PyPI under `crforest`
- Optional: conda-forge recipe after v0.2

## 7. Documentation

### 7.1 Required Documentation

- **README:** Project overview, install, quickstart, comparison with R, scope/non-goals
- **User Guide** (on readthedocs): Progressive tutorials
  - Installation
  - First competing risks analysis (5 min tutorial)
  - Understanding split rules
  - Variable importance
  - Visualization
  - Cross-validation and model selection
  - Comparison with randomForestSRC (for R users)
  - FAQ and troubleshooting
- **API Reference:** Auto-generated from docstrings
- **Example Notebooks (3–5):**
  - PBC dataset walkthrough (classic small example)
  - Cardiovascular cohort analysis (sanitized version of a real use case)
  - High-dimensional example (p > n)
  - Calibration and model diagnostics
  - Migration from randomForestSRC (side-by-side R and Python code)

### 7.2 Documentation Platform

- Hosted on ReadTheDocs
- Built with mkdocs-material or Sphinx with furo theme
- Example notebooks rendered via nbsphinx or myst-nb
- Versioned docs (latest, stable, and per-version)

### 7.3 Docstring Standards

Every public class and method documented in NumPy style, including:

- Parameters with types and defaults
- Returns with type and description
- Raises with conditions
- Examples (doctest-compatible where possible)
- References to relevant papers (Ishwaran 2014, Sverdrup 2025, Gray 1988) for algorithmic choices

## 8. Roadmap

Work is organized into phases. Each phase delivers a self-contained increment that can be released or evaluated on its own. Phase ordering reflects dependencies, not a fixed schedule.

- **P0 — Setup:** Repo structure, packaging, CI skeleton.
- **P1 — Reference mode:** Pure-NumPy exact-splitting implementation. Paired-seed validation against randomForestSRC on five benchmark datasets.
- **P2 — Histogram + memory:** CPU histogram split search, uint8 bin indices, compact leaf storage, level-wise tree construction, joblib parallelism.
- **P3 — CR feature completeness:** All three split rules, CIF/CHF prediction, permutation VIMP, cause-specific C-index.
- **P4 — Ergonomics and docs:** Plotting helpers, DataFrame I/O, warnings, progress bars, complete documentation.
- **P5 — v0.1 release:** PyPI publication and public announcement.
- **P6 — Field integration:** Real-workload integration, bug fixes, API refinement based on feedback.
- **P7 — v1.0 release:** API freeze, stable release.

## 9. Testing Strategy

### 9.1 Unit Tests

- Every public function tested on at least three inputs (normal, edge case, error case)
- Property-based testing (via hypothesis) for statistical invariants (CIF monotonicity, bounds, permutation invariance)
- Reference mode decisions validated against hand-computed truth on three toy datasets

### 9.2 Integration Tests

- Full fit/predict pipeline on PBC, follic, synthetic CR datasets
- DataFrame / NumPy / Polars input all produce identical results
- Reference vs default mode equivalence on fixed seeds

### 9.3 Regression Tests

- Benchmark suite tracking performance on a fixed machine
- Performance regression > 20% on any benchmark fails CI
- Memory regression > 50% fails CI

### 9.4 R Cross-Validation

- Offline test suite that runs the same analyses in R via rpy2 or pre-computed R outputs stored as parquet
- Run monthly, not on every commit (slow)
- Ensures statistical equivalence with randomForestSRC doesn't drift

### 9.5 Coverage

- Target ≥ 85% line coverage, ≥ 70% branch coverage
- Enforced in CI via pytest-cov

## 10. Release and Distribution

### 10.1 Versioning

- Semantic versioning (major.minor.patch)
- v0.x: API may change between minor versions (user warning in release notes)
- v1.0: API stable; breaking changes require major version bump

### 10.2 Release Process

- Releases tagged in git, published to PyPI via GitHub Actions
- Changelog maintained in `CHANGELOG.md` (keep-a-changelog format)
- Release notes summarize new features, breaking changes, migrations

### 10.3 License

MIT License. Permissive to maximize adoption.

## 11. Future Scope (Post-v1.0)

Candidates for v1.x and v2.x, explicitly out of scope for v1.0:

- **Modified Gray's test splitrule (`logrankgray`, v1.1 candidate):** IPCW-weighted Gray's test for subdistribution hazards, for CIF-focused splitting. Deferred from v1.0 per §5.1; rfSRC has no equivalent splitrule, so validation anchors on invariants + cross-mode equivalence rather than paired-seed ΔC.
- **GPU backend (v2.0 candidate):** `device="cuda"` or `device="auto"` option with Triton kernels, primarily for A6000/A100-class cards. Only if demand materializes.
- **Oblique splits:** Linear combinations of features, following Jaeger et al. (2022).
- **Missing data handling:** Surrogate splits or MIA (missing incorporated in attributes).
- **Time-varying covariates:** Analogous to randomForestSRC's tdc support.
- **Dynamic prediction / landmarking:** For use with longitudinal data.
- **Survival-specific feature engineering:** Built-in utilities for common transforms.
- **R bindings:** Reverse direction — let R users call crforest from R. Low priority.

## 12. Open Questions

These remain unresolved and may be revisited during development:

1. **Time grid default policy:** Fixed 200 points quantile-spaced, or adaptive based on data? Default 200 for now; revisit if users report artifacts.
2. **OOB calibration check in CI:** Should every fit check calibration on OOB and warn if poorly calibrated? Probably not (expensive); make it opt-in.
3. **DataFrame dependency:** Hard require pandas, or only require if DataFrame input used? Decision: hard requirement in v1.0; too central to modern Python workflow to make optional.
4. **Polars first-class or lazy adapter:** Full polars support or just conversion at input? Decision: conversion at input for v1.0, revisit.

## 13. Success Criteria (v1.0)

- Library installs cleanly on Linux, macOS, Windows via `pip install`.
- Benchmark performance targets (section 6.1) met.
- Statistical validation against randomForestSRC passes (section 6.2).
- Documentation deployed on ReadTheDocs, tutorials runnable end-to-end.
- At least one real competing-risks workflow that previously required R can be completed end-to-end in Python using crforest.

## 14. Governance

### 14.1 Contributions

Contributions are welcome once the project reaches v0.1. Guidelines (`CONTRIBUTING.md`) and a code of conduct (Contributor Covenant) will be published at that point. Substantive changes require maintainer review.

### 14.2 Breaking Changes Policy

- v0.x: Breaking changes allowed with release note flag.
- v1.x: Breaking changes require a deprecation warning in one minor version before removal.
- v2.0+: Clear migration guide required for any breaking change.

### 14.3 Issue and PR Response

Aspirational, not contractual:

- Acknowledge new issues within 1 week.
- Triage bug reports within 2 weeks.
- Release fix for severity-high bugs within 1 month.
- Feature requests: no SLA; evaluated at minor version planning.

## 15. Appendix

### 15.1 References

- Ishwaran, H. et al. (2008). Random Survival Forests. *Annals of Applied Statistics.*
- Ishwaran, H. et al. (2014). Random Survival Forests for Competing Risks. *Biostatistics.*
- Gray, R. J. (1988). A class of k-sample tests for comparing the cumulative incidence of a competing risk. *Annals of Statistics.*
- Sverdrup, E., Yang, J., LeBlanc, M. (2025). Efficient Log-Rank Updates for Random Survival Forests. arXiv:2510.03665.
- randomForestSRC R package documentation: https://www.randomforestsrc.org/

### 15.2 Glossary

- **CIF:** Cumulative Incidence Function. Probability of experiencing a specific cause of event by time t.
- **Competing risks:** Setting where multiple mutually exclusive event types can occur, and occurrence of one precludes others.
- **Cause-specific hazard:** Hazard for a specific cause in the presence of other causes.
- **Subdistribution hazard:** Hazard computed keeping individuals who experienced competing events in the risk set (Fine-Gray framework).
- **OOB:** Out-of-bag. Samples not included in a tree's bootstrap sample, used for unbiased performance estimates.
- **VIMP:** Variable importance, typically permutation-based for survival forests.
