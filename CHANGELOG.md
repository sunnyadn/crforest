# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

> **Project rename in 0.3.1**: this package was published as `crforest` for
> 0.1.0 → 0.3.0. From 0.3.1 it is named `comprisk`. All earlier entries
> describe releases that shipped on PyPI as `crforest`; the import path
> `from crforest import …` was the supported form for those versions. See
> the 0.3.1 entry below for the migration recipe.

## [Unreleased]

### Added

- `PenalizedFineGrayRegression` — penalized variable selection for the
  Fine-Gray subdistribution-hazards model (SUN-73), the first such
  estimator in the Python competing-risks ecosystem. Cyclic coordinate
  descent on the IPCW-weighted partial (pseudo-)likelihood with
  soft-/firm-thresholding updates, warm-started along a data-driven
  `lambda` path from `lambda_max` (the smallest value zeroing every
  penalized coefficient) down to `lambda_min_ratio * lambda_max`.
  Penalties: LASSO, ridge, elastic-net (`l1_ratio` blend), MCP
  (Zhang 2010) and SCAD (Fan & Li 2001). Reuses the v0.4 Fine-Gray IPCW
  machinery (no materialized Geskus expansion). `cv=K` selects `lambda`
  by the Verweij–van Houwelingen cross-validated partial-likelihood
  deviance (`lambda_min_`, `lambda_1se_`), with `n_jobs` parallelizing the
  CV folds; otherwise BIC over the path.
  Sandwich SEs along the path follow Fu et al. (2017). sklearn-compatible
  (`BaseEstimator`; `fit` / `predict` / `predict_cumulative_incidence`;
  `coef_path_`, `lambdas_`, `bic_path_`, …). Matches R `crrp::crrp()`
  (Fu et al. 2017) coefficients **and** sandwich SEs along the full path
  to ≤ 3e-6 on `pbc` (n = 276) and `follic` (n = 541) for LASSO / MCP /
  SCAD; the `lambda → 0` limit reproduces `FineGrayRegression` to ≤ 1e-3.

### Changed

- `CompetingRiskForest.shap_values()` is dramatically faster (SUN-74),
  with no change in output (additivity and bit-exact agreement with
  `shap.TreeExplainer` at a fixed `(cause, time)` slice are preserved).
  The TreeSHAP recursion now emits only the *structural* per-`(leaf,
  feature)` weights; the `(n_causes, n_times)` leaf tensors are folded
  back in by a single BLAS matmul per tree (`phi = W @ leaf_table_2d`),
  taking the `n_causes × n_times` factor out of the hot recursion. When a
  small `times=` subset is requested the leaf table is projected onto
  those columns *before* the matmul. Two further fixes: the recursion's
  scratch path-arrays are now sized by tree height rather than node count
  (they were over-allocated by ~10⁴× on deep, wide trees — the dominant
  cost), and per-tree contributions are reduced in worker-local
  accumulators instead of one full-array `+=` per tree. Measured ~14×
  faster on an 80-tree, depth-15 forest (n_test = 500: 171 s → 12 s).

## [0.4.0] — 2026-05-11

Broadens comprisk from "CR random forest" to a CR toolkit: the four
canonical CR regression / non-parametric methods (Fine-Gray, Aalen-
Johansen, Gray's test, cause-specific Cox — SUN-45), exact TreeSHAP for
the forest (SUN-43), and a CR-aware model-evaluation harness with time-
dependent AUC / Brier / IBS / iAUC and quantile-decile calibration data
(SUN-60, SUN-64).

### Added

- `FineGrayRegression` — proportional subdistribution-hazards regression
  (Fine & Gray 1999) via IPCW-weighted Breslow partial likelihood with
  Newton-Raphson + Armijo line search. Mathematically equivalent to
  Geskus (2011)'s expanded-data + weighted-Cox formulation but without
  the row-blowup. Matches `R cmprsk::crr()` defaults to floating-point
  noise on three reference datasets (synth, pbc, follic): max |Δβ| =
  1.4e-15, max |Δlog-lik| = 6.6e-12.
- `FineGrayRegression(robust_se=True)` returns a per-subject score-residual
  cluster sandwich; agrees with `cmprsk`'s IPCW-corrected sandwich SE to
  ~3 digits per Geskus 2011 (Therneau, R `survival::finegray` docs).
  Worst-case observed Δse = 4.07e-04 (within the 1e-3 acceptance bar).
- `FineGrayRegression.predict_cumulative_incidence(X, times=)` — predicted
  CIF curves under the cmprsk closed-form
  `F(t|x) = 1 - exp(-Λ̂_0(t) * exp(x'β))`.
- `CumulativeIncidence` — non-parametric Aalen-Johansen estimator
  (Aalen 1978; Aalen & Johansen 1978) with optional group stratification
  and the Pepe (1991) Greenwood-corrected pointwise variance.
  Independent implementation; bit-identical (atol 1e-9) to
  `R cmprsk::cuminc()` on grouped synthetic and follic.
- `CauseSpecificCox` — standard Cox PH on `Surv(time, event == cause)`
  with cause-specific censoring of competing events. Matches
  `R survival::coxph(method="breslow")` to 1e-9 on pbc and follic.
- `gray_test` — Gray's K-sample test for cumulative incidence functions
  (Gray 1988). Independent implementation derived from the paper plus
  counting-process martingale theory; matches `R cmprsk::cuminc()$Tests`
  (statistic and p-value) to floating-point noise on grouped synthetic
  and follic-by-clinstg fixtures.
- `CompetingRiskForest.shap_values(X)` — exact cause-specific TreeSHAP
  attributions (Lundberg et al. 2018, Algorithm 2; O(L·D²) per tree)
  returning `(shap, base)` of shape `(n, p, n_times, n_causes)`;
  attributions + baseline reconstruct the CIF (additivity). numba+nogil
  per-tree kernel, parallel over the ensemble. New private module
  `comprisk._shap`.
- `score_cr` — CR-aware model evaluation: IPCW-weighted time-dependent
  AUC and Brier score (Uno-style censoring correction under competing
  risks), plus integrated AUC (iAUC) and integrated Brier score (IBS),
  with optional bootstrap confidence intervals. One-call replacement for
  the AUC/Brier block of `R riskRegression::Score()` in CR mode; takes a
  dict of named candidate models. Returns a `ScoreResult` dataclass.
- `calibration_cr` (also reachable via `score_cr(..., calibration_at=)`)
  — tidy / long-form quantile-decile calibration table per
  `(model, time, bin)`: predicted bin midpoint, Aalen-Johansen empirical
  CIF on the bin's subjects, and a per-bin Wilson 95% interval. One-call
  replacement for `R riskRegression::plotCalibration(method="quantile",
  q=10)`; per-bin observed frequency matches the R reference within
  5.55e-16 on the pbc held-out fold.
- Test fixtures: `tests/cross_check_cmprsk.R` /
  `tests/cross_check_calibration.R` (R-side reference generators),
  `tests/fixtures/cmprsk_*`, `cuminc_*`, `csc_*`, `gray_*`, `calib_pbc.csv`
  (committed CSV reference fits).

### Fixed

- `CumulativeIncidence(cause_codes=[k])` silently overestimated the
  cause-`k` CIF (and got the variance wrong) when the data contained
  events outside `cause_codes`: the Kaplan-Meier survival decrement was
  summed only over the requested causes, hiding competing events from the
  at-risk dynamics. The recursion now always runs over every observed
  cause; `cause_codes` only selects which curves are returned.
  `cause_codes=None` is unchanged (SUN-71).

## [0.3.1] — 2026-05-04

Project rename and reposition. Code is identical to 0.3.0; this release
exists to claim the new package name on PyPI, broaden the package scope
beyond "competing-risks Random Forest" to "Python toolkit for competing
risks," and announce the v0.4 roadmap (Fine-Gray subdistribution-hazard
regression + stand-alone Aalen-Johansen CIF + Gray's K-sample test +
cause-specific Cox PH).

### Changed

- **Package renamed `crforest` → `comprisk`.** `pip install comprisk`,
  `from comprisk import CompetingRiskForest`. The `crforest` PyPI package
  is deprecated with a pointer to `comprisk`; the GitHub URL
  `github.com/sunnyadn/crforest` auto-redirects to
  `github.com/sunnyadn/comprisk`.
- README rewritten to lead with the CR-toolkit framing and add an explicit
  Roadmap section.
- pyproject `description` rewritten to match the new framing.
- `CITATION.cff` title and abstract updated; version bumped to 0.3.1.

### Migration

```python
# before (crforest 0.1.0 – 0.3.0)
from crforest import CompetingRiskForest

# after (comprisk ≥ 0.3.1)
from comprisk import CompetingRiskForest
```

API surface is unchanged; the rename is a one-line sed across user code.

## [0.3.0] — 2026-05-03

Adds Ishwaran-style minimal-depth variable selection. Partner-blocked
feature (SUN-42); ships ahead of SHAP support (SUN-43).

### Added

- `CompetingRiskForest.minimal_depth(threshold='md', return_extra=False) -> pd.DataFrame`
  — variable selection via mean minimal split depth across the forest, with
  the forest-averaged null-distribution threshold from Ishwaran et al.
  (2010, JASA, "High-Dimensional Variable Selection for Survival Data",
  Theorem 1 + Section 3).
- Sentinel for unused variables follows the paper's Eq. (2) convention
  (depth = D(T), the deepest leaf depth in the tree). The threshold is
  computed once from forest-averaged ℓ̄_d and D̄ per Section 3, not as a
  per-tree mean — matching the paper's recommendation. `randomForestSRC`'s
  default `max.subtree` aggregation is tree-averaged and produces a
  different numeric threshold; variable rankings tend to agree.
- Works on all three tree backends (default `FlatTree`,
  `equivalence='rfsrc'` `HistTreeNode`, `mode='reference'` `RefTreeNode`).
- Bit-equivalent ranking + per-feature mean minimal depth values vs
  `randomForestSRC::max.subtree(max.order=1)` under `equivalence='rfsrc'`
  with matched fit config (`bootstrap=False`, `min_samples_split=2*nodesize`,
  `min_samples_leaf=1`, `max_depth=None`). Verified on the bundled `follic`
  dataset (oracle: `tests/fixtures/rfsrc_var_select_follic.json`); per-tree
  trees are bit-identical at ntree=100. Note: the threshold *scalar* differs
  because comprisk implements the paper's forest-averaged threshold
  (Section 3) while rfSRC defaults to tree-averaged; rankings agree.
- Known limitation: `bootstrap=True` retains a residual ~0.003 p95 ΔCIF
  (RNG stream B shift); SUN-44 tracks the fix.

## [0.2.0] — 2026-05-02

Performance + scope expansion. Single-machine fit is ~6–7× faster than
0.1.2 on real EHR-shaped data; cross-library speedup vs randomForestSRC
is now anchored on real-cohort matched-pair benchmarks (CHF n=75k,
SEER n=238k) instead of synthetic Gaussian data.

### Added

- `predict_oob_risk()` and `oob_score()` on `CompetingRiskForest`,
  exposing the existing per-tree OOB infrastructure for out-of-bag risk
  prediction and OOB-based hyperparameter selection (no inner CV
  required). Drops fits-per-candidate from 3 to 1 in forest-internal
  hyperparameter tuning, with outer-val C-index equivalent to 3-fold CV
  within ±0.001 across folds.
- Real-cohort matched-pair benchmark harnesses:
  - SEER breast cancer 2010–2015, n=238 057 / p=17, paired with rfSRC
    (`validation/comparisons/seer_path_b.py`,
    `_seer_path_b_rfsrc.R`, `validation/gen_seer_breast.py`,
    `validation/comparisons/SEER_README.md`).
  - n75k path-b matched-pair (CHF cohort, n=75 278 / p=58),
    reproducible across mac M4 / i7-WSL2 / HPC Xeon.
- `bench/` subtree: aligned vs-rfSRC reference benchmark scripts
  (R + Python) and results CSV.
- Scaling-curve figure (`docs/figures/scaling_curve.svg`) plus
  reproducible generator (`validation/figures/scaling_curve.py`).
- Profiling helper `validation/profile_fit.py` (RSS + wall + per-stage
  timing).
- Tests: `test_estimators.py` (sklearn estimator interface) and
  `test_oob_predict_score.py` (OOB API contract).
- `CITATION.cff` for canonical citation metadata.

### Changed

- Per-leaf Aalen–Johansen prediction is now vectorized — eliminates the
  previous ntree cliff at large forests. Single-machine fit on the
  partner CHF feature-selection workload is ~6–7× faster than 0.1.2
  with bit-identical outputs (validated fold-5 outer-val C-index
  unchanged at 0.8650 to four decimal places).
- Cross-library perf claim revised to honest real-data band. README
  headline now reads "10–22× faster than randomForestSRC on real
  EHR-shaped data" with per-cohort breakdown (CHF 14–22× across three
  machines; SEER 11.6× on HPC). The previous synthetic-Gaussian
  ~200–375× headline is supplementary; that workload doesn't represent
  clinical-EHR feature mixes.
- README split into a tight homepage + `docs/benchmarks.md` for the
  deep tables.
- Project status bumped pre-alpha → alpha
  (`Development Status :: 3 - Alpha`).

### Fixed

- pyproject `description` field updated from the stale "4.5–6× faster"
  tagline to match the current README headline (same bug class as the
  v0.1.1 README fix — corrected at the source this time).
- Validation scripts pass CI ruff lint (`seer_path_b.py`,
  `gen_seer_breast.py`).

## [0.1.2] — 2026-04-28

### Changed

- PyPI license metadata now uses the PEP 639 SPDX expression
  (`license = "Apache-2.0"` + explicit `license-files = ["LICENSE", "NOTICE"]`)
  instead of the legacy `license = { file = "LICENSE" }` form, which
  caused the entire Apache-2.0 license body to be dumped into the
  rendered `License:` field on the PyPI project page. Drops the now-
  redundant `License :: OSI Approved :: Apache Software License`
  classifier per PEP 639. Bumps build requirement to `hatchling>=1.27`
  for SPDX support.

### Note

This release is also the first one Zenodo archives — the GitHub-Zenodo
integration was enabled after 0.1.1, so 0.1.0 and 0.1.1 do not have
DOIs. Cite v0.1.2 (or later) as the canonical reference.

## [0.1.1] — 2026-04-28

### Fixed

- README tagline cited "5–7× faster than randomForestSRC", inconsistent
  with the Highlights bullet ("4.5–6×") and the pyproject Summary
  metadata that already shipped the corrected number. README tagline
  now matches.

## [0.1.0] — 2026-04-27

Initial public release. Pre-alpha; API may change before 1.0.

### Added

- **`CompetingRiskForest`** — scikit-learn-compatible competing-risks
  random forest. Two split modes (`mode="default"`: histogram with
  uint8-binned features; `mode="reference"`: pure-NumPy exact splitting)
  and two split rules (`splitrule="logrankCR"` composite competing-risks
  log-rank with Lau-inclusive at-risk, and `splitrule="logrank"`
  cause-specific log-rank with optional `cause_weights`).
- **sklearn drop-in surface** — `fit(X, y)` and `score(X, y)` accept the
  scikit-survival-style structured ``y`` (build via ``Surv.from_arrays(event,
  time)``); the legacy ``fit(X, time, event)`` form keeps working.
  ``predict(X)`` aliases ``predict_risk(X, cause=1)`` so the estimator
  drops into ``cross_val_score`` / ``KFold`` / ``Pipeline`` without a
  wrapper.
- **Predict API** — `predict_cif`, `predict_chf`, `predict_risk`, `score`.
  Cumulative incidence (Aalen-Johansen) and cumulative hazard
  (Nelson-Aalen) tables are materialised lazily on first predict from
  per-leaf integer event/at-risk counts; right-continuous step
  interpolation onto user-supplied times.
- **Permutation variable importance** (`compute_importance`) — OOB
  Breiman or held-out, scored with the Uno IPCW C-index. Returns a
  DataFrame with per-cause and composite columns. Bit-equivalent across
  `n_jobs` for a fixed `random_state`.
- **Concordance metrics** (`comprisk.metrics`) — Wolbers cause-specific
  C-index, Uno IPCW weights with ESS-truncation gating, and Uno IPCW
  cause-specific C-index for competing risks.
- **rfSRC equivalence preset** — `equivalence="rfsrc"` reproduces
  randomForestSRC's per-tree mtry/nsplit RNG stream and exposes the
  `inbag_` attribute for `bootstrap="by.user"` paired fits.
- **Thread-parallel fit** — `n_jobs` parallelises tree building over
  joblib threads in default mode (numba split kernel releases the GIL);
  fit output is bit-identical across `n_jobs` for a fixed `random_state`.
- **Coarse-grid split search** — `split_ntime` parameter coarsens the
  log-rank time grid for split selection while leaves keep the full
  grid for CIF/CHF output. Default `10`.
- **CUDA preview** — optional `device="cuda"` backend for default-mode
  fitting via the `comprisk[gpu]` extra (cupy + CUDA 12). Faster only at
  low feature count today; full GPU rewrite scheduled for v1.1.

### Performance (v0.1 reference workload)

Same-machine benchmark, real CHF cohort with HF/death competing risks
(n=75 000, p=58, ntree=100, 24-thread CPU): comprisk **22.5 s** vs
randomForestSRC 111.7 s = **4.96× faster** at tied HF Harrell C-index
(0.8642 vs 0.8643). Apples-to-apples vs rfSRC's best `ntime` config:
**6.13× faster**. UKB-scale feasibility check (n=1 000 000) completes in
122 s on commodity CPU.

### Documentation

- README with runnable Quickstart.
- `docs/quickstart.md` — data format, prediction shapes, scoring,
  cross-validation, VIMP, performance levers, GPU preview, rfSRC
  migration recipe.
- `docs/REFERENCES.md` — algorithmic provenance with full paper
  citations (Park-Miller, Bays-Durham, Numerical Recipes, Knuth,
  Wolbers 2009, Uno 2011, Cole & Hernán 2008, Kaplan-Meier 1958,
  Breiman 2001, Ishwaran 2008/2014).
- `docs/equivalence-vs-rfsrc.md` — cross-library validation methodology.
- `docs/prd.md` — product requirements scope through v1.0.

### License

Apache-2.0 (see [LICENSE](LICENSE) and [NOTICE](NOTICE)).
