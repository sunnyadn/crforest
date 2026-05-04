# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

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
  because crforest implements the paper's forest-averaged threshold
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
- **Concordance metrics** (`crforest.metrics`) — Wolbers cause-specific
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
  fitting via the `crforest[gpu]` extra (cupy + CUDA 12). Faster only at
  low feature count today; full GPU rewrite scheduled for v1.1.

### Performance (v0.1 reference workload)

Same-machine benchmark, real CHF cohort with HF/death competing risks
(n=75 000, p=58, ntree=100, 24-thread CPU): crforest **22.5 s** vs
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
