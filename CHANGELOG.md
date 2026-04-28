# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.1.0] — 2026-04-27

Initial public release. Pre-alpha; API may change before 1.0.

### Added

- **`CompetingRiskForest`** — scikit-learn-compatible competing-risks
  random forest. Two split modes (`mode="default"`: histogram with
  uint8-binned features; `mode="reference"`: pure-NumPy exact splitting)
  and two split rules (`splitrule="logrankCR"` composite competing-risks
  log-rank with Lau-inclusive at-risk, and `splitrule="logrank"`
  cause-specific log-rank with optional `cause_weights`).
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
