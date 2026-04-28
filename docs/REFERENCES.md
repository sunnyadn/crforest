# Algorithmic References

This document lists the published references that ground each non-trivial
algorithm in `src/crforest/`. It is the authoritative source for the
mathematical provenance of the package; every numerical literal,
distributional assumption, and algorithmic step in the cited modules
traces to one of the entries below.

The package is also informed by reference behaviour from
[`randomForestSRC`](https://cran.r-project.org/package=randomForestSRC)
(rfSRC; Ishwaran & Kogalur, GPL-3 licensed), which was used as a
benchmarking target during development. The implementation under
`src/crforest/` is independent — see the per-module clean-room rewrite
commits — and its public API uses the string `equivalence='rfsrc'`
purely as a behavioural label, not as a redistributed component.

The validation harness under `validation/alignment/_rfsrc_patches/`
contains GPL-3 patches against the rfSRC C source for instrumentation
during equivalence diagnostics. That subtree is **excluded from the
PyPI sdist allowlist** (`pyproject.toml [tool.hatch.build.targets.sdist]
include`); see `validation/alignment/_rfsrc_patches/README.md`.

---

## Random number generation (`src/crforest/_aligned_rng.py`)

* **Park, S.K. & Miller, K.W.** (1988). "Random number generators: good
  ones are hard to find." *Communications of the ACM* 31(10): 1192–1201.
  → Source of the minimum-standard LCG parameters
  `(IA, IM, IQ, IR) = (16807, 2**31 − 1, 127773, 2836)` used by `_Ran1Stream`.

* **Bays, C. & Durham, S.D.** (1976). "Improving a poor random number
  generator." *ACM Transactions on Mathematical Software* 2(1): 59–64.
  → 32-slot shuffle-table convention layered on top of Park-Miller.

* **Press, W.H., Teukolsky, S.A., Vetterling, W.T., Flannery, B.P.** (1992).
  *Numerical Recipes in C: The Art of Scientific Computing*, 2nd ed.,
  Cambridge University Press. §7.1 ("Uniform Deviates").
  → Combines Park-Miller + Bays-Durham as the routine named `ran1`
    (constants and warmup procedure exactly as implemented here);
    fixes the endpoint cap `RNMX = 1 − 1.2e-7`; tabulates the auxiliary
    Lehmer-LCG parameters `(m, a, c) = (714025, 1366, 150889)` used by
    `derive_per_tree_seeds`.

* **Knuth, D.E.** (1997). *The Art of Computer Programming, Vol. 2:
  Seminumerical Algorithms*, 3rd ed., Addison-Wesley. §3.4.2.
  → Reservoir sampling / partial Fisher-Yates SWOR procedure used by
    `AlignedRng.choice` (swap-with-last index pool).

* **Lehmer, D.H.** (1951). "Mathematical methods in large-scale computing
  units." *Proc. 2nd Symposium on Large-Scale Digital Computing
  Machinery*, 141–146.
  → Foundational reference for the multiplicative-congruential LCG
    family used throughout `_aligned_rng.py`.

### Implementation choice not from a publication

The two-stage seed-derivation bookkeeping in `derive_per_tree_seeds`
(advance the auxiliary LCG `2*ntree` times, discard; then advance
another `2*ntree` times and record the negation of each post-step
state, with a skip-zero loop) is empirically determined to align our
stream-B output with a specific reference implementation we benchmark
against. It is six lines of integer arithmetic — far below the
threshold of copyrightable expression — and is documented inline at
the function's docstring.

---

## Concordance metrics (`src/crforest/metrics.py`)

* **Kaplan, E.L. & Meier, P.** (1958). "Nonparametric estimation from
  incomplete observations." *Journal of the American Statistical
  Association* 53(282): 457–481.
  → Underlying KM estimator for the censoring distribution used by
    `_km_censor_fit`.

* **Wolbers, M., Koller, M.T., Witteman, J.C.M., Schemper, M.** (2009).
  "Concordance for prognostic models with competing risks."
  *Biostatistics* 10(4): 715–727.
  → Cause-specific concordance pair structure used by both
    `concordance_index_cr` (unweighted) and `concordance_index_uno_cr`
    (IPCW-weighted).

* **Uno, H., Cai, T., Pencina, M.J., D'Agostino, R.B., Wei, L.J.** (2011).
  "On the C-statistics for evaluating overall adequacy of risk
  prediction procedures with censored survival data." *Statistics in
  Medicine* 30(10): 1105–1117.
  → IPCW C-statistic principle used by `concordance_index_uno_cr`;
    weighting form `1/G(t^-)^2`.

* **Cole, S.R. & Hernán, M.A.** (2008). "Constructing inverse probability
  weights for marginal structural models." *American Journal of
  Epidemiology* 168(6): 656–664.
  → ESS-truncation principle used by `_choose_gmin_auto` (the
    `gmin='auto'` lower-clip selection).

### Implementation choices not from a publication

* The "events-first" tie convention in the KM-of-censoring (cause-1
  events removed from the risk pool first; cause ≥ 2 events plus true
  censorings lumped as the "censoring" update) is the de-facto standard
  in the R `survival` package and CR-IPCW reference implementations,
  but is rarely spelled out in published papers. Annotated inline in
  `_km_censor_fit`.

* The symmetric `sqrt(w_i) * sqrt(w_j)` weighting in Branch C of
  `concordance_index_uno_cr` (case-vs-competing pairs) is a natural
  extension of Uno (2011) IPCW to the Wolbers (2009) competing-pair
  structure, but the specific sqrt-sqrt symmetrisation appears to be
  an implementation choice rather than a directly-cited formula.
  Annotated inline.

* The time-tie tolerance `_EPS_T = 1e-9` is an empirically chosen
  IEEE-754 double-comparison threshold, defensible across the range
  `1e-12..1e-6`. Annotated inline.

---

## Permutation variable importance (`src/crforest/_importance.py`)

* **Breiman, L.** (2001). "Random forests." *Machine Learning* 45(1):
  5–32.
  → OOB permutation-importance algorithm used by
    `_compute_importance_oob_impl`.

* **Ishwaran, H., Kogalur, U.B., Blackstone, E.H., Lauer, M.S.** (2008).
  "Random survival forests." *Annals of Applied Statistics* 2(3):
  841–860.
  → Random Survival Forest extension; mortality scoring concept.

* **Ishwaran, H., Gerds, T.A., Kogalur, U.B., Moore, R.D., Gange, S.J.,
  Lau, B.M.** (2014). "Random survival forests for competing risks."
  *Biostatistics* 15(4): 757–773.
  → Competing-risk extension of RSF; integrated CIF (left-Riemann
    integral over the time grid) used by `_predict_tree_mortality` as
    the per-cause mortality score for OOB scoring.

### Implementation discipline not from a publication

* Per-(tree, feature) permutation seed matrix derived once upfront
  (`_derive_perm_seeds`) so the result is bit-equivalent across
  `n_jobs` values. Implementation invariant tested by
  `tests/test_importance_oob.py`.

* `prefer="threads"` for the per-feature joblib outer loop (so the
  forest's per-tree process pool isn't over-subscribed). Standard
  joblib-nesting hygiene; no specific paper.

---

## Survival-tree primitives

The histogram-tree builder, splitting heuristics, time-grid coarsening,
GPU kernels, and persistence layers under `src/crforest/_*.py` are
project-internal designs informed by the RSF literature above and by
standard gradient-boosted-tree histogram engineering (LightGBM,
XGBoost, scikit-learn). They are not ports of any specific
implementation.
