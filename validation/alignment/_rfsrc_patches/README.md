# rfSRC 3.6.2 trace-instrumentation patches

> **License notice.** These patches modify source files from the CRAN
> package `randomForestSRC` (GPL >= 3) and are therefore distributed
> under **GPL-3.0**, *not* the Apache-2.0 license that covers the rest
> of this repository. They are diagnostic-only and are excluded from
> the PyPI sdist/wheel (see `pyproject.toml` `tool.hatch.build.targets`).

Patches against pristine CRAN `randomForestSRC_3.6.2` that add an
opt-in trace-event stream. Used by the equivalence-gate diagnostics:

- `../cascade_diagnostic.py`
- `../rank_flip_diagnostic.py`
- `../grid_mismatch_falsification.py`

## Rebuild

```sh
bash regen.sh
```

Defaults: extracts patched source to `/tmp/rfsrc_instrumented`,
installs library to `/tmp/rfsrc_patched_lib`. Override via
`RFSRC_SRC_DIR` / `RFSRC_LIB_DIR` / `RFSRC_VERSION` env vars.

The diagnostic scripts hard-code `/tmp/rfsrc_patched_lib` as the
library location — if you change `RFSRC_LIB_DIR`, update the scripts
to match.

## Usage

After install, set `RFSRC_TRACE=<path>` before calling `rfsrc()` to
capture trace events to that file. Unset to disable (default).

## What each patch adds

- `random.c.patch` — defines `rfsrc_trace_event()` + `rfsrc_trace_maybe_open()`
  helpers; instruments `randomUChainParallel` to emit `ran1B tree=<T> val=<V>`
  on every RNG draw. File is re-opened whenever `RFSRC_TRACE` env var changes,
  so multiple fits in one R session can each get their own trace.
- `splitSurv.c.patch` — per-feature `feat_stat_CR`, per-bin `bin_stat_CR`,
  per-node `node_start` events in `logRankCR` (and `feat_stat_CR` parity
  in `logRankNCR`).
- `splitUtil.c.patch` — `mtry_pick` after each covariate draw + `nsplit_start`
  / `nsplit_pick` inside the SWOR split-point loop.
- `importancePerm.c.patch` — `vimp_perm tree=T a=feat b=dst_sample c=src_sample`
  emitted in `getPermuteMembership` after the `permute()` call. Used by
  `vimp_perm_replay.py` to replay rfSRC's permutation choices through
  comprisk trees and verify algorithmic equivalence.
- `survivalE.c.patch` — Uno IPCW C-index trace, gated on a separate
  `RFSRC_TRACE_UNO=<path>` env var (the per-row volume would drown
  the main `RFSRC_TRACE` stream). Emits three event kinds:
    - `RFSRC_TRACE_UNO_INPUTS eventType=<k> obsSize=<n>` — once per
      `getCRPerformance` call, per cause.
    - `RFSRC_TRACE_UNO_OBS i=<row> t=<time> s=<status> pred=<predOutcome>
      denom=<denomGate> w=<unoWeight>` — per observation per call.
    - `RFSRC_TRACE_UNO_RESULT eventType=<k> n=<n_kept> denomW=<denomW>
      numerW=<numerW>` — at end of `getCRConcordanceIndexIPCW_Fenwick`;
      concordance is `numerW/denomW` (rfSRC's reported value is the
      err.rate `1 - numerW/denomW`).
  Used by `uno_cindex_check.py` to assert per-cell `|Δw|<1e-10` and
  `|Δc|<1e-6` between rfSRC and `compute_uno_weights` /
  `concordance_index_uno_cr`.

Trace format: plain text, one event per line, space-separated kv pairs.
See each diagnostic script for the parse logic.
