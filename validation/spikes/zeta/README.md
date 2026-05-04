# ζ — Head-to-head fit-wall benchmark vs rfSRC

Spec: `docs/superpowers/specs/2026-04-23-zeta-head-to-head-benchmark-design.md` (gitignored).

## What's here

```
run_comprisk.py         # comprisk gen + bench + self-test
make_rfsrc_timings.py   # transcribe /tmp rfSRC numbers into parquet
compare.py              # load both timings, log-log extrap, exit rule, report
data/                   # Weibull CR parquet per (n, seed), gitignored
timings/                # comprisk_timings.parquet, rfsrc_timings.parquet
                        #   + rfsrc_timings.parquet.provenance.md
reports/                # zeta_report_<timestamp>.md
```

The spec's planned `run_rfsrc.sh` / `run_rfsrc.R` were not materialized —
rfSRC timings came from an earlier OpenMP-anchored bench at `/tmp/rfsrc_openmp_bench.R`.
See `timings/rfsrc_timings.parquet.provenance.md` for divergences (DGP,
splitrule, seeds, holdout, peak_rss). A spec-compliant rfSRC re-run is a
follow-up to tighten the anchoring.

## Reproduce

```sh
cd validation/spikes/zeta
uv run python run_comprisk.py --step gen
uv run python run_comprisk.py --step bench       # ~35 min on Apple M4 10-core
uv run python make_rfsrc_timings.py              # transcribes known numbers
uv run python compare.py --self-test
uv run python compare.py                         # writes reports/zeta_report_*.md
```

See the latest `reports/zeta_report_*.md` for the decision token.
