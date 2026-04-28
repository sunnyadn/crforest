# validation/

Reproducibility surface for everything published in the project README and
the paper. Two kinds of content live here:

| Subdirectory | What's in it | Stability |
|---|---|---|
| `comparisons/` | Cross-library benchmarks (crforest vs rfSRC, vs scikit-survival). One canonical script per published comparison. | Stable. README + paper cite these by path. |
| `scaling/` *(planned)* | crforest standalone scaling matrices (n, p, ntree, device axes). | Stable once added. |
| `alignment/` | Equivalence diagnostics vs rfSRC (`equivalence='rfsrc'` correctness). | Stable. |
| `baselines/` | rfSRC reference outputs as parquet. Inputs to alignment tests. | Stable. |
| `data/` | Static datasets used by alignment + small benches (pbc, follic, hd, synthetic). | Stable. |
| `benches/`, `bench_*.py`, `runner.py`, `report.py` | Older internal benches, kept for backward compat. | Frozen; new work goes in `comparisons/` or `scaling/`. |
| `spikes/` | Sprint-coded exploration logs (`eta`, `iota`, `kappa`, `lambda`, `theta`, `zeta`). | **Lab-notebook tier.** Numbers may have been retracted; do not cite without checking the corresponding canonical bench in `comparisons/` or `scaling/`. |

Every script in `comparisons/` calls `_fingerprint.dump_fingerprint(out)` at
start, which writes `<out>.fingerprint.json` next to the parquet with: git
SHA + dirty flag, machine, CPU brand, RAM, OS, Python version, library
versions. README/paper claims should always reference the parquet AND its
fingerprint sidecar.

## Provenance: README perf claims → reproduction script

| README claim | Reproduction | Output | Status |
|---|---|---|---|
| crforest 22.5 s vs rfSRC 111.7 s @ n=75k real CHF | `validation/spikes/kappa/exp4c_win_crforest_dump.py` + `exp2_rfsrc.R` | `/tmp/kappa_exp4*.{parquet,Rds}` | ⚠ in spikes/, promote to `comparisons/rfsrc_wall.{py,R}` for v0.2 |
| 6.13× vs rfSRC best `ntime` | `validation/spikes/lambda/exp9c_rfsrc_ntime_sweep.R` | `/tmp/lambda_exp9c.parquet` | ⚠ in spikes/, promote for v0.2 |
| crforest n=1M = 122 s CPU | `validation/spikes/lambda/exp5_paper_scale_bench.py` | `/tmp/lambda_exp5_walls.parquet` | ⚠ in spikes/, promote to `scaling/n_axis.py` for v0.2 |
| rfSRC 14.7 GB @ n=75k → 80 GB @ n=500k | not scripted yet (peak RSS read out-of-band during κ runs) | — | ✗ ship `comparisons/rfsrc_rss.py` for v0.2 |
| pickle ~3.6 GB @ n=100k, n_jobs=1 | `validation/comparisons/sksurv_oom.py` (parallel column at n=100k row) | `/tmp/sksurv_oom*.parquet` | ✓ scripted (this script) |
| crforest 5.7×–64× faster than sksurv `low_memory=True` (n=5k…25k) | `validation/comparisons/sksurv_oom.py` `--low-memory-sksurv --n-jobs -1` | `/tmp/sksurv_oom_lowmem_njobsminus1_win.parquet` | ✓ scripted |
| sksurv `low_memory=False` 16.8 GB peak RSS @ n=5k, OOM ≥ n=10k | `validation/comparisons/sksurv_oom.py` (defaults; no `--low-memory-sksurv`) | `/tmp/sksurv_oom_win.parquet` | ✓ scripted |

Legend: ✓ canonical script + fingerprinted parquet • ⚠ script exists in
`spikes/` and is correct, but path is sprint-coded • ✗ no script yet.

## Running a comparison

```bash
# scikit-survival vs crforest, RSS+wall scaling
PYTHONUNBUFFERED=1 uv run --with scikit-survival --extra dev \
    python -u validation/comparisons/sksurv_oom.py \
    --machine $(hostname) --ns 5000,10000,25000,50000,100000

# Output:
#   /tmp/sksurv_oom.parquet
#   /tmp/sksurv_oom.parquet.fingerprint.json
```

If the fingerprint sidecar is absent or stale, the parquet should not be
cited — re-run the script.
