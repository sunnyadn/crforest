# crforest benchmarks

Reproducible head-to-head benchmarks vs `randomForestSRC`. Wall-time + parallel-efficiency rows append to `results/results.csv` so you can stack runs across hardware, parameter regimes, and crforest commits.

## Quick start

From the repo root:

```bash
# crforest at its default leaf size, all cores, ntree=100
python -m bench.run_crforest --ntree 100 --leaf 3 --jobs -1 --label "$(hostname)"

# rfSRC at its default nodesize, all cores, ntree=100
Rscript bench/run_rfsrc.R --ntree 100 --nodesize 15 --rfcores "$(nproc 2>/dev/null || sysctl -n hw.physicalcpu)"
```

Each call appends one row to `bench/results/results.csv` with timestamp, library version, hardware label, parameters, wall seconds, CPU seconds, and parallel ratio.

## Defaults that matter

| Knob | Default | Notes |
|---|---|---|
| `n` | 60,000 | Wanqi-cr-typical scale; the regime where the algorithmic gap is wide |
| `p` | 30 | Same |
| `ntree` | 100 | Quick smoke; bump to 300 / 500 for production-scale numbers |
| `leaf` (crforest) / `nodesize` (rfSRC) | 3 / 15 | Each library's own default; vary to study sensitivity |
| `nsplit` | 10 | Random-split candidates per feature; matches rfSRC v3+ default |
| `n_bins` | 256 | crforest histogram resolution (rfSRC has no analog) |
| `splitrule` | `logrankCR` | Competing-risks log-rank; aligned across libraries |

The DGP is a 2-cause Weibull with 5 informative features per cause (`bench/dgp.py`, `bench/dgp.R`). Both files use the same conceptual generator; due to RNG differences between Python and R, raw rows differ at the same seed but censoring proportions and problem difficulty match.

## Parallelism caveats

- **rfSRC parallelizes across trees, single-thread per tree.** Per-thread CPU performance dominates wall time, so high-clock consumer chips (i7-14700K, M-series P-cores) can outperform server Xeons at small ntree. The `parallel_ratio = (user+sys)/elapsed` column reports effective parallelism — values near 1.0 mean OpenMP is off; values near `n_cores_used` mean it's saturating.
- **crforest parallelizes both across trees (joblib) and within trees (numba).** The bench harness only logs main-process CPU time; for full-machine CPU usage, use `validation/profile_fit.py` with cProfile.
- `OPENBLAS_NUM_THREADS` / `MKL_NUM_THREADS` should be left at default (1) inside the rfSRC OpenMP block to avoid oversubscription; rfSRC uses BLAS only in lightweight setup paths.

## What's NOT a fair "stock vs stock"

- Forcing rfSRC to `nodesize=3` (crforest's default leaf) at large n triggers OOM in the rfSRC ensemble at modest ntree. We saw this at n=60k, ntree=500, 128 GB allocated: rfSRC `OUT_OF_MEMORY`. The aligned-deep-trees regime is **crforest-friendly** by design and should not be the headline number.
- Forcing crforest to skip its histogram and use exact split scans is not exposed via the API; that algorithmic difference is permanent and not a knob.

The honest framings (each defensible for different audiences):

1. **Stock-vs-stock**: each library at its own recommended defaults. Most relatable to typical users.
2. **rfSRC-defaults aligned**: both at `nodesize=leaf=15`. Most charitable to rfSRC; matches its design point.
3. **crforest-defaults aligned**: both at `nodesize=leaf=3`. Stresses rfSRC outside its design point; useful for showing scaling failure modes (OOM + super-linear walls).

We ship results from all three and let users pick.

## Reproducing the v0.2 release numbers

```bash
# All on the same machine, in this order:

# crforest, both leaf settings, both ntree=100/300/500
for leaf in 3 15; do
  for ntree in 100 300 500; do
    python -m bench.run_crforest --ntree "$ntree" --leaf "$leaf" --jobs -1 \
      --label "$(hostname)"
  done
done

# rfSRC, both nodesize settings, both ntree=100/300/500
for nodesize in 3 15; do
  for ntree in 100 300 500; do
    Rscript bench/run_rfsrc.R --ntree "$ntree" --nodesize "$nodesize" \
      --rfcores "$(nproc 2>/dev/null || sysctl -n hw.physicalcpu)"
  done
done
```

For HPC reproductions, wrap each loop iteration in a SLURM sbatch with `#SBATCH -c <cores>` matching the `--jobs` / `--rfcores` value.

## Result schema

`bench/results/results.csv` columns:

| Column | Notes |
|---|---|
| `timestamp` | ISO 8601 with timezone |
| `library` | `crforest` or `rfSRC` |
| `version` | Package version (or commit hash via the `commit` column) |
| `hardware` | Free-form label (hostname by default; `--label` override for bench runs) |
| `n_cores_used` | Effective core count given to the library |
| `n`, `p`, `ntree` | Workload size |
| `leaf_or_nodesize` | crforest `min_samples_leaf` / rfSRC `nodesize` |
| `nsplit` | Random-split candidates per feature |
| `n_bins` | crforest only; blank for rfSRC |
| `splitrule` | Should be `logrankCR` for cross-library comparisons |
| `wall_s` | Elapsed wall-clock seconds |
| `cpu_s` | (R) `user+sys` seconds; (Python) main-process `process_time` only |
| `parallel_ratio` | (R) `cpu_s / wall_s`; ≈ effective parallelism |
| `commit` | crforest git commit hash (short); blank for rfSRC |
| `notes` | Free-form |
