# SEER breast cancer matched-pair benchmark

Cross-dataset external-validity benchmark for comprisk. Mirrors the CHF
matched-pair (`n75k_path_b.py`) but on an oncology cohort with cancer-specific
vs other-cause mortality as the competing risks.

## Reproducibility status

This benchmark requires you to have **your own** SEER Research Data access
agreement. The SEER Data Use Agreement prohibits redistribution of the data;
this repo contains the methodology (variable selection, cohort filters,
encoding logic, harness) but no data.

Apply for access at https://seerdataaccess.cancer.gov/ — the independent
researcher tier ("SEER Research Data") is free, accepts any email, and is
typically approved in 2 business days.

## Cohort spec

Database: **Incidence - SEER Research Data, 17 Registries, Nov 2025 Sub
(2000-2023)** (or any later submission with the same variable layout).

SEER\*Stat session: **Case Listing**, with these variables in the Table tab:

```
SEER cause-specific death classification        # event 1 source
SEER other cause of death classification        # event 2 source
Vital status recode (study cutoff used)
Survival months                                 # time
Survival months flag                            # quality filter
Age recode with <1 year olds and 90+
Sex
Race recode (W, B, AI, API)
Year of diagnosis
Marital status at diagnosis
Summary stage 2000 (1998-2017)
Derived AJCC Stage Group, 7th ed (2010-2015)
Derived EOD 2018 Stage Group Recode (2018+)     # exported but unused (out of cohort window)
CS tumor size (2004-2015)
Regional nodes positive (1988+)
Regional nodes examined (1988+)
Histology recode - broad groupings
ER Status Recode Breast Cancer (1990+)
PR Status Recode Breast Cancer (1990+)
Derived HER2 Recode (2010+)
RX Summ--Surg Prim Site (1998-2022)
Radiation recode
Chemotherapy recode (yes, no/unk)
```

Selection tab filters:
- `Site recode ICD-O-3/WHO 2008` = `Breast`
- `Sequence number` = `One primary only` (or `0` — first-primary only)
- (Cohort window applied in code — export 2010-2018, the build script
  trims to 2010-2015 for homogeneous staging)

Export options:
- File format: DOS/Windows
- Field delimiter: comma
- Variable format: labels
- Missing character: space
- Variable names included: yes

Save the export as `~/data/seer/export.csv` (or pass `--src` to the build
script).

## Build and stage

```bash
# Vendoring + staging in one step (writes /tmp/seer_breast_*.parquet + idx)
python validation/gen_seer_breast.py

# Memory-constrained box: subsample to 75k for rfSRC tractability
python validation/gen_seer_breast.py --subsample 75000
```

Cohort filter chain:
1. Year of diagnosis ∈ [2010, 2015] — homogeneous staging via AJCC 7th
   plus CS tumor size (both valid all years in window).
2. Drop `Dead (missing/unknown COD)` — analytically intractable for
   competing risks.
3. Drop non-`Complete dates` survival months — quality filter.
4. Median-impute the three numeric features with missingness
   (`nodes_pos` ~16%, `nodes_exam` ~4%, `cs_tumor_size` ~6%).

Final cohort: ~238k cases × 17 features. Status distribution:
~71% censored / 16% cancer-specific death / 13% other-cause death.

## Run the benchmark

```bash
PYTHONUNBUFFERED=1 python -u validation/comparisons/seer_path_b.py \
    --seeds 42,43,44 --cells rfsrc_on,comprisk
```

Output: `/tmp/seer_path_b.parquet` plus a fingerprint of the run config.

## Reference results

Real EHR-shaped data, breast cancer cohort. Run on HPC fc7
(Xeon Gold 6148 @ 2.4 GHz, 32 cores, 187 GB RAM):

| Lib       | Wall (s)        | RSS (GB)        | C₁     | C₂     |
|-----------|-----------------|-----------------|--------|--------|
| comprisk  | 7.02 ± 0.31     | 8.83 ± 0.11     | 0.8652 | 0.8370 |
| rfSRC     | 81.56 ± 3.40    | 55.17 ± 0.17    | 0.8450 | 0.8090 |

Speedup: **11.6×** wall, **6.25×** memory; comprisk also wins on accuracy
(+0.020 C₁, +0.028 C₂).

Cross-dataset comparison with the CHF matched-pair: CHF (n=75k, p=58)
hits **19.8×**, SEER (n=238k, p=17) hits **11.6×**. The gap shrinks at
lower p because rfSRC's per-split full-scan cost scales with feature
count — this is the same mechanism that explains why synthetic-Gaussian
benchmarks at p=30 produce inflated 200× ratios that don't generalize.

## Known environmental constraints

**WSL2 / 23 GB visible RAM:** rfSRC OOMs at full SEER cohort scale
(needs ~55 GB at n=238k). Use `--subsample 75000` for the matched-pair;
comprisk itself runs cleanly at full N (~9 s, 8.7 GB RSS) and reports
the same C-index numbers — that pair (full-N comprisk + 75k matched) is
itself a reportable headline.

**HPC clusters without C++20 compilers:** the R `arrow` package will
fail to build from source. Patch `_seer_path_b_rfsrc.R` to read CSV
instead, and convert parquet → CSV at job start:

```bash
.venv/bin/python -c \
  "import pandas as pd; pd.read_parquet('/tmp/seer_breast_clean.parquet')\
   .to_csv('/tmp/seer_breast_clean.csv', index=False)"
sed -i "s|arrow::read_parquet(\"/tmp/seer_breast_clean.parquet\")|read.csv(\"/tmp/seer_breast_clean.csv\", stringsAsFactors=FALSE)|" \
  validation/comparisons/_seer_path_b_rfsrc.R
```

(Comment out the `arrow::write_parquet(...)` block at the end too — the
risk-path output is optional.)

## Acknowledgment

Publications using this benchmark must include the SEER acknowledgment
required by the Data Use Agreement, e.g.:

> This research used data from the Surveillance, Epidemiology, and End
> Results (SEER) Program of the National Cancer Institute, Research Data,
> 17 Registries, Nov 2025 Sub (2000-2023).
