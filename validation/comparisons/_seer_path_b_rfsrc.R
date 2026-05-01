# Single-seed rfSRC fit on staged SEER breast cancer cohort.
#
# CLI: Rscript _seer_path_b_rfsrc.R <seed> <rf.cores>
#   <seed>     integer (rfSRC RNG seed; passed -seed for reproducibility)
#   <rf.cores> integer; rf.cores option for OMP threading
#
# Reads /tmp/seer_breast_clean.parquet + train/test idx files, fits a
# 100-tree rfsrc, predicts on holdout, emits a single line "RESULT_JSON {...}"
# the Python harness parses.
#
# NOTE: requires the R `arrow` package. If your cluster lacks a C++20
# compiler (HPC clusters often do), patch this script to read.csv after
# converting the parquet to CSV via `pandas.read_parquet(...).to_csv(...)`.

suppressPackageStartupMessages({
  library(arrow)
  library(randomForestSRC)
  library(jsonlite)
})

args <- commandArgs(trailingOnly = TRUE)
if (length(args) != 2L) stop("usage: _seer_path_b_rfsrc.R <seed> <rf.cores>")
seed     <- as.integer(args[[1]])
rf_cores <- as.integer(args[[2]])
options(rf.cores = rf_cores)

df_full   <- as.data.frame(arrow::read_parquet("/tmp/seer_breast_clean.parquet"))
train_idx <- scan("/tmp/seer_breast_train_idx.txt", what = integer(), quiet = TRUE) + 1L
test_idx  <- scan("/tmp/seer_breast_test_idx.txt",  what = integer(), quiet = TRUE) + 1L
train_df  <- df_full[train_idx, ]
test_df   <- df_full[test_idx, ]
rm(df_full); gc(verbose = FALSE)

n_feat   <- ncol(train_df) - 2L
mtry_val <- ceiling(sqrt(n_feat))

t0 <- proc.time()
fit <- rfsrc(
  Surv(time, status) ~ ., data = train_df,
  ntree = 100, mtry = mtry_val, nodesize = 3, nsplit = 10,
  samptype = "swor", importance = "none", seed = -seed
)
fit_wall <- unname((proc.time() - t0)["elapsed"])

pred <- predict(fit, newdata = test_df, importance = "none")
c1 <- 1 - tail(pred$err.rate[, "event.1"], 1)
c2 <- 1 - tail(pred$err.rate[, "event.2"], 1)

risk_path <- sprintf("/tmp/rfsrc_seer_seed%d_cores%d_risk.parquet", seed, rf_cores)
arrow::write_parquet(
  data.frame(test_idx = test_idx - 1L,
             risk1 = pred$predicted[, 1],
             risk2 = pred$predicted[, 2]),
  risk_path
)

cat("RESULT_JSON ", jsonlite::toJSON(list(
  lib       = "rfsrc",
  seed      = seed,
  rf_cores  = rf_cores,
  fit_wall  = fit_wall,
  harrell_c1 = unname(c1),
  harrell_c2 = unname(c2),
  risk_path = risk_path
), auto_unbox = TRUE), "\n", sep = "")
