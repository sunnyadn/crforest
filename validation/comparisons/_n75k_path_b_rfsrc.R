# Single-seed rfSRC fit at n=75k for the n75k_path_b harness.
#
# CLI: Rscript _n75k_path_b_rfsrc.R <seed> <rf.cores>
#   <seed>     integer (-s seeds rfSRC's RNG; we pass the same seed crforest uses)
#   <rf.cores> integer; 1 = OMP-off (R-on-macOS default), $(nproc) = OMP-on
#
# Reads the staged CHF cohort at /tmp/chf_2012_*.parquet and the
# split-idx text files, fits one 100-tree rfsrc, predicts on holdout,
# emits a single line "RESULT_JSON {...}" the Python harness parses.

suppressPackageStartupMessages({
  library(arrow)
  library(randomForestSRC)
  library(jsonlite)
})

args <- commandArgs(trailingOnly = TRUE)
if (length(args) != 2L) stop("usage: _n75k_path_b_rfsrc.R <seed> <rf.cores>")
seed     <- as.integer(args[[1]])
rf_cores <- as.integer(args[[2]])
options(rf.cores = rf_cores)

df_full   <- as.data.frame(arrow::read_parquet("/tmp/chf_2012_clean.parquet"))
train_idx <- scan("/tmp/chf_2012_train_idx.txt", what = integer(), quiet = TRUE) + 1L
test_idx  <- scan("/tmp/chf_2012_test_idx.txt",  what = integer(), quiet = TRUE) + 1L
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

cat("RESULT_JSON ", jsonlite::toJSON(list(
  lib       = "rfsrc",
  seed      = seed,
  rf_cores  = rf_cores,
  fit_wall  = fit_wall,
  harrell_c1 = unname(c1),
  harrell_c2 = unname(c2)
), auto_unbox = TRUE), "\n", sep = "")
