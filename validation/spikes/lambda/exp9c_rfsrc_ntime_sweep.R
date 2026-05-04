# λ.exp9c — rfSRC ntime sweep (apples-to-apples vs comprisk split_ntime).
#
# comprisk's split_ntime constrains the time-grid for split-search; rfSRC has
# `ntime` but unclear whether it affects split-search or only prediction grid.
# This sweep tells us empirically: if rfSRC's wall drops with smaller ntime,
# its split-search uses the grid; otherwise ntime is prediction-only and
# comprisk's split_ntime IS a real algorithmic innovation.
#
# Sweep:
#   ntime ∈ {NULL (default = all unique event times), 50, 20, 10}
#   × seed=42 only
#   × ntree=100, rf.cores=8 (to avoid the OOM hit at higher cores)
#
# Output: /tmp/lambda_exp9c_rfsrc_ntime.parquet (long format) + per-cell
# wall + native-err c-index dump.
#
# Run: ssh win 'cd ~/comprisk && \
#        Rscript validation/spikes/lambda/exp9c_rfsrc_ntime_sweep.R \
#        2>&1 | tee /tmp/lambda_exp9c.log'

suppressPackageStartupMessages({
  library(arrow)
  library(randomForestSRC)
})

SEED <- 42L
NTIMES <- list(NULL, 50L, 20L, 10L)  # NULL = use all unique event times
RF_CORES <- min(parallel::detectCores(), 8L)
options(rf.cores = RF_CORES)

cat(sprintf("rfSRC %s, seed=%d × ntime sweep, rf.cores=%d on %s\n",
            as.character(packageVersion("randomForestSRC")),
            SEED, RF_CORES, Sys.info()[["nodename"]]))

df_full   <- as.data.frame(arrow::read_parquet("/tmp/chf_2012_clean.parquet"))
train_idx <- scan("/tmp/chf_2012_train_idx.txt", what = integer(), quiet = TRUE) + 1L
test_idx  <- scan("/tmp/chf_2012_test_idx.txt",  what = integer(), quiet = TRUE) + 1L
train_df  <- df_full[train_idx, ]
test_df   <- df_full[test_idx, ]
rm(df_full); gc(verbose = FALSE)

n_feat   <- ncol(train_df) - 2L
mtry_val <- ceiling(sqrt(n_feat))

run_one <- function(ntime_arg) {
  label <- if (is.null(ntime_arg)) "NULL" else as.character(ntime_arg)
  t0 <- proc.time()
  fit <- if (is.null(ntime_arg)) {
    rfsrc(
      Surv(time, status) ~ ., data = train_df,
      ntree = 100, mtry = mtry_val, nodesize = 3, nsplit = 10,
      samptype = "swor", importance = "none", seed = -SEED
    )
  } else {
    rfsrc(
      Surv(time, status) ~ ., data = train_df,
      ntree = 100, mtry = mtry_val, nodesize = 3, nsplit = 10,
      samptype = "swor", importance = "none", seed = -SEED,
      ntime = ntime_arg
    )
  }
  fit_wall <- (proc.time() - t0)["elapsed"]
  pred <- predict(fit, newdata = test_df, importance = "none")
  cat(sprintf("[ntime=%s] wall=%.1fs  rf-err c1=%.4f c2=%.4f  pred_grid_size=%d\n",
              label, fit_wall,
              1 - tail(pred$err.rate[, "event.1"], 1),
              1 - tail(pred$err.rate[, "event.2"], 1),
              ncol(pred$predicted)))
  flush.console()
  list(
    ntime    = label,
    fit_wall = unname(fit_wall),
    risk1    = pred$predicted[, 1],
    risk2    = pred$predicted[, 2]
  )
}

res_list <- lapply(NTIMES, run_one)

cat("\n=== Summary ===\n")
walls <- sapply(res_list, function(r) r$fit_wall)
ntime_labels <- sapply(res_list, function(r) r$ntime)
for (i in seq_along(walls)) {
  cat(sprintf("  ntime=%s  wall=%.1fs  speedup_vs_NULL=%.2fx\n",
              ntime_labels[i], walls[i], walls[1] / walls[i]))
}

risk_long <- do.call(rbind, lapply(res_list, function(r) {
  data.frame(
    ntime    = r$ntime,
    seed     = SEED,
    test_idx = test_idx - 1L,
    risk1    = r$risk1,
    risk2    = r$risk2
  )
}))
arrow::write_parquet(risk_long, "/tmp/lambda_exp9c_rfsrc_ntime.parquet")

walls_df <- data.frame(
  ntime    = ntime_labels,
  seed     = SEED,
  fit_wall = walls
)
arrow::write_parquet(walls_df, "/tmp/lambda_exp9c_rfsrc_ntime_walls.parquet")

cat(sprintf("\n[dump] /tmp/lambda_exp9c_rfsrc_ntime.parquet (%d rows)\n", nrow(risk_long)))
cat(sprintf("[dump] /tmp/lambda_exp9c_rfsrc_ntime_walls.parquet\n"))
