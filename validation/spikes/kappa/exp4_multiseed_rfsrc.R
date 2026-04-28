# κ.exp4 — rfSRC multi-seed bench, SERIAL.
#
# Runs 5 rfsrc fits one at a time with rf.cores=10 (full bandwidth). Earlier
# parallel mclapply variant was abandoned after it OOM'd the Mac (5 workers
# each holding the 75k training df + a 100-tree forest peaked > available
# RAM). Serial keeps memory bounded to a single fit's footprint at a time.
# Wall-time: ~5 × 220s ≈ 18-20 min total. Dumps per-seed risk vectors so
# Python can re-score with identical metrics.
#
# Run: Rscript validation/spikes/kappa/exp4_multiseed_rfsrc.R

suppressPackageStartupMessages({
  library(arrow)
  library(randomForestSRC)
})

SEEDS <- c(42L, 43L, 44L, 45L, 46L)
# Cap at 16: rfSRC's OpenMP scaling plateaus around 8-16 cores, and per-thread
# split workspace × 16 fits within WSL's 24 GB allocation (host has 32 GB,
# .wslconfig sets memory=24GB). On Mac (10 cores) this auto-caps at 10.
RF_CORES <- min(parallel::detectCores(), 16L)

options(rf.cores = RF_CORES)
cat(sprintf("rfSRC %s, 5 seeds SERIAL × rf.cores=%d on %s (detected=%d, capped=16)\n",
            as.character(packageVersion("randomForestSRC")), RF_CORES,
            Sys.info()[["nodename"]], parallel::detectCores()))

df_full   <- as.data.frame(arrow::read_parquet("/tmp/chf_2012_clean.parquet"))
train_idx <- scan("/tmp/chf_2012_train_idx.txt", what = integer(), quiet = TRUE) + 1L
test_idx  <- scan("/tmp/chf_2012_test_idx.txt",  what = integer(), quiet = TRUE) + 1L
train_df  <- df_full[train_idx, ]
test_df   <- df_full[test_idx, ]
rm(df_full); gc(verbose = FALSE)  # drop the full df once we have the splits

n_feat <- ncol(train_df) - 2L
mtry_val <- ceiling(sqrt(n_feat))

run_one_seed <- function(s) {
  t0 <- proc.time()
  fit <- rfsrc(
    Surv(time, status) ~ ., data = train_df,
    ntree = 100, mtry = mtry_val, nodesize = 3, nsplit = 10,
    samptype = "swor", importance = "none", seed = -s
  )
  fit_wall <- (proc.time() - t0)["elapsed"]
  pred <- predict(fit, newdata = test_df, importance = "none")
  result <- list(
    seed     = s,
    fit_wall = unname(fit_wall),
    risk1    = pred$predicted[, 1],
    risk2    = pred$predicted[, 2],
    err1     = 1 - tail(pred$err.rate[, "event.1"], 1),
    err2     = 1 - tail(pred$err.rate[, "event.2"], 1)
  )
  rm(fit, pred); gc(verbose = FALSE)  # release the forest before next iter
  result
}

t_outer <- proc.time()
res_list <- vector("list", length(SEEDS))
for (i in seq_along(SEEDS)) {
  s <- SEEDS[i]
  cat(sprintf("[seed %d] starting (%d/%d)...\n", s, i, length(SEEDS)))
  flush.console()
  r <- run_one_seed(s)
  res_list[[i]] <- r
  cat(sprintf("[seed %d] done in %.1fs  c1(rf-err)=%.4f c2(rf-err)=%.4f\n",
              s, r$fit_wall, r$err1, r$err2))
  flush.console()
}
total_wall <- (proc.time() - t_outer)["elapsed"]

cat("\n=== rfSRC per-seed wall + native err.rate-based C-index ===\n")
for (r in res_list) {
  cat(sprintf("  seed=%d  wall=%.2fs  c1(rf-err)=%.4f  c2(rf-err)=%.4f\n",
              r$seed, r$fit_wall, r$err1, r$err2))
}
cat(sprintf("[wall] outer total (serial): %.2fs\n", total_wall))

# Long-format risk dump for Python.
risk_long <- do.call(rbind, lapply(res_list, function(r) {
  data.frame(
    seed     = r$seed,
    test_idx = test_idx - 1L,
    risk1    = r$risk1,
    risk2    = r$risk2
  )
}))
arrow::write_parquet(risk_long, "/tmp/chf_2012_rfsrc_risks_multiseed.parquet")

walls <- data.frame(
  seed     = sapply(res_list, function(r) r$seed),
  fit_wall = sapply(res_list, function(r) r$fit_wall),
  err1     = sapply(res_list, function(r) r$err1),
  err2     = sapply(res_list, function(r) r$err2)
)
arrow::write_parquet(walls, "/tmp/chf_2012_rfsrc_walls_multiseed.parquet")

cat(sprintf("\n[dump] /tmp/chf_2012_rfsrc_risks_multiseed.parquet (%d rows)\n", nrow(risk_long)))
cat(sprintf("[dump] /tmp/chf_2012_rfsrc_walls_multiseed.parquet (%d rows)\n", nrow(walls)))
