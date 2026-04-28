# κ.exp9b — rfSRC ntree plateau check on real CHF.
#
# Companion to exp9 (crforest 3 seeds × ntree {100,200,500,1000} on real CHF).
# Quick single-seed rfSRC runs at ntree {500, 1000} so we can compare the
# converged C-index against crforest's plateau (HF Harrell ~0.8647).
#
# Single seed=42 to keep wall under 30 min:
#   ntree=500  ≈ 9 min  (linear extrap from canonical 100/111s)
#   ntree=1000 ≈ 19 min
#
# Dumps risk vectors so Python re-scores with the same Harrell+Uno C-index
# functions used for exp9.
#
# Run: ssh win 'cd ~/crforest && \
#        Rscript validation/spikes/kappa/exp9b_rfsrc_ntree_curve.R \
#        2>&1 | tee /tmp/exp9b_rfsrc_ntree_curve.log'

suppressPackageStartupMessages({
  library(arrow)
  library(randomForestSRC)
})

SEED <- 42L
NTREES <- c(500L, 1000L)
RF_CORES <- min(parallel::detectCores(), 16L)
options(rf.cores = RF_CORES)

cat(sprintf("rfSRC %s, seed=%d × ntree {%s}, rf.cores=%d on %s\n",
            as.character(packageVersion("randomForestSRC")),
            SEED, paste(NTREES, collapse = ", "), RF_CORES,
            Sys.info()[["nodename"]]))

df_full   <- as.data.frame(arrow::read_parquet("/tmp/chf_2012_clean.parquet"))
train_idx <- scan("/tmp/chf_2012_train_idx.txt", what = integer(), quiet = TRUE) + 1L
test_idx  <- scan("/tmp/chf_2012_test_idx.txt",  what = integer(), quiet = TRUE) + 1L
train_df  <- df_full[train_idx, ]
test_df   <- df_full[test_idx, ]
rm(df_full); gc(verbose = FALSE)

n_feat   <- ncol(train_df) - 2L
mtry_val <- ceiling(sqrt(n_feat))

run_one <- function(ntree) {
  t0  <- proc.time()
  fit <- rfsrc(
    Surv(time, status) ~ ., data = train_df,
    ntree = ntree, mtry = mtry_val, nodesize = 3, nsplit = 10,
    samptype = "swor", importance = "none", seed = -SEED
  )
  fit_wall <- (proc.time() - t0)["elapsed"]
  pred <- predict(fit, newdata = test_df, importance = "none")
  cat(sprintf("[ntree=%d] wall=%.1fs  rf-err c1=%.4f c2=%.4f\n",
              ntree, fit_wall,
              1 - tail(pred$err.rate[, "event.1"], 1),
              1 - tail(pred$err.rate[, "event.2"], 1)))
  flush.console()
  list(
    ntree    = ntree,
    fit_wall = unname(fit_wall),
    risk1    = pred$predicted[, 1],
    risk2    = pred$predicted[, 2]
  )
}

res_list <- lapply(NTREES, run_one)

risk_long <- do.call(rbind, lapply(res_list, function(r) {
  data.frame(
    ntree    = r$ntree,
    seed     = SEED,
    test_idx = test_idx - 1L,
    risk1    = r$risk1,
    risk2    = r$risk2
  )
}))
arrow::write_parquet(risk_long, "/tmp/chf_2012_rfsrc_ntree_curve.parquet")

walls <- data.frame(
  ntree    = sapply(res_list, function(r) r$ntree),
  seed     = SEED,
  fit_wall = sapply(res_list, function(r) r$fit_wall)
)
arrow::write_parquet(walls, "/tmp/chf_2012_rfsrc_ntree_walls.parquet")

cat(sprintf("\n[dump] /tmp/chf_2012_rfsrc_ntree_curve.parquet (%d rows)\n",
            nrow(risk_long)))
cat(sprintf("[dump] /tmp/chf_2012_rfsrc_ntree_walls.parquet (%d rows)\n",
            nrow(walls)))
