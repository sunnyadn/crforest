# κ.exp2 — rfSRC counterpart to exp1.
#
# Reads the SAME preprocessed parquet + train/test indices that exp1 dumped
# to /tmp, fits randomForestSRC with hyperparams aligned to crforest's
# defaults (ntree=100, mtry=ceil(sqrt(p)), nodesize=3, nsplit=10, swor),
# computes holdout cause-1/cause-2 Harrell's C-index, prints wall-time.
#
# Run: Rscript validation/spikes/kappa/exp2_rfsrc.R

suppressPackageStartupMessages({
  library(arrow)
  library(randomForestSRC)
})

cat("rfSRC version:", as.character(packageVersion("randomForestSRC")), "\n")
cat("R cores opt:  ", getOption("rf.cores", "(unset)"), "\n")
options(rf.cores = parallel::detectCores())
cat("rf.cores set to:", getOption("rf.cores"), "\n\n")

# --- Load shared artifacts ---
df_full   <- as.data.frame(arrow::read_parquet("/tmp/chf_2012_clean.parquet"))
train_idx <- scan("/tmp/chf_2012_train_idx.txt", what = integer(), quiet = TRUE) + 1L
test_idx  <- scan("/tmp/chf_2012_test_idx.txt",  what = integer(), quiet = TRUE) + 1L

cat(sprintf("[load] full df: %d rows x %d cols\n", nrow(df_full), ncol(df_full)))
cat(sprintf("[split] train n=%d, test n=%d\n", length(train_idx), length(test_idx)))

train_df <- df_full[train_idx, ]
test_df  <- df_full[test_idx, ]

# Sanity: same status mapping (0/1/2 already encoded in Python).
cat(sprintf("[split] test status counts: 0=%d, 1=%d, 2=%d\n",
            sum(test_df$status == 0), sum(test_df$status == 1), sum(test_df$status == 2)))

n_feat <- ncol(train_df) - 2L  # minus time, status
mtry_val <- ceiling(sqrt(n_feat))
cat(sprintf("[config] ntree=100, mtry=%d (ceil(sqrt(%d))), nodesize=3, nsplit=10, swor\n\n",
            mtry_val, n_feat))

# --- Fit ---
t0 <- proc.time()
fit <- rfsrc(
  Surv(time, status) ~ ., data = train_df,
  ntree = 100,
  mtry = mtry_val,
  nodesize = 3,            # crforest min_samples_leaf=3 equivalent
  nsplit = 10,
  samptype = "swor",
  importance = "none",
  seed = -42
)
fit_wall <- (proc.time() - t0)["elapsed"]
cat(sprintf("[fit] wall: %.2fs (%.1fms/tree avg)\n",
            fit_wall, fit_wall / 100 * 1000))

# --- Predict on holdout ---
t0 <- proc.time()
pred <- predict(fit, newdata = test_df, importance = "none")
pred_wall <- (proc.time() - t0)["elapsed"]
cat(sprintf("[predict] wall: %.2fs\n", pred_wall))

# rfSRC's predict() for CR returns:
#   pred$predicted: matrix [n_test, n_causes] = expected CR mortality per cause
#   pred$err.rate:  per-tree-running 1 - C-index (last row = final ensemble)
final_err <- tail(pred$err.rate, 1)
cat(sprintf("[rfsrc native] predict$err.rate (1 - C-index, rfSRC convention):\n"))
print(final_err)

c1_rfsrc <- 1 - final_err[1, "event.1"]
c2_rfsrc <- 1 - final_err[1, "event.2"]

# --- Dump risk scores for Python-side apples-to-apples C-index re-scoring ---
risk_df <- data.frame(
  test_idx = test_idx - 1L,                       # back to 0-indexed for Python
  risk_cause1 = pred$predicted[, 1],
  risk_cause2 = pred$predicted[, 2]
)
arrow::write_parquet(risk_df, "/tmp/chf_2012_rfsrc_risk.parquet")
cat("[dump] /tmp/chf_2012_rfsrc_risk.parquet written for Python re-scoring\n")

# --- Dump full CIF arrays for per-subject curve comparison ---
# pred$cif dim is [n_test, n_times, n_causes]. Flatten to long format
# (subject-major) so Python can reshape back to (n_test, n_times).
n_test <- dim(pred$cif)[1]
nt     <- length(pred$time.interest)
cif1_subj_major <- as.vector(t(pred$cif[, , 1]))   # length n_test * nt
cif2_subj_major <- as.vector(t(pred$cif[, , 2]))
cif_long <- data.frame(
  subj    = rep(0L:(n_test - 1L), each = nt),
  t_idx   = rep(0L:(nt - 1L), times = n_test),
  t_value = rep(pred$time.interest, times = n_test),
  cif1    = cif1_subj_major,
  cif2    = cif2_subj_major
)
arrow::write_parquet(cif_long, "/tmp/chf_2012_rfsrc_cif.parquet")
cat(sprintf("[dump] /tmp/chf_2012_rfsrc_cif.parquet (%d rows, n_test=%d × nt=%d)\n",
            nrow(cif_long), n_test, nt))
cat(sprintf("[dump] rfSRC time.interest: min=%.0f, max=%.0f, n=%d\n\n",
            min(pred$time.interest), max(pred$time.interest), nt))

cat("=== rfSRC summary ===\n")
cat(sprintf("  fit wall:                       %.2fs\n", fit_wall))
cat(sprintf("  predict wall:                   %.2fs\n", pred_wall))
cat(sprintf("  cause-1 (HF) 1-err.rate:        %.4f\n", c1_rfsrc))
cat(sprintf("  cause-2 (death) 1-err.rate:     %.4f\n", c2_rfsrc))
