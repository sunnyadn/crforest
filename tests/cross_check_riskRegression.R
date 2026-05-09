# tests/cross_check_riskRegression.R
#
# Generate the riskRegression::Score reference numbers used by
# tests/test_evaluation.py to gate `comprisk.evaluation.score_cr`.
#
# Outputs (committed under tests/fixtures/):
#   pbc_test_data.csv               --  (time, status) for the held-out fold
#   pbc_predictions.csv             --  CIF for cause=1 at eval_times, two models
#   riskregression_pbc_score.csv    --  AUC + Brier per (model, time)
#
# Usage:
#   Rscript tests/cross_check_riskRegression.R
#
# Versions used for the committed fixture:
#   randomForestSRC 3.6.x , riskRegression 2024.x , survival 3.x , R 4.5.x

suppressPackageStartupMessages({
    library(survival)
    library(randomForestSRC)
    library(riskRegression)
    library(prodlim)
})

set.seed(20260508)

data(pbc, package = "survival")
keep_cols <- c("time", "status", "age", "edema", "bili",
               "albumin", "protime", "stage")
df <- pbc[, keep_cols]
df <- df[complete.cases(df), ]

# pbc status: 0 = censored, 1 = transplant, 2 = dead.
# We treat status 2 (dead) as the cause of interest (cause=1 in our cause
# coding) because death is the dominant event and matches partner's
# convention; status 1 (transplant) is the competing event (cause=2).
df$event <- ifelse(df$status == 2L, 1L,
            ifelse(df$status == 1L, 2L, 0L))
df$status <- NULL

n <- nrow(df)
idx <- sample(seq_len(n), size = floor(0.7 * n))
train_df <- df[idx, ]
test_df  <- df[-idx, ]

eval_times <- c(365, 730, 1460, 2190, 2920, 3650)

# ---- Model 1: rfSRC (richer features) ----
fmla_rsf <- as.formula(
    "Surv(time, event) ~ age + edema + bili + albumin + protime + stage"
)
rsf_fit <- rfsrc(fmla_rsf, data = train_df, ntree = 200, seed = 1)

cif_at_times <- function(rfsrc_fit, newdata, times) {
    pr <- predict(rfsrc_fit, newdata = newdata)
    grid <- pr$time.interest
    cif1 <- pr$cif[, , 1]
    out <- matrix(NA_real_, nrow = nrow(newdata), ncol = length(times))
    for (k in seq_along(times)) {
        idx_t <- which.min(abs(grid - times[k]))
        out[, k] <- cif1[, idx_t]
    }
    out
}
probs_rsf <- cif_at_times(rsf_fit, test_df, eval_times)

# ---- Model 2: rfSRC with fewer features ----
fmla_small <- as.formula("Surv(time, event) ~ age + bili + albumin")
rsf_small <- rfsrc(fmla_small, data = train_df, ntree = 200, seed = 2)
probs_small <- cif_at_times(rsf_small, test_df, eval_times)

# ---- riskRegression::Score ----
score_obj <- Score(
    object = list(
        "RSF_full"  = probs_rsf,
        "RSF_small" = probs_small
    ),
    formula = Hist(time, event) ~ 1,
    data = test_df,
    cause = 1L,
    times = eval_times,
    metrics = c("auc", "brier"),
    summary = "ibs",
    se.fit = FALSE,
    plots = NULL
)

auc_df   <- as.data.frame(score_obj$AUC$score)
brier_df <- as.data.frame(score_obj$Brier$score)

auc_out <- subset(auc_df, model != "Null model",
                  select = c("model", "times", "AUC"))
brier_out <- subset(brier_df, model != "Null model",
                    select = c("model", "times", "Brier", "IBS"))
score_long <- merge(auc_out, brier_out, by = c("model", "times"), all = TRUE)

out_dir <- "tests/fixtures"
dir.create(out_dir, showWarnings = FALSE, recursive = TRUE)

write.csv(test_df[, c("time", "event")],
          file.path(out_dir, "pbc_test_data.csv"), row.names = FALSE)

pred_out <- data.frame(
    row_id = seq_len(nrow(test_df)),
    time   = test_df$time,
    event  = test_df$event
)
for (k in seq_along(eval_times)) {
    pred_out[[paste0("RSF_full_t",  eval_times[k])]] <- probs_rsf[,   k]
    pred_out[[paste0("RSF_small_t", eval_times[k])]] <- probs_small[, k]
}
write.csv(pred_out, file.path(out_dir, "pbc_predictions.csv"), row.names = FALSE)

write.csv(score_long, file.path(out_dir, "riskregression_pbc_score.csv"),
          row.names = FALSE)

cat(sprintf(">>> Wrote %s/{pbc_test_data,pbc_predictions,riskregression_pbc_score}.csv\n",
            out_dir))
cat(sprintf(">>> n_test=%d, eval_times=%s\n", nrow(test_df),
            paste(eval_times, collapse = ",")))
print(score_long)
