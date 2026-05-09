# tests/cross_check_calibration.R
#
# Generate the riskRegression::plotCalibration reference numbers used by
# tests/test_evaluation.py to gate `comprisk.evaluation.calibration_cr`.
#
# Output (committed under tests/fixtures/):
#   calib_pbc.csv  --  model, time_days, bin_idx, Pred, Obs, bin_n
#                      (bin_idx is 1..10, sorted by Pred ascending)
#
# Reuses the same pbc held-out fold and stacked CIF predictions produced by
# tests/cross_check_riskRegression.R; that script must be run first so
# tests/fixtures/{pbc_test_data,pbc_predictions}.csv exist.
#
# Usage:
#   Rscript tests/cross_check_calibration.R
#
# Versions used for the committed fixture:
#   riskRegression 2024.x , prodlim 2024.x , survival 3.x , R 4.5.x

suppressPackageStartupMessages({
    library(survival)
    library(riskRegression)
    library(prodlim)
    library(data.table)
})

fix_dir <- "tests/fixtures"
test_path <- file.path(fix_dir, "pbc_test_data.csv")
pred_path <- file.path(fix_dir, "pbc_predictions.csv")
if (!file.exists(test_path) || !file.exists(pred_path)) {
    stop("Run tests/cross_check_riskRegression.R first to generate ",
         "pbc_test_data.csv + pbc_predictions.csv.")
}

test_df <- read.csv(test_path)
preds   <- read.csv(pred_path)

# Calibration target: 1, 3, 5, 10 years in days (partner workflow).
target_years <- c(1, 3, 5, 10)
target_days  <- target_years * 365L
# Round to integer days to match the test harness.

stack_cif <- function(df, prefix, days) {
    cols <- paste0(prefix, "_t", days)
    if (!all(cols %in% names(df))) {
        stop("Missing predicted-CIF columns for prefix=", prefix,
             "; need columns ", paste(cols, collapse = ","))
    }
    as.matrix(df[, cols])
}

# To use riskRegression::plotCalibration we need a Score object that holds
# predictions at the target times. Re-fit Score() on the held-out fold with
# the pre-stacked CIF columns at target_days.
probs_full  <- stack_cif(preds, "RSF_full",  c(365, 730, 1460, 2190, 2920, 3650))
probs_small <- stack_cif(preds, "RSF_small", c(365, 730, 1460, 2190, 2920, 3650))

# pbc_predictions.csv only has CIF columns at the SUN-60 times. For the
# calibration target_days (1y=365, 3y=1095, 5y=1825, 10y=3650) we need
# 1095 and 1825 — not in the committed fixture. Approximate via right-
# continuous step: use the prediction at the latest committed eval time
# <= target_day. For 1095 use t730; for 1825 use t1460; for 365 use t365;
# for 3650 use t3650.
sun60_days <- c(365, 730, 1460, 2190, 2920, 3650)
src_days   <- sapply(target_days, function(t) max(sun60_days[sun60_days <= t]))
col_idx    <- match(src_days, sun60_days)
probs_full_at  <- probs_full[,  col_idx, drop = FALSE]
probs_small_at <- probs_small[, col_idx, drop = FALSE]
colnames(probs_full_at)  <- paste0("t", target_days)
colnames(probs_small_at) <- paste0("t", target_days)

score_obj <- Score(
    object = list(
        "RSF_full"  = probs_full_at,
        "RSF_small" = probs_small_at
    ),
    formula = Hist(time, event) ~ 1,
    data = test_df,
    cause = 1L,
    times = target_days,
    metrics = c("brier"),
    plots = "calibration",
    se.fit = FALSE
)

# plotCalibration is called per time; collect plotFrames into long form.
out_rows <- list()
for (k in seq_along(target_days)) {
    d <- target_days[k]
    cal <- plotCalibration(score_obj, times = d, method = "quantile",
                           q = 10, plot = FALSE)
    # cal$plotFrames is a named list keyed by model.
    for (m in names(cal$plotFrames)) {
        pf <- as.data.frame(cal$plotFrames[[m]])
        # plotFrames typically carries Pred, Obs columns; sort by Pred so
        # bin_idx is reproducible.
        pf <- pf[order(pf$Pred), , drop = FALSE]
        bin_n <- if (!is.null(pf$bin_n)) pf$bin_n else NA_real_
        out_rows[[length(out_rows) + 1]] <- data.frame(
            model     = m,
            time_days = d,
            bin_idx   = seq_len(nrow(pf)),
            Pred      = pf$Pred,
            Obs       = pf$Obs,
            bin_n     = bin_n,
            stringsAsFactors = FALSE
        )
    }
}
calib_df <- do.call(rbind, out_rows)

# Recover a reliable bin_n: count how many test subjects fall in each bin
# under R's cut(quantile(Pred, probs=seq(0, 1, length.out=q+1)), include.lowest=TRUE).
# This mirrors the binning convention plotCalibration uses internally.
bin_counts <- function(p_vec, q) {
    breaks <- quantile(p_vec, probs = seq(0, 1, length.out = q + 1),
                       names = FALSE, type = 7)
    bins <- cut(p_vec, breaks = breaks, include.lowest = TRUE,
                labels = FALSE)
    as.integer(table(factor(bins, levels = seq_len(q))))
}
for (k in seq_along(target_days)) {
    d <- target_days[k]
    full_p  <- probs_full_at[,  paste0("t", d)]
    small_p <- probs_small_at[, paste0("t", d)]
    full_n  <- bin_counts(full_p, 10)
    small_n <- bin_counts(small_p, 10)
    sel_full  <- calib_df$model == "RSF_full"  & calib_df$time_days == d
    sel_small <- calib_df$model == "RSF_small" & calib_df$time_days == d
    if (sum(sel_full)  == length(full_n))  calib_df$bin_n[sel_full]  <- full_n
    if (sum(sel_small) == length(small_n)) calib_df$bin_n[sel_small] <- small_n
}

write.csv(calib_df, file.path(fix_dir, "calib_pbc.csv"), row.names = FALSE)
cat(sprintf(">>> Wrote %s/calib_pbc.csv (rows=%d)\n", fix_dir, nrow(calib_df)))
print(head(calib_df, 12))
