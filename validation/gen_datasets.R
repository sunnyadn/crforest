# validation/gen_datasets.R
# One-shot: extract PBC, follic, hd from randomForestSRC to CSV.
# Run: Rscript validation/gen_datasets.R
#
# Output: validation/data/{pbc,follic,hd}.csv
#   Columns: x0..x{p-1}, time, event (event: 0 censored, 1..K cause)
#
# Notes:
#   - PBC is taken from survival::pbc (status: 0=censored, 1=transplant, 2=dead).
#     randomForestSRC::pbc only has {0,1} in this version.
#   - Factor columns in follic and hd are integer-encoded via as.integer().

suppressPackageStartupMessages(library(randomForestSRC))
suppressPackageStartupMessages(library(survival))

out_dir <- "validation/data"
dir.create(out_dir, showWarnings = FALSE, recursive = TRUE)

encode_factors <- function(df) {
    for (col in colnames(df)) {
        if (is.factor(df[[col]]) || is.character(df[[col]])) {
            df[[col]] <- as.integer(as.factor(df[[col]]))
        }
    }
    df
}

rename_features <- function(df, time_col, event_col) {
    feat_cols <- setdiff(colnames(df), c(time_col, event_col))
    out <- df[, feat_cols, drop = FALSE]
    out <- encode_factors(out)
    colnames(out) <- paste0("x", seq_len(ncol(out)) - 1)
    out$time <- df[[time_col]]
    out$event <- df[[event_col]]
    out
}

# --- PBC (from survival package: status 0=censored, 1=transplant, 2=dead) ---
data(pbc, package = "survival")
pbc <- pbc[complete.cases(pbc), ]
# Drop the 'id' column before renaming features
pbc$id <- NULL
pbc_out <- rename_features(pbc, time_col = "time", event_col = "status")
write.csv(pbc_out, file.path(out_dir, "pbc.csv"), row.names = FALSE)
cat(sprintf("pbc: %d rows, %d features\n", nrow(pbc_out), ncol(pbc_out) - 2))

# --- follic ---
data(follic, package = "randomForestSRC")
follic <- follic[complete.cases(follic), ]
follic_out <- rename_features(follic, time_col = "time", event_col = "status")
write.csv(follic_out, file.path(out_dir, "follic.csv"), row.names = FALSE)
cat(sprintf("follic: %d rows, %d features\n", nrow(follic_out), ncol(follic_out) - 2))

# --- hd ---
data(hd, package = "randomForestSRC")
hd <- hd[complete.cases(hd), ]
hd_out <- rename_features(hd, time_col = "time", event_col = "status")
write.csv(hd_out, file.path(out_dir, "hd.csv"), row.names = FALSE)
cat(sprintf("hd: %d rows, %d features\n", nrow(hd_out), ncol(hd_out) - 2))
