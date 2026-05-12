# tests/cross_check_crrp.R
#
# Generate crrp::crrp() reference fits used by tests/test_penalized_fine_gray.py
# to gate `comprisk.PenalizedFineGrayRegression`.
#
# `crrp` (Fu, Z. 2017, "Penalized variable selection in competing risks
# regression", Lifetime Data Analysis 23:353-376) implements cyclic coordinate
# descent on the IPCW-weighted Fine-Gray partial likelihood with LASSO / SCAD /
# MCP penalties -- the same algorithm comprisk's PenalizedFineGrayRegression
# implements. It is GPL-licensed and used here only as a developer-side
# numerical oracle (the committed fixtures are plain numbers), exactly as
# tests/cross_check_cmprsk.R uses cmprsk.
#
# `crrp` 1.0 was archived from CRAN; install from source archive (one tiny
# patch needed for modern compilers -- swap the deprecated `Calloc`/`Free`
# macros for `R_Calloc`/`R_Free`):
#
#   url <- "https://cran.r-project.org/src/contrib/Archive/crrp/crrp_1.0.tar.gz"
#   download.file(url, dest <- tempfile(fileext=".tar.gz"))
#   untar(dest, exdir = tmp <- tempfile()); src <- file.path(tmp, "crrp/src/crrp.c")
#   writeLines(c("#define Calloc R_Calloc", "#define Free R_Free",
#                "#define Realloc R_Realloc", readLines(src)), src)
#   install.packages(file.path(tmp, "crrp"), repos = NULL, type = "source")
#
# Reads the datasets already committed by cross_check_cmprsk.R; writes one
# fixture per (dataset, penalty):
#   crrp_<name>_<penalty>_fit.csv  -- columns: feature, lambda, coef, se
#                                     plus a `lambda` row group "__lambda__"
#
# Usage:
#   Rscript tests/cross_check_crrp.R

suppressPackageStartupMessages(library(crrp))

fixtures_dir <- "tests/fixtures"

write_crrp_fit <- function(fit, cov_cols, path) {
    beta <- fit$beta            # p x nlambda
    se   <- fit$SE              # p x nlambda
    lam  <- fit$lambda          # nlambda
    nl <- length(lam)
    rows <- list()
    for (l in seq_len(nl)) {
        rows[[l]] <- data.frame(
            feature = cov_cols,
            lambda  = lam[l],
            coef    = as.numeric(beta[, l]),
            se      = as.numeric(se[, l]),
            stringsAsFactors = FALSE
        )
    }
    out <- do.call(rbind, rows)
    write.csv(out, path, row.names = FALSE)
}

run_dataset <- function(name, cov_cols, event_col) {
    df <- read.csv(file.path(fixtures_dir, paste0("cmprsk_", name, "_data.csv")))
    X <- as.matrix(df[, cov_cols])
    time <- df[["time"]]
    fstatus <- df[[event_col]]
    for (pen in c("LASSO", "SCAD", "MCP")) {
        fit <- crrp(time, fstatus, X, failcode = 1, cencode = 0,
                    penalty = pen, alpha = 1, nlambda = 50,
                    lambda.min = 0.001, eps = 1e-6, max.iter = 5000)
        write_crrp_fit(fit, cov_cols,
                       file.path(fixtures_dir,
                                 paste0("crrp_", name, "_", tolower(pen), "_fit.csv")))
        cat("Wrote crrp", name, pen, "fixture\n")
    }
}

run_dataset("pbc", c("age", "edema", "bili", "albumin", "protime", "stage"), "event")
run_dataset("follic", c("age", "hgb", "clinstg", "ch"), "status")
cat("Done.\n")
