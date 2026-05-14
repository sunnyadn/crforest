# tests/cross_check_fastcmprsk.R
#
# Generate fastcmprsk::fastCrrp() reference fits used by
# tests/test_penalized_fine_gray.py to gate `comprisk.PenalizedFineGrayRegression`
# *independently* of the existing crrp oracle.
#
# `fastcmprsk` (Kawaguchi, E.S., Shen, J.I., Suchard, M.A., Li, G. 2021.
# "Scalable algorithms for large competing risks data." JCGS 30(3):685-693)
# implements the same cyclic coordinate descent on the IPCW-weighted Fine-Gray
# partial likelihood as `crrp`, but uses the linear-time cumulative-sum
# reformulation of the score and information (no Geskus row blowup, no O(n^2)
# risk-set scan) -- the same trick comprisk's `_psh_working` uses. Two
# independently implemented oracles for the same penalised path lets us
# triangulate: crforest should match BOTH within numerical tolerance, and if a
# disagreement shows up it tells us which oracle (or our impl) is off.
#
# What this script found when first authored (pbc, fixed lambda grid from crrp):
#   * LASSO: fastcmprsk == crrp to 3.3e-14 (bit-identical)
#   * MCP  : fastcmprsk == crrp to 4.0e-14 (bit-identical)
#   * SCAD : fastcmprsk != crrp, max |Δβ| ~ 0.5 along the path
# Both packages document Fan & Li (2001) SCAD with default gamma = 3.7, so the
# SCAD prox is implemented differently by the two authors. Whichever is "right"
# is unresolved here (a third reference -- ncvreg / picasso / hand-derived
# prox -- would settle it). crforest's SCAD currently matches crrp; the Python
# side marks the SCAD cell xfail so the gate still flags drift.
#
# `fastcmprsk` was archived from CRAN on 2026-04-10. Install from the CRAN
# source archive (current latest is 1.24.6 -- adjust the URL if a newer archived
# tarball exists):
#
#   url <- paste0("https://cran.r-project.org/src/contrib/Archive/",
#                 "fastcmprsk/fastcmprsk_1.24.6.tar.gz")
#   download.file(url, dest <- tempfile(fileext=".tar.gz"))
#   install.packages(dest, repos = NULL, type = "source")
#
# (If the package source needs a `Calloc`/`Free` -> `R_Calloc`/`R_Free` patch
# under modern R the way `crrp` does, apply the same untar-then-rewrite recipe
# as in tests/cross_check_crrp.R.)
#
# Reads the datasets already committed by cross_check_cmprsk.R; writes one
# fixture per (dataset, penalty):
#   fastcmprsk_<name>_<penalty>_fit.csv
#       columns: feature, lambda, coef, se
#       - `coef` is fastCrrp's coefficient path
#       - `se` is NA: fastCrrp does NOT emit standard errors along the
#         penalised path (its variance machinery is bootstrap-only and only
#         exposed via the unpenalised `fastCrr`). The column is kept so the
#         fixture schema matches `crrp_*` fixtures one-for-one.
#
# Notes on API differences vs `crrp` that matter for parity:
#   * fastCrrp uses a formula interface with `Crisk(ftime, fstatus) ~ .`
#     instead of crrp's `(time, fstatus, X)` triple.
#   * fastCrrp's `alpha` is the L2 admixture in elastic-net (`alpha=0` = pure
#     LASSO/MCP/SCAD; we use the default). crrp's `alpha` was the L1 weight
#     (`alpha=1` = pure L1 in our crrp script). The two scripts intend the
#     same model -- pure non-convex / pure LASSO -- they just spell it
#     differently.
#   * fastCrrp's default `gamma` is `switch(penalty, scad=3.7, 2.7)`, which
#     matches crrp's defaults and Fan & Li (2001) / Zhang (2010).
#   * When fastCrrp is left to generate its own lambda grid on pbc/follic via
#     the Crisk() formula interface with defaults (this script's original
#     attempt), `lambda.path` came out in the 1e-15 to 1e-18 range with
#     `standardize=TRUE` (every fit then sat at the unpenalised limit), and
#     `standardize=FALSE` produced NaN and crashed in seq.default. Root
#     cause not traced (could be an upstream lambda_max bug, an artefact of
#     this script's data prep, or an API misuse on our side). We sidestep
#     by passing the crrp-generated lambda grid in explicitly, which keeps
#     the comparison meaningful (and turns this script into a true
#     triangulation oracle: at matched lambdas, fastcmprsk and crrp agree
#     bit-identically on LASSO/MCP -- max |Δβ| ~3e-14 on pbc when authored,
#     far below crforest's 1e-3 gating tolerance). The Python test reads
#     the resulting fixture and feeds those lambdas to
#     `PenalizedFineGrayRegression(lambdas=...)`, mirroring how the crrp test
#     consumes its fixture.
#
# Usage:
#   Rscript tests/cross_check_fastcmprsk.R

suppressPackageStartupMessages(library(fastcmprsk))

fixtures_dir <- "tests/fixtures"

write_fastcrrp_fit <- function(fit, cov_cols, path) {
    # fastCrrp returns `coef` as a (p x nlambda) matrix and `lambda.path` as
    # a length-nlambda vector. No SE field on the penalised path.
    beta <- fit$coef
    lam  <- fit$lambda.path
    nl <- length(lam)
    rows <- list()
    for (l in seq_len(nl)) {
        rows[[l]] <- data.frame(
            feature = cov_cols,
            lambda  = lam[l],
            coef    = as.numeric(beta[, l]),
            se      = NA_real_,
            stringsAsFactors = FALSE
        )
    }
    out <- do.call(rbind, rows)
    write.csv(out, path, row.names = FALSE)
}

read_crrp_lambda_grid <- function(name, penalty) {
    # Re-use the lambda grid the crrp fixture already pins down -- avoids
    # fastcmprsk's broken `lambda_max` auto-resolution on these datasets.
    fp <- file.path(fixtures_dir,
                    paste0("crrp_", name, "_", tolower(penalty), "_fit.csv"))
    if (!file.exists(fp)) {
        stop("Missing crrp fixture ", fp,
             " -- run cross_check_crrp.R first.")
    }
    ref <- read.csv(fp)
    sort(unique(ref$lambda), decreasing = TRUE)
}

run_dataset <- function(name, cov_cols, event_col) {
    df <- read.csv(file.path(fixtures_dir, paste0("cmprsk_", name, "_data.csv")))
    df$.fstatus <- df[[event_col]]
    rhs <- paste(cov_cols, collapse = " + ")
    form <- as.formula(paste0("Crisk(time, .fstatus) ~ ", rhs))

    for (pen in c("LASSO", "SCAD", "MCP")) {
        lam <- read_crrp_lambda_grid(name, pen)
        fit <- fastCrrp(
            form, data = df,
            penalty = pen,
            lambda = lam,
            eps = 1e-6,
            max.iter = 5000
        )
        write_fastcrrp_fit(fit, cov_cols,
                           file.path(fixtures_dir,
                                     paste0("fastcmprsk_", name, "_",
                                            tolower(pen), "_fit.csv")))
        cat("Wrote fastcmprsk", name, pen, "fixture\n")
    }
}

run_dataset("pbc", c("age", "edema", "bili", "albumin", "protime", "stage"), "event")
run_dataset("follic", c("age", "hgb", "clinstg", "ch"), "status")
cat("Done.\n")
