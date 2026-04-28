# validation/gen_rfsrc_baselines.R
# rfSRC cause-1 risk baselines for the paired-seed validation harness.
#
# Regenerate when any of the following change:
#   randomForestSRC version: 3.6.1
#   R version:               R version 4.5.2 (2025-10-31)
#   Datasets file:           validation/data/*.parquet
#   Splits file:             validation/splits/*.parquet
#   Config:                  ntree=500, nodesize=15, mtry=ceiling(sqrt(p))
#                            (default nsplit=10 random-split as used in practice)
#
# Output files (per dataset):
#   validation/baselines/{dataset}.parquet                      -- splitrule="logrankCR", nsplit=10 (production, 100 seeds)
#   validation/baselines/{dataset}_logrank_cause1.parquet       -- splitrule="logrank", cause=1, nsplit=10 (production, 100 seeds)
#   validation/baselines/{dataset}_ns0.parquet                  -- splitrule="logrankCR", nsplit=0  (diagnostic, 20 seeds)
#   validation/baselines/{dataset}_logrank_cause1_ns0.parquet   -- splitrule="logrank", cause=1, nsplit=0  (diagnostic, 20 seeds)
#
# Usage:
#   Full regeneration:           Rscript validation/gen_rfsrc_baselines.R
#   Only nsplit=0 diagnostics:   P3A5_NS0_ONLY=1 Rscript validation/gen_rfsrc_baselines.R
#
# Runtime: full ~3-5h; nsplit=0 only ~1-2h (20 seeds × 4 datasets × 2 splitrules;
# synthetic nsplit=0 is the slowest single pass).

suppressPackageStartupMessages({
    library(randomForestSRC)
    library(arrow)
    library(parallel)
})

MC_CORES <- as.integer(Sys.getenv("MC_CORES", "8"))

DATASETS <- c("pbc", "follic", "hd", "synthetic")
N_ESTIMATORS <- 500
NODESIZE <- 15
DATA_DIR <- "validation/data"
SPLITS_DIR <- "validation/splits"
OUT_DIR <- "validation/baselines"
dir.create(OUT_DIR, showWarnings = FALSE, recursive = TRUE)

fit_one_seed <- function(df, splits, seed, splitrule = "logrankCR", cause = NULL, nsplit = 10) {
    sub <- splits[splits$seed == seed, ]
    train_idx <- sub$sample_id[sub$fold == "train"] + 1  # R is 1-based
    test_idx <- sub$sample_id[sub$fold == "test"] + 1
    train <- df[train_idx, ]
    test <- df[test_idx, ]
    p <- ncol(train) - 2
    rfsrc_args <- list(
        formula    = Surv(time, event) ~ .,
        data       = train,
        ntree      = N_ESTIMATORS,
        nodesize   = NODESIZE,
        mtry       = ceiling(sqrt(p)),
        splitrule  = splitrule,
        nsplit     = nsplit,
        samptype   = "swr",
        seed       = -as.integer(seed)
    )
    if (splitrule == "logrank" && !is.null(cause)) {
        rfsrc_args$cause <- cause
    }
    fit <- do.call(rfsrc, rfsrc_args)
    pred <- predict(fit, newdata = test)
    last_t <- length(fit$time.interest)
    # cif shape is (n_test, n_time, n_cause); cause-1 at final time:
    risk_cause_1 <- pred$cif[, last_t, 1]
    data.frame(
        seed = rep(as.integer(seed), length(risk_cause_1)),
        sample_id = as.integer(sub$sample_id[sub$fold == "test"]),
        risk_cause_1 = as.numeric(risk_cause_1)
    )
}

run_pass <- function(splitrule, cause = NULL, out_suffix = "", nsplit = 10, max_seeds = NULL) {
    label <- if (is.null(cause)) splitrule else sprintf("%s/cause%d", splitrule, cause)
    label <- sprintf("%s ns%d", label, nsplit)
    for (name in DATASETS) {
        cat(sprintf("[%s %s] loading...\n", name, label))
        df <- as.data.frame(read_parquet(file.path(DATA_DIR, paste0(name, ".parquet"))))
        splits <- as.data.frame(read_parquet(file.path(SPLITS_DIR, paste0(name, ".parquet"))))
        seeds <- sort(unique(splits$seed))
        if (!is.null(max_seeds)) {
            seeds <- seeds[seq_len(min(max_seeds, length(seeds)))]
        }
        t0 <- Sys.time()
        cat(sprintf("[%s %s] dispatching %d seeds across %d workers...\n",
                    name, label, length(seeds), MC_CORES))
        rows <- mclapply(
            seeds,
            function(seed) fit_one_seed(
                df, splits, seed,
                splitrule = splitrule, cause = cause, nsplit = nsplit
            ),
            mc.cores = MC_CORES,
            mc.preschedule = FALSE
        )
        # mclapply returns try-error on worker failure — detect and fail loudly.
        errs <- sapply(rows, function(r) inherits(r, "try-error") || is.null(r))
        if (any(errs)) {
            stop(sprintf("[%s %s] %d worker(s) failed", name, label, sum(errs)))
        }
        dt <- as.numeric(difftime(Sys.time(), t0, units = "secs"))
        cat(sprintf("[%s %s] all %d seeds done (%.1fs elapsed, %.1fs/seed avg)\n",
                    name, label, length(seeds), dt, dt / length(seeds)))
        out <- do.call(rbind, rows)
        out_pq <- file.path(OUT_DIR, paste0(name, out_suffix, ".parquet"))
        write_parquet(out, out_pq)
        cat(sprintf("[%s %s] wrote %s (%d rows)\n", name, label, out_pq, nrow(out)))
    }
}

ns0_only <- identical(Sys.getenv("P3A5_NS0_ONLY", "0"), "1")

if (!ns0_only) {
    # Production baselines (rfSRC default nsplit=10, 100 seeds):
    run_pass(splitrule = "logrankCR", cause = NULL, out_suffix = "",                nsplit = 10)
    run_pass(splitrule = "logrank",   cause = 1,    out_suffix = "_logrank_cause1", nsplit = 10)
}

# Diagnostic baselines (rfSRC exhaustive nsplit=0, 20 seeds):
run_pass(splitrule = "logrankCR", cause = NULL, out_suffix = "_ns0",                nsplit = 0, max_seeds = 20)
run_pass(splitrule = "logrank",   cause = 1,    out_suffix = "_logrank_cause1_ns0", nsplit = 0, max_seeds = 20)
