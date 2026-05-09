# tests/cross_check_cmprsk.R
#
# Generate cmprsk::crr() reference fits used by tests/test_fine_gray.py to
# gate `comprisk.FineGrayRegression`.
#
# Outputs (committed under tests/fixtures/):
#   cmprsk_synth_data.csv          --  synthetic seed=2 dataset (n=200)
#   cmprsk_synth_fit.csv           --  cmprsk::crr(ss, cc, cv) coef/se/LL
#   cmprsk_pbc_data.csv            --  preprocessed survival::pbc (cause=2 = death)
#   cmprsk_pbc_fit.csv             --  cmprsk::crr() reference
#   cmprsk_follic_data.csv         --  randomForestSRC::follic (cause=1 = relapse)
#   cmprsk_follic_fit.csv          --  cmprsk::crr() reference
#
# Usage:
#   Rscript tests/cross_check_cmprsk.R

suppressPackageStartupMessages({
    library(cmprsk)
    library(survival)
})

if (!requireNamespace("randomForestSRC", quietly = TRUE)) {
    stop("randomForestSRC package required for `follic`")
}

fixtures_dir <- "tests/fixtures"
if (!dir.exists(fixtures_dir)) {
    dir.create(fixtures_dir, recursive = TRUE)
}

write_fit <- function(fit, path) {
    coefs <- fit$coef
    ses   <- sqrt(diag(fit$var))
    out <- data.frame(
        feature = names(coefs),
        coef    = as.numeric(coefs),
        se      = as.numeric(ses),
        stringsAsFactors = FALSE
    )
    attr_row <- data.frame(
        feature = c("__loglik__", "__loglik_null__", "__n__", "__n_iter__"),
        coef    = c(fit$loglik, fit$loglik.null, fit$n, NA_real_),
        se      = c(NA_real_, NA_real_, NA_real_, NA_real_),
        stringsAsFactors = FALSE
    )
    write.csv(rbind(out, attr_row), path, row.names = FALSE)
}

# ---- Dataset 1: synthetic, mirrors cmprsk's built-in tests/test.R ----
set.seed(2026)
n_synth <- 500
# Mild signal so crr's Newton step stays well-conditioned.
cv <- matrix(rnorm(3 * n_synth), ncol = 3)
colnames(cv) <- c("x1", "x2", "x3")
linpred <- 0.4 * cv[, 1] - 0.3 * cv[, 2]
u1 <- runif(n_synth)
u2 <- runif(n_synth)
t1 <- -log(1 - u1) / exp(linpred)
t2 <- -log(1 - u2) / 0.5
ce <- rexp(n_synth, rate = 0.2)
ss <- pmin(t1, t2, ce)
cc <- ifelse(ss == t1, 1L, ifelse(ss == t2, 2L, 0L))

synth_df <- data.frame(time = ss, event = cc, cv)
write.csv(synth_df, file.path(fixtures_dir, "cmprsk_synth_data.csv"),
          row.names = FALSE)

fit_synth <- crr(ss, cc, cv)
write_fit(fit_synth, file.path(fixtures_dir, "cmprsk_synth_fit.csv"))

# ---- Dataset 2: survival::pbc, cause=1 := death (status==2 in raw pbc) ----
data(pbc, package = "survival")
keep <- c("time", "status", "age", "edema", "bili", "albumin", "protime", "stage")
pbc_use <- pbc[, keep]
pbc_use <- pbc_use[complete.cases(pbc_use), ]
# pbc status: 0=censored, 1=transplant, 2=dead. Recode to comprisk:
# 0=censored, 1=death (cause-of-interest), 2=transplant.
pbc_use$event <- ifelse(pbc_use$status == 2L, 1L,
                ifelse(pbc_use$status == 1L, 2L, 0L))
pbc_use$status <- NULL
write.csv(pbc_use, file.path(fixtures_dir, "cmprsk_pbc_data.csv"),
          row.names = FALSE)

cov_cols <- c("age", "edema", "bili", "albumin", "protime", "stage")
fit_pbc <- crr(
    pbc_use$time, pbc_use$event,
    as.matrix(pbc_use[, cov_cols])
)
write_fit(fit_pbc, file.path(fixtures_dir, "cmprsk_pbc_fit.csv"))

# ---- Dataset 3: randomForestSRC::follic ----
data(follic, package = "randomForestSRC")
# follic columns: age, hgb, clinstg, ch, rt, time, status
follic_use <- follic
follic_use$ch <- ifelse(follic_use$ch == "Y", 1L, 0L)
follic_use$rt <- ifelse(follic_use$rt == "Y", 1L, 0L)
follic_use <- follic_use[complete.cases(follic_use), ]
write.csv(follic_use, file.path(fixtures_dir, "cmprsk_follic_data.csv"),
          row.names = FALSE)

# `rt` is constant (all "Y") in follic, drop it.
cov_cols <- c("age", "hgb", "clinstg", "ch")
fit_follic <- crr(
    follic_use$time, follic_use$status,
    as.matrix(follic_use[, cov_cols])
)
write_fit(fit_follic, file.path(fixtures_dir, "cmprsk_follic_fit.csv"))

cat("Wrote fixtures to", fixtures_dir, "\n")

# ---- Dataset 4: cuminc cross-check (reads Python-generated synth) ----
if (file.exists(file.path(fixtures_dir, "cuminc_synth_data.csv"))) {
    cd <- read.csv(file.path(fixtures_dir, "cuminc_synth_data.csv"))
    res <- cuminc(cd$time, cd$event, cd$group)
    eval_t <- c(0.5, 1.0, 2.0)
    tp <- timepoints(res, eval_t)
    out <- data.frame()
    for (key in rownames(tp$est)) {
        # key is "{group} {cause}"
        out <- rbind(out, data.frame(
            curve = key,
            t = eval_t,
            est = as.numeric(tp$est[key, ]),
            var = as.numeric(tp$var[key, ]),
            stringsAsFactors = FALSE
        ))
    }
    write.csv(out, file.path(fixtures_dir, "cuminc_synth_fit.csv"),
              row.names = FALSE)
    cat("Wrote cuminc fixture for synth\n")
}

# ---- Dataset 5: cuminc on follic (cause=1, no groups) ----
follic_data_path <- file.path(fixtures_dir, "cmprsk_follic_data.csv")
if (file.exists(follic_data_path)) {
    fd <- read.csv(follic_data_path)
    res <- cuminc(fd$time, fd$status)  # no group => single curve
    eval_t <- c(2, 5, 10)
    tp <- timepoints(res, eval_t)
    out <- data.frame()
    for (key in rownames(tp$est)) {
        out <- rbind(out, data.frame(
            curve = key,
            t = eval_t,
            est = as.numeric(tp$est[key, ]),
            var = as.numeric(tp$var[key, ]),
            stringsAsFactors = FALSE
        ))
    }
    write.csv(out, file.path(fixtures_dir, "cuminc_follic_fit.csv"),
              row.names = FALSE)
    cat("Wrote cuminc fixture for follic\n")
}

# ---- Dataset 6: cause-specific Cox cross-check ----
csc_fit <- function(time, event, X_mat, cause) {
    is_event <- ifelse(event == cause, 1L, 0L)
    fit <- coxph(Surv(time, is_event) ~ X_mat, ties = "breslow")
    coef_se <- summary(fit)$coefficients
    out <- data.frame(
        feature = rownames(coef_se),
        coef    = as.numeric(coef_se[, "coef"]),
        se      = as.numeric(coef_se[, "se(coef)"]),
        stringsAsFactors = FALSE
    )
    attr_row <- data.frame(
        feature = c("__loglik__", "__loglik_null__"),
        coef    = c(fit$loglik[2], fit$loglik[1]),
        se      = c(NA_real_, NA_real_),
        stringsAsFactors = FALSE
    )
    rbind(out, attr_row)
}

# pbc, cause=1 (death)
write.csv(
    csc_fit(pbc_use$time, pbc_use$event,
            as.matrix(pbc_use[, c("age","edema","bili","albumin","protime","stage")]),
            cause = 1L),
    file.path(fixtures_dir, "csc_pbc_fit.csv"), row.names = FALSE
)

# follic, cause=1
write.csv(
    csc_fit(follic_use$time, follic_use$status,
            as.matrix(follic_use[, c("age","hgb","clinstg","ch")]),
            cause = 1L),
    file.path(fixtures_dir, "csc_follic_fit.csv"), row.names = FALSE
)
cat("Wrote cause-specific Cox fixtures\n")

# ---- Dataset 7: Gray's K-sample test cross-check ----
if (file.exists(file.path(fixtures_dir, "cuminc_synth_data.csv"))) {
    cd <- read.csv(file.path(fixtures_dir, "cuminc_synth_data.csv"))
    res <- cuminc(cd$time, cd$event, cd$group)
    out <- data.frame(
        cause = seq_len(nrow(res$Tests)),
        stat  = as.numeric(res$Tests[, "stat"]),
        pv    = as.numeric(res$Tests[, "pv"]),
        df    = as.integer(res$Tests[, "df"]),
        stringsAsFactors = FALSE
    )
    write.csv(out, file.path(fixtures_dir, "gray_synth_fit.csv"),
              row.names = FALSE)
    cat("Wrote gray_test fixture for synth\n")
}

# follic-with-groups: split on clinstg (1 vs 2)
if (file.exists(follic_data_path)) {
    fd <- read.csv(follic_data_path)
    fd$g <- factor(fd$clinstg)
    res <- cuminc(fd$time, fd$status, fd$g)
    out <- data.frame(
        cause = seq_len(nrow(res$Tests)),
        stat  = as.numeric(res$Tests[, "stat"]),
        pv    = as.numeric(res$Tests[, "pv"]),
        df    = as.integer(res$Tests[, "df"]),
        stringsAsFactors = FALSE
    )
    write.csv(out, file.path(fixtures_dir, "gray_follic_fit.csv"),
              row.names = FALSE)
    cat("Wrote gray_test fixture for follic\n")
}
