#!/usr/bin/env Rscript
# Single-config rfSRC bench. Emits one CSV row to bench/results/results.csv.
#
# Usage:
#   Rscript bench/run_rfsrc.R --n 60000 --p 30 --ntree 100 \
#     --nodesize 15 --nsplit 10 --rfcores 10 --label mac-m3pro
#
# All knobs default to v0.2-canonical values; override per run.

suppressMessages({
  library(randomForestSRC)
  library(survival)
})

# Locate repo root (one above bench/) so dgp.R works from any cwd.
script_path <- normalizePath(sys.frames()[[1]]$ofile %||% "")
if (!nzchar(script_path)) {
  args0 <- commandArgs(trailingOnly = FALSE)
  fa <- args0[grep("--file=", args0)]
  script_path <- sub("--file=", "", fa)
}
bench_dir <- dirname(normalizePath(script_path))
source(file.path(bench_dir, "dgp.R"))

# Parse CLI args (simple key=val style for portability)
args <- commandArgs(trailingOnly = TRUE)
get_arg <- function(name, default) {
  i <- match(paste0("--", name), args)
  if (is.na(i) || i == length(args)) return(default)
  args[i + 1]
}

n          <- as.integer(get_arg("n", 60000))
p          <- as.integer(get_arg("p", 30))
ntree      <- as.integer(get_arg("ntree", 100))
nodesize   <- as.integer(get_arg("nodesize", 15))
nsplit     <- as.integer(get_arg("nsplit", 10))
ntime      <- as.integer(get_arg("ntime", 200))
mtry       <- as.integer(get_arg("mtry", floor(sqrt(p))))
rfcores    <- as.integer(get_arg("rfcores", parallel::detectCores()))
seed       <- as.integer(get_arg("seed", 20260417))
label      <- get_arg("label", Sys.info()[["nodename"]])
splitrule  <- get_arg("splitrule", "logrankCR")

cat(sprintf("rfSRC %s | n=%d p=%d ntree=%d nodesize=%d nsplit=%d rfcores=%d label=%s\n",
            as.character(packageVersion("randomForestSRC")),
            n, p, ntree, nodesize, nsplit, rfcores, label))

df <- make_synthetic_cr(n, p, seed = seed)
options(rf.cores = rfcores)

tt <- system.time(
  fit <- rfsrc(Surv(time, event) ~ ., data = df,
               ntree = ntree, splitrule = splitrule,
               nsplit = nsplit, nodesize = nodesize,
               ntime = ntime, save.memory = FALSE,
               mtry = mtry, importance = "none",
               do.trace = FALSE)
)
ratio <- (tt["user.self"] + tt["sys.self"]) / tt["elapsed"]

cat(sprintf("DONE elapsed=%.2fs cpu=%.2fs ratio=%.2f\n",
            tt["elapsed"], tt["user.self"] + tt["sys.self"], ratio))

# Append to results CSV (header on first write)
results_path <- file.path(bench_dir, "results", "results.csv")
header <- !file.exists(results_path)
row <- data.frame(
  timestamp = format(Sys.time(), "%Y-%m-%dT%H:%M:%S%z"),
  library   = "rfSRC",
  version   = as.character(packageVersion("randomForestSRC")),
  hardware  = label,
  n_cores_used = rfcores,
  n = n, p = p, ntree = ntree,
  leaf_or_nodesize = nodesize,
  nsplit = nsplit,
  n_bins = NA,
  splitrule = splitrule,
  wall_s    = round(as.numeric(tt["elapsed"]), 3),
  cpu_s     = round(as.numeric(tt["user.self"] + tt["sys.self"]), 3),
  parallel_ratio = round(ratio, 3),
  commit = NA,
  notes = ""
)
write.table(row, results_path, sep = ",", row.names = FALSE,
            col.names = header, append = !header, quote = TRUE)
cat(sprintf("appended -> %s\n", results_path))
