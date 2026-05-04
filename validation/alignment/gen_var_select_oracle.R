# Generate rfSRC var.select(method = "md") oracle for SUN-42 bit-equivalence test.
#
# Run with: Rscript validation/alignment/gen_var_select_oracle.R
# Requires: randomForestSRC (>= 3.6.2) and the bundled `follic` dataset.
#
# Output: tests/fixtures/rfsrc_var_select_follic.json
#
# The comprisk test reads this JSON and asserts that
# forest.minimal_depth(equivalence='rfsrc')['feature'].tolist() matches
# `ranking` exactly (entire ranking, not just selected subset).
#
# NOTE: randomForestSRC >= 3.x renamed var.select(method="md") to max.subtree().
#       max.subtree(max.order = 1) reproduces the same mean-minimal-depth
#       values and Ishwaran threshold that the old var.select API returned.
#       - ms$order[,1]  : mean minimal depth per feature (used for ranking)
#       - ms$threshold  : Ishwaran analytical threshold (feature selected iff
#                         mean_min_depth <= threshold)
#       - ms$topvars    : selected feature names (subset with mean_md <= threshold)

suppressMessages(library(randomForestSRC))
suppressMessages(library(jsonlite))
data(follic, package = "randomForestSRC")

set.seed(42)
fit <- rfsrc(
  Surv(time, status) ~ .,
  data = follic,
  ntree = 100,
  nodesize = 15,
  splitrule = "logrankCR",
  bootstrap = "by.user",
  samp = matrix(1L, nrow = nrow(follic), ncol = 100),  # full bootstrap (deterministic)
  seed = -42L
)
# max.subtree(max.order = 1) is the rfSRC 3.x equivalent of var.select(method = "md")
ms <- max.subtree(fit, max.order = 1)

# Full ordering by mean min-depth ascending
# ms$order[,1] = mean minimal depth per feature (the value used for selection)
minDepthVar <- ms$order[, 1]
ord <- order(minDepthVar)
ranking <- names(minDepthVar)[ord]
mean_md <- minDepthVar[ord]
threshold <- ms$threshold
selected_set <- ms$topvars

out <- list(
  dataset = "follic",
  ntree = 100L,
  seed = 42L,
  splitrule = "logrankCR",
  ranking = ranking,
  mean_min_depth = as.numeric(mean_md),
  threshold = threshold,
  selected = selected_set,
  rfsrc_version = as.character(packageVersion("randomForestSRC")),
  note = "var.select(method='md') renamed to max.subtree(max.order=1) in rfSRC >= 3.x"
)
writeLines(toJSON(out, auto_unbox = TRUE, pretty = TRUE),
           "tests/fixtures/rfsrc_var_select_follic.json")
cat("Wrote tests/fixtures/rfsrc_var_select_follic.json\n")
