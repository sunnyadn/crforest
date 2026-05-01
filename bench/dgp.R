# Synthetic 2-cause Weibull competing-risks DGP — R version.
# Matches bench/dgp.py exactly when seed = 20260417.
#
# NOTE: R and Python's standard normal / uniform / exponential RNGs differ,
# so the realized event/covariate values DIFFER between bench/dgp.py and
# bench/dgp.R even at the same seed. Distribution / proportions / problem
# difficulty match; raw rows do not. This is acceptable for benchmarking
# wall time but not for cross-language correctness checks.

make_synthetic_cr <- function(n, p, seed = 20260417L) {
  set.seed(seed)
  X <- matrix(rnorm(n * p), n, p)
  beta_1 <- numeric(p)
  beta_2 <- numeric(p)
  beta_1[1:5]  <- c(0.8, 0.4, -0.3, 0.0, 0.0)
  beta_2[1:5]  <- c(0.0, 0.0, 0.0, -0.5, 0.6)
  if (p >= 10) {
    beta_1[6:10] <- c(0.0, 0.3, -0.5, 0.4, 0.0)
    beta_2[6:10] <- c(0.5, -0.4, 0.3, 0.0, -0.6)
  }
  alpha1 <- 1.2
  alpha2 <- 0.9
  inter1 <- -3.0
  inter2 <- -3.5
  censor_rate <- 0.06

  lam1 <- exp(inter1 + X %*% beta_1)
  lam2 <- exp(inter2 + X %*% beta_2)
  u1 <- runif(n)
  u2 <- runif(n)
  t1 <- (-log(u1) / lam1) ^ (1 / alpha1)
  t2 <- (-log(u2) / lam2) ^ (1 / alpha2)
  cc <- rexp(n, rate = censor_rate)
  times <- pmin(t1, t2, cc)
  event <- ifelse(times == t1, 1L, ifelse(times == t2, 2L, 0L))

  data.frame(time = as.numeric(times), event = event, X)
}
