#!/usr/bin/env Rscript
# Synthetic interaction-design microbenchmark; not a production wall-time claim.
# Usage: Rscript benchmarks/profile_r_knockoffs.R <dated-output-directory>
args <- commandArgs(trailingOnly = TRUE)
if (length(args) != 1L) stop("provide a dated output directory")
out <- args[[1]]
dir.create(out, recursive = TRUE, showWarnings = FALSE)
script <- sub("^--file=", "", grep("^--file=", commandArgs(), value = TRUE)[[1]])
helper <- file.path(dirname(dirname(normalizePath(script))), "src", "slide_py",
                    "slide_r", "knockoffPerformance.R")
perf <- new.env(parent = globalenv())
sys.source(helper, perf)
suppressPackageStartupMessages({
  library(SLIDE)
  library(foreach)
})

# Products with a common marginal produce a fixed interaction subproblem.
set.seed(20260912)
n <- 80L
p <- 100L
niter <- 100L
repeats <- 3L
z <- matrix(rnorm(n * (p + 1L)), n)
X <- z[, -1L] * z[, 1L]
colnames(X) <- paste0("z1.z", 2:(p + 1L))
y <- 2 * X[, 1L] - X[, 2L] + rnorm(n)
mu <- colMeans(X)
Sigma <- matrix(as.numeric(corpcor::cov.shrink(X, verbose = FALSE)), p)
diag_s <- knockoff::create.solve_asdp(Sigma)
prepared <- perf$prepareGaussianKnockoffs(X, mu, Sigma, diag_s)
Xk <- prepared(X)
sparse_stat <- perf$sparseKnockoffStatistic()
fit <- glmnet::glmnet(cbind(X, Xk), y, nlambda = 500)

tasks <- list(
  preparation = list(
    stock = function() perf$prepareGaussianKnockoffs(X, mu, Sigma, diag_s),
    optimized = function() perf$prepareGaussianKnockoffs(X, mu, Sigma, diag_s)),
  generation = list(
    stock = function() knockoff::create.gaussian(X, mu, Sigma, diag_s = diag_s),
    optimized = function() prepared(X)),
  coefficient_extraction = list(
    stock = function() apply(fit$beta, 1, function(x) match(TRUE, abs(x) > 0)),
    optimized = function() perf$firstNonzeroSparse(fit$beta)),
  statistic = list(
    stock = function() knockoff::stat.glmnet_lambdasmax(X, Xk, y),
    optimized = function() sparse_stat(X, Xk, y)),
  secondKO = list(
    stock = function() SLIDE::secondKO(X, y, niter = niter, parallel = FALSE),
    optimized = function() perf$secondKOPrepared(X, y, niter = niter, parallel = FALSE))
)
rows <- list()
for (stage in names(tasks)) {
  count <- if (stage == "secondKO") 1L else niter
  for (repeat_id in seq_len(repeats)) {
    for (mode in c("stock", "optimized")) {
      task <- tasks[[stage]][[mode]]
      invisible(gc())
      set.seed(42)
      elapsed <- system.time(for (i in seq_len(count)) task())[["elapsed"]]
      rows[[length(rows) + 1L]] <- data.frame(
        stage = stage, mode = mode, repeat_id = repeat_id,
        calls = count, seconds = elapsed)
    }
  }
}
timings <- do.call(rbind, rows)
write.csv(timings, file.path(out, "timings.csv"), row.names = FALSE)
writeLines(c(sprintf("Synthetic interaction design: n=%d, p=%d; niter=%d; repeats=%d",
                     n, p, niter, repeats),
             "Seed=20260912 (inputs), 42 (each timed batch).",
             "Preparation reports repeated setup cost, not an optimization comparison.",
             "Generation excludes once-per-subproblem setup in optimized mode.",
             "secondKO includes setup, fitting and voting draw collection.",
             trimws(capture.output(sessionInfo()), which = "right")),
           file.path(out, "environment.txt"))
summary <- aggregate(seconds ~ stage + mode, timings, median)
write.csv(summary, file.path(out, "median_timings.csv"), row.names = FALSE)
print(summary, row.names = FALSE)
