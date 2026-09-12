# Run from the repository root with TMPDIR pointing into project .tmpfiles/.
stopifnot(nzchar(Sys.getenv("TMPDIR")), !startsWith(normalizePath(tempdir()), "/tmp/"))
source("src/loveslide/slide_r/knockoffPerformance.R")
library(foreach)
sparse_path <- sparseLambdaPath()

paired <- function(stock, optimized, label, seed = 819L, equal = identical) {
  set.seed(seed)
  expected <- stock()
  expected_rng <- .Random.seed
  set.seed(seed)
  actual <- optimized()
  stopifnot(equal(expected, actual), identical(expected_rng, .Random.seed))
  message("PASS: ", label)
  invisible(actual)
}

set.seed(181)
X <- matrix(rnorm(120 * 8), 120, dimnames = list(paste0("r", 1:120), paste0("x", 1:8)))
X <- sweep(X, 2, seq_len(ncol(X)) / 10, "+")
y <- 8 * X[, 1] - 6 * X[, 2] + rnorm(nrow(X), sd = 0.3)
mu <- colMeans(X)
Sigma <- cov(X)
s <- knockoff::create.solve_equi(Sigma)
for (diag_s in list(s, diag(s, length(s)))) {
  set.seed(180)
  before <- .Random.seed
  prepared <- prepareGaussianKnockoffs(X, mu, Sigma, diag_s)
  stopifnot(identical(before, .Random.seed))
  paired(function() replicate(4, knockoff::create.gaussian(X, mu, Sigma, diag_s = diag_s), simplify = FALSE),
         function() replicate(4, prepared(X), simplify = FALSE), "Gaussian draws and RNG; vector/matrix s")
}
zero <- prepareGaussianKnockoffs(X, mu, Sigma, rep(0, ncol(X)))
paired(function() suppressWarnings(knockoff::create.gaussian(X, mu, Sigma, diag_s = rep(0, ncol(X)))) ,
       function() suppressWarnings(zero(X)), "zero covariance identity and RNG")
set.seed(181)
before <- .Random.seed
stopifnot(identical(suppressWarnings(zero(X)), X), identical(before, .Random.seed))

beta <- methods::new("dgCMatrix", Dim = c(5L, 4L),
                     p = c(0L, 3L, 5L, 7L, 8L),
                     i = c(0L, 1L, 2L, 0L, 3L, 1L, 2L, 3L),
                     x = c(0, NA_real_, 2, -1, 0, 3, 4, -2))
for (b in list(beta, Matrix::Matrix(matrix(0, 5, 4), sparse = TRUE), as.matrix(beta))) {
  expected <- apply(as.matrix(b), 1, function(row) match(TRUE, abs(row) > 0))
  stopifnot(identical(firstNonzeroSparse(b), expected))
}
message("PASS: first-entry extraction; explicit zeros, NA, ties and never-entering rows")
for (standardize in c(TRUE, FALSE)) {
  paired(function() knockoff:::lasso_max_lambda_glmnet(X, y, standardize = standardize),
         function() sparse_path(X, y, standardize = standardize), "Gaussian lambda entry values")
}
for (family in c("binomial", "multinomial")) {
  response <- if (family == "binomial") as.integer(y > median(y)) else factor(rep(1:3, length.out = nrow(X)))
  paired(function() knockoff:::lasso_max_lambda_glmnet(X, response, family = family),
         function() sparse_path(X, response, family = family), paste(family, "lambda entry values"))
}
set.seed(182)
X_k <- knockoff::create.gaussian(X, mu, Sigma, diag_s = s)
statistic <- sparseKnockoffStatistic()
paired(function() knockoff::stat.glmnet_lambdasmax(X, X_k, y),
       function() statistic(X, X_k, y), "W statistics and swap RNG")

originals <- lapply(c("secondKO", "selectShortFreq", "interactionSLIDE", "SLIDE"), function(n) get(n, asNamespace("SLIDE")))
fullrank <- paired(function() SLIDE::secondKO(X, y, niter = 6, parallel = FALSE),
                   function() secondKOPrepared(X, y, niter = 6, parallel = FALSE), "full-rank draw selections, counts and RNG")
stopifnot(length(unlist(fullrank$selected_list)) > 0)
paired(function() SLIDE::secondKO(as.data.frame(X), y, niter = 4, parallel = FALSE),
       function() secondKOPrepared(as.data.frame(X), y, niter = 4, parallel = FALSE), "data-frame draws, counts and RNG")
Z <- X[1:6, ]
stopifnot(!knockoff:::is_posdef(cov(Z)))
paired(function() SLIDE::secondKO(Z, y[1:6], niter = 4, parallel = FALSE),
       function() secondKOPrepared(Z, y[1:6], niter = 4, parallel = FALSE), "shrinkage draws, counts and RNG")

# Exercise the real PSOCK backend with deterministic, static worker assignment.
# secondKO exposes no foreach options, so set the scheduler at the backend seam.
cl <- parallel::makePSOCKcluster(2, outfile = file.path(Sys.getenv("TMPDIR"), "workers.log"))
tryCatch({
  stopifnot(all(unlist(parallel::clusterCall(cl, function() !startsWith(normalizePath(tempdir()), "/tmp/")))))
  doParallel::registerDoParallel(cl)
  foreach::setDoPar(function(obj, expr, envir, data) {
    obj$options$snow <- list(preschedule = TRUE)
    doParallel:::doParallelSNOW(obj, expr, envir, data)
  }, cl, doParallel:::snowinfo)
  stopifnot(foreach::getDoParWorkers() == 2L)
  parallel::clusterSetRNGStream(cl, 185)
  initial_workers <- parallel::clusterCall(cl, function() .Random.seed)
  set.seed(186)
  expected <- SLIDE::secondKO(X, y, niter = 6, parallel = TRUE)
  expected_rng <- .Random.seed
  expected_workers <- parallel::clusterCall(cl, function() .Random.seed)
  stopifnot(all(vapply(seq_along(cl), function(i) !identical(initial_workers[[i]], expected_workers[[i]]), logical(1))))
  parallel::clusterSetRNGStream(cl, 185)
  set.seed(186)
  actual <- secondKOPrepared(X, y, niter = 6, parallel = TRUE)
  stopifnot(identical(expected, actual), identical(expected_rng, .Random.seed),
            identical(expected_workers, parallel::clusterCall(cl, function() .Random.seed)))
  message("PASS: PSOCK draw selections, counts and master/worker RNG")
}, finally = {
  parallel::stopCluster(cl)
  foreach::registerDoSEQ()
})

helper_env <- new.env(parent = globalenv())
sys.source("src/loveslide/slide_r/knockoffPerformance.R", envir = helper_env)
runner <- helper_env$withPreparedInteractionKnockoffs()
interaction_y <- 10 * X[, 1] * X[, 2] + 8 * X[, 1] * X[, 3] + y / 10
interaction_X <- X
colnames(interaction_X) <- paste0("Z", seq_len(ncol(X)))
args <- list(z = interaction_X, y = interaction_y, m = 8, marginals = "Z1", method = 2,
             spec = 0.3, fdr = 0.3, niter = 8, parallel = FALSE)
for (method in c(2, 4)) {
  args$method <- method
  interaction <- paired(function() do.call(SLIDE::interactionSLIDE, args),
                        function() do.call(runner$interactionSLIDE, args),
                        paste("interaction method", method, "integration and RNG"),
                        # lm terms carry fresh call-frame environments; require
                        # exact numerical equality rather than environment identity.
                        equal = if (method == 4) function(a, b) isTRUE(all.equal(a, b, tolerance = 0)) else identical)
  stopifnot(length(interaction$interaction_vars) > 0)
}
stopifnot(identical(originals, lapply(c("secondKO", "selectShortFreq", "interactionSLIDE", "SLIDE"), function(n) get(n, asNamespace("SLIDE")))))
stopifnot(identical(environment(knockoff::stat.glmnet_lambdasmax), asNamespace("knockoff")))
message("PASS: installed namespaces unchanged")
