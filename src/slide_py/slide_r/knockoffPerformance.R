# Performance-only helpers for the native-R SLIDE interaction path.
# Gaussian arithmetic and glmnet fitting follow knockoff 0.3.6; keep paired
# verification in tests/test_r_knockoff_performance.R when changing either.

prepareGaussianKnockoffs <- function(X, mu, Sigma, diag_s) {
  if (is.null(dim(diag_s))) diag_s <- diag(diag_s, length(diag_s))
  if (all(diag_s == 0)) {
    return(function(X) {
      warning("The conditional knockoff covariance matrix is not positive definite. Knockoffs will have no power.")
      X
    })
  }
  SigmaInv_s <- solve(Sigma, diag_s)
  mu_k <- X - sweep(X, 2, mu, "-") %*% SigmaInv_s
  Sigma_k <- 2 * diag_s - diag_s %*% SigmaInv_s
  chol_k <- chol(Sigma_k)
  n <- nrow(X)
  p <- ncol(X)
  # This closure belongs to one fixed subproblem, not a cache across inputs.
  function(X) mu_k + matrix(rnorm(p * n), n) %*% chol_k
}

firstNonzeroSparse <- function(beta) {
  # CSC entries are ordered by column. Explicit stored zeros (and NA entries)
  # must not count: stock match(TRUE, abs(row) > 0) skips both.
  if (!inherits(beta, "dgCMatrix")) {
    return(apply(beta, 1, function(x) match(TRUE, abs(x) > 0)))
  }
  nonzero <- which(abs(beta@x) > 0)
  rows <- beta@i[nonzero] + 1L
  first <- !duplicated(rows)
  columns <- rep.int(seq_len(ncol(beta)), diff(beta@p))
  indices <- rep(NA_integer_, nrow(beta))
  indices[rows[first]] <- columns[nonzero[first]]
  indices
}

sparseLambdaPath <- function() {
  # Reuse the installed fitting code, including its defaults and family handling.
  # Replace only the two coefficient scans; fail closed if upstream changes them.
  path <- get("lasso_max_lambda_glmnet", asNamespace("knockoff"))
  replacements <- 0L
  sparse_scan <- function(expr) {
    if (identical(expr, quote(apply(fit$beta, 1, first_nonzero)))) {
      replacements <<- replacements + 1L
      return(quote(firstNonzeroSparse(fit$beta)))
    }
    if (identical(expr, quote(apply(beta, 1, first_nonzero)))) {
      replacements <<- replacements + 1L
      return(quote(firstNonzeroSparse(beta)))
    }
    if (is.call(expr)) as.call(lapply(expr, sparse_scan)) else expr
  }
  body(path) <- sparse_scan(body(path))
  if (replacements != 2L) {
    stop("Unsupported knockoff coefficient extractor; use SLIDE_R_OPTIMIZE=0.")
  }
  path_env <- new.env(parent = environment(path))
  path_env$firstNonzeroSparse <- firstNonzeroSparse
  environment(path) <- path_env
  path
}

sparseKnockoffStatistic <- function() {
  # Reuse the installed swap/W-statistic implementation, changing only its
  # glmnet coefficient extractor. Do not mutate the package namespace.
  stat_env <- new.env(parent = asNamespace("knockoff"))
  stat_env$firstNonzeroSparse <- firstNonzeroSparse
  stat_env$lasso_max_lambda_glmnet <- sparseLambdaPath()
  environment(stat_env$lasso_max_lambda_glmnet) <- stat_env
  stat_env$lasso_max_lambda <- get("lasso_max_lambda", asNamespace("knockoff"))
  environment(stat_env$lasso_max_lambda) <- stat_env
  statistic <- knockoff::stat.glmnet_lambdasmax
  environment(statistic) <- stat_env
  statistic
}

secondKOPrepared <- function(z, y, niter = 100, fdr = 0.2, parallel = TRUE) {
  if (is.null(z) || is.null(y)) stop("Z and Y must not be empty.")
  z_mat <- as.matrix(z)
  mu <- colMeans(z_mat)
  Sigma <- cov(z_mat)
  if (!knockoff:::is_posdef(Sigma)) {
    if (!requireNamespace("corpcor", quietly = TRUE)) {
      stop("corpcor is not installed", call. = FALSE)
    }
    Sigma <- tryCatch(
      suppressWarnings(matrix(as.numeric(corpcor::cov.shrink(z_mat, verbose = FALSE)),
                              nrow = ncol(z_mat))),
      error = function(e) stop("Shrinkage estimation of covariance matrix failed: ",
                               e$message, call. = FALSE)
    )
  }
  diag_s <- tryCatch(
    knockoff::create.solve_asdp(Sigma),
    error = function(e) {
      warning("ASDP solver failed, falling back to equi method: ", e$message,
              immediate. = TRUE)
      knockoff::create.solve_equi(Sigma)
    }
  )
  # Match knockoff.filter's data-frame conversion, including automatic row names.
  X <- if (is.data.frame(z)) as.matrix(z, rownames.force = FALSE) else z
  create_knockoffs_cached <- prepareGaussianKnockoffs(X, mu, Sigma, diag_s)
  statistic <- sparseKnockoffStatistic()
  `%dopar%` <- foreach::`%dopar%`
  `%do%` <- foreach::`%do%`
  if (parallel == TRUE) {
    selected_list <- foreach::foreach(i = 1:niter) %dopar% {
      result <- knockoff::knockoff.filter(
        X = z, y = as.matrix(y), knockoffs = create_knockoffs_cached,
        statistic = statistic, offset = 0, fdr = fdr)
      names(result$selected)
    }
  } else {
    selected_list <- foreach::foreach(i = 1:niter) %do% {
      result <- knockoff::knockoff.filter(
        X = z, y = y, knockoffs = create_knockoffs_cached,
        statistic = statistic, offset = 0, fdr = fdr)
      names(result$selected)
    }
  }
  tab_data <- table(unlist(selected_list))
  list(selected_list = selected_list, tab_data = as.matrix(t(tab_data)))
}

withPreparedInteractionKnockoffs <- function(slide_env = asNamespace("SLIDE")) {
  # Limit the override to interactions. Marginals and installed namespaces stay
  # untouched; each spec escalation still resamples through selectShortFreq.
  interaction_env <- new.env(parent = slide_env)
  for (name in c("interactionSLIDE", "selectShortFreq")) {
    fun <- get(name, slide_env)
    environment(fun) <- interaction_env
    assign(name, fun, interaction_env)
  }
  interaction_env$secondKO <- secondKOPrepared
  runner_env <- new.env(parent = slide_env)
  for (name in c("optimizeSLIDE", "runSLIDE", "SLIDE")) {
    fun <- get(name, slide_env)
    environment(fun) <- runner_env
    assign(name, fun, runner_env)
  }
  runner_env$interactionSLIDE <- interaction_env$interactionSLIDE
  runner_env
}
