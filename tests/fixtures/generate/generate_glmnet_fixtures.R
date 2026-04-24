#!/usr/bin/env Rscript
# Generate deterministic reference fixtures from R glmnet for slide-glmnet parity tests.

suppressPackageStartupMessages(library(glmnet))

args <- commandArgs(trailingOnly = FALSE)
file_arg <- grep("^--file=", args, value = TRUE)
if (length(file_arg) > 0) {
  script_dir <- normalizePath(dirname(sub("^--file=", "", file_arg)))
} else {
  script_dir <- getwd()
}
fixture_root <- normalizePath(file.path(script_dir, "..", "glmnet"), mustWork = FALSE)
dir.create(fixture_root, recursive = TRUE, showWarnings = FALSE)

write_csv_matrix <- function(mat, path) {
  write.table(
    as.matrix(mat),
    file = path,
    sep = ",",
    row.names = FALSE,
    col.names = FALSE
  )
}

build_foldid <- function(y, family, cv_folds) {
  n <- length(y)
  foldid <- integer(n)
  if (family == "gaussian") {
    foldid <- ((seq_len(n) - 1L) %% cv_folds) + 1L
    return(foldid)
  }

  zero_idx <- which(y == 0)
  one_idx <- which(y == 1)
  foldid[zero_idx] <- ((seq_along(zero_idx) - 1L) %% cv_folds) + 1L
  foldid[one_idx] <- ((seq_along(one_idx) - 1L) %% cv_folds) + 1L
  foldid
}

validation_losses <- function(x, y, family, lambda, cv_folds) {
  foldid <- build_foldid(y, family, cv_folds)
  loss_mat <- matrix(0, nrow = cv_folds, ncol = length(lambda))

  for (fold in seq_len(cv_folds)) {
    train <- foldid != fold
    valid <- foldid == fold
    fit <- glmnet(
      x[train, , drop = FALSE],
      y[train],
      family = family,
      alpha = 1,
      standardize = TRUE,
      intercept = TRUE,
      lambda = lambda
    )

    if (family == "gaussian") {
      pred <- predict(fit, newx = x[valid, , drop = FALSE], s = lambda, type = "link")
      residual <- pred - matrix(y[valid], nrow = sum(valid), ncol = length(lambda))
      loss_mat[fold, ] <- colMeans(residual^2)
    } else {
      prob <- predict(fit, newx = x[valid, , drop = FALSE], s = lambda, type = "response")
      prob <- pmin(pmax(prob, 1e-5), 1 - 1e-5)
      target <- matrix(y[valid], nrow = sum(valid), ncol = length(lambda))
      loss_mat[fold, ] <- colMeans(-(target * log(prob) + (1 - target) * log(1 - prob)))
    }
  }

  colMeans(loss_mat)
}

write_path_fixture <- function(name, x, y, family) {
  out_dir <- file.path(fixture_root, name)
  dir.create(out_dir, recursive = TRUE, showWarnings = FALSE)

  old_control <- glmnet.control()
  on.exit(do.call(glmnet.control, old_control), add = TRUE)
  glmnet.control(fdev = 0, devmax = 1, mnlam = 250)

  fit <- glmnet(
    x,
    y,
    family = family,
    alpha = 1,
    standardize = TRUE,
    intercept = TRUE,
    nlambda = 250
  )
  coef_mat <- as.matrix(coef(fit, s = fit$lambda))

  write_csv_matrix(x, file.path(out_dir, "input_x.csv"))
  write_csv_matrix(matrix(y, ncol = 1), file.path(out_dir, "input_y.csv"))
  write_csv_matrix(matrix(fit$lambda, ncol = 1), file.path(out_dir, "expected_lambda.csv"))
  write_csv_matrix(coef_mat[-1, , drop = FALSE], file.path(out_dir, "expected_beta.csv"))
  write_csv_matrix(matrix(coef_mat[1, ], ncol = 1), file.path(out_dir, "expected_intercept.csv"))
}

write_knockoff_fixture <- function(name, x, x_tilde, y, family, cv_folds) {
  out_dir <- file.path(fixture_root, name)
  dir.create(out_dir, recursive = TRUE, showWarnings = FALSE)

  old_control <- glmnet.control()
  on.exit(do.call(glmnet.control, old_control), add = TRUE)
  glmnet.control(fdev = 0, devmax = 1, mnlam = 250)

  x_aug <- cbind(x, x_tilde)
  fit <- glmnet(
    x_aug,
    y,
    family = family,
    alpha = 1,
    standardize = TRUE,
    intercept = TRUE,
    nlambda = 250
  )

  beta <- as.matrix(fit$beta)
  p <- ncol(x)
  lambda <- fit$lambda
  entry_lambda <- apply(abs(beta) > 1e-10, 1, function(active) {
    idx <- which(active)[1]
    if (is.na(idx)) {
      0
    } else {
      lambda[idx]
    }
  })

  z <- entry_lambda[seq_len(p)]
  z_tilde <- entry_lambda[p + seq_len(p)]
  w_lambdasmax <- pmax(z, z_tilde) * sign(z - z_tilde)
  w_lambdadiff <- rowSums(abs(beta[seq_len(p), , drop = FALSE])) -
    rowSums(abs(beta[p + seq_len(p), , drop = FALSE]))

  mean_deviance <- validation_losses(x_aug, y, family, lambda, cv_folds)
  lambda_min <- lambda[which.min(mean_deviance)]
  coef_at_lambda <- as.matrix(coef(fit, s = lambda_min))[-1, 1]
  w_coefdiff <- abs(coef_at_lambda[seq_len(p)]) - abs(coef_at_lambda[p + seq_len(p)])

  write_csv_matrix(x, file.path(out_dir, "input_x.csv"))
  write_csv_matrix(x_tilde, file.path(out_dir, "input_x_tilde.csv"))
  write_csv_matrix(matrix(y, ncol = 1), file.path(out_dir, "input_y.csv"))
  write_csv_matrix(matrix(lambda, ncol = 1), file.path(out_dir, "expected_lambda.csv"))
  write_csv_matrix(matrix(w_lambdasmax, ncol = 1), file.path(out_dir, "expected_lambdasmax.csv"))
  write_csv_matrix(matrix(w_lambdadiff, ncol = 1), file.path(out_dir, "expected_lambdadiff.csv"))
  write_csv_matrix(matrix(mean_deviance, ncol = 1), file.path(out_dir, "expected_mean_deviance.csv"))
  write_csv_matrix(matrix(lambda_min, ncol = 1), file.path(out_dir, "expected_lambda_min.csv"))
  write_csv_matrix(matrix(w_coefdiff, ncol = 1), file.path(out_dir, "expected_coefdiff.csv"))
}

gaussian_x <- matrix(c(
  0.2,  1.1, -0.4,
  1.3,  0.7,  0.5,
 -0.8,  0.4,  1.2,
  0.5, -1.0,  0.3,
  1.7,  0.2, -0.6,
 -1.1, -0.3,  0.8,
  0.9,  1.5, -1.2,
 -0.4,  0.6,  0.9
), ncol = 3, byrow = TRUE)
gaussian_y <- c(0.7, 1.8, -0.9, 0.2, 2.1, -1.2, 1.0, -0.3)

binomial_x <- matrix(c(
 -1.2,  0.3,  0.5,
 -0.7, -0.5,  1.1,
 -0.1,  0.8, -0.9,
  0.4, -1.0,  0.2,
  0.9,  0.5, -0.4,
  1.4, -0.2,  0.7,
 -1.0,  1.2, -0.3,
 -0.3, -0.7,  0.6,
  0.2,  0.4,  1.0,
  0.8, -0.4, -1.1,
  1.1,  0.9,  0.1,
 -0.5,  0.1, -0.8
), ncol = 3, byrow = TRUE)
binomial_y <- c(0, 0, 1, 0, 1, 1, 1, 0, 1, 0, 1, 0)

set.seed(123)
gaussian_knockoff_x <- matrix(rnorm(18 * 6), nrow = 18, ncol = 6)
gaussian_x_tilde <- 0.75 * gaussian_knockoff_x + matrix(rnorm(18 * 6, sd = 0.35), nrow = 18)
gaussian_knockoff_y <- drop(gaussian_knockoff_x %*% c(1.2, -0.8, 0.0, 0.6, 0.0, 0.4)) +
  rnorm(18, sd = 0.25)

set.seed(321)
binomial_knockoff_x <- matrix(rnorm(24 * 6), nrow = 24, ncol = 6)
binomial_x_tilde <- 0.7 * binomial_knockoff_x + matrix(rnorm(24 * 6, sd = 0.4), nrow = 24)
binomial_eta <- drop(binomial_knockoff_x %*% c(1.1, -0.9, 0.0, 0.7, 0.0, -0.5))
binomial_prob <- 1 / (1 + exp(-binomial_eta))
binomial_knockoff_y <- rbinom(24, size = 1, prob = binomial_prob)

write_path_fixture("gaussian_path", gaussian_x, gaussian_y, "gaussian")
write_path_fixture("binomial_path", binomial_x, binomial_y, "binomial")
write_knockoff_fixture(
  "gaussian_knockoff",
  gaussian_knockoff_x,
  gaussian_x_tilde,
  gaussian_knockoff_y,
  "gaussian",
  cv_folds = 4
)
write_knockoff_fixture(
  "binomial_knockoff",
  binomial_knockoff_x,
  binomial_x_tilde,
  binomial_knockoff_y,
  "binomial",
  cv_folds = 4
)

cat("Glmnet fixtures written to:", fixture_root, "\n")
