#!/usr/bin/env Rscript

suppressPackageStartupMessages({
  library(glmnet)
  library(knockoff)
})

args <- commandArgs(trailingOnly = TRUE)
if (length(args) != 12L) {
  stop("usage: paired_xtilde.R <z.raw> <y.raw> <a.raw> <n> <p> <a_n> <a_p> <lf_names> <out> <niter> <seed> <fdr>")
}
z_path <- args[[1]]
y_path <- args[[2]]
a_path <- args[[3]]
n <- as.integer(args[[4]])
p <- as.integer(args[[5]])
a_n <- as.integer(args[[6]])
a_p <- as.integer(args[[7]])
names_path <- args[[8]]
out <- args[[9]]
niter <- as.integer(args[[10]])
seed <- as.integer(args[[11]])
fdr <- as.numeric(args[[12]])

read_raw_f64 <- function(path, count) {
  con <- file(path, "rb")
  on.exit(close(con))
  values <- readBin(con, "double", n = count, size = 8L, endian = "little")
  if (length(values) != count) stop(path, ": expected ", count, " values")
  values
}

write_raw_matrix <- function(path, value) {
  con <- file(path, "wb")
  on.exit(close(con))
  writeBin(as.double(t(value)), con, size = 8L, endian = "little")
}

sha256 <- function(path) {
  strsplit(system2("sha256sum", path, stdout = TRUE), "[[:space:]]+")[[1]][[1]]
}

z <- matrix(read_raw_f64(z_path, n * p), nrow = n, ncol = p, byrow = TRUE)
y <- read_raw_f64(y_path, n)
a <- matrix(read_raw_f64(a_path, a_n * a_p), nrow = a_n, ncol = a_p, byrow = TRUE)
lf_names <- readLines(names_path)
if (ncol(a) != p || length(lf_names) != p) stop("A/Z/LF dimensions or names differ")
dir.create(file.path(out, "xtilde"), recursive = TRUE, showWarnings = FALSE)
dir.create(file.path(out, "diagnostics", "r_raw"), recursive = TRUE, showWarnings = FALSE)

mu <- colMeans(z)
sigma <- cov(z)
covariance_method <- "sample"
if (!knockoff:::is_posdef(sigma)) {
  shrunk <- corpcor::cov.shrink(z, verbose = FALSE)
  sigma <- matrix(as.numeric(shrunk), nrow = ncol(z), ncol = ncol(z))
  covariance_method <- "corpcor shrinkage"
}
sampler_method <- "asdp"
diag_s <- tryCatch(
  knockoff::create.solve_asdp(sigma),
  error = function(e) {
    sampler_method <<- "equi fallback"
    knockoff::create.solve_equi(sigma)
  }
)
write.csv(data.frame(covariance = covariance_method, method = sampler_method,
                     seed = seed, draws = niter), file.path(out, "sampler.csv"), row.names = FALSE)
glmnet::glmnet.control(fdev = 0)

manifest <- vector("list", niter)
set.seed(seed)
for (iteration in seq_len(niter)) {
  xtilde <- knockoff::create.gaussian(z, mu, sigma, diag_s = diag_s)
  swap <- as.raw(rbinom(p, 1L, 0.5))
  stem <- sprintf("iter_%03d", iteration - 1L)
  xtilde_path <- file.path(out, "xtilde", paste0(stem, ".raw"))
  swap_path <- file.path(out, "xtilde", paste0(stem, "_swap.raw"))
  write_raw_matrix(xtilde_path, xtilde)
  writeBin(swap, swap_path)

  swap_logical <- as.logical(as.integer(swap))
  left <- z
  right <- xtilde
  left[, swap_logical] <- xtilde[, swap_logical, drop = FALSE]
  right[, swap_logical] <- z[, swap_logical, drop = FALSE]
  design <- scale(cbind(left, right))
  lambda_max <- max(abs(crossprod(design, y))) / n
  lambda <- lambda_max * ((lambda_max / 2000) / lambda_max)^((0:499) / 500)
  fit <- glmnet(
    design, y, lambda = lambda, intercept = TRUE, standardize = FALSE,
    standardize.response = FALSE, thresh = 1e-7, maxit = 100000
  )
  coefficients <- as.matrix(fit$beta)
  entry_swapped <- apply(coefficients, 1L, function(value) {
    index <- match(TRUE, abs(value) > 0)
    if (is.na(index)) 0 else fit$lambda[[index]] * n
  })
  w <- pmax(entry_swapped[seq_len(p)], entry_swapped[p + seq_len(p)]) *
    sign(entry_swapped[seq_len(p)] - entry_swapped[p + seq_len(p)])
  w <- w * (1 - 2 * as.integer(swap))
  cutoff <- knockoff::knockoff.threshold(w, fdr = fdr, offset = 0)
  selected <- if (is.finite(cutoff)) which(w >= cutoff) - 1L else integer()
  entry_original <- entry_swapped[seq_len(p)]
  entry_knockoff <- entry_swapped[p + seq_len(p)]
  entry_original[swap_logical] <- entry_swapped[p + which(swap_logical)]
  entry_knockoff[swap_logical] <- entry_swapped[which(swap_logical)]

  raw_dir <- file.path(out, "diagnostics", "r_raw")
  write_raw_matrix(file.path(raw_dir, paste0(stem, "_lambda.raw")), matrix(fit$lambda, nrow = 1L))
  write_raw_matrix(file.path(raw_dir, paste0(stem, "_coefficients.raw")), coefficients)
  write_raw_matrix(file.path(raw_dir, paste0(stem, "_entry_original.raw")), matrix(entry_original, nrow = 1L))
  write_raw_matrix(file.path(raw_dir, paste0(stem, "_entry_knockoff.raw")), matrix(entry_knockoff, nrow = 1L))
  write_raw_matrix(file.path(raw_dir, paste0(stem, "_w.raw")), matrix(w, nrow = 1L))
  write_raw_matrix(
    file.path(raw_dir, paste0(stem, "_design_mean.raw")),
    matrix(colMeans(design), nrow = 1L)
  )
  write_raw_matrix(
    file.path(raw_dir, paste0(stem, "_design_sd.raw")),
    matrix(apply(design, 2L, sd), nrow = 1L)
  )
  write.csv(
    data.frame(
      lambda_count = length(fit$lambda),
      solver_error = fit$jerr,
      solver_sweeps = fit$npasses,
      threshold = if (is.finite(cutoff)) cutoff else NA_real_,
      selected = paste(selected, collapse = ";"),
      a_sha256 = sha256(a_path),
      lf_names_sha256 = sha256(names_path)
    ),
    file.path(raw_dir, paste0(stem, "_meta.csv")),
    row.names = FALSE
  )
  manifest[[iteration]] <- data.frame(
    iteration = iteration - 1L,
    xtilde_file = file.path("xtilde", paste0(stem, ".raw")),
    xtilde_sha256 = sha256(xtilde_path),
    swap_file = file.path("xtilde", paste0(stem, "_swap.raw")),
    swap_sha256 = sha256(swap_path),
    rows = n,
    columns = p
  )
}
write.csv(do.call(rbind, manifest), file.path(out, "xtilde_manifest.csv"), row.names = FALSE)
