#!/usr/bin/env Rscript
# CD4 over-selection attribution (R side of the bundled emit_knockoff_attribution probe).
#
# Given rust's EXACT sigma/mu/diag_s and the byte-identical X-tilde draws rust emitted,
# produce the two native-R vote-rate vectors that decompose the rust-vs-pure_r gap:
#   (c) R-glmnet stat.glmnet_lambdasmax on rust's X-tilde  -> r_on_rust_counts.csv
#   (b) R-glmnet on R's OWN create.gaussian draws (rust's sigma/mu/diag_s) -> r_on_r_counts.csv
# plus R's realized mean |corr(Z_j, X~_j)| over the (b) draws -> r_realized_corr.csv.
#
# Both R statistics use the SAME internal random swap knockoff::stat.glmnet_lambdasmax does
# and offset=0; so the only thing that varies between (c) and rust (a) is the lasso engine,
# and between (c) and (b) the X-tilde source. Decomposition (attrib_report.py):
#   H2 lasso = (a)-(c)   same X-tilde, rust-CD vs R-glmnet (both swap)
#   H1 samp  = (c)-(b)   same R-glmnet, rust X-tilde vs R X-tilde
#
# Uses the real `knockoff` package when loadable; otherwise falls back to a faithful
# glmnet-only reimplementation of stat.glmnet_lambdasmax / knockoff.threshold /
# create.gaussian (so it runs in the sandbox where only glmnet is visible). nlambda=500
# matches knockoff's lasso_max_lambda default.
#
# Usage: Rscript attrib_xinject.R <attr_dir> [ndraws] [fdr]

suppressPackageStartupMessages(library(glmnet))
HAVE_KO <- requireNamespace("knockoff", quietly = TRUE)
NLAMBDA <- 500L

if (HAVE_KO) {
  ko_stat   <- function(X, Xk, y) knockoff::stat.glmnet_lambdasmax(X, Xk, y)
  ko_thresh <- function(W, fdr) knockoff::knockoff.threshold(W, fdr = fdr, offset = 0)
  ko_sample <- function(X, mu, Sigma, diag_s) {
    # rust's SDP returns EXACT 0 at s~=0 coords -> Sigma_k exactly singular ->
    # knockoff::create.gaussian's chol() fails (rust jitters; R does not). Floor to
    # R's own ~1e-9 scale so (b) runs; near-copy knockoffs (corr~=1) are unchanged.
    ds <- pmax(diag_s, 1e-9 * max(diag(Sigma)))
    knockoff::create.gaussian(X, mu, Sigma, diag_s = diag(ds, length(ds)))
  }
  engine <- "knockoff-pkg"
} else {
  lasso_max_lambda <- function(Xa, y, nlambda = NLAMBDA) {
    n <- nrow(Xa)
    lmax <- max(abs(crossprod(Xa, y))) / n
    lmin <- lmax / 2e3
    lambda <- lmax * (lmin / lmax)^((0:(nlambda - 1)) / nlambda)
    fit <- glmnet::glmnet(Xa, y, lambda = lambda, intercept = TRUE,
                          standardize = TRUE, family = "gaussian")
    beta <- as.matrix(fit$beta)
    apply(beta, 1, function(b) {
      idx <- which(abs(b) > 0)[1]
      if (is.na(idx)) 0 else fit$lambda[idx] * n
    })
  }
  ko_stat <- function(X, Xk, y) {
    p <- ncol(X)
    swap <- rbinom(p, 1, 0.5)
    sM <- matrix(swap, nrow = nrow(X), ncol = p, byrow = TRUE)
    Xs  <- X * (1 - sM) + Xk * sM
    Xks <- X * sM + Xk * (1 - sM)
    Z <- lasso_max_lambda(cbind(Xs, Xks), y)
    Zo <- Z[1:p]; Zk <- Z[(p + 1):(2 * p)]
    W <- pmax(Zo, Zk) * sign(Zo - Zk)
    W * (1 - 2 * swap)
  }
  ko_thresh <- function(W, fdr) {
    ts <- sort(c(0, abs(W[W != 0])))
    for (t in ts) {
      if ((sum(W <= -t)) / max(1, sum(W >= t)) <= fdr) return(t)
    }
    Inf
  }
  ko_sample <- function(X, mu, Sigma, diag_s) {
    n <- nrow(X); p <- ncol(X)
    Ds <- diag(diag_s, p)
    SigmaInv_s <- solve(Sigma, Ds)
    mu_k <- X - sweep(X, 2, mu, "-") %*% SigmaInv_s
    V <- 2 * Ds - Ds %*% SigmaInv_s
    V <- (V + t(V)) / 2
    C <- tryCatch(chol(V), error = function(e) chol(V + diag(1e-9 * max(diag(V)), p)))
    mu_k + matrix(rnorm(n * p), n) %*% C
  }
  engine <- "glmnet-fallback"
}

args     <- commandArgs(trailingOnly = TRUE)
attr_dir <- args[1]
ndraws   <- if (length(args) >= 2) as.integer(args[2]) else 100L
fdr      <- if (length(args) >= 3) as.numeric(args[3]) else 0.1

Z     <- as.matrix(read.csv(file.path(attr_dir, "z_numeric.csv"), header = FALSE))
y     <- as.matrix(read.csv(file.path(attr_dir, "y_numeric.csv"), header = FALSE))[, 1]
Sigma <- as.matrix(read.csv(file.path(attr_dir, "sigma.csv"),  header = FALSE))
mu    <- as.matrix(read.csv(file.path(attr_dir, "mu.csv"),     header = FALSE))[, 1]
diags <- as.matrix(read.csv(file.path(attr_dir, "diag_s.csv"), header = FALSE))[, 1]
p <- ncol(Z)
cat(sprintf("engine=%s  Z=%dx%d y=%d Sigma=%dx%d diag_s=%d ndraws=%d fdr=%g nlambda=%d\n",
            engine, nrow(Z), p, length(y), nrow(Sigma), ncol(Sigma), length(diags),
            ndraws, fdr, NLAMBDA))

select_on <- function(Xt) {
  W  <- ko_stat(Z, Xt, y)
  Th <- ko_thresh(W, fdr)
  if (is.null(Th) || !is.finite(Th)) integer(0) else which(W >= Th)
}

# (c) R-glmnet on rust's byte-identical X-tilde draws.
set.seed(101L)
counts_c <- integer(p)
for (k in 0:(ndraws - 1L)) {
  Xt <- as.matrix(read.csv(file.path(attr_dir, "xtilde", sprintf("xt_%d.csv", k)),
                           header = FALSE))
  for (j in select_on(Xt)) counts_c[j] <- counts_c[j] + 1L
}
write.csv(data.frame(lf_index = 0:(p - 1L), count = counts_c),
          file.path(attr_dir, "r_on_rust_counts.csv"), row.names = FALSE)
cat(sprintf("(c) R-on-rust-Xtilde nonzero: %s\n",
            paste(sprintf("Z%d=%d", which(counts_c > 0) - 1L, counts_c[counts_c > 0]),
                  collapse = " ")))

# (b) R-glmnet on R's OWN create.gaussian draws (rust's Sigma/mu/diag_s) + realized corr.
set.seed(42L)
counts_b   <- integer(p)
corr_accum <- numeric(p)
for (k in 1:ndraws) {
  Xk <- ko_sample(Z, mu, Sigma, diags)
  for (j in select_on(Xk)) counts_b[j] <- counts_b[j] + 1L
  for (j in 1:p) corr_accum[j] <- corr_accum[j] + abs(cor(Z[, j], Xk[, j]))
}
write.csv(data.frame(lf_index = 0:(p - 1L), count = counts_b),
          file.path(attr_dir, "r_on_r_counts.csv"), row.names = FALSE)
write.csv(data.frame(lf_index = 0:(p - 1L), mean_abs_corr = corr_accum / ndraws),
          file.path(attr_dir, "r_realized_corr.csv"), row.names = FALSE)
cat(sprintf("(b) R-on-R-draws nonzero: %s\n",
            paste(sprintf("Z%d=%d", which(counts_b > 0) - 1L, counts_b[counts_b > 0]),
                  collapse = " ")))
cat("done\n")
