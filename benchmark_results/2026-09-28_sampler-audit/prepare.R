#!/usr/bin/env Rscript
# Audit the installed R sampler on a saved, row-major design.
args <- commandArgs(trailingOnly=TRUE)
out <- args[[1]]
n <- as.integer(args[[2]])
p <- as.integer(args[[3]])
draws <- as.integer(args[[4]])
read_raw <- function(name) readBin(file.path(out, name), "double", n=n*p, endian="little")
write_raw <- function(name, x) writeBin(as.double(t(x)), file.path(out, name), endian="little")
z <- matrix(read_raw("z.raw"), n, p, byrow=TRUE)
mu <- colMeans(z)
sigma <- cov(z)
method <- "sample"
if (!knockoff:::is_posdef(sigma)) {
    sigma <- matrix(as.numeric(corpcor::cov.shrink(z, verbose=FALSE)), p, p)
    method <- "shrinkage"
}
s <- knockoff::create.solve_asdp(sigma)
options(digits=17)
writeLines(apply(sigma, 1, function(row) paste(sprintf("%.17g", row), collapse=",")), file.path(out,"sigma.csv"))
writeLines(sprintf("%.17g", s), file.path(out,"r_s.csv"))
b <- solve(sigma, diag(s))
mean <- z - sweep(z, 2, mu, "-") %*% b
cov <- 2*diag(s) - diag(s) %*% b
factor <- chol(cov)
write_raw("r_mean.raw", mean)
write_raw("r_cov.raw", cov)
write_raw("r_factor.raw", factor)
con <- file(file.path(out,"r_draws.raw"), "wb")
noise_con <- file(file.path(out,"r_noise.raw"), "wb")
max_error <- 0
for (i in seq_len(draws)) {
    set.seed(42 + i - 1L)
    noise <- matrix(rnorm(n*p), n, p)
    set.seed(42 + i - 1L)
    actual <- knockoff::create.gaussian(z, mu, sigma, diag_s=s)
    max_error <- max(max_error, abs(actual - (mean + noise %*% factor)))
    writeBin(as.double(t(actual)), con, endian="little")
    writeBin(as.double(t(noise)), noise_con, endian="little")
}
close(con)
close(noise_con)
stopifnot(max_error < 1e-12)
writeLines(c(paste("covariance", method), paste("manual_create_gaussian_max_error", max_error),
             paste("R", getRversion()), paste("knockoff", packageVersion("knockoff")),
             paste("Rdsdp", packageVersion("Rdsdp")),
             paste("knockoff_path", find.package("knockoff")),
             paste("RNG", paste(RNGkind(), collapse=","))), file.path(out,"r_metadata.txt"))
writeLines(c(deparse(knockoff::create.gaussian), deparse(knockoff::create.solve_asdp),
             deparse(knockoff::create.solve_sdp)), file.path(out,"r_sampler_source.txt"))
