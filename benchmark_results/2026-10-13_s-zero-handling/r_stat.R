# R knockoff statistic on fixed X~ draws and swap masks (stat.glmnet_lambdasmax with the swap supplied).
# Columns: W_1..W_p, threshold, max(entry orig, entry knockoff)_1..p.
# usage: Rscript r_stat.R <inputs_dir> <arm> <draws> <n> <p> <fdr>; writes <inputs_dir>/w_r_<arm>.csv
args <- commandArgs(trailingOnly = TRUE)
dir <- args[1]; arm <- args[2]
draws <- as.integer(args[3]); n <- as.integer(args[4]); p <- as.integer(args[5]); fdr <- as.numeric(args[6])
rd <- function(f, k) readBin(file.path(dir, f), "double", k, endian = "little")
X <- matrix(rd("z.raw", n * p), n, p, byrow = TRUE)
y <- rd("y.raw", n)
xt <- rd(sprintf("xt_%s.raw", arm), draws * n * p)
sw <- matrix(as.integer(readBin(file.path(dir, "swaps.raw"), "raw", draws * p)), draws, p, byrow = TRUE)
out <- matrix(NA_real_, draws, 2 * p + 1)
for (d in seq_len(draws)) {
  Xk <- matrix(xt[((d - 1) * n * p + 1):(d * n * p)], n, p, byrow = TRUE)
  swap.M <- matrix(sw[d, ], n, p, byrow = TRUE)
  X.swap <- X * (1 - swap.M) + Xk * swap.M
  Xk.swap <- X * swap.M + Xk * (1 - swap.M)
  Z <- knockoff:::lasso_max_lambda(cbind(X.swap, Xk.swap), y, method = "glmnet")
  W <- pmax(Z[1:p], Z[1:p + p]) * sign(Z[1:p] - Z[1:p + p]) * (1 - 2 * sw[d, ])
  out[d, ] <- c(W, knockoff::knockoff.threshold(W, fdr = fdr, offset = 0), pmax(Z[1:p], Z[1:p + p]))
}
write.csv(out, file.path(dir, sprintf("w_r_%s.csv", arm)), row.names = FALSE)
