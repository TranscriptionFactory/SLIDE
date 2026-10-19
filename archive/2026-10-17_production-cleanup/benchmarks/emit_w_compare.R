#!/usr/bin/env Rscript
# Attribution probe (Option B step 5): given the EXACT knockoff draw X-tilde that rust
# emitted (emit_knockoff_w), compute R's knockoff W statistic (stat.glmnet_lambdasmax) on
# the identical [Z, X-tilde, y]. Compare to rust's emitted W. Same X-tilde -> same W means
# the rust-vs-R per-draw selection gap is the knockoff DRAW/construction (not W); different
# W means it is the lasso W statistic (rust Gaussian-CD nlambda=250 vs glmnet path).
#
# Usage: Rscript emit_w_compare.R <z_numeric.csv> <xtilde.csv> <y_numeric.csv> <out_R_W.csv>

suppressPackageStartupMessages(library(knockoff))
args <- commandArgs(trailingOnly = TRUE)
z_path  <- args[1]; xt_path <- args[2]; y_path <- args[3]; out_path <- args[4]

Z  <- as.matrix(read.csv(z_path,  header = FALSE))
Xt <- as.matrix(read.csv(xt_path, header = FALSE))
y  <- as.matrix(read.csv(y_path,  header = FALSE))[, 1]
stopifnot(ncol(Z) == ncol(Xt), nrow(Z) == nrow(Xt), nrow(Z) == length(y))
cat(sprintf("Z=%dx%d  Xt=%dx%d  y=%d\n", nrow(Z), ncol(Z), nrow(Xt), ncol(Xt), length(y)))

W <- knockoff::stat.glmnet_lambdasmax(Z, Xt, y)
write.table(W, out_path, row.names = FALSE, col.names = FALSE)
cat(sprintf("R W: len=%d range=[%.5g, %.5g]\n", length(W), min(W), max(W)))

Th <- knockoff::knockoff.threshold(W, fdr = 0.1, offset = 0)
sel <- which(W >= Th)
cat(sprintf("R knockoff.threshold(fdr=0.1, offset=0) T=%.6g\n", Th))
cat(sprintf("R selected this draw (0-based): %s\n", paste(sel - 1, collapse = ",")))
