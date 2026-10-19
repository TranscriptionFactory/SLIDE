#!/usr/bin/env Rscript
# Attribution probe (Option B): compute R's knockoff diag_s (create.solve_asdp, the same
# call secondKO uses) on the EXACT Sigma = cov(Z) that rust solves, for head-to-head with
# rust's emitted diag_s. A systematically LARGER s -> knockoffs less correlated with the
# originals -> originals beat their knockoffs more often -> MORE per-draw selections.
# This localizes the rust-vs-R per-draw over-selection to the ASDP construction.
#
# Usage: Rscript emit_diags_compare.R <sigma_cov.csv> <out_asdp.csv> <out_sdp.csv>

suppressPackageStartupMessages(library(knockoff))
args <- commandArgs(trailingOnly = TRUE)
sigma_path <- args[1]; out_asdp <- args[2]; out_sdp <- args[3]

Sigma <- as.matrix(read.csv(sigma_path, header = FALSE))
colnames(Sigma) <- NULL
cat(sprintf("Sigma=%dx%d  diag range [%.5g, %.5g]\n",
            nrow(Sigma), ncol(Sigma), min(diag(Sigma)), max(diag(Sigma))))

s_asdp <- knockoff::create.solve_asdp(Sigma)
write.table(s_asdp, out_asdp, row.names = FALSE, col.names = FALSE)
cat(sprintf("ASDP diag_s: len=%d range [%.6g, %.6g] sum=%.6g\n",
            length(s_asdp), min(s_asdp), max(s_asdp), sum(s_asdp)))

s_sdp <- tryCatch(knockoff::create.solve_sdp(Sigma), error = function(e) {
  cat("create.solve_sdp failed:", conditionMessage(e), "\n"); rep(NA_real_, ncol(Sigma))
})
write.table(s_sdp, out_sdp, row.names = FALSE, col.names = FALSE)
if (!all(is.na(s_sdp)))
  cat(sprintf("SDP  diag_s: len=%d range [%.6g, %.6g] sum=%.6g\n",
              length(s_sdp), min(s_sdp), max(s_sdp), sum(s_sdp)))
