#!/usr/bin/env Rscript
# Rank-flip diagnostic: run SLIDE-R's REAL two-stage chunked selectShortFreq on the EXACT
# Z rust used (delta=0.1, K=88, n=24), dumping per-chunk tab_data + stage-one union +
# stage-two tab_data, for head-to-head comparison with rust's per-LF selection_counts.
#
# Sources the INSTRUMENTED clone (vendor/SLIDE-jishnu/R) — not the installed SLIDE — so the
# dumping (SLIDE_DUMP_DIR-gated) is exercised. Params match rust exactly: spec=0.1, fdr=0.1,
# elbow=FALSE, niter=500, f_size=24 (= calcDefaultFsize(24,88)), sequential (deterministic;
# pixi R lacks doParallel — counts are an expectation over niter so parallelism is irrelevant).
#
# Usage: Rscript diag_rank_flip_r.R <rust_z.csv> <Y.csv> <dump_dir> [niter]

suppressPackageStartupMessages({
  library(knockoff); library(glmnet); library(corpcor); library(foreach)
})

args <- commandArgs(trailingOnly = TRUE)
z_path   <- args[1]
y_path   <- args[2]
dump_dir <- args[3]
niter    <- if (length(args) >= 4) as.integer(args[4]) else 500L

SLIDE_R <- "/ix/djishnu/Aaron/1_general_use/omni-SLIDE/vendor/SLIDE-jishnu/R"
for (f in c("secondKO.R", "findFrequent.R", "findOptIter.R", "selectShortFreq.R")) {
  source(file.path(SLIDE_R, f))
}

z <- as.matrix(read.csv(z_path, row.names = 1, check.names = FALSE))
ydf <- read.csv(y_path, row.names = 1)
y <- as.matrix(ydf[, 1, drop = FALSE])
cat(sprintf("Z: %d x %d (cols %s..%s), y: %d\n",
            nrow(z), ncol(z), colnames(z)[1], colnames(z)[ncol(z)], length(y)))

# f_size = calcDefaultFsize(n=24, K=88) = 24  (24<=88 & 88<100 & |24-88|>2 -> n)
f_size <- 24L
Sys.setenv(SLIDE_DUMP_DIR = dump_dir)

t0 <- proc.time()
final <- selectShortFreq(z = z, y = y, spec = 0.1, fdr = 0.1, elbow = FALSE,
                         niter = niter, f_size = f_size, parallel = FALSE)
cat(sprintf("selectShortFreq done in %.1fs\n", (proc.time() - t0)["elapsed"]))
cat(sprintf("FINAL marginal LFs (%d): %s\n", length(final), paste(sort(final), collapse = ", ")))
cat(sprintf("Dumps written to %s\n", dump_dir))
