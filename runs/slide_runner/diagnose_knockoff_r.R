#!/usr/bin/env Rscript
# Compute knockoff diagnostics from R side for comparison with Python.
# Saves diag_s, selection_counts, and per-iteration W-statistics.
#
# Usage:
#   Rscript diagnose_knockoff_r.R <z_matrix.csv> <y.csv> <out_dir> [niter]

suppressPackageStartupMessages({
  library(knockoff)
  library(glmnet)
})

args <- commandArgs(trailingOnly = TRUE)
if (length(args) < 3) stop("Usage: diagnose_knockoff_r.R <z.csv> <y.csv> <out_dir> [niter]")

z_path <- args[1]
y_path <- args[2]
out_dir <- args[3]
niter <- if (length(args) >= 4) as.integer(args[4]) else 500L

dir.create(out_dir, recursive = TRUE, showWarnings = FALSE)

cat("Loading data...\n")
z <- as.matrix(read.csv(z_path, row.names = 1))
y_df <- read.csv(y_path, row.names = 1)
y <- as.matrix(y_df)

cat(sprintf("Z: %d x %d, y: %d\n", nrow(z), ncol(z), length(y)))

# ── Step 1: Compute covariance and diag_s ──────────────────────────────
cat("\n=== Computing diag_s (ASDP) ===\n")
mu <- colMeans(z)
Sigma <- cov(z)  # ddof = n-1

t0 <- proc.time()
diag_s <- tryCatch(
  knockoff::create.solve_asdp(Sigma),
  error = function(e) {
    cat("ASDP failed, trying equi:", e$message, "\n")
    knockoff::create.solve_equi(Sigma)
  }
)
dt <- (proc.time() - t0)["elapsed"]
cat(sprintf("  Time: %.1fs\n", dt))
cat(sprintf("  diag_s: range=[%.6f, %.6f], mean=%.6f\n",
            min(diag_s), max(diag_s), mean(diag_s)))

write.csv(data.frame(diag_s = diag_s), file.path(out_dir, "diag_s_r.csv"),
          row.names = FALSE)

# ── Step 2: Run knockoff voting ────────────────────────────────────────
cat(sprintf("\n=== Running %d knockoff iterations ===\n", niter))

p <- ncol(z)
selection_counts <- rep(0L, p)
names(selection_counts) <- colnames(z)

# Pre-cache the knockoff generator (like R SLIDE's secondKO)
create_knockoffs_cached <- function(X) {
  knockoff::create.gaussian(X, mu, Sigma, diag_s = diag_s)
}

# Store W-statistics for first 20 iterations
n_save_w <- min(20L, niter)
W_matrix <- matrix(0, nrow = n_save_w, ncol = p)
colnames(W_matrix) <- colnames(z)

selected_list <- vector("list", niter)

t0 <- proc.time()
for (i in seq_len(niter)) {
  result <- knockoff.filter(
    X = z,
    y = y,
    knockoffs = create_knockoffs_cached,
    statistic = stat.glmnet_lambdasmax,
    offset = 0,
    fdr = 0.1
  )

  sel_names <- names(result$selected)
  selected_list[[i]] <- sel_names

  if (!is.null(sel_names)) {
    for (nm in sel_names) {
      if (nm %in% names(selection_counts)) {
        selection_counts[nm] <- selection_counts[nm] + 1L
      }
    }
  }

  # Save W-statistics for first iterations
  if (i <= n_save_w) {
    W_matrix[i, ] <- result$statistic
  }

  if (i %% 50 == 0) cat(sprintf("  %d/%d\n", i, niter))
}
dt <- (proc.time() - t0)["elapsed"]
cat(sprintf("  Total time: %.1fs (%.2fs/iter)\n", dt, dt / niter))

# ── Save results ────────────────────────────────────────────────────────
write.csv(data.frame(variable = names(selection_counts),
                     count = as.integer(selection_counts)),
          file.path(out_dir, "selection_counts_r.csv"), row.names = FALSE)

write.csv(W_matrix, file.path(out_dir, "W_matrix_r.csv"), row.names = FALSE)

# ── Step 3: Apply frequency threshold and findOptIter ──────────────────
spec <- 0.1
freq_threshold <- niter * spec
freq_inds <- which(selection_counts >= freq_threshold)
freq_vars <- names(selection_counts)[freq_inds]

cat(sprintf("\n=== Frequency threshold (>= %.0f) ===\n", freq_threshold))
cat(sprintf("  freq_vars (%d): %s\n", length(freq_vars), paste(freq_vars, collapse = ", ")))

# findOptIter
source("/ix/djishnu/Aaron/1_general_use/SLIDE/R/findOptIter.R")
if (length(freq_vars) > 0) {
  opt_vars <- findOptIter(freq_vars = freq_vars, selected_list = selected_list)
  cat(sprintf("  findOptIter (%d): %s\n", length(opt_vars), paste(opt_vars, collapse = ", ")))
} else {
  opt_vars <- character(0)
  cat("  findOptIter: no freq_vars, skipping\n")
}

# Summary
cat(sprintf("\n=== Top 30 by selection count ===\n"))
top <- head(sort(selection_counts, decreasing = TRUE), 30)
for (i in seq_along(top)) {
  cat(sprintf("  %s: %d (%.1f%%)\n", names(top)[i], top[i], 100 * top[i] / niter))
}

cat("\nDone.\n")
