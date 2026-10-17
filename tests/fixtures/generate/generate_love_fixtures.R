#!/usr/bin/env Rscript
# Generate reference fixtures from the canonical R LOVE package (CRAN).
#
# Usage:
#   Rscript tests/fixtures/generate/generate_love_fixtures.R
#
# Prerequisites:
#   - R >= 4.4 with the LOVE package installed
#   - SSc X.csv at the path below (or override via SSC_X_PATH env var)

library(LOVE)

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
ssc_x_path <- Sys.getenv(
  "SSC_X_PATH",
  unset = "/ix/djishnu/Aaron/1_general_use/SLIDE/Data_Scripts/SSc/UnTx/X.csv"
)
stopifnot(file.exists(ssc_x_path))

# Derive script dir from commandArgs when run via Rscript
args <- commandArgs(trailingOnly = FALSE)
file_arg <- grep("^--file=", args, value = TRUE)
if (length(file_arg) > 0) {
  script_dir <- normalizePath(dirname(sub("^--file=", "", file_arg)))
} else {
  script_dir <- getwd()
}
fixture_root <- normalizePath(file.path(script_dir, ".."), mustWork = TRUE)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
write_csv_matrix <- function(mat, path) {
  write.csv(mat, path, row.names = FALSE)
}

write_json_scalar <- function(value, path) {
  writeLines(as.character(value), path)
}

write_json_vec <- function(vec, path) {
  # Write as a JSON array of numbers
  writeLines(paste0("[", paste(vec, collapse = ","), "]"), path)
}

# ---------------------------------------------------------------------------
# Load SSc X
# ---------------------------------------------------------------------------
X_raw <- as.matrix(read.csv(ssc_x_path, row.names = 1))
cat("SSc X dimensions:", nrow(X_raw), "x", ncol(X_raw), "\n")

# ---------------------------------------------------------------------------
# LOVE fixtures: two delta configs, pure_homo=TRUE
# ---------------------------------------------------------------------------
configs <- list(
  list(delta = 0.01, dir_name = "ssc_delta_0_01"),
  list(delta = 0.1,  dir_name = "ssc_delta_0_10")
)

for (cfg in configs) {
  cat("\n--- Running LOVE with delta =", cfg$delta, "---\n")
  out_dir <- file.path(fixture_root, "love", cfg$dir_name)
  dir.create(out_dir, recursive = TRUE, showWarnings = FALSE)

  res <- LOVE(X_raw, pure_homo = TRUE, delta = cfg$delta)

  # Input X (centered, as LOVE processes it internally)
  X_centered <- scale(X_raw, center = TRUE, scale = FALSE)
  write_csv_matrix(X_centered, file.path(out_dir, "input_x.csv"))

  # Config
  config_json <- paste0(
    '{"delta":', cfg$delta,
    ',"pure_homo":true',
    ',"n":', nrow(X_raw),
    ',"p":', ncol(X_raw),
    '}'
  )
  writeLines(config_json, file.path(out_dir, "config.json"))

  # K
  write_json_scalar(res$K, file.path(out_dir, "expected_k.json"))
  cat("  K =", res$K, "\n")

  # pureVec (R uses 1-based indexing; convert to 0-based for Rust)
  pure_vec_0 <- res$pureVec - 1L
  write_json_vec(pure_vec_0, file.path(out_dir, "expected_pure_vec.json"))
  cat("  |pureVec| =", length(res$pureVec), "\n")

  # A matrix
  write_csv_matrix(res$A, file.path(out_dir, "expected_a.csv"))
  cat("  A dims:", nrow(res$A), "x", ncol(res$A), "\n")

  # C matrix
  write_csv_matrix(res$C, file.path(out_dir, "expected_c.csv"))

  # Omega (may be NULL if all pure)
  if (!is.null(res$Omega)) {
    write_csv_matrix(res$Omega, file.path(out_dir, "expected_omega.csv"))
  }

  # Gamma
  write_json_vec(res$Gamma, file.path(out_dir, "expected_gamma.json"))

  # optDelta
  write_json_scalar(res$optDelta, file.path(out_dir, "expected_opt_delta.json"))
  cat("  optDelta =", res$optDelta, "\n")
}

# ---------------------------------------------------------------------------
# Prescreen fixture: Screen_X on SSc data
# ---------------------------------------------------------------------------
cat("\n--- Running Screen_X ---\n")
pre_dir <- file.path(fixture_root, "prescreen", "ssc")
dir.create(pre_dir, recursive = TRUE, showWarnings = FALSE)

set.seed(42)  # Screen_X uses random CV folds
screen_res <- Screen_X(X_raw)

write_csv_matrix(scale(X_raw, center = TRUE, scale = FALSE),
                 file.path(pre_dir, "input_x.csv"))
writeLines('{"nfolds":10,"nthresh":50,"max_prop":0.5,"seed":42}',
           file.path(pre_dir, "config.json"))

# noise_ind: convert to 0-based
noise_ind_0 <- screen_res$noise_ind - 1L
write_json_vec(noise_ind_0, file.path(pre_dir, "expected_noise_ind.json"))
write_json_scalar(screen_res$thresh_min, file.path(pre_dir, "expected_thresh_min.json"))
cat("  |noise_ind| =", length(screen_res$noise_ind), "\n")
cat("  thresh_min =", screen_res$thresh_min, "\n")

cat("\nDone. Fixtures written to:", fixture_root, "\n")
