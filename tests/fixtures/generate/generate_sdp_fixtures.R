#!/usr/bin/env Rscript
# Generate deterministic reference fixtures from R knockoff for slide-sdp parity tests.

suppressPackageStartupMessages(library(knockoff))

args <- commandArgs(trailingOnly = FALSE)
file_arg <- grep("^--file=", args, value = TRUE)
if (length(file_arg) > 0) {
  script_dir <- normalizePath(dirname(sub("^--file=", "", file_arg)))
} else {
  script_dir <- getwd()
}
fixture_root <- normalizePath(file.path(script_dir, "..", "sdp"), mustWork = FALSE)
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

write_fixture <- function(name, sigma) {
  out_dir <- file.path(fixture_root, name)
  dir.create(out_dir, recursive = TRUE, showWarnings = FALSE)

  write_csv_matrix(sigma, file.path(out_dir, "input_sigma.csv"))
  write_csv_matrix(
    matrix(knockoff::create.solve_equi(sigma), ncol = 1),
    file.path(out_dir, "expected_equi.csv")
  )
  write_csv_matrix(
    matrix(knockoff::create.solve_sdp(sigma), ncol = 1),
    file.path(out_dir, "expected_sdp.csv")
  )
  write_csv_matrix(
    matrix(knockoff::create.solve_asdp(sigma), ncol = 1),
    file.path(out_dir, "expected_asdp.csv")
  )
}

write_fixture(
  "corr3",
  matrix(c(
    1.0, 0.8, 0.2,
    0.8, 1.0, 0.1,
    0.2, 0.1, 1.0
  ), nrow = 3, byrow = TRUE)
)

write_fixture(
  "block4",
  matrix(c(
    1.0, 0.95, 0.05, 0.02,
    0.95, 1.0, 0.04, 0.03,
    0.05, 0.04, 1.0, 0.92,
    0.02, 0.03, 0.92, 1.0
  ), nrow = 4, byrow = TRUE)
)

write_fixture(
  "cov4",
  matrix(c(
    4.0, 1.2, 0.4, 0.1,
    1.2, 3.0, 0.2, 0.0,
    0.4, 0.2, 2.5, 1.1,
    0.1, 0.0, 1.1, 1.8
  ), nrow = 4, byrow = TRUE)
)

cat("SDP fixtures written to:", fixture_root, "\n")
