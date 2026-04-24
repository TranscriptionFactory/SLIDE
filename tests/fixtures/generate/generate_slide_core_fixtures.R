options(warn = 1)

suppressPackageStartupMessages({
  library(knockoff)
  library(MASS)
})

script_arg <- commandArgs(trailingOnly = FALSE)
script_path <- sub("^--file=", "", script_arg[grep("^--file=", script_arg)])
repo_root <- dirname(dirname(dirname(dirname(normalizePath(script_path)))))
slide_r_dir <- file.path(repo_root, "src", "loveslide", "slide_r")
for (path in list.files(slide_r_dir, pattern = "\\.[Rr]$", full.names = TRUE)) {
  source(path)
}

write_json_array <- function(values, path) {
  if (length(values) == 0) {
    writeLines("[]", path)
  } else {
    writeLines(paste0("[", paste(values, collapse = ", "), "]"), path)
  }
}

write_json_scalar <- function(value, path) {
  writeLines(as.character(value), path)
}

out_dir <- file.path(repo_root, "tests", "fixtures", "slide_core", "hiv_delta_0_10_lambda_0_50")
dir.create(out_dir, recursive = TRUE, showWarnings = FALSE)

x <- read.csv(
  file.path(repo_root, "example", "HIV+cytokines_X.csv"),
  row.names = 1,
  check.names = FALSE
)
y_df <- read.csv(
  file.path(repo_root, "example", "HIV+cytokines_y.csv"),
  row.names = 1,
  check.names = FALSE
)
y <- as.integer(factor(y_df[[1]], levels = sort(unique(y_df[[1]])))) - 1L

config_lines <- c(
  "{",
  "  \"dataset\": \"hiv_cytokines\",",
  "  \"delta\": 0.1,",
  "  \"lambda\": 0.5,",
  "  \"thresh_fdr\": 0.2,",
  "  \"fdr\": 0.1,",
  "  \"spec\": 0.2,",
  "  \"niter\": 20,",
  "  \"do_interacts\": false,",
  "  \"knockoff_method\": \"equi\",",
  "  \"knockoff_offset\": 0,",
  "  \"statistic\": \"stat.glmnet_lambdasmax\",",
  "  \"response_type\": \"binary\"",
  "}"
)
writeLines(config_lines, file.path(out_dir, "config.json"))

love_result <- getLatentFactors(
  x = x,
  delta = c(0.1),
  thresh_fdr = 0.2,
  lbd = 0.5,
  rep_cv = 5,
  verbose = FALSE
)

x_std <- scale(x, center = TRUE, scale = TRUE)
sigma <- cor(x_std)
A_hat <- as.matrix(love_result$A)
C_hat <- as.matrix(love_result$C)
Gamma_hat <- as.numeric(love_result$Gamma)
I_hat <- as.integer(love_result$pureVec)
non_pure <- setdiff(seq_len(ncol(A_hat)), I_hat)

if (length(non_pure) > 0) {
  A_J <- A_hat[non_pure, , drop = FALSE]
  sigma_diag <- diag(sigma[non_pure, non_pure, drop = FALSE])
  aca_diag <- diag(A_J %*% C_hat %*% t(A_J))
  Gamma_hat[non_pure] <- sigma_diag - aca_diag
}

Gamma_hat[Gamma_hat < 0] <- 1e2
Gamma_hat[Gamma_hat == 0] <- 1e-10
Gamma_inv <- diag(Gamma_hat ^ (-1), nrow = length(Gamma_hat))
G_hat <- t(A_hat) %*% Gamma_inv %*% A_hat + solve(C_hat)
Z_hat <- x_std %*% Gamma_inv %*% A_hat %*% MASS::ginv(G_hat)

selection_counts <- integer(ncol(Z_hat))
for (iter in seq_len(20)) {
  set.seed(42 + iter - 1)
  result <- knockoff.filter(
    X = Z_hat,
    y = y,
    knockoffs = function(X) create.second_order(X, method = "equi"),
    statistic = stat.glmnet_lambdasmax,
    offset = 0,
    fdr = 0.1
  )
  selected <- as.integer(result$selected)
  if (length(selected) > 0) {
    selection_counts[selected] <- selection_counts[selected] + 1L
  }
}

selected_zero_based <- which(selection_counts >= ceiling(0.2 * 20)) - 1L

write.csv(x, file.path(out_dir, "input_x.csv"), row.names = TRUE)
write.csv(data.frame(y = y, row.names = rownames(x)), file.path(out_dir, "input_y.csv"), row.names = TRUE)
write_json_scalar(as.integer(love_result$K), file.path(out_dir, "expected_k.json"))
write_json_array(as.integer(selected_zero_based), file.path(out_dir, "expected_sig_lfs.json"))
