options(warn = 1)

suppressPackageStartupMessages({
  library(knockoff)
  library(MASS)
})

script_arg <- commandArgs(trailingOnly = FALSE)
script_path <- sub("^--file=", "", script_arg[grep("^--file=", script_arg)])
repo_root <- dirname(dirname(dirname(dirname(normalizePath(script_path)))))
slide_r_dir <- file.path(repo_root, "src", "slide_py", "slide_r")
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

args <- commandArgs(trailingOnly = TRUE)
if (length(args) > 0 && args[1] == "--interaction-cap") {
  stopifnot(length(args) == 2)
  capture <- normalizePath(args[2])
  z <- as.matrix(read.csv(file.path(capture, "z.csv"), header = FALSE))
  y <- scan(file.path(capture, "y.csv"), quiet = TRUE)
  colnames(z) <- paste0("Z", 0:(ncol(z) - 1))
  reference <- new.env(parent = globalenv())
  for (name in c("correctIt", "interUnion", "interactionSLIDE", "selectShortFreq")) {
    sys.source(file.path(repo_root, "vendor", "SLIDE-jishnu", "R", paste0(name, ".R")), reference)
  }
  corrected <- as.numeric(reference$correctIt(z[, 1], y))
  x <- z[, 1] * z[, -1]
  selected <- vector("list", 4)
  statistics <- matrix(0, 4, ncol(x))
  for (draw in 0:3) {
    xt <- as.matrix(read.csv(file.path(capture, "xtilde", paste0("xt_", draw, ".csv")), header = FALSE))
    swap <- as.integer(readBin(file.path(capture, "xtilde", paste0("swap_", draw, ".raw")), "raw", ncol(x)))
    mask <- matrix(swap, nrow(x), ncol(x), byrow = TRUE)
    augmented <- cbind(x * (1 - mask) + xt * mask, xt * (1 - mask) + x * mask)
    entry <- knockoff:::lasso_max_lambda(augmented, corrected, nlambda = 30)
    p <- ncol(x)
    w <- pmax(entry[1:p], entry[p + 1:p]) * sign(entry[1:p] - entry[p + 1:p]) * (1 - 2 * swap)
    statistics[draw + 1, ] <- w
    selected[[draw + 1]] <- colnames(x)[which(w >= knockoff::knockoff.threshold(w, fdr = 0.9, offset = 0))]
  }
  counts <- vapply(colnames(x), function(name) sum(vapply(selected, function(s) name %in% s, logical(1))), integer(1))
  # Replay the fitted draws through R's cap controller and frequency-selection mode.
  Sys.setenv(SLIDE_OPT_ITER = "0")
  ladder <- numeric()
  reference$secondKO <- function(z, y, fdr, niter, parallel) {
    stopifnot(niter == 4, isTRUE(all.equal(unname(z), unname(x))), max(abs(y - corrected)) < 1e-12)
    tab <- matrix(counts[counts > 0], nrow = 1, dimnames = list(NULL, colnames(z)[counts > 0]))
    list(tab_data = tab, selected_list = lapply(selected, function(s) colnames(z)[match(s, colnames(x))]))
  }
  select_reference <- reference$selectShortFreq
  reference$selectShortFreq <- function(...) {
    ladder <<- c(ladder, list(...)$spec)
    select_reference(...)
  }
  result <- reference$interactionSLIDE(z, y, m = 3, marginals = "Z0", method = 3, spec = 0.1,
                                       fdr = 0.9, niter = 4, parallel = FALSE)
  jsonlite::write_json(list(corrected_y = corrected, counts = unname(counts),
    selected = result$interaction_vars, spec_ladder = ladder,
    r_version = R.version.string, glmnet_version = as.character(packageVersion("glmnet"))),
    file.path(capture, "r_reference.json"), auto_unbox = TRUE, digits = 17, pretty = TRUE)
  write.table(statistics, file.path(capture, "r_w.csv"), sep = ",", row.names = FALSE, col.names = FALSE)
  print(list(counts = counts, selected = result$interaction_vars, spec_ladder = ladder))
  quit(save = "no")
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
