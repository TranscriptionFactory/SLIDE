#!/usr/bin/env Rscript

suppressPackageStartupMessages({
  library(jsonlite)
  library(pROC)
})

root <- normalizePath(file.path(dirname(sub("^--file=", "", commandArgs(FALSE)[grep("^--file=", commandArgs(FALSE))])), "../../.."), mustWork = TRUE)
fixture_dir <- file.path(root, "tests/fixtures/sample_cv/2026-11-01_r-reference")
tmp_dir <- file.path(root, ".tmpfiles/2026-11-01_samplecv-parity/tmp")
dir.create(fixture_dir, recursive = TRUE, showWarnings = FALSE)
dir.create(tmp_dir, recursive = TRUE, showWarnings = FALSE)

cases <- list(
  list(name = "binary_auc_reversed", condition = "auc",
       z = cbind(0:7, 1:8), y = c(0, 0, 0, 1, 1, 1, 1, 0), marginals = 1L, pairs = list(),
       train = rep(list(1:6), 2)),
  list(name = "binary_auc_ties", condition = "auc",
       z = cbind(c(0:6, 6), 1:8), y = c(0, 0, 0, 1, 1, 1, 1, 0), marginals = 1L, pairs = list(),
       train = rep(list(1:6), 2)),
  list(name = "continuous_spearman_raw_interaction", condition = "corr",
       z = cbind(c(-2, -1, 0, 1, 2, 3, 4, 5), c(3, 1, 4, 2, 5, 0, 6, 7)),
       y = c(1, 2, 1, 4, 3, 5, 4, 8), marginals = c(1L, 2L), pairs = list(c(1L, 2L)),
       train = list(c(1L, 2L, 3L, 5L, 6L, 7L), c(1L, 3L, 4L, 5L, 7L, 8L))),
  list(name = "constant_response_missing_metric", condition = "corr",
       z = cbind(0:7, (0:7)^2), y = rep(2, 8), marginals = 1L, pairs = list(),
       train = list(c(1L, 2L, 3L, 5L, 6L, 7L), c(1L, 3L, 4L, 5L, 7L, 8L))),
  list(name = "rank_deficient_heldout_nonestimable", condition = "corr",
       z = cbind(c(0, 1, 2, 3, 4, 5, 6, 7), c(0, 2, 4, 6, 8, 1e-10 + 10, 12, 14), c(0, 2, 4, 6, 8, 99, 12, 14)),
       y = c(0, 1, 2, 4, 3, 6, 5, 8), marginals = c(1L, 2L, 3L), pairs = list(),
       train = list(c(1L, 2L, 3L, 5L, 6L, 7L), c(1L, 2L, 4L, 5L, 7L, 8L))),
  list(name = "endpoint_union_guard", condition = "corr",
       z = matrix(seq_len(16), ncol = 4), y = c(1, 2, 3, 4), marginals = c(1L, 2L, 3L, 4L), pairs = list(), train = list())
)

sample_cv_env <- new.env(parent = globalenv())
sample_cv_env$sample <- sample
sys.source(file.path(root, "R/pairwiseInteractions.R"), envir = sample_cv_env)
sys.source(file.path(root, "R/sampleCV.R"), envir = sample_cv_env)
run_case <- function(case) {
  n <- length(case$y)
  z <- as.data.frame(case$z)
  names(z) <- paste0("Z", seq_len(ncol(z)))
  interaction_vars <- if (length(case$pairs)) paste0("Z", vapply(case$pairs, `[[`, integer(1), 1), ".Z", vapply(case$pairs, `[[`, integer(1), 2)) else character()
  slide_res <- list(marginal_vals = case$marginals,
                    interaction = list(p1 = if (length(case$pairs)) vapply(case$pairs, `[[`, integer(1), 1) else integer(),
                                       p2 = if (length(case$pairs)) vapply(case$pairs, `[[`, integer(1), 2) else integer()),
                    SLIDE_res = list(interaction_vars = interaction_vars))
  queue <- c(list(seq_len(ceiling((1 - 1/4) * n))), case$train)
  sample_cv_env$sample <- function(x, size, replace = FALSE, ...) {
    if (!length(queue)) stop("unexpected sample() call after fixed splits")
    result <- queue[[1]]
    queue <<- queue[-1]
    result
  }
  if (length(case$train) == 0) {
    value <- sample_cv_env$sampleCV(matrix(case$y, ncol = 1), z, slide_res, sampleCV_K = 4, condition = case$condition, sampleCV_iter = 1, out_path = paste0(tmp_dir, "/"))
    stopifnot(identical(as.character(value), "NA"))
    return(list(metric = NA_real_, predictions = list(), last_coefficients = list(), last_rank = NA_integer_, last_aliased_terms = integer(), last_train_indices = integer(), term_order = character(), guard = "model_too_large"))
  }
  value <- withCallingHandlers(
    sample_cv_env$sampleCV(matrix(case$y, ncol = 1), z, slide_res, sampleCV_K = 4, condition = case$condition, sampleCV_iter = length(case$train), out_path = paste0(tmp_dir, "/")),
    warning = function(w) invokeRestart("muffleWarning"))
  model <- readRDS(file.path(tmp_dir, "sampleCV_model.RDS"))
  test_rows <- unlist(lapply(case$train, function(train) setdiff(seq_len(n), train)), use.names = FALSE)
  predictions <- data.frame(iteration = rep(seq_along(case$train) - 1L, each = n - length(case$train[[1]])),
                            row_index = test_rows - 1L, y_true = as.numeric(model$samplesCV$true_y),
                            y_pred = as.numeric(model$samplesCV$pred_y))
  stopifnot(length(test_rows) == length(model$samplesCV$pred_y))
  coefficients <- as.list(coef(model))
  term_order <- c("(Intercept)", paste0("Z", case$marginals),
                  unlist(lapply(case$pairs, function(pair) paste0("Z", pair[1], ".Z", pair[2]))))
  names(coefficients) <- term_order
  list(metric = as.numeric(value), predictions = lapply(seq_len(nrow(predictions)), function(i) as.list(predictions[i, , drop = FALSE])),
       last_coefficients = coefficients, last_rank = model$rank,
       last_aliased_terms = I(which(is.na(coef(model))) - 1L), last_train_indices = I(as.integer(case$train[[length(case$train)]] - 1L)),
       term_order = term_order, guard = "ok")
}

out <- lapply(cases, function(case) {
  expected <- run_case(case)
  list(name = case$name, condition = case$condition, y = unname(case$y), z = unname(split(case$z, row(case$z))),
       marginal_indices = I(as.integer(case$marginals - 1L)),
       interaction_pairs = lapply(case$pairs, function(pair) as.integer(pair - 1L)),
       splits = lapply(case$train, function(train) list(train_indices = as.integer(train - 1L))), expected = expected)
})
write_json(list(schema_version = 1L, cases = out), file.path(fixture_dir, "sample_cv.json"), auto_unbox = TRUE, pretty = TRUE, na = "null", digits = 16)
writeLines(c("# R sampleCV reference fixtures", "", "Generated with `Rscript tests/fixtures/generate/generate_sample_cv_fixtures.R`.", "The generator sources production `R/sampleCV.R`, uses `K=4`, and supplies the listed explicit zero-based train splits (iteration count equals the split count). It records R `lm`/`predict`, pooled Spearman, and fixed-direction pROC AUC results. Cases cover reversed AUC (0), tied AUC (0.5), raw interactions, constant-response missing metric, ordered exact/near aliases, and the endpoint-union guard. R >= 4.4, jsonlite, and pROC are required."), file.path(fixture_dir, "README.md"))
