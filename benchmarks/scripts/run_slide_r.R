#!/usr/bin/env Rscript
# Pure-R SLIDE benchmark runner: one (delta, lambda) combo per invocation.
#
# Writes the same per-combo files the Python/R-knockoffs backends produce so
# benchmarks/scripts/benchmark_backends.py:load_python_result can pick them up:
#   <out_dir>/<delta>_<lambda>_out/
#     sig_LFs.txt
#     sig_interacts.txt
#     z_matrix.csv
#     A.csv  (best-effort)
#     SLIDE_LFs.rds + AllLatentFactors.rds  (native SLIDE outputs)

# Lightweight base-R arg parser (avoids optparse/yaml deps not in pixi R env).
parse_cli <- function(argv) {
  out <- list()
  i <- 1
  while (i <= length(argv)) {
    a <- argv[[i]]
    if (!startsWith(a, "--")) {
      stop("unexpected positional arg: ", a)
    }
    key <- sub("^--", "", a)
    # boolean flags have no value
    if (key %in% c("y-factor")) {
      out[[key]] <- TRUE
      i <- i + 1
      next
    }
    if (i + 1 > length(argv)) stop("missing value for --", key)
    out[[key]] <- argv[[i + 1]]
    i <- i + 2
  }
  out
}

defaults <- list(
  workflow         = "full-fit",
  `cv-repeats`     = "15",
  `cv-folds`       = "5",
  `sample-cv-iterations` = "500",
  `sample-cv-k`    = "4",
  niter            = "500",
  spec             = "0.1",
  fdr              = "0.1",
  `thresh-fdr`     = "0.2",
  `slide-top-feats`= "10",
  threads          = "10",
  `base-seed`      = NA_character_,
  `y-factor`       = FALSE
)
opts <- modifyList(defaults, parse_cli(commandArgs(trailingOnly = TRUE)))

# Coerce numeric/int fields
opts$delta             <- as.numeric(opts$delta)
opts$lambda            <- as.numeric(opts$lambda)
opts$spec              <- as.numeric(opts$spec)
opts$fdr               <- as.numeric(opts$fdr)
opts[["thresh-fdr"]]   <- as.numeric(opts[["thresh-fdr"]])
opts$niter             <- as.integer(opts$niter)
opts[["slide-top-feats"]] <- as.integer(opts[["slide-top-feats"]])
opts$threads           <- as.integer(opts$threads)
opts[["base-seed"]]    <- as.integer(opts[["base-seed"]])
opts[["y-factor"]]     <- isTRUE(opts[["y-factor"]])
for (key in c("cv-repeats", "cv-folds", "sample-cv-iterations", "sample-cv-k")) {
  opts[[key]] <- as.integer(opts[[key]])
  if (is.na(opts[[key]]) || opts[[key]] < 1L) stop("--", key, " must be positive")
}
if (!opts$workflow %in% c("full-fit", "cv")) stop("--workflow must be full-fit or cv")
if (opts[["cv-folds"]] < 2L || opts[["sample-cv-k"]] < 2L) stop("fold counts must be at least 2")
if (opts$workflow == "cv" && is.null(opts[["z-path"]])) stop("--z-path is required for cv")
if (is.na(opts$threads) || opts$threads < 1L) stop("--threads must be a positive integer")

for (k in c("x-path", "y-path", "out-dir", "delta", "lambda")) {
  if (is.null(opts[[k]])) stop("missing required arg --", k)
}

suppressPackageStartupMessages(library(SLIDE))
if (!is.na(opts[["base-seed"]])) set.seed(opts[["base-seed"]])

# Checkout/vendor functions resolve definitions locally, then installed package imports.
# Installed mode copies function bindings to avoid mutating installed namespaces.
slide_source <- Sys.getenv("SLIDE_R_SOURCE", unset = "installed")
script_arg <- grep("^--file=", commandArgs(trailingOnly = FALSE), value = TRUE)
script_dir <- if (length(script_arg) > 0) {
  dirname(normalizePath(sub("^--file=", "", script_arg[[1]])))
} else {
  getwd()
}
if (!slide_source %in% c("installed", "vendor", "checkout")) stop("unknown SLIDE_R_SOURCE: ", slide_source)
repo_dir <- dirname(dirname(script_dir))
source_dir <- find.package("SLIDE")
if (slide_source %in% c("vendor", "checkout")) {
  vendor_dir <- Sys.getenv("SLIDE_VENDOR_DIR", unset = "")
  if (!nzchar(vendor_dir)) {
    vendor_dir <- file.path(dirname(dirname(script_dir)), "vendor", "SLIDE-jishnu", "R")
  }
  if (slide_source == "checkout") vendor_dir <- file.path(repo_dir, "R")
  source_dir <- normalizePath(vendor_dir, mustWork = FALSE)
  if (!dir.exists(vendor_dir)) {
    stop("SLIDE_R_SOURCE source files missing at: ", vendor_dir)
  }
  slide_env    <- new.env(parent = parent.env(asNamespace("SLIDE")))
  vendor_files <- sort(list.files(vendor_dir, pattern = "\\.[Rr]$", full.names = TRUE))
  for (vendor_file in vendor_files) sys.source(vendor_file, envir = slide_env)
  optimize_slide <- get("optimizeSLIDE", envir = slide_env)
  # vendor/ is gitignored, so a fresh checkout has the pristine upstream tree and
  # SLIDE_OPT_ITER=0 would silently produce stock findOptIter numbers. Fail loudly
  # instead; benchmarks/scripts/vendor_slide_opt_iter.patch reapplies the gate.
  if (slide_source == "vendor" && identical(Sys.getenv("SLIDE_OPT_ITER", unset = "1"), "0") &&
      !any(grepl("SLIDE_OPT_ITER",
                 deparse(get("selectShortFreq", envir = slide_env)), fixed = TRUE))) {
    stop("SLIDE_OPT_ITER=0 requested but the vendored selectShortFreq does not honour ",
         "it. Apply benchmarks/scripts/vendor_slide_opt_iter.patch to ", vendor_dir, ".")
  }
  cat(sprintf("[pure_r] SLIDE_R_SOURCE=%s: sourced %d files from %s (SLIDE_OPT_ITER=%s)\n",
              slide_source, length(vendor_files), vendor_dir,
              Sys.getenv("SLIDE_OPT_ITER", unset = "1")))
} else {
  slide_env <- new.env(parent = asNamespace("SLIDE"))
  for (name in ls(asNamespace("SLIDE"), all.names = TRUE)) {
    fun <- get(name, asNamespace("SLIDE"))
    if (is.function(fun) && identical(environment(fun), asNamespace("SLIDE"))) {
      environment(fun) <- slide_env
      assign(name, fun, slide_env)
    }
  }
  optimize_slide <- slide_env$optimizeSLIDE
  cat("[pure_r] SLIDE_R_SOURCE=installed: using SLIDE::optimizeSLIDE\n")
}

# Preserve a stock comparison route without editing installed package files.
# Opt in with SLIDE_R_OPTIMIZE=1; copied runners then also need
# src/slide_py/slide_r/knockoffPerformance.R alongside benchmarks/scripts/.
optimize_interactions <- Sys.getenv("SLIDE_R_OPTIMIZE", unset = "0")
if (!optimize_interactions %in% c("0", "1")) stop("SLIDE_R_OPTIMIZE must be 0 or 1")
if (slide_source == "checkout" && optimize_interactions != "0") {
  stop("SLIDE_R_SOURCE=checkout requires SLIDE_R_OPTIMIZE=0 (unchanged R reference)")
}
if (identical(optimize_interactions, "1")) {
  performance_env <- new.env(parent = globalenv())
  sys.source(file.path(dirname(dirname(script_dir)), "src", "slide_py", "slide_r",
                       "knockoffPerformance.R"), envir = performance_env)
  optimized_env <- performance_env$withPreparedInteractionKnockoffs(slide_env)
  optimize_slide <- optimized_env$optimizeSLIDE
  cat("[pure_r] SLIDE_R_OPTIMIZE=1: cached Gaussian preparation and sparse interaction statistics\n")
} else {
  cat("[pure_r] SLIDE_R_OPTIMIZE=0: unchanged R interaction implementation\n")
}

# Upstream optimizeSLIDE drops ncore, and runSLIDE passes a hard-coded 10.
# Rebind only these entry points, preserving the selected stock/optimized SLIDE.
worker_env <- new.env(parent = environment(optimize_slide))
worker_env$SLIDE <- local({
  implementation <- get("SLIDE", envir = worker_env)
  workers <- opts$threads
  function(..., ncore = NULL) implementation(..., ncore = workers)
})
for (name in c("optimizeSLIDE", "runSLIDE", "benchCV2", "runSLIDEBeta")) {
  fun <- get(name, envir = worker_env)
  environment(fun) <- worker_env
  assign(name, fun, envir = worker_env)
}
optimize_slide <- worker_env$optimizeSLIDE
cat(sprintf("[pure_r] effective SLIDE workers: %d\n", opts$threads))

# Each real foreach task gets a reproducible stream; no installed namespace is changed.
dorng_seed <- Sys.getenv("SLIDE_R_DORNG_SEED", unset = "")
if (nzchar(dorng_seed) && !is.na(opts[["base-seed"]]) &&
    !identical(as.integer(dorng_seed), opts[["base-seed"]])) {
  stop("SLIDE_R_DORNG_SEED conflicts with --base-seed")
}
if (nzchar(dorng_seed) || !is.na(opts[["base-seed"]])) {
  if (nzchar(dorng_seed)) set.seed(as.integer(dorng_seed))
  requireNamespace("doRNG", quietly = FALSE)
  slide_env$`%dopar%` <- doRNG::`%dorng%`
  cat("[pure_r] deterministic doRNG streams for every foreach task\n")
}

out_path <- normalizePath(opts$`out-dir`, mustWork = FALSE)
dir.create(out_path, recursive = TRUE, showWarnings = FALSE)

input_params <- list(
  x_path          = normalizePath(opts$`x-path`),
  y_path          = normalizePath(opts$`y-path`),
  delta           = opts$delta,
  lambda          = opts$lambda,
  spec            = opts$spec,
  fdr             = opts$fdr,
  thresh_fdr      = opts$`thresh-fdr`,
  SLIDE_iter      = opts$niter,
  SLIDE_top_feats = opts$`slide-top-feats`,
  sampleCV_iter   = opts[["sample-cv-iterations"]],
  sampleCV_K      = opts[["sample-cv-k"]],
  ncore           = opts$threads,
  do_interacts    = TRUE,
  pure_homo       = TRUE,
  out_path        = out_path,
  y_factor        = opts[["y-factor"]],
  # Set eval_type explicitly to sidestep a SLIDE bug where the auto-detect
  # branch references `y` before it is loaded from disk.
  eval_type       = if (opts[["y-factor"]]) "auc" else "corr"
)

# Instrument local entry points without changing model construction or RNG draws.
diagnostics <- new.env(parent = emptyenv())
diagnostics$timings <- list()
diagnostics$selection_calls <- 0L
diagnostics$selection_errors <- 0L
diagnostics$sample_metric <- NA_real_
diagnostics$sample_rows <- list()
timed <- function(implementation, stage) {
  force(implementation); force(stage)
  function(...) {
    start <- proc.time()[["elapsed"]]
    on.exit({
      diagnostics$timings[[length(diagnostics$timings) + 1L]] <- data.frame(
        stage = stage, elapsed_s = proc.time()[["elapsed"]] - start)
    })
    implementation(...)
  }
}
worker_env$getLatentFactors <- timed(get("getLatentFactors", worker_env), "LOVE")
worker_env$SLIDE <- local({
  implementation <- worker_env$SLIDE
  timed(function(...) {
    diagnostics$selection_calls <- diagnostics$selection_calls + 1L
    tryCatch(implementation(...), error = function(e) {
      diagnostics$selection_errors <- diagnostics$selection_errors + 1L
      stop(e)
    })
  }, "selection")
})
worker_env$sampleCV <- local({
  implementation <- get("sampleCV", worker_env)
  function(...) {
    capture <- new.env(parent = worker_env)
    splits <- list()
    last_train <- integer()
    capture$sample <- function(x, size, ...) {
      indices <- base::sample(x, size, ...)
      caller <- parent.frame()
      diagnostics$sample_settings <- list(k = caller$sampleCV_K, iterations = caller$sampleCV_iter,
                                         n = nrow(caller$z_matrix))
      if (exists("iter", caller, inherits = FALSE)) {
        holdout <- setdiff(seq_len(nrow(caller$z_matrix)), indices)
        splits[[length(splits) + 1L]] <<- data.frame(
          iteration = caller$iter - 1L, row_index = holdout - 1L,
          sample_id = rownames(caller$z_matrix)[holdout])
        last_train <<- indices
      }
      indices
    }
    environment(implementation) <- capture
    value <- timed(implementation, "sampleCV")(...)
    diagnostics$sample_metric <- suppressWarnings(as.numeric(value))
    diagnostics$sample_rows <- splits
    diagnostics$last_train <- last_train
    value
  }
})
fmt_py <- function(x) {
  if (x == floor(x)) sprintf("%.1f", x) else format(x, trim = TRUE, scientific = FALSE)
}
combo_dir <- file.path(out_path, sprintf("%s_%s_out", fmt_py(opts$delta), fmt_py(opts$lambda)))
write_metadata <- function() {
  revision <- function(path, frozen = FALSE) {
    revision_file <- file.path(path, "REVISION")
    if (frozen && file.exists(revision_file)) return(trimws(readLines(revision_file, n = 1L)))
    result <- system2("git", c("-C", shQuote(path), "rev-parse", "HEAD"), stdout = TRUE, stderr = FALSE)
    if (!is.null(attr(result, "status"))) NA_character_ else result[[1]]
  }
  packages <- sort(loadedNamespaces())
  jsonlite::write_json(list(
    source = slide_source, source_path = normalizePath(source_dir),
    source_revision = if (slide_source == "installed") NA_character_ else revision(dirname(source_dir), frozen = TRUE),
    checkout_revision = revision(repo_dir),
    source_dirty = if (slide_source == "installed" || file.exists(file.path(dirname(source_dir), "REVISION"))) NA_character_ else
      system2("git", c("-C", shQuote(dirname(source_dir)), "status", "--porcelain", "--", "R"), stdout = TRUE),
    SLIDE_package_path = find.package("SLIDE"),
    packages = setNames(lapply(packages, function(p) as.character(utils::packageVersion(p))), packages),
    seed = opts[["base-seed"]], dorng_seed_override = dorng_seed, rng = RNGkind(),
    foreach_rng = if (nzchar(dorng_seed) || !is.na(opts[["base-seed"]])) "doRNG task streams" else "unseeded",
    workflow = opts$workflow, workers = opts$threads,
    sample_cv_iterations = opts[["sample-cv-iterations"]], sample_cv_k = opts[["sample-cv-k"]],
    actual_sample_cv_settings = diagnostics$sample_settings,
    selection_calls = diagnostics$selection_calls, selection_errors = diagnostics$selection_errors
  ), file.path(combo_dir, "benchmark_provenance.json"), auto_unbox = TRUE, pretty = TRUE, na = "null")
  if (length(diagnostics$timings)) utils::write.csv(do.call(rbind, diagnostics$timings),
    file.path(combo_dir, "benchmark_stage_timings.csv"), row.names = FALSE)
}

if (opts$workflow == "cv") {
  dir.create(combo_dir, recursive = TRUE, showWarnings = FALSE)
  x <- as.matrix(utils::read.csv(opts[["x-path"]], row.names = 1, check.names = FALSE))
  y <- as.matrix(utils::read.csv(opts[["y-path"]], row.names = 1, check.names = FALSE))
  z <- as.matrix(utils::read.csv(opts[["z-path"]], row.names = 1, check.names = FALSE))
  if (!identical(rownames(x), rownames(y)) || !identical(rownames(x), rownames(z))) {
    stop("CV x, y and saved Z sample IDs/order must match")
  }
  colnames(z) <- paste0("Z", seq_len(ncol(z)))
  z <- scale(z, TRUE, TRUE)
  f_size <- get("calcDefaultFsize", worker_env)(y, list(K = ncol(z)))
  cv_capture <- new.env(parent = worker_env)
  predictions <- list()
  cv_capture$runSLIDEBeta <- function(...) {
    caller <- parent.frame()
    rows <- data.frame(repeat_id = caller$rep, fold = caller$i, method = caller$method_j,
      sample_id = rownames(x)[caller$valid_ind],
      truth = if (opts[["y-factor"]]) as.numeric(as.character(caller$valid_y_labs)) else as.numeric(caller$valid_y),
      prediction = NA_real_, status = "failed", error = "")
    result <- tryCatch(worker_env$runSLIDEBeta(...), error = function(e) {
      rows$error <<- conditionMessage(e)
      predictions[[length(predictions) + 1L]] <<- rows
      stop(e)
    })
    rows$prediction <- as.numeric(result$valid_pred)
    rows$status <- ifelse(is.finite(rows$prediction), "ok", "nonfinite_prediction")
    predictions[[length(predictions) + 1L]] <<- rows
    result
  }
  bench_cv <- worker_env$benchCV2
  environment(bench_cv) <- cv_capture
  scores <- list()
  for (rep in seq_len(opts[["cv-repeats"]])) {
    error <- ""
    result <- tryCatch(timed(bench_cv, "finalCV")(
      k = opts[["cv-folds"]], x = x, y = y, z = z,
      std_y = TRUE, std_cv = TRUE, delta = opts$delta, lambda = opts$lambda,
      permute = TRUE, eval_type = input_params$eval_type,
      y_levels = if (opts[["y-factor"]]) sort(unique(as.numeric(y))) else NULL,
      out_path = paste0(combo_dir, "/SLIDE_CV_replicate_data/"),
      rep_cv = 50, alpha_level = 0.05, thresh_fdr = opts[["thresh-fdr"]],
      rep = rep, benchmark = FALSE, niter = opts$niter, spec = opts$spec,
      fdr = opts$fdr, f_size = f_size, parallel = TRUE, ncore = opts$threads,
      method = 4, do_interacts = TRUE), error = function(e) {
        error <<- conditionMessage(e); NULL
      })
    for (method in c("SLIDE", "SLIDE_y")) {
      metric <- if (is.null(result)) NA_real_ else as.numeric(result[result$method == method, 2])
      raw_auc <- NA_real_
      imputed_count <- 0L
      if (opts[["y-factor"]] && !is.null(result)) {
        p <- do.call(rbind, predictions)
        p <- p[p$repeat_id == rep & p$method == method, ]
        values <- p$prediction
        imputed_count <- sum(is.na(values))
        values[is.na(values)] <- median(values, na.rm = TRUE)
        raw_auc <- ROCR::performance(ROCR::prediction(values, p$truth), "auc")@y.values[[1]]
      }
      scores[[length(scores) + 1L]] <- data.frame(repeat_id = rep, method = method,
        metric = input_params$eval_type, score = metric, raw_auc = raw_auc,
        mse = if (!is.null(result$mse)) as.numeric(result$mse[result$method == method]) else NA_real_,
        imputed_count = imputed_count,
        flipped_auc = if (is.finite(raw_auc)) max(raw_auc, 1 - raw_auc) else NA_real_,
        status = if (nzchar(error)) "failed" else if (is.finite(metric)) "ok" else "nonfinite_metric",
        error = error, f_size = f_size)
    }
    # A failed repeat keeps explicit missing sample/method rows rather than disappearing.
    actual <- if (length(predictions)) do.call(rbind, predictions) else NULL
    for (method in c("SLIDE", "SLIDE_y")) {
      seen <- if (is.null(actual)) character() else actual$sample_id[actual$repeat_id == rep & actual$method == method]
      missing <- which(!rownames(x) %in% seen)
      if (length(missing)) predictions[[length(predictions) + 1L]] <- data.frame(
        repeat_id = rep, fold = NA_integer_, method = method, sample_id = rownames(x)[missing],
        truth = as.numeric(y[missing, ]), prediction = NA_real_, status = "not_run", error = error)
    }
    utils::write.csv(do.call(rbind, scores), file.path(combo_dir, "finalCV_repeat_scores.csv"), row.names = FALSE)
    utils::write.csv(do.call(rbind, predictions), file.path(combo_dir, "finalCV_predictions.csv"), row.names = FALSE)
    write_metadata()
  }
  quit(save = "no", status = if (any(do.call(rbind, scores)$status == "failed")) 1L else 0L)
}

cat(sprintf("[pure_r] delta=%g lambda=%g niter=%d out=%s\n",
            opts$delta, opts$lambda, opts$niter, out_path))
fullfit_error <- ""
t0 <- Sys.time()
tryCatch(optimize_slide(input_params, sink_file = FALSE), error = function(e) {
  fullfit_error <<- conditionMessage(e)
  warning("full-fit failed: ", fullfit_error, immediate. = TRUE)
})
elapsed <- as.numeric(difftime(Sys.time(), t0, units = "secs"))
cat(sprintf("[pure_r] optimizeSLIDE elapsed: %.2fs\n", elapsed))

# SLIDE names the per-combo dir as paste0(d, "_", l, "_out") with R's default
# numeric formatting (so 1.0 -> "1"). Python's benchmark loader expects the
# exact tokens that python's f"{delta}_{lam}_out" produces (so 1.0 -> "1.0").
# Rename the SLIDE-created dir to the Python-expected dir name if needed.
expected_dir <- file.path(out_path, sprintf("%s_%s_out", fmt_py(opts$delta),
                                            fmt_py(opts$lambda)))
slide_dir    <- file.path(out_path, sprintf("%s_%s_out",
                                            format(opts$delta, trim = TRUE),
                                            format(opts$lambda, trim = TRUE)))
if (!identical(slide_dir, expected_dir) && dir.exists(slide_dir) &&
    !dir.exists(expected_dir)) {
  file.rename(slide_dir, expected_dir)
}
combo_dir <- expected_dir
dir.create(combo_dir, recursive = TRUE, showWarnings = FALSE)

slide_rds <- file.path(combo_dir, "SLIDE_LFs.rds")
lf_rds    <- file.path(combo_dir, "AllLatentFactors.rds")

slide_res <- if (file.exists(slide_rds)) readRDS(slide_rds) else NULL
inner <- if (!is.null(slide_res$SLIDE_res)) slide_res$SLIDE_res else slide_res
clean_selection <- function(v) {
  values <- as.character(v)
  values[!is.na(values) & nzchar(values) & values != "NA"]
}
marginal_vars <- clean_selection(inner$marginal_vars)
interaction_vars <- clean_selection(inner$interaction_vars)

# SLIDE post-processing in getTopFeatures coerces marginal_vars from "Z26" to
# numeric 26. Re-prefix with "Z" AND shift to 0-indexed to match the
# python/rust backends, which name columns Z0..Z(K-1). SLIDE uses Z1..ZK.
shift_to_0idx <- function(v) {
  if (length(v) == 0) return(character(0))
  s <- as.character(v)
  # Strip any leading "Z"/"z" prefix, treat what remains as integer index, -1.
  nums <- as.integer(sub("^[Zz]", "", s))
  paste0("Z", nums - 1L)
}

shift_interacts_to_0idx <- function(v) {
  if (length(v) == 0) return(character(0))
  vapply(as.character(v), function(s) {
    parts <- strsplit(s, ".", fixed = TRUE)[[1]]
    shifted <- vapply(parts, function(p) {
      n <- as.integer(sub("^[Zz]", "", p))
      paste0("Z", n - 1L)
    }, character(1))
    paste(shifted, collapse = ".")
  }, character(1), USE.NAMES = FALSE)
}

writeLines(shift_to_0idx(marginal_vars),
           file.path(combo_dir, "sig_LFs.txt"))
writeLines(shift_interacts_to_0idx(interaction_vars),
           file.path(combo_dir, "sig_interacts.txt"))

# A cell-local summary is safe when a SLURM array runs multiple cells in the
# same backend directory. SLIDE's shared summary_table.csv is not.
z_path <- file.path(combo_dir, "z_matrix.csv")
zdf <- if (file.exists(z_path)) {
  utils::read.csv(z_path, row.names = 1, check.names = FALSE)
} else {
  NULL
}
cell_summary <- data.frame(
  delta = opts$delta,
  lambda = opts$lambda,
  workers = opts$threads,
  Num_of_LFs = if (!is.null(zdf)) ncol(zdf) else NA_integer_,
  Num_of_Sig_LFs = length(marginal_vars),
  Num_of_Interactors = length(interaction_vars),
  f_size = if (!is.null(slide_res$SLIDE_param)) unname(slide_res$SLIDE_param["f_size"]) else NA_real_,
  sampleCV_Performance = diagnostics$sample_metric,
  sampleCV_Provenance = "pooled_holdout_last_split_ols",
  sampleCV_Status = if (nzchar(fullfit_error) || is.null(slide_res)) "failed" else
    if (!length(marginal_vars)) "no_marginals" else
    if (is.finite(diagnostics$sample_metric)) "ok" else
    if (!length(diagnostics$sample_rows)) "model_too_large" else "missing_metric",
  error = fullfit_error,
  selection_calls = diagnostics$selection_calls,
  selection_errors = diagnostics$selection_errors
)
utils::write.csv(cell_summary, file.path(combo_dir, "benchmark_summary.csv"),
                 row.names = FALSE)

# Rewrite z_matrix.csv with 0-indexed column names so the loader's z-corr
# comparison aligns column-for-column with the python/rust z_matrix.csv.
if (!is.null(zdf)) {
  cn <- colnames(zdf)
  if (all(grepl("^[Zz][0-9]+$", cn))) {
    colnames(zdf) <- paste0("Z", as.integer(sub("^[Zz]", "", cn)) - 1L)
    utils::write.csv(zdf, z_path, row.names = TRUE)
  }
}

# z_matrix.csv and A.csv are already written by SLIDE::optimizeSLIDE into
# combo_dir; no need to recompute or rewrite them here.

cat(sprintf("[pure_r] wrote sig_LFs (%d), sig_interacts (%d) to %s\n",
            length(marginal_vars), length(interaction_vars), combo_dir))

# The native RDS contains the last split's OLS model and pooled holdout values.
model_path <- file.path(combo_dir, "sampleCV_model.RDS")
model <- if (file.exists(model_path)) readRDS(model_path) else NULL
predictions <- data.frame(iteration = integer(), row_index = integer(), sample_id = character(),
                          y_true = numeric(), y_pred = numeric())
if (!is.null(model)) {
  predictions <- do.call(rbind, diagnostics$sample_rows)
  if (is.null(predictions) || nrow(predictions) != length(model$samplesCV$pred_y)) {
    stop("actual sampleCV holdout indices do not match the saved pooled predictions")
  }
  predictions$y_true <- as.numeric(model$samplesCV$true_y)
  predictions$y_pred <- as.numeric(model$samplesCV$pred_y)
  y_original <- utils::read.csv(opts[["y-path"]], row.names = 1)
  expected_truth <- y_original[, 1]
  if (opts[["y-factor"]]) expected_truth <- readRDS(file.path(out_path, "binary_y_mapping.rds"))$cont_y
  if (!isTRUE(all.equal(predictions$y_true, as.numeric(expected_truth[match(predictions$sample_id, rownames(y_original))])))) {
    stop("sampleCV holdout truth does not match original sample IDs")
  }
}
utils::write.csv(predictions, file.path(combo_dir, "sampleCV_predictions.csv"), row.names = FALSE)
jsonlite::write_json(list(metric = diagnostics$sample_metric,
  status = cell_summary$sampleCV_Status, guard = cell_summary$sampleCV_Status,
  settings = list(condition = input_params$eval_type, k = diagnostics$sample_settings$k,
    iterations = diagnostics$sample_settings$iterations, seed = opts[["base-seed"]],
    model = "last_split_ols", score_provenance = "pooled_holdout"),
  term_order = if (!is.null(model)) names(coef(model)) else character(),
  last_coefficients = if (!is.null(model)) unname(coef(model)) else NULL,
  last_rank = if (!is.null(model)) model$rank else NULL,
  last_train_indices = if (!is.null(model)) diagnostics$last_train - 1L else integer(),
  last_train_ids = if (!is.null(model)) rownames(y_original)[diagnostics$last_train] else character(),
  prediction_ids_provenance = "captured_actual_sample_indices_no_rng_replay"),
  file.path(combo_dir, "sampleCV_model.json"), auto_unbox = TRUE, pretty = TRUE, na = "null")
write_metadata()

if (cell_summary$sampleCV_Status == "failed") quit(save = "no", status = 1L)
