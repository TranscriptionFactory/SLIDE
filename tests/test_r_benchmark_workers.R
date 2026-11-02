# Run from the repository root with TMPDIR pointing into project .tmpfiles/.
stopifnot(nzchar(Sys.getenv("TMPDIR")), !startsWith(normalizePath(tempdir()), "/tmp/"))
runner <- normalizePath("benchmarks/scripts/run_slide_r.R")
expressions <- parse(runner)
# Evaluate runner setup only, without reading data or executing a benchmark.
starts <- which(vapply(expressions, function(x) grepl("^suppressPackageStartupMessages", deparse(x)[1]), logical(1)))
ends <- which(vapply(expressions, function(x) grepl("^out_path <-", deparse(x)[1]), logical(1))) - 1L
stopifnot(length(starts) == 1L, length(ends) == 1L, starts < ends)
originals <- lapply(c("optimizeSLIDE", "runSLIDE", "SLIDE"), function(n) get(n, asNamespace("SLIDE")))
sources <- c("installed", "checkout")
if (dir.exists("vendor/SLIDE-jishnu/R")) {
  sources <- c(sources, "vendor")
  Sys.setenv(SLIDE_VENDOR_DIR = normalizePath("vendor/SLIDE-jishnu/R"))
} else {
  message("SKIP: optional vendor source is not installed")
}
Sys.setenv(SLIDE_OPT_ITER = "1")
for (source in sources) {
  for (optimized in if (source == "checkout") "0" else c("0", "1")) {
    for (workers in c(1L, 4L, 10L)) {
      Sys.setenv(SLIDE_R_SOURCE = source, SLIDE_R_OPTIMIZE = optimized)
      e <- new.env(parent = globalenv())
      e$opts <- list(threads = workers, `base-seed` = 42L)
      e$commandArgs <- function(trailingOnly = FALSE) paste0("--file=", runner)
      for (i in starts:ends) eval(expressions[[i]], e)
      original <- get("SLIDE", envir = parent.env(e$worker_env))
      wrapper_env <- environment(e$worker_env$SLIDE)
      stopifnot(identical(wrapper_env$implementation, original))
      # Observe real runSLIDE dispatch without running stochastic knockoff fits.
      observed <- NULL
      wrapper_env$implementation <- function(..., ncore) {
        observed <<- ncore
        list(marginal_vars = character(), interaction_vars = character())
      }
      e$worker_env$runSLIDE(y = matrix(1:6), z_matrix = matrix(1:12, 6),
                            all_latent_factors = list(), f_size = 2)
      stopifnot(identical(observed, workers), identical(environment(e$optimize_slide), e$worker_env))
      stopifnot(identical(environment(e$worker_env$benchCV2), e$worker_env),
                identical(environment(e$worker_env$runSLIDEBeta), e$worker_env),
                identical(e$slide_env$`%dopar%`, doRNG::`%dorng%`))
      if (source == "checkout") stopifnot(identical(e$source_dir, normalizePath("R")))
      message("PASS: ", source, ", optimized=", optimized, ", workers=", workers)
    }
  }
}
stopifnot(identical(originals, lapply(c("optimizeSLIDE", "runSLIDE", "SLIDE"), function(n) get(n, asNamespace("SLIDE")))))
message("PASS: installed namespace unchanged")

# Checkout forbids an optimized replacement in the reference arm.
Sys.setenv(SLIDE_R_SOURCE = "checkout", SLIDE_R_OPTIMIZE = "1")
e <- new.env(parent = globalenv())
e$opts <- list(threads = 2L, `base-seed` = 42L)
e$commandArgs <- function(trailingOnly = FALSE) paste0("--file=", runner)
error <- tryCatch({for (i in starts:ends) eval(expressions[[i]], e); NULL}, error = identity)
stopifnot(inherits(error, "error"), grepl("requires SLIDE_R_OPTIMIZE=0", conditionMessage(error)))
Sys.setenv(SLIDE_R_OPTIMIZE = "0")
for (i in starts:ends) eval(expressions[[i]], e)

# Real PSOCK execution proves task streams do not depend on worker scheduling.
draws <- function(workers) {
  cluster <- parallel::makePSOCKcluster(workers)
  on.exit(parallel::stopCluster(cluster))
  doParallel::registerDoParallel(cluster)
  set.seed(101)
  fun <- function() foreach::foreach(i = 1:6) %dopar% stats::runif(3)
  environment(fun) <- e$slide_env
  fun()
}
stopifnot(identical(draws(1L), draws(2L)),
          grepl("%dopar%", paste(deparse(e$slide_env$secondKO), collapse = ""), fixed = TRUE))
message("PASS: actual PSOCK doRNG task streams")

# Evaluate only instrumentation; compare real sampleCV with and without capture.
e$out_path <- "."
e$opts$delta <- 0.1
e$opts$lambda <- 0.1
index <- function(prefix) which(vapply(expressions, function(x) startsWith(deparse(x)[1], prefix), logical(1)))
for (i in index("diagnostics <-"):index("write_metadata <-")) eval(expressions[[i]], e)
scratch <- file.path(".tmpfiles/2026-11-01_ssc-cd4-benchmarks", "r-adapter-tests")
dir.create(scratch, recursive = TRUE, showWarnings = FALSE)
set.seed(10)
z <- matrix(rnorm(72), 24, 3, dimnames = list(paste0("sample_", 1:24), paste0("Z", 1:3)))
y <- matrix(z[, 1] + rnorm(24), 24, 1, dimnames = list(rownames(z), "y"))
selected <- list(marginal_vals = 1:2, interaction = data.frame(p1 = numeric(), p2 = numeric()),
                 SLIDE_res = list(interaction_vars = character()))
for (subdir in c("raw", "captured")) dir.create(file.path(scratch, subdir), showWarnings = FALSE)
set.seed(91)
raw <- e$slide_env$sampleCV(y, z, selected, sampleCV_K = 4, condition = "corr", sampleCV_iter = 3,
                           out_path = paste0(scratch, "/raw/"))
raw_seed <- .Random.seed
set.seed(91)
captured <- e$worker_env$sampleCV(y, z, selected, sampleCV_K = 4, condition = "corr", sampleCV_iter = 3,
                                 out_path = paste0(scratch, "/captured/"))
stopifnot(identical(raw, captured), identical(raw_seed, .Random.seed))
model <- readRDS(file.path(scratch, "captured/sampleCV_model.RDS"))
rows <- do.call(rbind, e$diagnostics$sample_rows)
stopifnot(nrow(rows) == length(model$samplesCV$pred_y),
          identical(as.numeric(y[rows$sample_id, 1]), as.numeric(model$samplesCV$true_y)),
          identical(rownames(z)[rows$row_index + 1L], rows$sample_id),
          identical(e$diagnostics$timings[[1]]$stage, "sampleCV"))
message("PASS: real sampleCV capture preserves predictions and RNG state")

# Use the actual fold/scoring engine with a cheap deterministic predictor.
utils::write.csv(z, file.path(scratch, "x.csv"))
utils::write.csv(y, file.path(scratch, "y.csv"))
utils::write.csv(z, file.path(scratch, "z.csv"))
e$opts <- list(workflow = "cv", `x-path` = file.path(scratch, "x.csv"),
  `y-path` = file.path(scratch, "y.csv"), `z-path` = file.path(scratch, "z.csv"),
  `cv-repeats` = 2L, `cv-folds` = 3L, `sample-cv-iterations` = 3L, `sample-cv-k` = 4L,
  `thresh-fdr` = 0.2, `y-factor` = FALSE, delta = 0.1, lambda = 0.1, niter = 1L,
  spec = 0.1, fdr = 0.1, threads = 2L, `base-seed` = 42L)
e$combo_dir <- file.path(scratch, "cv")
e$input_params <- list(eval_type = "corr")
e$worker_env$runSLIDEBeta <- function(train_y, train_z, valid_z, ncore, ...) {
  stopifnot(ncore == 2L)
  list(valid_pred = as.numeric(valid_z[, 1]))
}
e$quit <- function(...) stop("test_cv_complete")
set.seed(42)
error <- tryCatch(eval(expressions[[index('if (opts$workflow == "cv") {')]], e), error = identity)
stopifnot(inherits(error, "error"), identical(conditionMessage(error), "test_cv_complete"))
p <- utils::read.csv(file.path(e$combo_dir, "finalCV_predictions.csv"))
scores <- utils::read.csv(file.path(e$combo_dir, "finalCV_repeat_scores.csv"))
stopifnot(nrow(p) == 2L * 2L * nrow(z), nrow(scores) == 4L, all(scores$status == "ok"),
          all(p$status == "ok"), !anyDuplicated(p[c("repeat_id", "method", "sample_id")]),
          all(table(p$repeat_id, p$method) == nrow(z)))
provenance <- jsonlite::read_json(file.path(e$combo_dir, "benchmark_provenance.json"))
stopifnot(provenance$source == "checkout", provenance$source_path == normalizePath("R"),
          nzchar(provenance$source_revision), provenance$workers == 2L,
          nzchar(provenance$packages$SLIDE))
message("PASS: actual benchCV2 fold identity, scores, provenance and exports")

# A failed repeat must retain every sample/method and explicit missing scores.
e$combo_dir <- file.path(scratch, "cv-failure")
e$worker_env$runSLIDEBeta <- function(...) stop("expected model failure")
error <- tryCatch(eval(expressions[[index('if (opts$workflow == "cv") {')]], e), error = identity)
stopifnot(identical(conditionMessage(error), "test_cv_complete"))
p <- utils::read.csv(file.path(e$combo_dir, "finalCV_predictions.csv"))
scores <- utils::read.csv(file.path(e$combo_dir, "finalCV_repeat_scores.csv"))
stopifnot(nrow(p) == 2L * 2L * nrow(z), all(scores$status == "failed"), all(is.na(scores$score)),
          all(p$status %in% c("failed", "not_run")))
message("PASS: failed CV repeats retain missing sample rows and scores")

# Frozen snapshots identify executed code with REVISION, not their enclosing Git tree.
snapshot <- file.path(scratch, "snapshot")
dir.create(file.path(snapshot, "R"), recursive = TRUE, showWarnings = FALSE)
writeLines("frozen-reference-revision", file.path(snapshot, "REVISION"))
source_dir <- e$source_dir
e$source_dir <- normalizePath(file.path(snapshot, "R"))
e$write_metadata()
provenance <- jsonlite::read_json(file.path(e$combo_dir, "benchmark_provenance.json"))
stopifnot(provenance$source_revision == "frozen-reference-revision")
e$source_dir <- source_dir
message("PASS: frozen REVISION takes precedence for executed source provenance")

# Export a real sampleCV model through the full-fit tail without fitting LOVE/knockoffs.
e$out_path <- file.path(scratch, "full-fit")
e$opts$workflow <- "full-fit"
e$opts$lambda <- 1
native <- file.path(e$out_path, "0.1_1_out")
dir.create(native, recursive = TRUE, showWarnings = FALSE)
selected$SLIDE_res$marginal_vars <- 1:2
selected$SLIDE_param <- c(f_size = 3)
saveRDS(selected, file.path(native, "SLIDE_LFs.rds"))
utils::write.csv(z, file.path(native, "z_matrix.csv"))
file.copy(file.path(scratch, "captured/sampleCV_model.RDS"), native, overwrite = TRUE)
e$fullfit_error <- ""
for (i in index("expected_dir <-"):length(expressions)) eval(expressions[[i]], e)
summary <- utils::read.csv(file.path(e$combo_dir, "benchmark_summary.csv"))
p <- utils::read.csv(file.path(e$combo_dir, "sampleCV_predictions.csv"))
m <- jsonlite::read_json(file.path(e$combo_dir, "sampleCV_model.json"))
stopifnot(basename(e$combo_dir) == "0.1_1.0_out", summary$sampleCV_Status == "ok",
          isTRUE(all.equal(summary$sampleCV_Performance, raw)), nrow(p) == 18L,
          length(m$last_coefficients) == 3L, m$settings$model == "last_split_ols",
          isTRUE(all.equal(as.numeric(y[p$sample_id, 1]), p$y_true)))
message("PASS: full-fit real sampleCV model, summary, IDs and native directory rename")

# A completed but unscorable screening fit is not a process failure.
e$diagnostics$sample_metric <- NA_real_
eval(expressions[[index("cell_summary <-")]], e)
stopifnot(e$cell_summary$sampleCV_Status == "missing_metric")
eval(expressions[[length(expressions)]], e)
message("PASS: completed undefined sampleCV remains missing without a failed exit")
