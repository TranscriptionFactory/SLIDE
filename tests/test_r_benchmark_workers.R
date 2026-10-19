# Run from the repository root with TMPDIR pointing into project .tmpfiles/.
stopifnot(nzchar(Sys.getenv("TMPDIR")), !startsWith(normalizePath(tempdir()), "/tmp/"))
runner <- normalizePath("benchmarks/scripts/run_slide_r.R")
expressions <- parse(runner)
# Evaluate runner setup only, without reading data or executing a benchmark.
starts <- which(vapply(expressions, function(x) grepl("^suppressPackageStartupMessages", deparse(x)[1]), logical(1)))
ends <- which(vapply(expressions, function(x) grepl("^out_path <-", deparse(x)[1]), logical(1))) - 1L
stopifnot(length(starts) == 1L, length(ends) == 1L, starts < ends)
originals <- lapply(c("optimizeSLIDE", "runSLIDE", "SLIDE"), function(n) get(n, asNamespace("SLIDE")))
sources <- "installed"
if (dir.exists("vendor/SLIDE-jishnu/R")) {
  sources <- c(sources, "vendor")
  Sys.setenv(SLIDE_VENDOR_DIR = normalizePath("vendor/SLIDE-jishnu/R"))
} else {
  message("SKIP: optional vendor source is not installed")
}
Sys.setenv(SLIDE_OPT_ITER = "1")
for (source in sources) {
  for (optimized in c("0", "1")) {
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
      message("PASS: ", source, ", optimized=", optimized, ", workers=", workers)
    }
  }
}
stopifnot(identical(originals, lapply(c("optimizeSLIDE", "runSLIDE", "SLIDE"), function(n) get(n, asNamespace("SLIDE")))))
message("PASS: installed namespace unchanged")
