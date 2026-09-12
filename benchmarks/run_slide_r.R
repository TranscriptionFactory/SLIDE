#!/usr/bin/env Rscript
# Pure-R SLIDE benchmark runner: one (delta, lambda) combo per invocation.
#
# Writes the same per-combo files the Python/R-knockoffs backends produce so
# benchmarks/benchmark_backends.py:load_python_result can pick them up:
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

for (k in c("x-path", "y-path", "out-dir", "delta", "lambda")) {
  if (is.null(opts[[k]])) stop("missing required arg --", k)
}

suppressPackageStartupMessages(library(SLIDE))
if (!is.na(opts[["base-seed"]])) set.seed(opts[["base-seed"]])

# SLIDE_R_SOURCE=vendor runs the repo's vendored SLIDE clone (vendor/SLIDE-jishnu/R)
# instead of the installed package, so the env-gated edits there -- notably
# SLIDE_OPT_ITER in selectShortFreq -- actually take effect. The vendored files are
# evaluated into an environment whose parent is the installed package's imports
# environment, so every unqualified call inside them resolves exactly as it does
# inside the package: vendored definition first, then the package's imports, then
# base. library(SLIDE) above stays -- it is what makes that imports environment (and
# the package's dependencies) available. Default "installed" is stock behaviour.
slide_source <- Sys.getenv("SLIDE_R_SOURCE", unset = "installed")
script_arg <- grep("^--file=", commandArgs(trailingOnly = FALSE), value = TRUE)
script_dir <- if (length(script_arg) > 0) {
  dirname(normalizePath(sub("^--file=", "", script_arg[[1]])))
} else {
  getwd()
}
if (identical(slide_source, "vendor")) {
  vendor_dir <- Sys.getenv("SLIDE_VENDOR_DIR", unset = "")
  if (!nzchar(vendor_dir)) {
    vendor_dir <- file.path(dirname(script_dir), "vendor", "SLIDE-jishnu", "R")
  }
  if (!dir.exists(vendor_dir)) {
    stop("SLIDE_R_SOURCE=vendor but no vendored R sources at: ", vendor_dir)
  }
  slide_env    <- new.env(parent = parent.env(asNamespace("SLIDE")))
  vendor_files <- sort(list.files(vendor_dir, pattern = "\\.[Rr]$", full.names = TRUE))
  for (vendor_file in vendor_files) sys.source(vendor_file, envir = slide_env)
  optimize_slide <- get("optimizeSLIDE", envir = slide_env)
  # vendor/ is gitignored, so a fresh checkout has the pristine upstream tree and
  # SLIDE_OPT_ITER=0 would silently produce stock findOptIter numbers. Fail loudly
  # instead; benchmarks/vendor_slide_opt_iter.patch reapplies the gate.
  if (identical(Sys.getenv("SLIDE_OPT_ITER", unset = "1"), "0") &&
      !any(grepl("SLIDE_OPT_ITER",
                 deparse(get("selectShortFreq", envir = slide_env)), fixed = TRUE))) {
    stop("SLIDE_OPT_ITER=0 requested but the vendored selectShortFreq does not honour ",
         "it. Apply benchmarks/vendor_slide_opt_iter.patch to ", vendor_dir, ".")
  }
  cat(sprintf("[pure_r] SLIDE_R_SOURCE=vendor: sourced %d files from %s (SLIDE_OPT_ITER=%s)\n",
              length(vendor_files), vendor_dir,
              Sys.getenv("SLIDE_OPT_ITER", unset = "1")))
} else {
  slide_env <- asNamespace("SLIDE")
  optimize_slide <- SLIDE::optimizeSLIDE
  cat("[pure_r] SLIDE_R_SOURCE=installed: using SLIDE::optimizeSLIDE\n")
}

# Preserve a stock comparison route without editing installed package files.
# Opt in with SLIDE_R_OPTIMIZE=1; copied runners then also need
# src/loveslide/slide_r/knockoffPerformance.R alongside benchmarks/.
optimize_interactions <- Sys.getenv("SLIDE_R_OPTIMIZE", unset = "0")
if (!optimize_interactions %in% c("0", "1")) stop("SLIDE_R_OPTIMIZE must be 0 or 1")
if (identical(optimize_interactions, "1")) {
  performance_env <- new.env(parent = globalenv())
  sys.source(file.path(dirname(script_dir), "src", "loveslide", "slide_r",
                       "knockoffPerformance.R"), envir = performance_env)
  optimized_env <- performance_env$withPreparedInteractionKnockoffs(slide_env)
  optimize_slide <- optimized_env$optimizeSLIDE
  cat("[pure_r] SLIDE_R_OPTIMIZE=1: cached Gaussian preparation and sparse interaction statistics\n")
} else {
  cat("[pure_r] SLIDE_R_OPTIMIZE=0: unchanged R interaction implementation\n")
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
  ncore           = opts$threads,
  do_interacts    = TRUE,
  pure_homo       = TRUE,
  out_path        = out_path,
  y_factor        = opts[["y-factor"]],
  # Set eval_type explicitly to sidestep a SLIDE bug where the auto-detect
  # branch references `y` before it is loaded from disk.
  eval_type       = if (opts[["y-factor"]]) "auc" else "corr"
)

cat(sprintf("[pure_r] delta=%g lambda=%g niter=%d out=%s\n",
            opts$delta, opts$lambda, opts$niter, out_path))

t0 <- Sys.time()
optimize_slide(input_params, sink_file = FALSE)
elapsed <- as.numeric(difftime(Sys.time(), t0, units = "secs"))
cat(sprintf("[pure_r] optimizeSLIDE elapsed: %.2fs\n", elapsed))

# SLIDE names the per-combo dir as paste0(d, "_", l, "_out") with R's default
# numeric formatting (so 1.0 -> "1"). Python's benchmark loader expects the
# exact tokens that python's f"{delta}_{lam}_out" produces (so 1.0 -> "1.0").
# Rename the SLIDE-created dir to the Python-expected dir name if needed.
fmt_py <- function(x) {
  # Mimic Python's default float-to-str: integer floats keep the ".0".
  if (x == floor(x)) sprintf("%.1f", x) else format(x, trim = TRUE,
                                                   scientific = FALSE)
}
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
if (!dir.exists(combo_dir)) {
  stop("SLIDE did not create the expected combo dir: ", combo_dir,
       " (tried ", slide_dir, ")")
}

slide_rds <- file.path(combo_dir, "SLIDE_LFs.rds")
lf_rds    <- file.path(combo_dir, "AllLatentFactors.rds")

if (!file.exists(slide_rds)) {
  stop("Expected SLIDE_LFs.rds not found in ", combo_dir)
}

slide_res <- readRDS(slide_rds)
inner     <- if (!is.null(slide_res$SLIDE_res)) slide_res$SLIDE_res else slide_res

marginal_vars    <- if (!is.null(inner$marginal_vars))    inner$marginal_vars    else character(0)
interaction_vars <- if (!is.null(inner$interaction_vars)) inner$interaction_vars else character(0)

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
  Num_of_LFs = if (!is.null(zdf)) ncol(zdf) else NA_integer_,
  Num_of_Sig_LFs = length(marginal_vars),
  Num_of_Interactors = length(interaction_vars)
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
