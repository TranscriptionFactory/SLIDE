suppressPackageStartupMessages({library(knockoff); library(Rdsdp)})
args <- commandArgs(trailingOnly = TRUE)
post_only <- length(args) > 0L && identical(args[[1]], "--postprocess")
if (post_only && length(args) != 4L) stop("usage: trace_r_asdp.R --postprocess <sigma-dir-or-file> <raw-vector.csv> <output-dir>")
if (!post_only && !(length(args) %in% 2:4)) stop("usage: trace_r_asdp.R <sigma-dir-or-file> <output-dir> [raw-vector.csv] [cached-vector.csv]")
offset <- if (post_only) 1L else 0L
sigma_input <- args[[1L + offset]]
out <- if (post_only) args[[4]] else args[[2]]
dir.create(out, recursive = TRUE, showWarnings = FALSE)
writeLines(c(
  paste("R", getRversion()),
  paste("knockoff", packageVersion("knockoff"), find.package("knockoff")),
  paste("Rdsdp", packageVersion("Rdsdp"), find.package("Rdsdp")),
  "",
  "--- knockoff:::create.solve_asdp ---", deparse(knockoff:::create.solve_asdp),
  "--- knockoff:::create.solve_sdp ---", deparse(knockoff:::create.solve_sdp),
  "--- knockoff:::is_posdef ---", deparse(knockoff:::is_posdef),
  "--- Rdsdp::dsdp wrapper ---", deparse(get("dsdp", asNamespace("Rdsdp"))),
  "--- sessionInfo ---", capture.output(sessionInfo())
), file.path(out, "r_provenance.txt"))
read_raw <- function(file, n) {
  con <- file(file, "rb"); on.exit(close(con))
  x <- readBin(con, "double", n = n, size = 8L, endian = "little")
  if (length(x) != n) stop("short read: ", file)
  x
}
write_vec <- function(path, x) write.table(format(x, digits = 17L, scientific = TRUE),
  path, row.names = FALSE, col.names = FALSE, quote = FALSE)
write_mat <- function(path, x) write.table(format(x, digits = 17L, scientific = TRUE),
  path, row.names = FALSE, col.names = FALSE, quote = FALSE, sep = ",")
if (dir.exists(sigma_input)) {
  raw_sigma <- file.path(sigma_input, "rust_sigma.raw")
  sigma_path <- if (file.exists(raw_sigma)) raw_sigma else file.path(sigma_input, "input_sigma.csv")
} else {
  sigma_path <- sigma_input
  raw_sigma <- file.path(dirname(sigma_path), "rust_sigma.raw")
}
if (grepl("\\.raw$", sigma_path)) {
  p <- as.integer(sqrt(file.info(sigma_path)$size / 8))
  Sigma <- matrix(read_raw(sigma_path, p * p), nrow = p, byrow = TRUE)
} else {
  Sigma <- as.matrix(read.csv(sigma_path, header = FALSE))
}
p <- nrow(Sigma)
dimnames(Sigma) <- NULL
stopifnot(ncol(Sigma) == p, isSymmetric(Sigma), all(diag(Sigma) > 0))
G <- cov2cor(Sigma)
write_mat(file.path(out, "sigma.csv"), Sigma)
write_mat(file.path(out, "correlation.csv"), G)
postprocess_r <- function(y, G, Sigma) {
  s <- y
  s[s < 0] <- 0
  s[s > 1] <- 1
  psd <- 0
  s_eps <- 1e-08
  while ((psd == 0) & (s_eps <= 0.1)) {
    if (knockoff:::is_posdef(2 * G - diag(s * (1 - s_eps), length(s)), tol = 1e-09)) psd <- 1
    else s_eps <- s_eps * 10
  }
  repaired <- s * (1 - s_eps)
  repaired[repaired < 0] <- 0
  list(clipped = s, shrink = 1 - s_eps, repaired = repaired,
       final = repaired * diag(Sigma))
}
if (post_only) {
  post_input <- scan(args[[3]], quiet = TRUE)
  stopifnot(length(post_input) == p, all(is.finite(post_input)))
  stages <- postprocess_r(post_input, G, Sigma)
  write_vec(file.path(out, "input_raw.csv"), post_input)
  write_vec(file.path(out, "r_clipped.csv"), stages$clipped)
  write_vec(file.path(out, "r_repaired.csv"), stages$repaired)
  write_vec(file.path(out, "r_rescaled.csv"), stages$final)
  writeLines("Faithful extracted create.solve_sdp postprocessor; verified against full installed entrypoint in traced runs.",
    file.path(out, "postprocess_check.txt"))
  quit(status = 0)
}
# Bounded repeat control: two ordinary entrypoint calls, then the traced call below.
fresh1 <- knockoff::create.solve_asdp(Sigma)
fresh2 <- knockoff::create.solve_asdp(Sigma)
write_vec(file.path(out, "r_fresh1_final.csv"), fresh1)
write_vec(file.path(out, "r_fresh2_final.csv"), fresh2)
# Capture the installed Rdsdp call at the actual knockoff::create.solve_sdp boundary.
trace("dsdp", where = asNamespace("Rdsdp"), print = FALSE,
  tracer = quote({
    d <- Sys.getenv("R_SDP_CAPTURE_DIR")
    saveRDS(list(A = A, b = b, C = C, K = K, OPTIONS = OPTIONS), file.path(d, "Rdsdp_call.rds"))
    write_vec <- function(path, x) write.table(format(x, digits = 17L, scientific = TRUE),
      path, row.names = FALSE, col.names = FALSE, quote = FALSE)
    write_vec(file.path(d, "b.csv"), b)
    write_vec(file.path(d, "objective.csv"), C)
    # Matrix is sparse CSC in this API; persist exact coordinates and values.
    rows <- A@i
    columns <- rep(seq_len(length(A@p) - 1L) - 1L, diff(A@p))
    write.table(data.frame(row = rows, column = columns,
      value = format(A@x, digits = 17L, scientific = TRUE)),
      file.path(d, "constraints.csv"), sep = ",", row.names = FALSE, col.names = TRUE, quote = FALSE)
    writeLines(capture.output(str(K)), file.path(d, "cone.txt"))
    writeLines(capture.output(str(OPTIONS)), file.path(d, "options.txt"))
  }),
  exit = quote({
    d <- Sys.getenv("R_SDP_CAPTURE_DIR")
    sol <- returnValue()
    write.table(format(sol$y, digits = 17L, scientific = TRUE), file.path(d, "rdsdp_raw.csv"),
      row.names = FALSE, col.names = FALSE, quote = FALSE)
    writeLines(c(sol$STATS$stype, capture.output(str(sol$STATS))), file.path(d, "solver.txt"))
  }))
Sys.setenv(R_SDP_CAPTURE_DIR = normalizePath(out, mustWork = TRUE))
# Exercise the actual installed entrypoint, including its exact R postprocessing.
actual <- knockoff::create.solve_asdp(Sigma)
write_vec(file.path(out, "r_actual_final.csv"), actual)
raw <- scan(file.path(out, "rdsdp_raw.csv"), quiet = TRUE)
post_input <- if (length(args) >= 3L) scan(args[[3]], quiet = TRUE) else raw
stopifnot(length(post_input) == p, all(is.finite(post_input)))
full_stages <- postprocess_r(raw, G, Sigma)
stages <- postprocess_r(post_input, G, Sigma)
write_vec(file.path(out, "r_clipped.csv"), stages$clipped)
write_vec(file.path(out, "r_repaired.csv"), stages$repaired)
write_vec(file.path(out, "r_rescaled.csv"), stages$final)
write_vec(file.path(out, "input_raw.csv"), post_input)
untrace("dsdp", where = asNamespace("Rdsdp"))
stopifnot(isTRUE(all.equal(full_stages$final, actual, tolerance = 0)))
cached_path <- if (length(args) == 4L) args[[4]] else if (dir.exists(sigma_input)) file.path(sigma_input, "r_s.csv") else ""
cached_delta <- if (nzchar(cached_path) && file.exists(cached_path)) max(abs(actual - scan(cached_path, quiet = TRUE))) else NA_real_
writeLines(sprintf(paste0("postprocess_matches_installed_entrypoint=TRUE\n",
  "feasibility_shrink=%.17g\ninput_is_traced_R_raw=%s\n",
  "fresh_repeat_max_abs=%.17g\ntraced_vs_untraced_max_abs=%.17g\n",
  "traced_vs_cached_max_abs=%.17g"), stages$shrink,
  identical(post_input, raw), max(abs(fresh1 - fresh2)), max(abs(actual - fresh1)), cached_delta),
  file.path(out, "postprocess_check.txt"))
