# Run from the repository root with TMPDIR pointing into project .tmpfiles/.
stopifnot(nzchar(Sys.getenv("TMPDIR")), !startsWith(normalizePath(tempdir()), "/tmp/"))
e <- new.env(parent = globalenv())
sys.source("R/runSLIDE.R", e)
sys.source("R/optimizeSLIDE.R", e)
calls <- numeric()
failures <- 0L
e$calcDefaultFsize <- function(...) 2
e$SLIDE <- function(..., f_size) {
  calls <<- c(calls, f_size)
  if (length(calls) <= failures) stop("injected fit failure")
  list(marginal_vars = "Z1", interaction_vars = "Z1.Z2")
}
fit <- function() e$runSLIDE(y = matrix(c(0, 1, 0, 1)),
                            z_matrix = matrix(1:8, 4),
                            all_latent_factors = list(K = 2), f_size = 3)
result <- fit()
stopifnot(identical(calls, 3), result$SLIDE_param[["f_size"]] == 3,
          identical(result$marginal_vals, 1), result$interaction$p2 == 2)
calls <- numeric()
failures <- 1L
result <- suppressWarnings(fit())
stopifnot(identical(calls, c(3, 2)), result$SLIDE_param[["f_size"]] == 2,
          identical(result$marginal_vals, 1), result$interaction$p2 == 2)
calls <- numeric()
failures <- 2L
error <- tryCatch(suppressWarnings(fit()), error = identity)
stopifnot(inherits(error, "error"), conditionMessage(error) == "injected fit failure",
          identical(calls, c(3, 2)))
message("PASS: success, default-size retry and metadata, terminal error propagation")

# Exercise the real grid loop and file outputs without stochastic factor fitting.
e$getLatentFactors <- function(...) list(K = 2)
e$calcZMatrix <- function(x_std, ...) x_std
attempt <- 0L
e$runSLIDE <- function(...) {
  attempt <<- attempt + 1L
  if (attempt == 1L) stop("injected grid failure")
  list(SLIDE_res = list(marginal_vars = character()), SLIDE_param = c(f_size = 2))
}
out <- tempfile("fit-failures-", tmpdir = tempdir())
dir.create(out)
write.csv(matrix(1:8, 4, dimnames = list(NULL, c("a", "b"))), file.path(out, "x.csv"))
write.csv(matrix(c(0, 1, 0, 1)), file.path(out, "y.csv"))
params <- list(x_path = file.path(out, "x.csv"), y_path = file.path(out, "y.csv"),
               out_path = out, delta = c(0.1, 0.2), lambda = 1,
               eval_type = "auc", y_factor = FALSE)
warnings <- character()
summary <- withCallingHandlers(e$optimizeSLIDE(params), warning = function(w) {
  warnings <<- c(warnings, conditionMessage(w))
  invokeRestart("muffleWarning")
})
stopifnot(attempt == 2L, nrow(summary) == 2L,
          all(is.na(summary[1, c("f_size", "Num_of_Sig_LFs", "Num_of_Interactors", "sampleCV_Performance")])),
          summary$Num_of_LFs[1] == 2, as.numeric(summary$f_size[2]) == 2,
          any(grepl("injected grid failure", warnings)),
          !file.exists(file.path(out, "0.1_1_out/SLIDE_LFs.rds")),
          file.exists(file.path(out, "0.2_1_out/SLIDE_LFs.rds")),
          nrow(read.csv(file.path(out, "summary_table.csv"))) == 2L)
message("PASS: failed combination retained as NA; next combination and summary saved")
