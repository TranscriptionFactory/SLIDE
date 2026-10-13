# usage: Rscript r_gaptol.R <scratch_dir> <out_dir>; DSDP gaptol sweep via knockoff:::create.solve_sdp on each sigma_*.csv
suppressMessages(library(knockoff))
a <- commandArgs(TRUE); d <- a[1]; out <- a[2]
rows <- list()
for (f in list.files(d, pattern = "^sigma_.*\\.csv$", full.names = TRUE)) {
  S <- unname(as.matrix(read.csv(f, header = FALSE)))
  tag <- sub("\\.csv$", "", sub("sigma_", "", basename(f)))
  for (gt in c(1e-6, 1e-8, 1e-10)) for (mi in c(1000, 10000)) {
    s <- tryCatch(knockoff:::create.solve_sdp(S, gaptol = gt, maxit = mi), error = function(e) rep(NA_real_, nrow(S)))
    writeLines(sprintf("%.17g", s), file.path(out, sprintf("sR_%s_gt%g_mi%d.txt", tag, gt, mi)))
  }
}
