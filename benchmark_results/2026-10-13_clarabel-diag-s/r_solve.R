# usage: Rscript r_solve.R <scratch_dir> ; solves every sigma_*.csv with knockoff::create.solve_asdp (DSDP, gaptol 1e-6)
suppressMessages(library(knockoff))
d <- commandArgs(TRUE)[1]
for (f in list.files(d, pattern = "^sigma_.*\\.csv$", full.names = TRUE)) {
  S <- unname(as.matrix(read.csv(f, header = FALSE)))
  s <- knockoff:::create.solve_asdp(S)
  writeLines(sprintf("%.17g", as.numeric(s)), sub("sigma_", "sR_", f))
}
