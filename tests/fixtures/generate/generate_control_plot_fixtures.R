# Small lm/pROC/bw.nrd0 reference for calcControlPerformance.R, without random draws.
args <- commandArgs(trailingOnly = FALSE)
script <- sub("^--file=", "", args[grep("^--file=", args)])
root <- dirname(dirname(dirname(dirname(normalizePath(script)))))
out <- file.path(root, "tests", "fixtures", "control_plot")
source(file.path(root, "vendor", "SLIDE-jishnu", "R", "pairwiseInteractions.R"))
i <- seq_len(12)
z <- cbind(Z0 = sin(i), Z1 = cos(i * 0.7), Z2 = (i %% 5) - 2)
y <- z[, 1] + 0.7 * z[, 1] * z[, 3] + sin(i * 2) / 4
ints <- pairwiseInteractions(c(1, 2), z)$interaction[, c("Z0.Z2", "Z1.Z0")]
real <- data.frame(y = y, z[, 1:2], ints)
binary_x <- c(-5, 8, 3, -7, 6, -7, -7, 2, 1, 4, -8, 4)
binary_y <- rep(c(0, 1), each = 6)
binary_fit <- predict(lm(binary_y ~ binary_x))
write.csv(data.frame(z, y, binary_x, binary_y), file.path(out, "input.csv"), row.names = FALSE)
expected <- data.frame(
  corr = sqrt(summary(lm(y ~ ., real))$r.squared),
  auc = as.numeric(pROC::auc(binary_y, binary_fit, quiet = TRUE)),
  bw = bw.nrd0(c(0.2, 0.3, 0.5, 0.8)),
  bw_constant = bw.nrd0(rep(0.7, 4)))
write.csv(expected, file.path(out, "expected.csv"), row.names = FALSE)
