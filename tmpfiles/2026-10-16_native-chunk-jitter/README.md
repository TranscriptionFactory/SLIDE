# Native-chunk jitter replay

Reusable diagnostic driver, promoted from the project-local scratch run.
Reuses fit_draw, paired_vote_summary and aggregate_selection; does not change
production sampling or fitting. Inputs are the existing 24×22 saved native
chunks 1 and 4 (500 draws). Assertions verify the geometry and native controls.

Command and results:
../../benchmark_results/2026-10-16_native-chunk-jitter/README.md

This tests delta 0.01, lambda 0.1 with the corrected Gaussian fitter, unlike the
older jitter decomposition's saved design. Raw audit outputs remain under
.tmpfiles/2026-10-16_native-chunk-jitter/.

Optional --r-s adds the audit's R-S arm on the exact native covariance.
It checks bitwise covariance equality and the unregularized factor residual,
then uses the same innovations, swaps and fitter. Results:
../../benchmark_results/2026-10-16_native-chunk-r-s/README.md.
