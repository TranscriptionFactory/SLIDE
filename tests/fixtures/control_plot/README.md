# R control-plot reference

Regenerate from the repository root with:
`TMPDIR="$PWD/.tmpfiles" Rscript tests/fixtures/generate/generate_control_plot_fixtures.R`.

The generator follows `vendor/SLIDE-jishnu/R/calcControlPerformance.R` and calls
its `pairwiseInteractions.R`: real model uses marginals Z0/Z1 and exactly
Z0.Z2/Z1.Z0, fitted once with an intercept. Binary fixture demonstrates that
pROC's median-based automatic direction can produce AUC below 0.5; taking the
maximum orientation is incorrect. Bandwidth references use R's `bw.nrd0`.
No random-stream equality is asserted across languages.
