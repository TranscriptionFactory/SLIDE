# Gaussian scheduling regression

Captured 2026-10-15 from the saved `.tmpfiles/diag_c8` draw (24 rows,
18 original/knockoff pairs). Its full benchmark-cell provenance is unknown.
The inputs contain the swapped augmented design after installed R 4.5.3
`scale()`. `lambda.csv` contains the exact grid passed by knockoff 0.3.6 to
glmnet 5.0. `expected_sign.csv` contains all 18 R signed-max W signs before
undoing swaps. No factors or coefficients were filtered or perturbed.

Reproduce with `tmpfiles/2026-10-15_gaussian-parity/replay.py`; exports live in
`benchmark_results/2026-10-15_gaussian-parity/fixed-draws/saved_c8/`:
`r_x.raw`, `r_lambda.raw`, `y.raw`, `swaps.raw`, `w_r_fixed.csv`.
Split the 24x36 row-major R design into its two 18-column halves. Undo the
saved swaps on the reported W signs before comparing with these swapped inputs.
All CSV values use 17 significant digits. The Rust test holds supplied R
preprocessing fixed to isolate the scheduling correction; it does not assert
that native Rust preprocessing or all first-entry indices match R.
