# Gaussian arithmetic entry regression

Raw inputs and swap mask are draw index 1 of
`benchmark_results/2026-10-15_rust-covariance/fixed-draws/saved_highdelta`.
`expected_entry.csv` contains R glmnet 5.0's zero-based first-nonzero indices
from that directory's `r_entries.raw` (44 columns; -1 means never entered).

The cached and uncached native Rust paths must recover all 44 entry indices
with sample scaling, 500 lambdas, tolerance 1e-7, and max_iter 100000. The baseline
enters column 22 at index 1 rather than R's 0, and column 39 at 14 rather than 3.
No R-prepared design or lambda grid is supplied to either Rust path.

The inputs are the saved `z.raw` (24 × 22), `xt_fixed.raw` draw 1,
`y.raw`, and `swaps.raw` draw 1, exported as round-trip `%.17g` CSVs.
The existing fixed-draw replay regenerates the R reference on these saved inputs.
