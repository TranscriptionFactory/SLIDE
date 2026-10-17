# Five-backend latent-factor selection

![Marginal selection counts and rates](factor_rates.png)

This figure adds Python with R knockoffs to the four-arm
[October 16 replicate campaign](../2026-10-16_backend-replicates/README.md).
It does not pool that campaign with the production-readiness pilot.
Rust and Python each appear with their native knockoffs and with R knockoffs;
the fifth arm is pure R.

Annotations give selected/completed runs. Colors show selection rates for
factors selected in at least 20% of completed runs in any arm within that cell.
The [CSV](factor_rates.csv) includes all factors selected at least once,
including those below the display threshold, and pointwise 95% Wilson intervals.
The figure does not display uncertainty or test significance.

Python with R knockoffs completed 10, 7, 0 and 0 runs across the four cells.
Gray N/A entries mean no completed runs, not zero selections. The seven-run
cell is completion-conditioned and subject to timeout-related selection bias.
See [coverage and alignment checks](../2026-10-16_backend-replicates/SETUP.md#python-with-r-knockoffs-partial-evidence-only).
Factor identities compare only within a parameter cell.

## Reproduce

From the repository root, using the benchmark Python environment (NumPy,
pandas, SciPy, matplotlib and seaborn):

```bash
python benchmarks/scripts/analyze_seed_replicates.py \
  benchmarks/results/2026-10-17_hybrid-selection \
  --saved-selection-results benchmarks/results/2026-10-16_backend-replicates
```

Inputs are the retained four-arm `timings.csv` and successful Python-hybrid
`results/r_knockoffs/seed_*/r_knockoffs/timing_shards/*.json` files with their
corresponding `sig_LFs.txt` files. No benchmark jobs or statistical tests reran.
