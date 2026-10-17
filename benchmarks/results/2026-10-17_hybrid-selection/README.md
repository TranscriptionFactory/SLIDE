# Five-backend latent-factor selection

![Marginal selection counts and rates](factor_rates.png)

Compare Rust and Python with native or R-generated knockoffs against pure R.
The results come from the [October 16 replicate campaign](../2026-10-16_backend-replicates/README.md);
we did not pool them with the production-readiness pilot.

Annotations give selected/completed runs. Colors show selection rates for
factors selected in at least 20% of completed runs in any arm within that cell.
The [CSV](factor_rates.csv) includes all factors selected at least once,
including those below the display threshold, and pointwise 95% Wilson intervals.
Use the CSV intervals to assess uncertainty; the colors show rates alone.
This figure includes no significance tests.

Python with R knockoffs completed 10, 7, 0 and 0 runs across the four cells.
Gray N/A entries mean no completed runs, not zero selections. The seven-run
cell is completion-conditioned and subject to timeout-related selection bias.
See [coverage and alignment checks](../2026-10-16_backend-replicates/SETUP.md#python-with-r-knockoffs-partial-evidence-only).
Compare factor identities within a parameter cell.

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

## Extended-limit rerun

We submitted a [ten-task Python/R-knockoff rerun](../2026-10-17_python-r-knockoffs-extended/README.md)
as array 11967812 with two hours per task. This figure retains the original
results until we validate the completed rerun.
