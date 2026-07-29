# Prompt: Investigate SSc Significant-LF Divergence

Investigate why Rust and Python SLIDE select different significant latent
factors from fully native R SLIDE on the canonical SSc UnTx benchmark.

## Objective

Localize each discrepancy to a precise stage, separately for Rust and Python,
then propose the smallest parity fix backed by regression tests. Native R is
the reference. Do not change LOVE/Z code unless evidence contradicts the
observed pairwise Z correlation of 1.0.

## Inputs and Baselines

- Current five-backend run:
  `benchmark_results/20260729_ssc-five-backend-release`
- Interim saved-R report:
  `benchmark_results/20260729_ssc-five-backend-release/interim_saved_native_r_comparison.md`
- Saved native-R reference:
  `benchmark_results/lfcompare_tolfix_20260617_233303`
- Dataset:
  `/ix/djishnu/Aaron/1_general_use/SLIDE/Data_Scripts/SSc/UnTx/X.csv`
  and `Y.csv`
- Grid: delta `{0.01, 0.1}` × lambda `{0.1, 1.0}`
- Canonical settings: `niter=500`, `spec=0.1`, `fdr=0.1`, offset `0`,
  ASDP knockoffs, four threads
- Relevant scripts:
  `benchmarks/benchmark_backends.py`,
  `benchmarks/attrib_xinject.R`,
  `benchmarks/attrib_report.py`,
  `benchmarks/emit_w_compare.R`,
  `benchmarks/emit_diags_compare.R`,
  `benchmarks/compare_overselect_cd4.py`,
  and the existing attribution sbatch scripts

## Required Experiments

1. **Reconfirm the boundary**
   - Compare A, Z, y, LF names/order, and selected sets for all backends.
   - Assert identical shapes, ordering, scaling, and Z values before examining
     knockoffs.
   - Report selection counts/frequencies for every LF, not only thresholded
     sets.

2. **Separate knockoff construction from downstream statistics**
   - Record `mu`, `Sigma`, `diag_s`, conditional covariance/Cholesky, seeds,
     and realized per-column `corr(X_j, Xtilde_j)`.
   - Compare native R, Rust-native, Rust-with-R-draws, Python-native, and
     Python-with-R-draws.
   - Verify that “R knockoffs” means R computes covariance, SDP, and Gaussian
     draws—not only the SDP diagonal.

3. **Use paired X-tilde injection**
   - Generate and cache the exact same R X-tilde matrices.
   - Feed those byte-identical matrices into:
     a. native-R `stat.glmnet_lambdasmax`,
     b. Rust's statistic/voting path, and
     c. Python's statistic/voting path.
   - Compare per-iteration W vectors, thresholds, selected indices, and
     cumulative vote counts.
   - Repeat in the opposite direction with Rust/Python-generated X-tilde when
     useful.

4. **Audit lasso/statistic parity**
   - Match centering, sample-standard-deviation convention (`ddof=1`),
     response centering, column swapping, W sign correction, lambda-max
     definition, convergence tolerance, maximum iterations, and coefficient
     zero tolerance.
   - Explicitly sweep/match `nlambda` at 100, 250, and 500; R's knockoff
     statistic commonly uses 500 while Rust defaults may differ.
   - Compare the full lambda path and the first-entry lambda for each original
     and knockoff feature.
   - Determine whether divergences are caused by Rust coordinate descent,
     Python/sklearn or glmnet behavior, or configuration mismatch.

5. **Audit voting and final selection**
   - Match RNG streams or use paired fixed masks for column swapping.
   - Check offset, FDR threshold inequality, treatment of infinite/no
     thresholds, `findOptIter`, `slide_selection`, feature chunking, and
     `selection_count` versus `spec * niter` boundary semantics.
   - Highlight borderline LFs whose counts are near 50 votes.

6. **Account for native-R Monte Carlo variability**
   - Run multiple native-R seeds/replicates or use the six saved SSc runs.
   - Distinguish stable native-R LFs from stochastic boundary LFs.
   - Report both single-run Jaccard and agreement with the native-R selection
     frequency distribution.

## Rust-Specific Deliverables

- Stage-by-stage difference table against R using identical X-tilde.
- Per-LF vote-count deltas and per-iteration first divergence.
- Evidence for or against the Rust lasso path as the remaining cause.
- Minimal Rust/config fix and focused regression fixture.

## Python-Specific Deliverables

- Equivalent stage-by-stage table against R.
- Verify the R sampler is actually invoked on every cached draw.
- Compare Python statistic/lasso behavior against R under identical X-tilde.
- Minimal Python/config fix and focused regression test.

## Final Output

Produce:

1. Runtime and selected-LF table for all five backends.
2. Per-parameter Jaccard against native R.
3. Per-LF native-R frequency and backend vote-count table.
4. Attribution table: covariance/SDP, sampling, swap RNG, lasso statistic,
   voting threshold, or Monte Carlo boundary.
5. Ranked root causes with direct evidence.
6. Minimal proposed fixes; no speculative refactors.
7. Tests that fail before each fix and pass afterward.

