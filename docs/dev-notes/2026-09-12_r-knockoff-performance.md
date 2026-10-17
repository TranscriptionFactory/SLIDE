# R interaction knockoff optimizations

The native-R runner supports cached Gaussian preparation and sparse coefficient
extraction. **Stock R remains the default.** Opt in with `SLIDE_R_OPTIMIZE=1`;
set it to `0` or leave it unset for the reference implementation.

## Scope and equivalence

`src/loveslide/slide_r/knockoffPerformance.R` creates a private interaction
environment. It does not modify installed SLIDE/knockoff namespaces or the frozen
September 12 stability sweep. Marginals retain their original implementation.
The optimization applies to interaction methods 2–4 through `selectShortFreq`;
method 1's `selectLongest` is unchanged. Hybrid Python/R and Rust/R-generated
knockoff routes do not use this layer.

- Compute the Gaussian solve, conditional mean/covariance, and Cholesky factor
  once per `secondKO` subproblem. Keep the original arithmetic order, covariance
  shrinkage, ASDP/equi fallback, and normal draw dimensions/order. Zero covariance
  still returns X, warns per draw, and consumes no random numbers.
- Reuse the installed glmnet fitting function and swap/W-statistic functions.
  Replace only the two coefficient-scan expressions with a CSC sparse scan.
  Explicit stored zeros and NA entries do not count as entry; never-entering rows
  remain NA until stock code maps them to zero. Multinomial NA propagation stays
  unchanged. The loader rejects an unfamiliar coefficient-scan structure rather
  than silently applying a partial optimization.
- Do not cache selections or draws. Each spec escalation still reruns selection
  and samples new knockoffs. Lambda paths, tolerances, FDR, chunking, and foreach
  task dispatch are unchanged.

The helper defines functions without loading knockoff at source time, since the
legacy LOVE wrapper sources this directory even when it does not run knockoffs.

## Verification

`tests/test_r_knockoff_performance.R` passes 16 checks with R 4.5.3, knockoff 0.3.6,
SLIDE 0.1.1, glmnet 5.0, and single-threaded OpenBLAS:

- Exact generated matrices and RNG state for vector/matrix diagonal input,
  nonzero means, named inputs, and zero covariance.
- Exact sparse entry indices, Gaussian/binomial/multinomial entry values,
  W statistics, and swap RNG state.
- Exact per-draw selections, frequency tables, and RNG state on full-rank,
  shrinkage, and data-frame subproblems.
- Real two-worker PSOCK execution: exact selections and both master/worker RNG
  states with fixed worker assignment. Each worker's RNG advances.
- Nonempty method 2 and default method 4 interaction outputs. Method 2 uses
  `identical`; method 4 uses zero-tolerance `all.equal` because fitted-model terms
  contain fresh call-frame environments. Installed namespaces remain unchanged.

These are paired numerical checks, not a universal equivalence proof. Dynamic
worker assignment can change which RNG stream serves a draw when tasks run
faster. A shared seed alone therefore does not establish full-run reproducibility
under load-balanced scheduling. The optimization does not change the scheduler
or introduce a new RNG policy. Tests use fixed worker assignment to isolate
numerical equivalence. No full production grid was rerun.

## Measured speed

Synthetic interaction design: 80 observations, 100 predictors, 100 draws,
three repeats, serial execution and one BLAS thread. Times below are medians
for a 100-call batch, or one 100-draw `secondKO` call. This is a microbenchmark,
not a production interaction runtime estimate.

| Component | Stock (seconds) | Optimized (seconds) |
|---|---:|---:|
| Gaussian generation | 0.186 | 0.078 |
| Coefficient extraction | 0.385 | 0.038 |
| Complete W statistic | 2.926 | 2.622 |
| `secondKO`, including setup and fitting | 3.381 | 2.970 |

The complete subproblem took about 12% less time at the median. Its three repeats
ranged from 3.324–5.144 seconds stock and 2.953–2.996 seconds optimized; this is
not a confidence interval. Separate kernel measurements
are not additive: the statistic benchmark fixes Xk, while `secondKO` generates
new Xk each draw. Optimized generation excludes its once-per-subproblem setup;
`secondKO` includes it. Raw repeats and environment details are in
[`benchmark_results/2026-09-12_r-knockoff-performance`](../benchmark_results/2026-09-12_r-knockoff-performance/).

## Reproduce

Use an R environment with SLIDE, knockoff, glmnet, foreach, doParallel, and
corpcor installed; set `R_LIBS` if these live outside its default libraries.
From the repository root:

```bash
mkdir -p .tmpfiles/2026-09-12_r-knockoff-perf
export TMPDIR="$PWD/.tmpfiles/2026-09-12_r-knockoff-perf"
export TMP="$TMPDIR" TEMP="$TMPDIR"
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
Rscript --vanilla tests/test_r_knockoff_performance.R
Rscript --vanilla archive/2026-10-17_production-cleanup/benchmarks/profile_r_knockoffs.R \
  .tmpfiles/2026-09-12_r-knockoff-perf/repeat-benchmark
SLIDE_R_OPTIMIZE=1 Rscript --vanilla benchmarks/run_slide_r.R <existing runner arguments>
```

Copied runners that enable the optimization must also copy
`src/loveslide/slide_r/knockoffPerformance.R` relative to their repository root.
Rerun paired tests before using another dependency environment as a numerical
reference.
