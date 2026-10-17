# Five-arm backend statistical repeats

Status: 40 runs / 160 cells completed; all ten Python-with-R-knockoffs runs
timed out. The user excluded that arm. Final four-arm analysis completed in
job 11958124; tables and figures verified. See
[final results](../2026-10-16_backend-replicates-four-arms/README.md).
Source: c4726e3 (Gaussian arithmetic and interaction-residual fixes).

## Design

Reuses benchmarks/benchmark_backends.py and the October 13 run.sbatch,
with the arm list extended and array range changed to 0–49 (10 concurrent).
No model or analysis code changes.

- Arms: rust, python (DSDP), pure_r, rust_r_knockoffs, r_knockoffs.
- Ten runs each: seed labels 42, 1000, 2000, 3000, 4000, 5000, 6000,
  7000, 8000, 9000.
- Delta 0.01/0.1 × lambda 0.1/1.0: 50 tasks, 200 cells.
- Parity settings, 500 iterations, specificity 0.1, SLIDE selection,
  four workers/threads; BLAS threads 1.
- HTC Turin, four CPUs, 4 GB, 20 minutes per task.
- Frozen X/Y, R SLIDE package and Python 3.11 DSDP/glmnet extensions
  reused from the October 13 run. Fresh Rust build from frozen source.
- Fresh pure_r runs are the primary reference; no historical pooling.

Staging: .tmpfiles/2026-10-16_backend-replicates-all-arms/
under the repository root. This preserves all previous runs.

## Jobs and checks

- Build attempts 11957806 and 11957808 failed because Homebrew binutils
  require GLIBC_2.38, unavailable on the compute node.
- Build 11957812 uses PATH=$HOME/.cargo/bin:/usr/bin:/bin, system cc/c++/ar/ranlib,
  a new Cargo target directory, and the existing with-rust-blas-env.sh wrapper.
- Benchmark array: 11957807, dependent on the successful build.
- Original analysis: 11957810, cancelled after its dependency became unsatisfiable.
- Final four-arm analysis: 11958124, completed successfully.

Python DSDP, compiled glmnet and rpy2 import checks passed. R imports passed:
SLIDE 0.1.1, glmnet 5.0, knockoff 0.3.6, Rdsdp 1.0.6.
Each task checks the frozen input/extension manifest, all four cell statuses,
and the presence of four selection outputs.

The original plan was for analyze_seed_replicates.py to compare all four non-reference
arms with fresh pure_r, plus rust:python, rust:rust_r_knockoffs and
python:r_knockoffs. It requires ten runs per arm/cell and K/Z agreement
with R within 1e-8. It writes count tests, factor frequencies and Wilson
intervals, Fisher BH q-values, Jaccard/permutation comparisons, bootstrap
runtime intervals, and three figures here. It uses 10,000 permutations
and 10,000 bootstrap samples. Inspect these outputs after completion.

## Reproduction

Set RUN to the absolute staging directory. Environment:
ENV=/vast/djishnu/Aaron/.tmpfiles/20260902_knockoff-generator-parity/project/.pixi/envs/dev
R_DEPS=/vast/djishnu/Aaron/.tmpfiles/20260902_r-generated-knockoff-backend-evaluation/R-library

Submit the prepared configuration:
sbatch --output="$RUN/logs/%A_%a.out" "$RUN/run.sbatch" "$RUN" "$ENV" "$R_DEPS"

The source snapshot, REVISION, inputs, run.sbatch, compiled extensions,
build logs and job IDs are in staging. SLURM retains the submitted analysis
command in job 11957810.

## Limits

These are independent backend draws, not seed-paired comparisons; stock R
PSOCK workers are not fully controlled by seed labels. Ten repeats on one
dataset cannot establish equivalence or FDR control. Runtime includes node
variation and backend-specific timing boundaries. The analysis does not
extract knockoff-stage timing for rust_r_knockoffs; total wall timing remains
available. Permutation p-values are exploratory and not multiplicity-adjusted.
