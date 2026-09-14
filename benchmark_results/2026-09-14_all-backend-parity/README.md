# Gaussian parity backend benchmark

Submitted as **SLURM job 11241717** on cluster `htc` at 08:49 EDT, September 14.
Verified RUNNING on `htc-n70`, starting native Rust. Frozen source: `a74c521`.
Live job metadata confirms parity settings and four workers. Results pending.

- Same SSc UnTx inputs and four-cell grid as September 12:
  delta {0.01, 0.1} × lambda {0.1, 1.0}.
- Six routes: rust, rust_r_knockoffs, python_serial, python, r_knockoffs, pure_r.
- `--configuration parity`: 500 Gaussian lambda points, tolerance 1e-7,
  iteration cap value 100000, fixed lambda ratio 1/2000, and positive-definite-only
  raw covariance gate. Four workers, except serial Python uses one; BLAS=1.
- 500 knockoff draws, seed 42, spec/FDR 0.1, findOptIter, interaction fitting enabled.
- Installed SLIDE with `SLIDE_R_OPTIMIZE=1`; the updated runner enforces its worker budget.
- One pass per cell, fixed backend order, sequential in one four-CPU Turin job.
  Request: 64 GB, 12 hours on cluster htc. No builds inside the timing job.

Rust release CLI rebuilt from `a74c521` with Clarabel Faer enabled; SHA256:
`a3ea14f001c5c520c5f6a296731a52e10c217fe5a6e8b9d1d57674a7d1652b3c`.
The frozen Python 3.11 glmnet extension retains the verified September 12 O3 build.
The existing Python/R environment and R dependencies remain shared; source, CLI,
inputs, and installed SLIDE package are frozen in the run directory.

Preflight passed: release build, CLI startup and lambda override marker,
release cached/uncached lambda test under parity settings, compiled Python
500-point Gaussian path, R package/default checks, and installed R worker
regressions with optimization off/on. Shell syntax and all 321 manifest hashes pass.

Run directory:
`/vast/djishnu/Aaron/1_general_use/omni-SLIDE/.tmpfiles/2026-09-14_all-backend-parity/`

`manifest.sha256` uses paths relative to that directory. Logs and runtime results
stay there. The job checks hashes before running, rejects failed/missing cells,
and requires 24 results plus 60 within-cell comparisons. Each backend retains its
own environment metadata, including executed source revision `a74c521`.

Submission from the repository root:

```bash
RUN="$PWD/.tmpfiles/2026-09-14_all-backend-parity"
ENV=/vast/djishnu/Aaron/.tmpfiles/20260902_knockoff-generator-parity/project/.pixi/envs/dev
R_DEPS=/vast/djishnu/Aaron/.tmpfiles/20260902_r-generated-knockoff-backend-evaluation/R-library
sbatch --output="$RUN/logs/slurm-%j.out" "$RUN/run.sbatch" "$RUN" "$ENV" "$R_DEPS"
```

This matches configurations, not stochastic work or achieved convergence. RNG
streams/swaps differ. Rust caps iterations per lambda/refit; Fortran counts sweeps
across the path. Python excludes imports from timing; Rust/R include subprocess
startup. No repeated or randomized-order cells. Interpret comparisons accordingly.
