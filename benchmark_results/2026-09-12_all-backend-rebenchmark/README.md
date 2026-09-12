# Full-backend rebenchmark

Frozen source revision: `4c07012`. The Python 3.11 glmnet binary uses `-O3`;
the release Rust CLI includes Clarabel `faer-sparse` with automatic solver
selection. Native R uses the verified `SLIDE_R_OPTIMIZE=1` interaction changes.

- Dataset: the saved SSc UnTx X/Y from the September 12 R stability run.
- Grid: delta 0.01 and 0.1; lambda 0.1 and 1.0.
- 500 knockoff draws, seed 42, FDR 0.1, spec 0.1, `findOptIter` selection.
- Six routes: rust, rust_r_knockoffs, python_serial, python, r_knockoffs, pure_r.
- Four workers (one for python_serial), one BLAS thread per worker.
- Preserve backend lasso-grid defaults: Rust 250, Gaussian Python/R 500.
- One pass per cell, 24 full-pipeline runs, sequential on one Turin SLURM node.
  No build runs inside the timing job. This is a current-configuration comparison,
  not an isolated speedup or numerical-equivalence experiment.

The run script reuses the existing benchmark harness, rejects failed or missing
cells, then verifies 24 loaded results and 60 within-cell pairwise comparisons.
Pipeline-selected models may differ. Per-backend wall timers retain the existing
harness boundaries; subprocess startup is included for Rust and native R.

Run directory:
`/vast/djishnu/Aaron/1_general_use/omni-SLIDE/.tmpfiles/2026-09-12_all-backend-rebenchmark/`

It contains frozen source, inputs, Rust executable, and the installed SLIDE R
package. `manifest.sha256` verifies those artifacts relative to the run directory.
The external Python/R environment and R dependencies remain shared; preflight
logs identify versions and confirm the frozen Python extension and R package.

Submission (from the repository root):

```bash
RUN="$PWD/.tmpfiles/2026-09-12_all-backend-rebenchmark"
ENV=/vast/djishnu/Aaron/.tmpfiles/20260902_knockoff-generator-parity/project/.pixi/envs/dev
R_DEPS=/vast/djishnu/Aaron/.tmpfiles/20260902_r-generated-knockoff-backend-evaluation/R-library
sbatch --output="$RUN/logs/slurm-%j.out" "$RUN/run.sbatch" "$RUN" "$ENV" "$R_DEPS"
```

Preflight: shell syntax passes; frozen Python/compiled-glmnet and rpy2 imports
succeed; R loads SLIDE, knockoff and corpcor; the copied Rust CLI starts.
Results and logs will remain in the run directory until the completed outputs
are verified and summarized here.
