# Gaussian parity backend benchmark after the tied-entry statistic fix

Submitted as **SLURM job 11422548** on cluster `htc` at 11:45 EDT, September 22,
2026. It started on `htc-n86` within seconds. Frozen source: `605557c`, whose
Rust tree equals the fix commits `b1d0129` and `4a18274`. This record was
written at submission time; the job had not completed. Do not treat it as a
completed benchmark until a results readout is added.

## Purpose

Isolate the production fix. Commit `b1d0129` zeroes the knockoff signed-max
statistic when the original and knockoff entry lambdas tie exactly, matching
R/Python `sign(0)`. The [replay record](../2026-09-22_tie-fix-replay/README.md)
shows the fix reproduces the tie-zero counterfactual on all 5,000 saved paired
draws. This run asks what the fix does to the full six-backend pipeline.

Everything else copies the [September 14 parity run](../2026-09-14_all-backend-parity/README.md):

- Same SSc UnTx `X.csv`/`Y.csv` inputs, byte-identical to the September 14 copies.
- Same four cells: delta {0.01, 0.1} × lambda {0.1, 1.0}.
- Same six routes in the same order: rust, rust_r_knockoffs, python_serial,
  python, r_knockoffs, pure_r.
- `--configuration parity`: 500 Gaussian lambda points, tolerance 1e-7,
  iteration cap 100000, lambda ratio 1/2000, positive-definite-only covariance gate.
- 500 knockoff draws, seed 42, spec/FDR 0.1, findOptIter, interaction fitting.
- Four workers, except serial Python; BLAS threads 1, Rayon threads 4.
- Same installed SLIDE R package (built September 3, `SLIDE_R_OPTIMIZE=1`),
  same Python/R environment and R dependency library.
- 64 GB, 12 hours, one Turin node on cluster htc; no builds inside the job.

The only source difference from `a74c521` besides the fix is documentation and
benchmark tooling committed since then. Draw counts and the paired diagnostic
(1,000 draws per design) are unchanged. `run.sbatch` differs from the
September 14 script in one token, `--source-revision 605557c`.

## Frozen run directory

`/vast/djishnu/Aaron/1_general_use/omni-SLIDE/.tmpfiles/2026-09-22_all-backend-parity-tiefix/`

- `source/`: 300 tracked files from `605557c` (`src/`, `benchmarks/`,
  `crates/`, `scripts/`, `Cargo.toml`, `Cargo.lock`, and the two frozen test
  files); two more than September 14 because of `benchmarks/prepare_paired_cell.py`
  and `benchmarks/replay_paired_xtilde.py`.
- `bin/slide`: release CLI rebuilt from the fixed source. SHA256
  `62c498014cb320cc0454458015a348a43217507aab486b314def71c0cc477169`.
- `inputs/`, `R-library/`: copied from the September 14 run directory.
- `manifest.sha256`: 323 hashes, paths relative to the run directory. The job
  verifies them before running.

Preflight, logged under `logs/`: all 323 hashes pass, `bash -n run.sbatch`,
`bin/slide --help`, the installed R package loads from the frozen library,
`tests/tests/test_knockoff_glmnet_defaults.py` passes against the frozen source,
and the Rust fixed-ratio lambda parity test passes. The full Rust test suites
for `slide-glmnet` and `slide-rs` passed before the freeze, except the
pre-existing `compare_python_lasso` integration test, which fails only because
its `/tmp` CSV fixture is absent; that failure is independent of this change.

## Submission

```bash
RUN="$PWD/.tmpfiles/2026-09-22_all-backend-parity-tiefix"
ENV=/vast/djishnu/Aaron/.tmpfiles/20260902_knockoff-generator-parity/project/.pixi/envs/dev
R_DEPS=/vast/djishnu/Aaron/.tmpfiles/20260902_r-generated-knockoff-backend-evaluation/R-library
sbatch --output="$RUN/logs/slurm-%j.out" "$RUN/run.sbatch" "$RUN" "$ENV" "$R_DEPS"
```

The job rejects failed or missing cells and requires 24 results and 60
within-cell comparisons. Expect roughly 30 minutes based on September 14.

## What this run can and cannot show

Compare its `delta=0.1, lambda=1` Rust selections against the September 14
values (11 marginal LFs, 42 interactions for Rust; 2 or 3 LFs for R and Python).
Movement toward R/Python there is consistent with the paired evidence. RNG
streams and swaps still differ across backends, findOptIter picks a single
draw, and the union design in the replay still disagrees with R on one
near-threshold feature. Residual per-draw differences from coefficient
convergence or entry ordering remain unexplained. Do not claim full-pipeline
parity or complete selection stability from this run alone.
