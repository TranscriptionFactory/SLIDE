# Native-R iteration stability sweep

Submitted SLURM array **11232664** on `htc` on September 12, 2026.
Submission is verified; production results are pending.

## Experiment

- SSc UnTx: 24 samples, 804 input features.
- Native installed SLIDE 0.1.1, R 4.5.3, unchanged `findOptIter` selection.
- Seeds: 42, 1000, 2000.
- Knockoff iterations: 1000, 2000, 5000.
- Delta: 0.01, 0.1; lambda: 0.1, 1.0.
- Nine array tasks, one iteration/seed combination per task, four cells each:
  36 pipeline runs. At most three tasks run concurrently.
- Four CPUs per task, BLAS/OMP threads pinned to one, Turin nodes, 32 GB RAM,
  two-hour task limit. No changes to solver tolerances or iteration caps.
- FDR/spec remain 0.1; other R runner settings retain their existing defaults.

The frequency-only experiment from August was a deliberate algorithm change,
not closer emulation of stock R. Stock R uses the frequent set to choose a
single draw via `findOptIter`; the experimental branch returned the frequent
set instead, affecting both screening and aggregation. This sweep does not
repeat that intervention.

## Frozen artifacts and verification

Run directory, relative to the current `/vast` checkout:

`.tmpfiles/2026-09-12_r-stability/`

It contains copied X/Y inputs, the installed SLIDE package, unchanged benchmark
and R runners from `096085e`, the submission script, and a SHA-256 manifest
covering 25 frozen files. Each job verifies the manifest, records R session
information, and writes isolated results under `results/niter_N/seed_S/`.
The script rejects failed cells or missing result records even if the shared
benchmark harness exits successfully.

Both input hashes match the original September grid Rust checkpoint:

- X: `9cc14385d91db2be81ee10454cbafccde8a135699f95bbfa3c8f170aec0665b8`
- Y: `c4a141c7c55777746ebac95ebf0d25dc034a7643e422d37a36aede8371d4ae1a`

X labels are V1–V24 and Y labels are SC identifiers. A direct label-equality
preflight failed; exact file hashes establish that this run preserves the
previously verified positional pairing. This is not an external patient-ID
audit, and no row relabeling or reordering was performed.

Verified shell syntax, embedded Python syntax, 36-run mapping, R package loading,
installed `findOptIter` presence, input dimensions/hashes, and SLURM `--test-only`.
No local full-pipeline smoke run was performed.

Environment:
`/vast/djishnu/Aaron/.tmpfiles/20260902_knockoff-generator-parity/project/.pixi/envs/dev`

Additional R dependencies:
`/vast/djishnu/Aaron/.tmpfiles/20260902_r-generated-knockoff-backend-evaluation/R-library`

These environment/dependency directories are reused, not copied. The run's
copied SLIDE package takes precedence. All explicit source/input/runtime paths
are on `/vast`; scratch stays under the run directory.

## Analysis after completion

Check all 36 timing records and outputs before comparing iteration settings.
For each cell and iteration count, report the three between-seed marginal and
interaction Jaccards, selected-count spread, and per-factor selection rates
across the three runs. Compare these measures as iterations increase. Three
seeds provide a preliminary stability check, not a precise variance estimate.
More iterations do not guarantee that `findOptIter` returns a stable discrete
set. No stability conclusion is available at submission time.
