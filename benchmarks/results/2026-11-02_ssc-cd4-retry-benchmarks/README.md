# SSc/CD4 fresh-retry benchmark campaign

Status: submission stopped at preparation; resubmission requires approval.
Preparation job `12090713` failed before compilation because rustup could not
lock its settings file on the shared home filesystem (`No locks available`).
The scheduler cancelled all 44 dependent production jobs, `12090714`–`12090757`.
No benchmark cells ran. The preparation script now invokes the installed Rust
1.99.1 toolchain directly, but this repair has not been tested on a compute node.

The frozen source is
`81e1fc97fc13e3e2be3e579907f4608a1ece3515`, which gives native full-fit
marginal size-cap retries fresh draws and separates marginal/interaction retry
seed ranges. R is unchanged. This campaign does not yet have results.

## Approved experiments

Both SSc (24 subjects, 804 features) and CD4 (2,482 subjects, 1,776 features)
use delta {0.01, 0.1} × lambda {0.1, 1.0}. Backends are checked-in R,
native Python and native Rust, with homogeneous LOVE, continuous/Spearman
responses, parity settings, FDR 0.1, specificity 0.1, `findOptIter`, interactions,
and no outcome-based candidate selection.

| Workflow | Settings | Total |
|---|---|---|
| Production full fitting | 1,000 selection draws; 10 seeds per backend/cell | 240 fits in 20 bundled jobs |
| sampleCV within each fit | 500 holdouts, K=4; fixed selected raw terms | Included in each fit |
| Saved-factor final CV | 500 selection draws; 10 repeats × 5 folds; observed and training-response-permuted branches | 24 jobs |
| Preparation pilot | Seed 42; full-fitting settings above | 24 fits after the Rust build |

Full-fit seeds are 10,000,000 through 910,000,000 in steps of 100,000,000,
the first ten seeds of the previous campaign. Final-CV base seed remains
20,261,101. Equal seeds do not synchronize cross-language draws or folds.

## Inputs and saved factors

`inputs/` copies the previous campaign's staged data without changing row order
or values. Its provenance and SSc row map retain the user-confirmed positional
mapping; CD4 already has matching ordered IDs. `input-files.sha256` protects the
staged files. Patient-level inputs remain on the shared filesystem.

Each CV cell uses its predetermined R seed-42 pilot factor matrix:
`pilot/full-fit/<dataset>/pure_r/seed_42/pure_r/<delta>_<lambda>_out/z_matrix.csv`.
All three backends use that same matrix. The choice does not depend on scores.
The pilot checks all 24 cell outcomes, sample IDs, factor names/order and
absolute factor agreement within 1e-8 before writing `saved-factors.sha256`.
An undefined sampleCV score is not a process failure.

## Submission and resource limits

`jobs.csv` records accepted scheduler IDs and dependencies. One preparation job
builds the frozen Rust source and runs the pilot. Every production job requires
that preparation job to succeed. Production jobs additionally form a sequential
`afterany` chain so a failed production job does not suppress later comparisons.
Invalid dependencies cancel waiting jobs. Requeue and automatic resubmission
are disabled.

All 45 submitted jobs requested four CPUs, 32 GiB, the `htc` partition and `turin` constraint,
with one BLAS/OpenMP thread per worker. Preparation has a four-hour limit and
a 300-second per-cell pilot timeout. Each full-fit bundle contains 12 cells,
with a 1,190-second child timeout and a 4-hour-2-minute scheduler limit.
Each CV job has a 7,100-second child timeout and a 2-hour-2-minute scheduler
limit. Total maximum reservation is about 534 CPU-hours, below the previous
560-CPU-hour cap. These are caps, not runtime forecasts; queue time is extra.

The three additional CD4 settings have no prior timing measurements. A failed
pilot blocks production rather than triggering an automatic retry or a larger
resource request. Saved-factor CV has no new pilot: its unchanged runner has
existing focused coverage; the new full-fit pilot validates its factor inputs.

## Execution and provenance

The campaign reuses `benchmarks/scripts/benchmark_backends.py` and
`run_backend_array.sh --manifest`; the manifests extend the previous campaign's
JSON task format. No new benchmark implementation or per-run Python driver is
introduced. `prepare.sbatch` performs the build and pilot gate.

`source/` is a frozen Git snapshot plus the installed Python glmnet/DSDP native
extensions. `source-files.sha256`, `build-artifacts.sha256` and the input/factor
hashes guard jobs against altered inputs or executables. `environment.sh` fixes
the frozen Python/R source and dependency paths. Rust uses its native backend,
not Python. Build-only local pkg-config metadata selects the installed Conda
OpenBLAS library; no dependency or tracked package configuration changes.

To execute a fresh manifest within an allocation:

```bash
source benchmarks/results/2026-11-02_ssc-cd4-retry-benchmarks/environment.sh
cd "$RUN/source"
bash benchmarks/scripts/run_backend_array.sh --manifest "$MANIFEST"
```

Manifests and output/status paths are single-use. Do not resubmit active or
completed manifests. Preserve failures, timeouts, empty selections and undefined
scores in denominators. Logs retain timing, memory and retry diagnostics;
Python marginal retries are logged, but ordinary Python interaction retries
are not, so their incidence cannot be reported completely from these logs.

## Interpretation

Compare selected-factor overlap against within-R repeat variability. Full-fit
sampleCV conditions on terms selected from the full data; final CV conditions
on a full-data factor representation. Neither is end-to-end nested validation.

The [previous report](../2026-11-01_ssc-cd4-backends/analysis/2026-11-02_backend-report/README.md)
uses 500 full-fit draws and 20 seeds, with fewer CD4 settings and 15 final-CV
repeats. This campaign changes draw count as well as retry policy, so a change
in agreement cannot be attributed solely to the fix. No equivalence, FDR or
power claim follows from this comparison.
