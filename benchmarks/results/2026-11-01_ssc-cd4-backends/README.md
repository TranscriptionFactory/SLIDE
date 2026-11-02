# SSc/CD4 backend campaign

Status: all 31 production jobs completed: 300 full-fit cells and 15 saved-factor
CV runs. The [backend report](analysis/2026-11-02_backend-report/README.md) covers
selection, runtime, memory and held-out performance with figures and CSV tables.
272/300 sampleCV scores are defined; final CV has 450 finite branch/repeat scores.
This campaign measures current implementations, not
the historical October results. The user approved the matrix and resource caps
below, conditional on the matched pilot.

## Submitted jobs

`jobs.csv` records all 31 accepted Slurm jobs and dependencies:

- `12087132`–`12087147`: 16 jobs covering 300 full-fit cells.
- `12087148`–`12087162`: 15 saved-factor CV jobs.

The jobs use a sequential `afterany` chain. Slurm
requeue is disabled (`Requeue=0`) for every job. These submissions contain no
automatic resubmissions and reserve at most 560 CPU-hours in total. Preparation
allocations `12086757` and `12086872` are closed. The descriptive report is
complete; solver/retry diagnosis and any equivalence decision remain outside it.

The source, binary, saved R factor matrices and external environments named in
`environment.sh` must remain available while the jobs run. Each job checks
`build-artifacts.sha256` and `saved-factors.sha256` before execution and records
hardware/affinity under `logs/`. Manifests are single-use and refuse existing
output/status paths. To reproduce with fresh paths, use the maintained runner:

```bash
source benchmarks/results/2026-11-01_ssc-cd4-backends/environment.sh
cd "$RUN/source"
bash benchmarks/scripts/run_backend_array.sh --manifest "$MANIFEST"
```

Run that command inside an allocation; choose the manifest from `jobs.csv`.
Do not resubmit an existing manifest while its job is active.

## Pilot stop and next decision

Executed source: `3a739ae6770d003794bc26753919dbb213890667`, with a prebuilt
Rust executable and frozen sources retained under `pilot/original/`.
Allocation `12086757` used four CPUs and 32 GiB; it was cancelled after 18
minutes when source inspection confirmed the mismatch. All four SSc cells
completed for each native backend. R completed three SSc cells; its fourth
was interrupted. CD4 and saved-factor CV did not start. These are partial
diagnostics, not the approved production seed study.

R `interactionSLIDE` constructs numeric exclusion names such as `2.1`, but
`interUnion` names columns `Z2.Z1`. The exclusion therefore does not remove
reverse pairs: each marginal tests every other factor. Python's full-fit
`_find_interaction_LFs` and Rust's full-fit `select_interactions` exclude
previously used marginals. This changes candidate sets, covariance designs,
selection opportunities and potentially sampleCV. It is not just different RNG.
The separate native final-CV implementations already follow R's reverse-pair
behavior; the mismatch is in full fitting.

The user approved aligning native full-fit interaction candidates and fresh
retry draws with current R. R and LOVE remain unchanged. Native marginal
size-limit retries still reuse their seeds, unlike R; this is a remaining
retry-policy difference, not cross-language random-draw matching. Record its
incidence rather than attributing any resulting differences solely to solvers.

Preparation verification: 32 harness tests, allocated R adapter checks, and a
release Rust build passed. Campaign-local R runtime prerequisites are rngtools
1.5.2 and doRNG 1.8.6.3, installed with
`install.packages(c("rngtools", "doRNG"), lib=CAMPAIGN_R_LIBRARY, repos="https://cloud.r-project.org")`.
The R regression for a completed-but-undefined sampleCV metric passed: the
score remains missing without turning a completed fit into a process failure.
No scientific model construction changes accompany that status change.

## Matched pilot and production source

All 15 full-fit cells passed at `7c89e73` in allocation `12086872`; its source
snapshot is retained under `pilot/aligned/source/`. Screening predictions have
the expected IDs and 500 holdouts (3,000 rows for SSc, 310,000 for CD4). Pooled
scores reproduce from CSV within 1e-8; that is a serialization tolerance, not
a practical equivalence margin. Maximum absolute factor-matrix difference
against R is 4.69e-11. No marginal size-limit retry was observed in this pilot.

The full-fit pilot exposed four-decimal R JSON rounding. Native CSV/RDS values
remain intact; use the CSV screening scores for that pilot. `d5c9d67` fixes JSON
export precision, with a real-model metric/coefficient round-trip regression.
It changes no fits or random draws. Production and all 15 saved-factor CV
pilot cells use `d5c9d675c28ba63a7594fc8c5f1387af7459fc44`.
The Rust executable uses the identical Rust source tree from `7c89e73`.

All CV pilot cells passed: both branches cover every sample exactly once per
repeat; folds match between branches; held-out targets use the training-fold
mean/SD; pooled Spearman and MSE reproduce from predictions. All backends use
the same predetermined R factor matrix per cell. The Python CD4 pilot figure
was rendered and inspected; one repeat provides no uncertainty estimate.
These diagnostics do not establish statistical equivalence or replacement.

## Inputs and fixed settings

- SSc: 24 samples × 804 features; delta {0.01, 0.1} × lambda {0.1, 1.0}.
- CD4: 2,482 samples × 1,776 features; delta 0.1, lambda 0.1.
- Both responses use continuous/Spearman scoring, including CD4's ordinal
  response. No outcome-dependent candidate selection: `auto_select=false`.
- Pure checked-in R, native Python and native Rust; no hybrid arms.
- Homogeneous LOVE, covariance filtering threshold 0.2, interactions enabled,
  FDR 0.1, spec 0.1, 500 selection draws, `findOptIter`, Gaussian knockoff
  statistics, offset zero and the R-compatible 500-point lambda/covariance
  policy. The harness's `parity` setting matches these policies, not RNG draws.
- sampleCV: fixed selected raw terms, unweighted OLS, 500 uniform holdouts,
  K=4, pooled Spearman. Keep its metric separate from in-sample control scores.

The input source is the historical staging directory
`/vast/djishnu/Aaron/1_general_use/omni-SLIDE/.tmpfiles/2026-10-16_production-readiness/inputs/`.
The user confirmed that SSc X and Y correspond by row position. Staging retains
that order and assigns Y's subject IDs to the corresponding X rows; it never
sorts or joins the differing labels. `inputs/ssc/row-map.csv` records the mapping.
CD4 already has matching ordered IDs. `inputs/provenance.json` records source
paths, hashes and staged dimensions. Numerical values and feature order remain
unchanged; input CSVs live on the shared filesystem.

## Run matrix and resource limits

Full fits: 20 seeds per backend/cell, 10,000,000 through 1,910,000,000 in steps
of 100,000,000: 300 runs. Wide spacing avoids overlap between native per-block
and per-retry seed ranges for these cells. Equal numeric seeds do not pair
cross-language random draws. Earlier closely spaced, unsubmitted manifests
were discarded before production.

Final CV: each backend evaluates the same predetermined R pilot factor matrix
(seed 42) for each cell, with 15 repeats, 5 folds and 500 selection draws.
Both observed and training-response-permuted branches run. There are 15 final-CV
runs, not 300. Each backend generates independent folds; no shared-fold claim.
The matrices are chosen before seeing their scores. No native-factor CV arm.
CV base seed: 20,261,101.

Pilot: one full-fit seed (42) per backend/cell with production scientific
settings, capped at 180 seconds each. Saved-factor CV uses one repeat, five
folds and ten selection draws, capped at 120 seconds per backend/cell. These
CV pilot scores are diagnostic, not production performance estimates.

Each task has four CPUs, 32 GiB and one BLAS/OpenMP thread per worker. Build
shared native dependencies serially, outside benchmark timers. Serialize arms
and record effective concurrency, hardware and software versions. All runs use
project `.tmpfiles/tmp/` for temporary files.

Production full-fit timeout: 600 seconds per run. Final-CV timeout: 21,600
seconds per run. The maximum compute reservation is 560 CPU-hours, plus an
eight-CPU-hour preparation/pilot allocation. No automatic job retries. Retain
timeouts, failures, empty selections and undefined scores in denominators.
Child limits are 590 seconds and 21,500 seconds, respectively, leaving time
for final artifact/report writes inside those scheduler caps. Production uses
the `htc` partition's `turin` constraint, with sequential `afterany` dependencies.

## Execution and interpretation

Reuse `benchmarks/scripts/benchmark_backends.py`, `run_slide_r.R` and
`run_backend_array.sh --manifest`. Task manifests under `manifests/` specify
the complete arguments and output paths. Results use separate `pilot/`,
`full-fit/` and `cv/` trees, then dataset/cell/backend subdivisions.

Capture fresh-process end-to-end elapsed time and sampled process-group peak
RSS, plus available internal stage times. Sampling can miss short peaks and
double-count shared pages; it is not an exact cgroup peak. Startup/import costs
remain in end-to-end time, not internal stage times. Unavailable measurements
stay unavailable rather than becoming zeros.

R full fitting also constructs/selects upsilon submodels that do not change
the exported raw term sets used by sampleCV. This consumes time and RNG draws;
full-fit runtimes are not an equal-instruction-count comparison. R's intrinsic
selection retry/skip behavior and native solver failure policies must be
reported separately. Final CV uses the method-4 submodel predictor in all arms.
The R CV engine also reads/scales raw X although the SLIDE predictor uses saved
Z. R exports failed-repeat placeholders and can continue later repeats; native
CV aborts on a fold error. Do not average only the surviving R repeats or treat
those failure-path runtimes as equal-work comparisons. R's early-return cluster
cleanup can leave idle workers until cleanup; the measured group RSS includes
them. The CPU allowance remains four, with four configured voting workers.

Verification covers input order, effective settings, sampleCV IDs/statuses,
saved-factor compatibility, both CV branches, prediction coverage, pooled
scores, timings and memory records. Preserve historical results unchanged.
Analysis reuses the maintained timing and seed-analysis tools, compares against
R-to-R variability, and does not count all pairwise overlaps as independent
observations. No practical equivalence margins are agreed, so this campaign
cannot claim equivalence from a nonsignificant difference. No FDR/power claim
without a separate known-truth simulation analysis.

Analysis must gate selection comparisons on successful cell timings; partial
files from failed cells are not empty selections. Keep those failed cells in
the attempted-run denominators. The seed-analysis tools' legacy `score` field
is in-sample: read `cell_artifacts.sampleCV` and the final-CV repeat-score CSVs
for held-out performance, rather than relabeling that field.
