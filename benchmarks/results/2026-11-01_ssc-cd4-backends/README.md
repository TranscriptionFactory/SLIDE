# SSc/CD4 backend campaign

Status: native interaction correction approved; repeating the matched pilot
before production submission. This campaign measures current implementations, not
the historical October results. The user approved the matrix and resource caps
below, conditional on the matched pilot.

## Pilot stop and next decision

Executed source: `3a739ae6770d003794bc26753919dbb213890667`, with a prebuilt
Rust executable and frozen Python/R/runner sources under `source/`.
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

Full fits: 20 seeds per backend/cell, 1,000 through 20,000 in steps of 1,000:
300 runs. Equal numeric seeds do not pair cross-language random draws.

Final CV: each backend evaluates the same predetermined R pilot factor matrix
(seed 42) for each cell, with 15 repeats, 5 folds and 500 selection draws.
Both observed and training-response-permuted branches run. There are 15 final-CV
runs, not 300. Each backend generates independent folds; no shared-fold claim.
The matrices are chosen before seeing their scores. No native-factor CV arm.

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

Verification covers input order, effective settings, sampleCV IDs/statuses,
saved-factor compatibility, both CV branches, prediction coverage, pooled
scores, timings and memory records. Preserve historical results unchanged.
Analysis reuses the maintained timing and seed-analysis tools, compares against
R-to-R variability, and does not count all pairwise overlaps as independent
observations. No practical equivalence margins are agreed, so this campaign
cannot claim equivalence from a nonsignificant difference. No FDR/power claim
without a separate known-truth simulation analysis.
