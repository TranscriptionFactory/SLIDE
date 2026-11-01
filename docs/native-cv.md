# Native Rust SLIDEcv

`slide cv` evaluates a saved factor matrix without rerunning LOVE or selecting a
new candidate. It follows the method-4 procedure in `R/SLIDEcv.R`, `benchCV2.R`,
`SLIDE.R`, and `runSLIDEBeta.R`. Rust uses its own random draws and numerical
solvers; equal seeds do not reproduce R's selected factors or predictions.

## Run one saved candidate

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
pixi run --locked -e rust ./target/release/slide cv \
  --config path/to/input.yaml \
  --z path/to/z_matrix.csv \
  --out results/2026-10-27_native-cv \
  --nrep 15 --k 5 --seed 20261002 --threads 20
```

Use an existing parent directory and a new output directory. The command refuses
to overwrite a directory. It does not submit a scheduler job. On an HPC system,
run production CV inside an allocation rather than on a login node.

## Progress logs

`slide cv` writes progress to stderr by default; `--verbose` is not required.
Logs include input dimensions, worker count, repeat/fold starts and completions,
observed (`Slide`) versus permuted (`SlideY`) fits, selection stages, and elapsed
seconds. Repeat and fold numbers in logs are one-based. Completed fold pairs
count both fits; parallel workers can finish folds out of order.

Each knockoff voting call reports successful draws every 100 draws and at its
last draw. Use the thread ID to follow a fold's selection stages, and the voting
seed to distinguish voting calls. The draw counter is local to that call, not
the whole CV run. It is not a wall-clock heartbeat: one slow draw or covariance
setup can still leave a gap between messages. These logs do not provide an ETA.
Result files still appear only after the full CV computation finishes.

For example, append `2>cv.log` to capture progress. Already running processes
keep their old logging behavior after a rebuild; no restart occurs automatically.

## Inputs

The input is the R single-run YAML: `y_path`, `out_path`, `eval_type` (`auc` or
`corr`), `SLIDE_iter`, `spec`, optional `fdr` (default 0.1), and optional
`do_interacts` (default true). Other R fields do not affect saved-factor CV.
`--z` defaults to `out_path/z_matrix.csv`. Paths resolve against the working
directory, as with R. Inputs are numeric CSVs; binary responses must be 0/1,
with `y_levels: [0, 1]` when specified. Z and y row IDs must match in order and
be unique, or both inputs must omit IDs. R CSVs with an empty first header and
numeric row names are supported. No RDS input is needed: K is Z's column count.
This command handles one candidate, not R's recursive scan of optimization runs.

## Statistical contract

- Scale saved Z once over all rows using sample SD, as R does. This retains
  R's full-data factor conditioning; it is not independent-animal validation.
- Generate repeated stratified folds. Numeric responses use caret-style
  quantile strata; invalid training splits are redrawn with a bounded retry
  limit instead of hanging on impossible inputs. Standardize continuous y
  using training-fold mean/SD, including its held-out response values.
- Run `SLIDE` and `SLIDE_y` on the same folds. Only the null branch's training
  response is permuted. Each fit gets fresh, deterministic native RNG draws.
- Use Gaussian knockoff lasso statistics, offset zero, the R covariance gate
  and 500-point lambda-grid policy, and `findOptIter` selection. Compute marginal
  `f_size` from full N and K; interaction screening uses R's default 100.
  Reuse the existing cached sampler and a shared Rayon pool.
- Increase `spec` when selections exceed R's model-size limits. An empty
  marginal selection triggers R's random-one-factor fallback, not a dropped
  fold. Interaction selection residualizes y against each marginal with an
  intercept, fits the per-marginal Gaussian submodels (upsilons), selects among
  them, and fits a final Gaussian predictor. If no upsilon survives, use the
  marginals. Like the current R source, reverse interaction pairs are tested
  separately for each marginal (its numeric exclusion names do not match the
  Z-prefixed column names).
- Use Gaussian predictors even for binary y; predictions are not probabilities
  and can lie outside [0, 1]. Drop aliased design columns in input order before
  the least-squares solve. Near-singular numerical results can differ from R.
- Pool all held-out predictions within each repeat and method before scoring.
  For AUC, median-impute missing predictions and flip scores below 0.5, as R
  does; also retain the unflipped AUC. For continuous y, report Spearman
  correlation and MSE without a direction flip.
- Compute the observed mean and SE across repeat scores, never across folds
  or together with `SLIDE_y`. Keep null summaries separate. Missing sample
  predictions cause an error; undefined continuous repeat scores remain null
  and prevent a partial-repeat mean from being reported.

## Outputs and existing callers

- `cv-result.json`: observed `mean`, `standard_error`, `folds`, `predictions`;
  separate `permuted_*` fields; and `repeat_scores` for both methods.
- `SLIDECV_boxplot_data.csv`: R-style `method` and `auc`/`corr` columns, with
  one-based repeat IDs, raw scores, MSE, and imputation counts.
- `predictions.csv`: both methods, identified explicitly. Sample/repeat/fold
  indices here and in JSON are zero-based. Sample indices refer to input order.
- `requested-config.json` and `README.md`: requested settings, active knockoff environment overrides, and invocation.
  The statistical rules above override generic knockoff family, offset,
  stability-selection, chunking, and covariance policies inside CV.

CSV/JSON exports do not replace R's serialized RDS files or its PDF boxplot.
Existing Rust `cv.enabled: true` callers use the same implementation on their
best saved candidate; their configured `base_seed` now reaches final CV.
The dashboard's mean/SE and prediction plots use only observed results. Its
ROC/PR curves pool observed predictions across repeats and therefore need not
equal the mean of per-repeat, direction-flipped AUCs.

Old checkpoints remain readable. Their missing `repeat_scores` identifies the
legacy fold-average procedure. Resuming an already completed checkpoint does
not recompute its CV. Use `slide cv` with a fresh output directory to reevaluate
those factors. Do not combine old and new CV summaries.

## Verification

```bash
pixi run --locked -e rust cargo +1.99.1 test -p slide-cv --release --locked
# Optional reference test: requires R with dplyr; uses this checkout's R helpers.
SLIDE_RSCRIPT=/path/to/Rscript pixi run --locked -e rust \
  cargo +1.99.1 test -p slide-cv --release --locked \
  gaussian_submodels_and_scores_match_r_reference -- --ignored
```

Tests cover both branches, full prediction coverage, repeat aggregation, AUC
flipping/imputation, Spearman scoring, random fallback, rank-deficient prediction,
seeding, input alignment, and output protection. The R reference test fixes
selected interactions and compares submodel/final predictions; it does not
assert cross-language equality of random knockoff selections. These checks do
not establish a speedup or scientific acceptance on a production dataset.

## Memory and numerical solver

Omit `--max-active-folds` to size fold concurrency at startup. Explicit positive
values are additional ceilings, not overrides of CPU or memory limits. Future
launchers should omit historical hard-coded fold caps when automatic sizing is
intended; archived launchers and benchmark commands retain their original caps.

CPU discovery uses Rust's [standard parallelism detector](https://doc.rust-lang.org/std/thread/fn.available_parallelism.html) (including
supported affinity/cgroup restrictions), capped by `--threads` or
`RAYON_NUM_THREADS`, and tighter Slurm CPU allowances. One Rayon pool serves
folds and voting. There is no load polling or ordinary-run benchmarking.

[`sysinfo` 0.37.2](https://docs.rs/sysinfo/0.37.2/sysinfo/) supplies OS-available memory on Linux, macOS and Windows.
Linux cgroup v1/v2 memory limits are read from the process membership through
visible ancestors, including mounted subtrees. Slurm memory, inherited Linux
address-space/data limits, and `--memory-budget-mib` can tighten that allowance.
The application budget is total process memory; the resident bytes already in
use are subtracted. Container usage is subtracted from each container limit;
virtual address usage is subtracted from the address-space limit. Host available
memory alone does not establish a Slurm job's allowance. If that allowance is
unknown, use one fold, or supply `--memory-budget-mib`. If a known budget cannot
fit one estimated fold, the CLI fails before CV and creates no result directory.

The policy reserves 25% of detected headroom. Its sizing heuristic separates
shared/retained data, active fold workspaces and extra voting workspaces:

- Fold: 128 MiB + 128 × (N × K + K²) bytes + 8 × draws × (K + 3) bytes.
- Retained: saved factor matrix, split indices, both methods' predictions and
  worst-case retained interaction indices across folds/repeats.
- Each extra voting worker: another full fold estimate, conservatively.

These are conservative sizing estimates, not measured peak bounds. Other
processes can consume memory after startup. Slurm metadata without cgroup usage
cannot account for other processes sharing that allocation. Native macOS process
limits and Windows Job Object memory limits are not probed; use an explicit
application budget when those impose a tighter limit. Production peak memory
and laptop behavior remain unverified.

Fixed worker loops take indexed folds from a shared queue: a completed fold
admits the next one without a batch barrier. Results retain split order and
per-fold/per-draw seeds. On error, admission stops, running folds finish, and
no complete result is published. ASDP blocks and interaction submodels remain
sequential. Automatic inner voting uses spare startup CPU capacity only when
its additional workspaces fit the same memory budget. It passed the acceptance
gate below; the hidden `--voting-workers` diagnostic ceiling can restrict it
(for example, `--voting-workers 1` forces serial voting) on the same pool. Outer workers × voting workers never exceeds the CPU
budget. Non-CV callers retain their existing voting behavior; the serde-defaulted
`KnockoffConfig.max_voting_workers` optionally bounds their voting workspaces.

The CLI records requested limits, effective CPU/fold/voting counts, memory
estimates, reserve and budget sources in `requested-config.json`, and logs the
resource decision before computation. Library callers keep explicit configurations
and must set BLAS/OpenMP thread counts to 1 before starting worker threads. The
CV CLI rejects unset or non-1 `OPENBLAS_NUM_THREADS`, `OMP_NUM_THREADS` and
`MKL_NUM_THREADS` rather than oversubscribing nested native threads.

SDP blocks up to 64 variables retain Clarabel. Larger blocks use the existing
vendored DSDP C implementation through `slide-dsdp`, without Python or R at
runtime. The formulation is unchanged: maximize sum(s), with 0 ≤ s ≤ 1 and
2·correlation − diag(s) positive semidefinite. DSDP uses a quadratic-size Schur
system instead of Clarabel's quartic-size PSD scaling workspace. Calls are
serialized because vendored DSDP has process-global logging state. No feature
count, ASDP block limit, selection iteration count or CV split count changes.

The binding requires convergence, primal/dual feasibility status, finite output,
and the existing absolute/relative objective-gap tolerance. Existing correlation
and covariance feasibility repairs still run afterward. Numerical failures keep
the existing fallback semantics. A resource-limit error does not trigger an
equicorrelated fallback: dimensions above 4096 fail before solver allocation.
This is a conservative resource limit, not a promise that every configuration
below it fits every allocation. Use `--max-active-folds 1 --voting-workers 1`
when validating an unfamiliar workload with minimum concurrency.

See `sdp-memory-status.md` for measured memory, numerical comparisons and limits
of the validation. R's public feasibility shrinkage differs from Rust's existing
repair, so raw SDP agreement does not imply identical knockoff samples or AUCs.


### Combined concurrency acceptance

Build the CLI and selected Rust test executables once, then run the existing
portable smoke entrypoint:

```bash
python scripts/slide_artifact_smoke.py --binary target/release/slide \
  --workdir .tmpfiles/YYYY-MM-DD_cv-acceptance --cv-concurrency \
  --test-binary path/to/prebuilt-test-executable --profile-voting
```

Repeat `--test-binary` for the CV, knockoff, SDP, DSDP and CLI suites. The CLI
suite is filtered to `run::tests::`; unrelated TUI tests are outside this batch.
`--cv-fixture` reuses an existing small CLI fixture; `--cv-reference` compares
its 101-draw outputs with accepted files. Without it, native CI reuses the
committed HIV saved factors. `--saved-config`, `--saved-z`, and
`--saved-reference` add the existing 778-cell/425-factor, two-draw diagnostic
(one repeat, five folds, seed 20261602), once with automatic queued folds.
No fitting or production submission occurs.

`--profile-voting` runs three alternating serial/parallel pairs with 1,000 draws,
identical CPU/memory budgets and a per-child timeout (600 seconds by default).
POSIX `wait4` measures each process's CPU time and peak RSS. The existing
`SLIDE_KNOCKOFF_PROFILE` switch adds DSDP lock-wait and solve timing; zero calls
mean the fixture did not enter DSDP, not a measured DSDP speedup. The gate requires
baseline CPU utilization below 80% of budget, every pair faster, median paired
elapsed improvement of at least 10%, byte-identical outputs and respected
resource bounds. A failed or timed-out gate does not authorize enabling automatic inner voting. The portable
resource/queue smoke runs in the existing native-platform CI; profiling is an
engineering acceptance test, not a scientific-equivalence margin.


#### Linux acceptance, 2026-11-01

The combined run passed 59 focused tests (seven opt-in tests skipped), resource
ceilings, inherited Linux process limits, default progress, queue admission and
failure handling, and exact-result checks across concurrency settings. The saved
778-cell/425-factor diagnostic used four queued folds and matched the accepted
result JSON, predictions CSV and scores CSV byte-for-byte. Its settings remained
two draws, one repeat and five folds; it was not a production CV run.

The existing 12-row/3-factor CLI fixture passed three alternating profile pairs
at 1,000 draws, one outer fold, four CPUs and a 2 GiB application budget:

| Pair | Serial seconds | Four voting workers, seconds | Baseline CPU budget used |
| --- | ---: | ---: | ---: |
| 1 | 1.668 | 0.637 | 24.2% |
| 2 | 1.667 | 0.628 | 24.3% |
| 3 | 1.657 | 0.647 | 24.3% |

Median paired elapsed improvement was 61.8%; all pairs were faster and their
outputs matched exactly. Peak RSS was at most 30.2 MiB. The small fixture made
zero DSDP calls: lock-wait and solve totals were zero, not evidence of improved
DSDP throughput. These results authorize the startup-budgeted inner-voting
default, not a production speedup claim.

Evidence: `.tmpfiles/2026-11-01_auto-cv-acceptance/combined/`, including the
command in `README.md`, per-check logs, and paired measurements in `summary.json`.
macOS, Windows, Rust 1.88 execution and production memory/runtime are unverified
locally. Native-platform CI is wired but has not been run in this session.

After enabling the default, the 44 affected CLI/CV/knockoff tests passed again,
including serial-versus-bounded equality for cached and uncached voting. The
small accepted fixture and the portable HIV fixture both passed the smoke;
a one-fold ceiling used four budgeted voting workers with identical outputs.
Only affected checks were repeated; the saved-factor diagnostic and paired
profiling were not rerun. See `enabled-default/` and `portable-fixture/` beside
the combined run for those checks.
