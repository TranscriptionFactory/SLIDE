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
  --nrep 15 --k 5 --seed 20261002 --threads 20 --max-active-folds 1
```

Use an existing parent directory and a new output directory. The command refuses
to overwrite a directory. It does not submit a scheduler job. On an HPC system,
run production CV inside an allocation rather than on a login node.

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

`--max-active-folds` (default 1) limits active folds independently of `--threads`.
Increasing the thread count does not increase fold concurrency. Batches retain
split order, per-fold seeds and observed/permuted pairing. Each fold disables
parallel knockoff voting; ASDP blocks run sequentially. Library callers must
also set BLAS/OpenMP thread counts to 1 before starting worker threads. The CV
CLI rejects unset or non-1 `OPENBLAS_NUM_THREADS`, `OMP_NUM_THREADS` and
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
below it fits every allocation. Keep the default fold limit until the intended
workload has been measured.

See `sdp-memory-status.md` for measured memory, numerical comparisons and limits
of the validation. R's public feasibility shrinkage differs from Rust's existing
repair, so raw SDP agreement does not imply identical knockoff samples or AUCs.
