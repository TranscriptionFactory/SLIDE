# Proposal: Port loveslide to Rust as `slide-rs`

## Why

The current `loveslide` Python package carries three compounding problems.

**Dependency fragility.** SLIDE's statistical core relies on R packages (glmnet, knockoff, CVXR) accessed through `rpy2`. The `rpy2` bridge is version-sensitive: minor R or rpy2 upgrades routinely break the interface, requiring patched call sites and environment pins. Reproducing results on a new HPC cluster often means hours of R-library reconciliation before a single analysis can run.

**Performance ceiling.** The Python layer is largely a coordinator: it marshals data into R, waits for R to compute, then unmarshals results back. Parallelism is limited to whatever R exposes. NumPy vectorization helps in a few utility functions, but the hot paths (SDP solve, lasso path, knockoff sampling) are owned by R. Moving the full computation into Rust removes the interop overhead and opens the full parallelism surface.

**Reproducibility gap.** Matching results to the canonical R SLIDE implementation requires the same R build, the same R package versions, and the same LAPACK/BLAS linkage. That constraint makes long-term archival and cross-site collaboration difficult. A pure Rust port that mirrors R's numerical choices (MT19937 RNG, glmnet coordinate-descent schedule, LAPACK-compatible eigendecomposition) can be deterministically locked to a git SHA and produce bit-reproducible output regardless of the host environment.

## What Changes

`loveslide` is rewritten from scratch as a self-contained Rust crate named `slide-rs`. The Python package is not modified during the port; it remains available as a reference baseline. Once `slide-rs` passes numerical correctness tests against the R reference, `loveslide` is superseded for production use. A PyO3 thin-wrapper crate (`slide-py`) can expose `slide-rs` to Python callers without any R dependency, but that wrapper is out of scope for the initial port.

The rewrite targets the same algorithmic surface: LOVE decomposition, knockoff generation and filtering, interaction detection, cross-validation, and scoring. No new statistical methods are introduced.

## Capabilities

### New Capabilities

- `love-core`: Dense matrix decomposition implementing the LOVE (Latent-model based OVErlapping clustering) algorithm. Computes the factor matrix A and latent matrix Z from an input feature matrix X via alternating-least-squares with an SDP-regularized covariance step.
- `knockoff-filter`: Generates model-X knockoff copies of the latent factors Z and runs the knockoff filter with a voting scheme to identify statistically significant factors at a user-supplied FDR threshold.
- `slide-pipeline`: Top-level orchestrator that chains LOVE decomposition, knockoff filtering, interaction detection, and scoring into a single end-to-end call matching the `SLIDE()` interface in the R reference.
- `cross-validation`: K-fold cross-validation wrapper around the slide pipeline, selecting the regularization parameter lambda via held-out prediction loss.
- `scoring`: Computes prediction performance metrics (R-squared, RMSE, held-out likelihood) from fitted factor weights and a response vector.
- `sdp-solvers`: Rust-native semidefinite program solver for the LOVE covariance constraint, either a port of DSDP or a thin wrapper around the `clarabel` crate.
- `glmnet-stats`: Coordinate-descent lasso solver matching glmnet's warm-start, active-set, and lambda-grid conventions, used by the knockoff statistic computation.
- `data-io`: Readers and writers for the CSV/TSV matrices and response vectors that feed the pipeline, plus serialization of fitted model objects to JSON or a binary format.

### Modified Capabilities

- `loveslide` (Python): No source changes. Its role shifts from production tool to numerical reference baseline used in integration tests that compare `slide-rs` output against known-good R results.

## Impact

**New artifact.** A Rust workspace at `slide-rs/` containing the sub-crates listed above. Each sub-crate is independently versioned and testable.

**Removed runtime dependencies.** `rpy2`, R, glmnet, knockoff, and CVXR are no longer required at runtime. They remain as optional dev dependencies for generating reference fixtures used in correctness tests.

**API surface.** The primary public API is a Rust library crate. A CLI binary (`slide`) that accepts matrix files and a response file is included for scripted use. PyO3 bindings are a post-port deliverable.

**Affected code.** All of `loveslide/slide.py`, `love.py`, `knockoffs.py`, `cv.py`, `score.py`, and the bundled `love_python/` and `knockoff/` sub-packages are being replaced in function. No files are deleted from the Python package; they serve as algorithmic documentation during the port.
