## 1. Project Setup
- [x] 1.1 Initialize Rust workspace with Cargo.toml (slide-rs) — done: added a root Cargo workspace manifest with shared metadata and validated it with `cargo metadata`/`cargo test --workspace`
- [x] 1.2 Set up sub-crates: slide-love, slide-knockoff, slide-core, slide-glmnet, slide-sdp — done: scaffolded the core library crates with strict lint placeholders and smoke tests
- [x] 1.3 Add dependencies: nalgebra, ndarray, rand, rayon, thiserror, anyhow, serde, csv — done: added shared workspace versions and wired crate-level dependencies for the scaffolded crates
- [~] 1.3a Add design-specific supporting deps when implementation begins — partial: clarabel is wired for LOVE/SDP, but proptest, criterion, and MT19937-compatible RNG are not present
- [ ] 1.4 Set up CI with cargo test, clippy, fmt — stale: no active `.github/workflows` files are present in this checkout
- [x] 1.5 Create test fixtures from R reference outputs (SSc dataset) — done: committed `tests/fixtures/love`, `tests/fixtures/prescreen`, and generator scripts
- [x] 1.2a Align workspace scaffolding with design.md extras: slide-types, cv, score, data-io, slide-cli — done: added design-aligned support crates plus the placeholder binary crate
- [x] 1.5a Add committed `tests/fixtures/` generation assets or a portable SSc fixture snapshot so correctness tests do not depend on external `/ix/...` paths — done: fixture snapshots and `tests/fixtures/generate/*.R` scripts are committed

## 2. Linear Algebra Foundation
- [x] 2.1 Matrix types and conversion (nalgebra ↔ ndarray) — done: added `slide-types` Matrix/Vector newtypes with ndarray↔nalgebra conversions, shape validation, and unit tests
- [x] 2.2 Eigendecomposition wrapper (matching LAPACK dsyev) — done: added a LAPACK-backed symmetric eigensolver wrapper with square/symmetry validation, descending eigenpair ordering, sign normalization, and unit tests
- [x] 2.3 Cholesky decomposition with PD checks — done: added a LAPACK-backed Cholesky wrapper with explicit minimum-eigenvalue PD checks, descriptive errors, triangular factor access, factored solves, and unit tests
- [x] 2.4 Pseudoinverse (pinv) implementation — done: added a LAPACK-SVD-based pseudoinverse with configurable singular-value cutoff and unit tests for full-rank, singular, and rectangular matrices
- [x] 2.5 Graphical lasso (glasso) implementation for precision matrix — done: added the slide-love estOmega column-wise LP solver with Clarabel retries, rayon parallel column solves, and targeted unit tests

## 3. LOVE Implementation
- [x] 3.1 Delta cross-validation (CV_delta, KfoldCV_delta) — done: `slide-love` exports homogeneous `cv_delta`/`cv_lbd` and heterogeneous `kfold_cv_delta`
- [x] 3.2 Pure variable detection — homogeneous (EstAI, EstC, FindSignPureNode) — done: added slide-love pure_homo module with deterministic sign partitioning, EstAI/EstC helpers, and targeted unit tests
- [x] 3.3 Pure variable detection — heterogeneous (Est_BI_C, Re_Est_Pure) — done: `slide-love` exports `est_bi_c`, `re_est_pure`, `post_est_pure`, and `est_k`
- [x] 3.4 Non-pure row estimation (EstY, EstAJInv, EstAJDant, Dantzig LP) — done: added slide-love non-pure estimation helpers plus Clarabel-backed ST/Dantzig LP solvers with focused tests for sign adjustment, normalization, and closed-form identity cases
- [x] 3.5 Precision matrix estimation (estOmega via glasso) — done: `slide-love` exports `est_omega` with row LP solving and tests
- [x] 3.6 Z-matrix computation (MAP estimate) — done: added slide-love MAP Z-score helper with Gamma-weighted loadings, pseudoinverse-based C/G solves, and focused tests for diagonal closed forms plus singular-C behavior
- [x] 3.7 Pre-screening (Screen_X) — done: `slide-love` exports `screen_x` and committed prescreen SSc fixtures
- [~] 3.8 LOVE integration tests against R reference — partial: LOVE fixture data is committed, but no dedicated `slide-love` R-parity test consumes the LOVE fixtures yet

## 4. Glmnet / Lasso
- [x] 4.1 Coordinate descent lasso (Gaussian family) — done: `gaussian_lasso_path`
- [x] 4.2 Coordinate descent lasso (logistic family) — done: `binomial_lasso_path`
- [x] 4.3 Lambda path generation (matching R glmnet) — done: Gaussian and binomial lambda grid builders use glmnet-compatible defaults
- [x] 4.4 W-statistic: lambdasmax — done: `stat_glmnet_lambdasmax`
- [x] 4.5 W-statistic: lambdadiff, coefdiff — done: `stat_glmnet_lambdadiff` and `stat_glmnet_coefdiff`
- [x] 4.6 Cross-validated lambda selection — done: `cross_validated_lambda`
- [x] 4.7 Glmnet correctness tests against R — done: `crates/slide-glmnet/tests/r_parity.rs` consumes committed glmnet fixtures

## 5. SDP Solvers
- [x] 5.1 Equi-correlated solver — done: `create_solve_equi`
- [x] 5.2 SDP solver (DSDP5 port or Clarabel integration) — done: `create_solve_sdp` with Clarabel PSD cones
- [x] 5.3 ASDP solver (approximate, block-diagonal) — done: `create_solve_asdp`
- [x] 5.4 Binary-search feasibility scaling — done: covariance/correlation feasibility enforcement and scaling are implemented in `slide-sdp`
- [x] 5.5 SDP correctness tests against R — done: `crates/slide-sdp/tests/r_parity.rs` consumes committed SDP fixtures

## 6. Knockoff Filter
- [x] 6.1 Knockoff matrix generation (create_gaussian) — done: `create_gaussian` and reusable `KnockoffCache`
- [x] 6.2 Knockoff filter (threshold computation) — done: `knockoff_threshold` and `knockoff_filter`
- [x] 6.3 Voting procedure with SDP caching — done: `knockoff_filter_voting` reuses cached knockoff covariance data
- [x] 6.4 findOptIter refinement — done: `find_opt_iter`
- [x] 6.5 Feature chunking (select_short_freq) — done: `select_short_freq`
- [x] 6.6 Two-stage screening — done: chunked `select_short_freq` performs stage-one screening and stage-two selection
- [~] 6.7 Knockoff integration tests — partial: unit coverage exists and slide-core R fixture exercises selection, but there is no dedicated knockoff R-parity fixture test

## 7. SLIDE Pipeline
- [x] 7.1 Data I/O (CSV read/write, standardization) — done: `slide-data-io` reads/writes matrices, responses, selected outputs, and standardizes inputs
- [x] 7.2 Marginal LF selection (find_standalone_LFs) — done: `find_standalone_lfs`
- [x] 7.3 Interaction detection (find_interaction_LFs, correct_y) — done: `correct_y` and parallel `find_interaction_lfs`
- [x] 7.4 OptimizeSLIDE grid search — done: `optimize_slide`/`optimize_slide_resume` iterate delta/lambda grids
- [x] 7.5 State save/load (checkpoint/resume) — done: JSON checkpoint save/load, input hashes, progress sidecar, and live evaluation sidecar
- [~] 7.6 End-to-end pipeline tests against R — partial: `crates/slide-core/tests/r_parity.rs` checks an HIV fixture, but acceptance remains approximate rather than full R parity

## 8. Cross-Validation & Scoring
- [~] 8.1 Repeated k-fold CV (SLIDEcv) — partial: `slide-cv` now defines serializable final-CV config/result/prediction types, deterministic repeated k-fold split generation, fold-local standardization, marginal/interaction selection on precomputed Z, score aggregation, and focused unit tests; `slide-core` exposes `run_final_slide_cv_for_best` and now has optional `OptimizeSlideConfig::final_cv` plumbing that runs after best selection and persists into checkpointed execution; `cargo test -p slide-cv`, `cargo test -p slide-core --lib`, and `cargo test -p slide-cli` pass
- [~] 8.2 Estimator (logistic/linear regression scoring) — partial: `slide-score` now provides reusable selected-LF/interaction feature construction, linear/logistic prediction, AUC, and Pearson scoring with unit coverage; `slide-core` still has its private legacy scorer
- [ ] 8.3 Control comparison (random LF scoring)
- [~] 8.4 CV correctness tests — partial: added focused `slide-cv` tests for deterministic stratified folds, binary AUC aggregation, continuous correlation aggregation, and empty selected-feature handling; `slide-core` checkpoints/results now carry optional `final_cv` with backward-compatible missing-field tests plus config/checkpoint integration tests that attach and persist final CV only when enabled

## 9. Performance & Polish
- [x] 9.1 Rayon parallelization for knockoff iterations — done: knockoff voting, SDP block solving, LOVE CV repeats, and interaction scans use rayon
- [ ] 9.2 Criterion benchmarks for hot paths
- [ ] 9.3 Memory profiling for large datasets
- [ ] 9.4 Documentation (rustdoc)
- [ ] 9.5 PyO3 bindings (optional, future)
