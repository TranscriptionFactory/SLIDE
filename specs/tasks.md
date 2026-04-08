## 1. Project Setup
- [ ] 1.1 Initialize Rust workspace with Cargo.toml (slide-rs)
- [ ] 1.2 Set up sub-crates: slide-love, slide-knockoff, slide-core, slide-glmnet, slide-sdp
- [ ] 1.3 Add dependencies: nalgebra, ndarray, rand, rayon, thiserror, anyhow, serde, csv
- [ ] 1.4 Set up CI with cargo test, clippy, fmt
- [ ] 1.5 Create test fixtures from R reference outputs (SSc dataset)

## 2. Linear Algebra Foundation
- [ ] 2.1 Matrix types and conversion (nalgebra ↔ ndarray)
- [ ] 2.2 Eigendecomposition wrapper (matching LAPACK dsyev)
- [ ] 2.3 Cholesky decomposition with PD checks
- [ ] 2.4 Pseudoinverse (pinv) implementation
- [ ] 2.5 Graphical lasso (glasso) implementation for precision matrix

## 3. LOVE Implementation
- [ ] 3.1 Delta cross-validation (CV_delta, KfoldCV_delta)
- [ ] 3.2 Pure variable detection — homogeneous (EstAI, EstC, FindSignPureNode)
- [ ] 3.3 Pure variable detection — heterogeneous (Est_BI_C, Re_Est_Pure) — must match R
- [ ] 3.4 Non-pure row estimation (EstY, EstAJInv, EstAJDant, Dantzig LP)
- [ ] 3.5 Precision matrix estimation (estOmega via glasso)
- [ ] 3.6 Z-matrix computation (MAP estimate)
- [ ] 3.7 Pre-screening (Screen_X) — must match R
- [ ] 3.8 LOVE integration tests against R reference

## 4. Glmnet / Lasso
- [ ] 4.1 Coordinate descent lasso (Gaussian family)
- [ ] 4.2 Coordinate descent lasso (logistic family)
- [ ] 4.3 Lambda path generation (matching R glmnet)
- [ ] 4.4 W-statistic: lambdasmax
- [ ] 4.5 W-statistic: lambdadiff, coefdiff
- [ ] 4.6 Cross-validated lambda selection
- [ ] 4.7 Glmnet correctness tests against R

## 5. SDP Solvers
- [ ] 5.1 Equi-correlated solver
- [ ] 5.2 SDP solver (DSDP5 port or Clarabel integration)
- [ ] 5.3 ASDP solver (approximate, block-diagonal)
- [ ] 5.4 Binary-search feasibility scaling
- [ ] 5.5 SDP correctness tests against R

## 6. Knockoff Filter
- [ ] 6.1 Knockoff matrix generation (create_gaussian)
- [ ] 6.2 Knockoff filter (threshold computation)
- [ ] 6.3 Voting procedure with SDP caching
- [ ] 6.4 findOptIter refinement
- [ ] 6.5 Feature chunking (select_short_freq)
- [ ] 6.6 Two-stage screening
- [ ] 6.7 Knockoff integration tests

## 7. SLIDE Pipeline
- [ ] 7.1 Data I/O (CSV read/write, standardization)
- [ ] 7.2 Marginal LF selection (find_standalone_LFs)
- [ ] 7.3 Interaction detection (find_interaction_LFs, correct_y)
- [ ] 7.4 OptimizeSLIDE grid search
- [ ] 7.5 State save/load (checkpoint/resume)
- [ ] 7.6 End-to-end pipeline tests against R

## 8. Cross-Validation & Scoring
- [ ] 8.1 Repeated k-fold CV (SLIDEcv)
- [ ] 8.2 Estimator (logistic/linear regression scoring)
- [ ] 8.3 Control comparison (random LF scoring)
- [ ] 8.4 CV correctness tests

## 9. Performance & Polish
- [ ] 9.1 Rayon parallelization for knockoff iterations
- [ ] 9.2 Criterion benchmarks for hot paths
- [ ] 9.3 Memory profiling for large datasets
- [ ] 9.4 Documentation (rustdoc)
- [ ] 9.5 PyO3 bindings (optional, future)
