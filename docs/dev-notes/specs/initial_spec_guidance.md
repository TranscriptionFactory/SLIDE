# slide-rs Specification Index

This document is the entry point for the `slide-rs` Rust port specifications.
All authoritative design and correctness details live in the documents listed below.

## Architecture & Rationale

- **[design.md](design.md)** — Top-level design document: crate structure, matrix library choices, SDP/lasso solver decisions, RNG strategy, testing strategy, and risk assessment.

## Key Architectural Decisions

| Decision | Choice | Rationale |
|---|---|---|
| Implementation language | Pure Rust (no FFI to C/Fortran) | Eliminates build-time dependencies on DSDP5 and Fortran glmnet |
| SDP solver | Clarabel (interior-point, conic) | Actively maintained, MIT-licensed, no C dependencies |
| Lasso solver | Native coordinate descent matching glmnet conventions | Knockoff W statistics require identical lambda grid and convergence criteria |
| Matrix libraries | `nalgebra` (decompositions) + `ndarray` (slicing/broadcasting) | LAPACK parity for eigendecomposition; NumPy-like ergonomics for data wrangling |
| RNG | MT19937 matching R's `set.seed()` protocol | Enables deterministic fixture comparison against R reference |
| Parallelism | Rayon (work-stealing thread pool) | Zero-copy sharing of knockoff cache; no serialization overhead |

## Sub-Specifications

Each sub-spec defines the behavioral contract for one pipeline stage:

| Spec | Covers |
|---|---|
| [love/spec.md](specs/love/spec.md) | LOVE decomposition: covariance/correlation, FDR thresholding, pure variable estimation, loading matrix recovery |
| [sdp-solvers/spec.md](specs/sdp-solvers/spec.md) | Equi-correlated, full SDP, and approximate SDP solvers; binary-search feasibility scaling |
| [knockoffs/spec.md](specs/knockoffs/spec.md) | Knockoff matrix generation, W statistics (lasso max-lambda), knockoff filter, voting, `findOptIter`, feature chunking |
| [glmnet/spec.md](specs/glmnet/spec.md) | Coordinate-descent lasso: lambda grid, warm starts, active-set cycling, standardization |
| [cv/spec.md](specs/cv/spec.md) | Cross-validation wrapper |
| [scoring/spec.md](specs/scoring/spec.md) | Held-out prediction metrics |
| [data-io/spec.md](specs/data-io/spec.md) | Matrix I/O and model serialization |
| [slide-core/spec.md](specs/slide-core/spec.md) | Pipeline orchestrator, interaction detection |
| [correctness/spec.md](specs/correctness/spec.md) | Per-component numerical tolerances and acceptance criteria (authoritative source for all tolerance thresholds) |

## History

This file replaces an earlier brainstorming document (`initial_spec_guidance.md`, ~900 lines) that contained exploratory Rust code sketches and ecosystem mapping notes. All algorithmic content from that document has been absorbed into the sub-specs above, and implementation-specific Rust patterns (CSC matrix construction for Clarabel, Rayon map-reduce for knockoff voting) are captured in the relevant sub-spec's "Rust Implementation Notes" sections.
