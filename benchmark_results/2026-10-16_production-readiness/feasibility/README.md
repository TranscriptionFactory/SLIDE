# CUDA and feature-scale preflight

Read-only audit of source `4b32111`, 2026-10-16. No new scaling measurements.
Commands: scoped source inspection of the paths below; official dependency documentation checked on this date. Unlike prior solver studies, this audit evaluates input-feature scaling, not a changed solver tolerance.

## Current support

Python's native route uses CPU NumPy/SciPy, compiled DSDP and Fortran glmnet; optional Torch dependencies do not provide a device-enabled LOVE/knockoff path (`pyproject.toml:66`, `src/loveslide/knockoff/create.py:269`). Rust uses CPU ndarray/OpenBLAS, nalgebra, Rayon and Clarabel (`Cargo.toml:28–36`, `crates/slide-sdp/Cargo.toml:19–24`). Neither audited route currently offers CUDA execution.

Clarabel 0.11.1's [documented features](https://docs.rs/crate/clarabel/0.11.1/features) list BLAS/LAPACK implementations, not a CUDA backend. SCS 3.3.1 documents [GPU indirect and cuDSS direct linear solvers](https://www.cvxgrp.org/scs/linear_solver/index.html), but these require the appropriate build and integration. Their availability is not evidence that omni-SLIDE uses them, or that its SDP and selection results would remain equivalent.

## Dimension-specific limits

| Stage | Dimension and cost | Evidence |
| --- | --- | --- |
| LOVE covariance | Input p; O(np²) work, O(p²) storage | `src/loveslide/love_python/love/love.py:203,250`; `crates/slide-love/src/love.rs:630` |
| FDR threshold | Input p; dense p² values and sorting, O(p² log p) | `src/loveslide/love_python/love/love.py:47–78` |
| Heterogeneous score | Input p; dense p×p product, O(p³) | `src/loveslide/love_python/love/score.py:35`; `crates/slide-love/src/hetero_cv.rs:69` |
| Knockoff sampler | Factor/chunk width d; covariance/factor storage O(d²), dense factorization O(d³), each draw O(nd²) | `crates/slide-knockoff/src/lib.rs:1820–1865` |

No hard 10,000-feature cap was identified. Completion above that scale remains unverified. A single float64 p×p matrix needs 0.8 GB at p=10,000 and 1.152 GB at p=12,000; copies, sorting and concurrent workers increase peak usage. The historical `benchmark_results/2026-10-12_fdr-memory-love-cache/README.md` measures only a Python FDR helper at p=8,000, not complete capacity.

## Minimal changes, ranked

1. Replace Rust's covariance-for-diagonal calculation (`crates/slide-love/src/love.rs:633`) with per-column sums of squares: small effort, reduces unnecessary O(np²) work; verify rounding-sensitive selections.
2. Bound concurrent folds/workers and reuse existing caches: small-to-medium effort, lower memory; preserve split/randomness and cache invalidation semantics.
3. Block exact dense processing: medium effort; preserve BH multiplicities, arithmetic and thresholds. Does not automatically remove all p² storage.
4. Evaluate GPU covariance/heterogeneous matrix products and batched sampling only after profiling: substantial effort; transfers, float64 throughput and CPU solver/fitter stages limit benefit. Device-resident arrays/interfaces are necessary.
5. Solver replacement, low-rank covariance or reduced precision: high statistical/numerical risk; renewed known-truth and R-equivalence validation required. Do not substitute rejected tighter Clarabel feasibility settings.

No new execution, figures or end-to-end speedup estimate exists for this audit. Source-derived complexity is not a memory guarantee.
