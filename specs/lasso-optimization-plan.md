# Lasso Coordinate Descent Optimization Plan

## Problem Statement

The Rust SLIDE pipeline is **slower than both Python and R** on the same datasets. Profiling shows **99.9% of runtime** is in the knockoff voting loop, which runs lasso coordinate descent:

| Stage | Time (HIV dataset, 36 features) | % |
|-------|----------------------------------|---|
| LOVE | 0.01s | 0% |
| Z-map | 0.00s | 0% |
| Knockoff marginal | 55.73s | 14% |
| Knockoff interactions | 338.87s | 86% |
| **TOTAL** | **394.60s** | |

The bottleneck is `crates/slide-glmnet/src/lib.rs` — a hand-rolled lasso coordinate descent implementation called `niter` × `nlambda` times per evaluation, multiplied again by `|sig_lfs|` for interactions.

## Current Implementation Analysis

**File:** `crates/slide-glmnet/src/lib.rs` (~1100 lines)

### What it implements:
1. **Gaussian lasso path** (`gaussian_lasso_path`): Coordinate descent along a geometric lambda grid (default 250 lambdas)
2. **Binomial lasso path** (`binomial_lasso_path`): IRLS + weighted coordinate descent
3. **Knockoff statistics**: `stat_glmnet_lambdasmax_prebuilt`, `stat_glmnet_lambdadiff_prebuilt`, `stat_glmnet_coefdiff_prebuilt` — all call the lasso path on an augmented `[X | X_tilde]` matrix
4. **Cross-validated lambda selection** (`cross_validated_lambda`)

### How it's called in the hot path:
```
knockoff_filter_voting()                     # slide-knockoff/src/lib.rs:267
  ├── (0..niter).into_par_iter()             # 100-500 parallel iterations
  │   └── compute_statistic_prebuilt()       # per iteration
  │       └── gaussian_lasso_path()          # fits 250-lambda path on [X|X̃] (2p columns)
  │           └── coordinate_descent_step()  # per lambda point
  │               └── sweep_coordinates()    # O(n×p) per sweep, multiple sweeps until convergence
```

For the HIV dataset (n=67, p=36 → augmented is 67×72):
- Marginal: 100 iterations × 1 lasso path each = **100 lasso paths**
- Interactions: 100 iterations × ~6 interaction tests × lasso path each ≈ **600 lasso paths**
- Total: ~700 lasso paths, each fitting up to 250 lambda points

### Current optimizations already applied (Phases 1-3):
- Column-major (Fortran) layout for contiguous column access
- Active set strategy (only sweep non-zero coefficients)
- Early stopping when deviance ratio > 0.9999 or model saturated
- IRLS restructuring for binomial (fewer sigmoid evaluations)
- Warm-starting along lambda path (coefficients carry forward)
- Pre-computed column squared norms

### What's missing vs R glmnet:
R's `glmnet` is a **Fortran 90 library** (by Friedman, Hastie, Tibshirani) with decades of optimization:
- **Covariance updates**: Uses `X^T X` precomputed (naïve update) for small n — avoids O(n) dot products
- **Strong rules**: Skips coordinates guaranteed to remain zero, reducing per-lambda cost dramatically
- **BLAS dot products**: Inner loop uses BLAS ddot/daxpy
- **Memory layout**: Fortran-native column-major with no abstraction overhead

## Decision: Crate vs Hand-Rolled

### Option A: Use R's glmnet via FFI (link to Fortran)

**Pros:**
- Exact numerical parity with R — eliminates all glmnet-related divergence
- Battle-tested, maximally optimized (30+ years of development)
- Handles edge cases (near-singular, zero-variance columns) robustly

**Cons:**
- Requires Fortran compiler and R glmnet source in the build chain
- FFI boundary adds complexity (data marshaling, error handling)
- Links the project to R's ecosystem — defeats the "pure Rust" goal
- Hard to cross-compile

**Verdict:** Reject. Defeats the purpose of a Rust port.

### Option B: Use a Rust lasso crate

**Candidates to evaluate:**
- `linfa-elasticnet` (part of `linfa` ML framework) — implements coordinate descent elastic net
- `smartcore` — has lasso/elastic net, but API-heavy and less maintained
- `argmin` — optimization framework, could implement lasso but overkill
- `glmnet-rs` — check if this exists (unlikely to be mature)

**Evaluation criteria:**
1. Does it support the full lasso **path** (not just single-lambda fit)?
2. Does it support the `lambdasmax` statistic (entry-lambda per coefficient)?
3. Can it warm-start along the path?
4. Does it support both Gaussian and binomial families?
5. Is it actively maintained?
6. Performance: does it use BLAS, strong rules, covariance updates?

**Action:** Search crates.io for each. If `linfa-elasticnet` supports lasso paths with warm start, benchmark it against the current implementation. If no crate provides the full path API, keep hand-rolled.

**Likely verdict:** No existing Rust crate provides the specialized glmnet-style path interface needed. The knockoff statistic requires the full coefficient path along a lambda grid, not just a single best-lambda fit.

### Option C: Optimize the current hand-rolled implementation (RECOMMENDED)

This is the most pragmatic path. The current implementation is correct and well-structured but is missing key algorithmic optimizations that glmnet uses.

**Priority optimizations (ordered by expected impact):**

#### C.1: Strong rules for coordinate screening (HIGH IMPACT)
Skip coordinates where `|X_j^T r| < 2λ_k - λ_{k-1}` — they're guaranteed to stay zero at `λ_k` if they were zero at `λ_{k-1}`. For the typical knockoff augmented matrix (2p columns, most coefficients zero at most lambda values), this can skip 80-95% of coordinates.

**Implementation:**
- After each lambda step, compute the strong rule threshold
- Mark coordinates violating the threshold as "screened out"
- After sweeping active + strong-rule survivors, check KKT on screened-out coordinates
- If any KKT violations, add them back and re-sweep

**Expected speedup:** 3-10x on the lambda path (fewer coordinates touched per lambda)

**Files:** `crates/slide-glmnet/src/lib.rs` — `coordinate_descent_step()` and `gaussian_lasso_path()`

#### C.2: Covariance/naive update for n < p (MEDIUM IMPACT)
When `n < p` (the knockoff case: augmented matrix has 2p columns), precompute the Gram matrix `X^T X` once (O(n p²)) and use it for gradient computation instead of `X_j^T r` which is O(n) per coordinate. The gradient becomes `(X^T X)_j β - (X^T y)_j` with O(p) update per coordinate change.

**When beneficial:** `n < 2p` (which is true for the augmented matrix in most SLIDE use cases)
**Trade-off:** Requires O(p²) memory for the Gram matrix

**Implementation:**
- Precompute `gram = X^T X / n` and `Xty = X^T y / n` once
- In `sweep_coordinates`, replace `column.dot(residual)` with `gram.row(j).dot(beta) - Xty[j]`
- Update is O(p) instead of O(n) but avoids touching the full X matrix

**Expected speedup:** 1.5-3x when n << p

#### C.3: BLAS acceleration for inner dot products (MEDIUM IMPACT)
The `column.dot(residual)` and `residual.scaled_add(-new_beta, &column)` calls in `sweep_coordinates` are the innermost hot loop. These should dispatch to BLAS (`ddot` + `daxpy`) automatically via ndarray's BLAS feature, but only if columns are contiguous in memory.

**Verify:**
- Is `x` in column-major order? (Yes — `PreparedData` converts to `.f()` layout)
- Is `x.column(j)` returning a contiguous slice? (Should be, given F-layout)
- Is ndarray actually dispatching to BLAS? Check with `RUST_LOG=ndarray=trace`

**If not dispatching to BLAS:**
- Use explicit `unsafe` BLAS calls via `blas-sys` or `openblas-src`
- Or ensure ndarray's `blas` feature is propagated correctly

**Expected speedup:** 1.5-2x on the inner loop (SIMD + cache optimization)

#### C.4: Reduce allocation in the knockoff loop (LOW-MEDIUM IMPACT)
Each knockoff iteration calls `gaussian_lasso_path` which allocates:
- `PreparedData` (copies and standardizes the augmented matrix)
- `beta_path` (p × nlambda output)
- `residual` vector

For the knockoff use case, `X` columns don't change between iterations (only `X_tilde` columns change, and the augmented matrix is pre-standardized). Consider:
- Caching `PreparedData` across iterations (X portion is invariant)
- Pre-allocating beta_path and reusing
- Making `coordinate_descent_step` take pre-allocated buffers

**Expected speedup:** 1.2-1.5x (less allocation pressure, better cache locality)

#### C.5: Reduce nlambda for knockoff statistics (LOW IMPACT, EASY)
The knockoff `lambdasmax` statistic only needs the **entry lambda** per coefficient — the first lambda at which each coefficient becomes non-zero. This doesn't require fitting the full path to completion. Options:
- Use a coarser lambda grid (50 instead of 250) — the entry point is insensitive to grid resolution
- Stop the path early once all 2p entry lambdas have been recorded (no need to continue to small lambdas where many coefficients are non-zero)

**Expected speedup:** 2-5x for `lambdasmax` statistic specifically

#### C.6: Batch lasso paths across knockoff iterations (SPECULATIVE)
Instead of running 100 independent lasso paths, batch them: the X portion is shared, only X_tilde varies. A batched coordinate descent could share the Gram matrix `X^T X` across iterations. This is architecturally complex but could yield large gains.

**Expected speedup:** Unknown, requires prototyping

## Recommended Execution Order

```
C.1 (strong rules)      — highest impact, algorithmic improvement     ✅ DONE
  ↓
C.5 (reduce nlambda)    — easy, reduces work proportionally           ✅ DONE
  ↓
C.2 (covariance update) — medium impact, n < p case                   ✅ DONE (Gram-based path when n < 2p)
  ↓
C.3 (BLAS verification) — may already be working, just verify         ✅ VERIFIED (ndarray blas feature + openblas-src + F-layout)
  ↓
C.4 (reduce allocation) — incremental improvement                     ⏭️ SKIPPED (allocations are small, <100KB per iteration)
  ↓
C.6 (batch paths)       — speculative, only if still needed           ⏭️ DEFERRED
```

### Implementation notes

**C.1 (strong rules):** Implemented for both Gaussian and binomial paths.
- Before each lambda step, screen out coordinates where `|X_j^T r / n| < 2λ_k - λ_{k-1}`
- After convergence on the strong set, check KKT on excluded coordinates
- If KKT violated, add violators and re-run (loop until clean)
- Also: residual maintained across lambda steps (avoids O(n×p) recomputation per step)

**C.5 (reduce nlambda + early stopping):**
- `stat_glmnet_lambdasmax_prebuilt` uses `min(nlambda, 100)` instead of 250
- `early_stop_all_entered` flag: path stops once every coefficient has been non-zero at least once
- Applied to both Gaussian and binomial paths

**C.2 (done):** Gram matrix (covariance update) implemented for Gaussian lasso when n < 2p.
Precomputes G = X^T X / n via BLAS dgemm (one-time O(n·p²)) and Xy = X^T y / n via dgemv.
Inner CD loop maintains grad_cache = Xy - G·β; gradient reads are O(1), updates are O(p)
axpy on grad_cache. Strong rules and KKT checks use grad_cache directly (no residual needed).
Deviance ratio computed via quadratic form: RSS/n = y^Ty/n - 2·β^T·Xy + β^T·G·β.
Falls back to residual-based path when n ≥ 2p. Key win for SSc dataset (n=24, 2p=48-344):
all inner loop operations shift from n=24 (below BLAS threshold) to p≥48 (above threshold).

**C.3 (verified):** ndarray `blas` feature is enabled in workspace Cargo.toml, OpenBLAS is the
backend (`openblas-src`), and `PreparedData` converts to Fortran order (`.f()`), so
`x.column(j)` returns contiguous slices that dispatch to BLAS `ddot`/`daxpy`.

**C.4 (skipped):** Per-iteration allocations are ~100KB (PreparedData copy + beta_path + residual).
With 100-700 iterations, total allocation churn is <100MB — not a bottleneck.

## Verification at each step

1. `cargo test -p slide-glmnet` — existing tests pass ✅ (all 20 pass including R parity)
2. `cargo test --workspace` — no regressions ✅
3. Re-run HIV profiling: `target/release/slide run --config ... 2>&1 | grep timing`
4. Compare lasso path output (beta, lambda) against reference to ensure correctness

## Files modified

| File | Change |
|------|--------|
| `crates/slide-glmnet/src/lib.rs` | Strong rules, reduced nlambda, early stopping, residual persistence, Gram-matrix CD path |

## Success criteria

- Rust pipeline ≤ Python time on HIV dataset (currently ~394s Rust vs ~50s Python)
- Rust pipeline ≤ 2x R time on SSc dataset at delta=0.1
- Z-matrix correlation ≥ 0.95 with Python reference (correctness maintained)
- All existing tests pass ✅
