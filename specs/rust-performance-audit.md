# Rust SLIDE Performance Audit

## Summary

The Rust implementation (`slide-rs`) is **2-4x slower than R** and **1.5x slower
than Python** on the SSc benchmark (4 threads, 500 knockoff iterations). This is
the opposite of what a compiled implementation should deliver. Additionally, the
Z-matrix correlation between Rust and R/Python is 0.0096 — essentially
uncorrelated — indicating a **correctness issue** that may also contribute to
the performance gap (wrong code paths, different problem sizes, etc.).

### Benchmark Numbers (wall seconds, 4 threads)

| delta | lambda | Python | R | Rust | Rust/R |
|------:|-------:|-------:|----:|-----:|-------:|
| 0.01 | 0.1 | 1013 | 700 | 2477 | 3.5x |
| 0.01 | 1.0 | 985 | 672 | 1748 | 2.6x |
| 0.1 | 0.1 | 1052 | 236 | 1555 | 6.6x |
| 0.1 | 1.0 | 333 | 367 | 1396 | 3.8x |

---

## Critical Issues (each likely 2-5x slowdown contribution)

### 1. Cholesky pre-validates with full eigendecomposition

**File:** `crates/slide-types/src/lib.rs:185-203`

`Matrix::cholesky()` calls `self.minimum_eigenvalue()` before attempting
Cholesky. `minimum_eigenvalue()` calls `symmetric_eigendecomposition()` — a full
O(n³) LAPACK `dsyev` — just to check that the smallest eigenvalue is positive.
Then it calls `LapackCholesky::new()` — another O(n³) factorization.

**Impact:** Every Cholesky call costs **2x** what it should. Cholesky is called
in knockoff cache construction (`compute_l_trans`, `solve_sigma_inv_diag_s`),
in LOVE (`est_c`, `est_omega`), and in SDP feasibility. The `cholesky()` method
is also called in a retry loop (`regularized_cholesky`) which compounds the cost.

**Fix:** Attempt Cholesky first; if it fails (LAPACK returns error), then either
report not-PD or retry with regularization. R does exactly this — `chol()` tries
and fails fast. Remove the eigendecomposition pre-check entirely.

Additionally, `cholesky()` calls `to_nalgebra()` twice: once for the eigendecomp
PD check, once for the actual factorization. Each `to_nalgebra()` is a full
matrix copy. With the fix above, this drops to a single copy.

### 2. SDP feasibility binary search calls eigendecomposition up to 128 times

**File:** `crates/slide-sdp/src/lib.rs:300-368`

`enforce_correlation_feasibility` and `enforce_covariance_feasibility` each do
a 64-step binary search. Each step calls `minimum_eigenvalue_of_shifted_matrix`,
which:
1. Converts to nalgebra (copies the matrix)
2. Applies the diagonal shift (O(n²) loop)
3. Calls `minimum_eigenvalue` → `symmetric_eigendecomposition` (full O(n³) LAPACK)
4. Validates square + symmetric (redundant, already checked)

For a block of size p=100 in ASDP, each eigendecomp is ~10⁶ flops. With 64
steps × 2 feasibility checks = 128 eigendecompositions per SDP block solve.

**Impact:** For ASDP with multiple blocks, this is the dominant cost of the SDP
stage. R's `knockoff.filter` uses a single eigendecomposition to compute the
minimum eigenvalue and scales once — not a binary search.

**Fix:**
- Compute the minimum eigenvalue of `2Σ - diag(s)` once. If negative, scale
  `s` by `γ = λ_min(Σ) / (λ_min(Σ) - λ_min_deficit)` in closed form. No
  binary search needed.
- At minimum, skip validation (the matrix is known square and symmetric),
  use `dsyev` directly instead of going through the full
  `symmetric_eigendecomposition` wrapper, and reuse the shifted matrix buffer
  instead of allocating fresh each step.

### 3. Binomial CD recomputes sigmoid for entire sample on every coordinate

**File:** `crates/slide-glmnet/src/lib.rs` (sweep_binomial_coordinates, ~line 903-962)

For each coordinate `j` in the active set, `sweep_binomial_coordinates`
recomputes `eta.mapv(sigmoid)` — an O(n) computation involving `exp()` — and
allocates a fresh `weights` array. This runs for every active coordinate at every
CD iteration for every lambda value.

For 2p=344 coordinates, 250 lambda values, ~5 CD iterations per lambda:
344 × 250 × 5 = 430,000 full-vector sigmoid evaluations. Each is O(n=300) with
an `exp()` per element.

**Impact:** This is the dominant cost in the knockoff voting inner loop (100
iterations × the above). R's `glmnet` uses IRLS: it computes weights once per
Newton outer iteration, then runs a full CD convergence cycle with those fixed
weights. The per-coordinate cost in R is just a dot product + AXPY.

**Fix:** Restructure to match glmnet's IRLS: compute probabilities/weights once
per outer Newton step, form working response `z = η + (y - p)/w`, then run
coordinate descent on the weighted least-squares problem with fixed weights.
This reduces sigmoid evaluations from O(np × CD_iters) to O(n × Newton_iters).

### 4. Knockoff voting loop clones the augmented matrix every iteration

**File:** `crates/slide-knockoff/src/lib.rs:274-296`

The voting loop pre-allocates a per-thread buffer `buf` and copies the X columns
into it. But on line 291: `let augmented = Matrix::from_ndarray(buf.clone())` —
this clones the entire `(n, 2p)` buffer just to wrap it in the `Matrix` newtype.

For n=300, p=172: each clone is 300 × 344 × 8 = 825 KB. With 100 iterations
across 4 threads, that's ~80 MB of wasted allocation per voting call.

**Impact:** The clone exists only because `compute_statistic_prebuilt` takes
`&Matrix` instead of `&Array2`. The actual computation doesn't need ownership.

**Fix:** Change `compute_statistic_prebuilt` (and the lasso path functions it
calls) to accept `&Array2<Scalar>` or make `Matrix` wrap a `Cow` so
`from_ndarray` can borrow. Alternatively, wrap a reference:
`Matrix::from_ndarray_ref(buf.view())` — but this requires a lifetime-bearing
`Matrix` type. The simplest fix is to clone only once outside the loop and
mutate in place, since the `X` portion doesn't change.

Actually, the simplest fix: wrap the view without cloning by having the stat
functions accept `&Array2` directly, or by creating the `Matrix` wrapper once
and mutating its `X_tilde` portion in place.

---

## High-Impact Issues (each ~1.5-2x slowdown)

### 5. No early stopping on the lasso path

**File:** `crates/slide-glmnet/src/lib.rs:121-137`

The Gaussian and binomial lasso paths always walk all `nlambda=250` points. R's
`glmnet` stops early when:
- The deviance ratio exceeds 0.999 (all variance explained)
- The maximum number of non-zero coefficients exceeds `n` (overfit)
- The fractional deviance change between lambdas is below `fdev=1e-5`

For the knockoff statistic, only the lambda at which each variable enters matters
(`lambdasmax`). Once all 2p variables have entered, the remaining lambda values
are wasted computation.

**Fix:** Add early termination criteria matching glmnet's `fdev`, `dfmax`, and
`devmax` defaults. For `lambdasmax`, stop as soon as all coefficient rows are
non-zero.

### 6. Matrix newtype conversion overhead in hot paths

**File:** `crates/slide-types/src/lib.rs:96-149`

`Matrix::to_nalgebra()` copies all data into a new `DMatrix` by iterating
row-by-row. `Matrix::from_nalgebra()` copies back by iterating element-wise.
Neither uses `memcpy`-friendly layouts.

These conversions happen on every call to `cholesky()`, `pseudoinverse()`,
`symmetric_eigendecomposition()`, and `solve_matrix()` / `solve_vector()`.
In the SDP binary search alone, this means 128 matrix copies per block.

**Fix:**
- `to_nalgebra()`: nalgebra `DMatrix` is column-major. Use
  `DMatrix::from_column_slice` with a column-order copy or, if ndarray is
  already column-major (it is in the CD path!), use a direct memcpy.
- `from_nalgebra()`: similarly, use `as_slice()` for column-major data and
  construct ndarray directly.
- Better yet: for decomposition-only paths, keep data in nalgebra format and
  avoid the round-trip entirely.

### 7. Nested rayon parallelism causes thread contention

**File:** `crates/slide-knockoff/src/lib.rs:360-373` and `:274-296`

`select_short_freq` parallelizes chunks with `par_iter`. Each chunk calls
`knockoff_filter_voting`, which also uses `par_iter` over iterations. This
creates nested rayon parallelism.

Rayon handles nested parallelism reasonably, but when the outer parallelism
already saturates all cores, inner parallelism adds scheduling overhead and
cache thrashing without benefit.

Additionally, `find_interaction_lfs` in `slide-core` parallelizes across
marginal LFs, each calling `select_short_freq` which may itself parallelize.
This creates three levels of nesting.

**Fix:**
- For `select_short_freq`: run chunks sequentially and let `knockoff_filter_voting`
  handle the parallelism (it has more work to distribute).
- Or use rayon's `join` / scoping to limit concurrency depth.
- Consider making the inner iteration loop sequential when called from an
  already-parallel context (pass a flag or use `iter()` vs `par_iter()`).

---

## Medium-Impact Issues

### 8. Redundant symmetry/square validation on every decomposition call

Every call to `symmetric_eigendecomposition()`, `cholesky()`, etc. validates the
matrix is square and symmetric with O(n²) element comparisons. In hot loops
(SDP feasibility), the same matrix is validated hundreds of times. Separate
validation from computation — validate once at the API boundary, then use
unchecked internal methods.

### 9. Box-Muller normal sampling instead of Ziggurat

`sample_standard_normal` (slide-knockoff:761-777) uses a hand-rolled Box-Muller
transform. The `rand_distr::StandardNormal` distribution uses the Ziggurat
algorithm, which is ~40% faster (no `ln` or trig calls on the fast path).

For 100 iterations × n × p normal draws per iteration, this adds up.

### 10. KnockoffCache stores unnecessary clones

`build_knockoff_cache` (slide-knockoff:406-423) clones `mu`, `sigma`, and
`diag_s` into the cache struct. Only `mu_k` and `l_trans` are needed for
sampling. The clones waste memory and cache lines.

### 11. Interaction term computation allocates per-pair

In `find_interaction_lfs` (slide-core:472-477), each interaction column is
computed as `z.column(a) * z.column(b)` and pushed into a Vec. This allocates
a fresh Array1 per pair. Pre-allocate the interaction matrix.

---

## Correctness Concern: Z matrix uncorrelated with R/Python

The benchmark comparison shows:
```
| python | rust | 0.1 | 1.0 | Z overall r=0.0096 | Z median col r=0.0277 |
```

This means the Rust Z-matrix is essentially uncorrelated with the Python/R
reference. This is almost certainly a bug in LOVE or in the RNG/seed handling
that causes Rust to produce completely different factor structure.

**This must be investigated before performance optimization**, because:
1. If the Z matrix is wrong, downstream knockoff/interaction times are measuring
   the wrong computation (different K, different feature dimensions).
2. A correctness fix might change the performance profile entirely.

Possible causes:
- Eigenvector sign/ordering mismatch in LOVE eigendecomposition
- Different Cholesky/SVD branch taken due to the PD pre-check
- RNG (StdRng vs MT19937) producing different normal draws
- Different LOVE delta selected due to CV differences

---

## Implementation Plan

### Phase 0: Correctness (must precede performance work)

1. **Diagnose Z-matrix divergence**
   - Run LOVE on a small fixture (4×4 or 6×4) and compare intermediate outputs
     (eigenvalues, eigenvectors, pure variables, A, C, gamma) step by step
     against R/Python
   - Check if the sign normalization in `symmetric_eigendecomposition` produces
     the same convention as R's `eigen()`
   - Verify LOVE config parameters (delta, lambda, pure_homo) match between
     backends

### Phase 1: Critical fixes (expected 4-8x cumulative speedup)

2. **Remove eigendecomposition from `cholesky()`** — attempt Cholesky directly,
   fall back on failure
   - Files: `crates/slide-types/src/lib.rs`
   - Verify: `cargo test -p slide-types`, `cargo test -p slide-sdp`

3. **Replace SDP feasibility binary search with closed-form scaling**
   - Files: `crates/slide-sdp/src/lib.rs`
   - Compute `λ_min(2Σ - diag(s))` once, scale `s` analytically
   - Verify: `cargo test -p slide-sdp`

4. **Restructure binomial CD to IRLS** (compute weights once per Newton step)
   - Files: `crates/slide-glmnet/src/lib.rs`
   - Match R glmnet's outer-Newton / inner-CD structure
   - Verify: `cargo test -p slide-glmnet`, R parity tests

5. **Eliminate augmented matrix clone in voting loop**
   - Files: `crates/slide-knockoff/src/lib.rs`, `crates/slide-glmnet/src/lib.rs`
   - Change `compute_statistic_prebuilt` to accept `&Array2` or use zero-copy
     `Matrix` wrapping
   - Verify: `cargo test -p slide-knockoff`

### Phase 2: High-impact fixes (expected 1.5-2x additional)

6. **Add early stopping to lasso path** — deviance ratio, max active set, fdev
   - Files: `crates/slide-glmnet/src/lib.rs`

7. **Optimize `to_nalgebra()` / `from_nalgebra()` conversions** — use
   column-major memcpy when layouts match
   - Files: `crates/slide-types/src/lib.rs`

8. **Flatten nested rayon parallelism** — run chunks sequentially in
   `select_short_freq`, keep only the iteration-level parallelism
   - Files: `crates/slide-knockoff/src/lib.rs`

### Phase 3: Polish (expected 10-20% additional)

9. Remove redundant validation in inner loops
10. Switch normal sampling to Ziggurat
11. Trim `KnockoffCache` to only store `mu_k` and `l_trans`
12. Pre-allocate interaction term matrices

### Expected total improvement

Conservative estimate: **5-10x** from Phase 1 alone, bringing Rust to parity
with or faster than R/Python. Phase 2 should push to **1.5-2x faster than R**.

### Verification at each step

- `cargo test --workspace` must pass after each change
- Re-run the SSc benchmark (delta=0.1, lambda=1.0 is fastest) after each phase
- Compare timing regression with the baseline numbers above
- After Phase 0, verify Z correlation with R/Python is >0.95
