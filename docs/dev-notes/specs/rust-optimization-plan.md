# Rust SLIDE Optimization Plan

Detailed implementation plan for the issues identified in `rust-performance-audit.md`.
Each step references the Python implementation in `src/loveslide/` as the
behavioral specification.

**Target:** Rust faster than Python (≥2x), at parity with or faster than R.
**Baseline:** Rust is 2-6x slower than R, 1.5x slower than Python.

---

## Phase 0: Correctness — Z-matrix divergence ✅ COMPLETED

**Root cause identified and fixed.** Two bugs found:

### Bug 1: F-layout memory-order mismatch in `pure_hetero.rs` (Steps 0.1-0.2)

In `left_inverse()` and `schur_complement_diagonal()`, `.t().to_owned()` creates
an F-layout (column-major) array. `.iter()` yields elements in memory order
(column-major), but `DMatrix::from_row_slice` interprets them as row-major.
Result: the DMatrix was the **transpose** of what was intended.

**Fix:** Replaced `DMatrix::from_row_slice(... .iter() ...)` with the safe
`Matrix::from_ndarray(...).to_nalgebra()` path, which correctly handles any
memory layout.

**Affected:** `b_left_inv` → cascaded to C_hat → A_hat → gamma → Z.

### Bug 2: Z-matrix used `pinv(C)` instead of `inv(C)`, and hard-errored on zero gamma

- Python uses `np.linalg.inv(C_hat)` — regular inverse.
  Rust used `c.pseudoinverse()` — Moore-Penrose pseudoinverse.
  **Fix:** Use `try_inverse()` with `pseudoinverse()` fallback.

- Python replaces gamma==0 with 1e-10. Rust returned `Err(NonPositiveGamma)`.
  **Fix:** Clamp zero gamma to 1e-10 (matching Python behavior).

**Verification:**
- `cargo test -p slide-love` — all 49 tests pass
- `cargo clippy -p slide-love` — no new warnings

---

## Phase 1: Critical performance fixes ✅ COMPLETED

Each step is independently committable and testable.

### Step 1.1: Remove eigendecomposition pre-check from `cholesky()` ✅

**Commit:** `c5daba2` — Removed eigendecomp + symmetry pre-check. LAPACK
cholesky now fails naturally for non-PD matrices (returns `NotPositiveDefinite`
with `NAN` eigenvalue). Removed orphaned `minimum_eigenvalue()` and
`POSITIVE_DEFINITE_TOLERANCE`. All 16 slide-types tests pass.

**Audit issue:** #1 — Cholesky costs 2x due to eigendecomp PD check.

**File:** `crates/slide-types/src/lib.rs:185-203`

**Python reference:** `create.py:347` — `linalg.cholesky()` in a try/except.
No pre-validation. On failure, adds scaled `eps * I` and retries.

**Change:**

```rust
// Before (current):
pub fn cholesky(&self, matrix_name: &str) -> Result<...> {
    self.validate_square("cholesky")?;
    self.validate_symmetric("cholesky")?;
    let min_eigenvalue = self.minimum_eigenvalue()?;  // ← REMOVE
    if min_eigenvalue <= POSITIVE_DEFINITE_TOLERANCE {  // ← REMOVE
        return Err(...);                                // ← REMOVE
    }                                                   // ← REMOVE
    let factor = LapackCholesky::new(self.to_nalgebra()).ok_or_else(|| ...)?;
    Ok(CholeskyDecomposition { factor })
}

// After:
pub fn cholesky(&self, matrix_name: &str) -> Result<...> {
    self.validate_square("cholesky")?;
    // Skip symmetric check in hot paths (caller responsibility)
    let factor = LapackCholesky::new(self.to_nalgebra()).ok_or_else(|| {
        LinearAlgebraError::NotPositiveDefinite {
            matrix_name: matrix_name.to_owned(),
            min_eigenvalue: f64::NAN, // actual eigenvalue unknown
        }
    })?;
    Ok(CholeskyDecomposition { factor })
}
```

**Also update `regularized_cholesky`** in `slide-knockoff/src/lib.rs:594-615`:
currently calls `Matrix::from_ndarray(symmetrize(matrix.clone()))` then
`candidate.cholesky()` — each call does eigendecomp + Cholesky + 2 matrix copies.
After the fix, each retry is just one Cholesky + one copy.

**Python pattern for reference** (`create.py:346-364`):
```python
try:
    L = linalg.cholesky(Sigma_k, lower=True)
except linalg.LinAlgError:
    max_diag = np.max(np.diag(Sigma_k))
    eps = 1e-10 * max(1.0, max_diag)
    try:
        L = linalg.cholesky(Sigma_k + eps * np.eye(p), lower=True)
    except linalg.LinAlgError:
        while eps < 1:
            eps *= 10
            try:
                L = linalg.cholesky(Sigma_k + eps * np.eye(p), lower=True)
                break
            except linalg.LinAlgError:
                continue
```

**Verify:** `cargo test -p slide-types -p slide-knockoff -p slide-sdp -p slide-love`

**Impact:** ~2x speedup on all Cholesky-dependent paths (SDP, knockoff cache,
LOVE omega estimation).

### Step 1.2: Replace SDP feasibility binary search with closed-form scaling ✅

**Commit:** `c656a4d` — Replaced 64-step binary search with closed-form
γ = (λ_min(2Σ) - floor) / (λ_min(2Σ) - λ_min(2Σ - diag(s))). Uses identity
λ_min(2Σ) = 2·λ_min(Σ) to avoid second eigendecomp. Unified
`enforce_correlation_feasibility` and `enforce_covariance_feasibility` into
single `enforce_feasibility` function. R parity tests updated to verify
feasibility invariants (solution is valid but numerically different from
binary search). All 7 slide-sdp tests pass.

**Audit issue:** #2 — 128 eigendecompositions per SDP solve.

**File:** `crates/slide-sdp/src/lib.rs:300-368`

**Python reference:** `knockoff/solve.py` — the Python SDP solver (DSDP via C
extension) returns a feasible solution directly. The R knockoffs package
(`create.solve_sdp`) computes `λ_min(2Σ - diag(s))` once and scales `s` by
`min(1, 2*λ_min(Σ))` if needed — no binary search.

**Change:** Replace `enforce_correlation_feasibility` and
`enforce_covariance_feasibility` with:

```rust
fn enforce_feasibility(matrix: &Matrix, solution: &mut [Scalar]) -> Result<(), SdpError> {
    // Build the shifted matrix 2*matrix - diag(solution) once
    let shifted = build_shifted_matrix(matrix, solution);
    let min_eig = minimum_eigenvalue(&shifted)?;

    if min_eig >= FEASIBILITY_FLOOR {
        return Ok(());
    }

    // Closed-form scale: γ such that λ_min(2Σ - γ*diag(s)) ≥ floor
    // λ_min(2Σ - γ*diag(s)) = (1-γ)*λ_min(2Σ) + γ*λ_min(2Σ - diag(s))
    // For linear interpolation: γ = (λ_min(2Σ) - floor) / (λ_min(2Σ) - min_eig)
    let unshifted_min = minimum_eigenvalue(
        &Matrix::from_ndarray(matrix.as_ndarray() * 2.0)
    )?;
    let gamma = if unshifted_min > min_eig {
        ((unshifted_min - FEASIBILITY_FLOOR) / (unshifted_min - min_eig))
            .clamp(0.0, 1.0)
    } else {
        0.0
    };

    for value in solution.iter_mut() {
        *value *= gamma;
    }
    Ok(())
}
```

This reduces eigendecompositions from 128 to **2** (one for the shifted matrix,
one for the unshifted 2Σ — though 2Σ's eigenvalues are just 2× Σ's, so we can
use the already-computed Σ eigenvalues, making it **1** eigendecomp total).

**Also optimize `minimum_eigenvalue_of_shifted_matrix`:**
- Currently converts to nalgebra, applies shift, converts back, then calls
  `minimum_eigenvalue` which does eigendecomp via `symmetric_eigendecomposition`
  (which validates square+symmetric, sorts, sign-normalizes — all unnecessary
  when we only need the minimum eigenvalue).
- Replace with a direct `dsyev` call on the shifted nalgebra matrix without
  the round-trip.

**Verify:** `cargo test -p slide-sdp` — existing tests check feasibility
invariants (λ_min ≥ floor, 0 ≤ s ≤ 1).

**Impact:** ~50-100x speedup on the SDP stage itself. For the overall pipeline,
this matters mainly for the initial cache construction (one-time cost per
knockoff run), so pipeline impact is moderate (~10-20% overall) but it removes
a major latency spike.

### Step 1.3: Restructure binomial CD to IRLS ✅

**Commit:** `68016e0` — Replaced `sweep_binomial_coordinates` (sigmoid per
coordinate) with IRLS structure: compute sigmoid/weights once per outer Newton
iteration, then solve weighted LS via new `sweep_weighted_coordinates`. All 20
slide-glmnet tests pass including R parity (binomial path + knockoff stats).

**Audit issue:** #3 — sigmoid recomputed per coordinate.

**File:** `crates/slide-glmnet/src/lib.rs:842-962`

**Python reference:** Uses vendored Fortran `glmnet5.f90` which implements
the IRLS (iteratively reweighted least squares) structure:
- **Outer Newton loop:** compute `p = sigmoid(η)`, `w = p*(1-p)`,
  working response `z = η + (y - p)/w`
- **Inner CD loop:** solve weighted least squares `min Σ w_i (z_i - x_i β)²`
  via coordinate descent with fixed weights — identical to the Gaussian path
  but with per-observation weights.

**Change:** Restructure `binomial_coordinate_descent_step` to:

```rust
fn binomial_coordinate_descent_step(
    x: &Array2<Scalar>, y: &Array1<Scalar>,
    beta: &mut Array1<Scalar>, intercept: &mut Scalar,
    lambda: Scalar, alpha: Scalar, tolerance: Scalar, max_iter: usize,
) {
    let n = x.nrows();
    let p = x.ncols();
    let mut eta = x.dot(beta);
    eta += *intercept;

    for _newton in 0..LOGISTIC_MAX_OUTER_ITERATIONS {
        // Compute weights and working response ONCE per Newton step
        let prob = eta.mapv(sigmoid);
        let weights = prob.mapv(|p| (p * (1.0 - p)).max(MIN_PROBABILITY));
        let working_response = &eta + &(y - &prob) / &weights;

        // Update intercept
        let wsum = weights.sum();
        let residual_sum = (&weights * &(&working_response - *intercept)).sum();
        // ... intercept Newton step ...

        // Inner CD on weighted least squares (reuse Gaussian sweep logic)
        let mut active = beta.iter().map(|v| v.abs() > 0.0).collect::<Vec<_>>();
        let max_change = sweep_weighted_coordinates(
            x, &working_response, &weights, column_weighted_sq_norm,
            beta, &mut residual, &active, lambda, alpha, n,
        );

        // Recompute eta from beta (not incrementally, to avoid drift)
        eta = x.dot(beta);
        eta += *intercept;

        if max_change <= tolerance { break; }
    }
}
```

**Key insight from Python/R:** The weighted CD inner loop per-coordinate update
becomes:
```
gradient_j = Σ w_i * x_ij * (z_i - x_i·β) / n
new_beta_j = soft_threshold(gradient_j, λα) / (weighted_norm_j + λ(1-α))
```
This is just the Gaussian CD formula with `w_i` weighting — no sigmoid in the
inner loop.

**`sweep_weighted_coordinates`** is a new function nearly identical to
`sweep_coordinates` but with a `weights: &Array1<Scalar>` parameter that scales
the dot products and norms. The column squared norms become
`Σ w_i * x_ij²` (precomputed once per Newton step).

**Verify:**
- `cargo test -p slide-glmnet` — existing binomial lasso path tests
- Write a small parity test: compare Rust binomial path output against Python
  on a 20×5 binary-response fixture.

**Impact:** Reduces sigmoid evaluations from O(n × p × CD_iters × λ_count)
to O(n × Newton_iters × λ_count). For typical numbers (p=344, CD_iters=5,
Newton_iters=5), this is ~344/5 = **~70x fewer sigmoid calls** per lambda step.

### Step 1.4: Eliminate augmented matrix clone in voting loop ✅

**Commit:** `998dc24` — Changed `map_init` to create a `Matrix` directly
(instead of `Array2`), then mutate columns in place each iteration via
`as_ndarray_mut()`. The `compute_statistic_prebuilt` borrows `&Matrix`
immutably — no clone needed. Simpler than the plan's `_raw` variant approach.

**Audit issue:** #4 — 825 KB clone per iteration.

**File:** `crates/slide-knockoff/src/lib.rs:274-296`

**Python reference:** `knockoff/stats/glmnet.py` just does
`X_combined = np.hstack([X_swap, Xk_swap])` — one allocation, no clone. The
combined matrix is passed to the Fortran glmnet which reads it directly.

**Change:** Add `Matrix::from_ndarray_owned(array)` that takes ownership of an
`Array2` without cloning:

```rust
// In slide-types:
impl Matrix {
    pub fn from_ndarray_owned(array: Array2<Scalar>) -> Self {
        Self(array)
    }
}
```

Then in the voting loop, instead of `buf.clone()`, move ownership:

Actually, a simpler approach: change `compute_statistic_prebuilt` and the
glmnet `_prebuilt` functions to accept `&Array2<Scalar>` directly instead of
`&Matrix`. This avoids the wrapping entirely:

```rust
// Before:
let augmented = Matrix::from_ndarray(buf.clone());  // clone!
let w = compute_statistic_prebuilt(&augmented, p, y, config)?;

// After:
let w = compute_statistic_prebuilt_raw(buf, p, y, config)?;
```

Where `compute_statistic_prebuilt_raw` calls glmnet functions that take
`&Array2<Scalar>` for the design matrix. The `PreparedData::from_inputs` already
calls `x.as_ndarray()` immediately — so the `Matrix` wrapper is pure overhead.

**Scope of change:**
- `compute_statistic_prebuilt` → `compute_statistic_prebuilt_raw(&Array2, ...)`
- `stat_glmnet_lambdasmax_prebuilt` → add `_raw` variant taking `&Array2`
- `stat_glmnet_lambdadiff_prebuilt` → same
- `stat_glmnet_coefdiff_prebuilt` → same
- `gaussian_lasso_path` / `binomial_lasso_path` → add `_raw` variants
- Keep the `&Matrix` versions as thin wrappers for the public API

**Verify:** `cargo test -p slide-knockoff -p slide-glmnet`

**Impact:** Eliminates 100 × 825 KB = 80 MB of allocation per voting call.
Moderate direct speedup (~5-10%), but reduces GC/allocator pressure which
improves cache behavior for the CD inner loop.

### Step 1.5: Add missing column swap in knockoff statistic (correctness) ✅

**Commit:** `998dc24` (combined with 1.4) — Added Bernoulli(0.5) column swap
between X and X_tilde before computing the knockoff statistic. Swap uses
deterministic seed `base_seed + iteration + SWAP_SEED_OFFSET`. After computing
the stat, signs are corrected: `W[j] *= -1` for swapped columns. Columns are
un-swapped in place to restore X for the next iteration. All 8 slide-knockoff
tests pass (determinism preserved).

**Audit issue:** not in original audit, discovered during Python review.

**File:** `crates/slide-knockoff/src/lib.rs:274-296`

**Python reference:** `knockoff/stats/glmnet.py:391` and `base.py:7-48`

The Python knockoff statistic implementation:
1. Randomly swaps columns of X and X_k (`swap[j] ~ Bernoulli(0.5)`)
2. Concatenates `[X_swap | Xk_swap]`
3. Computes the W statistic on the swapped version
4. Corrects signs: `W = W * (1 - 2*swap)`

The Rust implementation does NOT swap columns. It just concatenates `[X | X_tilde]`
directly. This breaks the knockoff symmetry property and may contribute to the
statistical divergence (different feature selection patterns, though NOT the
Z-matrix divergence which is upstream in LOVE).

**Change:** In `knockoff_filter_voting`, before computing the statistic:

```rust
// Per iteration, after filling X_tilde into buf:
let mut rng = StdRng::seed_from_u64(config.base_seed + iteration as u64 + SWAP_SEED_OFFSET);
let swap: Vec<bool> = (0..p).map(|_| rng.gen_bool(0.5)).collect();

// Swap columns in buf
for j in 0..p {
    if swap[j] {
        for i in 0..n {
            buf.swap((i, j), (i, p + j));
        }
    }
}

let w = compute_statistic_prebuilt_raw(buf, p, y, config)?;

// Correct for swap: flip sign where swapped
let mut w_array = w.into_ndarray();
for j in 0..p {
    if swap[j] {
        w_array[j] = -w_array[j];
    }
}
```

**Note:** The swap is done in-place on the pre-allocated buffer. After
computing the statistic, the buffer's X portion may be corrupted by the swap,
so we need to re-copy X columns at the start of each iteration (or un-swap
after computing the stat). The simplest approach: always copy both X and X_tilde
into buf per iteration (the X copy is already done in `map_init`, but we need
to redo it if we swapped).

Actually, the most efficient approach:
1. `map_init`: copy X into buf once (left half)
2. Per iteration: copy X_tilde into right half, then swap in-place, compute stat
3. After stat: un-swap to restore X (swap the same columns back)

**Verify:**
- `cargo test -p slide-knockoff`
- On the SSc benchmark, the knockoff W-statistic distribution should now be
  symmetric under permutation of (X, X_tilde) for null features.

---

## Phase 2: High-impact performance fixes ✅ COMPLETED

### Step 2.1: Add early stopping to lasso path

**Audit issue:** #5

**File:** `crates/slide-glmnet/src/lib.rs`

**Python reference:** The Fortran glmnet has built-in early stopping. The Python
wrapper just sets `n_lambda=250` and lets Fortran decide when to stop. Key
stopping criteria (from `glmnet5.f90`):

1. **All variables entered** — for `lambdasmax`, once all 2p coefficients are
   non-zero, the entry lambdas are determined. Stop.
2. **Deviance ratio** — `dev_ratio > 0.999` (explained ≥ 99.9% of null deviance)
3. **Max active set** — number of non-zero coefficients exceeds `dfmax = p + 1`
   (for lasso) or a user-specified limit
4. **Fractional deviance change** — `|dev_ratio[k] - dev_ratio[k-1]| < fdev`
   where `fdev = 1e-5` by default

**Change:** Add early stopping in the lambda loop:

```rust
// In gaussian_lasso_path, after coordinate_descent_step:
let nnz = beta_scaled.iter().filter(|b| b.abs() > 1e-10).count();
let rss = residual.dot(&residual);
let dev_ratio = 1.0 - rss / null_deviance;

if nnz >= n {
    // More variables than observations — stop
    break;
}
if dev_ratio > 0.999 {
    break;
}
if lambda_index > 0 {
    let prev_dev = /* tracked from previous iteration */;
    if (dev_ratio - prev_dev).abs() < 1e-5 {
        break;
    }
}
```

For `lambdasmax` specifically, add a targeted criterion: once all rows of
`beta_path` have at least one non-zero entry, every feature has a defined
entry lambda. Stop immediately.

**Verify:** `cargo test -p slide-glmnet` — regression test that the truncated
path produces the same `lambdasmax` statistic as the full path.

**Impact:** For the knockoff voting loop, the lasso path typically saturates
well before lambda 250. On the SSc data with 2p=344 columns, expect ~50-70%
of lambda steps to be skippable → **~2x speedup** on the glmnet component.

### Step 2.2: Optimize `to_nalgebra()` / `from_nalgebra()` conversions

**Audit issue:** #6

**File:** `crates/slide-types/src/lib.rs:96-149`

**Current code:**
```rust
pub fn to_nalgebra(&self) -> DMatrix<Scalar> {
    let (rows, cols) = self.shape();
    DMatrix::from_fn(rows, cols, |r, c| self.0[(r, c)])
    // ↑ element-wise copy via indexing — no SIMD, no memcpy
}

pub fn from_nalgebra(matrix: DMatrix<Scalar>) -> Self {
    let (rows, cols) = (matrix.nrows(), matrix.ncols());
    let mut array = Array2::zeros((rows, cols));
    for row in 0..rows {
        for col in 0..cols {
            array[(row, col)] = matrix[(row, col)];
        }
    }
    Self(array)
}
```

**Change:** Use layout-aware bulk copies:

```rust
pub fn to_nalgebra(&self) -> DMatrix<Scalar> {
    let (rows, cols) = self.shape();
    // ndarray default is row-major; nalgebra is column-major
    // Use from_iterator with column-order traversal
    let data: Vec<Scalar> = (0..cols)
        .flat_map(|c| self.0.column(c).iter().copied())
        .collect();
    DMatrix::from_vec(rows, cols, data)
}

pub fn from_nalgebra(matrix: DMatrix<Scalar>) -> Self {
    let (rows, cols) = (matrix.nrows(), matrix.ncols());
    // nalgebra stores column-major; build ndarray from columns
    let data = matrix.as_slice(); // contiguous column-major
    let array = Array2::from_shape_vec(
        (rows, cols).f(),  // column-major (Fortran order)
        data.to_vec(),
    ).unwrap_or_else(|_| unreachable!("shape matches"));
    // Convert to row-major default if needed, or keep col-major
    Self(array)
}
```

**Better yet** for hot paths: keep data in column-major throughout. The
`PreparedData` already converts to column-major for CD. If we keep `Matrix`
in column-major order, `to_nalgebra` becomes a zero-copy view (both are
column-major), and `from_nalgebra` is a memcpy.

For this to work, `Matrix` needs to be column-major by default. This is a
larger refactor but eliminates the conversion cost entirely for the
eigendecomp/cholesky paths.

**Phased approach:**
1. First: optimize `to_nalgebra` / `from_nalgebra` with bulk column copies
2. Later: consider switching `Matrix` to column-major as the default

**Verify:** `cargo test --workspace` — all matrix operations must be unchanged.

**Impact:** Each conversion goes from O(n²) indexed access to O(n²) memcpy.
For a 100×100 matrix, this is ~10x faster per conversion. Combined with
reduced conversion count (from Step 1.1 removing eigendecomp), the total
conversion overhead drops dramatically.

### Step 2.3: Flatten nested rayon parallelism

**Audit issue:** #7

**Files:** `crates/slide-knockoff/src/lib.rs`, `crates/slide-core/src/lib.rs`

**Python reference:** Python doesn't have nested parallelism at all. The
knockoff iteration loop is the parallelism unit (via `multiprocessing` or
sequential). Chunks are run sequentially.

**Three nesting levels in Rust:**
1. `find_interaction_lfs` → `par_iter` over marginal LFs
2. `select_short_freq` → `par_iter` over chunks (when n ≤ p)
3. `knockoff_filter_voting` → `par_iter` over iterations

**Change:**

1. **`select_short_freq`:** Run chunks sequentially (`.iter()` instead of
   `.par_iter()`). The iteration parallelism inside `knockoff_filter_voting`
   is the right level to parallelize — it has 100-1000 independent tasks.

2. **`find_interaction_lfs`:** Keep the `par_iter` over marginal LFs, but
   ensure `knockoff_filter_voting` uses `par_iter` only when not already inside
   a parallel context. Add a `parallel: bool` parameter:

```rust
pub fn knockoff_filter_voting(
    x: &Matrix, y: &Vector, config: &KnockoffConfig, parallel: bool,
) -> Result<VotingResult, KnockoffError> {
    // ...
    let iter = 0..config.niter;
    let selected_list = if parallel {
        iter.into_par_iter().map_init(/* ... */).collect()?
    } else {
        iter.map(|iteration| { /* same logic but sequential */ }).collect()?
    };
    // ...
}
```

3. **In `find_interaction_lfs`:** Call `knockoff_filter_voting` with
   `parallel=false` since the outer `par_iter` over marginals already uses all
   cores. Each marginal's voting loop runs sequentially on one core.

**Verify:** `cargo test --workspace` + benchmark with `RAYON_NUM_THREADS=4`.

**Impact:** Eliminates thread pool contention and cache thrashing. Expected
~20-30% improvement on multi-core runs.

---

## Phase 3: Polish ✅ COMPLETED

### Step 3.1: Remove redundant validation in decomposition inner loops ✅

**Commit:** `fe849b8`

**Audit issue:** #8

**File:** `crates/slide-types/src/lib.rs`

Added `symmetric_eigendecomposition_unchecked()` and `cholesky_unchecked()` to
`Matrix`. Original checked methods now delegate to unchecked after validation.
Internal callers in slide-sdp, slide-knockoff, and slide-love switched to
unchecked variants.

### Step 3.2: Switch normal sampling to Ziggurat ✅

**Commit:** `fe849b8`

**Audit issue:** #9

**File:** `crates/slide-knockoff/src/lib.rs`

Replaced hand-rolled Box-Muller (`sample_standard_normal` with spare caching)
with `rand_distr::StandardNormal` (Ziggurat). Added `rand_distr = "0.4"` to
`slide-knockoff/Cargo.toml`.

**Impact:** ~40% faster normal generation.

### Step 3.3: Trim KnockoffCache ✅

**Commit:** `fe849b8`

**Audit issue:** #10

**File:** `crates/slide-knockoff/src/lib.rs`

Removed `mu`, `sigma`, `diag_s` from `KnockoffCache`. Only `mu_k` and `l_trans`
are retained for sampling. Saves ~3 matrix clones per cache construction and
reduces working set size.

### Step 3.4: Pre-allocate interaction term matrix ✅

**Commit:** `fe849b8`

**Audit issue:** #11

**File:** `crates/slide-core/src/lib.rs`

Pre-allocate `Array2::zeros((nrows, total_interactions))` and fill columns
in-place via `ndarray::azip!` instead of collecting `Vec<Array1>` then
assembling via `columns_to_matrix`. Removed the now-dead helper.

---

## Verification Protocol

### After each step:

1. `cargo test --workspace` — all tests pass
2. `cargo clippy --workspace` — no new warnings
3. Commit with step number in message

### After Phase 0:

4. Z-matrix column correlations ≥ 0.95 (absolute) with Python reference on
   the SSc dataset
5. LOVE outputs (K, pure_vec, A nonzero structure) match Python

### After Phase 1:

6. Re-run SSc benchmark (delta=0.1, lambda=1.0) — expect ≤ 400s (vs 1396s
   baseline and 367s for R)
7. Knockoff W-statistics should be sign-symmetric for null features

### Phase 2 completion notes:

**Step 2.1 (early stopping):** Implemented two criteria: nnz >= n (saturated
model) and dev_ratio > 0.9999 (near-perfect fit). The fdev (fractional deviance
change) criterion was omitted as it triggered too aggressively on small datasets
without matching R's exact Fortran implementation. CV lambda selection pads
truncated folds with Inf to avoid selecting un-computed lambdas. All 20
slide-glmnet tests pass including R parity fixtures.

**Step 2.2 (nalgebra conversion):** `to_nalgebra` now iterates column-major and
uses `DMatrix::from_vec` (zero-copy column-major). `from_nalgebra` uses bulk
`as_slice().to_vec()` with column-major Array2 construction instead of
per-element row/col indexing. All 16 slide-types tests pass.

**Step 2.3 (rayon flattening):** Added `parallel: bool` field to
`KnockoffConfig` (default true). `knockoff_filter_voting` uses par_iter or
sequential iter based on the flag. `select_short_freq` chunk iteration changed
from par_iter to plain iter. `find_interaction_lfs` sets `parallel=false` since
the outer par_iter over marginals already uses all cores. All knockoff and core
tests pass.

### After Phase 2:

8. Full SSc benchmark (all 4 configs) — expect all configs faster than R
9. Profile with `perf` or `flamegraph` to confirm no remaining hot spots

### Phase 3 completion notes:

**Step 3.1 (unchecked decompositions):** Added `symmetric_eigendecomposition_unchecked()`
and `cholesky_unchecked()` to `Matrix` that skip square/symmetry validation. Original
methods now delegate to `_unchecked` after validation. Callers in slide-sdp
(`minimum_eigenvalue`, `minimum_eigenvalue_of_shifted_matrix`), slide-knockoff
(`minimum_eigenvalue`, `regularized_cholesky`), and slide-love (`est_k`) switched
to `_unchecked` variants.

**Step 3.2 (Ziggurat sampling):** Removed hand-rolled `sample_standard_normal()`
(Box-Muller with spare value caching). Replaced with `rand_distr::StandardNormal`
(Ziggurat algorithm) via `Array2::from_shape_fn`. Added `rand_distr = "0.4"` to
`slide-knockoff/Cargo.toml`.

**Step 3.3 (cache trim):** Removed `mu`, `sigma`, `diag_s` from `KnockoffCache`
struct. Only `mu_k` and `l_trans` are retained — the only fields needed for
knockoff sampling. Saves ~3 matrix clones per cache construction.

**Step 3.4 (interaction alloc):** Pre-allocate `Array2::zeros((nrows, total_interactions))`
and fill columns in-place with `ndarray::azip!` instead of collecting `Vec<Array1>`
and assembling via `columns_to_matrix()`. Removed the now-dead `columns_to_matrix`
helper.

### After Phase 3:

10. Final benchmark + memory profiling (`heaptrack`)
11. Update `specs/rust-performance-audit.md` with final numbers

---

## Dependency Order

```
Phase 0 (correctness)
  └→ Step 0.1 (diagnostic harness)
  └→ Step 0.2 (fix divergences)
     ↓
Phase 1 (critical perf) — steps are independent, can be done in any order
  ├→ Step 1.1 (cholesky)
  ├→ Step 1.2 (SDP feasibility)
  ├→ Step 1.3 (binomial IRLS)
  ├→ Step 1.4 (eliminate clone)
  └→ Step 1.5 (column swap — correctness)
     ↓
Phase 2 (high-impact perf) — steps are independent
  ├→ Step 2.1 (early stopping)
  ├→ Step 2.2 (nalgebra conversion)
  └→ Step 2.3 (rayon nesting)
     ↓
Phase 3 (polish) — steps are independent
  ├→ Step 3.1 (validation)
  ├→ Step 3.2 (Ziggurat)
  ├→ Step 3.3 (cache trim)
  └→ Step 3.4 (interaction alloc)
```

## Summary Table

| Step | Issue | Est. Impact | Crate(s) | Status |
|------|-------|-------------|----------|--------|
| 0.1-0.2 | Z-matrix wrong | correctness | slide-love, slide-core | pending |
| 1.1 | Eigen before Cholesky | 2x on Cholesky paths | slide-types | ✅ done |
| 1.2 | SDP binary search | 50-100x on SDP stage | slide-sdp | ✅ done |
| 1.3 | Sigmoid per coordinate | ~70x fewer sigmoid calls | slide-glmnet | ✅ done |
| 1.4 | Augmented clone | ~80 MB/call saved | slide-knockoff | ✅ done |
| 1.5 | Missing column swap | correctness | slide-knockoff | ✅ done |
| 2.1 | No early stopping | ~2x on glmnet | slide-glmnet | ✅ done |
| 2.2 | Slow nalgebra conversion | ~10x per conversion | slide-types | ✅ done |
| 2.3 | Nested rayon | ~20-30% overall | slide-knockoff, slide-core | ✅ done |
| 3.1 | Redundant validation | 5-10% on decomp paths | slide-types | ✅ done |
| 3.2 | Box-Muller sampling | ~40% on RNG | slide-knockoff | ✅ done |
| 3.3 | Cache bloat | memory savings | slide-knockoff | ✅ done |
| 3.4 | Interaction alloc | minor | slide-core | ✅ done |
