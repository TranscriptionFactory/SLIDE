# Correctness Criteria & Test Vectors

Acceptance criteria for the Rust port against the R SLIDE reference implementation. All criteria are derived from verified comparisons on the SSc dataset (24 samples, 172/88 latent factors depending on delta).

---

## 1. LOVE Decomposition Correctness

### EstC (Covariance Matrix)

**WHEN** LOVE runs on the SSc dataset with any valid delta
**THEN** the estimated covariance matrix `EstC` matches the R reference with max absolute difference < 1e-14

**WHEN** the verified difference is 5.83e-16
**THEN** this tolerance bound (1e-14) should be treated as a hard acceptance threshold, not a target

### estOmega (Precision Matrix)

**WHEN** LOVE computes the precision matrix `estOmega` (inverse of `EstC`)
**THEN** max absolute difference vs R reference is < 1e-12

**WHEN** the verified difference is 3.53e-13
**THEN** any implementation producing max diff > 1e-12 must be investigated for accumulation of floating-point error

### Score / Moments Matrix

**WHEN** LOVE computes the score or moments matrix
**THEN** max absolute difference vs R reference is < 1e-14

**WHEN** the verified difference is approximately 2e-15
**THEN** this confirms that direct matrix computations (not iterative) are required to preserve precision

### K (Number of Factors)

**WHEN** LOVE runs on SSc data at delta=0.01
**THEN** `K` exactly matches the R reference value (172 latent factors)

**WHEN** LOVE runs on SSc data at delta=0.1
**THEN** `K` exactly matches the R reference value (88 latent factors)

**WHEN** `K` does not match R for any tested delta
**THEN** the run is rejected as non-conforming; numerical tolerance on continuous outputs does not compensate for structural mismatches in factor count

### Z-Matrix (pure_homo=True)

**WHEN** `pure_homo=True` and LOVE runs on SSc data at delta=0.01
**THEN** column-wise Pearson correlation between the Rust Z-matrix and the R reference Z-matrix has mean > 0.98

**WHEN** the verified mean column correlation is 0.9988
**THEN** implementations with mean < 0.98 indicate a sign-flip or ordering issue in the eigenvectors and must be corrected

### pureVec (pure_homo=True)

**WHEN** `pure_homo=True`
**THEN** `pureVec` (the vector identifying pure variables for each factor) matches R exactly

**WHEN** `pureVec` does not match R under `pure_homo=True`
**THEN** the selection logic for identifying anchor variables is incorrect and must be debugged before proceeding

### pureVec (pure_homo=False)

**WHEN** `pure_homo=False`
**THEN** `pureVec` matches R exactly

**WHEN** this criterion is tested against the current Python implementation
**THEN** a mismatch is expected (known Python bug); the Rust port must fix this bug and produce R-matching output

### Screen_X Noise Detection

**WHEN** `screen_X` is called to detect noisy features
**THEN** the set of flagged noise features matches R exactly

**WHEN** this criterion is tested against the current Python implementation
**THEN** a mismatch is expected (known Python bug); the Rust port must fix this bug and produce R-matching output

---

## 2. Knockoff Filter Correctness

### SDP diag_s Values

**WHEN** the SDP solver computes `diag_s` for a given covariance structure
**THEN** max absolute difference from R reference values is < 1e-6

**WHEN** binary-search scaling is applied to ensure `2 * diag_s <= eigenvalue_min`
**THEN** the resulting `diag_s` values satisfy the positive-semidefinite constraint and match R to within 1e-6

### W-Statistic Correlation

**WHEN** the W-statistic is computed for a dataset where n > p
**THEN** Pearson correlation between Rust W-statistics and R reference W-statistics is > 0.97

**WHEN** run on SSc data at delta=0.05
**THEN** the verified correlation is 0.9885; values below 0.97 indicate a lasso path or knockoff construction error

### Knockoff Filter Threshold

**WHEN** the knockoff filter threshold `T` is computed
**THEN** it is evaluated using the exact formula: `T = min{ t : #{W <= -t} / max(1, #{W >= t}) <= q }`

**WHEN** `T` is computed using any other formula or approximation
**THEN** the implementation is non-conforming regardless of empirical results

### Voting Frequency Counting

**WHEN** `select_short_freq` completes all voting iterations
**THEN** the frequency count for each feature (how many iterations selected it) is computed exactly with no off-by-one errors

**WHEN** frequencies are aggregated across iterations
**THEN** the result is deterministic for the same seed and matches Python verified output

### findOptIter Logic

**WHEN** `findOptIter` is called to determine the optimal number of voting iterations
**THEN** the logic matches R's reference implementation exactly

**WHEN** the `findOptIter` result differs from R
**THEN** the stopping criterion or loop bounds are incorrect and must be corrected before the voting loop is trusted

### calc_default_fsize

**WHEN** `calc_default_fsize` is called to determine feature chunk size
**THEN** the result matches R exactly for all tested (n, p) combinations

### Feature Chunking

**WHEN** features are split into chunks for voting
**THEN** the split formula matches R exactly: `floor(p / n_chunks)` features per chunk with remainder distributed to last chunk

**WHEN** any chunk boundary differs from R
**THEN** W-statistic distributions will diverge and downstream results are invalid

---

## 3. End-to-End Concordance

### delta=0.01

**WHEN** the full SLIDE pipeline (LOVE + knockoff selection) runs on SSc data at delta=0.01
**THEN** the Jaccard similarity of selected marginal LFs between Rust and R is >= 0.70

**WHEN** Jaccard < 0.70 at delta=0.01
**THEN** the shortfall must be attributed to a specific component (LOVE K mismatch, W-statistic divergence, or threshold error) before release

### delta=0.1

**WHEN** the full SLIDE pipeline runs on SSc data at delta=0.1
**THEN** the Jaccard similarity of selected marginal LFs between Rust and R is >= 0.40

**WHEN** Jaccard < 0.40 at delta=0.1
**THEN** LOVE sensitivity at higher delta is the primary suspect; the lower threshold reflects inherent stochastic divergence amplification

### Python Internal Consistency

**WHEN** the Python implementation is run and compared against itself across seeds
**THEN** Jaccard similarity for repeated runs with the same seed is 1.0 (deterministic)

**WHEN** Python results are compared against R results
**THEN** Jaccard >= 0.80 is required for Python to serve as an intermediate validation layer

### Stochastic Reproducibility

**WHEN** 500 voting iterations are run with the same base seed using the pattern `base_seed + iteration`
**THEN** results are fully deterministic within a single implementation

**WHEN** results are compared across implementations (Python vs Rust vs R)
**THEN** stochastic divergence is expected unless RNG sequences match exactly; Jaccard thresholds above define acceptable divergence

---

## 4. Numerical Precision Requirements

### Matrix Multiplication

**WHEN** any matrix multiplication is performed in the LOVE or knockoff pipeline
**THEN** BLAS level-3 operations are used (e.g., `dgemm`)

**WHEN** non-BLAS matrix multiplication is used
**THEN** numerical drift will accumulate and precision thresholds in section 1 will not be achievable

### Eigendecomposition

**WHEN** eigendecomposition is required (e.g., for covariance matrix factoring)
**THEN** LAPACK `dsyev` or an equivalent symmetric eigensolver is used

**WHEN** a non-symmetric or general eigensolver is used on a symmetric matrix
**THEN** the implementation is non-conforming even if results appear numerically close

### Cholesky Factorization

**WHEN** Cholesky decomposition is applied
**THEN** LAPACK `dpotrf` is used with an explicit positive-definite check beforehand

**WHEN** the positive-definite check fails
**THEN** a descriptive error is raised identifying the matrix and its minimum eigenvalue

### Linear Solves

**WHEN** a linear system must be solved
**THEN** factored forms (e.g., LU, Cholesky) are used rather than explicit matrix inversion

**WHEN** `inv(A) @ b` appears in code
**THEN** it must be replaced with `solve(A, b)` using the appropriate LAPACK routine

### Floating-Point Precision

**WHEN** any intermediate result is computed in the LOVE or knockoff pipeline
**THEN** it is stored and operated on in f64 (double precision)

**WHEN** f32 or reduced precision is used anywhere in the pipeline
**THEN** the implementation is non-conforming

---

## 5. RNG Reproducibility

### MT19937 Matching

**WHEN** the Rust implementation uses the same integer seed as the R implementation
**THEN** the Rust MT19937 RNG produces the same sequence of random numbers as R's `set.seed(n)` + `rnorm()` calls

**WHEN** MT19937 matching is implemented and verified
**THEN** knockoff matrix construction becomes deterministic and W-statistic correlation is expected to reach > 0.99

**WHEN** MT19937 matching is not implemented
**THEN** stochastic divergence is accepted and Jaccard thresholds in section 3 apply instead

### Deterministic Seeding Pattern

**WHEN** voting iterations are run
**THEN** each iteration `i` uses seed `base_seed + i` for its knockoff generation step

**WHEN** any other seeding strategy is used
**THEN** results will not be reproducible across machines or runs and the voting frequency counts cannot be validated

---

## 6. Known Bugs and Improvements

### SDP Caching

**WHEN** multiple knockoff iterations run with the same covariance structure
**THEN** `diag_s` is computed once and cached, not recomputed per iteration

**WHEN** caching is disabled or removed
**THEN** runtime increases linearly with iteration count and the behavior diverges from the Python implementation

**WHEN** the cache is used
**THEN** the cached `diag_s` values are identical to what would be computed on every call (no staleness)

### Deterministic Seeding (Python Improvement)

**WHEN** the Rust port is implemented
**THEN** the `base_seed + iteration` seeding pattern from the Python implementation is preserved

**WHEN** R's original random seeding (no per-iteration seed) is used instead
**THEN** results are not reproducible across runs and the improvement is lost

### correct_y / correctIt

**WHEN** `correct_y(y, z_m)` is called
**THEN** it returns OLS residuals of regressing `y` on `z_m`, matching R's `correctIt` function to floating-point precision

**WHEN** the residual computation uses a formula other than `y - z_m * (z_m' z_m)^{-1} z_m' y`
**THEN** numerical differences from R will appear in interaction detection results

### pureVec and Screen_X Bug Fixes

**WHEN** the Rust port implements `pureVec` under `pure_homo=False`
**THEN** it produces output matching R, fixing the known Python divergence

**WHEN** the Rust port implements `screen_X`
**THEN** it produces noise feature sets matching R exactly, fixing the known Python divergence

**WHEN** these fixes are validated
**THEN** the test vectors from SSc data are used as ground truth, not the Python output
