# Knockoff Filter

The knockoff filter provides FDR-controlled variable selection. SLIDE uses it with a voting procedure over multiple iterations to obtain stable, reproducible selections.

---

## 1. Knockoff Matrix Generation (`create_gaussian`)

**WHEN** `create_gaussian(X, mu, Sigma, diag_s)` is called
**THEN** a knockoff copy `X_tilde` is generated such that `[X, X_tilde]` has the required second-moment properties:
1. `E[X_tilde] = mu`
2. `Cov([X, X_tilde]) = [[Sigma, Sigma - diag(s)], [Sigma - diag(s), Sigma]]`

**WHEN** the knockoff sample is drawn
**THEN** it is computed as:
```
X_tilde = X - (X - mu) @ Sigma_inv @ diag(s) + C_tilde @ U
```
where `C_tilde = cholesky(2*diag(s) - diag(s) @ Sigma_inv @ diag(s))` and `U ~ N(0, I)`

**WHEN** `diag_s` contains any value outside `[0, 1]`
**THEN** a `ValueError` is raised before sampling begins

**WHEN** `2*diag(s) - diag(s) @ Sigma_inv @ diag(s)` is not positive semi-definite due to numerical error
**THEN** a small regularization `epsilon` is added to the diagonal before the Cholesky factorization

---

## 2. Equi-Correlated SDP Solver (`create_solve_equi`)

**WHEN** `create_solve_equi(Sigma)` is called
**THEN** it returns a scalar `s` such that `s_j = min(1, 2 * lambda_min(Sigma))` for all `j`

**WHEN** `lambda_min(Sigma) >= 0.5`
**THEN** `s_j = 1.0` for all `j`

**WHEN** `lambda_min(Sigma) < 0.5`
**THEN** `s_j = 2 * lambda_min(Sigma)` for all `j`

**WHEN** the resulting `diag(s)` causes `2*Sigma - diag(s)` to not be positive semi-definite
**THEN** `s` is scaled down by a binary search until feasibility is achieved

---

## 3. SDP Solver (`create_solve_sdp`)

**WHEN** `create_solve_sdp(Sigma)` is called
**THEN** it solves the semidefinite program:
```
maximize   sum(s)
subject to 0 <= s_j <= 1  for all j
           2*Sigma - diag(s) >= 0  (positive semidefinite)
```

**WHEN** the SDP solution `s` is returned by the solver
**THEN** a binary search feasibility scaling step is applied to ensure `2*Sigma - diag(s)` is strictly positive semi-definite

**WHEN** the binary search scaling is applied
**THEN** `s` is multiplied by the largest scalar `gamma in (0, 1]` such that `2*Sigma - gamma*diag(s) >= 0`

**WHEN** the binary search scaling is omitted
**THEN** the implementation is incorrect and will not match R's behavior; this step must not be removed

---

## 4. Approximate SDP Solver (`create_solve_asdp`)

**WHEN** `create_solve_asdp(Sigma)` is called
**THEN** it approximates the SDP solution using block-diagonal structure of `Sigma`

**WHEN** block structure is determined
**THEN** `Sigma` is partitioned into blocks and the SDP is solved independently per block

**WHEN** each block SDP is solved
**THEN** the same binary search feasibility scaling as in `create_solve_sdp` is applied per block

**WHEN** block solutions are assembled into a full `diag_s` vector
**THEN** the assembled vector satisfies `2*Sigma - diag(s) >= 0` globally

**WHEN** `create_solve_asdp` is used (the default method)
**THEN** its output matches `create_solve_sdp` to within acceptable numerical tolerance for well-conditioned `Sigma`

---

## 5. W-Statistic: Lasso Max-Lambda (`stat_glmnet_lambdasmax`)

**WHEN** `stat_glmnet_lambdasmax(X, X_tilde, y, nlambda=250)` is called
**THEN**:
1. The augmented matrix `[X, X_tilde]` of size `n x 2p` is assembled
2. A lasso path is fit on `[X, X_tilde]` against `y` with `nlambda` lambda values
3. For each original feature `j`, `lambda_j` = the largest lambda at which `X_j` enters the lasso path
4. For each knockoff `j`, `lambda_tilde_j` = the largest lambda at which `X_tilde_j` enters the path
5. `W_j = lambda_j - lambda_tilde_j`

**WHEN** feature `j` never enters the lasso path
**THEN** its entry lambda is set to `0`

**WHEN** `nlambda=250`
**THEN** the lambda grid resolution matches R's `stat.glmnet.lambdamax` with default settings

**WHEN** `nlambda < 250`
**THEN** the lambda grid is coarser and W statistics will diverge from R; `nlambda=250` must be used

**WHEN** comparing `W` statistics against R on the same data and SDP solution
**THEN** Pearson correlation between Python and R W vectors is >= 0.97 for `n > p` cases

---

## 6. W-Statistic: Lambda Difference (`stat_glmnet_lambdadiff`)

**WHEN** `stat_glmnet_lambdadiff(X, X_tilde, y, nlambda=250)` is called
**THEN** a lasso path is fit on `[X, X_tilde]` and `W_j` is computed as the sum of absolute coefficient differences across all lambda values

**WHEN** the lasso path is fit
**THEN** `nlambda=250` is used for consistency with `stat_glmnet_lambdasmax`

---

## 7. W-Statistic: Coefficient Difference (`stat_glmnet_coefdiff`)

**WHEN** `stat_glmnet_coefdiff(X, X_tilde, y)` is called
**THEN** a single lasso model is fit at a fixed regularization strength and `W_j = |beta_j| - |beta_tilde_j|`

---

## 8. Knockoff Filter (`knockoff_filter`)

**WHEN** `knockoff_filter(W, q, offset)` is called
**THEN** the knockoff threshold is computed as:
```
T = min{ t > 0 : #{j: W_j <= -t} / max(1, #{j: W_j >= t} + offset) <= q }
```

**WHEN** `offset=0` (original knockoff)
**THEN** the denominator is `max(1, #{j: W_j >= t})`

**WHEN** `offset=1` (knockoff+)
**THEN** the denominator is `#{j: W_j >= t} + 1`

**WHEN** no threshold `t` satisfies the FDR condition
**THEN** no variables are selected and an empty set is returned

**WHEN** threshold `T` is found
**THEN** the selected set is `{j : W_j >= T}`

**WHEN** `q` is outside `(0, 1)`
**THEN** a `ValueError` is raised

---

## 9. Knockoff Voting (`knockoff_filter_voting`)

**WHEN** `knockoff_filter_voting(X, y, niter, fdr, spec, knockoff_method, knockoff_offset, fstat, nlambda, base_seed)` is called
**THEN**:
1. The SDP solution `diag_s` is computed once for `X` and cached for all iterations
2. For each iteration `i` in `range(niter)`:
   a. Set random seed to `base_seed + i`
   b. Generate `X_tilde` via `create_gaussian` using cached `diag_s`
   c. Compute W statistics
   d. Run `knockoff_filter` with `fdr` and `knockoff_offset`
   e. Record the selected variable indices for this iteration
3. Compute selection frequency: `freq_j = #{iterations where j was selected} / niter`
4. Return variables where `freq_j >= spec`

**WHEN** the SDP solution is cached
**THEN** it is computed exactly once per `knockoff_filter_voting` call, not once per iteration

**WHEN** the seed is set to `base_seed + i` before each iteration
**THEN** results are fully deterministic: the same inputs always produce the same selections

**WHEN** `niter=100` and `spec=0.2`
**THEN** a variable must be selected in at least 20 iterations to be included in the output

---

## 10. SLIDE Voting Wrapper (`knockoff_filter_voting_slide`)

**WHEN** `knockoff_filter_voting_slide(X, y, **params)` is called
**THEN** it calls `knockoff_filter_voting` with SLIDE-specific defaults and additionally returns:
- `selected_list`: the per-iteration selection sets from all `niter` iterations
- `freq_selected`: the frequency-thresholded selected set

**WHEN** `slide_selection=False` (default)
**THEN** `freq_selected` is the final output and `findOptIter` is not applied

**WHEN** `slide_selection=True`
**THEN** `findOptIter` is applied to `selected_list` to refine the final selection

---

## 11. `findOptIter` Refinement

**WHEN** `findOptIter(selected_list, freq_selected)` is called
**THEN** it searches all iterations in `selected_list` and returns the iteration whose selection set maximizes Jaccard overlap with `freq_selected`

**WHEN** multiple iterations achieve the same maximum Jaccard overlap
**THEN** the iteration with the smallest selection set size is chosen as the tie-breaker

**WHEN** no iteration has any overlap with `freq_selected`
**THEN** the first iteration's selection set is returned

**WHEN** `findOptIter` output is compared against R's `findOptIter()`
**THEN** selected indices match exactly on the same input data

**WHEN** `slide_selection=False` (default)
**THEN** `findOptIter` is not called and its behavior does not affect the output

---

## 12. Feature Chunking (`select_short_freq`)

**WHEN** `select_short_freq(X, y, **params)` is called and `n > K` (more samples than features)
**THEN** no chunking occurs: a single call to `knockoff_filter_voting_slide` is made on the full `X`

**WHEN** `select_short_freq` is called and `n <= K` (fewer samples than features)
**THEN** `X` is split into chunks of approximately `f_size` columns each

**WHEN** chunking is required
**THEN** the number of chunks and chunk size are computed as:
```
n_splits = ceil(K / f_size)
feature_split = ceil(K / n_splits)
```

**WHEN** `f_size` is not provided
**THEN** it is computed via `calc_default_fsize(n, K)`, which matches R's default `fsize` calculation exactly

**WHEN** chunks are formed
**THEN** the column indices `0..K-1` are partitioned into `n_splits` contiguous groups of size `feature_split` (last group may be smaller)

**WHEN** per-chunk knockoff voting completes
**THEN** the union of all chunk-selected variables forms the candidate set for Stage 2

---

## 13. Two-Stage Screening

**WHEN** `n_splits == 1` (single chunk)
**THEN** Stage 2 is skipped and the Stage 1 result is the final output

**WHEN** `n_splits > 1` (multiple chunks)
**THEN** two-stage screening is performed:
- Stage 1: per-chunk knockoff voting on each chunk of `X`; take the union of selected variables
- Stage 2: knockoff voting on `X[:, union]` (the screened candidates) against `y`

**WHEN** Stage 2 runs
**THEN** `knockoff_filter_voting_slide` is called with the same parameters as Stage 1

**WHEN** the Stage 1 union is empty
**THEN** Stage 2 is skipped and an empty selection is returned

---

## 14. Default `f_size` Calculation (`calc_default_fsize`)

**WHEN** `calc_default_fsize(n, K)` is called
**THEN** it returns the default chunk size such that each chunk is safely underdetermined (n > chunk_size)

**WHEN** output is compared against R's corresponding function on the same `(n, K)` inputs
**THEN** the returned value matches exactly

---

## Parameters

| Parameter          | Default              | Description                                       |
|--------------------|----------------------|---------------------------------------------------|
| `fdr`              | `0.1`                | Target FDR for knockoff threshold                 |
| `spec`             | `0.2`                | Minimum selection frequency across iterations     |
| `niter`            | `100`                | Knockoff voting iterations per chunk              |
| `knockoff_method`  | `"asdp"`             | SDP solver: `"asdp"`, `"sdp"`, or `"equi"`       |
| `knockoff_offset`  | `0`                  | `0` = original knockoff, `1` = knockoff+          |
| `fstat`            | `"glmnet_lambdasmax"`| W-statistic type                                  |
| `nlambda`          | `250`                | Lambda grid size for glmnet lasso path            |
| `base_seed`        | `42`                 | Base random seed; iteration `i` uses `base_seed + i` |
| `slide_selection`  | `False`              | Apply `findOptIter` refinement to final selection |
| `f_size`           | computed             | Chunk size for feature splitting; default from `calc_default_fsize` |

---

## Correctness Benchmarks

**WHEN** Python W statistics are compared against R's `stat.glmnet.lambdamax` on the same data and SDP solution
**THEN** Pearson correlation between W vectors is >= 0.97 for `n > p` cases

**WHEN** Python knockoff selections are compared against R selections on the same data
**THEN** Jaccard similarity between selected sets is >= 0.80 when both use the same SDP solver output

**WHEN** Python `create_solve_asdp` output is compared against R's ASDP solver
**THEN** `diag_s` values match to within `1e-4` absolute tolerance

**WHEN** `findOptIter` is run on identical `selected_list` and `freq_selected` inputs in Python and R
**THEN** the returned iteration index and selected set match exactly

---

## Rust Implementation Notes

### KnockoffCache Struct

Pre-compute and cache the expensive matrix operations once per `knockoff_filter_voting` call, then share immutably across all Rayon threads:

```rust
/// Pre-computed matrices for fast knockoff sampling.
/// Shared immutably across all parallel iterations via &KnockoffCache.
pub struct KnockoffCache {
    /// mu_k = X - (X - mu) @ Sigma_inv @ diag(s)
    pub mu_k: Array2<f64>,
    /// l_trans = cholesky(2*diag(s) - diag(s) @ Sigma_inv @ diag(s)).T
    pub l_trans: Array2<f64>,
}

impl KnockoffCache {
    /// Generate a knockoff matrix X_tilde for a single iteration.
    /// Z is sampled from N(0, I) using a deterministic seed.
    pub fn sample(&self, n: usize, p: usize, seed: u64) -> Array2<f64> {
        let mut rng = Mt19937Rng::seed_from_u64(seed);
        let z = Array2::from_shape_fn((n, p), |_| rng.sample(StandardNormal));
        &self.mu_k + &z.dot(&self.l_trans)
    }
}
```

### Rayon Map-Reduce Voting Pattern

Each iteration is independent — generate knockoffs, compute W, threshold, record selections. The map-reduce pattern avoids locks entirely:

```rust
let selection_counts: Array1<i32> = (0..niter)
    .into_par_iter()
    .map(|i| {
        let seed = base_seed + (i as u64);
        let x_k = cache.sample(n, p, seed);
        let w = stat_lasso_lambdasmax(x, &x_k, y);
        let t = knockoff_threshold(&w, fdr, offset);

        let mut counts = Array1::<i32>::zeros(p);
        for j in 0..p {
            if w[j] >= t { counts[j] = 1; }
        }
        counts
    })
    .reduce(
        || Array1::<i32>::zeros(p),
        |acc, counts| &acc + &counts,
    );
```

### Deterministic Seed Strategy

Each iteration `i` uses `seed = base_seed + i`. This ensures:
1. Full determinism given the same `base_seed`
2. Independence between iterations (different PRNG streams)
3. Reproducibility regardless of thread scheduling or Rayon's work-stealing order
