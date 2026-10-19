# Glmnet / Lasso Statistics

## 1. Lasso Path (Core)

**WHEN** the lasso path solver is invoked with `X` (`n x p`), `y` (`n x 1`), and a lambda grid,
**THEN** run coordinate descent matching the Fortran glmnet algorithm and return a coefficient matrix of shape `(p, nlambda)` and the corresponding lambda values.

**WHEN** `family="gaussian"` (continuous `y`),
**THEN** minimize the squared-error loss with L1 penalty.

**WHEN** `family="binomial"` (binary `y`),
**THEN** minimize the logistic log-likelihood with L1 penalty.

**WHEN** `alpha` is set (elastic net mixing parameter, default `1.0`),
**THEN** apply the combined L1/L2 penalty `alpha * ||beta||_1 + (1 - alpha)/2 * ||beta||_2^2`; `alpha=1.0` is pure lasso.

**WHEN** `standardize=True` (default),
**THEN** standardize columns of `X` to unit variance within the solver before fitting, and back-transform coefficients to the original scale before returning.

**WHEN** fitting proceeds along the lambda path,
**THEN** use warm starts: initialize each lambda's coordinate descent from the solution at the previous (larger) lambda.

**WHEN** active set cycling is enabled,
**THEN** iterate only over the current active set of non-zero coefficients until convergence, then check all predictors for violations of the KKT conditions, and expand the active set if needed.

**WHEN** the coordinate descent update for a single coefficient changes by less than `1e-7` (convergence threshold),
**THEN** treat that coefficient as converged for that lambda.

---

## 2. Lambda Grid

**WHEN** a lambda grid is computed,
**THEN** use `nlambda=250` values (matching R's default).

**WHEN** computing `lambda_max`,
**THEN** set `lambda_max = max(|X.T @ y|) / n`.

**WHEN** `n >= p`,
**THEN** set `lambda_min_ratio = 0.01`.

**WHEN** `n < p`,
**THEN** set `lambda_min_ratio = 0.05`.

**WHEN** the grid is constructed,
**THEN** produce a geometric (log-linear) sequence from `lambda_max` down to `lambda_max * lambda_min_ratio`.

---

## 3. `stat_glmnet_lambdasmax` (Primary W-Statistic)

**WHEN** `stat_glmnet_lambdasmax` is called with original features `X`, knockoff features `X_tilde`, and response `y`,
**THEN** fit a single lasso path on the augmented matrix `[X, X_tilde]` (shape `n x 2p`) against `y` using `nlambda=250`.

**WHEN** the lasso path is fit,
**THEN** for each variable `j` (both original and knockoff), identify `lambda_entry_j` as the largest lambda value at which `coeff_j` first becomes non-zero.

**WHEN** a variable never enters the model across the full lambda path,
**THEN** set its `lambda_entry` to `0`.

**WHEN** W-statistics are computed,
**THEN** set `W_j = lambda_entry(original_j) - lambda_entry(knockoff_j)` for each `j` in `1..p`.

**WHEN** `W_j` is large and positive,
**THEN** this is interpreted as evidence that original variable `j` is important relative to its knockoff.

---

## 4. `stat_glmnet_lambdadiff`

**WHEN** `stat_glmnet_lambdadiff` is called,
**THEN** follow the same augmented lasso path procedure as `stat_glmnet_lambdasmax`, but use the lambda at which each coefficient is maximized (rather than first entry) to compute `W_j`.

---

## 5. `stat_glmnet_coefdiff`

**WHEN** `stat_glmnet_coefdiff` is called,
**THEN** select `lambda_opt` by cross-validation on the augmented `[X, X_tilde]` lasso.

**WHEN** `lambda_opt` is determined,
**THEN** set `W_j = |coeff_j(lambda_opt)| - |coeff_tilde_j(lambda_opt)|` for each `j`.

---

## 6. Vendored Glmnet

**WHEN** running in the Python implementation,
**THEN** use the vendored Fortran glmnet wrappers (`linear.py`, `logistic.py`) as the backend for the lasso path.

**WHEN** porting to Rust,
**THEN** implement native coordinate descent; the vendored Fortran wrappers are not available, and the Rust implementation must reproduce the same algorithm.

---

## 7. Correctness Requirements

**WHEN** the lambda path is computed on the same `X` and `y` as R's `glmnet`,
**THEN** the lambda sequence must be identical (to floating-point precision) to the R output.

**WHEN** coefficient values at each lambda are compared to R's `glmnet` output,
**THEN** the maximum relative difference must be less than `1e-6` for well-conditioned problems.

**WHEN** W-statistics produced by `stat_glmnet_lambdasmax` are compared to R's equivalent on the same inputs,
**THEN** the Pearson correlation between Python and R W-statistics must be `>= 0.97` for well-conditioned problems.
