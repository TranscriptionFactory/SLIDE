# Solver parity audit

## Lambda grid

Gaussian knockoff paths use

```text
k = 0:(nlambda - 1)
lambda[k] = lambda_max * (lambda_min / lambda_max)^(k / nlambda)
```

For `nlambda=500`, this produces exactly 500 strictly decreasing values, starts at `lambda_max`, and intentionally ends above `lambda_min`. `nlambda=1` returns `[lambda_max]`. General Gaussian and binomial endpoint-inclusive paths remain unchanged.

Iteration 0 index 1 after the fix:

- R: `4.3741044063721892`
- Python: `4.3741044063721901`
- Rust: `4.37410440637219`

The maximum R/Rust grid difference over the corrected 20-iteration run is `2.6645352591003757e-15`, below the documented `2e-14` floating-point tolerance.

## Entry-lambda convention

R/Python diagnostics report `lambda*n`; Rust production statistics retain raw lambda. Cross-backend CSV/JSON diagnostics multiply Rust entry lambdas, W, and thresholds by `n=24`. This is reporting-only and does not alter knockoff threshold selection.

## Early stopping

`early_stop_all_entered=true` records a lambda column before checking the all-entered condition, so it cannot revise an entry already observed; it only truncates later columns after every coefficient has entered. It did not trigger in iteration 0 or 14: only 28/27 of 176 coefficients had ever entered.

The actual old truncation was the unconditional df/deviance tail stop: iteration 0 ended at 355 lambdas and iteration 14 at 352, both with 23 nonzero coefficients (`n=24`). Comparison against R found 49 R nonzero entries recorded as zero by Rust across the fixture; several R entries occur below Rust's last retained lambda, proving tail truncation can change entry statistics. Knockoff paths therefore disable df/deviance and all-entered truncation and now retain all 500 requested columns. General Gaussian paths keep existing truncation.

## Audited solver operations

- **Objective normalization:** both use `(1/(2n)) ||y-Xb||² + lambda ||b||₁`; active-coordinate residual correlations target lambda.
- **Standardization:** identical paired design; ddof=1; Rust coefficients are back-transformed while R/Python diagnostic coefficients remain standardized. Comparisons below convert Rust back to standardized coefficient scale.
- **Response/intercept:** y centered; intercept back-transform is `y_mean - x_mean·beta`.
- **Coordinate order:** Rust deterministic ascending candidate/active indices; not changed.
- **Warm starts:** enabled across decreasing lambdas in all backends.
- **Convergence:** first remaining exact operation mismatch; details below.
- **Tolerance/max iterations:** Rust `1e-7` max absolute coefficient change and `100000`; glmnet uses its null-deviance-scaled coordinate-update convergence test. Neither reports a max-iteration failure.
- **Screening:** Rust strong-rule candidates plus full KKT checks; no safe screening. No excluded-coordinate KKT violation at the first material difference.
- **Coefficient-zero treatment/ties:** exact zero determines entry; no tie is involved at the first material difference.
- **Back-transform:** accounted for using Rust `beta_original * column_sd_ddof1`.

## First remaining coefficient difference

The first difference above `1e-12`, after matching grids and coefficient scale, is:

| field | value |
|---|---:|
| iteration | 0 |
| lambda index | 4 |
| lambda | `4.179101941747793` |
| feature | Z44, zero-based augmented index 44 |
| R standardized coefficient | `0.2540963347061765` |
| Python standardized coefficient | `0.25409633470617454` |
| Rust standardized coefficient | `0.2540349151415462` |
| R residual correlation | `4.179057048604609` |
| Python residual correlation | `4.17905704860461` |
| Rust residual correlation | `4.179101939844034` |
| `abs(correlation)-lambda`, R/Python/Rust | `-4.4893143184e-05 / -4.4893143183e-05 / -1.9037607046e-09` |
| convergence status | all returned normally; no max-iteration failure |
| Rust CD iterations | `118127` total over the 500-lambda iteration-0 path |
| R/Python CD iterations | not exposed by native glmnet diagnostics |

Rust's stopping rule is `max(abs(delta_beta)) <= tolerance`; native glmnet stops using a squared, column-norm-weighted coordinate update compared against `thresh * null_deviance`. The R/Python active-coordinate KKT gap is ~`4.49e-05`, while Rust continues to ~`1.90e-09`. This isolates the next solver-level divergence to the convergence criterion, before entry-lambda divergence (for example iteration 0/Z10 enters at grid index 75 in R/Python and 81 in Rust).

A tolerance sweep (`1e-7` through `1e-4`) changed Rust CD iteration totals but did not change its discrete entry lambdas; `1e-3` changed zero treatment abruptly. Therefore changing only the scalar tolerance is not a justified fix.

## Final behavior and next minimal change

Iteration 14/Z10 still disagrees: normalized W `0.9719992044484664 / 0.9719992044484665 / 0.8223245377197124` (R/Python/Rust), producing `false / false / true`. Selected sets differ at iterations 14 and 17; full cumulative votes differ, although Z33/Z44/Z61 remain `6/10/7` for every backend.

Next change should replace only the Gaussian CD convergence test with glmnet's column-norm-weighted squared-update criterion and null-deviance scaling, then rerun this unchanged fixture. Coordinate order, screening, tolerance value, zero threshold, and back-transform should remain untouched until that experiment identifies any subsequent divergence.
