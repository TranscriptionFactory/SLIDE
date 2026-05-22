# LOVE A-Matrix Sparsity Mismatch: Investigation & Resolution

## Summary

Two root causes were identified and fixed, eliminating all A-matrix NNZ divergence
between Rust and Python/R:

1. **Interior-point vs simplex LP solver** — Clarabel (interior-point) produced dense
   solutions; replaced with HiGHS (simplex) to match Python's scipy/HiGHS.
2. **Un-thresholded sigma in non-pure estimation** — Rust recomputed a fresh sigma
   without FDR thresholding for `estimate_non_pure_rows`, while R and Python reuse the
   same FDR-thresholded sigma throughout the LOVE pipeline.

## Results After Both Fixes

| Config (delta, lambda) | Python A NNZ | Rust A NNZ | A max diff | Z max diff |
|------------------------|-------------|------------|-----------|-----------|
| d=0.01, l=0.1          | 72,534      | 72,534     | 6.8e-14   | 4.4e-13   |
| d=0.01, l=1.0          | 66,434      | 66,434     | 2.1e-13   | 4.1e-13   |
| d=0.1, l=0.1           | 19,969      | 19,969     | 4.3e-14   | 1.8e-12   |
| d=0.1, l=1.0           | 2,269       | 2,269      | 4.8e-15   | 6.2e-12   |

All differences are at machine epsilon. A nonzero Jaccard = 1.0 across all configs.

---

## Root Cause 1: Interior-Point vs Simplex LP Solver

**Commit:** `15e94b4` — Replace Clarabel with HiGHS simplex solver in nonpure.rs

Interior-point solvers find solutions in the *interior* of the feasible region, where all
variables tend to be slightly nonzero (~1e-10). Simplex solvers find solutions at *vertices*
of the feasible polytope, where many variables are exactly zero. After L1 normalization,
interior-point artifacts get scaled up, preserving all nonzeros regardless of lambda.

An intermediate fix (`97cd476`) thresholded near-zero values before L1 normalization,
which partially addressed this, but switching to HiGHS was the proper solution.

## Root Cause 2: Un-thresholded Sigma in Non-Pure Estimation

**Commit:** `d6b3603` — Use FDR-thresholded sigma for non-pure row estimation

In `love.rs`, `estimate_non_pure_rows()` recomputed sigma from scratch:
```rust
// BEFORE (bug): recomputed un-thresholded sigma
let sigma = Matrix::from_ndarray(sample_covariance(x.as_ndarray()));
let y = est_y(&sigma, a_hat, pure_vec)?;
```

Both R (`getLatentFactors.R:139`) and Python (`love.py:325`) reuse the same
FDR-thresholded sigma that was computed earlier in the pipeline:
```r
# R: sigma was thresholded at line 67, reused at line 139
sigma_TJ <- estSigmaTJ(sigma = sigma, AI = A_hat, pure_vec = result_AI$pure_vec)
```
```python
# Python: Sigma was thresholded at line 199, reused at line 325
AJ = EstAJDant(C_hat, EstY(Sigma, A_hat, I_hat), ...)
```

FDR thresholding zeros out statistically insignificant off-diagonal entries in sigma.
Without it, the Dantzig LP response vector `y` (computed by `est_y`) has more non-zero
structure, leading to denser LP solutions and ultimately denser A non-pure rows.

**Fix:** Thread the thresholded sigma from `homogeneous_path`/`heterogeneous_path`
through to `estimate_non_pure_rows`, matching R and Python behavior.

### Divergence chain (before fix):

```
un-thresholded sigma (Rust) vs thresholded sigma (Python/R)
  → different est_y / sigma_tj response matrix
    → different Dantzig LP solutions
      → different A non-pure rows (Py: 2,269 NNZ vs Rust: 13,157 for d=0.1,l=1.0)
        → different Gamma (max diff: 0.51)
          → different Z (max diff: 21.6)
```

### What was verified as identical (before fix 2):

| Component | Status |
|-----------|--------|
| K (latent factors) | Identical |
| pure_vec | Jaccard = 1.0 |
| C matrix | max diff = 1.3e-15 |
| Pure A rows | max diff = 0.0 |
| Pure Gamma | max diff = 0.0 |
| est_y/sigma_tj logic | Mathematically identical |
| c_hat, se_est_j, sigma_bar_sup | Mathematically identical |
| Dantzig LP formulation | Mathematically identical |
| Z-matrix formula | Mathematically identical |

---

## Key File References

| File | Lines | What |
|------|-------|------|
| `crates/slide-love/src/love.rs` | 114-118 | Sigma returned from path functions |
| `crates/slide-love/src/love.rs` | 139-149 | Sigma passed to estimate_non_pure_rows |
| `crates/slide-love/src/love.rs` | 421-431 | estimate_non_pure_rows receives sigma |
| `crates/slide-love/src/nonpure.rs` | 180-244 | Dantzig LP (HiGHS simplex) |
| `crates/slide-love/src/nonpure.rs` | 140-181 | est_aj_dant + L1 normalization |
| `src/loveslide/love_python/love/love.py` | 192-202 | Python sigma + FDR thresholding |
| `src/loveslide/love_python/love/love.py` | 325 | Python reuses thresholded Sigma |
| `../SLIDE/R/getLatentFactors.R` | 45-71 | R sigma (cor + FDR threshold) |
| `../SLIDE/R/getLatentFactors.R` | 139 | R reuses thresholded sigma |

## Benchmark Details

Benchmark run: `benchmark_results/2026-05-22_184857_fdr-fix/`

Rust is ~3.9x faster than Python overall with identical LOVE results.
Remaining sig LF count differences at d=0.1 are in the knockoff filtering
stage (separate pipeline, not related to LOVE).
