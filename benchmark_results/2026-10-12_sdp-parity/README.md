# Python knockoff SDP now matches R's DSDP

**Python now selects R's latent factors at δ=0.01.** Before this change, Python's
selections shifted under a column rescale that should not matter
([2026-10-12 parity fixes](../2026-10-12_parity-fixes/README.md)). The cause was the
knockoff SDP: Python's DSDP call did not reproduce R's.

## Mechanism

- **Most knockoffs at δ=0.01 are near-copies.** Each marginal chunk has 22 factors and
  only n=23 samples, so the correlation matrix G is nearly singular. Because s ≤
  2·λmin(G), DSDP sets s_j/var_j ≈ 1e-10 for 14–19 of the 22 factors in every chunk,
  in R and Python alike.
- **Tiny solver differences decide selection.** For those factors the knockoff mean
  shift (X−μ)Σ⁻¹diag(s) depends on the ratio s/λmin, so s values near 1e-10 set which
  of X_j and its knockoff enters the lasso first. A 2× difference in a tiny s_j is
  enough to change which factors are selected. Each implementation is stable across
  seeds, but different implementations disagree.
- **Python's DSDP results differed from R's in two ways:**
  1. **SDPA sign convention.** `pydsdp`'s `sedumi2sdpa` writes b and A negated, while
     Rdsdp writes them unnegated. The problems are equivalent, but DSDP reaches them
     along different paths. Run on R's own file, Python's DSDP build matches Rdsdp to
     3e-7. Through the negated file the gap was 9e-6, and near-zero s entries differed
     3–20×.
  2. **The `is_posdef` rule.** R's `knockoff:::is_posdef(A, tol = 1e-9)` returns
     `lambda_min > tol * 10`; Python returned `lambda_min > tol`. The rule decides
     s_eps in the SDP feasibility loop, where s is scaled by (1−s_eps). It also gates
     raw versus shrunk covariance in R's `secondKO`. Rust's covariance gate used 1e-9
     with a comment that misread R the same way.

## Changes

| File | Change |
| --- | --- |
| `src/loveslide/knockoff/pydsdp/convert.py`, `dsdp5.py` | Write Rdsdp's SDPA file: unnegated b and A, `%.15g` entries (`write.table`) and `%.7g` b (`cat`). Negate the returned y. |
| `src/loveslide/knockoff/utils.py` | `is_posdef` returns `lambda_min > tol * 10`, as R does. |
| `src/loveslide/knockoff/solve.py` | DSDP `maxit` defaults to R's 1000 instead of 10000. It never binds here (no effect on s). |
| `crates/slide-knockoff/src/lib.rs` | Covariance positive-definite gate set to 1e-8 (R's 10·tol). |
| `tests/tests/test_knockoff_corpcor_routing.py` | The R-fixture `diag_s` tolerance tightens from 1e-5 to 1e-10. The previous commit fails it (4.7e-7) and this one passes (9.9e-13). |

## Evidence

**SDP output against R** (`cmp_s.py`, `tiny_s.py`). These use the δ=0.01 λ=0.1 Z
(identical in R and Python), split into eight 22-factor chunks, with R
`create.solve_asdp` output saved in hex.

| | max \|s_py − s_R\| (correlation units) | Near-zero s, Python/R ratio |
| --- | --- | --- |
| Before | 1e-6 to 1e-4 | 0.04–20× in 6 of 8 chunks |
| R `is_posdef` only | 2e-8 to 1e-5 | — |
| Both fixes | 2e-11 to 3e-6 | 0.94–1.15 in chunks 2–7; 0.24–10 in chunks 0–1 |

Chunks 0–1 still differ because the two DSDP builds differ: the same SDPA file gives
3e-7 between them. At gaptol 1e-6 those near-zero s values are solver noise in R too, so
only R's exact DSDP binary would reproduce them. Under a rescale of Z, chunks 2–7 are
now invariant (0.94–1.04, `scale_s.py`); chunks 0–1 are not.

**End-to-end** (`run.sbatch`, job 11918941; 2026-10-06 parity settings; Turin nodes).
- **Rust:** seeds 42 and 1000 give exactly the 2026-10-06 selections; the gate change
  never fires on this data.
- **Python:** five seeds before and after the fix, scored against R's 20 runs (stock +
  doRNG, 2026-10-06 `factor_rates.csv`). "Core" means factors R selects in ≥16/20 runs.

| Cell | Before (`aae2db9`): picks ×5 seeds; recall of R core | After: picks; recall | R, 20 runs |
| --- | --- | --- | --- |
| δ0.01 λ0.1 | 18 20 45 73 96 ×5; 0.40 | 18 27 45 136 ×5, 74 ×4; **0.96** | 18 27 45 136 20/20, 74 17/20 |
| δ0.01 λ1.0 | 45 96 136 ×5; 0.50 | 18 27 45 136 ×5, 73 ×2, 57 ×1; **1.00** | 18 27 20/20, 45 136 19/20, 73 13/20, 57 7/20 |
| δ0.1 λ1.0 | 5 44 ×5, 0 ×3; 0.50 | 44 ×5, 69 ×3, 33/25 ×2; 0.70 | 44 20/20, 33 16/20, 25 14/20, 69 11/20 |
| δ0.1 λ0.1 | 44 61 ×5, 25 ×1; 0.60 | 61 ×5, 44 ×3, others 1–2; 0.50 | 44 20/20, 25 18/20, 61 2/20 |

- **δ=0.01:** Python now reproduces R's distribution: the same core set, and Z73/Z57 at
  rates consistent with R.
- **δ=0.1:** Python now varies across seeds as R does, and it no longer selects Z0 or
  Z5. Z61 at δ0.1 λ0.1 (5/5 against R's 2/20) is the remaining Python outlier. Five
  seeds is too few to test, so the replicate experiment should settle it.

## Not addressed

**Rust uses Clarabel, not DSDP.** Its near-zero s values cannot match R's, which likely
explains why Rust picks Z20 at δ=0.01. One route is to link the bundled DSDP C source
(`src/loveslide/dsdp_solver/dsdp`) into `slide-sdp` and write Rdsdp's SDPA problem, as
Python now does. Another is to run Rust's existing R-knockoff backend in the experiment.
However, Rust with R knockoffs selected Z20 at seed 42 on 2026-09-22, so the fitter may
contribute as well.

Scratch: `.tmpfiles/2026-10-12_python-scale-bug/` (frozen tree, R hex outputs, results).
