# R-parity fixes to Python and Rust SLIDE

A source audit of the six benchmark routes under `--configuration parity` found that all
backends get the same nominal settings (δ, λ, thresh_fdr 0.2, spec 0.1, fdr 0.1, 500
draws, offset 0, lambdasmax statistic with 500 lambdas). It also found three places where
Python or Rust does something different from the vendored R SLIDE
(`vendor/SLIDE-jishnu/R`, which is byte-identical to the installed package):

| Fix | R reference | Before | After |
| --- | --- | --- | --- |
| Raw Z into knockoffs (Python) | `secondKO.R` takes `cov()` and shrinkage of unscaled Z | `Knockoffs.__init__`, `add_z1` and `select_short_freq_slide` ran `StandardScaler` first | raw Z and raw interaction products |
| Stage-two empty frequent set (Python, Rust) | `selectShortFreq.R` checks `is.null(freq_vars)`; `character(0)` is not NULL, so `findOptIter` returns the first shortest draw | returned nothing | first shortest draw |
| Marginal model size (Python, Rust) | `marginalSLIDE.R` adds 0.1 to spec until fewer than n marginals remain | no loop | same loop; returns nothing once spec passes 1 |

Not changed:
- Rust residualizes y through the origin, while R's `lm` includes an intercept. Z is
  centered and glmnet fits an intercept, so the selections are the same.
- The SDP solvers differ (Clarabel vs DSDP), and so do the RNG streams. These differences are by design.

## Checks (job scripts in this directory)

Tests (`tests.sbatch`, job 11917474): nine knockoff and pipeline test files, run on
frozen `HEAD` and on the patched tree. `HEAD` had 87 failed/93 passed/22 errors; the
patched tree had 86 failed/94 passed/22 errors. The two failure lists are the same except
that `test_parallel_different_random_seeds` passes after the patch. The pre-existing
failures are stale generated tests. The new test
`test_slide_knockoffs_receive_raw_latent_factors_like_r_secondko` fails on `HEAD` and
passes after the patch. `cargo test -p slide-knockoff -p slide-core` passes.

Real data (`run.sbatch` job 11917475 and `attrib.sbatch` jobs 11918538/11918541): SSc
inputs, the four parity cells, and the 2026-10-06 replicate settings on htc Turin nodes.
- **Rust**, seeds 42 and 1000: all 8 cell × seed selections (factors and interactions)
  equal the 2026-10-06 replicate outputs. Neither new rule fired, and the HiGHS and
  LoveCache speedups left the selections unchanged.
- **Python**, single-fix variants at seed 42 (`attrib.sbatch`): unmodified `HEAD`
  reproduces the 2026-09-22 Python results. The stage-two and marginal-size rules change
  nothing. Every change comes from passing raw Z.
- **Python**, multiple seeds (`attrib_seeds.sbatch`, job 11918559): `HEAD` at seeds
  42/1000/2000/3000 and raw Z at seeds 42/1000/2000/3000/4000. Marginal factors:

| Cell | `HEAD` Python | Raw-Z Python | R, 20 runs (2026-10-06 `factor_rates.csv`) |
| --- | --- | --- | --- |
| δ0.01 λ0.1 | 18 20 27 45 73 136 (4/4) | 18 20 45 73 96 (5/5) | 18 27 45 136 20/20; 74 17/20; 20 3/20; 73 0/20 |
| δ0.01 λ1.0 | none / 18 96 136 ×2 / 18 45 74 96 136 | 45 96 136 (5/5) | 18 27 20/20; 45 136 19/20; 73 13/20 |
| δ0.1 λ0.1 | 44 61 ×2 / 25 44 61 / 25 33 38 44 53 61 69 | 44 61 ×4 / 25 44 61 | 44 20/20; 25 18/20; 53 15/20; 61 2/20 |
| δ0.1 λ1.0 | 0 5 25 ×3 / 0 5 25 62 | 0 5 44 ×3 / 5 44 ×2 | 44 20/20; 33 16/20; 25 14/20; 5 0/20 |

**Interpretation.**
- **Raw Z is the R method.** R passes unscaled Z to `secondKO`, and Rust already did the
  same. Python's covariance equals `cov`/`corpcor` to about 1e-15 on both scales.
- **Raw Z does not bring Python closer to R overall.** Python changes little across seeds
  either way, but raw Z shifts the stable result. At δ0.1 λ1.0 it gains Z44 (R 20/20).
  At δ0.01 λ1.0 it removes an empty run. At δ0.01 λ0.1 it loses Z27 and Z136 (R 20/20).
- **Python carries its own systematic deviation from R, before and after.** It selects
  Z96, Z0 and Z5, which R never selects, and it misses Z27 or Z18 at δ0.01.
- **Python is scale-sensitive, which points to another bug.** Z column SDs are
  0.71–2.63, the SDP works on the correlation matrix, and the statistic standardizes its
  inputs. A rescale this small should not move a seed-stable selection. Some step in
  Python's sampler or statistic depends on column scale, and it is unidentified. Likely
  candidates are the Cholesky jitter escalation in `_prepare_knockoff_cache`
  (`filter.py:526-540`) and the glmnet lambda path. This is the next investigation for
  Python.

Scratch outputs: `.tmpfiles/2026-10-12_parity-fix-check/` (frozen trees, logs, results).
