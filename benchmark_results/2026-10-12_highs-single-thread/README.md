# Parallel Dantzig LPs with single-threaded HiGHS

What changed: `crates/slide-love/src/nonpure.rs` `est_aj_dant` solves the per-feature
Dantzig LPs with rayon and pins each HiGHS solve to one thread.

Why: perf on the K=172 cell put 39% of all samples in `HighsTaskExecutor::run_worker`
(HiGHS's own worker pool spinning) and about 2% in actual simplex work.

Inputs: Sep 22 parity cells (`.tmpfiles/2026-09-22_all-backend-parity-tiefix`), SSc n=24 p=804,
parity knockoff env, 4 rayon threads, BLAS threads 1. `run_compare.sh` is the command
(`ARMS=` selects binaries); binaries and checkpoints live in `.tmpfiles/2026-10-12_nonpure-parallel/`.

Same-node timings (one run each, htc):

| Cell | Baseline 471b42b LOVE / total | Patched LOVE / total |
|---|---|---|
| delta 0.01 lambda 0.1 (K=172) | 26.2 s / 75.3 s | 2.9 s / 51.2 s |
| delta 0.1 lambda 0.1 (K=88) | 2.5 s / 18.4 s | 0.58 s / 16.4 s |

Rayon alone (HiGHS default threads) only moved K=172 LOVE from 31.1 s to 28-29 s.

Equivalence: the full checkpoint (LOVE, Z, marginal, interactions, score) is identical
to the baseline for both cells across two baseline runs, two rayon-only runs and
three patched runs. slide-love tests: 49 pass.
