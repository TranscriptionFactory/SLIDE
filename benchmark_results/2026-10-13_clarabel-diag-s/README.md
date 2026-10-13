# Per-chunk diag_s at δ=0.01: Rust Clarabel vs R DSDP vs Python DSDP

Question: on the SSc UnTx δ=0.01 chunk covariances (K=172, 8 chunks of 22, last 18), how far is
Clarabel's `diag_s` from DSDP's, and does tightening Clarabel's tolerance close the gap?

## Inputs and provenance
- Z: `.tmpfiles/2026-10-12_python-scale-bug/results/python/seed_42/python/0.01_{0.1,1.0}_out/z_matrix.csv`
  (24x172, columns `Z0..Z171`; byte-identical across Python seeds 42/1000/2000).
- Σ per chunk (22 contiguous columns, the `f_size=24` split: 172 -> 8 splits of ceil(172/8)=22).
  - λ=0.1: R's own Σ from the sdp-parity run (`rSig_i.txt`, hex); differs from `np.cov` by <=3e-15.
  - λ=1.0: not saved, so `np.cov(chunk)` of the λ=1.0 Z (same recipe; verified equal to R's Σ at λ=0.1).
  - Every solver receives the same Σ (written `%.17g`).
- Raw covariance is fed in, as in R/Python's raw branch. The Rust pipeline may instead route a chunk
  through Schäfer-Strimmer shrinkage (condition gate); that routing is not exercised here.

## Commands
```
ENV=/vast/djishnu/Aaron/.tmpfiles/20260902_knockoff-generator-parity/project/.pixi/envs/dev
export R_LIBS=$PWD/.tmpfiles/2026-10-06_replicates/R-library:/vast/djishnu/Aaron/.tmpfiles/20260902_r-generated-knockoff-backend-evaluation/R-library:$ENV/lib/R/library
export PYTHONPATH=src PYTHONNOUSERSITE=1
$ENV/bin/python benchmark_results/2026-10-13_clarabel-diag-s/driver.py \
    $PWD/.tmpfiles/2026-10-13_clarabel-diag-s $PWD/benchmark_results/2026-10-13_clarabel-diag-s
$ENV/bin/python benchmark_results/2026-10-12_sdp-parity/cmp_s.py .tmpfiles/2026-10-12_python-scale-bug \
    .tmpfiles/2026-10-12_python-scale-bug/results/python/seed_42/python/0.01_0.1_out/z_matrix.csv   # control
```
`driver.py` runs R (`knockoff:::create.solve_asdp`, Rdsdp, gaptol 1e-6; `r_solve.R`), Python
(`loveslide.knockoff.solve.create_solve_asdp`), and Rust per chunk via
`scripts/with-rust-blas-env.sh cargo test --release -p slide-knockoff emit_diag_s -- --ignored`
(default settings = tol 1e-6; `SLIDE_S_TOL`=1e-8, 1e-9, 1e-10). No `cmp_s.py` change was needed.

## Control (task 4): passes
`cmp_s.py` unchanged (`control_cmp_s.txt`): max|s_py−s_R| 2.3e-11 to 3.2e-6 (R-Σ: 2.3e-11 to 8.1e-7),
matching the sdp-parity README (2e-11 to 3e-6). Re-run R reproduces the saved `rsh_i` exactly (0.0 diff).

## Results
Files: `chunk_summary.csv`, `pair_stats.csv` (all pairs), `factors_Z20_Z21_Z73_Z74.csv`, `s_lam*_chunk*.csv`
(all solved s vectors). Near-zero = s/var(Σ_jj) < 1e-6. "Δ" below is max|s−s_R| in correlation units.
Rust at tol 1e-10 failed on all 16 chunks (`InsufficientProgress`/`NumericalError`); at 1e-9 it failed on
8 of 16 (blank below). The default tolerance is 1e-6.

| λ | chunk | near-zero R (Py) | near-zero Rust 1e-6 / 1e-8 | Δ Py | Δ Rust 1e-6 | Δ Rust 1e-8 | Δ Rust 1e-9 | median tiny-s ratio Rust/R, 1e-6 | 1e-8 |
|---|---|---|---|---|---|---|---|---|---|
| 0.1 | 0 | 17 (17) | 17 / 17 | 3.3e-7 | 9.5e-6 | 9.3e-6 | fail | 0.21 | 8e-4 |
| 0.1 | 1 | 17 (17) | 17 / 17 | 6.0e-7 | 1.7e-5 | 2.9e-6 | 1.2e-6 | 0.23 | 1e-3 |
| 0.1 | 2 | 14 (14) | 14 / 14 | 3.4e-9 | 1.4e-5 | 3.6e-6 | 1.7e-6 | 0 | 0 |
| 0.1 | 3 | 14 (14) | 14 / 14 | 2.5e-8 | 1.2e-5 | 8.3e-6 | 7.9e-6 | 0.017 | 9e-4 |
| 0.1 | 4 | 19 (19) | 18 / 19 | 1.6e-8 | 9.9e-5 | 1.2e-4 | fail | 41.6 | 6e-3 |
| 0.1 | 5 | 15 (15) | 15 / 15 | 1.8e-11 | 3.9e-5 | 3.7e-6 | fail | 2.1 | 0.06 |
| 0.1 | 6 | 17 (17) | 17 / 17 | 1.2e-9 | 1.3e-5 | 1.0e-5 | 9.3e-6 | 0.42 | 1e-3 |
| 0.1 | 7 | 5 (5) | 5 / 6 | 8.7e-7 | 9.6e-6 | 1.1e-5 | 1.1e-5 | 0.024 | 5e-4 |
| 1.0 | 0 | 17 (17) | 16 / 17 | 3.0e-11 | 9.6e-6 | 1.9e-6 | fail | 0 | 0 |
| 1.0 | 1 | 17 (17) | 18 / 18 | 7.8e-7 | 1.9e-6 | 1.9e-6 | fail | 0.11 | 5e-3 |
| 1.0 | 2 | 16 (16) | 16 / 16 | 5.2e-8 | 2.5e-5 | 1.3e-5 | fail | 4.6 | 0.017 |
| 1.0 | 3 | 13 (13) | 13 / 14 | 7.6e-12 | 2.8e-6 | 1.9e-6 | 1.9e-6 | 0.57 | 4e-3 |
| 1.0 | 4 | 19 (19) | 17 / 19 | 3.6e-11 | 1.7e-4 | 1.4e-5 | fail | 16.4 | 0.024 |
| 1.0 | 5 | 15 (15) | 15 / 15 | 2.3e-11 | 2.4e-5 | 3.1e-6 | 9.5e-7 | 8.8 | 0.11 |
| 1.0 | 6 | 18 (18) | 18 / 18 | 2.2e-7 | 9.7e-6 | 1.1e-5 | fail | 0.64 | 0.018 |
| 1.0 | 7 | 7 (7) | 7 / 8 | 1.5e-8 | 2.5e-6 | 3.1e-6 | 3.2e-6 | 0.43 | 8e-3 |

Max relative Δ on s>1e-3 entries (full set in `chunk_summary.csv`): Python vs R 5e-10 to 7e-5; Rust 1e-6
vs R 1.5e-5 to 1.1e-3; Rust 1e-8 vs R 1.6e-5 to 1e-3.

### Factors (0-based names as in `z_matrix.csv`; s in covariance units)
Chunk membership is fixed (contiguous column split, a function of K, n and f_size only; Z is
seed-independent): Z20, Z21 are in chunk 0 (local idx 20, 21); Z73, Z74 in chunk 3 (local idx 7, 8).

| λ | factor | R DSDP | Py DSDP | Rust 1e-6 | Rust 1e-8 | Rust 1e-9 |
|---|---|---|---|---|---|---|
| 0.1 | Z20 | 4.57e-9 | 1.68e-8 | 4.54e-8 | 2.26e-10 | fail |
| 0.1 | Z21 | 4.49e-9 | 3.31e-9 | 0 | 0 | fail |
| 0.1 | Z73 | 1.906e-1 | 1.906e-1 | 1.906e-1 | 1.906e-1 | 1.906e-1 |
| 0.1 | Z74 | 1.18e-8 | 1.18e-8 | 0 | 0 | 0 |
| 1.0 | Z20 | 1.77e-10 | 1.77e-10 | 0 | 0 | fail |
| 1.0 | Z21 | 2.54e-10 | 2.54e-10 | 0 | 0 | fail |
| 1.0 | Z73 | 2.296e-1 | 2.296e-1 | 2.296e-1 | 2.296e-1 | 2.296e-1 |
| 1.0 | Z74 | 1.65e-8 | 1.65e-8 | 3.8e-9 | 2.6e-11 | 2.2e-12 |

## Conclusions
- The large-s entries agree (Z73 s=0.19/0.23 in all solvers; max relative Δ <=1e-3). The disagreement is
  confined to the 5-19 near-zero entries per chunk, where Clarabel is off by 10x or more in either
  direction (median Rust/R 0 to 42 per chunk at default tolerance), while Python-vs-R is 0.5-6x in chunks 0-1/6 and ~1.0 elsewhere.
- Clarabel clips several tiny entries to exactly 0 where DSDP returns 1e-10 to 1e-8: Z21 at both λ, Z74
  at λ=0.1, Z20 at λ=1.0. A zero s gives a knockoff identical to its original column. Z20 is 10x R's at
  λ=0.1 (default) and Z74 is 0.2x R's at λ=1.0.
- The near-zero count mostly matches DSDP (differs by 1-2 in 6 of 16 chunks), so the gap is in the magnitudes, not the count.
- Tightening does not move the tiny entries toward DSDP. Δ falls modestly (often 2-10x, e.g. chunk 5 λ=0.1 3.9e-5 to 3.7e-6),
  but median tiny-s ratio drops toward 0 (1e-3 or less at 1e-8), because Clarabel converges the tiny entries toward 0
  instead of DSDP's 1e-10 to 1e-8 values. Tolerance 1e-9 fails on half the chunks and 1e-10 on all.
- Consistent with the plan's hypothesis (Clarabel's tiny-s differ from DSDP's for Z21/Z74/Z20), but this
  only shows S differs; whether it changes selection needs the paired S-only replay (Job 2).

## Caveats
- λ=1.0 Σ is `np.cov` of the saved Python Z; R and Python Z are identical at λ=0.1 (sdp-parity README), assumed at λ=1.0.
- Rust ran on the login node's current build (`cargo test --release`); CPU/BLAS differences can move tiny entries.
- The 3.2e-6 control maximum in `cmp_s.py` is raw covariance units; sdp-parity labels it correlation units.
- The `nz_ratio` column in `pair_stats.csv` is unused (NaN).
