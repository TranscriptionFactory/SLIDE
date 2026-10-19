# How R, Python and Rust handle knockoff s_j ≈ 0, and whether 0 vs ~1e-9 changes W_j

Question (Job 4 of `.agent-handoff/2026-10-13_clarabel-plan.md`): at δ=0.01 the true SDP optimum is s_j = 0 for
Z20/Z21 (`../2026-10-13_clarabel-diag-s/README.md`, Mechanism). R's DSDP returns ~1e-10, Rust's Clarabel returns
exact 0. Does that difference change W_j, and through which step?

## Result
0 vs ~1e-9 does not change W_j. Every s arm gives X̃_j − X_j of 1e-5 to 5e-5 relative, and within one fitter the
W_j distributions match (paired sign agreement 0.89–1.0). The fitter is what differs. When X̃_j ≈ X_j, R's glmnet
and Python's vendored glmnet enter both columns at the same grid λ. That is an exact tie, so W_j = 0 and the
variable can never be selected. Rust's coordinate descent enters only one of the pair. W_j is then ±(entry λ),
with the sign set by the random swap. For Z20 this is the largest |W| in the chunk, and Z20 passes the threshold
in 33% of iterations.

## Code trace (file:line; R line numbers are of the deparsed functions dumped to `.tmpfiles/2026-10-13_s-zero/rsrc/`)
| step | R (SLIDE + knockoff 0.3.x, glmnet) | Python `loveslide` | Rust |
|---|---|---|---|
| 1 post-SDP s | `create.solve_sdp` 38-40 clip [0,1]; 44-54 shrink s·(1−s_eps), s_eps=1e-8·10^k until 2G−diag(s) PD (tol 1e-9); 62 ×diag(Σ). No floor. | `solve.py:345-361` same clip + s_eps shrink + clip. No floor. | `slide-sdp/src/lib.rs:194` `clip_unit_interval` (455) sends raw negatives to 0; 104-108 γ-scale feasibility, floor 1e-10 on λ_min (16, 319-349), not on s. No floor on s. |
| 2 sampler | `create.gaussian` 14 all-zero → return X; 18-22 `chol(Sigma_k)`, **no jitter**: errors on any exact s_j=0 ("leading minor of order 1 is not positive", measured with Rust s) | `create.py:330` all-zero → X; 346 Σ_k; 350-366 chol, on failure +1e-10·max(1,maxdiag)·I, then ×10 | `slide-knockoff/src/lib.rs:1131-1136` Σ_k; 1138-1159 `regularized_cholesky`: +1e-10·max(1,maxdiag)·I (cumulative ×10). Same first step as Python. |
| 3 statistic | `stat.glmnet_lambdasmax` 3-14: random swap; `lasso_max_lambda_glmnet` 6-7 `scale()` X, 19-22 grid λ_max..λ_max/2000, nlambda 500; glmnet default thresh 1e-7; entry = first nonzero ×n; W = pmax·sign(diff), sign(0)=0 | `stats/base.py:41` swap; `stats/glmnet.py:101-120` same scale + grid; 131-140 Fortran glmnet tol 1e-7; 158-162 entry ×n; `base.py:123` np.sign | `slide-knockoff:371-399` swap + sign flip; `slide-glmnet:200-211` grid; cyclic CD in column order (`run_gaussian_cd` 1280ff, KKT loop); tol 1e-4/max_iter 1000 default (517-518); entry not ×n (scale only); `signed_max_statistic` 903 gives 0 on exact tie (b1d0129) |
| 4 threshold/vote | `knockoff.threshold` 6-10 ts=c(0,|W|), offset 0 (SLIDE `secondKO` 32); `knockoff.filter` 50 W ≥ t; `selectShortFreq` 41 count ≥ niter·spec | `filter.py:97-110` same; `knockoffs.py:194` W ≥ t; 459 count ≥ spec·niter | `knockoff_threshold` 203-233 same; `build_voting_result` 702-706 count ≥ ceil(niter·spec) |

A W_j = 0 is counted on both sides of the ratio at t = 0 but is selected only if t = 0. A ±|W| coin flip is
selected whenever its sign is + and |W| ≥ t, so for a top-entry variable it is selected in about half of all iterations.

## Numerical check (`check.py`, `r_stat.R`; outputs `summary.csv`, `arms.csv`, `paired_sign_agreement_vs_rust_s.csv`)
Chunk 0 (Z0–Z21) at δ=0.01 λ=1.0: Z and Y from `.tmpfiles/2026-10-13_s-only-d001/designs/marginal_chunk_1`; Σ =
`sigma_1.0_0.csv` (asserted equal to np.cov(Z)); s from `../2026-10-13_clarabel-diag-s/s_lam1.0_chunk0.csv`.
200 draws, Python sampler for every arm. Innovations (`np.random.seed(20261013+d)`) and swap masks are shared
across arms. Each X̃ is scored with three fitters: Python glmnet, R `knockoff:::lasso_max_lambda` (real glmnet),
and the Rust `paired_xtilde` binary from `.tmpfiles/2026-10-13_s-only-d001/full/bin`. The Rust fitter runs at
replay settings (nlambda 500, tol 1e-7) and at the production default (tol 1e-4). Near-zero = s_R/Σ_jj < 1e-6
(17 entries). The Rust sampler was not run; it matches Python's to 1.9e-9 on shared inputs (2026-09-28 audit),
and its jitter step is the same.

```
ENV=/vast/djishnu/Aaron/.tmpfiles/20260902_knockoff-generator-parity/project/.pixi/envs/dev
PYTHONPATH=src R_LIBS=$PWD/.tmpfiles/2026-10-06_replicates/R-library:$ENV/lib/R/library \
 bash scripts/with-rust-blas-env.sh $ENV/bin/python benchmark_results/2026-10-13_s-zero-handling/check.py \
 $PWD/.tmpfiles/2026-10-13_s-zero $PWD/benchmark_results/2026-10-13_s-zero-handling 200 \
 $PWD/.tmpfiles/2026-10-13_s-only-d001/full/bin/paired_xtilde $ENV/bin/Rscript      # 1m46s on the login node
```

Z20 (W fractions +/0/−, fraction selected; Rust = production tol 1e-4):

| s arm | chol jitter | ‖X̃−X‖/‖X‖ med | R glmnet | Python glmnet | Rust |
|---|---|---|---|---|---|
| Rust s (Z20=Z21=0) | yes | 1.1e-5 | .005/.99/.005, 0 | .005/.99/.005, 0 | .52/.005/.475, .33 |
| R s (1.8e-10, 2.5e-10) | no | 2.0e-5 | .005/.98/.015, 0 | .005/.98/.015, 0 | .52/.005/.475, .335 |
| R s, near-zero → 0 | yes | 1.1e-5 | .005/.995/0, 0 | .005/.995/0, 0 | .52/.005/.475, .33 |
| Rust s, near-zero floored 1e-9 | no | 4.8e-5 | .02/.95/.03, .005 | .025/.94/.035, .01 | .52/.005/.475, .335 |

Z21: R and Python give W=0 in 93–99% of draws and select it in ≤1%. Rust gives W=0 in 60–73% of draws and
selects it in 5–10% (higher at tol 1e-7). Z21 has the chunk's largest entry λ (99.4); Z20 is next (94.9).
Whenever Rust's W_j ≠ 0, its sign follows the swap bit in 100% of draws. In other words, the column placed first
in [X|X̃] wins.

## Caveats
- Why Rust's CD admits only one of two near-identical columns, when glmnet admits both at the same λ, is
  **inferred** to come from cyclic-order updates plus the KKT/active-set check. Neither was traced line by line.
- The selection fractions are per-iteration rates for this chunk's knockoff filter, not SLIDE vote rates. Job 2
  (job 11935828) holds the 1000-draw paired vote replay.
