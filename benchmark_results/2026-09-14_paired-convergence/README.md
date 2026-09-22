# Paired solver convergence study: high-delta/high-lambda cell

Submitted as **SLURM array 11242920**, tasks 0–4, at 09:51 EDT on September 14.
All five tasks completed successfully on September 14 in 1m28s–2m51s, with
1,000 paired draws per design and all hash checks passed. Frozen source: `550d2ea`.
Production solvers are unchanged. See the [September 22 readout](../2026-09-22_paired-convergence-readout/README.md).

## Question and sample size

Does the Rust/Python/R marginal-selection discrepancy persist when the design,
knockoff draws, and swap masks are identical? Is it sensitive to solver tolerance,
iteration budget, or tied variable-entry statistics?

Use **1,000 paired draws per design**, not the legacy 2/20-draw smoke defaults.
Five designs mean 5,000 distinct saved realizations, each fitted by R, Python,
and three Rust solver configurations (25,000 paths). The tie-zero arm reuses the
baseline path; it is not another solver fit.

| Design | Shape | Features |
| --- | --- | --- |
| marginal_chunk_1 | 24 × 22 | Z0–Z21 |
| marginal_chunk_2 | 24 × 22 | Z22–Z43 |
| marginal_chunk_3 | 24 × 22 | Z44–Z65 |
| marginal_chunk_4 | 24 × 22 | Z66–Z87 |
| rust_stage_two_union | 24 × 11 | Z0, Z5, Z25, Z33, Z39, Z44, Z61, Z66, Z69, Z76, Z87 |

These are the actual balanced marginal chunks and saved Rust second-stage union
from the September 14 parity run at delta=0.1, lambda=1. The old full-88-column
replay does not represent those executed designs. Preparation verifies frozen Y
hash and rowwise Z/A checkpoint agreement before relabeling positional V1…V24
rows with response sample IDs; `designs.json` records provenance and tolerances.
No LOVE regeneration. This is a fixed-design marginal diagnostic, not another
end-to-end benchmark or an interaction replay.

Each design has an independent declared seed, 42 + 10000 × array index. Within a
design, every solver reads byte-identical R-generated knockoffs and swap masks.
Hashes verify A/Z/y, feature order, knockoffs, and swaps. R covariance/ASDP setup
runs once per design; `sampler.csv` records any covariance shrinkage/equi fallback.
All fits use 500 lambda points, ratio 1/2000, Gaussian response, FDR .1, offset 0.

## Solver adequacy and draw-count stability

| Fit | Tolerance | Iteration cap value |
| --- | ---: | ---: |
| R/Python compiled glmnet | 1e-7 | 100000 |
| Rust baseline | 1e-7 | 100000 |
| Rust higher cap | 1e-7 | 1000000 |
| Rust tighter tolerance, same higher cap | 1e-9 | 1000000 |

Rust uses production cached preparation. Higher-cap vs baseline changes only
the cap; tighter vs higher-cap changes only the tolerance. The diagnostic CLI
now binds its tolerance/cap arguments instead of silently inheriting fast-path
defaults. Rust's cap is not semantically equivalent to Fortran's global sweep
budget, so saved coefficients also receive an independent lasso KKT-residual
audit (absolute and lambda-normalized). R error/sweep codes and Python warnings
are retained. KKT stationarity does not imply identical coefficients for a
nonunique lasso optimum, nor is it the same as the update-based stopping metric.

Nested summaries at **100, 250, 500, and 1,000** report vote frequencies, pointwise
95% Wilson intervals, frequency changes, threshold-ambiguous features, and
findOptIter outputs. At 1,000 independent draws, the worst-case pointwise 95%
frequency margin is approximately 3.1 percentage points; it is not a simultaneous
bound across features/designs. Near-threshold decisions or moving prefixes remain
unresolved, even if the run completes. More draws do not by themselves guarantee
stability of findOptIter's single-draw choice.

## Confirmed replay finding; not yet a production fix

The smoke test exposed positive tied-entry W values in Rust. Both production
cached and uncached paths compute `(original - knockoff).signum()`, where Rust's
positive zero maps to +1. R/Python mathematical sign(0) maps to 0. A deterministic
production fixture gives entries 0.8529598164112223 / 0.8529598164112223 and
W=+0.8529598164112223. The first actual union smoke draw also contains such a tie.

The study preserves this production behavior. `rust_500_tie_zero` is an explicitly
labeled post-fit counterfactual: same baseline coefficients/entry times, tied W
set to zero, then the same threshold and aggregation. It measures how much this
mechanism explains without mixing a fix into the baseline. Other differences may
remain from coefficient convergence or entry ordering; the 1,000-draw results
must distinguish them before any production change.

## Execution and verification

Five-task SLURM array; one CPU and 16 GB per task, BLAS/Rayon=1, 12-hour limit,
cluster htc, Turin. No builds inside the job. Frozen source, prepared inputs,
release diagnostic binary, and SLIDE package under:
`.tmpfiles/2026-09-14_paired-convergence/`.
Shared Python/R dependencies match the benchmark environment.

Preflight: 11 Python tests; production tied-entry Rust test; cached/uncached
agreement at three tolerance/cap settings; all five real designs replayed for
two draws with hashes/configuration/KKT checks; R sampler metadata check;
shell syntax and 334 frozen-file hashes passed. Smoke draws are not adequacy evidence.

Binary SHA256: `0c338ede3bd7aed0e15d490f7b385c149db42a2bae2338e98884971d1ea420ab`.

```bash
RUN="$PWD/.tmpfiles/2026-09-14_paired-convergence"
ENV=/vast/djishnu/Aaron/.tmpfiles/20260902_knockoff-generator-parity/project/.pixi/envs/dev
R_DEPS=/vast/djishnu/Aaron/.tmpfiles/20260902_r-generated-knockoff-backend-evaluation/R-library
sbatch --output="$RUN/logs/slurm-%A_%a.out" "$RUN/run.sbatch" "$RUN" "$ENV" "$R_DEPS"
```

Per-design results include `paired_metrics.csv`, `convergence.csv`,
`prefix_stability.json`, `solver_sensitivity.json`, `pairwise_draw_jaccard.csv`,
and `attribution_summary.json`, plus original draw/fit diagnostics. Require five
successful 1,000-draw summaries and hash checks before the final readout. Do not
interpret the fixed Rust union as the union the other solvers would select.
