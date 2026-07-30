# SSc Gaussian knockoff lasso-path fix — 2026-07-30

## Scope and commits

- Attribution/evidence baseline: `217dc4b` (`Attribute SSc divergence with paired X-tilde`).
- Gaussian knockoff grid fix: `6f1e1f1` (`Match Gaussian knockoff grids to R`).
- Complete knockoff path fix: `64b61c7` (`Retain complete Gaussian knockoff paths`).
- Paired diagnostic bundle: `9effb02` (`Record SSc lasso-path parity diagnostics`).
- Input fixtures are byte-for-byte copies of `benchmark_results/20260730_ssc-paired-xtilde`; no knockoffs or swap masks were regenerated.
- The user-owned `.gitignore` modification was preserved and not staged.

## Environment

- Host working tree: `/ix/djishnu/Aaron/1_general_use/omni-SLIDE`
- `rustc 1.94.0 (4a4ef493e 2026-03-02)`
- `cargo 1.94.0 (85eff7c80 2026-01-15)`
- Python `3.11.15`, pixi `dev` environment
- Single-threaded diagnostic execution: `RAYON_NUM_THREADS=1`; original fixture used `OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1`.
- Rust path controls: `SLIDE_KNOCKOFF_NLAMBDA=500`, `SLIDE_KNOCKOFF_TOL=1e-7`, `SLIDE_KNOCKOFF_MAX_ITER=100000`.

## Exact commands

Before editing:

```bash
cargo test -p slide-glmnet --test paired_xtilde_grid_divergence -- --nocapture
```

This passed the old regression named `rust_grid_reproduces_paired_xtilde_r_denominator_divergence`, confirming the known mismatch.

After editing:

```bash
cargo build --release -p slide-rs --bin paired_xtilde

# For each row of the copied xtilde_manifest.csv:
SLIDE_KNOCKOFF_NLAMBDA=500 \
SLIDE_KNOCKOFF_TOL=1e-7 \
SLIDE_KNOCKOFF_MAX_ITER=100000 \
RAYON_NUM_THREADS=1 \
target/release/paired_xtilde \
  benchmark_results/20260730_ssc-lasso-path-fix/inputs/z.raw \
  benchmark_results/20260730_ssc-lasso-path-fix/xtilde/iter_NNN.raw \
  benchmark_results/20260730_ssc-lasso-path-fix/inputs/y.raw \
  benchmark_results/20260730_ssc-lasso-path-fix/inputs/a.raw \
  benchmark_results/20260730_ssc-lasso-path-fix/xtilde/iter_NNN_swap.raw \
  24 88 804 88 \
  benchmark_results/20260730_ssc-lasso-path-fix/inputs/lf_names.txt \
  0.1 \
  benchmark_results/20260730_ssc-lasso-path-fix/diagnostics/rust/iter_NNN.json

cargo test -p slide-glmnet --test paired_xtilde_grid_divergence
cargo test -p slide-glmnet --test r_parity
cargo test -p slide-glmnet --lib
PYTHONPATH=src .pixi/envs/dev/bin/python -m pytest \
  tests/tests/test_knockoff_threshold.py -q
```

## Production changes

1. Gaussian **knockoff** paths now use exactly `lambda_max * ratio.powf(k / nlambda)` for `k=0..nlambda-1`. General Gaussian paths retain their endpoint-inclusive grid; binomial behavior is unchanged.
2. Gaussian knockoff paths retain all requested lambdas. The old df/deviance truncation omitted 145–148 tail lambdas here and changed 49 zero/nonzero entry statuses relative to R across the 20 iterations. General path truncation remains unchanged.
3. Entry-lambda selection semantics remain raw lambda internally. Cross-backend artifacts normalize Rust entry lambdas, W, and thresholds to the R/Python convention `lambda*n` (`n=24`).
4. Knockoff generation was not changed.

## Before/after

- Iteration 0 grid maximum difference, R vs Rust: `0.0032708245936001745` before; `1.7763568394002505e-15` after.
- All 20 Rust paths now contain exactly 500 decreasing lambdas. First lambda equals `lambda_max`; last is intentionally above `lambda_min`.
- Maximum grid difference across all corrected iterations: Rust/R `2.6645352591003757e-15`; Python/R `4.440892098500626e-15`. Grid tolerance is `2e-14`.
- All A, LF-name, X-tilde, Z, and y hashes agree for every consumer and iteration.
- Z33/Z44/Z61 cumulative votes remain identical: R/Python/Rust = `6/6/6`, `10/10/10`, `7/7/7`.
- Remaining Rust selected-set mismatches occur at iterations 14 and 17. Cumulative votes therefore do not fully match.
- Iteration 14/Z10 remains divergent after the proven grid/path-tail fixes: normalized W R/Python/Rust = `0.9719992044484664 / 0.9719992044484665 / 0.8223245377197124`; normalized thresholds = `1.29749307709564 / 1.2974930770956392 / 0.8223245377197124`; selected = `false / false / true`.

Per instructions, solver changes stop here. `solver_parity.md` attributes the next exact difference to the coordinate-descent convergence criterion and proposes the minimal next fix. No speculative solver edit was stacked.

## Artifacts

- `before_after_summary.csv`: backend/iteration hashes, normalized differences, thresholds, selections, parity flags.
- `parity_diagnostics.json`: compact machine-readable normalized comparisons and cumulative votes.
- `diagnostics/{r,python,rust}/`: per-iteration path diagnostics on identical copied fixtures.
- `solver_parity.md`: grid, reporting, early-stop, and next-solver-operation audit.
- `cumulative_votes.json`: corrected-run counts.
