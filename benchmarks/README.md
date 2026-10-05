# Backend benchmark configurations

Use `--configuration parity` for a Gaussian configuration-matched comparison.
The default, `--configuration native`, preserves backend solver defaults and
inherited overrides. Neither mode pairs random draws across implementations.
Parity defaults to base seed 42; `--knockoff-base-seed` can change it.

| Setting | Parity | Native Rust default |
| --- | --- | --- |
| Gaussian knockoff λ points | 500 | 250 |
| CD tolerance | 1e−7 | 1e−4 |
| Iteration limit value | 100,000 | 1,000 |
| Gaussian λ ratio | 1/2000 for all dimensions | 1e−4 when n > 2p, otherwise 1/2000 |
| Raw covariance gate | Positive definite | Positive definite and condition number ≤ 100,000 |
| Default workers | 4; serial Python uses 1 | 10; serial Python uses 1 |

Here p counts original columns before adding knockoffs. The Gaussian grid uses
λk = λmax × (1/2000)^(k/500), k = 0,…,499; it excludes the lower endpoint.
Python/R already use the parity Gaussian settings. Parity requires compiled
Python glmnet rather than its sklearn fallback. Binary-response parity is not
implemented; `--y-factor` with parity fails before running.

Parity sets Rust's `SLIDE_KNOCKOFF_TOL`, `SLIDE_KNOCKOFF_MAX_ITER`,
`SLIDE_KNOCKOFF_MATCH_R_COV`, and `SLIDE_KNOCKOFF_MATCH_R_LAMBDA` explicitly,
overriding inherited values in child processes. It rejects conflicting Rust
λ-count/FDR arguments and makes the R selection flag follow `--slide-selection`.
Both modes now honor pure R's requested worker budget without modifying the
installed R package. SLURM submission rejects a worker budget above its CPU request.

## Running

**Rebuild the Rust executable from this source before timing.** Older executables
ignore the new λ-range environment setting; the harness cannot detect that.
Build outside the benchmark timer and pass `--rust-bin` for usable timing results.
Use the existing project BLAS/environment setup as needed.

```bash
cargo build --release -p slide-rs
python benchmarks/benchmark_backends.py \
  --configuration parity --threads 4 --knockoff-base-seed 42 \
  --rust-bin "$PWD/target/release/slide" \
  --out-dir benchmark_results/2026-09-14_gaussian-parity
```

Add input paths as needed. Use a fresh output directory for each configuration;
do not combine saved native and parity timings. For a frozen source copy, pass
`--source-revision <executed-commit>`; metadata keeps that declaration separate
from `enclosing_git_sha`. Metadata also records inherited SLIDE environment
variables, Rust solver overrides, λ counts, and worker budgets. It records
configuration, not a measurement of achieved convergence.

For native defaults, unset solver override variables before selecting
`--configuration native`. Native mode deliberately preserves inherited overrides.
The September 12 frozen executable and run remain native-configuration results.

## Interpretation and next measurement

Matching tolerance values does not establish equal work. The Gaussian stopping
metrics use comparable variance-scaled coefficient updates, but Rust applies its
iteration cap per λ and candidate-set refit. Fortran glmnet counts sweeps across
the whole path. Rust currently returns coefficients on cap exhaustion without a
convergence error; matching the cap value does not remove this limitation.

Keep the faster Rust defaults available: the tighter tolerance can cost much
more runtime. Profile the parity path on saved inputs before considering another
solver dependency. Compare convergence and W statistics as well as time.

Remaining benchmark limitations: backend-specific RNG streams and swaps;
Python excludes imports from timing while Rust/R include subprocess startup;
one pass per cell in fixed order. A future equal-work study needs paired saved
draws, convergence diagnostics, aligned timing boundaries, repeated cells, and
randomized backend order. No full parity benchmark has been run for this change.

## Saved-draw S-only selection replay

`replay_paired_xtilde.py --sampler-audit <saved-audit-root>` compares the
sampler audit's `rust_shared_sigma_draws.raw` (Rust S) against
`rust_shared_s_draws.raw` (R S). Both arms use Rust's Gaussian transform,
the saved common covariance and innovations, and identical saved swap masks.
Pass the paired study root with `--fixtures`, the design names with `--designs`,
the verified `paired_xtilde` executable with `--rust-bin`, and a new `--out`.
Optional `--draws N` selects a prefix for smoke checks; omit it for all draws.
The original tie-fix replay remains the default when `--sampler-audit` is absent.

The S-only mode checks input hashes, reconstructs draws from common innovations,
repeats each arm's first fit, and checks tied-entry statistics and thresholds.
It saves per-draw fits/W, paired vote differences and Monte Carlo SE,
prefix/findOptIter summaries, input hashes, and a vote-difference figure.
This diagnoses fixed-design sensitivity, not full-pipeline parity or FDR control.
