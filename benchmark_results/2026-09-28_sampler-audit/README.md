# Rust/R sampler and random-stream audit

The saved five-design audit found no large covariance or Gaussian-transform
mismatch. With identical covariance Σ, diagonal S and Gaussian innovations,
Rust and R samples agree within **1.89 × 10⁻⁹** across 5,000 draws.
Native SDP solutions differ slightly, and the native RNG streams differ by
design. This does **not** establish that those small SDP differences have no
selection effect, or that the remaining full-pipeline gap is entirely random.

We also reproduced an R worker-seeding gap: the runner seeds the main process,
but the frozen installed SLIDE entry point creates PSOCK workers without
seeding them. Historical nominal seed labels therefore do not establish
reproducible worker streams.

## Scope and controls

- Source baseline: `da77527`, plus the diagnostic-only extension of
  `tests::emit_diag_s` in `crates/slide-knockoff/src/lib.rs`.
- Four saved 24 × 22 marginal chunks and one fixed 24 × 11 Rust stage-two union,
  from the September 14 paired study, at δ=0.1 and λ=1.
- 1,000 draws per design per sampler arm; seeds 42 through 1041, reseeded for
  each draw. Same seed range for each design is intentional for this diagnostic;
  it is not the production chunk-offset schedule, nor five independent studies.
- Rust uses `SLIDE_KNOCKOFF_MATCH_R_COV=1`, as in the parity benchmark. All five
  designs use sample covariance, so this does not test the shrinkage branch.
- Native Rust arm calls `prepare_knockoff_cache` and `KnockoffCache::sample`.
  Shared-Σ arm uses R's saved covariance with Rust's ASDP S solution.
  Shared-Σ/S arm uses R's saved covariance and S. The identity-cache arm obtains
  innovations through the same production `sample` function, not a separate RNG.
- Covariance and S are serialized with 17 significant digits for float64
  round-tripping; both languages receive the same numerical inputs.
- R calls installed `knockoff::create.solve_asdp` and `create.gaussian`;
  R 4.5.3, knockoff 0.3.6. Its explicit mean/noise/Cholesky reconstruction equals
  `create.gaussian` exactly on every draw.
- No fitting, swaps, response-dependent selection, interaction designs or
  full-pipeline reruns. Production functions and defaults are unchanged.

## Numerical results

| Design | max absolute ΔS / feature variance | native conditional-covariance relative difference | common-innovation S-change RMS | different-stream RMS |
|---|---:|---:|---:|---:|
| Marginal 1 | 1.85e−5 | 0.0118% | 0.00271 | 0.24583 |
| Marginal 2 | 6.47e−6 | 0.0015% | 0.03417 | 0.29571 |
| Marginal 3 | 1.72e−5 | 0.0488% | 0.00025 | 0.18573 |
| Marginal 4 | 4.03e−5 | 0.1022% | 0.00122 | 0.14932 |
| Fixed stage-two union | 1.20e−4 | 0.0105% | 0.00085 | 0.87844 |

S differences use an identical supplied Σ. Conditional-covariance differences
are Frobenius norms relative to R's conditional covariance, comparing the
native preparations. RMS values are in saved latent-factor units: the S-change
comparison holds Σ and innovations fixed; the stream comparison holds Σ and S
fixed. They are coupling diagnostics, not selection-effect sizes or statistical
tests. Marginal 2 illustrates why small covariance differences need not imply
small same-noise Cholesky-coupled sample differences near singularity.

Native and R input covariances differ by at most 5.69e−14. Native solves can
amplify even that rounding: the largest resulting same-noise sample RMS is
1.83e−4. Rust clips 5–8 S entries to zero in the marginal chunks; R has no exact
zeros. The shared-Σ Rust-S Cholesky factor adds approximately 1e−10 diagonal
jitter in those chunks, versus ≤3.34e−16 residual in the union. We have not
changed clipping, regularization or solver settings.

For fixed R Σ/S, the same numeric seed does not produce the same normals:
Rust uses `StdRng`/`StandardNormal`, row-major filling; R uses Mersenne-Twister /
Inversion and column-major filling. Empirical cross-stream correlations are
−0.00143 to −0.00172. Observed sample-difference RMS is close to
sqrt(2 × trace(conditional covariance) / p), the independent-stream expectation
(0.24535, 0.29542, 0.18531, 0.14914 and 0.87732 respectively).

Both native samplers pass a six-standard-error moment screen on their own
conditional means/covariances. Across designs, the largest mean discrepancy is
2.16 SE and second-moment discrepancy 3.68 SE. With N = draws × rows, the
screen uses SE(meanᵢ) = sqrt(Cᵢᵢ/N) and
SE(second momentᵢⱼ) = sqrt((Cᵢᵢ Cⱼⱼ + Cᵢⱼ²)/N), after subtracting the exact
row-specific conditional mean. It assumes independent Gaussian innovations;
it is a diagnostic screen, not a calibrated omnibus p-value, a general RNG
test, or proof of distributional equality. The script self-check catches
injected mean and variance errors. Repeated Rust seed-42 draws match exactly.

## R full-pipeline stream setup

`benchmarks/run_slide_r.R:66` calls `set.seed(base-seed)` in the main R process.
The frozen installed `SLIDE::SLIDE` creates `makePSOCKcluster(ncore)` and
registers it with doParallel; it does not call `clusterSetRNGStream`.
`secondKO` dispatches draws via foreach. The interaction optimization preserves
that dispatch and does not initialize worker streams.

`stream_check.R` captures those installed function bodies and mirrors their
four-worker cluster setup, twice with master seed 42:

| Setup | Workers with initialized RNG before sampling | Repeated worker draws identical? |
|---|---:|---|
| Master seed only, as inspected | 0 / 4 in both executions | No |
| Explicit `clusterSetRNGStream(cl, 42)` control | 4 / 4 | Yes |

This directly demonstrates that the master seed alone does not seed these
workers. It is not a full-pipeline replay or a measurement of how much historical
selection variability this explains. Seeding each worker alone would still not
guarantee repeatability with load-balanced task assignment; per-task streams
are the appropriate next check. R's official [parallel RNG documentation](https://search.r-project.org/R/refmans/parallel/html/RngStream.html)
describes worker-stream initialization, and its [parallel vignette, §7](https://stat.ethz.ch/R-manual/R-patched/library/parallel/doc/parallel.pdf)
explains the load-balancing caveat.

Rust's R-generated-knockoff bridge is a different path: it calls
`set.seed(base_seed + iteration)` per draw and then uses Rust's separately
seeded swap masks. The sequential saved paired-fixture generator is different
again: one initial R seed, with swaps interleaved between draws. Their seed
labels alone cannot make these streams byte-paired.

## Verification and reproduction

`cargo test --release -p slide-knockoff`: 14 passed, four ignored diagnostic
harnesses. All five explicit sampler diagnostics and Python assertions passed.
The complete audit was rerun; final numerical summaries match the preceding run
exactly. Input hashes were unchanged. No full backend job was submitted.

From the repo root, using the existing environment:

```bash
ENV=/vast/djishnu/Aaron/.tmpfiles/20260902_knockoff-generator-parity/project/.pixi/envs/dev
export TMPDIR="$PWD/.tmpfiles/2026-09-28_sampler-audit"
export R_LIBS=/vast/djishnu/Aaron/.tmpfiles/20260902_r-generated-knockoff-backend-evaluation/R-library
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 RAYON_NUM_THREADS=1
export LD_LIBRARY_PATH="$ENV/lib:${LD_LIBRARY_PATH:-}"
mkdir -p "$TMPDIR"
scripts/cargo-rust-blas test --release -p slide-knockoff
# Use the test executable reported by that build if its hash suffix differs.
"$ENV/bin/python" benchmark_results/2026-09-28_sampler-audit/audit.py \
  --fixtures .tmpfiles/2026-09-14_paired-convergence/results \
  --out "$TMPDIR/reproduction" \
  --rscript "$ENV/bin/Rscript" \
  --rust-test target/release/deps/slide_knockoff-a77db14ddbd86859
R_LIBS="$PWD/.tmpfiles/2026-09-22_all-backend-parity-tiefix/R-library:$R_LIBS" \
  "$ENV/bin/Rscript" --vanilla benchmark_results/2026-09-28_sampler-audit/stream_check.R \
  "$TMPDIR/stream_reproduction"
```

The audit refuses an existing output directory. `summary.csv`, `manifest.json`
and `worker_rng.csv` preserve the compact results. Raw arrays/logs are in
`.tmpfiles/2026-09-28_sampler-audit/run_5/`; installed R function snapshots and
worker samples are in the sibling `stream_check/`. Earlier probes are retained.
The manifest records fixture, source, test-binary and R-function hashes.

## Next bounded experiment

Before another full-backend benchmark, compare W/vote frequencies using native
versus R S with common Σ, innovations and swap masks on these saved designs.
Separately test reproducible per-task R streams; do not silently reinterpret
historical nominal seeds as controlled streams or change production defaults.
Use the [historical R aggregation](../2026-09-28_r-history/README.md) as a
stratified variability reference, not one pooled gold-standard selection.

**October 5 follow-up:** the [S-only selection replay](../2026-10-05_s-selection/README.md)
completed this comparison. Marginal chunk 4 changes from eight selected factors
with R S to seven with Rust S; both arms still select ten on the fixed union.
Small S differences are therefore not selection-neutral, but the full-pipeline
discrepancy remains unresolved.
