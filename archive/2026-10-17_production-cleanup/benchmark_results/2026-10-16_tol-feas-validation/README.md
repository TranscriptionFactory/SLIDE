# Single feasibility-tolerance experiment

Started 2026-10-16 from `ff22081`. Production defaults remain unchanged.

## Decision rules recorded before candidate results

Compare only effective Clarabel `tol_feas`: baseline 1e-6 versus candidate
1e-8. Copy the complete baseline settings and change that field alone;
retain the exact binary64 baseline `reduced_tol_feas` (1e-6 * 0.1).
Do not change objective-gap tolerances, fallback acceptance, postprocessing,
covariance, innovations, swaps, fitting, or aggregation.

Required controls: native Z20 74/500 and Z74 38/500. Use the exact saved
`rust_sigma.raw` in `.tmpfiles/2026-10-16_native-chunk-r-s/chunk_{1,4}`;
never estimate experiment covariance from `z.raw`. Compare binary64 input
and normalized matrices across arms, and complete effective settings.

Candidate gates: no new failures or downgrade from Solved to AlmostSolved;
finite, bounded final S satisfying existing feasibility checks; relevant
regression tests pass. Numerical benefit requires improved solver residuals
or raw constraint feasibility, with objective/gap and postprocessing repair
reported alongside it. Smaller S, different votes, and R agreement alone
are not benefit. If the candidate produces identical solver trajectories
and outputs, stop for no benefit. Do not tune another setting.

The user requested runtime measurement before choosing a slowdown limit.
No numerical runtime threshold is approved. Report timings and variability;
if numerical benefit presents a runtime tradeoff, ask before declaring that
gate passed. The user agreed that selection changes are diagnostic, not an
automatic rejection when numerical checks and existing regressions pass.

Inputs: two diagnostic chunks, `tests/fixtures/sdp/{corr3,cov4}/input_sigma.csv`,
and `.tmpfiles/2026-10-12_python-scale-bug/rSig_4.txt` (float.fromhex).
Reuse `emit_sdp_stages`, the sampler audit, and
`tmpfiles/2026-10-16_native-chunk-jitter/replay_jitter.py`.

## Conditional full benchmark

Only after bounded gates pass: reuse `benchmarks/benchmark_backends.py` and
existing scheduler workflow, frozen SSc UnTx X/Y from
`.tmpfiles/2026-10-16_backend-replicates-all-arms/inputs`, parity settings,
delta 0.01/0.1, lambda 0.1/1.0, 500 iterations, specificity 0.1, SLIDE selection,
seeds 42 and 1000 through 9000. Four CPUs, one BLAS thread, HTC/Turin,
4 GB and 20 minutes per replicate. Ten Rust replicates / 40 cells.
Verify matching baseline outputs from the completed four-arm report and its
source snapshot before reuse. Do not rerun the excluded Python/R hybrid.

## Status

Bounded validation completed. The user accepted its measured runtime cost
and authorized continuation to the established full benchmark. Production
remains at tol_feas=1e-6. The completed full benchmark rejected the candidate
after ten new NumericalError failures; see the linked full report below.

## Bounded results

All three scheduler jobs completed with exit 0 on `htc-n70`:
control `11958560` (58 s), paired solver stages `11958570` (8 s), and
sampler plus paired fits `11958575` (94 s). No failed or incomplete cells.
The solver job ran five alternating-order baseline/candidate repetitions
per input (50 solves total), one BLAS/Rayon thread. Each arm reproduced its
residuals, objectives, iterations, raw S, and final S across repetitions.

All 50 solves returned **Solved**, not AlmostSolved. Complete settings dumps
differ only at `tol_feas`; `reduced_tol_feas` remains the baseline binary64
value of 1e-6 * 0.1. Input Sigma, normalized G, objective, RHS and constraints
match across each pair. The diagnostic also checks these settings in Rust.

| Input | Iterations baseline → candidate | Primal residual baseline → candidate | Dual residual baseline → candidate | Absolute objective gap baseline → candidate |
| --- | ---: | ---: | ---: | ---: |
| chunk 1 | 22 → 22 | 5.993e-9 → same | 1.571e-10 → same | 5.309e-7 → same |
| chunk 4 | 13 → 13 | 1.995e-9 → same | 7.675e-11 → same | 4.210e-8 → same |
| corr3 | 6 → 7 | 7.936e-8 → 4.170e-9 | 6.141e-8 → 3.227e-9 | 1.333e-6 → 7.007e-8 |
| cov4 | 6 → 7 | 3.859e-8 → 5.931e-10 | 2.940e-8 → 4.519e-10 | 9.798e-7 → 1.506e-8 |
| rSig4 | 20 → 20 | 4.027e-9 → same | 4.220e-11 → same | 2.430e-7 → same |

Both diagnostic chunks and rSig4 have **bitwise-identical raw and final S**
across arms. Their baseline residuals already satisfy 1e-8, so tighter
feasibility does not require another iteration. Negative-coordinate / final
zero counts stay 8 / 8, 3 / 3, and 9 / 9 respectively. No S floors were used.

The two small fixtures show a numerical benefit beyond smaller S:

| Input | Minimum eigenvalue of raw 2G−diag(s), baseline → candidate | Maximum correlation-repair change | Final normalized sum(S/diag(Sigma)), baseline → candidate |
| --- | ---: | ---: | ---: |
| corr3 | −1.807e-7 → −8.882e-9 | 4.660e-7 → 2.315e-8 | 1.75999953636 → 1.75999997379 |
| cov4 | −2.406e-8 → −5.244e-10 | 2.531e-8 → 6.563e-10 | 3.89300953339 → 3.89300958402 |

Their residuals and gaps improve about 19× and 65×, raw PSD violations and
repair magnitudes decrease, and final normalized objectives increase slightly.
Neither fixture gains negative coordinates or zeros. Extra covariance repair
is inactive on all ten input/arm combinations. Final shifted-Sigma minimum
eigenvalues are positive on all five inputs in both arms; the smallest is
2.110e-10. Candidate corr3/cov4 also satisfy the existing 1e-3 fixture
optimality comparison. Objective-gap settings remain 1e-6 absolute/relative;
an absolute gap above 1e-6 alone does not imply solver failure when its
relative criterion passes.

### Runtime

Clarabel-reported solve milliseconds, median [min, max], five repetitions.
These sub-millisecond fixture measurements do not establish end-to-end cost.

| Input | Baseline | Candidate |
| --- | ---: | ---: |
| chunk 1 | 42.610 [41.155, 44.892] | 42.148 [40.360, 42.998] |
| chunk 4 | 29.476 [28.446, 31.790] | 29.838 [28.012, 30.945] |
| corr3 | 0.140 [0.116, 0.158] | 0.136 [0.118, 0.176] |
| cov4 | 0.167 [0.161, 0.216] | 0.182 [0.171, 0.194] |
| rSig4 | 41.235 [39.665, 43.165] | 40.763 [39.320, 43.481] |

All min/max ranges overlap. cov4 requires one extra iteration and its median
increases about 0.015 ms (9%). No runtime acceptance threshold has been invented.
The user accepted this measured cost and confirmed proceeding to the
conditional full benchmark.

### Sampler and selection controls

The audit uses `SLIDE_S_AUDIT_FIXED_SIGMA=1` to bypass covariance estimation.
It feeds staged S through the existing shared-S cache. Saved covariance,
innovations, designs and swaps match exactly; the fitter and aggregation are
unchanged. For both chunks, fresh baseline and candidate S, generated draws,
all 500 W vectors, all per-draw selections, vote counts and preliminary
selected sets match the saved native control **bitwise/exactly**.

| Chunk | Native / baseline / candidate focus votes | Preliminary findOptIter set, all three arms |
| --- | ---: | --- |
| 1, Z20 | 74 / 74 / 74 of 500 | Z16, Z18, Z19, Z20 |
| 4, Z74 | 38 / 38 / 38 of 500 | Z70, Z73, Z75, Z78, Z79, Z81, Z83, Z87 |

Rust's initial conditional-covariance Cholesky fails for both chunks and
rSig4 in both arms; each actual factor requires the unchanged 1e-10 diagonal
jitter. corr3 and cov4 factor directly without jitter in both arms. The
maximum off-diagonal factor reconstruction residual is 1.388e-17. The
2.22e-16 measured diagonal offset on baseline cov4 is roundoff, not added
jitter; its direct Rust Cholesky succeeds. `sampler_measurements.csv` keeps
both measured offsets and actual added jitter.

For heldout factorization checks only, the audit uses the first p columns of
saved chunk-1 Z to satisfy its design interface. The supplied Sigma and S
alone determine the conditional covariance and factor; this design contributes
only unused means. Two draws exercise the existing audit, without fitting or
claiming heldout selection results. Diagnostic chunks retain all 500 original
draw seeds (42–541 and 1542–2041).

### Interpretation and current recommendation

**Retain the production baseline.** The candidate passes bounded
numerical and downstream checks and improves the two small fixtures, but does
not improve the two diagnostic chunks or near-singular heldout. This supports
the full-benchmark evaluation, but did not justify changing the production
default. The subsequent full run failed ten cells that the baseline completes.

The user accepted the bounded runtime cost after reviewing these results.
The [full benchmark](../2026-10-16_tol-feas-benchmark/README.md) completed
and rejected the candidate for ten new NumericalError failures. No R-agreement
criterion or statistical-equivalence claim was used. Existing benchmark
statistics measure selection variability and agreement, not ground-truth
power or FDR control.

## Verification and retained outputs

- Release slide-sdp: 6 ordinary unit tests plus 4 integration tests pass.
- Release slide-knockoff: 14 ordinary tests pass.
- Python paired attribution/preparation: 14 tests pass.
- Diagnostic settings regression, exact input/settings comparisons, final
  feasibility, saved control reproduction, and paired draw/score comparisons
  pass. Production settings and accepted solver statuses are unchanged.
- `solver_measurements.csv`: all 50 solves, residuals/objectives, both timers,
  negativity, clipping, repairs, feasibility and paired bitwise comparisons.
- `stages/<input>/<arm>/repeat_1/`: full effective settings, normalized input,
  constraints, raw S and each postprocessing stage. Repeats 2–5 remain under
  `.tmpfiles/2026-10-16_tol-feas-validation/archive/stage-repeats/`.
- `sampler_measurements.csv` and `sampler/`: actual Rust Cholesky outcomes,
  targets/factors, supplied S and covariance.
- `control/` and `paired/`: complete W arrays, per-draw selections, paired
  votes, aggregation and geometry. These contain repeated saved controls,
  not new estimates of statistical performance.

## Reproduction commands

Choose fresh dated scratch/result paths to avoid replacing this run. All
paths below are relative to the repository root unless absolute. Preserve
original saved inputs: the replay writes fit scratch alongside its inputs.

```bash
ENV=/vast/djishnu/Aaron/.tmpfiles/20260902_knockoff-generator-parity/project/.pixi/envs/dev
RUN="$PWD/.tmpfiles/2026-10-16_tol-feas-validation"
OUT="$PWD/benchmark_results/2026-10-16_tol-feas-validation"
export TMPDIR="$RUN/scratch" TMP="$RUN/scratch" TEMP="$RUN/scratch"
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 RAYON_NUM_THREADS=1
export PYTHONDONTWRITEBYTECODE=1 SLIDE_KNOCKOFF_MATCH_R_COV=1 SLIDE_KNOCKOFF_MATCH_R_LAMBDA=1
unset SLIDE_S_TOL
scripts/with-rust-blas-env.sh cargo test --locked --release -p slide-sdp
scripts/with-rust-blas-env.sh cargo test --locked --release -p slide-knockoff
```

Build output identifies test executables. This run used
`target/release/deps/slide_sdp-a1482001354cc402` and
`target/release/deps/slide_knockoff-3b28efba8069aae9`.
The unchanged fitter is
`.tmpfiles/2026-10-16_gaussian-arithmetic/final-paired-xtilde`.
Scheduler requests: `sbatch --cluster=htc --constraint=turin --mem=4G
--cpus-per-task=1 --time=00:20:00` (stage job: 10 minutes).

Input preparation uses the prior heldout loader conventions:

```python
# Execute with $ENV/bin/python; define run = Path(RUN).
from pathlib import Path
import numpy as np
for name in ('chunk_1', 'chunk_4', 'corr3', 'cov4', 'rSig4'):
    if name.startswith('chunk_'):
        sigma = np.fromfile(Path('.tmpfiles/2026-10-16_native-chunk-r-s') /
                            name / 'rust_sigma.raw', '<f8').reshape(22, 22)
    elif name == 'rSig4':
        text = Path('.tmpfiles/2026-10-12_python-scale-bug/rSig_4.txt').read_text()
        sigma = np.array([[float.fromhex(v) for v in line.split()]
                          for line in text.splitlines() if line.strip()])
    else:
        sigma = np.loadtxt(Path('tests/fixtures/sdp') / name / 'input_sigma.csv', delimiter=',')
    d = run / 'inputs' / name
    d.mkdir(parents=True)
    sigma.astype('<f8').tofile(d / 'sigma.raw')
    np.savetxt(d / 'sigma.csv', sigma, delimiter=',', fmt='%.17e')
    assert np.array_equal(np.loadtxt(d / 'sigma.csv', delimiter=',').view('u8'), sigma.view('u8'))
```

For each input and repetition, create its stage directory, copy `sigma.csv`,
and invoke the same binary on both arms. Alternate arm order on repetitions
2 and 4; use five repetitions total. Do not set `correlation.csv` or `raw_s.csv`.

```bash
SDP_DIAG_DIR="$OUT/stages/$fixture/$arm/repeat_$repeat" \
SDP_DIAG_TOL_FEAS="$tol" scripts/with-rust-blas-env.sh \
  target/release/deps/slide_sdp-a1482001354cc402 \
  --exact tests::emit_sdp_stages --ignored --nocapture
# tol=1e-6 for baseline; tol=1e-8 for candidate.
```

Prepare `$RUN/sampler/$arm/$fixture`: copy exact `sigma.csv` and stage
`covariance_repaired.csv` as `r_s.csv`; copy original diagnostic `z.raw` and
`swaps.raw`. Heldout factorization design convention is described above.
For each of the ten sampler input/arm combinations:

```bash
SLIDE_S_AUDIT_FIXED_SIGMA=1 SLIDE_S_SIGMA_CSV="$dir/sigma.csv" \
SLIDE_S_OUT="$dir/native_s.csv" SLIDE_S_AUDIT_DIR="$dir" \
SLIDE_S_AUDIT_DRAWS="$draws" SLIDE_S_AUDIT_BASE_SEED="$seed" \
  scripts/with-rust-blas-env.sh target/release/deps/slide_knockoff-3b28efba8069aae9 \
  --exact tests::emit_diag_s --ignored --nocapture
# Diagnostic chunks: draws=500; seeds 42 and 1542. Heldouts: draws=2, seed=42.
```

Copy original chunk directories to `$RUN/baseline_inputs` (no symlinks or
hardlinks to writable source files). First run the existing replay without
`--s-arms` into `control/`; then run the paired comparison:

```bash
scripts/with-rust-blas-env.sh "$ENV/bin/python" \
  tmpfiles/2026-10-16_native-chunk-jitter/replay_jitter.py \
  "$RUN/baseline_inputs" \
  "$PWD/.tmpfiles/2026-10-16_gaussian-arithmetic/final-paired-xtilde" \
  "$OUT/paired" --s-arms "$RUN/sampler"
```

Summary fields use NumPy `eigvalsh(2*Sigma - diag(S))`,
`eigvalsh(2*G - diag(raw_s))`, `sum(S/diag(Sigma))`, and maximum absolute
changes between saved clipping/repair stages. Actual factor residual is
`factor.T @ factor - rust_shared_s_target`; its median diagonal estimates
added jitter, with direct Rust Cholesky success distinguishing roundoff from
regularization. Timing rows retain individual repeats, not pooled averages.
