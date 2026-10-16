# Production-readiness pilot

Approved in conversation on 2026-10-16 after the three read-only audits.
Status: first 12 cells passed; remaining runs and analysis are queued. No final
equivalence or production-readiness verdict yet.

Preflight findings: SSc X uses V1…V24 while Y uses subject identifiers. Both
files match historical/original hashes exactly. Preserve their positional pairing
for computational comparison; biological X/Y identity is unverified. CD4 has
matching ordered IDs and actual n=2482,p=1776 (old comments counted headers).
Build job 11961811 compiled the release binary successfully, then failed its
optional `emit_sdp_stages` diagnostic because `sigma.csv` was absent. This setup
failure is retained; the build script now runs ordinary SDP tests instead of a
fixture-dependent ignored diagnostic. No model-cell retry follows this correction.

## Scope and frozen inputs

Production source: `4b321116fde6b92d5eb7bd771e0357cfb8dd72c4`.
Assessment branch: `feat/production-readiness-assessment`. Benchmark-only changes
are recorded separately from that production snapshot. No statistical defaults
change. In particular, Clarabel effective `tol_feas` remains `1e-6`.
The rejected `1e-8` candidate is not an assessment arm.

Staging: `.tmpfiles/2026-10-16_production-readiness/`.
The frozen SSc inputs, installed R SLIDE package and Python DSDP/glmnet extensions
come from `.tmpfiles/2026-10-16_backend-replicates-all-arms/`.
Historical results at `c4726e3` are background, not pooled with candidate runs.
The installed R package is SLIDE 0.1.1; its upstream commit is unknown. Runtime
versions and installed-package/input/binary hashes identify the executed reference.
The R harness rebinds worker counts and uses the existing
`SLIDE_R_OPTIMIZE=1` interaction-preparation optimization, as did the historical
comparison. Thus this is an installed-R reference with documented harness
adaptations, not an untouched upstream invocation. Those adaptations require
their existing regression checks.

## Approved cells and limits

- SSc: R/Python/Rust × delta {0.01, 0.1} × lambda {0.1, 1.0} × ten runs = 120.
- CD4: R/Python/Rust × delta 0.1/lambda 0.1 × ten runs = 30, conditional on
  matching preflight. Its ordinal response is treated as continuous, as in the
  existing CD4 workflow. Approved 500 draws/four workers differ from the older
  CD4 workflow's 1,000 draws/ten workers.
- Simulations: null/sparse response × ordinary/near-degenerate factors × twenty
  independent generated datasets × three implementations = 240 runs;
  n=100, p=500, planted K=10, delta=0.1, lambda=0.1.
- Scaling: n=50, planted K=20, p={2000,4000,8000,12000}, plus n=200,p=2000;
  three implementations, one run each. Report estimated K separately.
- Replay: saved native Z20/Z74 chunks, 500 draws each, common inputs and draws;
  fitter-only controls and S-construction interventions are distinct.

All full-pipeline cells use homogeneous LOVE, Gaussian parity settings,
500 knockoff draws, specificity 0.1, SLIDE/findOptIter selection, four CPU workers
and one BLAS thread. Final CV is excluded. Binary response, heterogeneous LOVE,
shipped native-default configurations and additional datasets remain untested.
Same integer seeds do not synchronize random draws across languages; R worker
RNG is not fully controlled by seed labels. Simulation datasets, not draws,
are the independent units for truth-based uncertainty.

Regular cells: 8 GiB and 300 seconds each. Scaling: 32 GiB and 600 seconds;
stop a backend's larger-p ladder after timeout/OOM. No automatic retries.
Maximum requested compute: approximately 142 CPU-hours including a two-CPU-hour
build/replay allowance. Queue time and whole-node charging are not included.
Use sequential benchmark processes on one allocated Turin node; record shared
node interference instead of claiming exclusive-node measurements.

## Simulation model (declared before execution)

Three pure indicators per factor; remaining loading rows have two distinct
factor entries with absolute weights 0.6 and 0.4 and random signs.
Latent variances are one. The near-degenerate regime correlates factors 0/1
at 0.999; other factors are independent. Feature noise has standard deviation
0.3. Sparse response: 2 Z0 + 1.5 Z1 + 2 Z0 Z2 + unit Gaussian noise.
Null response is independent unit Gaussian noise. Save loadings, latent scores,
feature noise and response noise with each input. The closest existing generator
is the fixed p=6/K=2 fixture in `tests/tests/test_love.py`; it has no response or
truth-export workflow. `benchmarks/generate_factor_inputs.py` supplies the
benchmark workflow without changing that fixture.

## Approved provisional margins and inference

- Factor selection-probability differences: ±0.05.
- Mean selected-factor count differences: ±1.
- Common-input mean Jaccard: at least 0.95, with factor-level checks.
- Independent-run overlap: cross-backend overlap no more than 0.05 below R–R.
- Power differences: ±0.05. Assess nominal FDR target separately; 0.02 excess
  is a practical allowance, not proof of nominal FDR control.

Use outcome-blind absolute-score-correlation matching (threshold 0.95), sign
alignment and explicit unmatched factors. For nonidentifiable factors, report
subspace comparisons rather than individual-factor claims. The existing strict
Z-identity check suffices only where it passes. Never compare raw factor IDs
across unmatched factors. Report both-empty agreement separately.

Simultaneous exact binomial-derived factor intervals use the full factor universe,
not only observed selections. Whole-run bootstrap count/overlap intervals are
exploratory; degenerate bootstrap intervals do not establish equivalence.
Ten runs and twenty simulation datasets can detect large problems but cannot
certify the approved narrow margins. Larger confirmation requires new approval.

## Measurement and ownership

Worker A extends `analyze_seed_replicates.py`; worker B owns measurement changes
in `benchmark_backends.py`; worker C audits replay/scaling feasibility.
Coordinator owns configuration, submission, truth evaluation, verification and
final classification. The generator worker owns only generator/tests.

Measured mode starts a fresh interpreter for every backend/cell and includes
startup, imports, input I/O and preprocessing. Build/setup are separate. The first
measured launch is distinct from later fresh-process launches. Filesystem/library
caches are not flushed: this is not a certified cold-cache measurement, and later
runs are not startup-free.
Process-group RSS is sampled every 50 ms: shared pages can be double-counted,
short peaks can be missed, and it is not an exact cgroup peak. Keep scheduler
accounting and every failed/timed-out cell alongside successful timings.

## Commands

Build (submitted job 11961811; script syntax checked):

```bash
RUN="$PWD/.tmpfiles/2026-10-16_production-readiness"
sbatch --parsable --output="$RUN/logs/build-%j.out" \
  benchmark_results/2026-10-16_production-readiness/build.sbatch "$RUN"
```

Raw per-cell outputs remain in staging, with
commands, configurations, summaries, failure rows and inspected plots retained
in this result directory.

## Execution checkpoint

| Job | Work | State at checkpoint |
| --- | --- | --- |
| 11961811 | Release build + copied optional diagnostic | Binary built; diagnostic lacked sigma.csv |
| 11962055 | Ordinary SDP tests, R worker/optimization tests, environment/input preflight | Completed, 61 seconds, 4 CPUs |
| 11962071 | First SSc seed, three implementations × four cells | Completed, 391 seconds, 4 CPUs; 12/12 success |
| 11962074 | Controlled native/R-S replay and fresh R fitter check | Running |
| 11962087 | Remaining 378 regular cells | Queued after first/replay |
| 11962088 | Up to 15 scaling cells | Queued after regular cells |
| 11962108 | Statistical/cost analysis | Queued after scaling |

Benchmark jobs use sequential processes on htc-n73 (AMD EPYC 9575F), four
allocated CPUs; other users can share the node. The release build ran on htc-n77.
R runtime is 4.5.3, Python 3.11.15, NumPy 2.4.3, SciPy 1.17.1 and pandas 3.0.2.
See session files for BLAS and dependency details. First-seed strict Z identity
checks pass in all four cells; maximum absolute difference is 4.69e-11.
Selection sets differ, and one run is not evidence of equivalence.
The three `first/*.png` figures were rendered and inspected: labels are legible,
backend colors distinguish singleton observations, and values match the CSVs.
These are one observation per backend/cell, not replicated speedup estimates;
no bootstrap intervals are estimated for these singleton arms.

The measurement/generator/analysis/runner tests pass (45 focused Python tests at
this checkpoint), as do ordinary SDP and existing R worker/optimization tests.
Full project suites were not run. Updated benchmark tools are committed; algorithm
sources remain the frozen production revision. The scaling ladder conservatively
stops an arm after any failure, not only diagnosed OOM/timeout.

Rerun preparation and execution use the recorded environment plus existing tools:

```bash
export PROJECT="$PWD"
export ASSESSMENT="$PROJECT/benchmark_results/2026-10-16_production-readiness"
export RUN="$PROJECT/.tmpfiles/2026-10-16_production-readiness"
export ENV=/vast/djishnu/Aaron/.tmpfiles/20260902_knockoff-generator-parity/project/.pixi/envs/dev
export R_DEPS=/vast/djishnu/Aaron/.tmpfiles/20260902_r-generated-knockoff-backend-evaluation/R-library
# For a new run, choose new dated RUN and output directories; do not overwrite this one.
# Freeze production with git archive 4b32111, copy the recorded R package/extensions
# and input CSVs, and overlay the committed benchmark tools before preparation.
bash "$ASSESSMENT/prepare-command.sh"
sbatch --output="$RUN/logs/preflight-%j.out" "$ASSESSMENT/preflight.sbatch"
# Within the recorded four-CPU allocation, one manifest at a time:
source "$ASSESSMENT/environment.sh"
bash scripts/with-rust-blas-env.sh bash benchmarks/run_backend_array.sh --manifest "$RUN/first.json"
bash scripts/with-rust-blas-env.sh bash benchmarks/run_backend_array.sh --manifest "$RUN/remaining.json"
# Scaling requires its separate 32-GiB allocation:
bash scripts/with-rust-blas-env.sh bash benchmarks/run_backend_array.sh --manifest "$RUN/scaling.json"
# Analysis uses frozen copies of the committed analysis tools:
bash "$ASSESSMENT/analysis-command.sh"
```

Submitted batch scripts and dependencies are recoverable with
`scontrol write batch_script JOB_ID -` while scheduler records remain available.
The first/remaining/scaling manifest caps were respectively 1h/31.5h/2.5h;
replay 30 minutes at one CPU; analysis 15 minutes at one CPU.
The nominal knockoff q=0.1 is not a demonstrated FDR guarantee for final
findOptIter-selected sets. Simulation outputs evaluate final selections rather
than assuming that per-draw guarantees survive aggregation.
