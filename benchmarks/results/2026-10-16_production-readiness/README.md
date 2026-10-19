# Production-readiness pilot

The 2026-10-17 results contain all 120 SSc and 30 CD4 observations and the
bounded scaling runs. Cancellation left the simulation study incomplete, so
the results do not establish production readiness or backend equivalence.
See [coverage and jobs](#coverage-and-jobs),
[replicate summaries](#completed-replicate-summaries), and
[factor identities and frequencies](#backend-selection-details).

## Input alignment

SSc X uses V1…V24 while Y uses subject identifiers. Both
files match historical/original hashes exactly. Preserve their positional pairing
for computational comparison; biological X/Y identity is unverified. CD4 has
matching ordered IDs and n=2482,p=1776; earlier comments included headers
in those counts.

## Scope and frozen inputs

Production source: `4b321116fde6b92d5eb7bd771e0357cfb8dd72c4`.
See the [original-to-rewritten revision mapping](../../README.md#source-revisions-after-history-cleanup).
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
adaptations, not an untouched upstream invocation. The regression checks for
these adaptations passed (see [coverage and jobs](#coverage-and-jobs)).

## Planned cells and limits

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
feature noise and response noise with each input using
[`generate_factor_inputs.py`](../../scripts/generate_factor_inputs.py).

## Provisional margins and inference

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

## Measurement

Measured mode starts a fresh interpreter for every backend/cell and includes
startup, imports, input I/O and preprocessing. Build/setup are separate. The first
measured launch is distinct from later fresh-process launches. Filesystem/library
caches are not flushed: this is not a certified cold-cache measurement, and later
runs are not startup-free.
All pilot cells enable `--profile`. Timings include profiling overhead, which
has not been measured separately and can differ by backend. Python writes
per-stage JSON records (`src/loveslide/knockoff/filter.py::_profile_stage` in
the frozen source, now `src/slide_py/knockoff/filter.py::_profile_stage`);
these instrumented costs do not establish uninstrumented production speedups.
An unprofiled timing confirmation would require a separately approved run.
Process-group RSS is sampled every 50 ms: shared pages can be double-counted,
short peaks can be missed, and it is not an exact cgroup peak. Keep scheduler
accounting and every failed/timed-out cell alongside successful timings.

## Commands

Build (submitted job 11961811; script syntax checked):

```bash
RUN="$PWD/.tmpfiles/2026-10-16_production-readiness"
sbatch --parsable --output="$RUN/logs/build-%j.out" \
  benchmarks/results/2026-10-16_production-readiness/build.sbatch "$RUN"
```

Raw per-cell outputs remain in staging, with
commands, configurations, summaries, failure rows and inspected plots retained
in this result directory.

## Coverage and jobs

Build job 11961811 compiled the release binary successfully, then failed its
optional `emit_sdp_stages` diagnostic because `sigma.csv` was absent. This setup
failure is retained; the build script now runs ordinary SDP tests instead of a
fixture-dependent ignored diagnostic. No model-cell retry follows this correction.

Job 11962087 was cancelled with `scancel 11962087` after about 7 hours
14 minutes to prioritize scaling above 5,000 input features. The saved results
include partial outputs; unrun cells remain excluded from successful counts.
The `afterany` dependency allowed analysis of the partial run.
Scaling job 11962088 had already completed.
Its 600-second cell cap was a pilot compute guardrail, not a capacity limit;
R's timeout at p=4,000 caused p=8,000 and p=12,000 to be skipped under the approved
stop rule. Those larger R cases therefore remain untested, not demonstrated
failures. The [separate capacity campaign](../2026-10-16_large-feature-scaling/README.md)
uses a larger cap and different feature counts; do not pool its results with this pilot.

| Job | Work | State at checkpoint |
| --- | --- | --- |
| 11961811 | Release build + copied optional diagnostic | Binary built; diagnostic lacked sigma.csv |
| 11962055 | Ordinary SDP tests, R worker/optimization tests, environment/input preflight | Completed, 61 seconds, 4 CPUs |
| 11962071 | First SSc seed, three implementations × four cells | Completed, 391 seconds, 4 CPUs; 12/12 success |
| 11962074 | Controlled native/R-S replay and fresh R fitter check | Completed, 212 seconds, 1 CPU |
| 11962087 | Remaining 378 regular cells | User cancelled; SSc/CD4 summaries available, simulations partial |
| 11962088 | Up to 15 scaling cells | Completed; saved cost summary includes failures and skipped cells |
| 11962108 | Statistical/cost analysis | Analysis logs and CSV summaries available; scheduler state not rechecked |

Regular benchmark jobs use sequential processes on htc-n73 (AMD EPYC 9575F), four
allocated CPUs; other users can share the node. The release build ran on htc-n77.
Scaling was allowed to run alongside simulations, excluding htc-n73 to avoid
competition between these jobs on one node. The `turin`
hardware constraint, four CPUs, 32 GiB and 2.5-hour limit remain unchanged;
verify the allocated node's exact CPU model before pooling timings. Applied:
`scontrol update JobId=11962088 Dependency= ReqNodeList= ExcNodeList=htc-n73`
and `scontrol update JobId=11962108 Dependency=afterany:11962087:11962088`.
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
Full project suites were not run. Algorithm sources use the frozen production
revision. The scaling ladder stops an arm after any failure, not only diagnosed OOM/timeout.

Rerun preparation and execution use the recorded environment plus existing tools:

```bash
export PROJECT="$PWD"
export ASSESSMENT="$PROJECT/benchmarks/results/2026-10-16_production-readiness"
export RUN="$PROJECT/.tmpfiles/2026-10-16_production-readiness"
export ENV=/vast/djishnu/Aaron/.tmpfiles/20260902_knockoff-generator-parity/project/.pixi/envs/dev
export R_DEPS=/vast/djishnu/Aaron/.tmpfiles/20260902_r-generated-knockoff-backend-evaluation/R-library
# For a new run, choose new dated RUN and output directories; do not overwrite this one.
# Freeze production with git archive 4b32111, copy the recorded R package/extensions
# and input CSVs, and overlay the committed benchmark tools before preparation.
bash "$ASSESSMENT/prepare-command.sh"
sbatch --output="$RUN/logs/preflight-%j.out" "$ASSESSMENT/preflight.sbatch"
# Within the recorded four-CPU allocation, one manifest at a time:
# environment.sh enters the frozen source tree; its runner paths retain that snapshot's layout.
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

## Completed controlled replay

Both native samplers at the frozen production revision reproduce the saved
native draws bitwise.
Native fitter selections match the saved R controls. On the freshly generated
R-S draws, fresh R and candidate Rust fitters agree on all 1,000 selected sets
(500 per chunk). This is a fitter-only result conditional on supplied draws,
not agreement of the native samplers or final full-pipeline selection.

| Fixed native chunk | Native S votes | R S votes | R-S minus native, percentage points ± paired MC SE |
| --- | ---: | ---: | ---: |
| Z20 | 74/500 | 40/500 | −6.8 ± 1.29 |
| Z74 | 38/500 | 55/500 | +3.4 ± 0.95 |

Both cross the preliminary 50-vote threshold in the R-like direction. The
intervention replaces all S entries and their downstream conditional mean/factor;
it does not isolate one coordinate or prove a solver replacement fixes SLIDE.
The fresh Z20 R-S count is 40, not the historical 43; historical and fresh
numerical outputs are not pooled. All settings remain unchanged.

`replay/vote_deltas.png` was rendered and inspected for labels, clipping and
agreement with `paired_votes.csv`. Its error bars quantify Monte Carlo variation
on two fixed designs, not uncertainty across biological datasets.

Regenerate the figure with the existing plotting helper, from the project root:

```python
from pathlib import Path
import sys
import pandas as pd
sys.path.insert(0, "benchmarks/scripts")
from replay_paired_xtilde import plot_sampler_votes
out = Path("benchmarks/results/2026-10-16_production-readiness/replay")
votes = pd.read_csv(out / "paired_votes.csv")
votes = votes[votes.contrast == "r_s-native"].rename(columns={"factor": "feature"})
votes["design"] = votes.chunk.map({1: "Native chunk 1 (Z20)", 4: "Native chunk 4 (Z74)"})
plot_sampler_votes(out, votes, title="R-S minus native Rust-S votes on candidate fitter",
                   note="Shared covariance, innovations and swaps; fixed designs only, not full-pipeline equivalence or FDR.")
```

## Completed replicate summaries

`cd4-analysis.log` reports n=2,482, p=1,776, K=39 for all three implementations,
with ten runs per backend. Median fresh-process wall seconds [Q1–Q3] are
Rust 5.31 [5.03–5.58], Python 22.07 [21.85–22.83], R 93.52 [91.32–98.60].
All score matrices agree within 9.1×10⁻¹²; Rust/Python select marginal Z7 ten
times each, while R adds a second marginal in one run. These are profiled
measurements, not uninstrumented speedups or predictive validation.

`ssc-analysis.log` reports score-matrix agreement within 4.7×10⁻¹¹ across all
four cells. Nevertheless, `ssc/factor_rates.csv` records substantial selection
differences at delta=0.01/lambda=0.1 (K=172): Rust/Python/R select Z20 in 10/0/2
of ten runs and Z74 in 0/8/8. Biological sample identity remains unverified as
noted above. Agreement in LOVE output does not establish knockoff-selection
agreement. The external `costs/summary.csv` (not bundled here) records
attempted, successful and unrun counts;
completed analysis files do not imply that cancelled simulations completed.

## Backend selection details

### Recovered analysis provenance

The analysis artifacts under `ssc/` and `cd4/`, together with
`ssc-analysis.log` and `cd4-analysis.log`, were recovered on 2026-10-17 from the
sibling `../omni-SLIDE/benchmarks/results/2026-10-16_production-readiness/`.
They are byte-identical copies of outputs from the original frozen
`4b321116fde6b92d5eb7bd771e0357cfb8dd72c4` run, not a rerun. The 60 raw
`benchmark_summary.json` files in `results/` were restored from that run's
sibling `.tmpfiles/2026-10-16_production-readiness/results/`. These are copies,
not regenerated artifacts. `analysis-source-revision.txt` records the analysis
code revision; `analysis-command.sh` records the command used to create the
saved plots and CSVs. The raw patient inputs and large Z matrices were not
copied. Raw JSON alone cannot reproduce the saved Z checks; those checks are
reported in the retained analysis logs and `z_check.csv` files.

### Marginal selection counts

Counts are runs with each factor selected, out of ten per backend and cell.
These list every factor selected at least once in that cell, not every factor
estimated. Full rates, uncertainty intervals, and inferential columns are in
[`ssc/factor_rates.csv`](ssc/factor_rates.csv) and
[`cd4/factor_rates.csv`](cd4/factor_rates.csv); the existing
[`ssc/factor_rates.png`](ssc/factor_rates.png) and
[`cd4/factor_rates.png`](cd4/factor_rates.png) show factors with rate at least
0.2, so they omit rarer selections. Existing saved uncertainty files include
[`ssc/equivalence_factors.csv`](ssc/equivalence_factors.csv),
[`ssc/equivalence_summary.csv`](ssc/equivalence_summary.csv),
[`cd4/equivalence_factors.csv`](cd4/equivalence_factors.csv), and
[`cd4/equivalence_summary.csv`](cd4/equivalence_summary.csv).

| SSc cell | Factor | Rust / 10 | Python / 10 | R / 10 |
|---|---:|---:|---:|---:|
| d=0.01 l=0.1 | Z18 | 10 | 10 | 10 |
| d=0.01 l=0.1 | Z20 | 10 | 0 | 2 |
| d=0.01 l=0.1 | Z27 | 10 | 10 | 10 |
| d=0.01 l=0.1 | Z45 | 10 | 10 | 10 |
| d=0.01 l=0.1 | Z73 | 2 | 0 | 0 |
| d=0.01 l=0.1 | Z74 | 0 | 8 | 8 |
| d=0.01 l=0.1 | Z136 | 10 | 10 | 10 |
| d=0.01 l=1.0 | Z18 | 9 | 10 | 10 |
| d=0.01 l=1.0 | Z27 | 10 | 10 | 10 |
| d=0.01 l=1.0 | Z45 | 9 | 10 | 10 |
| d=0.01 l=1.0 | Z57 | 2 | 2 | 4 |
| d=0.01 l=1.0 | Z73 | 4 | 5 | 8 |
| d=0.01 l=1.0 | Z74 | 0 | 0 | 1 |
| d=0.01 l=1.0 | Z136 | 9 | 10 | 10 |
| d=0.1 l=0.1 | Z5 | 4 | 4 | 6 |
| d=0.1 l=0.1 | Z10 | 3 | 1 | 2 |
| d=0.1 l=0.1 | Z25 | 10 | 4 | 10 |
| d=0.1 l=0.1 | Z32 | 1 | 0 | 2 |
| d=0.1 l=0.1 | Z33 | 6 | 3 | 6 |
| d=0.1 l=0.1 | Z38 | 4 | 3 | 4 |
| d=0.1 l=0.1 | Z44 | 10 | 7 | 10 |
| d=0.1 l=0.1 | Z53 | 8 | 3 | 8 |
| d=0.1 l=0.1 | Z55 | 0 | 1 | 0 |
| d=0.1 l=0.1 | Z61 | 4 | 8 | 2 |
| d=0.1 l=0.1 | Z62 | 3 | 2 | 1 |
| d=0.1 l=0.1 | Z66 | 1 | 1 | 1 |
| d=0.1 l=0.1 | Z78 | 2 | 1 | 2 |
| d=0.1 l=0.1 | Z84 | 1 | 0 | 0 |
| d=0.1 l=1.0 | Z0 | 0 | 1 | 0 |
| d=0.1 l=1.0 | Z25 | 5 | 3 | 4 |
| d=0.1 l=1.0 | Z33 | 9 | 5 | 7 |
| d=0.1 l=1.0 | Z38 | 2 | 1 | 0 |
| d=0.1 l=1.0 | Z44 | 10 | 10 | 10 |
| d=0.1 l=1.0 | Z55 | 0 | 1 | 0 |
| d=0.1 l=1.0 | Z66 | 2 | 1 | 0 |
| d=0.1 l=1.0 | Z69 | 8 | 7 | 5 |
| d=0.1 l=1.0 | Z78 | 0 | 1 | 0 |
| d=0.1 l=1.0 | Z86 | 0 | 1 | 0 |

| CD4 cell | Factor | Rust / 10 | Python / 10 | R / 10 |
|---|---:|---:|---:|---:|
| d=0.1 l=0.1 | Z7 | 10 | 10 | 10 |
| d=0.1 l=0.1 | Z26 | 0 | 0 | 1 |

For SSc, only Z20 and Z74 in the d=0.01/l=0.1 cell have Benjamini–Hochberg
q < 0.05 for Rust versus R in the saved factor table; both q values are
0.005001. The correction covers observed factors and both R-reference
comparisons within each cell, not all estimated factors or all cells together. The `*_flag`
columns mark an absolute estimated effect of at least 0.5, not statistical
significance. At this cell, Rust/Python/R select Z20 10/0/2 times and Z74
0/8/8 times. In CD4, Z7 is selected in all ten runs for each backend; R selects
Z26 once, versus zero for Rust and Python.

### Cross-run overlap

Each entry is the mean Jaccard similarity for each pairing of independent
ten-run arms. We average the full cross-product of ten runs versus ten runs
(100 combinations per cell, not 100 independent observations). Jaccard is
0 for no overlap and 1 for identical sets; empty/empty is defined as 1. These
are not paired-seed comparisons and do not establish equivalent backends.

| Dataset/cell | Pair | Marginal mean Jaccard | Interaction mean Jaccard |
|---|---|---:|---:|
| SSc d=0.01 l=0.1 | Rust–Python | 0.673 | 0.545 |
| SSc d=0.01 l=0.1 | Rust–R | 0.705 | 0.532 |
| SSc d=0.01 l=0.1 | Python–R | 0.904 | 0.801 |
| SSc d=0.01 l=1.0 | Rust–Python | 0.788 | 0.719 |
| SSc d=0.01 l=1.0 | Rust–R | 0.750 | 0.667 |
| SSc d=0.01 l=1.0 | Python–R | 0.819 | 0.719 |
| SSc d=0.1 l=0.1 | Rust–Python | 0.322 | 0.352 |
| SSc d=0.1 l=0.1 | Rust–R | 0.523 | 0.418 |
| SSc d=0.1 l=0.1 | Python–R | 0.310 | 0.286 |
| SSc d=0.1 l=1.0 | Rust–Python | 0.533 | 0.589 |
| SSc d=0.1 l=1.0 | Rust–R | 0.589 | 0.648 |
| SSc d=0.1 l=1.0 | Python–R | 0.533 | 0.623 |
| CD4 d=0.1 l=0.1 | Rust–Python | 1.000 | 0.723 |
| CD4 d=0.1 l=0.1 | Rust–R | 0.950 | 0.717 |
| CD4 d=0.1 l=0.1 | Python–R | 0.950 | 0.751 |

The identities in these tables compare only within the same dataset and
parameter cell, where Z agreement was checked. SSc sample biological pairing
is still unverified, so this is not a biological identity claim. The saved
R-reference cross-run marginal means in [`ssc/tests.csv`](ssc/tests.csv) and
[`cd4/tests.csv`](cd4/tests.csv) match the raw-summary recomputation within
floating-point precision.

Reproduce the overlap calculation from the project root with the existing
loader and Jaccard helper (the raw results are required):

```python
from pathlib import Path
import sys
sys.path.insert(0, "benchmarks/scripts")
from aggregate_seed_sweep import load_sweep, _jaccard

for dataset in ("ssc", "cd4"):
    root = Path("benchmarks/results/2026-10-16_production-readiness/results") / dataset
    arms = {name: load_sweep(root / name)["selections"] for name in ("rust", "python", "pure_r")}
    cells = sorted({(delta, lam) for arm in arms.values() for _, delta, lam in arm})
    for delta, lam in cells:
        for left, right in (("rust", "python"), ("rust", "pure_r"), ("python", "pure_r")):
            for kind in ("marginal", "interaction"):
                a = arms[left][(left, delta, lam)][kind]
                b = arms[right][(right, delta, lam)][kind]
                mean = sum(_jaccard(x, y) for _, x in a for _, y in b) / (len(a) * len(b))
                print(dataset, delta, lam, left, right, kind, mean)
```
