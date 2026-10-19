# Backend benchmarks

## Maintained tools and saved results

Use the tools in [`scripts/`](scripts/) and find saved measurements in
[`results/`](results/).

See the [retained-evidence index](results/README.md) for current
runtime, memory and significant-LF comparisons. Keep each new run in a dated,
descriptive directory under `benchmarks/results/`, with its command/configuration
and differences from the preceding run in a `README.md`.

- `benchmark_backends.py`, `run_slide_r.R`, `run_backend_array.sh`: backend
  execution, including bounded manifest runs. `vendor_slide_opt_iter.patch`
  supports the optional vendored-R selection mode.
- `generate_factor_inputs.py`: shared known-truth simulation inputs.
- `analyze_seed_replicates.py`, `aggregate_seed_sweep.py`: significant-LF/set
  comparisons and statistical summaries.
- `compare_timings.py`: runtime/memory summaries and figures from manifests.
- `prepare_paired_cell.py`, `paired_xtilde.py`, `paired_xtilde.R`,
  `analyze_paired_xtilde.py`, `replay_paired_xtilde.py`: saved-draw preparation,
  replay and analysis for investigating selection differences.

The 13 files above support benchmark execution, analysis and paired-draw checks.
The fixed-dataset diagnostics [replay_jitter.py](https://github.com/TranscriptionFactory/SLIDE/blob/0c030651bf4db5dd0d184e09f7274ab319d66bce/benchmarks/scripts/replay_jitter.py),
[replay_gaussian_parity.py](https://github.com/TranscriptionFactory/SLIDE/blob/0c030651bf4db5dd0d184e09f7274ab319d66bce/benchmarks/scripts/replay_gaussian_parity.py) and its
[r_stat.R helper](https://github.com/TranscriptionFactory/SLIDE/blob/0c030651bf4db5dd0d184e09f7274ab319d66bce/archive/2026-10-17_production-cleanup/benchmark_results/2026-10-13_s-zero-handling/r_stat.R) are historical source, not maintained
entrypoints. They require private scratch trees or fixed historical controls.

Find one-off diagnostics, old plotting tools and machine-specific launchers in
the historical [cleanup archive](https://github.com/TranscriptionFactory/SLIDE/blob/0c030651bf4db5dd0d184e09f7274ab319d66bce/archive/2026-10-17_production-cleanup/README.md).
Use the retained run commands rather than those historical launchers.
Commands recorded inside frozen runs may use the original `benchmarks/<script>`
layout after entering a frozen source tree; maintained current-checkout tools
live under `benchmarks/scripts/`.

## Runtime and memory comparisons

These are profiled measurements of frozen source `4b32111`, not benchmarks of
every subsequent change. All rows use 500 knockoff draws, four workers, one BLAS
thread and homogeneous LOVE; final predictive CV is excluded. CD4 and synthetic
capacity runs use Gaussian parity settings. The localized scleroderma (LS) dataset uses native binary-response
settings, whose solver/grid differences are intentional. Delta/lambda are 0.1/0.1.

| Dataset (samples n × features p) | Runs per backend | Estimated factors K (Rust / Python / R) | Rust wall seconds | Python wall seconds | R wall seconds |
| --- | ---: | --- | ---: | ---: | ---: |
| CD4 (2,482 × 1,776) | 10 | 39 / 39 / 39 | 5.31 [5.03–5.58] | 22.07 [21.85–22.83] | 93.52 [91.32–98.60] |
| Synthetic (50 × 10,000) | 1 | 178 / 178 / unavailable | 115.93 | 111.85 | timeout at 7,200 |
| Synthetic (50 × 15,000) | 1 | 136 / 136 / unavailable | 76.86 | 114.00 | timeout at 7,200 |
| Synthetic (50 × 20,000) | 1 | 109 / 109 / unavailable | 90.69 | 137.00 | timeout at 7,200 |
| LS skin transcriptomics (44 × 15,496) | 1 | 434 / 434 / unavailable | 1,305.27 | 4,817.10 | timeout at 7,200 |

The LS benchmark uses processed, cluster-specific expression summaries from
skin single-cell RNA sequencing: 44 samples and 15,496 features, with a binary
LS/control response. The study includes adult and pediatric participants;
see [Rosen et al., *JCI Insight* (2025), DOI: 10.1172/jci.insight.185758](https://doi.org/10.1172/jci.insight.185758).
The benchmark uses the full cohort, rather than an age subgroup; it does not
rerun the paper's predictive cross-validation.

CD4 entries are median [25th–75th percentile], not confidence intervals. Sources:
[CD4 analysis and provenance](results/2026-10-16_production-readiness/README.md),
[capacity results](results/2026-10-16_large-feature-scaling/README.md)
and [LS skin transcriptomics results and provenance](results/2026-10-17_torok-all-binary/README.md).
For original-to-rewritten source revisions, see the
[revision mapping](#source-revisions-after-history-cleanup).
Capacity factor counts come from the corresponding per-backend
`results/<dataset>/<backend>/seed_20000/benchmark_summary.json` files retained
on the shared filesystem. The synthetic inputs have planted K=20, far below
estimated K=109–178: successful execution does not establish factor recovery.
Different p values use nonnested datasets, so runtime need not increase with p.

Peak sampled process-group RSS for Rust/Python was 2.30/4.73 GiB at p=10,000,
5.02/10.31 GiB at p=15,000, 8.97/18.13 GiB at p=20,000, and 3.79/9.43 GiB for
LS skin transcriptomics (bytes divided by 2³⁰). Sampling every 50 ms can miss peaks and double-count
shared pages; these are not exact cgroup peaks. Timings include fresh-process
startup, imports, input I/O and preprocessing. Profiling overhead is unmeasured;
caches were not flushed. Shared AMD EPYC 9575F nodes can experience interference.
Singleton rows provide capacity observations, not timing uncertainty or universal
speedup claims. R timeouts are bounded observations, not completed runtimes.

## Latent-factor comparisons

### Rust, Python and R pilot

For CD4, all three implementations estimated K=39 with maximum absolute latent-score
matrix difference below 9.1×10⁻¹². Rust and Python selected marginal Z7 in all ten
runs; R selected it in all ten and one extra marginal in one run. Median selected
interaction count was two for each backend; equal counts do not prove equal sets.

For SSc (n=24, p=804), latent-score matrices agree within 4.7×10⁻¹¹
across four parameter cells, but marginal selections differ. At delta=0.01,
lambda=0.1, Rust selected Z20 in 10/10 runs, Python in 0/10 and R in 2/10;
Rust selected Z74 in 0/10 runs; Python and R each selected it in 8/10.
Selection frequencies alone do not establish statistical significance.

The [pilot selection report](results/2026-10-16_production-readiness/README.md#backend-selection-details)
contains full selection-count tables, marginal and interaction Jaccard
comparisons, CSVs, heatmaps and provenance for source `4b32111`.
Mean cross-run Jaccard averages all 100 combinations of ten runs from two arms.
Those pairs reuse runs; they are neither 100 independent observations nor
paired-seed comparisons. Ten repeats do not establish equivalence.

![SSc marginal selection rates](results/2026-10-16_production-readiness/ssc/factor_rates.png)

*Ten runs per backend. The heatmap shows factors with selection rates of at
least 20% in any arm. See the benchmark CSVs for pointwise Wilson intervals;
the colors show no uncertainty. Compare factor identities within a
dataset/parameter cell, not across cells.*

### R-knockoff hybrid comparisons

The separate `c4726e3` campaign compares native Rust, native Python, pure R,
Rust with R-generated knockoffs and Python with R knockoffs through `rpy2`.
Do not pool it with the pilot. At delta=0.01, lambda=0.1, native Rust selected
Z20 in 10/10 runs versus 0/10 for Python and R, 1/10 for the Rust hybrid and
0/10 for the Python hybrid. For Z74, the counts were Rust 0/10, Python 8/10,
R 9/10, Rust hybrid 6/10 and Python hybrid 8/10.

- [Four-arm statistical analysis](results/2026-10-16_backend-replicates/README.md):
  adjusted per-factor tests, overlap comparisons and complete Rust-hybrid counts.
- [Five-arm heatmap and rates](results/2026-10-17_hybrid-selection/README.md):
  includes both hybrids, with selected/completed denominators and missing cells.
- [Python-hybrid coverage and alignment](results/2026-10-16_backend-replicates/SETUP.md#python-with-r-knockoffs-partial-evidence-only):
  17/40 cells completed before the original ten tasks timed out. The seven-run
  cell has completion bias; missing cells are not zero selections. The 17
  completed score matrices match R within 1.12×10⁻¹¹ with matching row/column order.
- [Extended-limit rerun](results/2026-10-17_python-r-knockoffs-extended/README.md):
  we resubmitted ten Python-hybrid tasks with two hours each. Use the original
  figures until we validate the rerun.

Median wall seconds per completed cell (ten repeats unless noted):

| Delta | Lambda | Rust | Python | R | Rust with R knockoffs | Python with R knockoffs |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| 0.01 | 0.1 | 35.66 | 27.64 | 61.43 | 43.53 | 581.0 (10 completed) |
| 0.01 | 1.0 | 29.61 | 24.13 | 58.36 | 45.86 | 521.7 (7 completed) |
| 0.1 | 0.1 | 14.49 | 12.35 | 25.49 | 15.85 | No completed cells |
| 0.1 | 1.0 | 14.23 | 12.40 | 23.76 | 14.28 | Not started |

These timing boundaries differ from the pilot's fresh-process timings;
see the [replicate report](results/2026-10-16_backend-replicates/README.md)
for details.

### Interpretation limits

SSc input/response identifiers differ; the benchmark preserves historical positional
pairing, without verifying biological sample identity. For [LS skin transcriptomics](results/2026-10-17_torok-all-binary/README.md),
Rust and Python both report marginal Z294, but Rust reports six interactions and Python five.
Matching K or a factor label alone does not establish aligned factor scores.

Compare factor labels within an aligned dataset/parameter cell. Equal
selection counts do not imply equal factor sets, and matching scores does not
establish selection equivalence or FDR control. See the
[pilot limits and controlled replay](results/2026-10-16_production-readiness/README.md)
for solver interventions and incomplete simulation coverage.

Independent random streams and near-singular least-squares rank decisions can
produce different control-performance scores. KDE uses R's Gaussian kernel and
`bw.nrd0` bandwidth, not identical FFT discretization or pixel layout.

## Configuration modes

`SLIDE_KNOCKOFF_TOL`, `SLIDE_KNOCKOFF_MAX_ITER`, `SLIDE_KNOCKOFF_MATCH_R_LAMBDA`
and `SLIDE_KNOCKOFF_MATCH_R_COV` override knockoff fitting for R-parity
experiments. When any is set, `slide` warns and stores it under
`config.env_overrides` in the checkpoint.

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
Both modes honor pure R's requested worker budget without modifying the
installed R package. SLURM submission rejects a worker budget above its CPU request.

## Running

**Rebuild the Rust executable from this source before timing.** Older executables
ignore the λ-range environment setting; the harness cannot detect that.
Build outside the benchmark timer and pass `--rust-bin` for usable timing results.
Use the existing project BLAS/environment setup as needed.

```bash
cargo build --release -p slide-rs
python benchmarks/scripts/benchmark_backends.py \
  --configuration parity --threads 4 --knockoff-base-seed 42 \
  --rust-bin "$PWD/target/release/slide" \
  --out-dir benchmarks/results/2026-09-14_gaussian-parity
```

Add input paths as needed. Use a fresh output directory for each configuration;
do not combine saved native and parity timings. For a frozen source copy, pass
`--source-revision <executed-commit>`; metadata keeps that declaration separate
from `enclosing_git_sha`. Metadata also records inherited SLIDE environment
variables, Rust solver overrides, λ counts, and worker budgets. It records
configuration, not a measurement of achieved convergence.

For native defaults, unset solver override variables before selecting
`--configuration native`. Native mode preserves inherited overrides.
The September 12 frozen executable and run remain native-configuration results.

## Interpretation and limits

Matching tolerance values does not establish equal work. The Gaussian stopping
metrics use comparable variance-scaled coefficient updates, but Rust applies its
iteration cap per λ and candidate-set refit. Fortran glmnet counts sweeps across
the whole path. Rust currently returns coefficients on cap exhaustion without a
convergence error; matching the cap value does not remove this limitation.

Tighter tolerance can increase runtime. Compare convergence and W statistics
as well as time when evaluating parity settings.

The basic harness has these limitations: backend-specific RNG streams and swaps;
Python excludes imports from timing while Rust/R include subprocess startup;
one pass per cell in fixed order. A future equal-work study needs paired saved
draws, convergence diagnostics, aligned timing boundaries, repeated cells, and
randomized backend order. The [readiness pilot](results/2026-10-16_production-readiness/README.md#measurement)
uses fresh processes and repeated cells; consult each campaign for its timing boundaries.

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

## Source revisions after history cleanup

On 2026-10-17, `git-filter-repo` removed root `tmpfiles/` and `.tmpfiles/`
paths from local history. Measurement files did not change. Run reports retain
the original revisions that were executed; use these filtered revisions when
inspecting the corresponding source in this checkout:

| Recorded revision | Filtered revision | Evidence |
| --- | --- | --- |
| `4b321116fde6b92d5eb7bd771e0357cfb8dd72c4` | `bdc1054b3c175937f5dd8741a6dc840579da29ad` | Readiness pilot, scaling and LS skin transcriptomics production source |
| `c4726e34fde91e59429f88d2fd228f785ce845dc` | `f241643ae63cf8a2b592eeeb3ed5042f85bc9881` | Five-arm replicate campaign source |
| `5bcde18f9bef1290239b2d8b429abc50cec34dfc` | `836b0f8774e0ca864b95e503a89a8c0c08b59294` | Readiness analysis source |

The full local mapping is in `.git/filter-repo/commit-map`. The ignored
`archive/2026-10-17_history-backup/` preserves the unfiltered Git database,
all three stashes, retired scratch files and a copy of the mapping. Neither
backup nor local staging data is part of a fresh clone. Replay helpers live
under `benchmarks/scripts/`; historical scratch inputs remain external
reproduction requirements.
