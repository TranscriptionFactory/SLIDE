# SLIDE Backend Runtime Benchmark

## Evidence status (2026-10-17)

The August and May measurements below describe historical revisions and settings.
Do not use their 6–15× multipliers as current release claims. The
[README comparison](../../README.md#measured-backend-comparisons) summarizes the
[2026-10-16 frozen-candidate pilot](../../benchmarks/results/2026-10-16_production-readiness/README.md),
[large-feature capacity run](../../benchmarks/results/2026-10-16_large-feature-scaling/README.md),
and [localized scleroderma skin transcriptomics run](../../benchmarks/results/2026-10-17_torok-all-binary/README.md).
Those runs include profiling overhead and do not establish statistical equivalence.
The historical `loveslide` name below identifies the package used at run time;
the current distribution is `slide-py`, imported as `slide_py`.


Comparison of Python, R (rpy2), and Rust backends on the SSc UnTx dataset.

**Status (2026-08-20):** Rust remains substantially faster. Switching every
backend to the stability selection rule made Python and Rust individually more
reproducible but *widened* cross-backend divergence, so the change was measured
and reverted — see "2026-08-20 stability-rule seed sweep" below. `findOptIter`
remains the default everywhere; the stability rule is opt-in per run.

## 2026-08-20 stability-rule seed sweep

Same grid as the 2026-08-19 sweep plus `delta=0.01`: seeds 42, 1000, ..., 9000,
500 knockoff iterations, `spec=0.1`, 10 workers, BLAS pinned to 1, Turin nodes.
Every backend ran the stability rule (`--no-slide-selection`); all 40 `pure_r`
cells are confirmed to have run `SLIDE_R_SOURCE=vendor` with `SLIDE_OPT_ITER=0`.
Results in `benchmark_results/20260819_stability-rule-seeds/`.

Stable-core Jaccard against `pure_r`, versus the 2026-08-19 baseline:

| Backend | Delta | Lambda | Marginal (base → stability) | Interaction (base → stability) |
|---|---:|---:|---|---|
| Rust | 0.1 | 0.1 | 1.000 → **0.333** | 0.214 → **0.125** |
| Rust | 0.1 | 1.0 | 0.250 → 0.250 | 0.214 → 0.214 |
| Python | 0.1 | 0.1 | 0.500 → **0.333** | 0.333 → **0.167** |
| Python | 0.1 | 1.0 | 0.667 → 0.667 | 0.400 → 0.400 |

The plan's success criteria — Jaccard materially up at `lambda=1.0` and on both
interaction rows, with `lambda=0.1` holding at 1.000 — are **not met**.

### Why: the reference moved, the backends did not

Per-LF selection rates at `delta=0.1`/`lambda=0.1` locate the cause precisely:

| Backend | Distinct LFs (base → stability) | Stable core (base → stability) |
|---|---|---|
| `pure_r` | 11 → 12 | `{Z5, Z44, Z61}` → `{Z5, Z25, Z33, Z44, Z53}` |
| `rust` | 4 → 4 | `{Z5, Z44, Z61}` → `{Z5, Z44, Z61}` (unchanged) |
| `python` | 11 → 8 | `{Z25, Z44, Z61}` → `{Z25, Z44, Z61}` (unchanged) |

Rust's stable core is identical under both rules; Python's is too. The entire
Jaccard drop is `pure_r` moving away from them. The predicted mechanism — R's
11-LF scatter collapsing toward its stable core — did not occur: R's scatter is
essentially unchanged (11 → 12 distinct) while its core gained three factors and
lost `Z61`.

The rule did deliver reproducibility *within* the non-R backends, which is a real
if separate result: Python's marginal scatter fell from 11 to 8 distinct factors
and its interaction scatter from 65 to 47; Rust's `lambda=1.0` marginal scatter
fell to zero (2 distinct factors, both in the core).

R does not stabilize because `selectShortFreq` applies the frequency cut twice
under `SLIDE_OPT_ITER=0` — once per chunk, then again in stage-two aggregation —
whereas `findOptIter` hands stage two a whole draw including sub-threshold
variables. The two rules therefore screen different candidate sets, so this is
not a like-for-like threshold change. In a five-seed probe the stability rule
returned an empty selection on 2 of 5 seeds: the honest answer when nothing
clears 50/500 in stage two, where `findOptIter` instead returns an arbitrary
shortest draw.

At `lambda=1.0` every backend's numbers are bit-identical to the baseline: the
selection is already degenerate there, so the rule does not bite.

### It fails without privileging R either

The goal was bounded cross-backend divergence, which does not require treating
R as the reference. Measured over all three backend pairs on the four
comparable `delta=0.1` cells:

| Pair | Marginal (base → stability) | Interaction (base → stability) |
|---|---|---|
| `rust`/`python`, λ=0.1 | 0.500 → 0.500 | 0.278 → 0.294 |
| `rust`/`pure_r`, λ=0.1 | 1.000 → **0.333** | 0.214 → **0.125** |
| `python`/`pure_r`, λ=0.1 | 0.500 → **0.333** | 0.333 → **0.167** |
| `rust`/`python`, λ=1.0 | 0.333 → 0.333 | 0.308 → 0.308 |
| `rust`/`pure_r`, λ=1.0 | 0.250 → 0.250 | 0.214 → 0.214 |
| `python`/`pure_r`, λ=1.0 | 0.667 → 0.667 | 0.400 → 0.400 |

**Mean pairwise stable-core Jaccard: 0.416 → 0.327.** The R-free `rust`/`python`
pair is flat, so this is not "R is the odd one out" — the rule simply does not
move the backends toward each other.

### Disposition

The defaults were reverted to `findOptIter` on 2026-08-20. What was kept:

- the `slide_selection` flag on all three backends, and `SLIDE_R_SOURCE=vendor`,
  which is the first time the full-pipeline R benchmark can see *any* edit to
  the vendored tree (it previously called the installed `SLIDE::optimizeSLIDE`);
- the Rust interaction size cap, which closes the last structural difference as
  a hypothesis even though it never fires here;
- these measurements, so the hypothesis is not retried.

### delta=0.01 (new, no committed baseline)

Agreement at low delta is strong and was not measured before: Rust marginal
Jaccard 1.000 at `lambda=0.1` and 0.800 at `lambda=1.0`; Python 0.667 and 1.000.
Cores largely coincide, e.g. all three backends return `{Z18, Z27, Z45, Z136}`
plus one factor at `delta=0.01`/`lambda=0.1`.

### Interaction size cap

The Rust cap never fired in any of the 40 Rust cells: with `n=24` the cap is
`24 - #marginals` (about 20) and no marginal selected more than six interactors.
The change is therefore correct but unexercised on this dataset, and every
interaction movement above is attributable to the selection rule alone.

## 2026-08-19 high-delta seed sweep

The latest run used `delta=0.1`, `lambda in {0.1, 1.0}`, 500 knockoff
iterations, 10 workers, and seeds 42, 1000, ..., 9000. A stable core contains
features selected in at least half of the ten runs.

| Backend | Lambda | Marginal Jaccard | Marginal recall | Interaction Jaccard | Interaction recall |
|---|---:|---:|---:|---:|---:|
| Python | 0.1 | 0.500 | 0.667 | 0.333 | 1.000 |
| Python | 1.0 | 0.667 | 0.667 | 0.400 | 0.571 |
| Rust | 0.1 | 1.000 | 1.000 | 0.214 | 1.000 |
| Rust | 1.0 | 0.250 | 0.333 | 0.214 | 0.429 |

All comparisons use native R as the reference. The machine-readable report is
in `benchmark_results/20260819_high-delta-seeds/`.

Changes validated by this sweep:

- Benchmark cells write isolated timing/profile artifacts before aggregation,
  avoiding concurrent lost updates.
- Python now batches cached knockoff iterations into a bounded number of jobs,
  reducing dispatch and serialization overhead without changing seed order.
- Python and Rust assign disjoint deterministic seed ranges to chunks, stage-2
  voting, and interaction subproblems. The benchmark seed is now passed into
  the Python pipeline.
- Python interaction output now records the complete factor pair, matching the
  R and Rust artifact schema.

Profiling attributes 97--99% of Python knockoff time to Lasso-path fitting and
less than 1% to preparation, so no speculative preparation refactor was made.
Wall times varied materially across cluster nodes; batching is structurally
cheaper, but this sweep does not support a clean wall-time speedup claim.

Increasing Rust's path from 250 to 500 lambdas improved paired per-draw
statistics but did not improve the stable-core result, so the faster production
default remains unchanged. The remaining high-delta mismatch is therefore
most consistent with cumulative stochastic/path/threshold sensitivity and
needs a separately scoped numerical investigation rather than a special case.

## Final selection rule (`slide_selection`)

Every backend ends each knockoff stage with a choice of two rules:

- **stability rule (opt-in)** — report the frequency set: every factor whose
  selection count reaches `ceil(niter * spec)`.
- **`findOptIter`** — R SLIDE's original rule. Discard the frequency table and
  report a *single* draw out of `niter`: the one with maximal overlap with the
  frequent set, ties broken by shortest.

`findOptIter` is a variance amplifier by construction, and native R does not
reproduce itself under it: in the 2026-08-19 sweep `pure_r` selected 11 distinct
marginal factors across ten seeds at `delta=0.1`/`lambda=0.1`, of which only two
appeared in most runs.

That made the stability rule an obvious candidate for reducing cross-backend
divergence. It was tried as the default and **measured not to work** — mean
pairwise stable-core Jaccard fell from 0.416 to 0.327, and the one cell where
Rust and native R agreed perfectly dropped from 1.000 to 0.333. See the
2026-08-20 sweep below. **`findOptIter` therefore remains the default on every
backend**, and "parity with R" keeps its original meaning: parity with stock
native-R SLIDE.

The stability rule stays available per run, because it does reduce run-to-run
variance *within* a backend, which is a separate and sometimes useful property:

| Backend | `findOptIter` (default) | Stability rule |
|---|---|---|
| Rust | default | `slide_selection: false` in the run YAML |
| Python | default | `slide_selection=False` keyword |
| R | default (installed package) | `SLIDE_R_SOURCE=vendor` + `SLIDE_OPT_ITER=0` |

The vendored SLIDE clone keeps `findOptIter` as *its* default deliberately: it
is third-party source, and an unflagged change there would make the
vendor-vs-installed equivalence check meaningless. `benchmarks/run_slide_r.R`
evaluates the vendored tree into an environment parented on the installed
package's imports, so unqualified calls resolve exactly as they do inside the
package. `benchmarks/benchmark_backends.py --slide-selection /
--no-slide-selection` sets all of this for every backend at once (default
`--slide-selection`); an explicit `SLIDE_R_SOURCE` or `SLIDE_OPT_ITER` in the
environment wins over the derived value.

`vendor/` is gitignored, so the `SLIDE_OPT_ITER` gate itself lives only in the
working tree. Reapply it after a fresh checkout or a re-vendor with `git apply
benchmarks/vendor_slide_opt_iter.patch`. `run_slide_r.R` refuses to run with
`SLIDE_OPT_ITER=0` against an unpatched tree, so a forgotten patch fails loudly
rather than quietly reporting stock-rule numbers.

Note that `loveslide.knockoff.filter.knockoff_filter_voting_slide` — the
low-level wrapper, not the pipeline entry point — still defaults to
`slide_selection=True`. The pipeline always passes the flag explicitly, so this
affects only direct callers of that function.

## Interaction model size cap

R `interactionSLIDE` caps the interaction model at `m = n - #marginals` and
reruns the whole interaction pass with `spec + 0.1` whenever a marginal
overshoots it. Python mirrored this; Rust did not, and ran a single pass at a
fixed `spec` with no cap. Rust now implements the same control flow
(`MarginalLfSelectionConfig::interaction_size_cap`, default on), plus a
termination guard that returns no interactions once `spec` would exceed 1.0.
Set `interaction_size_cap: false` in the run YAML for the old single-pass
behaviour. Expect Rust interaction counts to differ from runs made before this
landed.

## Setup

- **Dataset:** SSc UnTx (X.csv / Y.csv), 24 samples × 804 features
- **Parameter grid:** delta ∈ {0.01, 0.1}, lambda ∈ {0.1, 1.0} → 4 combos per backend
- **Knockoff iterations:** 500
- **Worker budget:** 10 by default (`python_serial=1`, `python=n_workers`, Rust=Rayon threads)
- **BLAS threads:** 1 per worker to prevent nested oversubscription
- **Cluster:** htc (CRC), 150 GB RAM, 4 CPUs per task
- **SLURM job:** 9447800 (2026-05-22)

### Backend descriptions

| Backend | LOVE | Knockoffs | Notes |
|---------|------|-----------|-------|
| `python` | Python | Python | Pure Python pipeline via `loveslide` |
| `r_knockoffs` | Python | R (via rpy2) | Python LOVE/voting + R SDP and Gaussian draws |
| `rust` | Rust | Rust | Full Rust pipeline via `slide-rs` |
| `rust_r_knockoffs` | Rust | R (via `Rscript`) | Rust LOVE/voting + R SDP and Gaussian draws |
| `pure_r` | R | R | Native-R SLIDE reference |

The release comparison always runs all five modes on the same parameter grid.
The Rust YAML caller selects R-generated knockoffs with
`knockoff_backend: r` and an optional `rscript` path.

## Runtime Results

### Per-combination wall time (seconds)

| delta | lambda | Python | R knockoffs | Rust | Rust speedup (vs Python) |
|------:|-------:|-------:|------------:|-----:|-------------------------:|
| 0.01  | 0.1    | 689    | 713         | 106  | 6.5x                    |
| 0.01  | 1.0    | 651    | 701         | 107  | 6.1x                    |
| 0.1   | 0.1    | 530    | 663         | 43   | 12.3x                   |
| 0.1   | 1.0    | 550    | 130         | 41   | 13.4x                   |

### Totals

| Backend       | Total (s) | Mean per combo | Speedup vs Python |
|---------------|----------:|---------------:|------------------:|
| Python        | 2420      | 605s (~10 min) | 1.0x              |
| R knockoffs   | 2206      | 552s (~9 min)  | 1.1x              |
| Rust          | 298       | 75s (~1.2 min) | 8.1x              |

### Key findings

1. **Rust is 6-15x faster** than Python/R across all parameter combinations.
2. **Larger speedup at higher delta** (lower K): delta=0.1 (K=88) gives 12-13x vs delta=0.01 (K=172) gives 6x. Rust's knockoff implementation scales better with fewer latent factors.
3. **Python vs R knockoffs are similar** overall. R is slightly slower at delta=0.01 but much faster at (0.1, 1.0) — likely due to the R `knockoff` package's optimized Lasso path for smaller problems.

## Correctness

### Python vs R knockoffs (reference comparison)

Near-perfect agreement — these share the same LOVE step (Python), so differences are only in knockoff sampling.

| Metric | Range across param combos |
|--------|--------------------------|
| Z overall r | 1.000 |
| Z median col r | 1.000 |
| Z min col r | 1.000 |
| A nonzero Jaccard | 1.000 |
| Sig LF Jaccard | 0.25 – 1.00 |
| K | Identical (172 / 88) |

Sig LF variation is expected — knockoff selection is stochastic.

### Rust vs Python/R

| Metric | delta=0.01 | delta=0.1 | Notes |
|--------|-----------|-----------|-------|
| K | **Identical** (172 / 88) | **Identical** | LOVE rank agrees |
| A nonzero Jaccard | 0.993 | 0.986–0.996 | High structural agreement |
| Z overall r | 0.996–1.000 | 0.857–0.938 | Degrades at higher delta |
| Z median col r | 0.999 | 0.956–0.990 | Most columns match well |
| Z min col r | 0.979–0.999 | 0.560–0.793 | Some columns diverge significantly |
| Sig LF Jaccard | 0.33–0.75 | 0.11–0.50 | Substantial disagreement |
| Sig LF count | Rust: 2-3 | Rust: 2 | Rust is more conservative |
| | Python: 4-5 | Python: 4-6 | |

### Interpretation

- **LOVE step is consistent:** Same K, high A-matrix overlap, good Z correlations at small delta.
- **Knockoff selection diverges:** Rust finds fewer significant LFs and the sig LF overlap is poor, especially at delta=0.1.
- **Root cause candidates:**
  1. LOVE A-matrix differences (0.7–1.4% nonzero mismatch) propagate into Z = X × A
  2. Different SDP solver (Rust custom vs Python/R ASDP) produces different knockoff covariance
  3. Knockoff feature statistics (Lasso/Dantzig) differ in solver tolerance or path
  4. FDR threshold boundary effects amplify small upstream differences

**Next step:** Investigate where the Z/A divergence originates — LOVE decomposition vs knockoff sampling.

## How to reproduce

```bash
# Submit all 3 backends in parallel + comparison job
sbatch archive/2026-10-17_production-cleanup/benchmarks/run_benchmark.sbatch           # array job: tasks 0-2
sbatch --dependency=afterok:<ARRAY_JOB_ID> \
       archive/2026-10-17_production-cleanup/benchmarks/run_benchmark_compare.sbatch    # comparison + viz

# Timings are persisted to benchmark_results/{backend}/timings.json
# Plots saved to benchmark_results/figures/
```

Or run locally (sequential):
```bash
pixi run -e dev python benchmarks/benchmark_backends.py
```

### Quick smoke test for Rust performance

```bash
# Single-combo test: delta=0.01, lam=0.1 on SSc UnTx
time target/release/slide run \
    --config benchmark_results/rust/0.01_0.1_out/rust_config.yaml \
    --checkpoint .tmpfiles/bench_checkpoint.json \
    --threads 10
```

Add `--profile` to write `profile_summary.csv` with count, total, median, and
p95 time for sampling, preparation, lasso-path fitting, and thresholding.
Because workers overlap, summed stage time can exceed cell wall time.
