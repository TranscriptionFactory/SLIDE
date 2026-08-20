# SLIDE Backend Runtime Benchmark

Comparison of Python, R (rpy2), and Rust backends on the SSc UnTx dataset.

**Status (2026-08-19):** Rust remains substantially faster. A ten-seed,
high-delta sweep now measures parity by stable selections rather than by a
single stochastic run; marginal parity improved, but interaction parity and
the `lambda=1.0` Rust marginal core still diverge from native R.

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

- **stability rule (default)** — report the frequency set: every factor whose
  selection count reaches `ceil(niter * spec)`.
- **`findOptIter`** — R SLIDE's original rule. Discard the frequency table and
  report a *single* draw out of `niter`: the one with maximal overlap with the
  frequent set, ties broken by shortest.

`findOptIter` is a variance amplifier by construction, and it dominates
cross-backend disagreement. Native R cannot reproduce itself under it: in the
2026-08-19 sweep `pure_r` selected 11 distinct marginal factors across ten
seeds at `delta=0.1`/`lambda=0.1`, of which only two appeared in most runs.
No amount of numerical alignment can lift any backend above that floor while
`findOptIter` is the final rule.

The stability rule is therefore the default in Rust
(`KnockoffConfig::slide_selection`) and Python
(`Knockoffs.select_short_freq_slide`). **From here on, "parity with R" in this
document means parity with R running the stability rule.** Exact fidelity to
published native-R SLIDE is not the target; bounded cross-backend divergence
is.

Reaching the rule per backend:

| Backend | Stability rule (default) | `findOptIter` |
|---|---|---|
| Rust | default | `slide_selection: true` in the run YAML |
| Python | default | `slide_selection=True` keyword |
| R | `SLIDE_R_SOURCE=vendor` + `SLIDE_OPT_ITER=0` | default (installed package) |

The vendored SLIDE clone keeps `findOptIter` as *its* default deliberately: it
is third-party source, and an unflagged change there would make the
vendor-vs-installed equivalence check meaningless. `benchmarks/run_slide_r.R`
evaluates the vendored tree into an environment parented on the installed
package's imports, so unqualified calls resolve exactly as they do inside the
package. `benchmarks/benchmark_backends.py --slide-selection /
--no-slide-selection` sets all of this for every backend at once; an explicit
`SLIDE_R_SOURCE` or `SLIDE_OPT_ITER` in the environment wins over the derived
value.

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
sbatch benchmarks/run_benchmark.sbatch           # array job: tasks 0-2
sbatch --dependency=afterok:<ARRAY_JOB_ID> \
       benchmarks/run_benchmark_compare.sbatch    # comparison + viz

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
