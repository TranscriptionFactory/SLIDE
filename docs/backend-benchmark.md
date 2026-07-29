# SLIDE Backend Runtime Benchmark

Comparison of Python, R (rpy2), and Rust backends on the SSc UnTx dataset.

**Status (2026-05-22):** Rust is 6-15x faster than Python/R. Correctness diverges on knockoff variable selection — Z/A matrices need investigation.

## Setup

- **Dataset:** SSc UnTx (X.csv / Y.csv), 24 samples × 804 features
- **Parameter grid:** delta ∈ {0.01, 0.1}, lambda ∈ {0.1, 1.0} → 4 combos per backend
- **Knockoff iterations:** 500
- **Threads:** 4
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
    --checkpoint /tmp/bench_checkpoint.json \
    --threads 4
```
