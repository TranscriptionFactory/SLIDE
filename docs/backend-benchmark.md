# SLIDE Backend Runtime Benchmark

Comparison of Python, R (rpy2), and Rust backends on the SSc UnTx dataset.

**Status:** Rust is slower than expected — needs profiling and optimization.

## Setup

- **Dataset:** SSc UnTx (X.csv / Y.csv), 24 samples × 804 features
- **Parameter grid:** delta ∈ {0.01, 0.1}, lambda ∈ {0.1, 1.0} → 4 combos per backend
- **Knockoff iterations:** 500
- **Threads:** 4
- **Cluster:** htc (CRC), 150 GB RAM, 4 CPUs per task
- **Date:** 2026-05-21

### Backend descriptions

| Backend | LOVE | Knockoffs | Notes |
|---------|------|-----------|-------|
| `python` | Python | Python | Pure Python pipeline via `loveslide` |
| `r_knockoffs` | Python | R (via rpy2) | Python LOVE + R knockoff package |
| `rust` | Rust | Rust | Full Rust pipeline via `slide-rs` |

## Runtime Results

### Per-combination wall time (seconds)

| delta | lambda | python | r_knockoffs | rust |
|------:|-------:|-------:|------------:|-----:|
| 0.01 | 0.1 | 1012.6 | 699.9 | >1320 (still running at 22 min) |
| 0.01 | 1.0 | 984.5 | 671.7 | TBD |
| 0.1 | 0.1 | TBD | 236.2 | TBD |
| 0.1 | 1.0 | TBD | 367.0 | TBD |

### Totals (completed combos)

| Backend | Combos done | Total (s) | Mean per combo (s) |
|---------|:-----------:|----------:|--------------------:|
| python | 2/4 | 1997.1 | 998.6 (~16.6 min) |
| r_knockoffs | 4/4 | 1974.8 | 493.7 (~8.2 min) |
| rust | 0/4 | — | >22 min on first combo |

### Key finding: Rust is NOT faster

The Rust backend has been running for 22+ minutes on (delta=0.01, lam=0.1)
without completing, while:
- Python finished the same combo in **1012s (~17 min)**
- R knockoffs finished it in **700s (~12 min)**

This means the Rust implementation is **slower than both Python and R** on
the most expensive parameter combination. The bottleneck is likely in the
knockoff iteration loop (500 iterations of SDP solve + glmnet), not in
language overhead.

### Other observations

1. **R knockoffs are ~2x faster than pure Python** for the knockoff generation
   step. The LOVE step is identical (both use Python LOVE).

2. **delta=0.1 runs are much faster than delta=0.01** — smaller delta → higher
   rank LOVE decomposition → more knockoff variables → more work per iteration.

## Performance target

For the Rust backend to be competitive, it should complete (delta=0.01,
lam=0.1) on SSc UnTx in **under 700s** (matching R knockoffs). A minimal
first milestone is **under 22 minutes** (matching current Python).

### Quick smoke test for Rust performance

```bash
# Single-combo test: delta=0.01, lam=0.1 on SSc UnTx
# Target: <22 min (parity with Python), stretch goal <12 min (parity with R)
time target/release/slide run \
    --config benchmark_results/rust/0.01_0.1_out/rust_config.yaml \
    --checkpoint /tmp/bench_checkpoint.json \
    --threads 4
```

## Correctness

Pairwise correctness comparison (sig LF Jaccard, Z-matrix correlation,
A-matrix nonzero overlap) will be generated automatically by the comparison
job once all backends complete. Results will appear in
`benchmark_results/comparison_report.md`.

## How to reproduce

```bash
# Submit all 3 backends in parallel + comparison job
sbatch benchmarks/run_benchmark.sbatch           # array job: tasks 0-2
sbatch --dependency=afterok:<ARRAY_JOB_ID> \
       benchmarks/run_benchmark_compare.sbatch    # comparison
```

Or run locally (sequential):
```bash
pixi run -e dev python benchmarks/benchmark_backends.py
```
