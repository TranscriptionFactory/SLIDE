# SLIDE Backend Benchmark Report

Generated: 2026-06-13 08:05:36

## Timing

| Backend | Delta | Lambda | Seconds | Status |
|---------|------:|-------:|--------:|--------|
| rust | 0.01 | 0.1 | 56.61 | success |
| rust | 0.01 | 1.0 | 65.34 | success |
| rust | 0.1 | 0.1 | 31.82 | success |
| rust | 0.1 | 1.0 | 34.86 | success |

### Total time per backend

| Backend | Total (s) | Speedup vs slowest |
|---------|----------:|-------------------:|
| rust | 188.62 | 1.0x |

## Results per backend

| Backend | Delta | Lambda | K | Sig LFs | Sig Interacts |
|---------|------:|-------:|--:|--------:|--------------:|
| rust | 0.01 | 0.1 | 172 | 2 | 3 |
| rust | 0.01 | 1.0 | 172 | 2 | 3 |
| rust | 0.1 | 0.1 | 88 | 2 | 6 |
| rust | 0.1 | 1.0 | 88 | 2 | 8 |

## Pairwise correctness comparison

| Backend A | Backend B | Delta | Lambda | Sig LF Jaccard | Z overall r | Z median col r | Z min col r | A nonzero Jaccard |
|-----------|-----------|------:|-------:|---------------:|------------:|---------------:|------------:|------------------:|
