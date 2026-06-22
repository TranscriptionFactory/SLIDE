# SLIDE Backend Benchmark Report

Generated: 2026-06-20 14:25:04

## Timing

| Backend | Delta | Lambda | Seconds | Status |
|---------|------:|-------:|--------:|--------|
| rust | 0.01 | 0.1 | 167.86 | success |
| rust | 0.01 | 1.0 | 143.48 | success |
| rust | 0.1 | 0.1 | 21.15 | success |
| rust | 0.1 | 1.0 | 20.43 | success |

### Total time per backend

| Backend | Total (s) | Speedup vs slowest |
|---------|----------:|-------------------:|
| rust | 352.92 | 1.0x |

## Results per backend

| Backend | Delta | Lambda | K | Sig LFs | Sig Interacts |
|---------|------:|-------:|--:|--------:|--------------:|
| rust | 0.01 | 0.1 | 172 | 19 | 77 |
| rust | 0.01 | 1.0 | 172 | 15 | 57 |
| rust | 0.1 | 0.1 | 88 | 2 | 11 |
| rust | 0.1 | 1.0 | 88 | 2 | 22 |

## Pairwise correctness comparison

| Backend A | Backend B | Delta | Lambda | Sig LF Jaccard | Z overall r | Z median col r | Z min col r | A nonzero Jaccard |
|-----------|-----------|------:|-------:|---------------:|------------:|---------------:|------------:|------------------:|
