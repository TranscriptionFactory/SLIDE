# SLIDE Backend Benchmark Report

Generated: 2026-06-20 13:39:04

## Timing

| Backend | Delta | Lambda | Seconds | Status |
|---------|------:|-------:|--------:|--------|
| rust | 0.01 | 0.1 | 61.81 | success |
| rust | 0.01 | 1.0 | 64.51 | success |
| rust | 0.1 | 0.1 | 18.48 | success |
| rust | 0.1 | 1.0 | 20.49 | success |

### Total time per backend

| Backend | Total (s) | Speedup vs slowest |
|---------|----------:|-------------------:|
| rust | 165.30 | 1.0x |

## Results per backend

| Backend | Delta | Lambda | K | Sig LFs | Sig Interacts |
|---------|------:|-------:|--:|--------:|--------------:|
| rust | 0.01 | 0.1 | 172 | 2 | 3 |
| rust | 0.01 | 1.0 | 172 | 2 | 3 |
| rust | 0.1 | 0.1 | 88 | 2 | 7 |
| rust | 0.1 | 1.0 | 88 | 2 | 6 |

## Pairwise correctness comparison

| Backend A | Backend B | Delta | Lambda | Sig LF Jaccard | Z overall r | Z median col r | Z min col r | A nonzero Jaccard |
|-----------|-----------|------:|-------:|---------------:|------------:|---------------:|------------:|------------------:|
