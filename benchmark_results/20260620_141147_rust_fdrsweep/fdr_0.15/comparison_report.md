# SLIDE Backend Benchmark Report

Generated: 2026-06-20 14:19:11

## Timing

| Backend | Delta | Lambda | Seconds | Status |
|---------|------:|-------:|--------:|--------|
| rust | 0.01 | 0.1 | 101.37 | success |
| rust | 0.01 | 1.0 | 97.54 | success |
| rust | 0.1 | 0.1 | 33.20 | success |
| rust | 0.1 | 1.0 | 43.82 | success |

### Total time per backend

| Backend | Total (s) | Speedup vs slowest |
|---------|----------:|-------------------:|
| rust | 275.94 | 1.0x |

## Results per backend

| Backend | Delta | Lambda | K | Sig LFs | Sig Interacts |
|---------|------:|-------:|--:|--------:|--------------:|
| rust | 0.01 | 0.1 | 172 | 9 | 16 |
| rust | 0.01 | 1.0 | 172 | 8 | 19 |
| rust | 0.1 | 0.1 | 88 | 4 | 20 |
| rust | 0.1 | 1.0 | 88 | 7 | 29 |

## Pairwise correctness comparison

| Backend A | Backend B | Delta | Lambda | Sig LF Jaccard | Z overall r | Z median col r | Z min col r | A nonzero Jaccard |
|-----------|-----------|------:|-------:|---------------:|------------:|---------------:|------------:|------------------:|
