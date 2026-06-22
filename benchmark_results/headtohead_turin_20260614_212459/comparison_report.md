# SLIDE Backend Benchmark Report

Generated: 2026-06-14 21:29:26

## Timing

| Backend | Delta | Lambda | Seconds | Status |
|---------|------:|-------:|--------:|--------|
| pure_r | 0.01 | 0.1 | 50.60 | success |
| pure_r | 0.01 | 1.0 | 45.08 | success |
| pure_r | 0.1 | 0.1 | 18.74 | success |
| pure_r | 0.1 | 1.0 | 16.83 | success |
| rust | 0.01 | 0.1 | 38.32 | success |
| rust | 0.01 | 1.0 | 40.66 | success |
| rust | 0.1 | 0.1 | 18.50 | success |
| rust | 0.1 | 1.0 | 19.82 | success |

### Total time per backend

| Backend | Total (s) | Speedup vs slowest |
|---------|----------:|-------------------:|
| pure_r | 131.26 | 1.0x |
| rust | 117.31 | 1.1x |

## Results per backend

| Backend | Delta | Lambda | K | Sig LFs | Sig Interacts |
|---------|------:|-------:|--:|--------:|--------------:|
| pure_r | 0.01 | 0.1 | 172 | 5 | 14 |
| pure_r | 0.01 | 1.0 | 172 | 6 | 10 |
| pure_r | 0.1 | 0.1 | 88 | 3 | 3 |
| pure_r | 0.1 | 1.0 | 88 | 2 | 5 |
| rust | 0.01 | 0.1 | 172 | 2 | 3 |
| rust | 0.01 | 1.0 | 172 | 2 | 3 |
| rust | 0.1 | 0.1 | 88 | 2 | 6 |
| rust | 0.1 | 1.0 | 88 | 2 | 8 |

## Pairwise correctness comparison

| Backend A | Backend B | Delta | Lambda | Sig LF Jaccard | Z overall r | Z median col r | Z min col r | A nonzero Jaccard |
|-----------|-----------|------:|-------:|---------------:|------------:|---------------:|------------:|------------------:|
| pure_r | rust | 0.01 | 0.1 | 0.4000 | 1.0000 | 1.0000 | 1.0000 | N/A |
| pure_r | rust | 0.01 | 1.0 | 0.3333 | 1.0000 | 1.0000 | 1.0000 | N/A |
| pure_r | rust | 0.1 | 0.1 | 0.6667 | 1.0000 | 1.0000 | 1.0000 | N/A |
| pure_r | rust | 0.1 | 1.0 | 0.3333 | 1.0000 | 1.0000 | 1.0000 | N/A |
