# SLIDE Backend Benchmark Report

Generated: 2026-06-08 13:49:30

## Timing

| Backend | Delta | Lambda | Seconds | Status |
|---------|------:|-------:|--------:|--------|
| pure_r | 0.01 | 0.1 | 128.62 | success |
| pure_r | 0.01 | 1.0 | 108.63 | success |
| pure_r | 0.1 | 0.1 | 47.97 | success |
| pure_r | 0.1 | 1.0 | 45.69 | success |
| rust | 0.01 | 0.1 | 151.69 | success |
| rust | 0.01 | 1.0 | 175.06 | success |
| rust | 0.1 | 0.1 | 83.85 | success |
| rust | 0.1 | 1.0 | 89.51 | success |

### Total time per backend

| Backend | Total (s) | Speedup vs slowest |
|---------|----------:|-------------------:|
| pure_r | 330.91 | 1.5x |
| rust | 500.11 | 1.0x |

## Results per backend

| Backend | Delta | Lambda | K | Sig LFs | Sig Interacts |
|---------|------:|-------:|--:|--------:|--------------:|
| pure_r | 0.01 | 0.1 | 172 | 5 | 13 |
| pure_r | 0.01 | 1.0 | 172 | 5 | 15 |
| pure_r | 0.1 | 0.1 | 88 | 4 | 3 |
| pure_r | 0.1 | 1.0 | 88 | 3 | 7 |
| rust | 0.01 | 0.1 | 172 | 2 | 3 |
| rust | 0.01 | 1.0 | 172 | 2 | 3 |
| rust | 0.1 | 0.1 | 88 | 2 | 6 |
| rust | 0.1 | 1.0 | 88 | 2 | 8 |

## Pairwise correctness comparison

| Backend A | Backend B | Delta | Lambda | Sig LF Jaccard | Z overall r | Z median col r | Z min col r | A nonzero Jaccard |
|-----------|-----------|------:|-------:|---------------:|------------:|---------------:|------------:|------------------:|
| pure_r | rust | 0.01 | 0.1 | 0.4000 | 1.0000 | 1.0000 | 1.0000 | N/A |
| pure_r | rust | 0.01 | 1.0 | 0.4000 | 1.0000 | 1.0000 | 1.0000 | N/A |
| pure_r | rust | 0.1 | 0.1 | 0.5000 | 1.0000 | 1.0000 | 1.0000 | N/A |
| pure_r | rust | 0.1 | 1.0 | 0.2500 | 1.0000 | 1.0000 | 1.0000 | N/A |
