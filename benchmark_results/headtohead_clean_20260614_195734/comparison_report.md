# SLIDE Backend Benchmark Report

Generated: 2026-06-14 20:09:44

## Timing

| Backend | Delta | Lambda | Seconds | Status |
|---------|------:|-------:|--------:|--------|
| pure_r | 0.01 | 0.1 | 104.02 | success |
| pure_r | 0.01 | 1.0 | 88.75 | success |
| pure_r | 0.1 | 0.1 | 36.33 | success |
| pure_r | 0.1 | 1.0 | 38.06 | success |
| rust | 0.01 | 0.1 | 130.96 | success |
| rust | 0.01 | 1.0 | 151.59 | success |
| rust | 0.1 | 0.1 | 73.26 | success |
| rust | 0.1 | 1.0 | 79.34 | success |

### Total time per backend

| Backend | Total (s) | Speedup vs slowest |
|---------|----------:|-------------------:|
| pure_r | 267.16 | 1.6x |
| rust | 435.14 | 1.0x |

## Results per backend

| Backend | Delta | Lambda | K | Sig LFs | Sig Interacts |
|---------|------:|-------:|--:|--------:|--------------:|
| pure_r | 0.01 | 0.1 | 172 | 5 | 15 |
| pure_r | 0.01 | 1.0 | 172 | 5 | 12 |
| pure_r | 0.1 | 0.1 | 88 | 2 | 3 |
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
| pure_r | rust | 0.1 | 0.1 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | N/A |
| pure_r | rust | 0.1 | 1.0 | 0.2500 | 1.0000 | 1.0000 | 1.0000 | N/A |
