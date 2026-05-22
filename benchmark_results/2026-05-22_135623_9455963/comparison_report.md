# SLIDE Backend Benchmark Report

Generated: 2026-05-22 14:37:35

## Timing

| Backend | Delta | Lambda | Seconds | Status |
|---------|------:|-------:|--------:|--------|
| r_knockoffs | 0.01 | 0.1 | 716.01 | success |
| r_knockoffs | 0.01 | 1.0 | 660.17 | success |
| r_knockoffs | 0.1 | 0.1 | 722.08 | success |
| r_knockoffs | 0.1 | 1.0 | 371.55 | success |
| python | 0.01 | 0.1 | 698.21 | success |
| python | 0.01 | 1.0 | 686.40 | success |
| python | 0.1 | 0.1 | 586.69 | success |
| python | 0.1 | 1.0 | 100.54 | success |

### Total time per backend

| Backend | Total (s) | Speedup vs slowest |
|---------|----------:|-------------------:|
| python | 2071.84 | 1.2x |
| r_knockoffs | 2469.81 | 1.0x |

## Results per backend

| Backend | Delta | Lambda | K | Sig LFs | Sig Interacts |
|---------|------:|-------:|--:|--------:|--------------:|
| r_knockoffs | 0.01 | 0.1 | 172 | 5 | 18 |
| r_knockoffs | 0.01 | 1.0 | 172 | 6 | 10 |
| r_knockoffs | 0.1 | 0.1 | 88 | 8 | 22 |
| r_knockoffs | 0.1 | 1.0 | 88 | 3 | 8 |

## Pairwise correctness comparison

| Backend A | Backend B | Delta | Lambda | Sig LF Jaccard | Z overall r | Z median col r | Z min col r | A nonzero Jaccard |
|-----------|-----------|------:|-------:|---------------:|------------:|---------------:|------------:|------------------:|
