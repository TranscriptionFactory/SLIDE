# SLIDE Backend Benchmark Report

Generated: 2026-05-23 20:30:14

## Timing

| Backend | Delta | Lambda | Seconds | Status |
|---------|------:|-------:|--------:|--------|
| python | 0.01 | 0.1 | 1068.64 | success |
| python | 0.01 | 1.0 | 384.84 | success |
| python | 0.1 | 0.1 | 574.33 | success |
| python | 0.1 | 1.0 | 663.07 | success |
| r_knockoffs | 0.01 | 0.1 | 705.73 | success |
| r_knockoffs | 0.01 | 1.0 | 657.35 | success |
| r_knockoffs | 0.1 | 0.1 | 375.32 | success |
| r_knockoffs | 0.1 | 1.0 | 437.43 | success |
| rust | 0.01 | 0.1 | 129.61 | success |
| rust | 0.01 | 1.0 | 125.92 | success |
| rust | 0.1 | 0.1 | 153.23 | success |
| rust | 0.1 | 1.0 | 190.78 | success |

### Total time per backend

| Backend | Total (s) | Speedup vs slowest |
|---------|----------:|-------------------:|
| python | 2690.88 | 1.0x |
| r_knockoffs | 2175.83 | 1.2x |
| rust | 599.54 | 4.5x |

## Results per backend

| Backend | Delta | Lambda | K | Sig LFs | Sig Interacts |
|---------|------:|-------:|--:|--------:|--------------:|
| python | 0.01 | 0.1 | 172 | 4 | 14 |
| python | 0.01 | 1.0 | 172 | 1 | 4 |
| python | 0.1 | 0.1 | 88 | 4 | 14 |
| python | 0.1 | 1.0 | 88 | 3 | 7 |
| r_knockoffs | 0.01 | 0.1 | 172 | 5 | 12 |
| r_knockoffs | 0.01 | 1.0 | 172 | 5 | 11 |
| r_knockoffs | 0.1 | 0.1 | 88 | 4 | 13 |
| r_knockoffs | 0.1 | 1.0 | 88 | 3 | 7 |
| rust | 0.01 | 0.1 | 172 | 5 | 19 |
| rust | 0.01 | 1.0 | 172 | 5 | 20 |
| rust | 0.1 | 0.1 | 88 | 14 | 63 |
| rust | 0.1 | 1.0 | 88 | 12 | 55 |

## Pairwise correctness comparison

| Backend A | Backend B | Delta | Lambda | Sig LF Jaccard | Z overall r | Z median col r | Z min col r | A nonzero Jaccard |
|-----------|-----------|------:|-------:|---------------:|------------:|---------------:|------------:|------------------:|
| python | r_knockoffs | 0.01 | 0.1 | 0.8000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 |
| python | rust | 0.01 | 0.1 | 0.8000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 |
| r_knockoffs | rust | 0.01 | 0.1 | 0.6667 | 1.0000 | 1.0000 | 1.0000 | 1.0000 |
| python | r_knockoffs | 0.01 | 1.0 | 0.2000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 |
| python | rust | 0.01 | 1.0 | 0.2000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 |
| r_knockoffs | rust | 0.01 | 1.0 | 0.6667 | 1.0000 | 1.0000 | 1.0000 | 1.0000 |
| python | r_knockoffs | 0.1 | 0.1 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 |
| python | rust | 0.1 | 0.1 | 0.2857 | 1.0000 | 1.0000 | 1.0000 | 1.0000 |
| r_knockoffs | rust | 0.1 | 0.1 | 0.2857 | 1.0000 | 1.0000 | 1.0000 | 1.0000 |
| python | r_knockoffs | 0.1 | 1.0 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 |
| python | rust | 0.1 | 1.0 | 0.2500 | 1.0000 | 1.0000 | 1.0000 | 1.0000 |
| r_knockoffs | rust | 0.1 | 1.0 | 0.2500 | 1.0000 | 1.0000 | 1.0000 | 1.0000 |
