# SLIDE Backend Benchmark Report

Generated: 2026-06-22 15:08:49

## Timing

| Backend | Delta | Lambda | Seconds | Status |
|---------|------:|-------:|--------:|--------|
| pure_r | 0.01 | 0.1 | 96.26 | success |
| pure_r | 0.01 | 1.0 | 92.58 | success |
| pure_r | 0.1 | 0.1 | 72.94 | success |
| pure_r | 0.1 | 1.0 | 84.17 | success |
| python | 0.01 | 0.1 | 114.96 | success |
| python | 0.01 | 1.0 | 112.20 | success |
| python | 0.1 | 0.1 | 65.00 | success |
| python | 0.1 | 1.0 | 69.38 | success |
| r_knockoffs | 0.01 | 0.1 | 119.18 | success |
| r_knockoffs | 0.01 | 1.0 | 117.66 | success |
| r_knockoffs | 0.1 | 0.1 | 49.37 | success |
| r_knockoffs | 0.1 | 1.0 | 72.06 | success |
| rust | 0.01 | 0.1 | 52.71 | success |
| rust | 0.01 | 1.0 | 50.33 | success |
| rust | 0.1 | 0.1 | 38.54 | success |
| rust | 0.1 | 1.0 | 37.83 | success |

### Total time per backend

| Backend | Total (s) | Speedup vs slowest |
|---------|----------:|-------------------:|
| pure_r | 345.96 | 1.0x |
| python | 361.54 | 1.0x |
| r_knockoffs | 358.27 | 1.0x |
| rust | 179.42 | 2.0x |

## Results per backend

| Backend | Delta | Lambda | K | Sig LFs | Sig Interacts |
|---------|------:|-------:|--:|--------:|--------------:|
| pure_r | 0.01 | 0.1 | 36 | 4 | 7 |
| pure_r | 0.01 | 1.0 | 36 | 4 | 10 |
| pure_r | 0.1 | 0.1 | 39 | 1 | 3 |
| pure_r | 0.1 | 1.0 | 39 | 3 | 4 |
| python | 0.01 | 0.1 | 36 | 4 | 5 |
| python | 0.01 | 1.0 | 36 | 4 | 10 |
| python | 0.1 | 0.1 | 39 | 2 | 3 |
| python | 0.1 | 1.0 | 39 | 2 | 2 |
| r_knockoffs | 0.01 | 0.1 | 36 | 4 | 4 |
| r_knockoffs | 0.01 | 1.0 | 36 | 4 | 10 |
| r_knockoffs | 0.1 | 0.1 | 39 | 1 | 2 |
| r_knockoffs | 0.1 | 1.0 | 39 | 2 | 2 |
| rust | 0.01 | 0.1 | 36 | 5 | 14 |
| rust | 0.01 | 1.0 | 36 | 5 | 14 |
| rust | 0.1 | 0.1 | 39 | 3 | 8 |
| rust | 0.1 | 1.0 | 39 | 3 | 9 |

## Pairwise correctness comparison

| Backend A | Backend B | Delta | Lambda | Sig LF Jaccard | Z overall r | Z median col r | Z min col r | A nonzero Jaccard |
|-----------|-----------|------:|-------:|---------------:|------------:|---------------:|------------:|------------------:|
| python | r_knockoffs | 0.01 | 0.1 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 |
| python | pure_r | 0.01 | 0.1 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | N/A |
| python | rust | 0.01 | 0.1 | 0.8000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 |
| r_knockoffs | pure_r | 0.01 | 0.1 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | N/A |
| r_knockoffs | rust | 0.01 | 0.1 | 0.8000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 |
| pure_r | rust | 0.01 | 0.1 | 0.8000 | 1.0000 | 1.0000 | 1.0000 | N/A |
| python | r_knockoffs | 0.01 | 1.0 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 |
| python | pure_r | 0.01 | 1.0 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | N/A |
| python | rust | 0.01 | 1.0 | 0.8000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 |
| r_knockoffs | pure_r | 0.01 | 1.0 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | N/A |
| r_knockoffs | rust | 0.01 | 1.0 | 0.8000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 |
| pure_r | rust | 0.01 | 1.0 | 0.8000 | 1.0000 | 1.0000 | 1.0000 | N/A |
| python | r_knockoffs | 0.1 | 0.1 | 0.5000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 |
| python | pure_r | 0.1 | 0.1 | 0.5000 | 1.0000 | 1.0000 | 1.0000 | N/A |
| python | rust | 0.1 | 0.1 | 0.6667 | 1.0000 | 1.0000 | 1.0000 | 1.0000 |
| r_knockoffs | pure_r | 0.1 | 0.1 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | N/A |
| r_knockoffs | rust | 0.1 | 0.1 | 0.3333 | 1.0000 | 1.0000 | 1.0000 | 1.0000 |
| pure_r | rust | 0.1 | 0.1 | 0.3333 | 1.0000 | 1.0000 | 1.0000 | N/A |
| python | r_knockoffs | 0.1 | 1.0 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 |
| python | pure_r | 0.1 | 1.0 | 0.6667 | 1.0000 | 1.0000 | 1.0000 | N/A |
| python | rust | 0.1 | 1.0 | 0.2500 | 1.0000 | 1.0000 | 1.0000 | 1.0000 |
| r_knockoffs | pure_r | 0.1 | 1.0 | 0.6667 | 1.0000 | 1.0000 | 1.0000 | N/A |
| r_knockoffs | rust | 0.1 | 1.0 | 0.2500 | 1.0000 | 1.0000 | 1.0000 | 1.0000 |
| pure_r | rust | 0.1 | 1.0 | 0.2000 | 1.0000 | 1.0000 | 1.0000 | N/A |
