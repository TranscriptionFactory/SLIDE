# SLIDE Backend Benchmark Report

Generated: 2026-06-22 15:46:07

## Timing

| Backend | Delta | Lambda | Seconds | Status |
|---------|------:|-------:|--------:|--------|
| pure_r | 0.01 | 0.1 | 98.05 | success |
| pure_r | 0.01 | 1.0 | 96.59 | success |
| pure_r | 0.1 | 0.1 | 76.08 | success |
| pure_r | 0.1 | 1.0 | 87.85 | success |
| python | 0.01 | 0.1 | 112.87 | success |
| python | 0.01 | 1.0 | 115.33 | success |
| python | 0.1 | 0.1 | 66.69 | success |
| python | 0.1 | 1.0 | 71.00 | success |
| r_knockoffs | 0.01 | 0.1 | 118.19 | success |
| r_knockoffs | 0.01 | 1.0 | 112.22 | success |
| r_knockoffs | 0.1 | 0.1 | 47.46 | success |
| r_knockoffs | 0.1 | 1.0 | 69.97 | success |
| rust | 0.01 | 0.1 | 52.66 | success |
| rust | 0.01 | 1.0 | 51.69 | success |
| rust | 0.1 | 0.1 | 39.34 | success |
| rust | 0.1 | 1.0 | 39.18 | success |

### Total time per backend

| Backend | Total (s) | Speedup vs slowest |
|---------|----------:|-------------------:|
| pure_r | 358.57 | 1.0x |
| python | 365.89 | 1.0x |
| r_knockoffs | 347.84 | 1.1x |
| rust | 182.88 | 2.0x |

## Results per backend

| Backend | Delta | Lambda | K | Sig LFs | Sig Interacts |
|---------|------:|-------:|--:|--------:|--------------:|
| pure_r | 0.01 | 0.1 | 36 | 4 | 6 |
| pure_r | 0.01 | 1.0 | 36 | 4 | 10 |
| pure_r | 0.1 | 0.1 | 39 | 1 | 3 |
| pure_r | 0.1 | 1.0 | 39 | 3 | 3 |
| python | 0.01 | 0.1 | 36 | 4 | 4 |
| python | 0.01 | 1.0 | 36 | 4 | 10 |
| python | 0.1 | 0.1 | 39 | 2 | 3 |
| python | 0.1 | 1.0 | 39 | 2 | 2 |
| r_knockoffs | 0.01 | 0.1 | 36 | 4 | 5 |
| r_knockoffs | 0.01 | 1.0 | 36 | 4 | 10 |
| r_knockoffs | 0.1 | 0.1 | 39 | 1 | 1 |
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
