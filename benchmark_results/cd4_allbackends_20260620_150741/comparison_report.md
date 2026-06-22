# SLIDE Backend Benchmark Report

Generated: 2026-06-20 15:32:24

## Timing

| Backend | Delta | Lambda | Seconds | Status |
|---------|------:|-------:|--------:|--------|
| pure_r | 0.01 | 0.1 | 90.34 | success |
| pure_r | 0.01 | 1.0 | 84.43 | success |
| pure_r | 0.1 | 0.1 | 67.83 | success |
| pure_r | 0.1 | 1.0 | 77.27 | success |
| python | 0.01 | 0.1 | 90.68 | success |
| python | 0.01 | 1.0 | 92.79 | success |
| python | 0.1 | 0.1 | 42.05 | success |
| python | 0.1 | 1.0 | 42.40 | success |
| r_knockoffs | 0.01 | 0.1 | 93.13 | success |
| r_knockoffs | 0.01 | 1.0 | 93.60 | success |
| r_knockoffs | 0.1 | 0.1 | 31.17 | success |
| r_knockoffs | 0.1 | 1.0 | 42.84 | success |
| rust | 0.01 | 0.1 | 109.65 | success |
| rust | 0.01 | 1.0 | 104.11 | success |
| rust | 0.1 | 0.1 | 118.62 | success |
| rust | 0.1 | 1.0 | 162.62 | success |

### Total time per backend

| Backend | Total (s) | Speedup vs slowest |
|---------|----------:|-------------------:|
| pure_r | 319.88 | 1.5x |
| python | 267.92 | 1.8x |
| r_knockoffs | 260.73 | 1.9x |
| rust | 495.00 | 1.0x |

## Results per backend

| Backend | Delta | Lambda | K | Sig LFs | Sig Interacts |
|---------|------:|-------:|--:|--------:|--------------:|
| pure_r | 0.01 | 0.1 | 36 | 4 | 8 |
| pure_r | 0.01 | 1.0 | 36 | 4 | 9 |
| pure_r | 0.1 | 0.1 | 39 | 1 | 2 |
| pure_r | 0.1 | 1.0 | 39 | 3 | 3 |
| python | 0.01 | 0.1 | 36 | 4 | 5 |
| python | 0.01 | 1.0 | 36 | 4 | 10 |
| python | 0.1 | 0.1 | 39 | 2 | 4 |
| python | 0.1 | 1.0 | 39 | 2 | 2 |
| r_knockoffs | 0.01 | 0.1 | 36 | 4 | 5 |
| r_knockoffs | 0.01 | 1.0 | 36 | 4 | 10 |
| r_knockoffs | 0.1 | 0.1 | 39 | 1 | 2 |
| r_knockoffs | 0.1 | 1.0 | 39 | 2 | 2 |
| rust | 0.01 | 0.1 | 36 | 4 | 10 |
| rust | 0.01 | 1.0 | 36 | 4 | 10 |
| rust | 0.1 | 0.1 | 39 | 6 | 19 |
| rust | 0.1 | 1.0 | 39 | 5 | 17 |

## Pairwise correctness comparison

| Backend A | Backend B | Delta | Lambda | Sig LF Jaccard | Z overall r | Z median col r | Z min col r | A nonzero Jaccard |
|-----------|-----------|------:|-------:|---------------:|------------:|---------------:|------------:|------------------:|
| python | r_knockoffs | 0.01 | 0.1 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 |
| python | pure_r | 0.01 | 0.1 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | N/A |
| python | rust | 0.01 | 0.1 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 |
| r_knockoffs | pure_r | 0.01 | 0.1 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | N/A |
| r_knockoffs | rust | 0.01 | 0.1 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 |
| pure_r | rust | 0.01 | 0.1 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | N/A |
| python | r_knockoffs | 0.01 | 1.0 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 |
| python | pure_r | 0.01 | 1.0 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | N/A |
| python | rust | 0.01 | 1.0 | 0.6000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 |
| r_knockoffs | pure_r | 0.01 | 1.0 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | N/A |
| r_knockoffs | rust | 0.01 | 1.0 | 0.6000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 |
| pure_r | rust | 0.01 | 1.0 | 0.6000 | 1.0000 | 1.0000 | 1.0000 | N/A |
| python | r_knockoffs | 0.1 | 0.1 | 0.5000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 |
| python | pure_r | 0.1 | 0.1 | 0.5000 | 1.0000 | 1.0000 | 1.0000 | N/A |
| python | rust | 0.1 | 0.1 | 0.3333 | 1.0000 | 1.0000 | 1.0000 | 1.0000 |
| r_knockoffs | pure_r | 0.1 | 0.1 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | N/A |
| r_knockoffs | rust | 0.1 | 0.1 | 0.1667 | 1.0000 | 1.0000 | 1.0000 | 1.0000 |
| pure_r | rust | 0.1 | 0.1 | 0.1667 | 1.0000 | 1.0000 | 1.0000 | N/A |
| python | r_knockoffs | 0.1 | 1.0 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 |
| python | pure_r | 0.1 | 1.0 | 0.6667 | 1.0000 | 1.0000 | 1.0000 | N/A |
| python | rust | 0.1 | 1.0 | 0.1667 | 1.0000 | 1.0000 | 1.0000 | 1.0000 |
| r_knockoffs | pure_r | 0.1 | 1.0 | 0.6667 | 1.0000 | 1.0000 | 1.0000 | N/A |
| r_knockoffs | rust | 0.1 | 1.0 | 0.1667 | 1.0000 | 1.0000 | 1.0000 | 1.0000 |
| pure_r | rust | 0.1 | 1.0 | 0.3333 | 1.0000 | 1.0000 | 1.0000 | N/A |
