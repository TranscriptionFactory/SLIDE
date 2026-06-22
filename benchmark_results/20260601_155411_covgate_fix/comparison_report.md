# SLIDE Backend Benchmark Report

Generated: 2026-06-08 13:11:42

## Timing

| Backend | Delta | Lambda | Seconds | Status |
|---------|------:|-------:|--------:|--------|
| pure_r | 0.01 | 0.1 | 147.37 | success |
| pure_r | 0.01 | 1.0 | 123.91 | success |
| pure_r | 0.1 | 0.1 | 34.31 | success |
| pure_r | 0.1 | 1.0 | 52.22 | success |
| python | 0.01 | 0.1 | 1072.38 | success |
| python | 0.01 | 1.0 | 1073.79 | success |
| python | 0.1 | 0.1 | 915.52 | success |
| python | 0.1 | 1.0 | 394.63 | success |
| r_knockoffs | 0.01 | 0.1 | 1093.57 | success |
| r_knockoffs | 0.01 | 1.0 | 1095.88 | success |
| r_knockoffs | 0.1 | 0.1 | 458.52 | success |
| r_knockoffs | 0.1 | 1.0 | 156.61 | success |
| rust | 0.01 | 0.1 | 538.96 | success |
| rust | 0.01 | 1.0 | 542.58 | success |
| rust | 0.1 | 0.1 | 234.05 | success |
| rust | 0.1 | 1.0 | 268.70 | success |

### Total time per backend

| Backend | Total (s) | Speedup vs slowest |
|---------|----------:|-------------------:|
| pure_r | 357.81 | 9.7x |
| python | 3456.31 | 1.0x |
| r_knockoffs | 2804.58 | 1.2x |
| rust | 1584.29 | 2.2x |

## Results per backend

| Backend | Delta | Lambda | K | Sig LFs | Sig Interacts |
|---------|------:|-------:|--:|--------:|--------------:|
| pure_r | 0.01 | 0.1 | 172 | 5 | 15 |
| pure_r | 0.01 | 1.0 | 172 | 5 | 10 |
| pure_r | 0.1 | 0.1 | 88 | 0 | 0 |
| pure_r | 0.1 | 1.0 | 88 | 3 | 7 |
| python | 0.01 | 0.1 | 172 | 4 | 15 |
| python | 0.01 | 1.0 | 172 | 5 | 14 |
| python | 0.1 | 0.1 | 88 | 6 | 18 |
| python | 0.1 | 1.0 | 88 | 2 | 6 |
| r_knockoffs | 0.01 | 0.1 | 172 | 4 | 13 |
| r_knockoffs | 0.01 | 1.0 | 172 | 5 | 9 |
| r_knockoffs | 0.1 | 0.1 | 88 | 2 | 9 |
| r_knockoffs | 0.1 | 1.0 | 88 | 1 | 6 |
| rust | 0.01 | 0.1 | 172 | 2 | 4 |
| rust | 0.01 | 1.0 | 172 | 2 | 2 |
| rust | 0.1 | 0.1 | 88 | 2 | 4 |
| rust | 0.1 | 1.0 | 88 | 4 | 10 |

## Pairwise correctness comparison

| Backend A | Backend B | Delta | Lambda | Sig LF Jaccard | Z overall r | Z median col r | Z min col r | A nonzero Jaccard |
|-----------|-----------|------:|-------:|---------------:|------------:|---------------:|------------:|------------------:|
| python | r_knockoffs | 0.01 | 0.1 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 |
| python | rust | 0.01 | 0.1 | 0.5000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 |
| python | pure_r | 0.01 | 0.1 | 0.8000 | 1.0000 | 1.0000 | 1.0000 | N/A |
| r_knockoffs | rust | 0.01 | 0.1 | 0.5000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 |
| r_knockoffs | pure_r | 0.01 | 0.1 | 0.8000 | 1.0000 | 1.0000 | 1.0000 | N/A |
| rust | pure_r | 0.01 | 0.1 | 0.4000 | 1.0000 | 1.0000 | 1.0000 | N/A |
| python | r_knockoffs | 0.01 | 1.0 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 |
| python | rust | 0.01 | 1.0 | 0.4000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 |
| python | pure_r | 0.01 | 1.0 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | N/A |
| r_knockoffs | rust | 0.01 | 1.0 | 0.4000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 |
| r_knockoffs | pure_r | 0.01 | 1.0 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | N/A |
| rust | pure_r | 0.01 | 1.0 | 0.4000 | 1.0000 | 1.0000 | 1.0000 | N/A |
| python | r_knockoffs | 0.1 | 0.1 | 0.3333 | 1.0000 | 1.0000 | 1.0000 | 1.0000 |
| python | rust | 0.1 | 0.1 | 0.3333 | 1.0000 | 1.0000 | 1.0000 | 1.0000 |
| python | pure_r | 0.1 | 0.1 | 0.0000 | 1.0000 | 1.0000 | 1.0000 | N/A |
| r_knockoffs | rust | 0.1 | 0.1 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 |
| r_knockoffs | pure_r | 0.1 | 0.1 | 0.0000 | 1.0000 | 1.0000 | 1.0000 | N/A |
| rust | pure_r | 0.1 | 0.1 | 0.0000 | 1.0000 | 1.0000 | 1.0000 | N/A |
| python | r_knockoffs | 0.1 | 1.0 | 0.5000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 |
| python | rust | 0.1 | 1.0 | 0.2000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 |
| python | pure_r | 0.1 | 1.0 | 0.6667 | 1.0000 | 1.0000 | 1.0000 | N/A |
| r_knockoffs | rust | 0.1 | 1.0 | 0.2500 | 1.0000 | 1.0000 | 1.0000 | 1.0000 |
| r_knockoffs | pure_r | 0.1 | 1.0 | 0.3333 | 1.0000 | 1.0000 | 1.0000 | N/A |
| rust | pure_r | 0.1 | 1.0 | 0.1667 | 1.0000 | 1.0000 | 1.0000 | N/A |
