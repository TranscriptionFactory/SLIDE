# SLIDE Backend Benchmark Report

Generated: 2026-06-17 23:43:22

## Timing

| Backend | Delta | Lambda | Seconds | Status |
|---------|------:|-------:|--------:|--------|
| pure_r | 0.01 | 0.1 | 51.85 | success |
| pure_r | 0.01 | 1.0 | 45.40 | success |
| pure_r | 0.1 | 0.1 | 20.05 | success |
| pure_r | 0.1 | 1.0 | 16.23 | success |
| python | 0.01 | 0.1 | 475.10 | success |
| python | 0.01 | 1.0 | 444.68 | success |
| python | 0.1 | 0.1 | 322.46 | success |
| python | 0.1 | 1.0 | 495.21 | success |
| r_knockoffs | 0.01 | 0.1 | 500.94 | success |
| r_knockoffs | 0.01 | 1.0 | 489.78 | success |
| r_knockoffs | 0.1 | 0.1 | 418.88 | success |
| r_knockoffs | 0.1 | 1.0 | 403.35 | success |
| rust | 0.01 | 0.1 | 61.03 | success |
| rust | 0.01 | 1.0 | 57.42 | success |
| rust | 0.1 | 0.1 | 17.71 | success |
| rust | 0.1 | 1.0 | 19.23 | success |

### Total time per backend

| Backend | Total (s) | Speedup vs slowest |
|---------|----------:|-------------------:|
| pure_r | 133.53 | 13.6x |
| python | 1737.44 | 1.0x |
| r_knockoffs | 1812.95 | 1.0x |
| rust | 155.40 | 11.7x |

## Results per backend

| Backend | Delta | Lambda | K | Sig LFs | Sig Interacts |
|---------|------:|-------:|--:|--------:|--------------:|
| pure_r | 0.01 | 0.1 | 172 | 5 | 14 |
| pure_r | 0.01 | 1.0 | 172 | 6 | 11 |
| pure_r | 0.1 | 0.1 | 88 | 5 | 7 |
| pure_r | 0.1 | 1.0 | 88 | 2 | 5 |
| python | 0.01 | 0.1 | 172 | 4 | 13 |
| python | 0.01 | 1.0 | 172 | 4 | 8 |
| python | 0.1 | 0.1 | 88 | 5 | 15 |
| python | 0.1 | 1.0 | 88 | 6 | 13 |
| r_knockoffs | 0.01 | 0.1 | 172 | 5 | 16 |
| r_knockoffs | 0.01 | 1.0 | 172 | 5 | 14 |
| r_knockoffs | 0.1 | 0.1 | 88 | 7 | 19 |
| r_knockoffs | 0.1 | 1.0 | 88 | 4 | 13 |
| rust | 0.01 | 0.1 | 172 | 2 | 3 |
| rust | 0.01 | 1.0 | 172 | 2 | 3 |
| rust | 0.1 | 0.1 | 88 | 2 | 6 |
| rust | 0.1 | 1.0 | 88 | 2 | 8 |

## Pairwise correctness comparison

| Backend A | Backend B | Delta | Lambda | Sig LF Jaccard | Z overall r | Z median col r | Z min col r | A nonzero Jaccard |
|-----------|-----------|------:|-------:|---------------:|------------:|---------------:|------------:|------------------:|
| python | r_knockoffs | 0.01 | 0.1 | 0.8000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 |
| python | pure_r | 0.01 | 0.1 | 0.8000 | 1.0000 | 1.0000 | 1.0000 | N/A |
| python | rust | 0.01 | 0.1 | 0.5000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 |
| r_knockoffs | pure_r | 0.01 | 0.1 | 0.6667 | 1.0000 | 1.0000 | 1.0000 | N/A |
| r_knockoffs | rust | 0.01 | 0.1 | 0.4000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 |
| pure_r | rust | 0.01 | 0.1 | 0.4000 | 1.0000 | 1.0000 | 1.0000 | N/A |
| python | r_knockoffs | 0.01 | 1.0 | 0.8000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 |
| python | pure_r | 0.01 | 1.0 | 0.6667 | 1.0000 | 1.0000 | 1.0000 | N/A |
| python | rust | 0.01 | 1.0 | 0.5000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 |
| r_knockoffs | pure_r | 0.01 | 1.0 | 0.8333 | 1.0000 | 1.0000 | 1.0000 | N/A |
| r_knockoffs | rust | 0.01 | 1.0 | 0.4000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 |
| pure_r | rust | 0.01 | 1.0 | 0.3333 | 1.0000 | 1.0000 | 1.0000 | N/A |
| python | r_knockoffs | 0.1 | 0.1 | 0.7143 | 1.0000 | 1.0000 | 1.0000 | 1.0000 |
| python | pure_r | 0.1 | 0.1 | 0.4286 | 1.0000 | 1.0000 | 1.0000 | N/A |
| python | rust | 0.1 | 0.1 | 0.1667 | 1.0000 | 1.0000 | 1.0000 | 1.0000 |
| r_knockoffs | pure_r | 0.1 | 0.1 | 0.3333 | 1.0000 | 1.0000 | 1.0000 | N/A |
| r_knockoffs | rust | 0.1 | 0.1 | 0.1250 | 1.0000 | 1.0000 | 1.0000 | 1.0000 |
| pure_r | rust | 0.1 | 0.1 | 0.1667 | 1.0000 | 1.0000 | 1.0000 | N/A |
| python | r_knockoffs | 0.1 | 1.0 | 0.6667 | 1.0000 | 1.0000 | 1.0000 | 1.0000 |
| python | pure_r | 0.1 | 1.0 | 0.3333 | 1.0000 | 1.0000 | 1.0000 | N/A |
| python | rust | 0.1 | 1.0 | 0.1429 | 1.0000 | 1.0000 | 1.0000 | 1.0000 |
| r_knockoffs | pure_r | 0.1 | 1.0 | 0.5000 | 1.0000 | 1.0000 | 1.0000 | N/A |
| r_knockoffs | rust | 0.1 | 1.0 | 0.2000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 |
| pure_r | rust | 0.1 | 1.0 | 0.3333 | 1.0000 | 1.0000 | 1.0000 | N/A |
