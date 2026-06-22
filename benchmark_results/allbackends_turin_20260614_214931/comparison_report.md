# SLIDE Backend Benchmark Report

Generated: 2026-06-14 22:20:23

## Timing

| Backend | Delta | Lambda | Seconds | Status |
|---------|------:|-------:|--------:|--------|
| r_knockoffs | 0.01 | 0.1 | 500.94 | success |
| r_knockoffs | 0.01 | 1.0 | 489.78 | success |
| r_knockoffs | 0.1 | 0.1 | 418.88 | success |
| r_knockoffs | 0.1 | 1.0 | 403.35 | success |
| pure_r | 0.01 | 0.1 | 51.85 | success |
| pure_r | 0.01 | 1.0 | 45.40 | success |
| pure_r | 0.1 | 0.1 | 20.05 | success |
| pure_r | 0.1 | 1.0 | 16.23 | success |
| python | 0.01 | 0.1 | 475.10 | success |
| python | 0.01 | 1.0 | 444.68 | success |
| python | 0.1 | 0.1 | 322.46 | success |
| python | 0.1 | 1.0 | 495.21 | success |
| rust | 0.01 | 0.1 | 38.18 | success |
| rust | 0.01 | 1.0 | 40.44 | success |
| rust | 0.1 | 0.1 | 18.14 | success |
| rust | 0.1 | 1.0 | 19.77 | success |

### Total time per backend

| Backend | Total (s) | Speedup vs slowest |
|---------|----------:|-------------------:|
| pure_r | 133.53 | 13.6x |
| python | 1737.44 | 1.0x |
| r_knockoffs | 1812.95 | 1.0x |
| rust | 116.54 | 15.6x |

## Results per backend

| Backend | Delta | Lambda | K | Sig LFs | Sig Interacts |
|---------|------:|-------:|--:|--------:|--------------:|
| r_knockoffs | 0.01 | 0.1 | 172 | 5 | 16 |
| r_knockoffs | 0.01 | 1.0 | 172 | 5 | 14 |
| r_knockoffs | 0.1 | 0.1 | 88 | 7 | 19 |
| r_knockoffs | 0.1 | 1.0 | 88 | 4 | 13 |

## Pairwise correctness comparison

| Backend A | Backend B | Delta | Lambda | Sig LF Jaccard | Z overall r | Z median col r | Z min col r | A nonzero Jaccard |
|-----------|-----------|------:|-------:|---------------:|------------:|---------------:|------------:|------------------:|
