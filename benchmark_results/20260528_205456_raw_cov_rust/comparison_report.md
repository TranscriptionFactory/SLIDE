# SLIDE Backend Benchmark Report

Generated: 2026-05-28 21:39:34

## Timing

| Backend | Delta | Lambda | Seconds | Status |
|---------|------:|-------:|--------:|--------|
| r_knockoffs | 0.01 | 0.1 | 697.23 | success |
| r_knockoffs | 0.01 | 1.0 | 666.28 | success |
| r_knockoffs | 0.1 | 0.1 | 578.97 | success |
| r_knockoffs | 0.1 | 1.0 | 695.66 | success |
| pure_r | 0.01 | 0.1 | 162.17 | success |
| pure_r | 0.01 | 1.0 | 141.69 | success |
| pure_r | 0.1 | 0.1 | 62.31 | success |
| pure_r | 0.1 | 1.0 | 56.62 | success |
| python | 0.01 | 0.1 | 720.97 | success |
| python | 0.01 | 1.0 | 618.50 | success |
| python | 0.1 | 0.1 | 310.98 | success |
| python | 0.1 | 1.0 | 693.63 | success |
| rust | 0.01 | 0.1 | 271.53 | success |
| rust | 0.01 | 1.0 | 260.70 | success |
| rust | 0.1 | 0.1 | 116.95 | success |
| rust | 0.1 | 1.0 | 133.27 | success |

### Total time per backend

| Backend | Total (s) | Speedup vs slowest |
|---------|----------:|-------------------:|
| pure_r | 422.79 | 6.2x |
| python | 2344.08 | 1.1x |
| r_knockoffs | 2638.14 | 1.0x |
| rust | 782.45 | 3.4x |

## Results per backend

| Backend | Delta | Lambda | K | Sig LFs | Sig Interacts |
|---------|------:|-------:|--:|--------:|--------------:|
| r_knockoffs | 0.01 | 0.1 | 172 | 4 | 14 |
| r_knockoffs | 0.01 | 1.0 | 172 | 6 | 10 |
| r_knockoffs | 0.1 | 0.1 | 88 | 7 | 22 |
| r_knockoffs | 0.1 | 1.0 | 88 | 5 | 9 |

## Pairwise correctness comparison

| Backend A | Backend B | Delta | Lambda | Sig LF Jaccard | Z overall r | Z median col r | Z min col r | A nonzero Jaccard |
|-----------|-----------|------:|-------:|---------------:|------------:|---------------:|------------:|------------------:|
