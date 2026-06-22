# SLIDE Backend Benchmark Report

Generated: 2026-06-22 13:46:42

## Timing

| Backend | Delta | Lambda | Seconds | Status |
|---------|------:|-------:|--------:|--------|
| rust | 0.01 | 0.1 | 93.85 | success |
| rust | 0.01 | 1.0 | 86.38 | success |
| rust | 0.1 | 0.1 | 100.10 | success |
| rust | 0.1 | 1.0 | 135.21 | success |
| pure_r | 0.01 | 0.1 | 98.21 | success |
| pure_r | 0.01 | 1.0 | 93.63 | success |
| pure_r | 0.1 | 0.1 | 74.06 | success |
| pure_r | 0.1 | 1.0 | 83.84 | success |
| python | 0.01 | 0.1 | 118.72 | success |
| python | 0.01 | 1.0 | 116.27 | success |
| python | 0.1 | 0.1 | 48.98 | success |
| python | 0.1 | 1.0 | 72.00 | success |
| r_knockoffs | 0.01 | 0.1 | 126.79 | success |
| r_knockoffs | 0.01 | 1.0 | 119.78 | success |
| r_knockoffs | 0.1 | 0.1 | 50.92 | success |
| r_knockoffs | 0.1 | 1.0 | 73.35 | success |

### Total time per backend

| Backend | Total (s) | Speedup vs slowest |
|---------|----------:|-------------------:|
| pure_r | 349.74 | 1.2x |
| python | 355.98 | 1.2x |
| r_knockoffs | 370.85 | 1.1x |
| rust | 415.53 | 1.0x |

## Results per backend

| Backend | Delta | Lambda | K | Sig LFs | Sig Interacts |
|---------|------:|-------:|--:|--------:|--------------:|
| rust | 0.01 | 0.1 | 36 | 4 | 10 |
| rust | 0.01 | 1.0 | 36 | 4 | 10 |
| rust | 0.1 | 0.1 | 39 | 6 | 19 |
| rust | 0.1 | 1.0 | 39 | 5 | 16 |

## Pairwise correctness comparison

| Backend A | Backend B | Delta | Lambda | Sig LF Jaccard | Z overall r | Z median col r | Z min col r | A nonzero Jaccard |
|-----------|-----------|------:|-------:|---------------:|------------:|---------------:|------------:|------------------:|
