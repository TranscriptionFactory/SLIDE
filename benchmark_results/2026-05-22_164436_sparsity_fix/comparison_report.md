# SLIDE Backend Benchmark Report

Generated: 2026-05-22 16:55:59

## Timing

| Backend | Delta | Lambda | Seconds | Status |
|---------|------:|-------:|--------:|--------|
| python | 0.01 | 0.1 | 706.36 | success |
| python | 0.01 | 1.0 | 663.57 | success |
| r_knockoffs | 0.01 | 0.1 | 1095.71 | success |
| rust | 0.01 | 0.1 | 172.48 | success |
| rust | 0.01 | 1.0 | 199.31 | success |
| rust | 0.1 | 0.1 | 134.22 | success |
| rust | 0.1 | 1.0 | 125.47 | success |

### Total time per backend

| Backend | Total (s) | Speedup vs slowest |
|---------|----------:|-------------------:|
| python | 1369.93 | 1.0x |
| r_knockoffs | 1095.71 | 1.3x |
| rust | 631.48 | 2.2x |

## Results per backend

| Backend | Delta | Lambda | K | Sig LFs | Sig Interacts |
|---------|------:|-------:|--:|--------:|--------------:|
| python | 0.01 | 0.1 | 172 | 5 | 19 |
| python | 0.01 | 1.0 | 172 | 5 | 10 |
| python | 0.1 | 0.1 | 88 | 8 | 0 |
| r_knockoffs | 0.01 | 0.1 | 172 | 5 | 16 |
| r_knockoffs | 0.01 | 1.0 | 172 | 6 | 0 |
| rust | 0.01 | 0.1 | 172 | 5 | 20 |
| rust | 0.01 | 1.0 | 172 | 6 | 23 |
| rust | 0.1 | 0.1 | 88 | 13 | 54 |
| rust | 0.1 | 1.0 | 88 | 13 | 58 |

## Pairwise correctness comparison

| Backend A | Backend B | Delta | Lambda | Sig LF Jaccard | Z overall r | Z median col r | Z min col r | A nonzero Jaccard |
|-----------|-----------|------:|-------:|---------------:|------------:|---------------:|------------:|------------------:|
| python | r_knockoffs | 0.01 | 0.1 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 |
| python | rust | 0.01 | 0.1 | 0.6667 | 0.9997 | 0.9999 | 0.9989 | 0.9925 |
| r_knockoffs | rust | 0.01 | 0.1 | 0.6667 | 0.9997 | 0.9999 | 0.9989 | 0.9925 |
| python | r_knockoffs | 0.01 | 1.0 | 0.8333 | 1.0000 | 1.0000 | 1.0000 | 1.0000 |
| python | rust | 0.01 | 1.0 | 0.5714 | 0.9963 | 0.9991 | 0.9787 | 0.9925 |
| r_knockoffs | rust | 0.01 | 1.0 | 0.5000 | 0.9963 | 0.9991 | 0.9787 | 0.9925 |
| python | rust | 0.1 | 0.1 | 0.4000 | 0.9382 | 0.9898 | 0.7934 | 0.9963 |
