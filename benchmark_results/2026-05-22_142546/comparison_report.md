# SLIDE Backend Benchmark Report

Generated: 2026-05-22 15:00:01

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
| rust | 0.01 | 0.1 | 96.10 | success |
| rust | 0.01 | 1.0 | 111.52 | success |
| rust | 0.1 | 0.1 | 65.34 | success |
| rust | 0.1 | 1.0 | 62.02 | success |

### Total time per backend

| Backend | Total (s) | Speedup vs slowest |
|---------|----------:|-------------------:|
| python | 2690.88 | 1.0x |
| r_knockoffs | 2175.83 | 1.2x |
| rust | 334.98 | 8.0x |

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
| rust | 0.01 | 0.1 | 172 | 5 | 20 |
| rust | 0.01 | 1.0 | 172 | 6 | 23 |
| rust | 0.1 | 0.1 | 88 | 13 | 54 |
| rust | 0.1 | 1.0 | 88 | 13 | 58 |

## Pairwise correctness comparison

| Backend A | Backend B | Delta | Lambda | Sig LF Jaccard | Z overall r | Z median col r | Z min col r | A nonzero Jaccard |
|-----------|-----------|------:|-------:|---------------:|------------:|---------------:|------------:|------------------:|
| python | r_knockoffs | 0.01 | 0.1 | 0.8000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 |
| python | rust | 0.01 | 0.1 | 0.8000 | 0.9997 | 0.9999 | 0.9989 | 0.9925 |
| r_knockoffs | rust | 0.01 | 0.1 | 0.6667 | 0.9997 | 0.9999 | 0.9989 | 0.9925 |
| python | r_knockoffs | 0.01 | 1.0 | 0.2000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 |
| python | rust | 0.01 | 1.0 | 0.1667 | 0.9963 | 0.9991 | 0.9787 | 0.9925 |
| r_knockoffs | rust | 0.01 | 1.0 | 0.5714 | 0.9963 | 0.9991 | 0.9787 | 0.9925 |
| python | r_knockoffs | 0.1 | 0.1 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 |
| python | rust | 0.1 | 0.1 | 0.3077 | 0.9382 | 0.9898 | 0.7934 | 0.9963 |
| r_knockoffs | rust | 0.1 | 0.1 | 0.3077 | 0.9382 | 0.9898 | 0.7934 | 0.9963 |
| python | r_knockoffs | 0.1 | 1.0 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 |
| python | rust | 0.1 | 1.0 | 0.2308 | 0.8573 | 0.9562 | 0.5603 | 0.9863 |
| r_knockoffs | rust | 0.1 | 1.0 | 0.2308 | 0.8573 | 0.9562 | 0.5603 | 0.9863 |
