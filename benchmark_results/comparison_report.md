# SLIDE Backend Benchmark Report

Generated: 2026-05-22 13:53:45

## Timing

| Backend | Delta | Lambda | Seconds | Status |
|---------|------:|-------:|--------:|--------|
| rust | 0.01 | 0.1 | 595.74 | success |
| rust | 0.01 | 1.0 | 734.02 | success |
| rust | 0.1 | 0.1 | 595.19 | success |
| rust | 0.1 | 1.0 | 588.34 | success |
| python | 0.01 | 0.1 | 1068.64 | success |
| python | 0.01 | 1.0 | 384.84 | success |
| python | 0.1 | 0.1 | 574.33 | success |
| python | 0.1 | 1.0 | 663.07 | success |
| r_knockoffs | 0.01 | 0.1 | 705.73 | success |
| r_knockoffs | 0.01 | 1.0 | 657.35 | success |
| r_knockoffs | 0.1 | 0.1 | 375.32 | success |
| r_knockoffs | 0.1 | 1.0 | 437.43 | success |

### Total time per backend

| Backend | Total (s) | Speedup vs slowest |
|---------|----------:|-------------------:|
| python | 2690.88 | 1.0x |
| r_knockoffs | 2175.83 | 1.2x |
| rust | 2513.28 | 1.1x |

## Results per backend

| Backend | Delta | Lambda | K | Sig LFs | Sig Interacts |
|---------|------:|-------:|--:|--------:|--------------:|
| rust | 0.01 | 0.1 | 172 | 5 | 20 |
| rust | 0.01 | 1.0 | 172 | 6 | 23 |
| rust | 0.1 | 0.1 | 88 | 13 | 54 |
| rust | 0.1 | 1.0 | 88 | 13 | 58 |

## Pairwise correctness comparison

| Backend A | Backend B | Delta | Lambda | Sig LF Jaccard | Z overall r | Z median col r | Z min col r | A nonzero Jaccard |
|-----------|-----------|------:|-------:|---------------:|------------:|---------------:|------------:|------------------:|
