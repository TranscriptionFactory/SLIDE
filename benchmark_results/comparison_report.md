# SLIDE Backend Benchmark Report

Generated: 2026-05-21 04:34:28

## Timing

| Backend | Delta | Lambda | Seconds | Status |
|---------|------:|-------:|--------:|--------|
| python | 0.01 | 0.1 | 1012.60 | success |
| python | 0.01 | 1.0 | 984.53 | success |
| python | 0.1 | 0.1 | 1051.54 | success |
| python | 0.1 | 1.0 | 332.99 | success |

### Total time per backend

| Backend | Total (s) | Speedup vs slowest |
|---------|----------:|-------------------:|
| python | 3381.66 | 1.0x |

## Results per backend

| Backend | Delta | Lambda | K | Sig LFs | Sig Interacts |
|---------|------:|-------:|--:|--------:|--------------:|
| python | 0.01 | 0.1 | 172 | 5 | 18 |
| python | 0.01 | 1.0 | 172 | 6 | 15 |
| python | 0.1 | 0.1 | 88 | 8 | 19 |
| python | 0.1 | 1.0 | 88 | 2 | 6 |

## Pairwise correctness comparison

| Backend A | Backend B | Delta | Lambda | Sig LF Jaccard | Z overall r | Z median col r | Z min col r | A nonzero Jaccard |
|-----------|-----------|------:|-------:|---------------:|------------:|---------------:|------------:|------------------:|
